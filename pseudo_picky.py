#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pseudo_picky_bpe.py

A retrofit "pseudo‑Picky" tokenizer on top of a trained HuggingFace BPE tokenizer.
FIXED to support merges as strings OR lists, with robust parents mapping.
Adds PROGRESS STATEMENTS and .TXT INFERENCE SUPPORT.

Workflow:
  1) Load HF tokenizer .json (vocab + merges).
  2) Read training.txt (the corpus used to train the tokenizer).
  3) Build token_removal.json by one of:
     - mode="count": remove tokens with frequency < count_threshold
     - mode="ios":   remove tokens whose IoS >= ios_threshold
       IoS_left(x1|x1,x2)=count_bigrams[(x1,x2)]/count_tokens[x1] (and symmetric for right).
       (Computed from final BPE bigram counts—parent-free and robust.)
     - mode="ios_struct": structural IoS via merge-tree substructure counts
       IoS_left = occ(x1+x2) / occ(x1) using global subtoken counts.

  4) Inference:
     - (single string) --text "..."
     - (batch file)    --input-file file.txt  (one sample per line)
     In both cases: encode with normal BPE, then post-process:
       * split removed tokens into parents recursively until only "kept" tokens remain.
       * if parents missing, fallback to DP segmentation into allowed vocab tokens.
       * optional local forward re-merge (postprocess_mode="split_remerge") to
         recover some compression lost by pure split.

Dependencies:
  - tokenizers  (pip install tokenizers)

Example (build IoS removal):
  python pseudo_picky_bpe.py \
    --tokenizer tokenizer.json \
    --train training.txt \
    --mode ios \
    --ios-threshold 0.9 \
    --removal artifacts/token_removal_ios.json \
    --verbose

Example (prefer structural IoS if ios=0 removals):
  python pseudo_picky_bpe.py \
    --tokenizer tokenizer.json \
    --train training.txt \
    --mode ios_struct \
    --ios-threshold 0.85 \
    --removal artifacts/token_removal_ios_struct.json \
    --verbose

Example (batch inference from .txt, JSONL out):
  python pseudo_picky_bpe.py \
    --tokenizer tokenizer.json \
    --removal artifacts/token_removal_ios.json \
    --postprocess split_remerge \
    --remerge-window 16 \
    --input-file inference.txt \
    --output-file outputs.jsonl \
    --format jsonl \
    --verbose
"""

import os
import json
import time
import argparse
from typing import Dict, Tuple, List, Iterable, Optional, Set
from collections import Counter, defaultdict
from functools import lru_cache

try:
    from tokenizers import Tokenizer
except ImportError as e:
    raise ImportError(
        "This module requires the `tokenizers` package. Install with `pip install tokenizers`."
    ) from e


def _now() -> str:
    return time.strftime("%H:%M:%S")


class PseudoPickyBPE:
    """
    Pseudo‑Picky wrapper for a trained HF BPE tokenizer.

    Key fixes:
      - Robust merges parsing (string OR list form)
      - Robust parents reconstruction (direct concat + fallback split search)
      - IoS from final bigrams (parent-free) AND structural IoS (merge-tree)
      - Count mode can remove rare multi‑char tokens
      - DP fallback segmentation at inference for removed tokens without parents
    """

    def __init__(
        self,
        tokenizer_json: str,
        training_text: Optional[str] = None,
        mode: str = "count",
        count_threshold: Optional[int] = None,
        ios_threshold: float = 0.9,
        removal_json: str = "token_removal.json",
        sample_lines: Optional[int] = None,
        postprocess_mode: str = "split",
        remerge_window: int = 16,
        verbose: bool = True,
        progress_every_lines: int = 10_000,
        progress_every_merges: int = 5_000,
        debug_parents: bool = False,
    ):
        self.tokenizer_json = tokenizer_json
        self.training_text = training_text
        self.mode = mode
        self.count_threshold = count_threshold
        self.ios_threshold = ios_threshold
        self.removal_json = removal_json
        self.sample_lines = sample_lines
        self.postprocess_mode = postprocess_mode
        self.remerge_window = int(remerge_window)
        self.verbose = verbose
        self.progress_every_lines = max(1, int(progress_every_lines))
        self.progress_every_merges = max(1, int(progress_every_merges))
        self.debug_parents = debug_parents

        # Load tokenizer and JSON
        t0 = time.time()
        self.tk = Tokenizer.from_file(self.tokenizer_json)
        with open(self.tokenizer_json, "r", encoding="utf-8") as f:
            self.tk_config = json.load(f)
        t1 = time.time()

        model = self.tk_config.get("model", {})
        if str(model.get("type", "")).upper() != "BPE":
            raise ValueError("This wrapper expects a BPE tokenizer.json (model.type == 'BPE').")

        # vocab: token -> id
        self.vocab: Dict[str, int] = model.get("vocab", {})
        if not isinstance(self.vocab, dict) or not self.vocab:
            raise ValueError("Could not read 'model.vocab' from tokenizer.json.")

        # merges: normalize to a list of (left,right) tuples (strings)
        raw_merges = model.get("merges", [])
        if not isinstance(raw_merges, list) or not raw_merges:
            raise ValueError("Could not read 'model.merges' from tokenizer.json.")

        self.merges: List[Tuple[str, str]] = []
        for m in raw_merges:
            if isinstance(m, str):
                parts = m.split()
                if len(parts) == 2:
                    self.merges.append((parts[0], parts[1]))
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                self.merges.append((str(m[0]), str(m[1])))

        if not self.merges:
            raise ValueError("No valid merges parsed from tokenizer.json (check format).")

        # Merge ranks (training order)
        self.rank: Dict[Tuple[str, str], int] = {pair: i for i, pair in enumerate(self.merges)}

        # Special tokens (to never remove or split)
        self.special_tokens: Set[str] = self._extract_special_tokens(self.tk_config)

        # Parents/creation/base
        self.parents: Dict[str, Tuple[str, str]] = {}
        self.creation_step: Dict[str, int] = {}
        self.base_tokens: Set[str] = set()

        # Removal data
        self.removal_set: Set[str] = set()
        self.removal_meta: Dict[str, Dict] = {}

        # Caches
        self._substructure_cache: Dict[str, Dict[str, int]] = {}
        self._dp_cache: Dict[str, Optional[List[str]]] = {}

        # Build parents robustly
        self._build_parents_map_robust()

        if self.verbose:
            print(f"[init {_now()}] Loaded tokenizer JSON in {t1 - t0:.2f}s")
            print(f"[init] Vocab size: {len(self.vocab)} | #merges: {len(self.merges)}")
            print(f"[init] Special tokens: {len(self.special_tokens)}")
            print(f"[init] Parents coverage: {len(self.parents)} / {len(self.vocab)} "
                  f"({100*len(self.parents)/max(1,len(self.vocab)):.1f}%)")
            print(f"[init] Base tokens (derived): {len(self.base_tokens)}")
            print(f"[init] Mode={self.mode} | Postprocess={self.postprocess_mode}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def build_removals(self) -> None:
        """
        Build a removal list (self.removal_set + self.removal_meta) from the training corpus,
        using self.mode ("count", "ios", or "ios_struct"). Saves to self.removal_json.
        """
        if self.training_text is None:
            raise ValueError("training_text is required to build removals.")

        print("\n=== BUILD REMOVALS ===") if self.verbose else None
        if self.mode == "count":
            if self.count_threshold is None:
                raise ValueError("For mode='count', please provide --count-threshold (int).")
            self._build_removals_count()
        elif self.mode == "ios":
            self._build_removals_ios_bigram()
        elif self.mode == "ios_struct":
            self._build_removals_ios_structural()
        else:
            raise ValueError("mode must be 'count', 'ios', or 'ios_struct'.")

        self._save_removal_json()
        if self.verbose:
            print(f"[build_removals] Saved removal list to: {self.removal_json}")
            print(f"[build_removals] Removed tokens: {len(self.removal_set)}")
        print("=== DONE BUILD REMOVALS ===\n") if self.verbose else None

    def load_removals(self, path: Optional[str] = None) -> None:
        """
        Load a previously built token_removal.json into memory.
        """
        path = path or self.removal_json
        t0 = time.time()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        removed = data.get("removed", [])
        self.removal_set = set()
        self.removal_meta = {}
        for item in removed:
            t = item["token"]
            self.removal_set.add(t)
            self.removal_meta[t] = item
        if self.verbose:
            print(f"[load_removals] Loaded {len(self.removal_set)} removed tokens from {path} "
                  f"in {time.time() - t0:.2f}s")

    def tokenize(self, text: str) -> List[str]:
        """
        Inference-time tokenization for a single string:
          1) run normal BPE encode
          2) post-process: split removed tokens into parents recursively
             (or DP segmentation), optionally re-merge locally
        Returns list of token strings.
        """
        if self.verbose:
            print(f"[tokenize] Encoding text of length {len(text)}...")
        t0 = time.time()
        enc = self.tk.encode(text)
        toks = enc.tokens
        if self.verbose:
            print(f"[tokenize] Vanilla BPE produced {len(toks)} tokens in {time.time() - t0:.3f}s")

        if not self.removal_set:
            if self.verbose:
                print("[tokenize] No removals configured; returning vanilla tokens.")
            return toks

        if self.postprocess_mode == "split":
            out = self._postprocess_split_only(toks)
        elif self.postprocess_mode == "split_remerge":
            out = self._postprocess_split_with_remerge(toks)
        else:
            raise ValueError("postprocess_mode must be 'split' or 'split_remerge'.")

        if self.verbose:
            print(f"[tokenize] Final tokens: {len(out)} (Δ={len(out)-len(toks)})")
        return out

    def tokenize_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        output_format: str = "jsonl",   # "jsonl" or "txt"
        show_line_num: bool = False,
    ) -> None:
        """
        Batch inference from a .txt file: one input sample per line.
        Outputs either to stdout (default) or to output_path.
        """
        if self.verbose:
            print(f"\n=== TXT INFERENCE ===")
            print(f"[file] input={input_path} | output={output_path or 'stdout'} | format={output_format}")

        if output_format not in {"jsonl", "txt"}:
            raise ValueError("output_format must be 'jsonl' or 'txt'.")

        # Avoid per-line spam from tokenize(); temporarily mute and print periodic progress here.
        old_verbose = self.verbose
        self.verbose = False

        t0 = time.time()
        n_lines = 0
        fout = open(output_path, "w", encoding="utf-8") if output_path else None

        def emit_jsonl(outf, obj):
            s = json.dumps(obj, ensure_ascii=False)
            (outf.write(s + "\n") if outf else print(s))

        def emit_txt(outf, line_idx, tokens):
            s = f"{line_idx}\t" + " ".join(tokens) if show_line_num else " ".join(tokens)
            (outf.write(s + "\n") if outf else print(s))

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    line = line.rstrip("\n")
                    toks = self.tokenize(line)
                    if output_format == "jsonl":
                        emit_jsonl(fout, {"line": line_idx, "text": line, "tokens": toks})
                    else:
                        emit_txt(fout, line_idx, toks)

                    n_lines += 1
                    if old_verbose and (n_lines % self.progress_every_lines == 0):
                        print(f"[file] processed {n_lines} lines...")
        finally:
            if fout:
                fout.flush()
                fout.close()
            self.verbose = old_verbose
            if self.verbose:
                print(f"[file] done. processed={n_lines} | time={time.time() - t0:.2f}s")
                print(f"=== END TXT INFERENCE ===\n")

    # -------------------------------------------------------------------------
    # Building removals (COUNT)
    # -------------------------------------------------------------------------

    def _build_removals_count(self) -> None:
        if self.verbose:
            print(f"[count {_now()}] Counting token frequencies over: {self.training_text}")
        t0 = time.time()

        freqs = Counter()
        n_lines = 0
        with open(self.training_text, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    freqs.update(self.tk.encode(line).tokens)
                n_lines += 1

                if self.sample_lines is not None and n_lines >= self.sample_lines:
                    if self.verbose:
                        print(f"[count] Reached sample_lines={self.sample_lines}. Stopping early.")
                    break
                if self.verbose and (n_lines % self.progress_every_lines == 0):
                    print(f"[count] Encoded {n_lines} lines; unique tokens seen={len(freqs)}")

        # Protection: special tokens always; otherwise only 1-char tokens by default.
        def _is_protected(tok: str) -> bool:
            if tok in self.special_tokens:
                return True
            if tok in self.parents:
                return False  # splittable; safe to remove if rare
            return len(tok) <= 1  # treat single-char/byte as atomic

        removal = []
        for tok, c in freqs.items():
            if c < self.count_threshold and not _is_protected(tok):
                removal.append(tok)

        self.removal_set = set(removal)
        self.removal_meta = {}
        for t in removal:
            self.removal_meta[t] = {
                "token": t,
                "reason": "count",
                "count": int(freqs.get(t, 0)),
                "parents": list(self.parents.get(t, (None, None))),
                "creation_step": int(self.creation_step.get(t, -1)),
            }

        if self.verbose:
            kept = len(self.vocab) - len(self.removal_set)
            print(f"[count] Removed {len(self.removal_set)} tokens (kept {kept}). "
                  f"Threshold={self.count_threshold}. Time={time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # Building removals (IoS from BIGRAMS – parent-free)
    # -------------------------------------------------------------------------

    def _build_removals_ios_bigram(self) -> None:
        """
        IoS from final bigram counts:

          IoS_left(x1|x1,x2)  = count_bigrams[(x1,x2)] / count_tokens[x1]
          IoS_right(x2|x1,x2) = count_bigrams[(x1,x2)] / count_tokens[x2]

        We evaluate IoS only over merge pairs (x1,x2) from the tokenizer's merges,
        and remove sides whose IoS >= threshold (excluding special tokens).
        """
        if self.verbose:
            print(f"[ios {_now()}] Phase 1/2: encoding corpus & counting unigrams+bigrams…")
            print(f"[ios]   file={self.training_text}")

        t0 = time.time()
        tok_freq = Counter()
        pair_freq = Counter()
        n_lines = 0

        with open(self.training_text, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    n_lines += 1
                    continue
                toks = self.tk.encode(line).tokens
                tok_freq.update(toks)
                for i in range(len(toks) - 1):
                    pair_freq[(toks[i], toks[i+1])] += 1
                n_lines += 1
                if self.sample_lines is not None and n_lines >= self.sample_lines:
                    if self.verbose:
                        print(f"[ios]   reached sample_lines={self.sample_lines}.")
                    break
                if self.verbose and (n_lines % self.progress_every_lines == 0):
                    print(f"[ios]   encoded {n_lines} lines; uniq tokens={len(tok_freq)}; uniq bigrams={len(pair_freq)}")

        if self.verbose:
            print(f"[ios] Phase 2/2: evaluating IoS over {len(self.merges)} merges (threshold={self.ios_threshold})…")

        removal_candidates: Set[str] = set()
        total_merges = len(self.merges)
        for idx, (x1, x2) in enumerate(self.merges, 1):
            fp = pair_freq.get((x1, x2), 0)
            if fp == 0:
                if self.verbose and (idx % self.progress_every_merges == 0):
                    print(f"[ios]   merges evaluated: {idx}/{total_merges} | candidates={len(removal_candidates)}")
                continue
            ft1 = tok_freq.get(x1, 0)
            ft2 = tok_freq.get(x2, 0)

            if ft1 > 0:
                ios_l = fp / ft1
                if ios_l >= self.ios_threshold and x1 not in self.special_tokens:
                    removal_candidates.add(x1)
            if ft2 > 0:
                ios_r = fp / ft2
                if ios_r >= self.ios_threshold and x2 not in self.special_tokens:
                    removal_candidates.add(x2)

            if self.verbose and (idx % self.progress_every_merges == 0):
                print(f"[ios]   merges evaluated: {idx}/{total_merges} | candidates={len(removal_candidates)}")

        # Protect obvious atomic tokens; prefer removing multi-char or known-parent tokens
        final_removal = []
        for t in removal_candidates:
            if t in self.parents or len(t) > 1:
                final_removal.append(t)

        self.removal_set = set(final_removal)
        self.removal_meta = {}
        for t in final_removal:
            self.removal_meta[t] = {
                "token": t,
                "reason": "ios",
                "ios_threshold": self.ios_threshold,
                "parents": list(self.parents.get(t, (None, None))),
                "creation_step": int(self.creation_step.get(t, -1)),
            }

        if self.verbose:
            kept = len(self.vocab) - len(self.removal_set)
            print(f"[ios] Removed {len(self.removal_set)} tokens (kept {kept}). "
                  f"Total time={time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # Building removals (IoS STRUCTURAL – via merge tree)
    # -------------------------------------------------------------------------

    def _build_removals_ios_structural(self) -> None:
        """
        Structural IoS using global substructure counts:
          IoS_left = occ(x1+x2) / occ(x1)
          IoS_right = occ(x1+x2) / occ(x2)
        where occ() counts include occurrences inside merged tokens via the merge tree.
        """
        if self.verbose:
            print(f"[ios {_now()}] Phase 1/3: encoding corpus to collect final token frequencies…")
            print(f"[ios]   file={self.training_text}")
        t_phase1 = time.time()

        final_freq = Counter()
        n_lines = 0
        with open(self.training_text, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    final_freq.update(self.tk.encode(line).tokens)
                n_lines += 1
                if self.sample_lines is not None and n_lines >= self.sample_lines:
                    if self.verbose:
                        print(f"[ios]   reached sample_lines={self.sample_lines}.")
                    break
                if self.verbose and (n_lines % self.progress_every_lines == 0):
                    print(f"[ios]   encoded {n_lines} lines; unique final tokens={len(final_freq)}")

        if self.verbose:
            print(f"[ios]   Phase 1 complete: {len(final_freq)} unique final tokens seen. "
                  f"Time={time.time() - t_phase1:.2f}s\n")

        # Phase 2: global substructure counts
        if self.verbose:
            print(f"[ios] Phase 2/3: computing global substructure counts…")
        t_phase2 = time.time()

        global_occ: Dict[str, int] = Counter()

        @lru_cache(maxsize=None)
        def substructure_counts(t: str) -> Dict[str, int]:
            if t in self._substructure_cache:
                return self._substructure_cache[t]
            d = defaultdict(int)
            d[t] += 1
            if t in self.parents:
                left, right = self.parents[t]
                for k, v in substructure_counts(left).items():
                    d[k] += v
                for k, v in substructure_counts(right).items():
                    d[k] += v
            dd = dict(d)
            self._substructure_cache[t] = dd
            return dd

        processed = 0
        total_types = len(final_freq)
        last_log = time.time()
        for t, c in final_freq.items():
            sc = substructure_counts(t)
            if c:
                for sub, cnt_in_t in sc.items():
                    if cnt_in_t:
                        global_occ[sub] += cnt_in_t * c
            processed += 1
            if self.verbose and (processed % 2000 == 0 or (time.time() - last_log) > 5.0):
                pct = 100.0 * processed / max(1, total_types)
                print(f"[ios]   substructure {processed}/{total_types} ({pct:.1f}%) "
                      f"| global subtokens so far={len(global_occ)}")
                last_log = time.time()

        if self.verbose:
            print(f"[ios]   Phase 2 complete: global_occ for ~{len(global_occ)} subtokens. "
                  f"Time={time.time() - t_phase2:.2f}s\n")

        # Phase 3: IoS over merges
        if self.verbose:
            print(f"[ios] Phase 3/3: evaluating IoS over {len(self.merges)} merges (threshold={self.ios_threshold})…")
        t_phase3 = time.time()

        removal_candidates: Set[str] = set()
        for step_idx, (x1, x2) in enumerate(self.merges, 1):
            x3 = x1 + x2
            if (x1 not in self.vocab) or (x2 not in self.vocab) or (x3 not in self.vocab):
                continue
            fp = global_occ.get(x3, 0)
            ft_x1 = global_occ.get(x1, 0)
            ft_x2 = global_occ.get(x2, 0)

            if ft_x1 > 0 and (fp / ft_x1) >= self.ios_threshold and x1 not in self.special_tokens and len(x1) > 1:
                removal_candidates.add(x1)
            if ft_x2 > 0 and (fp / ft_x2) >= self.ios_threshold and x2 not in self.special_tokens and len(x2) > 1:
                removal_candidates.add(x2)

            if self.verbose and (step_idx % self.progress_every_merges == 0):
                print(f"[ios]   merges evaluated: {step_idx}/{len(self.merges)} | candidates={len(removal_candidates)}")

        self.removal_set = removal_candidates
        self.removal_meta = {
            t: {
                "token": t,
                "reason": "ios",
                "ios_threshold": self.ios_threshold,
                "parents": list(self.parents.get(t, (None, None))),
                "creation_step": int(self.creation_step.get(t, -1)),
                "approx_ft": int(global_occ.get(t, 0)),
            }
            for t in sorted(self.removal_set, key=lambda s: self.creation_step.get(s, 10**9))
        }

        if self.verbose:
            kept = len(self.vocab) - len(self.removal_set)
            print(f"[ios]   Phase 3 complete: removed {len(self.removal_set)} tokens (kept {kept}). "
                  f"Time={time.time() - t_phase3:.2f}s")

    # -------------------------------------------------------------------------
    # Post-processing at inference
    # -------------------------------------------------------------------------

    def _postprocess_split_only(self, tokens: List[str]) -> List[str]:
        out: List[str] = []
        if self.verbose:
            print(f"[post/split] Start split-only for {len(tokens)} tokens…")
        for i, t in enumerate(tokens, 1):
            out.extend(self._split_removed_token(t))
            if self.verbose and (i % (10 * self.progress_every_lines) == 0):
                print(f"[post/split]   processed {i}/{len(tokens)} input tokens…")
        return out

    def _postprocess_split_with_remerge(self, tokens: List[str]) -> List[str]:
        # 1) split pass
        if self.verbose:
            print(f"[post/split_remerge] Step 1/2: split removed tokens…")
        seq: List[str] = []
        for i, t in enumerate(tokens, 1):
            seq.extend(self._split_removed_token(t))
            if self.verbose and (i % (10 * self.progress_every_lines) == 0):
                print(f"[post/split_remerge]   split progress {i}/{len(tokens)}")

        # 2) local forward re-merge
        if self.verbose:
            print(f"[post/split_remerge] Step 2/2: local forward re-merge (window={self.remerge_window}) "
                  f"on {len(seq)} tokens…")
        blocked_new: Set[str] = set(self.removal_set)

        def apply_merges_window(arr: List[str], L: int, R: int) -> int:
            applied = 0
            for (a, b) in self.merges:
                new_tok = a + b
                if new_tok in blocked_new:
                    continue  # don't recreate removed tokens
                j = max(L, 0)
                R2 = min(R, len(arr))
                while j + 1 < R2:
                    if arr[j] == a and arr[j + 1] == b:
                        arr[j : j + 2] = [new_tok]
                        R2 -= 1
                        applied += 1
                    else:
                        j += 1
            return applied

        i = 0
        N = len(seq)
        total_applied = 0
        last_log = time.time()
        while i < N:
            L = max(0, i - self.remerge_window)
            R = min(N, i + self.remerge_window + 1)
            applied = apply_merges_window(seq, L, R)
            total_applied += applied
            i = R
            N = len(seq)
            if self.verbose and ((i % (50 * self.remerge_window) == 0) or (time.time() - last_log) > 5.0):
                print(f"[post/split_remerge]   window end@{i}/{N} | merges_applied_so_far={total_applied}")
                last_log = time.time()

        if self.verbose:
            print(f"[post/split_remerge]   total merges applied during re-merge: {total_applied}")
        return seq

    def _split_removed_token(self, token: str) -> List[str]:
        """
        Split a removed token into allowed pieces.
        Priority:
          1) If token not removed → keep.
          2) If parents known → use parents recursively.
          3) Else → DP segmentation fallback into allowed vocab tokens (not removed).
          4) If DP fails → keep token (last resort).
        """
        if token in self.special_tokens or token not in self.removal_set:
            return [token]

        # Parent-based split
        if token in self.parents:
            left, right = self.parents[token]
            return self._split_removed_token(left) + self._split_removed_token(right)

        # Fallback DP segmentation (avoid removed tokens)
        pieces = self._dp_segment(token, forbid=self.removal_set)
        if pieces:
            return pieces

        # Last resort: keep as-is
        return [token]

    # -------------------------------------------------------------------------
    # Utilities: parents/decomposition/IO
    # -------------------------------------------------------------------------

    def _build_parents_map_robust(self) -> None:
        """
        Two-pass parent reconstruction:
          Pass A: direct new_tok = left+right (fast, when vocab matches concat)
          Pass B: for each vocab token t with len>1 and no parent yet,
                  scan splits t[:k]/t[k:], and if (left,right) is a known merge
                  and both sides exist in vocab, accept the best-ranked pair.
        """
        t0 = time.time()
        parents = {}
        creation = {}

        # Pass A: direct concat
        direct_hits = 0
        for idx, (left, right) in enumerate(self.merges):
            new_tok = left + right
            if new_tok in self.vocab:
                parents[new_tok] = (left, right)
                creation[new_tok] = idx
                direct_hits += 1

        # Pass B: split search if coverage is low
        vocab_set = set(self.vocab.keys())
        merges_set = set(self.rank.keys())
        split_hits = 0

        if direct_hits < len(self.vocab) * 0.2:
            for t in self.vocab.keys():
                if t in parents or t in self.special_tokens or len(t) <= 1:
                    continue
                best_pair = None
                best_rank = -1
                L = len(t)
                for k in range(1, L):
                    left = t[:k]
                    right = t[k:]
                    if left in vocab_set and right in vocab_set and (left, right) in merges_set:
                        r = self.rank[(left, right)]
                        if r > best_rank:
                            best_rank = r
                            best_pair = (left, right)
                if best_pair:
                    parents[t] = best_pair
                    creation[t] = best_rank
                    split_hits += 1

        self.parents = parents
        self.creation_step = creation
        self.base_tokens = set(self.vocab.keys()) - set(self.parents.keys())

        if self.verbose:
            print(f"[parents {_now()}] Direct hits: {direct_hits} | Split hits: {split_hits} | "
                  f"Total parents: {len(self.parents)} | Base tokens: {len(self.base_tokens)} "
                  f"| Time={time.time() - t0:.2f}s")
            if self.debug_parents:
                n = 0
                for tok, (l, r) in self.parents.items():
                    print(f"[parents] {tok!r} <- ({l!r}, {r!r}) step={self.creation_step.get(tok,-1)}")
                    n += 1
                    if n >= 10:
                        break

    def _extract_special_tokens(self, tk_config: Dict) -> Set[str]:
        specials = set()
        for item in tk_config.get("added_tokens", []):
            if item.get("special", False):
                content = item.get("content", None)
                if content is not None:
                    specials.add(content)
        return specials

    def _dp_segment(self, token: str, forbid: Set[str]) -> Optional[List[str]]:
        """
        Segment `token` into a list of vocab tokens not in `forbid` by DP.
        Returns None if no segmentation exists.
        """
        if token in self._dp_cache:
            return self._dp_cache[token]

        vocab_allowed = set(self.vocab.keys()) - set(forbid)

        n = len(token)
        best: List[Optional[List[str]]] = [None] * (n + 1)
        best[0] = []

        # Limit sub-token length window to avoid O(n^2) blowups on long strings
        MAX_SEG = 50

        for i in range(1, n + 1):
            chosen = None
            j_start = max(0, i - MAX_SEG)
            for j in range(j_start, i):
                if best[j] is None:
                    continue
                sub = token[j:i]
                if sub in vocab_allowed:
                    cand = best[j] + [sub]
                    if (chosen is None) or (len(cand) < len(chosen)):
                        chosen = cand
            best[i] = chosen

        self._dp_cache[token] = best[n]
        return best[n]

    def _save_removal_json(self) -> None:
        t0 = time.time()
        data = {
            "mode": self.mode,
            "threshold": {
                "type": "count" if self.mode == "count" else "ios",
                "value": self.count_threshold if self.mode == "count" else self.ios_threshold,
            },
            "removed": [self.removal_meta[t] for t in sorted(self.removal_set)],
        }
        with open(self.removal_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if self.verbose:
            print(f"[save] Wrote {len(self.removal_set)} removals to {self.removal_json} "
                  f"in {time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # CLI helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _read_lines(path: str, limit: Optional[int] = None) -> Iterable[str]:
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")
                n += 1
                if limit is not None and n >= limit:
                    return


def _cli():
    ap = argparse.ArgumentParser(
        description="Pseudo‑Picky BPE: robust merges parsing, parents mapping, IoS (bigram/struct), DP fallback, .txt inference."
    )
    ap.add_argument("--tokenizer", required=True, help="Path to HF tokenizer.json (BPE).")
    ap.add_argument("--train", required=False, help="Path to training text.")
    ap.add_argument("--mode", choices=["count", "ios", "ios_struct"], default="count",
                    help="Removal mode: count, ios (bigram IoS), or ios_struct (structural IoS).")
    ap.add_argument("--count-threshold", type=int, default=None, help="Min freq to keep (mode=count).")
    ap.add_argument("--ios-threshold", type=float, default=0.9, help="IoS threshold (mode=ios / ios_struct).")
    ap.add_argument("--removal", default="token_removal.json", help="Removal JSON path (save & load).")
    ap.add_argument("--sample-lines", type=int, default=None, help="Use only first N lines of training text.")
    ap.add_argument("--postprocess", choices=["split", "split_remerge"], default="split",
                    help="Inference postprocess mode.")
    ap.add_argument("--remerge-window", type=int, default=16, help="Local window size for split_remerge.")
    ap.add_argument("--verbose", action="store_true", help="Enable progress logging.")
    ap.add_argument("--quiet", action="store_true", help="Disable all logging (overrides --verbose).")
    ap.add_argument("--progress-every-lines", type=int, default=10_000,
                    help="Log progress every N lines when reading corpus or txt inference.")
    ap.add_argument("--progress-every-merges", type=int, default=5_000,
                    help="Log progress every N merges when scanning merges.")
    ap.add_argument("--debug-parents", action="store_true", help="Print sample parent mappings.")

    # Inference
    ap.add_argument("--text", default=None, help="If set, tokenize this text and print tokens.")
    ap.add_argument("--input-file", default=None, help="Path to a .txt file with one text per line for inference.")
    ap.add_argument("--output-file", default=None, help="Optional path to write outputs (stdout if omitted).")
    ap.add_argument("--format", choices=["jsonl", "txt"], default="jsonl",
                    help="Output format for --input-file inference. Default=jsonl.")
    ap.add_argument("--show-line-num", action="store_true",
                    help="If set and --format=txt, prefix each line with the 1-based line number and a tab.")

    args = ap.parse_args()
    verbose = False if args.quiet else (True if args.verbose else False)

    pp = PseudoPickyBPE(
        tokenizer_json=args.tokenizer,
        training_text=args.train,
        mode=args.mode,
        count_threshold=args.count_threshold,
        ios_threshold=args.ios_threshold,
        removal_json=args.removal,
        sample_lines=args.sample_lines,
        postprocess_mode=args.postprocess,
        remerge_window=args.remerge_window,
        verbose=verbose,
        progress_every_lines=args.progress_every_lines,
        progress_every_merges=args.progress_every_merges,
        debug_parents=args.debug_parents,
    )

    # Build removal list if training file is provided
    if args.train:
        pp.build_removals()
    else:
        # Load existing removal json if exists; otherwise warn
        if os.path.exists(args.removal):
            pp.load_removals(args.removal)
        else:
            if verbose:
                print("[warn] No training file and no removal JSON; tokenization will be vanilla.")

    # Inference modes
    if args.input_file:
        pp.tokenize_file(
            input_path=args.input_file,
            output_path=args.output_file,
            output_format=args.format,
            show_line_num=args.show_line_num,
        )
    elif args.text is not None:
        toks = pp.tokenize(args.text)
        print("TOKENS:", toks)
    else:
        if verbose:
            print("[info] Nothing to do: provide --train to build removals, --text for single string, "
                  "or --input-file for batch .txt inference.")


if __name__ == "__main__":
    _cli()