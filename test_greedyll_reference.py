# -*- coding: utf-8 -*-
"""
Reference implementation of GreedyLL BPE (exact and approx) in pure Python.
No lazy heap - recomputes all scores from scratch each iteration.
This is slow but guaranteed correct, used to validate the Rust implementation.

Usage:
    python test_greedyll_reference.py

Compares the first N merges from:
  1. Pure Python reference (no optimizations)
  2. Rust tokenizers library

Uses the same ByteLevel pre-tokenization as fineweb_test.py.
"""

import os
import json
import math
import time
from collections import Counter
from typing import Dict, List, Tuple, Optional, Iterable
from dataclasses import dataclass

# ---------- CONFIG ----------
DATASET_PATH = "/Users/ahmetcanyavuz/Developer/tokenizers/fineweb_data/fineweb_en_sentences.txt"
NUM_MERGES = 1000  # Number of merges to compare
MIN_FREQ = 2
MAX_LINES = 100000  # Limit lines for faster testing (None = all)
# ----------------------------

# We need the tokenizers library for pre-tokenization and comparison
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.decoders import ByteLevel as ByteLevelDec


def xlogx(x: int) -> float:
    """x * log(x), with 0*log(0) = 0."""
    if x == 0:
        return 0.0
    return x * math.log(x)


def delta_ll_exact(nb: int, nc: int, nbc: int, n: int) -> float:
    """
    Exact change in log-likelihood for merging pair (b,c).

    ΔLL(b,c) = (nb - nbc)log(nb - nbc) - nb log nb
             + (nc - nbc)log(nc - nbc) - nc log nc
             + nbc log nbc
             - (N - nbc)log(N - nbc) + N log N
    """
    return (
        xlogx(nb - nbc) - xlogx(nb)
        + xlogx(nc - nbc) - xlogx(nc)
        + xlogx(nbc)
        - xlogx(n - nbc) + xlogx(n)
    )


def delta_ll_approx(nb: int, nc: int, nbc: int, n: int) -> float:
    """
    PMI-like approximation: nbc * log(nbc * N / (nb * nc))
    """
    if nbc == 0 or nb == 0 or nc == 0 or n == 0:
        return 0.0
    return nbc * math.log((nbc * n) / (nb * nc))


@dataclass
class MergeResult:
    """Result of a single merge operation."""
    pair: Tuple[int, int]
    new_id: int
    count: int
    score: float
    delta_ll_exact: float
    delta_ll_approx: float
    total_tokens: int = 0  # N at time of merge
    n_a: int = 0  # symbol count for left symbol
    n_b: int = 0  # symbol count for right symbol


class ReferenceBPE:
    """
    Pure Python reference implementation of BPE with GreedyLL scoring.
    No optimizations - recomputes everything from scratch each iteration.
    """

    def __init__(self, score_by: str = "count", min_frequency: int = 2):
        """
        Args:
            score_by: "count", "exact_ll", or "approx_ll"
            min_frequency: minimum pair count to consider for merging
        """
        self.score_by = score_by
        self.min_frequency = min_frequency

        # Vocabulary: token_id -> token_str (single characters from ByteLevel encoding)
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}

        # Training state
        self.words: List[List[int]] = []  # Each word is a list of token IDs
        self.word_counts: List[int] = []  # Count for each word
        self.merges: List[MergeResult] = []

    def _add_token(self, token: str) -> int:
        """Add a token to vocabulary, return its ID."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        new_id = len(self.id_to_token)
        self.id_to_token[new_id] = token
        self.token_to_id[token] = new_id
        return new_id

    def _init_vocab_from_pretokenized(self, pretokenized_words: Dict[str, int]):
        """
        Initialize vocabulary from pre-tokenized words.
        Each word is a string from ByteLevel pre-tokenizer (already Unicode chars).
        """
        # First pass: collect all unique characters and build initial vocab
        for word_str in pretokenized_words.keys():
            for ch in word_str:
                self._add_token(ch)

        # Second pass: convert words to token ID sequences
        for word_str, count in pretokenized_words.items():
            token_ids = [self.token_to_id[ch] for ch in word_str]
            self.words.append(token_ids)
            self.word_counts.append(count)

    def _compute_pair_counts(self) -> Counter:
        """Compute counts for all adjacent pairs in the corpus."""
        pair_counts = Counter()
        for word, count in zip(self.words, self.word_counts):
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += count
        return pair_counts

    def _compute_symbol_counts(self) -> Tuple[Counter, int]:
        """Compute marginal counts for each symbol and total token count."""
        sym_counts = Counter()
        total = 0
        for word, count in zip(self.words, self.word_counts):
            for token_id in word:
                sym_counts[token_id] += count
            total += len(word) * count
        return sym_counts, total

    def _compute_score(self, pair: Tuple[int, int], pair_count: int,
                       sym_counts: Counter, total_tokens: int) -> float:
        """Compute score for a pair based on scoring policy."""
        if self.score_by == "count":
            return float(pair_count)

        nb = sym_counts[pair[0]]
        nc = sym_counts[pair[1]]

        if self.score_by == "exact_ll":
            return delta_ll_exact(nb, nc, pair_count, total_tokens)
        elif self.score_by == "approx_ll":
            return delta_ll_approx(nb, nc, pair_count, total_tokens)
        else:
            raise ValueError(f"Unknown score_by: {self.score_by}")

    def _find_best_pair(self, pair_counts: Counter, sym_counts: Counter,
                        total_tokens: int) -> Optional[Tuple[Tuple[int, int], int, float]]:
        """
        Find the best pair to merge (highest score, meeting min_frequency).
        Returns (pair, count, score) or None if no valid pair.
        """
        best_pair = None
        best_count = 0
        best_score = float('-inf')

        for pair, count in pair_counts.items():
            if count < self.min_frequency:
                continue

            score = self._compute_score(pair, count, sym_counts, total_tokens)

            # Tie-breaking: prefer higher count, then lexicographically smaller pair
            if (score > best_score or
                (score == best_score and count > best_count) or
                (score == best_score and count == best_count and pair < best_pair)):
                best_score = score
                best_count = count
                best_pair = pair

        if best_pair is None:
            return None
        return best_pair, best_count, best_score

    def _apply_merge(self, pair: Tuple[int, int], new_id: int):
        """Apply a merge to all words in the corpus."""
        a, b = pair
        for i, word in enumerate(self.words):
            new_word = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and word[j] == a and word[j + 1] == b:
                    new_word.append(new_id)
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            self.words[i] = new_word

    def train(self, pretokenized_words: Dict[str, int], num_merges: int,
              verbose: bool = True) -> List[MergeResult]:
        """
        Train BPE for a fixed number of merges.

        Args:
            pretokenized_words: dict mapping pre-tokenized word strings to counts
            num_merges: number of merges to perform
            verbose: print progress

        Returns:
            List of MergeResult objects
        """
        # Initialize vocabulary from pre-tokenized words
        self._init_vocab_from_pretokenized(pretokenized_words)

        if verbose:
            print(f"Initial vocab size: {len(self.id_to_token)}")
            print(f"Number of unique words: {len(self.words)}")
            print(f"Score by: {self.score_by}")
            # Compute initial stats
            sym_counts, total_tokens = self._compute_symbol_counts()
            print(f"Total tokens (N): {total_tokens}")

        self.merges = []

        for step in range(num_merges):
            # Recompute everything from scratch (reference implementation)
            pair_counts = self._compute_pair_counts()
            sym_counts, total_tokens = self._compute_symbol_counts()

            # Find best pair
            result = self._find_best_pair(pair_counts, sym_counts, total_tokens)
            if result is None:
                if verbose:
                    print(f"No valid pairs left at step {step}")
                break

            pair, count, score = result

            # Create new token
            new_token = self.id_to_token[pair[0]] + self.id_to_token[pair[1]]
            new_id = self._add_token(new_token)

            # Compute both ΔLL values for telemetry
            nb = sym_counts[pair[0]]
            nc = sym_counts[pair[1]]
            dll_exact = delta_ll_exact(nb, nc, count, total_tokens)
            dll_approx = delta_ll_approx(nb, nc, count, total_tokens)

            # Record merge
            merge_result = MergeResult(
                pair=pair,
                new_id=new_id,
                count=count,
                score=score,
                delta_ll_exact=dll_exact,
                delta_ll_approx=dll_approx,
                total_tokens=total_tokens,
                n_a=nb,
                n_b=nc,
            )
            self.merges.append(merge_result)

            # Apply merge
            self._apply_merge(pair, new_id)

            # Debug: show N after first few merges
            if verbose and step < 5:
                new_sym_counts, new_total = self._compute_symbol_counts()
                new_pair_counts = self._compute_pair_counts()
                print(f"  After merge {step}: N changed from {total_tokens} to {new_total} (delta={total_tokens - new_total})")
                # Show stats for key pairs after first merge
                if step == 0 and self.score_by == "exact_ll":
                    # Look for pairs involving the new token
                    for (p0, p1), pcount in sorted(new_pair_counts.items(), key=lambda x: -x[1])[:10]:
                        t0, t1 = self.id_to_token[p0], self.id_to_token[p1]
                        n0, n1 = new_sym_counts[p0], new_sym_counts[p1]
                        score = self._compute_score((p0, p1), pcount, new_sym_counts, new_total)
                        print(f"    Top pair: '{t0}'+'{t1}' count={pcount}, n0={n0}, n1={n1}, N={new_total}, score={score:.2f}")

            if verbose and (step + 1) % 100 == 0:
                print(f"Step {step + 1}: merged {self.id_to_token[pair[0]]!r} + "
                      f"{self.id_to_token[pair[1]]!r} -> {new_token!r} "
                      f"(count={count}, score={score:.4f})")

        if verbose:
            print(f"Final vocab size: {len(self.id_to_token)}")
            print(f"Total merges: {len(self.merges)}")

        return self.merges


def pretokenize_corpus(lines: Iterable[str], max_lines: Optional[int] = None) -> Dict[str, int]:
    """
    Pre-tokenize corpus using ByteLevel pre-tokenizer (same as fineweb_test.py).
    Returns dict mapping pre-tokenized words to their counts.
    """
    # Use tokenizers library for pre-tokenization to ensure exact match
    pre_tok = ByteLevelPre(add_prefix_space=True)

    word_counts: Counter = Counter()

    for i, line in enumerate(lines):
        if max_lines is not None and i >= max_lines:
            break

        # Pre-tokenize
        pretokenized = pre_tok.pre_tokenize_str(line)
        for word, _ in pretokenized:
            word_counts[word] += 1

    return dict(word_counts)


def train_rust_bpe(lines: Iterable[str], max_lines: Optional[int], num_merges: int,
                   score_by: str, min_frequency: int) -> Tuple[List[dict], Dict[int, str]]:
    """
    Train BPE using the Rust tokenizers library on original lines (not pre-tokenized).
    Returns (list of merge events from telemetry, id_to_token mapping).
    """
    # Create tokenizer with BPE model
    model = BPE(byte_fallback=True)
    tok = Tokenizer(model)
    tok.pre_tokenizer = ByteLevelPre(add_prefix_space=True)
    tok.decoder = ByteLevelDec()

    # We need vocab_size = initial_vocab + num_merges
    # Initial vocab is 256 bytes + special handling
    vocab_size = 256 + num_merges + 100  # Add buffer for safety

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=False,
        score_by=score_by,
        stop_by="vocab_size",
        track_ll=True,
        score_snapshot_every=0,  # Disable snapshots for this test
    )

    # Collect lines (with limit)
    collected_lines = []
    for i, line in enumerate(lines):
        if max_lines is not None and i >= max_lines:
            break
        collected_lines.append(line)

    tok.train_from_iterator(collected_lines, trainer=trainer)

    # Get telemetry
    telemetry = tok.model.telemetry()
    if telemetry is None:
        return [], {}

    merge_trace = telemetry.get("merge_trace", [])

    # Get vocabulary for token ID -> string mapping
    vocab = tok.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    return merge_trace[:num_merges], id_to_token


def line_iterator(path: str) -> Iterable[str]:
    """Iterate over non-empty lines in a file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


def compare_merges(ref_merges: List[MergeResult], rust_merges: List[dict],
                   ref_bpe: ReferenceBPE, rust_id_to_token: Dict[int, str]) -> Tuple[int, List[str]]:
    """
    Compare merge sequences from reference and Rust implementations.
    Returns (num_matching, list_of_differences).
    """
    differences = []
    num_matching = 0

    min_len = min(len(ref_merges), len(rust_merges))

    for i in range(min_len):
        ref = ref_merges[i]
        rust = rust_merges[i]

        # Get token strings for comparison
        ref_a = ref_bpe.id_to_token[ref.pair[0]]
        ref_b = ref_bpe.id_to_token[ref.pair[1]]
        ref_merged = ref_a + ref_b

        rust_pair = rust.get("pair", (None, None))
        rust_a = rust_id_to_token.get(rust_pair[0], "?") if rust_pair[0] is not None else "?"
        rust_b = rust_id_to_token.get(rust_pair[1], "?") if rust_pair[1] is not None else "?"
        rust_merged = rust_a + rust_b

        # Compare by merged token string and count
        # Note: scores can have small floating-point precision differences between
        # Python and Rust implementations, so we only compare the merge identity
        ref_score = ref.score
        rust_score = rust.get("score", 0)

        ref_count = ref.count
        rust_count = rust.get("count", 0)

        # Check if this merge matches (by token pair and count only)
        token_match = ref_merged == rust_merged
        count_match = ref_count == rust_count

        # Get telemetry data for debugging
        ref_N = ref.total_tokens
        rust_N = rust.get("total_tokens") or 0  # Handle None from JSON null
        ref_na = ref.n_a
        rust_na = rust.get("n_a") or 0
        ref_nb = ref.n_b
        rust_nb = rust.get("n_b") or 0

        if token_match and count_match:
            num_matching += 1
            # Check for N/symbol count drift (for debugging)
            if rust_N and ref_N != rust_N:
                print(f"  N DRIFT at step {i}: ref_N={ref_N} rust_N={rust_N} (diff={ref_N - rust_N})")
            if rust_na and ref_na != rust_na:
                print(f"  n_a DRIFT at step {i}: ref={ref_na} rust={rust_na} (diff={ref_na - rust_na})")
            if rust_nb and ref_nb != rust_nb:
                print(f"  n_b DRIFT at step {i}: ref={ref_nb} rust={rust_nb} (diff={ref_nb - rust_nb})")
        else:
            diff = (f"Step {i}: "
                   f"ref='{ref_a}'+'{ref_b}'->'{ref_merged}' count={ref_count} score={ref_score:.4f} | "
                   f"rust='{rust_a}'+'{rust_b}'->'{rust_merged}' count={rust_count} score={rust_score:.4f}")
            differences.append(diff)
            if len(differences) <= 10:  # Only show first 10 differences
                print(f"  DIFF: {diff}")
                # Also print N and symbol counts for first mismatch
                if len(differences) == 1:
                    print(f"    ref: N={ref_N}, n_a={ref_na}, n_b={ref_nb}")
                    print(f"    rust: N={rust_N}, n_a={rust_na}, n_b={rust_nb}")

    return num_matching, differences


def main():
    print("=" * 70)
    print("Reference GreedyLL BPE Implementation Test")
    print("=" * 70)

    # Check if dataset exists
    if not os.path.isfile(DATASET_PATH):
        print(f"Dataset not found: {DATASET_PATH}")
        return

    print(f"\nDataset: {DATASET_PATH}")
    print(f"Max lines: {MAX_LINES}")
    print(f"Num merges to compare: {NUM_MERGES}")
    print(f"Min frequency: {MIN_FREQ}")

    # Pre-tokenize for reference implementation
    print(f"\n[Preprocessing] Loading and pre-tokenizing corpus...")
    t0 = time.time()
    pretokenized = pretokenize_corpus(line_iterator(DATASET_PATH), max_lines=MAX_LINES)
    t1 = time.time()
    print(f"Pre-tokenization took {t1-t0:.2f}s")
    print(f"Unique pre-tokenized words: {len(pretokenized)}")
    print(f"Total word occurrences: {sum(pretokenized.values())}")

    # Test each scoring method
    for score_by in ["count", "exact_ll", "approx_ll"]:
        print("\n" + "=" * 70)
        print(f"Testing score_by = '{score_by}'")
        print("=" * 70)

        # Train reference implementation
        print(f"\n[1] Training REFERENCE implementation ({NUM_MERGES} merges)...")
        ref_bpe = ReferenceBPE(score_by=score_by, min_frequency=MIN_FREQ)
        t0 = time.time()
        ref_merges = ref_bpe.train(pretokenized, num_merges=NUM_MERGES, verbose=True)
        t1 = time.time()
        print(f"    Reference took {t1-t0:.2f}s, produced {len(ref_merges)} merges")

        # Show first 5 merges from reference
        print("    First 5 reference merges:")
        for i, m in enumerate(ref_merges[:5]):
            a = ref_bpe.id_to_token[m.pair[0]]
            b = ref_bpe.id_to_token[m.pair[1]]
            print(f"      {i}: '{a}' + '{b}' -> '{a+b}' count={m.count} score={m.score:.4f} "
                  f"dll_exact={m.delta_ll_exact:.4f}")

        # Train Rust implementation (on original lines, not pre-tokenized)
        print(f"\n[2] Training RUST implementation ({NUM_MERGES} merges)...")
        t0 = time.time()
        rust_merges, rust_id_to_token = train_rust_bpe(
            line_iterator(DATASET_PATH), MAX_LINES, NUM_MERGES, score_by, MIN_FREQ
        )
        t1 = time.time()
        print(f"    Rust took {t1-t0:.2f}s, produced {len(rust_merges)} merges")

        # Show first 5 merges from Rust
        print("    First 5 Rust merges:")
        for i, m in enumerate(rust_merges[:5]):
            pair = m.get("pair", (None, None))
            a = rust_id_to_token.get(pair[0], "?") if pair[0] is not None else "?"
            b = rust_id_to_token.get(pair[1], "?") if pair[1] is not None else "?"
            print(f"      {i}: '{a}' + '{b}' -> '{a+b}' count={m.get('count')} "
                  f"score={m.get('score', 0):.4f} delta_ll={m.get('delta_ll', 0):.4f}")
            if i == 0:
                print(f"         [DEBUG] Rust merge keys: {list(m.keys())}")

        # Compare
        print(f"\n[3] Comparing merge sequences...")
        num_match, diffs = compare_merges(ref_merges, rust_merges, ref_bpe, rust_id_to_token)
        total = min(len(ref_merges), len(rust_merges))

        if num_match == total:
            print(f"    SUCCESS: All {total} merges match!")
        else:
            print(f"    MISMATCH: {num_match}/{total} merges match")
            print(f"    Total differences: {len(diffs)}")

        # Save detailed comparison to file
        output_file = f"comparison_{score_by}.json"
        comparison_data = {
            "score_by": score_by,
            "num_merges_requested": NUM_MERGES,
            "ref_merges": len(ref_merges),
            "rust_merges": len(rust_merges),
            "num_matching": num_match,
            "ref_first_10": [
                {
                    "step": i,
                    "pair_str": [ref_bpe.id_to_token[m.pair[0]], ref_bpe.id_to_token[m.pair[1]]],
                    "merged": ref_bpe.id_to_token[m.pair[0]] + ref_bpe.id_to_token[m.pair[1]],
                    "count": m.count,
                    "score": m.score,
                    "delta_ll_exact": m.delta_ll_exact,
                    "delta_ll_approx": m.delta_ll_approx,
                }
                for i, m in enumerate(ref_merges[:10])
            ],
            "rust_first_10": [
                {
                    "step": i,
                    "pair": m.get("pair"),
                    "pair_str": [
                        rust_id_to_token.get(m.get("pair", (None, None))[0], "?"),
                        rust_id_to_token.get(m.get("pair", (None, None))[1], "?"),
                    ] if m.get("pair") else None,
                    "count": m.get("count"),
                    "score": m.get("score"),
                    "delta_ll": m.get("delta_ll"),
                }
                for i, m in enumerate(rust_merges[:10])
            ],
        }
        with open(output_file, "w") as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        print(f"    Saved comparison to {output_file}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
