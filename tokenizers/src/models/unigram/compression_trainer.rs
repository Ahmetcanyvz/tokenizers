//! Greedy compression-based trainer for the Unigram model.
//!
//! This trainer minimizes the total number of tokens under **unit-cost decoding**.
//! It iteratively deletes one token at a time, always picking the token `t` that minimizes
//!
//!   ΔL(t) = c[t] * ( d[t] - 1 )
//!
//! where
//!   - `c[t]` is the current corpus count of `t` in the best (unit-cost) segmentations,
//!   - `d[t]` is the shortest decomposition length of the *string of `t`* using `V \ {t}`.
//!
//! After deleting a token, we **only re-segment the sentences that used it**, and we
//! **only recompute d[·]** for tokens whose decomposition depended on it.
//!
//! Notes:
//! - We realize *unit-cost decoding* by giving **every token the same negative score** (e.g. `-1.0`).
//!   Viterbi maximizes the total score, so with per-token cost `-1.0`, it prefers **fewer tokens**.
//! - We don't require any change to `Lattice` or `Trie` by filtering disallowed nodes
//!   out of the lattice **after** populating it.
//!
//! This file is self-contained w.r.t. lattice/trie implementations.

use crate::models::unigram::{lattice::Lattice, model::Unigram};
use crate::tokenizer::{AddedToken, Result, Trainer};
use crate::utils::parallelism::*;
use crate::utils::progress::{ProgressBar, ProgressStyle};

use ahash::{AHashMap, AHashSet};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;

// ----------------------------- Local types ------------------------------------

/// A full sentence/span and its count within the dataset
type Sentence = (String, u32);

/// A token candidate and a (dummy) score
type SentencePiece = (String, f64);

// ----------------------------- Trainer ----------------------------------------

/// Trainer for greedy compression-based Unigram.
/// Configuration mirrors the style of `UnigramTrainer` but replaces EM with greedy deletion.
#[derive(Builder, Debug, Clone, Serialize, Deserialize)]
pub struct CompressionTrainer {
    /// Show progress bars while training
    #[builder(default = "true")]
    pub show_progress: bool,

    /// Target vocabulary size (including special tokens)
    #[builder(default = "8000")]
    pub vocab_size: u32,

    /// Max piece length considered when seeding multi-char candidates
    #[builder(default = "16")]
    pub max_piece_length: usize,

    /// Special tokens to prepend in the vocabulary (kept; never deleted)
    #[builder(default = "vec![]")]
    pub special_tokens: Vec<AddedToken>,

    /// Characters to force-include in Σ (in addition to those seen in data)
    #[builder(default = "AHashSet::new()")]
    pub initial_alphabet: AHashSet<char>,

    /// Upper bound on seed list size (same default as UnigramTrainer)
    #[builder(default = "1_000_000")]
    pub seed_size: usize, // <-- public so Python bindings can access/get/set it

    /// If provided, training starts **exactly** from this vocabulary (scores forced to -1.0).
    /// You must include Σ yourself if you want single-char coverage guaranteed.
    #[builder(default)]
    pub seed_vocab: Option<Vec<String>>,

    /// Internal: word/span counts (populated by `feed`)
    #[builder(default = "AHashMap::new()")]
    words: AHashMap<String, u32>,
}

impl Default for CompressionTrainer {
    fn default() -> Self {
        Self::builder().build().expect("CompressionTrainer::default()")
    }
}

impl CompressionTrainer {
    /// Builder entry point, like other trainers in this crate.
    pub fn builder() -> CompressionTrainerBuilder {
        CompressionTrainerBuilder::default()
    }

    // ----------------------------- Progress helpers -----------------------------

    /// Setup a progress bar if asked to show progress
    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<36!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            Some(p)
        } else {
            None
        }
    }

    /// Update the progress bar with the new provided length and message
    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &'static str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
        }
    }

    /// Finish the progress bar
    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
            println!();
        }
    }

    // ----------------------------- Utilities -----------------------------------

    /// Required characters Σ = set of all characters seen in data + initial_alphabet.
    fn required_chars(&self, word_counts: &[Sentence]) -> AHashSet<String> {
        word_counts
            .iter()
            .flat_map(|(s, _)| s.chars())
            .chain(self.initial_alphabet.iter().copied())
            .map(|c| c.to_string())
            .collect()
    }

    /// Basic validity check for a candidate sentencepiece.
    fn is_valid_sentencepiece(&self, chars: &[char]) -> bool {
        let n = chars.len();
        if n == 0 || n > self.max_piece_length {
            return false;
        }
        true
    }

    /// Generate seed candidates: include Σ and frequent substrings (via suffix array).
    /// We mirror `UnigramTrainer::make_seed_sentence_pieces` in style.
    fn seed_pieces(&self, sentences: &[Sentence]) -> Vec<SentencePiece> {
        // Flatten sentences with a boundary char to prevent crossing spans
        let c_sentence_boundary = '\0';
        let boundary = c_sentence_boundary.to_string();

        // Collect char counts and build a flat string
        let total: usize = sentences
            .iter()
            .map(|(s, _)| s.chars().count())
            .sum::<usize>()
            + sentences.len();
        let mut flat = String::with_capacity(total);
        let mut all_chars: AHashMap<char, u32> = AHashMap::new();

        for (s, n) in sentences {
            if s.is_empty() {
                continue;
            }
            flat.push_str(s);
            // Keep the boundary to avoid cross-span substrings in suffix-array results
            flat.push_str(&boundary);
            // Count characters excluding boundary
            for c in s.chars() {
                if c != c_sentence_boundary {
                    *all_chars.entry(c).or_default() += *n;
                }
            }
        }
        flat.shrink_to_fit();

        // Compute substrings via suffix array (fast or pure-Rust fallback)
        #[cfg(feature = "esaxx_fast")]
        let suffix = esaxx_rs::suffix(&flat).expect("esaxx suffix failed");
        #[cfg(not(feature = "esaxx_fast"))]
        let suffix = esaxx_rs::suffix_rs(&flat).expect("esaxx_rs suffix_rs failed");

        // 1) Single characters, sorted by decreasing frequency
        let mut seed: Vec<SentencePiece> = Vec::with_capacity(self.seed_size);
        let mut sall_chars: Vec<(u32, char)> = all_chars.into_iter().map(|(c, f)| (f, c)).collect();
        // Reversed order by frequency
        sall_chars.sort_by_key(|&a| Reverse(a));
        for (_freq, ch) in sall_chars {
            seed.push((ch.to_string(), 0.0));
            if seed.len() >= self.seed_size {
                return seed;
            }
        }

        // 2) Multi-char substrings from suffix array, scored by freq * length (for ordering)
        //    Use `.iter()` (NOT `.into_iter()`) since `Suffix<T>` is not an iterator.
        let mut substr_index: Vec<_> = suffix
            .iter()
            .filter_map(|(string, freq)| {
                if string.len() <= 1 {
                    return None;
                }
                if string.contains(&c_sentence_boundary) {
                    return None;
                }
                if !self.is_valid_sentencepiece(string) {
                    return None;
                }
                // `freq` is a `u32`; no deref.
                let score = freq * (string.len() as u32);
                Some((score, string))
            })
            .collect();

        // sort by decreasing score (keep compatibility with unigram::trainer approach)
        substr_index.sort_by_key(|&a| Reverse(a));
        for (_score, char_string) in substr_index {
            // Just in case
            debug_assert!(self.is_valid_sentencepiece(char_string));
            // Build a String from `&[char]` / `&Vec<char>`
            let string: String = char_string.iter().copied().collect();
            seed.push((string, 0.0));
            if seed.len() >= self.seed_size {
                break;
            }
        }

        seed
    }

    /// Filter lattice nodes by an `allowed` mask on token ids.
    /// We prune both `begin_nodes` and `end_nodes` for every position. This avoids
    /// needing any private fields or methods from `Node`.
    fn filter_lattice_by_allowed_ids(lattice: &mut Lattice<'_>, allowed: &[bool]) {
        let len = lattice.len();
        for pos in 0..=len {
            lattice.begin_nodes[pos].retain(|node_rc| {
                let id = node_rc.borrow().id;
                if id < allowed.len() {
                    allowed[id]
                } else {
                    // Keep BOS/EOS (ids beyond vocab)
                    true
                }
            });
            lattice.end_nodes[pos].retain(|node_rc| {
                let id = node_rc.borrow().id;
                if id < allowed.len() {
                    allowed[id]
                } else {
                    true
                }
            });
        }
    }

    /// Run **unit-cost** Viterbi on `s` with a filter that disables some token ids.
    /// Returns the list of ids on the best path. With Σ present, this always returns non-empty
    /// unless `s` is empty.
    fn best_path_ids_with_filter<F>(model: &Unigram, s: &str, mut allow: F) -> Vec<usize>
    where
        F: FnMut(usize) -> bool,
    {
        // 1) Populate the lattice with all nodes
        let mut lattice = Lattice::from(s, model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);

        // 2) Build the allowed mask for current model ids
        let mut allowed = vec![true; model.len()];
        for id in 0..model.len() {
            allowed[id] = allow(id);
        }

        // 3) Remove disallowed nodes
        Self::filter_lattice_by_allowed_ids(&mut lattice, &allowed);

        // 4) Viterbi
        let path = lattice.viterbi();
        if path.is_empty() {
            return Vec::new();
        }
        path.into_iter().map(|n| n.borrow().id).collect()
    }

    /// Compute d[t] and its decomposition token ids (Dep[t]) for token `t` under the
    /// current disabled set. This runs a tiny unit-cost DP on the **string of token `t`**.
    fn compute_dt_with_deps(
        model: &Unigram,
        t: usize,
        disabled: &AHashSet<usize>,
    ) -> (usize, Vec<usize>) {
        let tok = &model.vocab[t].0;
        // Forbid `t` itself + all currently disabled tokens
        let ids = Self::best_path_ids_with_filter(model, tok, |id| id != t && !disabled.contains(&id));
        // With Σ present, ids.len() >= 1; guard against pathological cases anyway.
        let d = if ids.is_empty() { tok.chars().count().max(1) } else { ids.len() };
        (d, ids)
    }

    /// Build the initial Unigram model for training:
    /// - If `seed_vocab` is provided, use it **as-is** (plus special tokens), all scores `-1.0`.
    /// - Otherwise, start from Σ and substrings discovered via suffix array (scores `-1.0`).
    fn build_initial_model(
        &self,
        sentences: &[Sentence],
        required: &AHashSet<String>,
    ) -> Result<Unigram> {
        if let Some(seed) = &self.seed_vocab {
            // Use the provided seed exactly (plus special tokens), all scores = -1.0.
            let mut seen = AHashSet::new();
            let mut final_seed: Vec<(String, f64)> =
                Vec::with_capacity(seed.len() + self.special_tokens.len());

            // Special tokens first (kept forever)
            for t in &self.special_tokens {
                if seen.insert(t.content.clone()) {
                    final_seed.push((t.content.clone(), -1.0));
                }
            }
            // Then user-provided seed_vocab order
            for s in seed {
                if seen.insert(s.clone()) {
                    final_seed.push((s.clone(), -1.0));
                }
            }
            // No automatic Σ injection here: caller controls it explicitly via seed_vocab.
            let model = Unigram::from(final_seed, None, false)?;
            return Ok(model);
        }

        // Fallback: automatic seeding (Σ + substrings)
        let mut pieces = self.seed_pieces(sentences);

        // Ensure Σ is included first (deterministic order). Insert any missing required char at front.
        for ch in required.iter() {
            if !pieces.iter().any(|(s, _)| s == ch) {
                pieces.insert(0, (ch.clone(), 0.0));
            }
        }

        // Prepend special tokens (kept; never deleted)
        let mut specials: Vec<SentencePiece> = self
            .special_tokens
            .iter()
            .map(|t| (t.content.clone(), 0.0))
            .collect();
        specials.append(&mut pieces);

        // Convert every score to -1.0 so maximizing total score gives **minimal number of tokens**
        let final_seed: Vec<(String, f64)> = specials
            .into_iter()
            .map(|(s, _)| (s, -1.0))
            .collect();

        // No UNK during training; byte_fallback off.
        let model = Unigram::from(final_seed, /*unk_id*/ None, /*byte_fallback*/ false)?;
        Ok(model)
    }

    /// Main training logic; separated to keep `train` small.
    pub fn do_train(&self, sentences: Vec<Sentence>, model: &mut Unigram) -> Result<Vec<AddedToken>> {
        let progress = self.setup_progress();

        // 1) Prepare Σ and initial model
        let required = self.required_chars(&sentences); // Σ
        let m = self.build_initial_model(&sentences, &required)?;

        // Define non-deletable strings (Σ + special tokens)
        let non_deletable_strings: AHashSet<String> = required
            .iter()
            .cloned()
            .chain(self.special_tokens.iter().map(|t| t.content.clone()))
            .collect();

        // Identify ids belonging to Σ/specials
        let is_non_deletable = |id: usize, m: &Unigram| -> bool { non_deletable_strings.contains(&m.vocab[id].0) };

        // 2) Initial segmentation of the entire corpus (unit-cost via -1.0 per token)
        let n = sentences.len();
        self.update_progress(&progress, n, "Initial segmentation");
        let mut seg_tokens: Vec<Vec<usize>> = Vec::with_capacity(n); // per-sentence best path ids
        let mut seg_len: Vec<usize> = Vec::with_capacity(n);         // per-sentence path lengths
        // Global counts c[t] and inverted index D[t]
        let mut ct: Vec<u32> = vec![0; m.len()];
        let mut Dt: Vec<AHashSet<usize>> = vec![AHashSet::new(); m.len()];

        for (i, (s, cnt)) in sentences.iter().enumerate() {
            if let Some(p) = &progress { p.inc(1); }

            // Populate lattice and run Viterbi with *all* tokens enabled
            let mut lattice = Lattice::from(s, m.bos_id, m.eos_id);
            m.populate_nodes(&mut lattice);
            let path = lattice.viterbi();
            let ids: Vec<usize> = path.into_iter().map(|n| n.borrow().id).collect();

            seg_len.push(ids.len());
            for &id in &ids {
                ct[id] = ct[id].saturating_add(*cnt);
                Dt[id].insert(i);
            }
            seg_tokens.push(ids);
        }
        self.finalize_progress(&progress, n);

        // 3) Precompute d[t] and Dep[t] (dependencies), and build reverse dependency R[u]
        self.update_progress(&progress, m.len(), "Precompute d[t] & deps");
        let mut disabled: AHashSet<usize> = AHashSet::new();          // removed token ids
        let mut dt: Vec<usize> = vec![1; m.len()];                    // shortest decomposition length per token
        let mut deps: Vec<Vec<usize>> = vec![Vec::new(); m.len()];    // decomposition ids per token
        let mut R: Vec<AHashSet<usize>> = vec![AHashSet::new(); m.len()]; // reverse deps: u -> { t | u ∈ deps[t] }

        for t in 0..m.len() {
            if let Some(p) = &progress { p.inc(1); }
            if is_non_deletable(t, &m) {
                dt[t] = 1;
                deps[t].clear();
                continue;
            }
            let (d, used) = Self::compute_dt_with_deps(&m, t, &disabled);
            dt[t] = d;
            deps[t] = used.clone();
            for &u in &used {
                R[u].insert(t);
            }
        }
        self.finalize_progress(&progress, m.len());

        // 4) Iterative greedy deletion until we hit the target vocabulary size
        // We don't physically remove tokens; we mark them disabled and filter during DP.
        let mut enabled_count = m.len();
        let target = self.vocab_size as usize;
        if enabled_count <= target {
            // Nothing to delete; just return the current model
            *model = m.clone();
            return Ok(self.special_tokens.clone());
        }

        // Allow showing iteration count
        self.update_progress(&progress, enabled_count - target, "Greedy deletions");

        'outer: for _iter in 0..(enabled_count - target) {
            if let Some(p) = &progress { p.inc(1); }

            // 4.1) Pick the best token to delete: minimize ΔL(t) = c[t] * (d[t] - 1)
            let mut best_t: Option<usize> = None;
            let mut best_delta: u64 = u64::MAX;

            for t in 0..m.len() {
                if disabled.contains(&t) {
                    continue;
                }
                if is_non_deletable(t, &m) {
                    continue; // never delete Σ or specials
                }
                let c_t = ct[t] as u64;
                if c_t == 0 {
                    // Deleting an unused token is always safe and Δ=0
                    best_t = Some(t);
                    break;
                }
                let d_t = dt[t] as u64;
                let delta = c_t.saturating_mul(d_t.saturating_sub(1));
                if delta < best_delta {
                    best_delta = delta;
                    best_t = Some(t);
                }
            }

            let Some(t_star) = best_t else {
                // No deletable token remains
                break 'outer;
            };

            // 4.2) Disable t*
            disabled.insert(t_star);
            enabled_count -= 1;

            // 4.3) Resegment only sentences that used t*
            let affected: Vec<usize> = Dt[t_star].iter().copied().collect();
            for &s_idx in &affected {
                // Fetch sentence text & count without moving them
                let s: &str = &sentences[s_idx].0;
                let cnt: u32 = sentences[s_idx].1;

                // Remove old counts for this sentence
                for &old_id in &seg_tokens[s_idx] {
                    ct[old_id] = ct[old_id].saturating_sub(cnt);
                }
                // Remove sentence from all old D[·]
                for &old_id in &seg_tokens[s_idx] {
                    Dt[old_id].remove(&s_idx);
                }

                // Recompute best path with filtered tokens (exclude disabled set)
                let new_ids = Self::best_path_ids_with_filter(&m, s, |id| !disabled.contains(&id));

                // Update per-sentence path and length
                seg_len[s_idx] = new_ids.len();
                seg_tokens[s_idx] = new_ids.clone();

                // Add new counts and update inverted index
                for &id in &new_ids {
                    ct[id] = ct[id].saturating_add(cnt);
                    Dt[id].insert(s_idx);
                }
            }
            // Clear D[t*] after processing its sentences
            Dt[t_star].clear();

            // 4.4) Recompute d[·] only for tokens whose decomposition used t*
            let impacted: Vec<usize> = R[t_star].iter().copied().collect();
            for t in impacted {
                // Remove old reverse links: t depended on deps[t]
                for &u in &deps[t] {
                    R[u].remove(&t);
                }
                // Recompute d[t] and deps[t] under the new disabled set
                let (d, used) = Self::compute_dt_with_deps(&m, t, &disabled);
                dt[t] = d;
                deps[t] = used.clone();
                // Add new reverse links
                for &u in &used {
                    R[u].insert(t);
                }
            }
            // No need to touch R[t*] further; t* won't be used again.
        }

        self.finalize_progress(&progress, enabled_count.saturating_sub(target));

        // 5) Build the final compact model from enabled tokens only (keep order stable)
        let final_pieces: Vec<(String, f64)> = m
            .vocab
            .iter()
            .enumerate()
            .filter(|(id, _)| !disabled.contains(id))
            .map(|(_, p)| (p.0.clone(), -1.0)) // keep unit-cost via constant -1.0 scores
            .collect();

        let final_model = Unigram::from(final_pieces, /*unk_id*/ None, /*byte_fallback*/ false)?;
        *model = final_model;

        Ok(self.special_tokens.clone())
    }
}

// ----------------------------- Trainer impl -----------------------------------

impl Trainer for CompressionTrainer {
    type Model = Unigram;

    /// Public training entry point; delegates to `do_train`.
    fn train(&self, model: &mut Unigram) -> Result<Vec<AddedToken>> {
        let sentences: Vec<_> = self.words.iter().map(|(s, i)| (s.to_owned(), *i)).collect();
        self.do_train(sentences, model)
    }

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    /// Collect the word/span counts from an iterator, mirroring `UnigramTrainer::feed`.
    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        // For each input sequence, `process` returns a vector of "words"/spans.
        // We count their frequencies and merge across the dataset, in parallel when available.
        let words: Result<AHashMap<String, u32>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = AHashMap::new();
                for word in words {
                    *map.entry(word).or_default() += 1;
                }
                Ok(map)
            })
            .reduce(
                || Ok(AHashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        *acc.entry(k).or_default() += v;
                    }
                    Ok(acc)
                },
            );

        self.words = words?;
        Ok(())
    }
}