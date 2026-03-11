#![allow(clippy::map_entry)]

use super::{Pair, WithFirstLastIterator, Word, BPE};
use crate::parallelism::*;
use crate::tokenizer::{AddedToken, Result, Trainer};
use crate::utils::progress::{ProgressBar, ProgressStyle};
use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use dary_heap::OctonaryHeap;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashSet, VecDeque};

#[derive(Debug, Eq)]
struct PairMerge {
    pair: Pair,
    count: u64,
}
impl PartialEq for PairMerge {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}
impl PartialOrd for PairMerge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PairMerge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            other.pair.cmp(&self.pair)
        }
    }
}

/// Parity selection variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParityVariant {
    /// At each step, pick the language with the longest total dev-set token length.
    Base,
    /// Use a moving-window mechanism to prevent one language from monopolizing merges.
    Window,
}

/// Configuration for the parity-aware BPE trainer.
struct ParityConfig {
    min_frequency: u64,
    vocab_size: usize,
    num_merges: Option<usize>,
    show_progress: bool,
    special_tokens: Vec<AddedToken>,
    limit_alphabet: Option<usize>,
    initial_alphabet: AHashSet<char>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    max_token_length: Option<usize>,
    /// How many initial merges use global (concatenated) statistics.
    global_merges: usize,
    /// Parity variant (base or window).
    variant: ParityVariant,
    /// Window size for the moving-window variant.
    window_size: usize,
    /// Alpha parameter for the moving-window variant.
    alpha: f64,
    /// Desired compression ratios per language (alternative to dev set).
    ratio: Option<Vec<f64>>,
    /// If true, subtract unique char count from num_symbols.
    total_symbols: bool,
}

pub struct ParityBpeTrainerBuilder {
    config: ParityConfig,
}

impl Default for ParityBpeTrainerBuilder {
    fn default() -> Self {
        Self {
            config: ParityConfig {
                min_frequency: 0,
                vocab_size: 30000,
                num_merges: None,
                show_progress: true,
                special_tokens: vec![],
                limit_alphabet: None,
                initial_alphabet: AHashSet::new(),
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                max_token_length: None,
                global_merges: 0,
                variant: ParityVariant::Base,
                window_size: 100,
                alpha: 2.0,
                ratio: None,
                total_symbols: false,
            },
        }
    }
}

impl ParityBpeTrainerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.config.min_frequency = frequency;
        self
    }

    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    #[must_use]
    pub fn num_merges(mut self, n: usize) -> Self {
        self.config.num_merges = Some(n);
        self
    }

    #[must_use]
    pub fn show_progress(mut self, show: bool) -> Self {
        self.config.show_progress = show;
        self
    }

    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.config.special_tokens = tokens;
        self
    }

    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.config.limit_alphabet = Some(limit);
        self
    }

    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        let mut initial_alphabet = AHashSet::with_capacity(alphabet.len());
        initial_alphabet.extend(alphabet);
        self.config.initial_alphabet = initial_alphabet;
        self
    }

    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.config.continuing_subword_prefix = Some(prefix);
        self
    }

    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.config.end_of_word_suffix = Some(suffix);
        self
    }

    #[must_use]
    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        self.config.max_token_length = max_token_length;
        self
    }

    #[must_use]
    pub fn global_merges(mut self, n: usize) -> Self {
        self.config.global_merges = n;
        self
    }

    #[must_use]
    pub fn variant(mut self, variant: ParityVariant) -> Self {
        self.config.variant = variant;
        self
    }

    #[must_use]
    pub fn window_size(mut self, size: usize) -> Self {
        self.config.window_size = size;
        self
    }

    #[must_use]
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    #[must_use]
    pub fn ratio(mut self, ratio: Vec<f64>) -> Self {
        self.config.ratio = Some(ratio);
        self
    }

    #[must_use]
    pub fn total_symbols(mut self, total: bool) -> Self {
        self.config.total_symbols = total;
        self
    }

    pub fn build(self) -> ParityBpeTrainer {
        ParityBpeTrainer {
            min_frequency: self.config.min_frequency,
            vocab_size: self.config.vocab_size,
            num_merges: self.config.num_merges,
            show_progress: self.config.show_progress,
            special_tokens: self.config.special_tokens,
            limit_alphabet: self.config.limit_alphabet,
            initial_alphabet: self.config.initial_alphabet,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
            max_token_length: self.config.max_token_length,
            global_merges: self.config.global_merges,
            variant: self.config.variant,
            window_size: self.config.window_size,
            alpha: self.config.alpha,
            ratio: self.config.ratio,
            total_symbols: self.config.total_symbols,
            language_words: Vec::new(),
            dev_language_words: Vec::new(),
        }
    }
}

/// Parity-aware BPE trainer.
///
/// Unlike the standard BPE trainer which operates on a single corpus,
/// this trainer accepts multiple corpora (one per language) and selects
/// which language to optimize at each merge step, ensuring cross-lingual
/// fairness in tokenization.
///
/// Language selection is driven by dev set token lengths (matching the
/// Python reference implementation exactly). The language with the longest
/// total dev-set token length is selected for the next merge.
///
/// Key optimizations over the Python implementation:
/// - **Linked-list Word representation** for O(1) merge operations
///   (vs regex-based string join/split in Python)
/// - **Integer token IDs (u32)** throughout instead of string comparisons
/// - **Rayon parallelism** for initial pair counting
/// - **Efficient hash maps** (AHashMap) for pair counts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityBpeTrainer {
    pub min_frequency: u64,
    pub vocab_size: usize,
    pub num_merges: Option<usize>,
    pub show_progress: bool,
    pub special_tokens: Vec<AddedToken>,
    pub limit_alphabet: Option<usize>,
    pub initial_alphabet: AHashSet<char>,
    pub continuing_subword_prefix: Option<String>,
    pub end_of_word_suffix: Option<String>,
    pub max_token_length: Option<usize>,
    pub global_merges: usize,
    pub variant: ParityVariant,
    pub window_size: usize,
    pub alpha: f64,
    pub ratio: Option<Vec<f64>>,
    pub total_symbols: bool,

    /// Per-language training word counts: language_words[lang_idx] = {word -> count}
    #[serde(skip)]
    language_words: Vec<AHashMap<CompactString, u64>>,

    /// Per-language dev word counts: dev_language_words[lang_idx] = {word -> count}
    #[serde(skip)]
    dev_language_words: Vec<AHashMap<CompactString, u64>>,
}

impl Default for ParityBpeTrainer {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl ParityBpeTrainer {
    pub fn builder() -> ParityBpeTrainerBuilder {
        ParityBpeTrainerBuilder::new()
    }

    /// Feed word counts for a specific language index (training data).
    pub fn feed_language(&mut self, lang_idx: usize, words: AHashMap<CompactString, u64>) {
        if self.language_words.len() <= lang_idx {
            self.language_words.resize_with(lang_idx + 1, AHashMap::new);
        }
        self.language_words[lang_idx] = words;
    }

    /// Feed word counts for a specific language index (dev data).
    pub fn feed_dev_language(&mut self, lang_idx: usize, words: AHashMap<CompactString, u64>) {
        if self.dev_language_words.len() <= lang_idx {
            self.dev_language_words
                .resize_with(lang_idx + 1, AHashMap::new);
        }
        self.dev_language_words[lang_idx] = words;
    }

    /// Return number of languages currently fed.
    pub fn num_languages(&self) -> usize {
        self.language_words.len()
    }

    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<40!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            Some(p)
        } else {
            None
        }
    }

    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
            println!();
        }
    }

    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &'static str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
        }
    }

    fn add_special_tokens(
        &self,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
    ) {
        for token in &self.special_tokens {
            if !w2id.contains_key(&CompactString::from(&token.content)) {
                id2w.push(CompactString::from(&token.content));
                w2id.insert(CompactString::from(&token.content), (id2w.len() - 1) as u32);
            }
        }
    }

    fn compute_alphabet(
        &self,
        all_words: &[&AHashMap<CompactString, u64>],
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
    ) {
        let mut alphabet: AHashMap<char, usize> = AHashMap::new();
        for wc in all_words {
            for (word, count) in *wc {
                for c in word.chars() {
                    *alphabet.entry(c).or_default() += *count as usize;
                }
            }
        }

        for c in &self.initial_alphabet {
            *alphabet.entry(*c).or_default() = usize::MAX;
        }

        let mut kept = alphabet.iter().collect::<Vec<_>>();

        let to_remove = self
            .limit_alphabet
            .map(|limit| alphabet.len().saturating_sub(limit))
            .unwrap_or(0);

        if to_remove > 0 {
            kept.sort_unstable_by_key(|k| *k.1);
            kept.drain(..to_remove);
        }

        kept.sort_unstable_by_key(|k| *k.0 as u32);
        kept.into_iter().for_each(|(c, _)| {
            let s = c.to_string();
            if !w2id.contains_key(&CompactString::from(&s)) {
                id2w.push(CompactString::from(&s));
                w2id.insert(CompactString::from(&s), (id2w.len() - 1) as u32);
            }
        });
    }

    fn tokenize_words(
        &self,
        wc: &AHashMap<CompactString, u64>,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
    ) -> (Vec<Word>, Vec<u64>) {
        let mut words: Vec<Word> = Vec::with_capacity(wc.len());
        let mut counts: Vec<u64> = Vec::with_capacity(wc.len());

        for (word, count) in wc {
            let mut current_word = Word::new();
            counts.push(*count);

            for (is_first, is_last, c) in word.chars().with_first_and_last() {
                let mut s = c.to_string();
                if w2id.contains_key(&CompactString::from(&s)) {
                    if !is_first {
                        if let Some(prefix) = &self.continuing_subword_prefix {
                            s.insert_str(0, prefix);
                        }
                    }
                    if is_last {
                        if let Some(suffix) = &self.end_of_word_suffix {
                            s.push_str(suffix);
                        }
                    }

                    if !w2id.contains_key(&CompactString::from(&s)) {
                        id2w.push(CompactString::from(&s));
                        w2id.insert(CompactString::from(&s), (id2w.len() - 1) as u32);
                    }
                    current_word.add(w2id[&CompactString::from(&s)], 1);
                }
            }
            words.push(current_word);
        }

        (words, counts)
    }

    /// Count pairs for a single language, returning per-pair counts and positions.
    fn count_pairs(
        &self,
        words: &[Word],
        counts: &[u64],
    ) -> (AHashMap<Pair, i64>, AHashMap<Pair, AHashSet<usize>>) {
        words
            .maybe_par_iter()
            .enumerate()
            .map(|(i, word)| {
                let mut pair_counts: AHashMap<Pair, i64> = AHashMap::new();
                let mut where_to_update: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();

                for window in word.get_chars().windows(2) {
                    let cur_pair: Pair = (window[0], window[1]);
                    *pair_counts.entry(cur_pair).or_default() += counts[i] as i64;
                    where_to_update.entry(cur_pair).or_default().insert(i);
                }

                (pair_counts, where_to_update)
            })
            .reduce(
                || (AHashMap::new(), AHashMap::new()),
                |(mut pair_counts, mut where_to_update), (pc, wtu)| {
                    for (k, v) in pc {
                        *pair_counts.entry(k).or_default() += v;
                    }
                    for (k, v) in wtu {
                        where_to_update.entry(k).or_default().extend(v);
                    }
                    (pair_counts, where_to_update)
                },
            )
    }

    /// Select which language to optimize next using the moving-window approach.
    fn select_language_window(
        &self,
        lengths: &[i64],
        selected_indices: &VecDeque<usize>,
        selection_threshold: f64,
    ) -> usize {
        let num_langs = lengths.len();
        let mut mask = vec![true; num_langs];

        loop {
            let mut best_idx = 0;
            let mut best_val = i64::MIN;
            for i in 0..num_langs {
                if mask[i] && lengths[i] > best_val {
                    best_val = lengths[i];
                    best_idx = i;
                }
            }

            let count = selected_indices.iter().filter(|&&x| x == best_idx).count();
            let ratio = count as f64 / self.window_size as f64;
            if ratio <= selection_threshold {
                return best_idx;
            }

            mask[best_idx] = false;

            // If all masked out, fall back to the overall best
            if mask.iter().all(|&m| !m) {
                return best_idx;
            }
        }
    }

    /// Select which language to optimize next using the moving-window approach (f64 version for ratio mode).
    fn select_language_window_f64(
        &self,
        values: &[f64],
        selected_indices: &VecDeque<usize>,
        selection_threshold: f64,
    ) -> usize {
        let num_langs = values.len();
        let mut mask = vec![true; num_langs];

        loop {
            let mut best_idx = 0;
            let mut best_val = f64::NEG_INFINITY;
            for i in 0..num_langs {
                if mask[i] && values[i] > best_val {
                    best_val = values[i];
                    best_idx = i;
                }
            }

            let count = selected_indices.iter().filter(|&&x| x == best_idx).count();
            let ratio = count as f64 / self.window_size as f64;
            if ratio <= selection_threshold {
                return best_idx;
            }

            mask[best_idx] = false;

            if mask.iter().all(|&m| !m) {
                return best_idx;
            }
        }
    }

    /// Pop the best pair from a priority queue, lazily discarding stale entries.
    fn pop_best_pair(
        queue: &mut OctonaryHeap<PairMerge>,
        pair_counts: &AHashMap<Pair, i64>,
    ) -> Option<(Pair, u64)> {
        loop {
            let top = queue.pop()?;
            let current_count = pair_counts.get(&top.pair).copied().unwrap_or(0);
            if current_count <= 0 {
                continue;
            }
            if top.count != current_count as u64 {
                queue.push(PairMerge {
                    pair: top.pair,
                    count: current_count as u64,
                });
                continue;
            }
            return Some((top.pair, top.count));
        }
    }

    /// Find the best pair for a language using linear scan with string-based
    /// tie-breaking to match Python's `max(stats, key=lambda x: (stats[x][lang], x))`.
    fn find_best_pair_linear(
        &self,
        pair_counts: &AHashMap<Pair, i64>,
        id_to_word: &[CompactString],
    ) -> Option<(Pair, u64)> {
        let mut best_pair: Option<Pair> = None;
        let mut best_count: i64 = 0;
        let mut best_key: (CompactString, CompactString) =
            (CompactString::default(), CompactString::default());

        for (&pair, &count) in pair_counts {
            if count <= 0 {
                continue;
            }
            let str_a = &id_to_word[pair.0 as usize];
            let str_b = &id_to_word[pair.1 as usize];

            if count > best_count
                || (count == best_count && (str_a, str_b) > (&best_key.0, &best_key.1))
            {
                best_count = count;
                best_pair = Some(pair);
                best_key = (str_a.clone(), str_b.clone());
            }
        }

        best_pair.map(|p| (p, best_count as u64))
    }

    /// Find the best pair in global mode (summing counts across all languages)
    /// with string-based tie-breaking.
    fn find_best_pair_global_linear(
        &self,
        per_lang_pair_counts: &[AHashMap<Pair, i64>],
        id_to_word: &[CompactString],
    ) -> Option<(Pair, u64)> {
        // Collect all pairs and sum their counts
        let mut total_counts: AHashMap<Pair, i64> = AHashMap::new();
        for lang_counts in per_lang_pair_counts {
            for (&pair, &count) in lang_counts {
                if count > 0 {
                    *total_counts.entry(pair).or_default() += count;
                }
            }
        }

        self.find_best_pair_linear(&total_counts, id_to_word)
    }

    /// Apply a merge to the dev vocabulary.
    /// Returns the per-language length change (positive = words got shorter).
    fn replace_pair_dev(
        pair: Pair,
        new_token_id: u32,
        dev_vocab: &mut AHashMap<Vec<u32>, Vec<i64>>,
        num_langs: usize,
    ) -> Vec<i64> {
        let mut length_change = vec![0i64; num_langs];

        // Find all words containing the pair
        let words_to_update: Vec<Vec<u32>> = dev_vocab
            .keys()
            .filter(|word| {
                word.windows(2)
                    .any(|w| w[0] == pair.0 && w[1] == pair.1)
            })
            .cloned()
            .collect();

        for old_word in words_to_update {
            let freq = dev_vocab.remove(&old_word).unwrap();

            // Merge the pair in the word
            let mut new_word = Vec::with_capacity(old_word.len());
            let mut i = 0;
            while i < old_word.len() {
                if i + 1 < old_word.len() && old_word[i] == pair.0 && old_word[i + 1] == pair.1 {
                    new_word.push(new_token_id);
                    i += 2;
                } else {
                    new_word.push(old_word[i]);
                    i += 1;
                }
            }

            let old_len = old_word.len() as i64;
            let new_len = new_word.len() as i64;
            for lang in 0..num_langs {
                length_change[lang] += (old_len - new_len) * freq[lang];
            }

            dev_vocab.insert(new_word, freq);
        }

        length_change
    }

    /// Main training method. Returns (special_tokens, ordered_merge_strings).
    /// Each merge string is "token_a token_b" matching the Python output format.
    pub fn do_train(&self, model: &mut BPE) -> Result<(Vec<AddedToken>, Vec<String>)> {
        let num_langs = self.language_words.len();
        assert!(num_langs > 0, "No language data has been fed");

        let max_token_length: usize = self.max_token_length.unwrap_or(usize::MAX);
        let progress = self.setup_progress();

        let mut word_to_id: AHashMap<CompactString, u32> =
            AHashMap::with_capacity(self.vocab_size);
        let mut id_to_word: Vec<CompactString> = Vec::with_capacity(self.vocab_size);

        // 1. Add special tokens
        self.add_special_tokens(&mut word_to_id, &mut id_to_word);

        // 2. Compute alphabet from ALL languages (train + dev)
        let mut all_words_refs: Vec<&AHashMap<CompactString, u64>> =
            self.language_words.iter().collect();
        for dw in &self.dev_language_words {
            all_words_refs.push(dw);
        }
        self.compute_alphabet(&all_words_refs, &mut word_to_id, &mut id_to_word);

        // 3. Tokenize training words per language
        self.update_progress(&progress, 0, "Tokenize words");
        let mut per_lang_words: Vec<Vec<Word>> = Vec::with_capacity(num_langs);
        let mut per_lang_counts: Vec<Vec<u64>> = Vec::with_capacity(num_langs);

        for lang_wc in &self.language_words {
            let (words, counts) =
                self.tokenize_words(lang_wc, &mut word_to_id, &mut id_to_word);
            per_lang_words.push(words);
            per_lang_counts.push(counts);
        }

        // 4. Count pairs per language
        self.update_progress(&progress, 0, "Count pairs");
        let mut per_lang_pair_counts: Vec<AHashMap<Pair, i64>> = Vec::with_capacity(num_langs);
        let mut per_lang_where: Vec<AHashMap<Pair, AHashSet<usize>>> =
            Vec::with_capacity(num_langs);

        for lang in 0..num_langs {
            let (pc, wtu) = self.count_pairs(&per_lang_words[lang], &per_lang_counts[lang]);
            per_lang_pair_counts.push(pc);
            per_lang_where.push(wtu);
        }

        // 4b. Build per-language priority queues
        let mut per_lang_queues: Vec<OctonaryHeap<PairMerge>> =
            Vec::with_capacity(num_langs);
        for lang in 0..num_langs {
            let mut queue =
                OctonaryHeap::with_capacity(per_lang_pair_counts[lang].len());
            for (&pair, &count) in &per_lang_pair_counts[lang] {
                if count > 0 {
                    queue.push(PairMerge {
                        pair,
                        count: count as u64,
                    });
                }
            }
            per_lang_queues.push(queue);
        }

        // 5. Build dev vocab and compute initial lengths
        let has_dev = !self.dev_language_words.is_empty();
        let has_ratio = self.ratio.is_some();
        let parity_num_langs = if has_dev {
            self.dev_language_words.len()
        } else {
            num_langs
        };

        let mut dev_vocab: AHashMap<Vec<u32>, Vec<i64>> = AHashMap::new();
        let mut lengths: Vec<i64> = vec![0i64; parity_num_langs];

        // For ratio mode: track initial and current lengths as f64
        let mut initial_lengths_f64: Vec<f64> = Vec::new();
        let mut lengths_f64: Vec<f64> = Vec::new();

        if has_ratio {
            // Ratio mode: compute initial lengths from training data
            for lang in 0..num_langs {
                let total: f64 = per_lang_words[lang]
                    .iter()
                    .zip(per_lang_counts[lang].iter())
                    .map(|(word, &count)| word.get_chars().len() as f64 * count as f64)
                    .sum();
                initial_lengths_f64.push(total);
                lengths_f64.push(total);
            }
            eprintln!(
                "Ratio mode: initial lengths: {:?}, ratios: {:?}",
                initial_lengths_f64,
                self.ratio.as_ref().unwrap()
            );
        } else if has_dev {
            // Tokenize dev words into char ID sequences
            for (lang_idx, dev_wc) in self.dev_language_words.iter().enumerate() {
                for (word_str, &count) in dev_wc {
                    let mut char_ids = Vec::new();
                    let mut valid = true;
                    for c in word_str.chars() {
                        let s = CompactString::from(c.to_string());
                        if let Some(&id) = word_to_id.get(&s) {
                            char_ids.push(id);
                        } else {
                            valid = false;
                            break;
                        }
                    }
                    if valid && !char_ids.is_empty() {
                        let entry = dev_vocab
                            .entry(char_ids)
                            .or_insert_with(|| vec![0i64; parity_num_langs]);
                        entry[lang_idx] += count as i64;
                    }
                }
            }

            // Compute initial lengths from dev vocab: sum(word_len * freq) per language
            for (word, freqs) in &dev_vocab {
                for lang in 0..parity_num_langs {
                    lengths[lang] += word.len() as i64 * freqs[lang];
                }
            }
            eprintln!(
                "Dev vocab: {} unique words, initial lengths: {:?}",
                dev_vocab.len(),
                lengths
            );
        } else {
            // Fall back to training data lengths
            for lang in 0..num_langs {
                let total: i64 = per_lang_words[lang]
                    .iter()
                    .zip(per_lang_counts[lang].iter())
                    .map(|(word, &count)| word.get_chars().len() as i64 * count as i64)
                    .sum();
                lengths[lang] = total;
            }
        }

        // Moving-window state
        let selection_threshold = self.alpha / parity_num_langs as f64;
        let mut selected_indices: VecDeque<usize> =
            VecDeque::with_capacity(self.window_size);

        // 6. Handle --total-symbols: subtract unique char count from num_merges
        let mut num_merges = self
            .num_merges
            .unwrap_or_else(|| self.vocab_size.saturating_sub(word_to_id.len()));

        if self.total_symbols {
            let mut internal_chars: AHashSet<char> = AHashSet::new();
            let mut final_chars: AHashSet<char> = AHashSet::new();
            for lang_wc in &self.language_words {
                for (word, _) in lang_wc {
                    let chars: Vec<char> = word.chars().collect();
                    if !chars.is_empty() {
                        for &c in &chars[..chars.len() - 1] {
                            internal_chars.insert(c);
                        }
                        final_chars.insert(*chars.last().unwrap());
                    }
                }
            }
            let reduction = internal_chars.len() + final_chars.len();
            eprintln!(
                "Number of word-internal characters: {}",
                internal_chars.len()
            );
            eprintln!("Number of word-final characters: {}", final_chars.len());
            eprintln!(
                "Reducing number of merge operations by {}",
                reduction
            );
            num_merges = num_merges.saturating_sub(reduction);
        }

        self.update_progress(&progress, num_merges, "Compute merges");
        let mut merges: Vec<(Pair, u32)> = vec![];
        let mut exhausted: AHashSet<usize> = AHashSet::new();
        let mut merge_count = 0;

        while merge_count < num_merges {
            // Check if all languages are exhausted
            if exhausted.len() >= num_langs {
                eprintln!(
                    "All {} languages exhausted, stopping at {} merges",
                    num_langs, merge_count
                );
                break;
            }

            // Select which language to optimize
            let lang_idx = if merge_count < self.global_merges {
                usize::MAX // signals "use global"
            } else if has_ratio {
                let ratio_vec = self.ratio.as_ref().unwrap();
                // compression_rates = initial_lengths / lengths
                // adjusted = compression_rates / ratio
                let adjusted: Vec<f64> = initial_lengths_f64
                    .iter()
                    .zip(lengths_f64.iter())
                    .zip(ratio_vec.iter())
                    .map(|((&init, &cur), &r)| (init / cur) / r)
                    .collect();

                match self.variant {
                    ParityVariant::Base => {
                        // min(enumerate(adjusted)) — pick language with least adjusted compression
                        // Skip exhausted languages
                        let mut best_idx = 0;
                        let mut best_val = f64::INFINITY;
                        for (idx, &val) in adjusted.iter().enumerate() {
                            if !exhausted.contains(&idx) && val < best_val {
                                best_val = val;
                                best_idx = idx;
                            }
                        }
                        best_idx
                    }
                    ParityVariant::Window => {
                        // Python: select_language_index(-adjusted_compression_rates, ...)
                        let mut neg_adjusted: Vec<f64> =
                            adjusted.iter().map(|&v| -v).collect();
                        for &ex in &exhausted {
                            neg_adjusted[ex] = f64::NEG_INFINITY;
                        }
                        let idx = self.select_language_window_f64(
                            &neg_adjusted,
                            &selected_indices,
                            selection_threshold,
                        );
                        selected_indices.push_back(idx);
                        if selected_indices.len() > self.window_size {
                            selected_indices.pop_front();
                        }
                        idx
                    }
                }
            } else {
                match self.variant {
                    ParityVariant::Base => {
                        // Pick language with longest total token length.
                        // Skip exhausted languages.
                        let mut best_idx = 0;
                        let mut best_val = i64::MIN;
                        for (idx, &val) in lengths.iter().enumerate() {
                            if !exhausted.contains(&idx) && val > best_val {
                                best_val = val;
                                best_idx = idx;
                            }
                        }
                        best_idx
                    }
                    ParityVariant::Window => {
                        let mut effective_lengths = lengths.clone();
                        for &ex in &exhausted {
                            effective_lengths[ex] = i64::MIN;
                        }
                        let idx = self.select_language_window(
                            &effective_lengths,
                            &selected_indices,
                            selection_threshold,
                        );
                        selected_indices.push_back(idx);
                        if selected_indices.len() > self.window_size {
                            selected_indices.pop_front();
                        }
                        idx
                    }
                }
            };

            // Find the best pair
            let best_pair = if lang_idx == usize::MAX {
                self.find_best_pair_global_linear(&per_lang_pair_counts, &id_to_word)
            } else {
                Self::pop_best_pair(
                    &mut per_lang_queues[lang_idx],
                    &per_lang_pair_counts[lang_idx],
                )
            };

            let (best_pair, _best_count) = match best_pair {
                Some((p, c)) if c >= self.min_frequency => (p, c),
                _ => {
                    if lang_idx == usize::MAX {
                        // Global mode exhausted — no valid pairs across any language
                        break;
                    }
                    eprintln!(
                        "Language {} exhausted at merge {}, skipping",
                        lang_idx, merge_count
                    );
                    exhausted.insert(lang_idx);
                    continue;
                }
            };

            // Build new token
            let part_a = &id_to_word[best_pair.0 as usize];
            let mut part_b = id_to_word[best_pair.1 as usize].as_str();
            if let Some(prefix) = &self.continuing_subword_prefix {
                if let Some(rest) = part_b.strip_prefix(prefix) {
                    part_b = rest;
                }
            }
            let new_token = format!("{part_a}{part_b}");
            let new_token_id = word_to_id
                .get(&CompactString::from(&new_token))
                .copied()
                .unwrap_or(id_to_word.len() as u32);
            if !word_to_id.contains_key(&CompactString::from(&new_token)) {
                id_to_word.push(CompactString::from(&new_token));
                word_to_id.insert(CompactString::from(&new_token), new_token_id);
            }
            merges.push((best_pair, new_token_id));

            // Apply merge to ALL languages' training words and update pair counts
            for lang in 0..num_langs {
                let (train_length_change, changed_pairs) = self.apply_merge_to_language(
                    best_pair,
                    new_token_id,
                    max_token_length,
                    &mut per_lang_words[lang],
                    &per_lang_counts[lang],
                    &mut per_lang_pair_counts[lang],
                    &mut per_lang_where[lang],
                );

                // Push changed pairs into this language's heap
                for changed_pair in changed_pairs {
                    let count = per_lang_pair_counts[lang]
                        .get(&changed_pair)
                        .copied()
                        .unwrap_or(0);
                    if count > 0 {
                        per_lang_queues[lang].push(PairMerge {
                            pair: changed_pair,
                            count: count as u64,
                        });
                    }
                }

                if has_ratio {
                    // Ratio mode: update lengths from training data changes
                    lengths_f64[lang] -= train_length_change as f64;
                } else if !has_dev {
                    // No dev data and no ratio: update lengths from training data
                    lengths[lang] -= train_length_change;
                }
            }

            // Apply merge to dev vocab and update lengths (only when using dev set, not ratio)
            if has_dev && !has_ratio {
                let length_change = Self::replace_pair_dev(
                    best_pair,
                    new_token_id,
                    &mut dev_vocab,
                    parity_num_langs,
                );
                for lang in 0..parity_num_langs {
                    lengths[lang] -= length_change[lang];
                }
            }

            merge_count += 1;
            if let Some(p) = &progress {
                p.inc(1);
            }
        }

        self.finalize_progress(&progress, merges.len());
        eprintln!(
            "Training complete: {} merges, {} vocab size",
            merges.len(),
            id_to_word.len()
        );

        // Build ordered merge strings for output
        let merge_strings: Vec<String> = merges
            .iter()
            .map(|(pair, _)| {
                let a = &id_to_word[pair.0 as usize];
                let b = &id_to_word[pair.1 as usize];
                format!("{} {}", a, b)
            })
            .collect();

        // Transfer to model
        model.vocab = word_to_id
            .into_iter()
            .map(|(_key, val)| (id_to_word[val as usize].to_string(), val))
            .collect();
        model.vocab_r = model
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();
        model.merges = merges
            .into_iter()
            .enumerate()
            .map(|(i, (pair, new_token_id))| (pair, (i as u32, new_token_id)))
            .collect();

        model.continuing_subword_prefix = self.continuing_subword_prefix.clone();
        model.end_of_word_suffix = self.end_of_word_suffix.clone();

        Ok((self.special_tokens.clone(), merge_strings))
    }

    /// Apply a merge to one language's words, update pair counts.
    /// Returns (length_reduction, changed_pairs) for heap updates.
    fn apply_merge_to_language(
        &self,
        pair: Pair,
        new_token_id: u32,
        max_token_length: usize,
        words: &mut [Word],
        counts: &[u64],
        pair_counts: &mut AHashMap<Pair, i64>,
        where_to_update: &mut AHashMap<Pair, AHashSet<usize>>,
    ) -> (i64, Vec<Pair>) {
        let positions = match where_to_update.remove(&pair) {
            Some(pos) => pos,
            None => return (0, Vec::new()),
        };

        // --- Parallel phase: merge words at each position ---
        // Safety: same pattern as standard BPE (trainer.rs:521-544).
        // Each position appears at most once (AHashSet), so no two threads
        // mutate the same Word.
        let words_len = words.len();
        struct WordPtr(*mut Word);
        unsafe impl Sync for WordPtr {}
        let word_start = WordPtr(words.as_mut_ptr());

        let changes: Vec<(Vec<(Pair, i32)>, usize, i64)> = positions
            .maybe_par_iter()
            .map(|&i| unsafe {
                assert!(i < words_len);
                let word = word_start.0.add(i);
                let old_len = (*word).get_chars().len() as i64;
                let merge_changes =
                    (*word).merge(pair.0, pair.1, new_token_id, max_token_length);
                let new_len = (*word).get_chars().len() as i64;
                let reduction = (old_len - new_len) * counts[i] as i64;
                (merge_changes, i, reduction)
            })
            .collect();

        // --- Sequential phase: apply changes to pair_counts + where_to_update ---
        let mut length_reduction: i64 = 0;
        let mut changed_pairs = Vec::new();
        for (merge_changes, iw, reduction) in changes {
            length_reduction += reduction;
            for (change_pair, change) in merge_changes {
                let count = change as i64 * counts[iw] as i64;
                *pair_counts.entry(change_pair).or_default() += count;
                if change > 0 {
                    where_to_update
                        .entry(change_pair)
                        .or_default()
                        .insert(iw);
                }
                changed_pairs.push(change_pair);
            }
        }

        pair_counts.insert(pair, 0);
        (length_reduction, changed_pairs)
    }
}

impl Trainer for ParityBpeTrainer {
    type Model = BPE;

    fn train(&self, model: &mut BPE) -> Result<Vec<AddedToken>> {
        let (special_tokens, _merge_strings) = self.do_train(model)?;
        Ok(special_tokens)
    }

    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        // Default feed adds to language 0.
        // For multi-language use, call feed_language directly.
        let words: Result<AHashMap<CompactString, u64>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = AHashMap::new();
                for word in words {
                    *map.entry(CompactString::from(word)).or_default() += 1;
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

        self.feed_language(0, words?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parity_base_exact_merges() {
        // Symmetric two-language data, no dev set.
        // Lang 0: "aabb" x10, Lang 1: "ccdd" x10
        let lang0: AHashMap<CompactString, u64> = [("aabb".into(), 10u64)]
            .iter()
            .cloned()
            .collect();
        let lang1: AHashMap<CompactString, u64> = [("ccdd".into(), 10u64)]
            .iter()
            .cloned()
            .collect();

        let mut trainer = ParityBpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .num_merges(6)
            .variant(ParityVariant::Base)
            .build();

        trainer.feed_language(0, lang0);
        trainer.feed_language(1, lang1);

        let mut model = BPE::default();
        let (_special, merge_strings) = trainer.do_train(&mut model).unwrap();

        // Languages are tied in length, so lang 0 goes first (lower index wins ties).
        // ID-based tie-breaking: lower pair ID wins.
        // Chars: a(0), b(1), c(2), d(3). Pairs with same count → lowest ID wins.
        // Lang 0 first: a a, then lang 1: c c, then lang 0: b b, etc.
        assert_eq!(
            merge_strings,
            vec!["a a", "c c", "b b", "d d", "aa bb", "cc dd"],
            "expected alternating merges; got {:?}",
            merge_strings
        );
        assert!(
            model.vocab.contains_key("aabb"),
            "final token 'aabb' should be in vocab"
        );
        assert!(
            model.vocab.contains_key("ccdd"),
            "final token 'ccdd' should be in vocab"
        );
    }

    #[test]
    fn test_parity_base_dev_drives_selection() {
        // Asymmetric training data with inverted dev data.
        // Train lang 0 is larger, but dev lang 1 is larger — dev should win.
        let train0: AHashMap<CompactString, u64> = [("ab".into(), 10u64)]
            .iter()
            .cloned()
            .collect();
        let train1: AHashMap<CompactString, u64> = [("cd".into(), 5u64)]
            .iter()
            .cloned()
            .collect();

        // Dev inverts the priority: lang 1 has more data
        let dev0: AHashMap<CompactString, u64> = [("ab".into(), 1u64)]
            .iter()
            .cloned()
            .collect();
        let dev1: AHashMap<CompactString, u64> = [("cd".into(), 10u64)]
            .iter()
            .cloned()
            .collect();

        let mut trainer = ParityBpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .num_merges(2)
            .variant(ParityVariant::Base)
            .build();

        trainer.feed_language(0, train0);
        trainer.feed_language(1, train1);
        trainer.feed_dev_language(0, dev0);
        trainer.feed_dev_language(1, dev1);

        let mut model = BPE::default();
        let (_special, merge_strings) = trainer.do_train(&mut model).unwrap();

        // Dev lengths: lang 0 = 2 chars, lang 1 = 20 chars
        // Lang 1 selected first despite smaller training data
        // 'c' + 'd' -> 'cd' (lang 1 first)
        // 'a' + 'b' -> 'ab' (lang 0 second)
        assert_eq!(
            merge_strings,
            vec!["c d", "a b"],
            "dev set should drive language selection: lang 1 first"
        );
    }

    #[test]
    fn test_parity_window_ensures_fairness() {
        // Highly asymmetric data: lang 0 dominates in length.
        // Base would give lang 0 all its merges first.
        // Window (alpha=1.0, window_size=2) forces lang 1 to get turns earlier.
        //
        // threshold = alpha / num_langs = 1.0 / 2 = 0.5
        // After lang 0 fills >50% of the window, it gets masked.
        let lang0: AHashMap<CompactString, u64> = [("aabb".into(), 100u64)]
            .iter()
            .cloned()
            .collect();
        let lang1: AHashMap<CompactString, u64> = [("ccdd".into(), 1u64)]
            .iter()
            .cloned()
            .collect();

        // Base variant: lang 0 monopolizes until exhausted
        let mut trainer = ParityBpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .num_merges(6)
            .variant(ParityVariant::Base)
            .build();
        trainer.feed_language(0, lang0.clone());
        trainer.feed_language(1, lang1.clone());
        let mut model = BPE::default();
        let (_special, base_merges) = trainer.do_train(&mut model).unwrap();
        // Base: lang 0 always longest, takes all 3 merges before lang 1 gets any.
        // ID-based tie-breaking: a a (lowest pair ID) before b b.
        assert_eq!(
            base_merges,
            vec!["a a", "b b", "aa bb", "c c", "d d", "cc dd"],
            "Base should let lang 0 monopolize merges"
        );

        // Window variant: forces interleaving
        let mut trainer = ParityBpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .num_merges(6)
            .variant(ParityVariant::Window)
            .window_size(2)
            .alpha(1.0)
            .build();
        trainer.feed_language(0, lang0);
        trainer.feed_language(1, lang1);
        let mut model = BPE::default();
        let (_special, window_merges) = trainer.do_train(&mut model).unwrap();
        // Window: after 2 consecutive lang 0 picks, ratio=2/2=1.0 > 0.5 threshold,
        // so lang 0 is masked and lang 1 gets "c c" at step 3 instead of step 4.
        assert_eq!(
            window_merges,
            vec!["a a", "b b", "c c", "aa bb", "d d", "cc dd"],
            "Window should force lang 1's first merge earlier than Base"
        );
        // Key difference: lang 1's first merge is at index 2 (Window) vs 3 (Base)
        assert_ne!(
            base_merges, window_merges,
            "Window and Base should produce different merge orders"
        );
    }

    #[test]
    fn test_parity_exhausted_language_continues() {
        // Lang 0 has only 1 possible pair ("a"+"b"), lang 1 has 3 ("e"+"f", "d"+"ef", "c"+"def").
        // No dev set — training data lengths drive language selection.
        // Training should skip exhausted lang 0 and continue with lang 1.
        let lang0: AHashMap<CompactString, u64> = [("ab".into(), 10u64)]
            .iter()
            .cloned()
            .collect();
        let lang1: AHashMap<CompactString, u64> = [("cdef".into(), 10u64)]
            .iter()
            .cloned()
            .collect();

        let mut trainer = ParityBpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .num_merges(10) // request more merges than possible
            .variant(ParityVariant::Base)
            .build();

        trainer.feed_language(0, lang0);
        trainer.feed_language(1, lang1);

        let mut model = BPE::default();
        let (_special, merge_strings) = trainer.do_train(&mut model).unwrap();

        // Lang 1 selected first (longer: 4*10=40 vs 2*10=20)
        // ID-based tie-breaking: (2,3)="c d" wins over (3,4) and (4,5)
        // 'c' + 'd' -> 'cd' (lang 1, length now 3*10=30)
        // Lang 1 still longer (30 vs 20), selected again:
        // 'e' + 'f' -> 'ef' (lang 1, length now 2*10=20)
        // Tied at 20, lang 0 wins by index:
        // 'a' + 'b' -> 'ab' (lang 0, now exhausted)
        // Lang 0 exhausted, skip to lang 1:
        // 'cd' + 'ef' -> 'cdef' (lang 1)
        // Only 4 merges possible despite requesting 10
        assert_eq!(
            merge_strings,
            vec!["c d", "e f", "a b", "cd ef"],
            "should produce exactly 4 merges; exhausted lang 0 skipped"
        );
    }

    #[test]
    fn test_parity_global_merges() {
        // Same data trained with global_merges=1 vs global_merges=0.
        // Global warmup uses concatenated statistics, changing merge order.
        let make_data = || {
            let lang0: AHashMap<CompactString, u64> = [
                ("ab".into(), 5u64),
                ("cd".into(), 1),
            ]
            .iter()
            .cloned()
            .collect();
            let lang1: AHashMap<CompactString, u64> = [
                ("ab".into(), 1u64),
                ("cd".into(), 5),
            ]
            .iter()
            .cloned()
            .collect();
            (lang0, lang1)
        };

        // With global_merges=1: first merge uses global stats (ab:6, cd:6 — tied,
        // 'c'+'d' wins alphabetically), then per-language for the rest.
        let (lang0, lang1) = make_data();
        let mut trainer = ParityBpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .num_merges(2)
            .global_merges(1)
            .variant(ParityVariant::Base)
            .build();
        trainer.feed_language(0, lang0);
        trainer.feed_language(1, lang1);
        let mut model = BPE::default();
        let (_special, merge_strings) = trainer.do_train(&mut model).unwrap();
        assert_eq!(
            merge_strings,
            vec!["c d", "a b"],
            "global merge should pick 'c d' first (alphabetic tie-break)"
        );

        // With global_merges=0: per-language from the start.
        // Lang 0 selected first (longer: 5*2+1*2=12 vs 1*2+5*2=12 — tied,
        // lang 0 wins by index), lang 0's best pair is "ab" (freq 5).
        let (lang0, lang1) = make_data();
        let mut trainer = ParityBpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .num_merges(2)
            .global_merges(0)
            .variant(ParityVariant::Base)
            .build();
        trainer.feed_language(0, lang0);
        trainer.feed_language(1, lang1);
        let mut model = BPE::default();
        let (_special, merge_strings) = trainer.do_train(&mut model).unwrap();
        assert_eq!(
            merge_strings,
            vec!["a b", "c d"],
            "without global merges, lang 0 picks 'a b' first"
        );
    }

    #[test]
    fn test_parity_min_frequency() {
        // Lang 0: "ab" x10 (pair freq 10), "cd" x3 (pair freq 3 — below threshold)
        // Lang 1: "ef" x10, "gh" x10
        // min_frequency=5 filters out "cd" pair
        let lang0: AHashMap<CompactString, u64> = [
            ("ab".into(), 10u64),
            ("cd".into(), 3),
        ]
        .iter()
        .cloned()
        .collect();
        let lang1: AHashMap<CompactString, u64> = [
            ("ef".into(), 10u64),
            ("gh".into(), 10),
        ]
        .iter()
        .cloned()
        .collect();

        let mut trainer = ParityBpeTrainer::builder()
            .show_progress(false)
            .min_frequency(5)
            .num_merges(10)
            .variant(ParityVariant::Base)
            .build();

        trainer.feed_language(0, lang0);
        trainer.feed_language(1, lang1);

        let mut model = BPE::default();
        let (_special, merge_strings) = trainer.do_train(&mut model).unwrap();

        // Lang 1 total length: 10*2 + 10*2 = 40, Lang 0: 10*2 + 3*2 = 26
        // Lang 1 first: 'g'+'h' -> 'gh' (freq 10, tied with 'ef'; 'g'>'e' — actually
        // the trainer picks highest freq first, both 10; tie-break by pair)
        // Lang 0: 'a'+'b' -> 'ab' (freq 10; 'cd' freq 3 < min_frequency=5, filtered)
        // Lang 1: 'e'+'f' -> 'ef' (freq 10)
        // Lang 0 exhausted (only valid pair was "ab"), all done
        assert_eq!(
            merge_strings.len(),
            3,
            "expected 3 merges; 'cd' pair (freq 3) should be filtered by min_frequency=5"
        );
        assert!(
            merge_strings.contains(&"a b".to_string()),
            "'a b' merge should be present"
        );
        assert!(
            merge_strings.contains(&"e f".to_string()),
            "'e f' merge should be present"
        );
        assert!(
            merge_strings.contains(&"g h".to_string()),
            "'g h' merge should be present"
        );
        assert!(
            !model.vocab.contains_key("cd"),
            "'cd' should NOT be in vocab (pair freq 3 < min_frequency 5)"
        );
    }

    #[test]
    fn test_ratio_base_favors_high_ratio_language() {
        // Lang 0: "aabb" x10, ratio=1.0; Lang 1: "ccdd" x10, ratio=2.0
        // Lang 1 needs more compression → gets lower adjusted value → selected first
        // adjusted = [(init/cur)/ratio_i]: initially [1.0, 0.5], lang 1 wins.
        // After 2 merges on lang 1 (c c, d d): adjusted ties at [1.0, 1.0] → lang 0.
        // Then lang 1 finishes with "cc dd".
        let lang0: AHashMap<CompactString, u64> = [("aabb".into(), 10u64)]
            .iter()
            .cloned()
            .collect();
        let lang1: AHashMap<CompactString, u64> = [("ccdd".into(), 10u64)]
            .iter()
            .cloned()
            .collect();

        let mut trainer = ParityBpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .num_merges(4)
            .variant(ParityVariant::Base)
            .ratio(vec![1.0, 2.0])
            .build();

        trainer.feed_language(0, lang0);
        trainer.feed_language(1, lang1);

        let mut model = BPE::default();
        let (_special, merge_strings) = trainer.do_train(&mut model).unwrap();

        assert_eq!(
            merge_strings,
            vec!["c c", "d d", "a a", "cc dd"],
            "high-ratio language should be selected first"
        );
    }

    #[test]
    fn test_ratio_equal_ratios_matches_no_ratio_symmetric() {
        // Symmetric data: both langs "aabb" x10, ratio=[1.0, 1.0].
        // With equal ratios and equal initial lengths, ratio mode should produce
        // the same merge order as no-ratio mode.
        let make_data = || {
            let lang0: AHashMap<CompactString, u64> =
                [("aabb".into(), 10u64)].iter().cloned().collect();
            let lang1: AHashMap<CompactString, u64> =
                [("ccdd".into(), 10u64)].iter().cloned().collect();
            (lang0, lang1)
        };

        // No-ratio mode
        let (lang0, lang1) = make_data();
        let mut trainer = ParityBpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .num_merges(6)
            .variant(ParityVariant::Base)
            .build();
        trainer.feed_language(0, lang0);
        trainer.feed_language(1, lang1);
        let mut model = BPE::default();
        let (_special, no_ratio_merges) = trainer.do_train(&mut model).unwrap();

        // Ratio mode with equal ratios
        let (lang0, lang1) = make_data();
        let mut trainer = ParityBpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .num_merges(6)
            .variant(ParityVariant::Base)
            .ratio(vec![1.0, 1.0])
            .build();
        trainer.feed_language(0, lang0);
        trainer.feed_language(1, lang1);
        let mut model = BPE::default();
        let (_special, ratio_merges) = trainer.do_train(&mut model).unwrap();

        assert_eq!(
            no_ratio_merges, ratio_merges,
            "equal ratios with symmetric data should match no-ratio mode"
        );
    }

    #[test]
    fn test_ratio_asymmetric_data_compensated_by_ratio() {
        // Lang 0: "ab" x100 (ratio=1.0), Lang 1: "cd" x10 (ratio=0.5)
        // Despite lang 1 having much less data, its low ratio means it's "already ahead".
        // adjusted[0] = (200/200)/1.0 = 1.0, adjusted[1] = (20/20)/0.5 = 2.0
        // → lang 0 selected first.
        let lang0: AHashMap<CompactString, u64> = [("ab".into(), 100u64)]
            .iter()
            .cloned()
            .collect();
        let lang1: AHashMap<CompactString, u64> = [("cd".into(), 10u64)]
            .iter()
            .cloned()
            .collect();

        let mut trainer = ParityBpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .num_merges(2)
            .variant(ParityVariant::Base)
            .ratio(vec![1.0, 0.5])
            .build();

        trainer.feed_language(0, lang0);
        trainer.feed_language(1, lang1);

        let mut model = BPE::default();
        let (_special, merge_strings) = trainer.do_train(&mut model).unwrap();

        assert_eq!(
            merge_strings,
            vec!["a b", "c d"],
            "lang 0 should go first despite smaller lang 1 data because ratio=0.5 marks lang 1 as ahead"
        );
    }

    #[test]
    fn test_ratio_window_variant() {
        // Lang 0: "aabb" x10 (ratio=1.0), Lang 1: "ccdd" x10 (ratio=3.0)
        // window_size=2, alpha=1.0. threshold = 1.0/2 = 0.5.
        // Lang 1 dominates initial selections (lower adjusted), but after 2 picks
        // its window ratio = 2/2 = 1.0 > 0.5 → masked, forcing lang 0 at merge 3.
        let lang0: AHashMap<CompactString, u64> = [("aabb".into(), 10u64)]
            .iter()
            .cloned()
            .collect();
        let lang1: AHashMap<CompactString, u64> = [("ccdd".into(), 10u64)]
            .iter()
            .cloned()
            .collect();

        let mut trainer = ParityBpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .num_merges(4)
            .variant(ParityVariant::Window)
            .window_size(2)
            .alpha(1.0)
            .ratio(vec![1.0, 3.0])
            .build();

        trainer.feed_language(0, lang0);
        trainer.feed_language(1, lang1);

        let mut model = BPE::default();
        let (_special, merge_strings) = trainer.do_train(&mut model).unwrap();

        // Merge 1-2: lang 1 (lower adjusted). Merge 3: window masks lang 1, forces lang 0.
        // Merge 4: lang 1 unmasked, finishes with "cc dd".
        assert_eq!(
            merge_strings,
            vec!["c c", "d d", "a a", "cc dd"],
            "window should force lang 0 at merge 3 despite lang 1 having lower adjusted value"
        );
        // Verify the window masking actually mattered: merge 3 is from lang 0
        assert_eq!(merge_strings[2], "a a", "merge 3 should be from lang 0 due to window masking");
    }
}
