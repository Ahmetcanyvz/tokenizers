use ahash::AHashMap;
use arrow::array::Array;
use compact_str::CompactString;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;
use tokenizers::models::bpe::{ParityBpeTrainer, ParityVariant, TrainingConfig, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};

fn pre_tokenize_text(
    text: &str,
    pre_tokenizer: &Sequence,
    word_counts: &mut AHashMap<CompactString, u64>,
) {
    for line in text.lines() {
        let mut pretokenized = PreTokenizedString::from(line);
        pre_tokenizer
            .pre_tokenize(&mut pretokenized)
            .expect("Pre-tokenization failed");
        let splits = pretokenized.get_splits(OffsetReferential::Original, OffsetType::Byte);
        for (word, _, _) in splits {
            if !word.is_empty() {
                *word_counts.entry(CompactString::from(word)).or_default() += 1;
            }
        }
    }
}

fn pre_tokenize_file(path: &str, pre_tokenizer: &Sequence) -> AHashMap<CompactString, u64> {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open {}: {}", path, e));
    let reader = BufReader::new(file);
    let mut word_counts: AHashMap<CompactString, u64> = AHashMap::new();

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        pre_tokenize_text(&line, pre_tokenizer, &mut word_counts);
    }

    word_counts
}

fn pre_tokenize_parquet_file(
    path: &str,
    text_column: &str,
    pre_tokenizer: &Sequence,
) -> AHashMap<CompactString, u64> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open {}: {}", path, e));
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(file).expect("Failed to create parquet reader");
    let reader = builder.build().expect("Failed to build parquet reader");
    let mut word_counts: AHashMap<CompactString, u64> = AHashMap::new();

    for batch in reader {
        let batch = batch.expect("Failed to read batch");
        let col = batch
            .column_by_name(text_column)
            .unwrap_or_else(|| panic!("Column '{}' not found in {}", text_column, path));

        if let Some(arr) = col
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
        {
            for i in 0..arr.len() {
                if arr.is_null(i) {
                    continue;
                }
                pre_tokenize_text(arr.value(i), pre_tokenizer, &mut word_counts);
            }
        } else if let Some(arr) = col
            .as_any()
            .downcast_ref::<arrow::array::LargeStringArray>()
        {
            for i in 0..arr.len() {
                if arr.is_null(i) {
                    continue;
                }
                pre_tokenize_text(arr.value(i), pre_tokenizer, &mut word_counts);
            }
        } else {
            panic!(
                "Column '{}' in {} is not a string type",
                text_column, path
            );
        }
    }
    word_counts
}

fn pre_tokenize_auto(
    path: &str,
    text_column: &str,
    pre_tokenizer: &Sequence,
) -> AHashMap<CompactString, u64> {
    if path.ends_with(".parquet") {
        pre_tokenize_parquet_file(path, text_column, pre_tokenizer)
    } else {
        pre_tokenize_file(path, pre_tokenizer)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut train_files: Vec<String> = Vec::new();
    let mut dev_files: Vec<String> = Vec::new();
    let mut output_path = String::from("merges_rust.txt");
    let mut num_symbols: usize = 10000;
    let mut min_frequency: u64 = 2;
    let mut variant = ParityVariant::Base;
    let mut global_merges: usize = 0;
    let mut window_size: usize = 100;
    let mut alpha: f64 = 2.0;
    let mut ratio: Option<Vec<f64>> = None;
    let mut total_symbols: bool = false;
    let mut config_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                while i < args.len() && !args[i].starts_with('-') {
                    train_files.push(args[i].clone());
                    i += 1;
                }
            }
            "--dev" | "-d" => {
                i += 1;
                while i < args.len() && !args[i].starts_with('-') {
                    dev_files.push(args[i].clone());
                    i += 1;
                }
            }
            "--output" | "-o" => {
                i += 1;
                output_path = args[i].clone();
                i += 1;
            }
            "--symbols" | "-s" => {
                i += 1;
                num_symbols = args[i].parse().expect("Invalid --symbols");
                i += 1;
            }
            "--min-frequency" => {
                i += 1;
                min_frequency = args[i].parse().expect("Invalid --min-frequency");
                i += 1;
            }
            "--variant" => {
                i += 1;
                variant = match args[i].as_str() {
                    "base" => ParityVariant::Base,
                    "window" => ParityVariant::Window,
                    _ => panic!("Unknown variant: {}. Use 'base' or 'window'.", args[i]),
                };
                i += 1;
            }
            "--global-merges" | "-g" => {
                i += 1;
                global_merges = args[i].parse().expect("Invalid --global-merges");
                i += 1;
            }
            "--window-size" | "-w" => {
                i += 1;
                window_size = args[i].parse().expect("Invalid --window-size");
                i += 1;
            }
            "--alpha" => {
                i += 1;
                alpha = args[i].parse().expect("Invalid --alpha");
                i += 1;
            }
            "--ratio" => {
                i += 1;
                let mut ratios = Vec::new();
                while i < args.len() && !args[i].starts_with('-') {
                    ratios.push(args[i].parse::<f64>().expect("Invalid --ratio value"));
                    i += 1;
                }
                if !ratios.is_empty() {
                    ratio = Some(ratios);
                }
            }
            "--total-symbols" => {
                total_symbols = true;
                i += 1;
            }
            "--config" | "-c" => {
                i += 1;
                config_path = Some(args[i].clone());
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("Parity-aware BPE trainer (Rust optimized)");
                eprintln!("Usage: parity_bpe_train [--input <files...> | --config <json>] --output <file> --symbols <N>");
                eprintln!("  --config <file>    JSON config with language groups (overrides --input/--ratio)");
                eprintln!("  --variant base|window");
                eprintln!("  --global-merges N");
                eprintln!("  --min-frequency N");
                eprintln!("  --window-size N  --alpha F");
                eprintln!("  --ratio <floats...>  (compression ratios per language, alternative to --dev)");
                eprintln!("  --total-symbols  (subtract unique chars from --symbols)");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    // Build pre-tokenizer: Whitespace + ByteLevel(use_regex=False)
    let pre_tokenizer = Sequence::new(vec![
        PreTokenizerWrapper::Whitespace(Whitespace),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(true, true, false)),
    ]);

    let pretok_start = Instant::now();

    if let Some(ref cfg_path) = config_path {
        // Config-driven training
        let config = TrainingConfig::from_file(cfg_path)
            .unwrap_or_else(|e| panic!("Cannot load config {}: {}", cfg_path, e));

        let num_langs = config.languages.len();
        let config_ratios = config.ratios();

        eprintln!(
            "Parity-aware BPE (Rust) | variant={:?} | languages={} (from config) | symbols={}",
            variant, num_langs, num_symbols
        );

        let builder = ParityBpeTrainer::builder()
            .min_frequency(min_frequency)
            .num_merges(num_symbols)
            .show_progress(true)
            .variant(variant)
            .global_merges(global_merges)
            .window_size(window_size)
            .alpha(alpha)
            .total_symbols(total_symbols)
            .ratio(config_ratios);
        let mut trainer = builder.build();

        eprintln!("Pre-tokenizing training files from config...");
        for (lang_idx, lang_cfg) in config.languages.iter().enumerate() {
            eprintln!("  [{}] {} ({} files, ratio={})", lang_idx, lang_cfg.name, lang_cfg.input.len(), lang_cfg.ratio);
            let mut merged_counts: AHashMap<CompactString, u64> = AHashMap::new();
            for file_path in &lang_cfg.input {
                let file_counts = pre_tokenize_auto(file_path, &lang_cfg.text_column, &pre_tokenizer);
                for (word, count) in file_counts {
                    *merged_counts.entry(word).or_default() += count;
                }
            }
            eprintln!("    {} unique words", merged_counts.len());
            trainer.feed_language(lang_idx, merged_counts);
        }

        eprintln!("Pre-tokenization took: {:.2?}", pretok_start.elapsed());

        // Train
        let train_start = Instant::now();
        eprintln!("Training {} merges...", num_symbols);
        let mut model = BPE::default();
        let (_special_tokens, merge_strings) =
            trainer.do_train(&mut model).expect("Training failed");
        let train_duration = train_start.elapsed();
        eprintln!("Training took: {:.2?}", train_duration);

        // Write merge rules
        let mut out = File::create(&output_path).expect("Cannot create output file");
        writeln!(out, "#version: 0.2").unwrap();
        for merge_line in &merge_strings {
            writeln!(out, "{}", merge_line).unwrap();
        }

        eprintln!(
            "Wrote {} merges to {}",
            merge_strings.len(),
            output_path
        );
        eprintln!("Total time: {:.2?}", pretok_start.elapsed());
    } else {
        // Original --input file-per-language mode
        assert!(
            !train_files.is_empty(),
            "No training files specified (use --input or --config)"
        );
        let num_langs = train_files.len();
        eprintln!(
            "Parity-aware BPE (Rust) | variant={:?} | languages={} | symbols={}",
            variant, num_langs, num_symbols
        );

        eprintln!("Pre-tokenizing training files...");
        let mut builder = ParityBpeTrainer::builder()
            .min_frequency(min_frequency)
            .num_merges(num_symbols)
            .show_progress(true)
            .variant(variant)
            .global_merges(global_merges)
            .window_size(window_size)
            .alpha(alpha)
            .total_symbols(total_symbols);
        if let Some(r) = ratio {
            builder = builder.ratio(r);
        }
        let mut trainer = builder.build();

        for (lang, path) in train_files.iter().enumerate() {
            eprintln!("  [{}] {}", lang, path);
            let word_counts = pre_tokenize_file(path, &pre_tokenizer);
            eprintln!("    {} unique words", word_counts.len());
            trainer.feed_language(lang, word_counts);
        }

        // Pre-tokenize and feed dev files
        if !dev_files.is_empty() {
            eprintln!("Pre-tokenizing dev files...");
            for (lang, path) in dev_files.iter().enumerate() {
                eprintln!("  [{}] {}", lang, path);
                let word_counts = pre_tokenize_file(path, &pre_tokenizer);
                eprintln!("    {} unique words", word_counts.len());
                trainer.feed_dev_language(lang, word_counts);
            }
        }

        eprintln!("Pre-tokenization took: {:.2?}", pretok_start.elapsed());

        // Train
        let train_start = Instant::now();
        eprintln!("Training {} merges...", num_symbols);
        let mut model = BPE::default();
        let (_special_tokens, merge_strings) =
            trainer.do_train(&mut model).expect("Training failed");
        let train_duration = train_start.elapsed();
        eprintln!("Training took: {:.2?}", train_duration);

        // Write merge rules
        let mut out = File::create(&output_path).expect("Cannot create output file");
        writeln!(out, "#version: 0.2").unwrap();
        for merge_line in &merge_strings {
            writeln!(out, "{}", merge_line).unwrap();
        }

        eprintln!(
            "Wrote {} merges to {}",
            merge_strings.len(),
            output_path
        );
        eprintln!("Total time: {:.2?}", pretok_start.elapsed());
    }
}
