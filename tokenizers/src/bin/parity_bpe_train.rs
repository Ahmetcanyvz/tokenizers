use ahash::AHashMap;
use compact_str::CompactString;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use tokenizers::models::bpe::{
    parity_utils, ParityBpeTrainer, ParityVariant, TrainingConfig, BPE,
};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;

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
    let mut pretokenize: Vec<String> = Vec::new();

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
            "--pretokenize" => {
                i += 1;
                while i < args.len() && !args[i].starts_with('-') {
                    pretokenize.push(args[i].clone());
                    i += 1;
                }
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
                eprintln!("  --pretokenize <whitespace|bytelevel>...  (default: whitespace bytelevel)");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    // Build pre-tokenizer from --pretokenize args (default: whitespace bytelevel)
    if pretokenize.is_empty() {
        pretokenize = vec!["whitespace".into(), "bytelevel".into()];
    }
    let mut pretok_parts: Vec<PreTokenizerWrapper> = Vec::new();
    for p in &pretokenize {
        match p.as_str() {
            "whitespace" => pretok_parts.push(PreTokenizerWrapper::Whitespace(Whitespace)),
            "bytelevel" => pretok_parts.push(PreTokenizerWrapper::ByteLevel(ByteLevel::new(true, true, false))),
            _ => panic!("Unknown pretokenizer: {}. Use 'whitespace' or 'bytelevel'.", p),
        }
    }
    let pre_tokenizer = Sequence::new(pretok_parts);

    let pretok_start = Instant::now();

    if let Some(ref cfg_path) = config_path {
        // Config-driven training
        let config = TrainingConfig::from_file(cfg_path)
            .unwrap_or_else(|e| panic!("Cannot load config {}: {}", cfg_path, e));

        let num_langs = config.languages.len();
        let has_dev = config.has_dev();

        eprintln!(
            "Parity-aware BPE (Rust) | variant={:?} | languages={} (from config) | symbols={} | dev={}",
            variant, num_langs, num_symbols, has_dev
        );

        let mut builder = ParityBpeTrainer::builder()
            .min_frequency(min_frequency)
            .num_merges(num_symbols)
            .show_progress(true)
            .variant(variant)
            .global_merges(global_merges)
            .window_size(window_size)
            .alpha(alpha)
            .total_symbols(total_symbols);

        // Only use ratios if no dev files are present in the config
        if !has_dev {
            builder = builder.ratio(config.ratios());
        }

        let mut trainer = builder.build();

        eprintln!("Pre-tokenizing training files from config...");
        for (lang_idx, lang_cfg) in config.languages.iter().enumerate() {
            eprintln!("  [{}] {} ({} files, ratio={})", lang_idx, lang_cfg.name, lang_cfg.input.len(), lang_cfg.ratio.unwrap_or(1.0));
            let mut merged_counts: AHashMap<CompactString, u64> = AHashMap::new();
            for file_path in &lang_cfg.input {
                let file_counts = parity_utils::pre_tokenize_auto(file_path, &lang_cfg.text_column, None, Some(&pre_tokenizer)).expect("Pre-tokenization failed");
                for (word, count) in file_counts {
                    *merged_counts.entry(word).or_default() += count;
                }
            }
            eprintln!("    {} unique words", merged_counts.len());
            trainer.feed_language(lang_idx, merged_counts);
        }

        // Feed dev data from config
        if has_dev {
            eprintln!("Pre-tokenizing dev files from config...");
            for (lang_idx, lang_cfg) in config.languages.iter().enumerate() {
                if let Some(ref dev_paths) = lang_cfg.dev {
                    let mut dev_counts: AHashMap<CompactString, u64> = AHashMap::new();
                    for file_path in dev_paths {
                        let file_counts = parity_utils::pre_tokenize_auto(file_path, &lang_cfg.text_column, None, Some(&pre_tokenizer)).expect("Pre-tokenization failed");
                        for (word, count) in file_counts {
                            *dev_counts.entry(word).or_default() += count;
                        }
                    }
                    eprintln!("    [{}] {} dev words", lang_idx, dev_counts.len());
                    trainer.feed_dev_language(lang_idx, dev_counts);
                }
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
            let word_counts = parity_utils::pre_tokenize_file(path, None, Some(&pre_tokenizer)).expect("Pre-tokenization failed");
            eprintln!("    {} unique words", word_counts.len());
            trainer.feed_language(lang, word_counts);
        }

        // Pre-tokenize and feed dev files
        if !dev_files.is_empty() {
            eprintln!("Pre-tokenizing dev files...");
            for (lang, path) in dev_files.iter().enumerate() {
                eprintln!("  [{}] {}", lang, path);
                let word_counts = parity_utils::pre_tokenize_file(path, None, Some(&pre_tokenizer)).expect("Pre-tokenization failed");
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
