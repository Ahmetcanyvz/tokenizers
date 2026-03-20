//! Shared pre-tokenization utilities for the parity-aware BPE trainer.
//!
//! Used by both the Rust CLI binary and the Python bindings to avoid duplication.

use ahash::AHashMap;
use compact_str::CompactString;
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::tokenizer::Result;
use crate::{
    NormalizedString, Normalizer, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer,
};

/// Process a single text string: normalize, pre-tokenize, and accumulate word counts.
pub fn pre_tokenize_text(
    text: &str,
    normalizer: Option<&dyn Normalizer>,
    pre_tokenizer: Option<&dyn PreTokenizer>,
    word_counts: &mut AHashMap<CompactString, u64>,
) -> Result<()> {
    let normalized_text = if let Some(norm) = normalizer {
        let mut normalized = NormalizedString::from(text);
        norm.normalize(&mut normalized)?;
        normalized.get().to_string()
    } else {
        text.to_string()
    };

    if let Some(pretok) = pre_tokenizer {
        let mut pretokenized = PreTokenizedString::from(normalized_text.as_str());
        pretok.pre_tokenize(&mut pretokenized)?;

        let splits =
            pretokenized.get_splits(OffsetReferential::Original, OffsetType::Byte);
        for (word, _, _) in splits {
            if !word.is_empty() {
                *word_counts.entry(CompactString::from(word)).or_default() += 1;
            }
        }
    } else {
        let word = normalized_text.trim();
        if !word.is_empty() {
            *word_counts.entry(CompactString::from(word)).or_default() += 1;
        }
    }
    Ok(())
}

/// Read a text file line by line, pre-tokenize each line, and return word counts.
///
/// Uses `read_line` to preserve trailing newlines, matching Python's `for line in fobj`
/// behavior. This matters for ByteLevel pre-tokenizer where `\n` → `Ċ`.
pub fn pre_tokenize_file(
    path: &str,
    normalizer: Option<&dyn Normalizer>,
    pre_tokenizer: Option<&dyn PreTokenizer>,
) -> Result<AHashMap<CompactString, u64>> {
    let file =
        File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
    let mut reader = BufReader::new(file);
    let mut word_counts: AHashMap<CompactString, u64> = AHashMap::new();

    let mut line = String::new();
    loop {
        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .map_err(|e| format!("Read error in {}: {}", path, e))?;
        if bytes_read == 0 {
            break;
        }
        pre_tokenize_text(&line, normalizer, pre_tokenizer, &mut word_counts)?;
    }
    Ok(word_counts)
}

/// Read a Parquet file, extract text from the given column, pre-tokenize, and return word counts.
#[cfg(feature = "parquet")]
pub fn pre_tokenize_parquet_file(
    path: &str,
    text_column: &str,
    normalizer: Option<&dyn Normalizer>,
    pre_tokenizer: Option<&dyn PreTokenizer>,
) -> Result<AHashMap<CompactString, u64>> {
    use arrow::array::Array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file =
        File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("Failed to create parquet reader for {}: {}", path, e))?;
    let reader = builder
        .build()
        .map_err(|e| format!("Failed to build parquet reader for {}: {}", path, e))?;
    let mut word_counts: AHashMap<CompactString, u64> = AHashMap::new();

    for batch in reader {
        let batch =
            batch.map_err(|e| format!("Failed to read batch from {}: {}", path, e))?;
        let col = batch.column_by_name(text_column).ok_or_else(|| {
            format!("Column '{}' not found in {}", text_column, path)
        })?;

        if let Some(arr) = col
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
        {
            for i in 0..arr.len() {
                if arr.is_null(i) {
                    continue;
                }
                pre_tokenize_text(arr.value(i), normalizer, pre_tokenizer, &mut word_counts)?;
            }
        } else if let Some(arr) = col
            .as_any()
            .downcast_ref::<arrow::array::LargeStringArray>()
        {
            for i in 0..arr.len() {
                if arr.is_null(i) {
                    continue;
                }
                pre_tokenize_text(arr.value(i), normalizer, pre_tokenizer, &mut word_counts)?;
            }
        } else {
            return Err(format!(
                "Column '{}' in {} is not a string type",
                text_column, path
            )
            .into());
        }
    }
    Ok(word_counts)
}

/// Dispatch to the appropriate file reader based on extension.
pub fn pre_tokenize_auto(
    path: &str,
    #[allow(unused_variables)] text_column: &str,
    normalizer: Option<&dyn Normalizer>,
    pre_tokenizer: Option<&dyn PreTokenizer>,
) -> Result<AHashMap<CompactString, u64>> {
    #[cfg(feature = "parquet")]
    if path.ends_with(".parquet") {
        return pre_tokenize_parquet_file(path, text_column, normalizer, pre_tokenizer);
    }
    #[cfg(not(feature = "parquet"))]
    if path.ends_with(".parquet") {
        return Err(format!(
            "Parquet support not compiled in. Rebuild with --features parquet to read {}",
            path
        )
        .into());
    }
    pre_tokenize_file(path, normalizer, pre_tokenizer)
}
