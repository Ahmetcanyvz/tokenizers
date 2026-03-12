#!/usr/bin/env python3
"""Unified BPE training script supporting both standard and parity-aware BPE."""

import argparse
import json
import os
import tempfile

import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer, ParityBpeTrainer


def parquet_to_text(parquet_path, output_dir, text_column="text", batch_size=4096):
    """Convert a parquet file to a UTF-8 text file (one document per line)."""
    basename = os.path.splitext(os.path.basename(parquet_path))[0] + ".txt"
    out_path = os.path.join(output_dir, basename)
    pf = pq.ParquetFile(parquet_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for batch in pf.iter_batches(batch_size=batch_size, columns=[text_column]):
            for value in batch.column(text_column):
                text = value.as_py()
                if text:
                    f.write(text.replace("\n", " ") + "\n")
    print(f"  Converted {parquet_path} -> {out_path}")
    return out_path


def build_tokenizer(tokenizer_config_path):
    """Build a Tokenizer with components from an HF-format tokenizer config JSON.

    Uses Tokenizer.from_str() to leverage Rust serde deserializers for all
    component types, then extracts the deserialized components.
    """
    if tokenizer_config_path is None:
        return Tokenizer(BPE())

    with open(tokenizer_config_path) as f:
        components = json.load(f)

    # Build a minimal skeleton tokenizer JSON with an empty BPE model
    skeleton = {
        "version": "1.0",
        "model": {"type": "BPE", "vocab": {}, "merges": []},
    }
    for key in ("pre_tokenizer", "decoder", "normalizer", "post_processor"):
        if key in components and components[key] is not None:
            skeleton[key] = components[key]

    tokenizer = Tokenizer.from_str(json.dumps(skeleton))
    return tokenizer


def flatten_files_from_config(data_config_path):
    """Extract all unique input files from a pa_config.json, preserving text_column info."""
    with open(data_config_path) as f:
        config = json.load(f)

    seen = set()
    files = []  # list of (path, text_column)
    for lang in config["languages"]:
        text_column = lang.get("text_column", "text")
        for path in lang["input"]:
            if path not in seen:
                seen.add(path)
                files.append((path, text_column))
    return files


def train_standard_bpe(args):
    """Train a standard BPE tokenizer."""
    tokenizer = build_tokenizer(args.tokenizer_config)

    # Flatten and deduplicate input files from config
    file_entries = flatten_files_from_config(args.data_config)

    # Convert parquet files to text, pass text files through
    text_dir = tempfile.mkdtemp(prefix="bpe_text_")
    train_files = []
    print(f"Preparing training files in {text_dir} ...")
    for path, text_column in file_entries:
        if path.endswith(".parquet"):
            train_files.append(parquet_to_text(path, text_dir, text_column=text_column))
        else:
            train_files.append(path)

    # Build trainer
    trainer_kwargs = {
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "show_progress": True,
    }
    if args.max_token_length is not None:
        trainer_kwargs["max_token_length"] = args.max_token_length
    if args.special_tokens:
        trainer_kwargs["special_tokens"] = args.special_tokens

    trainer = BpeTrainer(**trainer_kwargs)

    print(f"Training standard BPE (vocab_size={args.vocab_size}) ...")
    tokenizer.train(train_files, trainer=trainer)

    # Clean up temp files
    for f in os.listdir(text_dir):
        os.remove(os.path.join(text_dir, f))
    os.rmdir(text_dir)
    print(f"Cleaned up temp directory {text_dir}")

    return tokenizer


def train_parity_bpe(args):
    """Train a parity-aware BPE tokenizer."""
    tokenizer = build_tokenizer(args.tokenizer_config)

    trainer = ParityBpeTrainer(
        num_merges=args.vocab_size,
        variant=args.variant,
        min_frequency=args.min_frequency,
        global_merges=args.global_merges,
        window_size=args.window_size,
        alpha=args.alpha,
        total_symbols=True,
    )

    print(f"Training parity-aware BPE (vocab_size={args.vocab_size}, variant={args.variant}) ...")
    trainer.train(tokenizer, config=args.data_config)

    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer (standard or parity-aware)"
    )

    # Required arguments
    parser.add_argument(
        "--data-config", required=True,
        help="Path to pa_config.json format data config",
    )
    parser.add_argument(
        "--trainer", required=True, choices=["bpe", "parity-bpe"],
        help="Training algorithm to use",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output tokenizer JSON path",
    )

    # Optional tokenizer component config
    parser.add_argument(
        "--tokenizer-config",
        help="HF-format JSON specifying pre_tokenizer, decoder, normalizer, post_processor",
    )

    # Shared training params
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Final vocabulary size (default: 32000)")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="Min pair frequency (default: 2)")
    parser.add_argument("--special-tokens", nargs="+", default=None,
                        help="Special tokens to add after training")

    # Parity-BPE specific
    parser.add_argument("--variant", choices=["base", "window"], default="base",
                        help="Parity-BPE algorithm variant (default: base)")
    parser.add_argument("--global-merges", type=int, default=0,
                        help="Standard merges before parity mode (default: 0)")
    parser.add_argument("--window-size", type=int, default=100,
                        help="Window size for window variant (default: 100)")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Alpha for window variant (default: 2.0)")

    # Standard BPE specific
    parser.add_argument("--max-token-length", type=int, default=None,
                        help="Max token length for standard BPE")

    args = parser.parse_args()

    if args.trainer == "bpe":
        tokenizer = train_standard_bpe(args)
    else:
        tokenizer = train_parity_bpe(args)

    # Add special tokens after training (for parity-bpe path where they
    # can't be passed to the trainer directly)
    if args.special_tokens and args.trainer == "parity-bpe":
        tokenizer.add_special_tokens(args.special_tokens)

    tokenizer.save(args.output)
    print(f"Saved tokenizer to {args.output}")

    # Quick sanity check
    loaded = Tokenizer.from_file(args.output)
    tokens = loaded.encode("Hello world").tokens
    print(f"Sanity check encode('Hello world'): {tokens}")
    print(f"Vocab size: {loaded.get_vocab_size()}")


if __name__ == "__main__":
    main()
