# Parity-Aware BPE

This fork adds **Parity-Aware BPE** to the HuggingFace tokenizers library — a modified BPE algorithm that ensures cross-lingual fairness in tokenization.

Paper: ["Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization"](https://arxiv.org/abs/2508.04796) (arXiv 2025)

## Installation

```bash
git clone https://github.com/swiss-ai/parity-aware-bpe.git
cd parity-aware-bpe/tokenizers/bindings/python
pip install -e .
```

Requires Rust 1.70+ and Python 3.9+.

## Python API

```python
from tokenizers.trainers import ParityBpeTrainer

trainer = ParityBpeTrainer(num_merges=32000, variant="base")
tokenizer = trainer.train(
    train_files=["train_en.txt", "train_de.txt", "train_fr.txt"],
    dev_files=["dev_en.txt", "dev_de.txt", "dev_fr.txt"],
)

# Returns a standard tokenizers.Tokenizer
encoded = tokenizer.encode("Hello world")
print(encoded.tokens)

# Save / load
tokenizer.save("my_tokenizer.json")

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("my_tokenizer.json")
```

## ParityBpeTrainer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_merges` | `32000` | Number of BPE merge operations to perform |
| `variant` | `"base"` | Algorithm variant: `"base"` or `"window"` (moving-window balancing) |
| `min_frequency` | `2` | Minimum pair frequency to merge |
| `ratio` | `None` | Target compression ratios per language (alternative to `dev_files`) |
| `global_merges` | `0` | Number of initial standard BPE merges before switching to parity mode |
| `window_size` | `100` | Window size for the `"window"` variant |
| `alpha` | `2.0` | Alpha parameter for the `"window"` variant |
| `total_symbols` | `False` | If True, subtract unique character count from `num_merges` |

## trainer.train() Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `train_files` | Yes | List of training file paths, one per language |
| `dev_files` | No | List of dev file paths for parity computation (multi-parallel) |
| `ratio` | No | Target compression ratios per language (alternative to `dev_files`) |
| `output` | No | Path to write merge rules to a file |

Either `dev_files` or `ratio` must be provided for parity computation. The tool assumes that the nth training file corresponds to the nth dev file (same language order).

## Rust CLI

```bash
cd tokenizers/tokenizers
cargo build --release --bin parity_bpe_train

./target/release/parity_bpe_train \
    --symbols 32000 \
    --variant base \
    --input train_en.txt train_de.txt train_fr.txt \
    --dev dev_en.txt dev_de.txt dev_fr.txt \
    --output merges.txt
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--symbols` | Number of BPE merges to perform |
| `--variant` | `base` (default) or `window` |
| `--input` | Space-separated training files (one per language) |
| `--dev` | Space-separated dev files for parity computation |
| `--ratio` | Space-separated target compression ratios per language |
| `--output` | Output file for merge rules |
| `--global-merges` | Initial standard BPE merges before parity mode |
| `--total-symbols` | Subtract character count from merge count |
| `--min-frequency` | Minimum pair frequency (default: 2) |
| `--window-size` | Window size for `window` variant (default: 100) |
| `--alpha` | Alpha for `window` variant (default: 2.0) |

## Algorithm Variants

- **base**: At each merge step, selects the language with the longest total token length on the dev set (or furthest from target compression ratio), then merges the most frequent pair in that language's training data.

- **window**: Uses a moving-window mechanism to prevent any single language from dominating merge selections. Controlled by `window_size` and `alpha`.

## Citation

```bibtex
@article{foroutan-meister-et-al-2025-parity-aware-bpe,
  title={Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization},
  author={Foroutan, Negar and Meister, Clara and Paul, Debjit and Niklaus, Joel and Ahmadi, Sina and Bosselut, Antoine and Sennrich, Rico},
  url={https://arxiv.org/abs/2508.04796},
  booktitle={arXiv},
  year={2025}
}
```
