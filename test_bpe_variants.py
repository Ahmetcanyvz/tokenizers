from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# ---- Hand-crafted corpus (characters only, whitespace separated) ----
CORPUS = """
aa aa aa aa aa aa aa aa
pq pq pq pq pq
uv uv uv uv u v
kl kl kl
mn mn
z z z
w w w w w w w w w w w w w w w
""".strip()

CORPUS_LINES = [ln.strip() for ln in CORPUS.splitlines() if ln.strip()]

EXPECTED = {
    "count":     ["a a", "p q", "u v", "k l", "m n"],
    "exact_ll":  ["p q", "k l", "u v", "m n", "a a"],
    "approx_ll": ["p q", "u v", "k l", "m n", "a a"],
}

def train_and_merges(score_by: str, stop_by: str, k: int = 5, tag: str = ""):
    # Plain BPE model + Whitespace pre-tokenizer to match hand math exactly
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Whitespace()

    tr = trainers.BpeTrainer(
        vocab_size=100,          # small cap is fine; we only look at the first 5 merges
        min_frequency=1,
        show_progress=True,
        special_tokens=[],
        score_by=score_by,       # "count" | "exact_ll" | "approx_ll"
        stop_by=stop_by,         # "vocab_size" | "delta_ll_exact" | "delta_ll_approx"
    )
    tok.train_from_iterator(CORPUS_LINES, tr)

    out_dir = Path(f"out_{tag or score_by}")
    out_dir.mkdir(exist_ok=True)
    _, merges_path = tok.model.save(str(out_dir), tag or score_by)

    merges = []
    with open(merges_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            merges.append(line.strip())  # "x y"
            if len(merges) == k:
                break
    return merges, merges_path

def main():
    runs = [
        ("count",     "vocab_size",      "count"),
        ("exact_ll",  "vocab_size",  "exact_ll"),
        ("approx_ll", "vocab_size", "approx_ll"),
    ]
    for score_by, stop_by, tag in runs:
        m, path = train_and_merges(score_by, stop_by, 5, tag)
        print(f"{tag:>10}  first 5 merges => {m}   [file: {path}]")
        assert m == EXPECTED[tag], f"{tag} mismatch:\n  got      {m}\n  expected {EXPECTED[tag]}"

    print("\nAll three variants matched the expected first-5 merges:")
    for tag, seq in EXPECTED.items():
        print(f"  {tag:>10} -> {seq}")

if __name__ == "__main__":
    main()