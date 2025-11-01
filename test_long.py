# test_prune_complex.py
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import CompressionTrainer

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_unit_cost_tokenizer(tokens):
    """
    Build a Tokenizer(Unigram) where every token has score -1.0 (unit-cost),
    unk_id=None, byte_fallback=False.
    """
    model = Unigram(vocab=[(t, -1.0) for t in tokens], unk_id=None, byte_fallback=False)
    tok = Tokenizer(model)
    # If your binding exposes this, it's a no-op; otherwise harmless:
    try:
        tok.model.set_unit_cost(True)
    except Exception:
        pass
    return tok

def get_vocab_list(tok):
    """Token list ordered by id (for stable display)."""
    v = tok.get_vocab()
    inv = {i: t for t, i in v.items()}
    return [inv[i] for i in sorted(inv)]

def segment(tok, spans):
    """Return list of tokens for each span."""
    return [tok.encode(s).tokens for s in spans]

def total_tokens(tok, spans):
    return sum(len(tok.encode(s).tokens) for s in spans)

def counts_c(tok, spans):
    c = Counter()
    for s in spans:
        for t in tok.encode(s).tokens:
            c[t] += 1
    return c

def d_t_under_v_minus_t(token_str, current_vocab, t_remove):
    """Length of best segmentation of token_str under V \\ {t_remove}."""
    v_minus = [x for x in current_vocab if x != t_remove]
    t2 = make_unit_cost_tokenizer(v_minus)
    seg = t2.encode(token_str).tokens
    return len(seg) if seg else max(1, len(token_str))  # guard

# ---------------------------------------------------------------------
# Test scenario (richer than abcd)
# ---------------------------------------------------------------------

# Σ (kept forever)
SIGMA = list("abcde")

# Multi-char pieces we'll seed (chain + a couple of 'off-path' pieces)
PIECES = [
    "ab","bc","cd","de",
    "abc","bcd","cde",
    "abcd","bcde","abcde",
    "abe","ace",    # off-path / rare
]

SEED = SIGMA + PIECES

# Corpus with varied lengths and frequencies (no spaces)
CORPUS = [
    "abcde","abcde","abcde",   # 3× abcde
    "abcd","abcd",             # 2× abcd
    "abc","abc",               # 2× abc
    "bcd","bcd",               # 2× bcd
    "cde","cde",               # 2× cde
    "ab","de",                 # singles
    "abe","ace",               # off-path pieces occur once
]

# ---------------------------------------------------------------------
# Greedy pruning on top of the tokenizer (model-only, uses library DP)
# ---------------------------------------------------------------------

def greedy_prune(spans, seed_tokens, keep_tokens):
    vocab = list(seed_tokens)
    tok = make_unit_cost_tokenizer(vocab)

    print("Seed vocab:", vocab)
    print("Corpus:", spans, "\n")

    # Print baseline
    print("Step 0 (no prune):")
    print("Vocab:", vocab)
    print("Segmentations:")
    for s in spans:
        print(f"  {s:<8} -> {tok.encode(s).tokens}")
    print("Total tokens:", total_tokens(tok, spans), "\n")

    prune_order = []
    step = 1

    while set(vocab) - set(keep_tokens):
        # Counts from current best paths
        c = counts_c(tok, spans)

        # Candidate tokens = everything except Σ
        cand = [t for t in vocab if t not in keep_tokens]

        # Compute Δ for each candidate (and d_t for trace)
        deltas = {}
        dt_map = {}
        for t in cand:
            d_t = d_t_under_v_minus_t(t, vocab, t)
            dt_map[t] = d_t
            deltas[t] = c.get(t, 0) * (d_t - 1)

        # Choose argmin Δ with a stable tiebreak
        # (Δ, count, length, lex) keeps behavior deterministic for display
        t_star = min(cand, key=lambda t: (deltas[t], c.get(t, 0), len(t), t))
        delta_star = deltas[t_star]

        # Sanity: actual increase in token usage equals Δ
        before = total_tokens(tok, spans)
        vocab = [t for t in vocab if t != t_star]
        tok = make_unit_cost_tokenizer(vocab)
        after = total_tokens(tok, spans)
        actual_delta = after - before

        # Trace
        print(f"Step {step}: removed -> '{t_star}' "
              f"(Δ = {delta_star}, d[{t_star}] = {dt_map[t_star]}, c[{t_star}] = {c.get(t_star,0)})")
        # Show the top few Δs to make the step explainable
        top = sorted(deltas.items(), key=lambda kv: (kv[1], c.get(kv[0], 0), len(kv[0]), kv[0]))[:6]
        print("Top Δ candidates:", top)
        print("Vocab:", vocab)
        print("Segmentations:")
        for s in spans:
            print(f"  {s:<8} -> {tok.encode(s).tokens}")
        print(f"Total tokens: {after} (Δ = {actual_delta})\n")

        # Assertion: theory vs reality
        assert actual_delta == delta_star, (
            f"Token '{t_star}': expected Δ={delta_star}, got {actual_delta}"
        )

        prune_order.append(t_star)
        step += 1

        # Stop when only Σ remains
        if set(vocab) == set(keep_tokens):
            break

    return prune_order, vocab, tok

# ---------------------------------------------------------------------
# Run the greedy test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    order, final_vocab, final_tok = greedy_prune(CORPUS, SEED, SIGMA)

    print("=== Summary ===")
    print("Prune order:", order)
    print("Final vocab:", final_vocab)
    print("Final segmentations:")
    for s in CORPUS:
        print(f"  {s:<8} -> {final_tok.encode(s).tokens}")
    print()

    # -----------------------------------------------------------------
    # Sanity check with the actual CompressionTrainer:
    # We train down to |Σ|, which (by design) must end with Σ only.
    # -----------------------------------------------------------------
    trainer = CompressionTrainer(
        vocab_size=len(SIGMA),
        show_progress=False,
        max_piece_length=max(len(t) for t in SEED),
        seed_size=10_000,  # any large cap
        # NOTE: If your Python wrapper accepts seed_vocab, you can pass it:
        # seed_vocab=SEED,
    )
    t_train = Tokenizer(Unigram())
    t_train.train_from_iterator(CORPUS, trainer=trainer)

    trained_vocab = set(get_vocab_list(t_train))
    expected_vocab = set(SIGMA)  # final vocab must be exactly Σ
    print("Trainer final vocab:", sorted(trained_vocab))
    assert expected_vocab.issubset(trained_vocab), "Σ missing from trained vocab!"
    assert trained_vocab == expected_vocab, \
        f"Trainer ended with {sorted(trained_vocab)}, expected {sorted(expected_vocab)}"

    print("\nAll checks passed ✅")