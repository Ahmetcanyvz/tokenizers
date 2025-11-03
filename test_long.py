# test_long.py
# Verifies greedy compression pruning against a precomputed expectation:
#   (1) percent pruning (single and two rounds) using trainer-like tie-break
#       AND per-step “removed token” equality with the trainer
#   (2) byte_fallback segmentation on OOV characters by injecting <0x00>.. <0xFF> tokens
#       AND ensuring <unk> is present + unk_id set (needed for DP over unknown chars)

from collections import Counter
import json
import math
import os
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import CompressionTrainer

# ---------------------------------------------------------------------
# Helpers (model-only simulator under unit-cost)
# ---------------------------------------------------------------------

UNIT_COST = -1.0  # every piece costs 1 token

def make_unit_cost_tokenizer(tokens, byte_fallback=False):
    """
    Build Tokenizer(Unigram) with EXACT vocab (token, UNIT_COST) pairs.
    unk_id=None, byte_fallback={byte_fallback}.
    """
    model = Unigram(vocab=[(t, UNIT_COST) for t in tokens],
                    unk_id=None,
                    byte_fallback=byte_fallback)
    tok = Tokenizer(model)
    try:
        tok.model.set_unit_cost(True)  # no-op if not supported
    except Exception:
        pass
    return tok

def vocab_list(tok):
    """Return vocab tokens ordered by id for readability."""
    v = tok.get_vocab()
    inv = {i: t for t, i in v.items()}
    return [inv[i] for i in sorted(inv)]

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

def remove_k_greedy_like_trainer(seed_vocab, keep_tokens, spans, k):
    """
    Delete exactly k tokens by repeated greedy Δ minimization,
    resegmenting after each deletion, using the trainer's tiebreak:
      (Δ, current vocab order).
    Return (removed_order, final_vocab).
    """
    vocab = list(seed_vocab)           # current vocab; order ~ id order
    tok = make_unit_cost_tokenizer(vocab)
    removed = []

    for _ in range(k):
        cand = [t for t in vocab if t not in keep_tokens]
        if not cand:
            break

        c = counts_c(tok, spans)
        deltas = {}
        for t in cand:
            d_t = d_t_under_v_minus_t(t, vocab, t)
            deltas[t] = c.get(t, 0) * (d_t - 1)

        # Trainer scans ids in order and picks the first minimal Δ.
        pos = {t: i for i, t in enumerate(vocab)}
        t_star = min(cand, key=lambda t: (deltas[t], pos[t]))

        before = total_tokens(tok, spans)
        vocab = [t for t in vocab if t != t_star]
        tok = make_unit_cost_tokenizer(vocab)
        after = total_tokens(tok, spans)
        assert (after - before) == deltas[t_star], \
            f"Expected Δ={deltas[t_star]} but got {after - before} for '{t_star}'"

        removed.append(t_star)

    return removed, vocab

def train_with_seed(corpus, seed_vocab, target_size):
    """
    Train the CompressionTrainer from a fixed seed to a target vocab size.
    """
    trainer = CompressionTrainer(
        vocab_size=target_size,
        show_progress=False,
        max_piece_length=max(len(t) for t in seed_vocab),
        seed_size=max(10_000, len(seed_vocab)*2),
        seed_vocab=seed_vocab,   # requires the binding we added
    )
    tok = Tokenizer(Unigram())  # model replaced by training
    tok.train_from_iterator(corpus, trainer=trainer)
    return tok

def trainer_remove_k_step_by_step(corpus, seed_vocab, k):
    """
    Remove exactly k tokens by calling the trainer k times, each time pruning 1 token,
    and diff the vocab between steps to record the exact removed token.
    Return (removed_order, final_vocab).
    """
    current = list(seed_vocab)
    removed = []
    for _ in range(k):
        tok = train_with_seed(corpus, current, target_size=len(current)-1)
        new_vocab = vocab_list(tok)
        diff = sorted(set(current) - set(new_vocab))
        assert len(diff) == 1, f"Expected 1 removal, got {diff}"
        removed.append(diff[0])
        current = new_vocab
    return removed, current

# ---------------------------------------------------------------------
# Controlled scenario
# ---------------------------------------------------------------------

# Σ (kept forever)
SIGMA = list("abcde")

# Seed pieces (chain+blocks + a couple of off-path)
PIECES = [
    "ab","bc","cd","de",
    "abc","bcd","cde",
    "abcd","bcde","abcde",
    "abe","ace",
]
SEED = SIGMA + PIECES

# Corpus with varied lengths/frequencies (no spaces)
CORPUS = [
    "abcde","abcde","abcde",   # 3× abcde
    "abcd","abcd",             # 2× abcd
    "abc","abc",               # 2× abc
    "bcd","bcd",               # 2× bcd
    "cde","cde",               # 2× cde
    "ab","de",                 # singles
    "abe","ace",               # rare/off-path
]

# ---------------------------------------------------------------------
# 1) Percent prune check (single round and two rounds)
# ---------------------------------------------------------------------
def test_percent_round(prune_pct=1/3):
    """
    Remove ceil(prune_pct * (#deletable)) tokens in one 'percent round' by:
      - simulating greedy deletions (trainer-like tiebreak) to compute the expected vocab,
      - training to the matching target and asserting vocab equality,
      - AND verifying per-step removed tokens by calling trainer k times
        and diffing the vocab each step.
    Then repeat a second round starting from the intermediate vocab.
    """
    # ---- Round 1 ----
    deletable = len(SEED) - len(SIGMA)
    k1 = math.ceil(prune_pct * deletable)

    # Simulate k1 deletions
    sim_removed1, expected_vocab1 = remove_k_greedy_like_trainer(SEED, SIGMA, CORPUS, k1)
    print(f"[Percent Round 1] deletable={deletable}, pct={prune_pct:.2f}, k={k1}")
    print(f"Removed (simulated): {sim_removed1}")
    print(f"Expected vocab size after round: {len(expected_vocab1)}")

    # One-shot train to the same target and check final vocab equality
    tok1 = train_with_seed(CORPUS, SEED, target_size=len(SEED) - k1)
    got_vocab1 = set(vocab_list(tok1))
    exp_vocab1 = set(expected_vocab1)
    assert got_vocab1 == exp_vocab1, \
        f"Trainer vocab != expected after round 1.\nGot: {sorted(got_vocab1)}\nExp: {sorted(exp_vocab1)}"
    print("✓ Round 1 final vocab matches expected.")

    # Per-step trainer check: call trainer k1 times to record removed tokens
    tr_removed1, tr_vocab_after_1 = trainer_remove_k_step_by_step(CORPUS, SEED, k1)
    assert tr_removed1 == sim_removed1, \
        f"Trainer per-step removals != simulated in round 1.\nGot: {tr_removed1}\nExp: {sim_removed1}"
    assert set(tr_vocab_after_1) == exp_vocab1, \
        "Trainer per-step final vocab != expected in round 1."
    print(f"✓ Round 1 per-step removals match: {tr_removed1}")

    # ---- Round 2 ----
    deletable2 = len(expected_vocab1) - len(SIGMA)
    k2 = math.ceil(prune_pct * deletable2)

    # Simulate k2 deletions from expected_vocab1
    sim_removed2, expected_vocab2 = remove_k_greedy_like_trainer(expected_vocab1, SIGMA, CORPUS, k2)
    print(f"[Percent Round 2] deletable={deletable2}, pct={prune_pct:.2f}, k={k2}")
    print(f"Removed (simulated): {sim_removed2}")

    # One-shot train from expected_vocab1 and check final vocab equality
    tok2 = train_with_seed(CORPUS, expected_vocab1, target_size=len(expected_vocab1) - k2)
    got_vocab2 = set(vocab_list(tok2))
    exp_vocab2 = set(expected_vocab2)
    assert got_vocab2 == exp_vocab2, \
        f"Trainer vocab != expected after round 2.\nGot: {sorted(got_vocab2)}\nExp: {sorted(exp_vocab2)}"
    print("✓ Round 2 final vocab matches expected.")

    # Per-step trainer check for round 2
    tr_removed2, tr_vocab_after_2 = trainer_remove_k_step_by_step(CORPUS, expected_vocab1, k2)
    assert tr_removed2 == sim_removed2, \
        f"Trainer per-step removals != simulated in round 2.\nGot: {tr_removed2}\nExp: {sim_removed2}"
    assert set(tr_vocab_after_2) == exp_vocab2, \
        "Trainer per-step final vocab != expected in round 2."
    print(f"✓ Round 2 per-step removals match: {tr_removed2}")

    # Return an intermediate tokenizer that still contains 'abc' and 'de'
    return tok1

# ---------------------------------------------------------------------
# 2) Byte fallback check
# ---------------------------------------------------------------------

def ensure_unk_in_json(model_dict):
    """
    Ensure <unk> exists in vocab and set model['unk_id'] to its index.
    This is REQUIRED so the DP can place an 'unknown' piece for OOV codepoints
    before tokenize() explodes them into bytes via byte_fallback.
    """
    vocab = model_dict.get("vocab")
    assert isinstance(vocab, list), "Malformed Unigram JSON: 'vocab' missing"
    unk_idx = None
    for i, (tok, _score) in enumerate(vocab):
        if tok == "<unk>":
            unk_idx = i
            break
    if unk_idx is None:
        vocab.insert(0, ["<unk>", 0.0])
        unk_idx = 0
    model_dict["unk_id"] = unk_idx

def ensure_byte_tokens_in_json(model_dict):
    """
    Append any missing <0xXX> tokens to the Unigram vocab (score = -1.0).
    """
    vocab = model_dict.get("vocab")
    assert isinstance(vocab, list), "Malformed Unigram JSON: 'vocab' missing"

    present = {t for (t, _score) in vocab}
    added = 0
    for b in range(256):
        token = f"<0x{b:02X}>"
        if token not in present:
            vocab.append([token, -1.0])
            added += 1
    return added

def enable_byte_fallback_on_saved_json(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model = data.get("model", {})
    assert model.get("type") == "Unigram", "Model is not Unigram"

    ensure_unk_in_json(model)
    model["byte_fallback"] = True
    ensure_byte_tokens_in_json(model)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def test_byte_fallback(tok_with_abc_de):
    """
    Ensure that with byte_fallback=True and byte tokens present (and <unk> set),
    an OOV char yields as many fallback pieces as its UTF-8 byte length, and
    boundary pieces remain intact.
    We'll use tokens 'abc' and 'de' that exist in the intermediate vocab after round 1.
    """
    tmp_plain = "tmp_unigram.json"
    tmp_fb = "tmp_unigram_fb.json"
    tok_with_abc_de.save(tmp_plain)
    enable_byte_fallback_on_saved_json(tmp_plain, tmp_fb)

    tok_fb = Tokenizer.from_file(tmp_fb)

    s1 = "abc🙂de"     # '🙂' is 4 bytes
    s2 = "abc\u00E9de" # 'é' is 2 bytes

    seg1 = tok_fb.encode(s1).tokens
    seg2 = tok_fb.encode(s2).tokens

    b1 = len("🙂".encode("utf-8"))
    b2 = len("é".encode("utf-8"))

    assert seg1[0] == "abc" and seg1[-1] == "de", f"Unexpected boundaries for {s1}: {seg1}"
    assert len(seg1) == 1 + b1 + 1, f"{s1} expected {1+b1+1} tokens, got {len(seg1)}: {seg1}"

    assert seg2[0] == "abc" and seg2[-1] == "de", f"Unexpected boundaries for {s2}: {seg2}"
    assert len(seg2) == 1 + b2 + 1, f"{s2} expected {1+b2+1} tokens, got {len(seg2)}: {seg2}"

    for p in (tmp_plain, tmp_fb):
        try:
            os.remove(p)
        except Exception:
            pass

    print("✓ Byte fallback counts match UTF-8 byte lengths and keep boundaries intact.")

# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Percent-based pruning checks (single and two rounds), including per-step removed tokens
    tok_intermediate = test_percent_round(prune_pct=1/3)

    # 2) Byte fallback checks using an intermediate model that still contains 'abc' and 'de'
    test_byte_fallback(tok_intermediate)

    print("\nAll checks passed ✅")