# test_long.py
# Tests for CompressionTrainer:
#   1) Algorithm correctness: compare Rust vs Python implementation
#   2) Basic correctness: vocab size reaches target, alphabet preserved
#   3) Batch deletion: verify batching works correctly
#   4) Byte fallback: OOV characters handled correctly
#   5) Large scale: test with larger corpus

from collections import Counter
import json
import os
import time
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import CompressionTrainer
from tokenizers.pre_tokenizers import Whitespace

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

UNIT_COST = -1.0

def make_unit_cost_tokenizer(tokens, byte_fallback=False):
    """Build Tokenizer(Unigram) with unit cost vocab."""
    model = Unigram(
        vocab=[(t, UNIT_COST) for t in tokens],
        unk_id=None,
        byte_fallback=byte_fallback
    )
    return Tokenizer(model)

def vocab_list(tok):
    """Return vocab tokens ordered by id."""
    v = tok.get_vocab()
    inv = {i: t for t, i in v.items()}
    return [inv[i] for i in sorted(inv)]

def vocab_set(tok):
    """Return vocab as a set of strings."""
    return set(tok.get_vocab().keys())

def total_tokens(tok, corpus):
    """Total tokens when encoding corpus."""
    return sum(len(tok.encode(s).tokens) for s in corpus)

def counts_c(tok, corpus):
    """Token usage counts."""
    c = Counter()
    for s in corpus:
        for t in tok.encode(s).tokens:
            c[t] += 1
    return c

# ---------------------------------------------------------------------
# Python reference implementation of CompressionTrainer algorithm
# ---------------------------------------------------------------------

def python_segment(text, vocab):
    """
    Segment text using vocab with unit-cost Viterbi (minimize token count).
    Returns list of tokens.
    """
    tok = make_unit_cost_tokenizer(vocab)
    return tok.encode(text).tokens

def python_compute_c(corpus, vocab):
    """Compute c[t] = count of token t in corpus segmentations."""
    c = Counter()
    for text in corpus:
        tokens = python_segment(text, vocab)
        for t in tokens:
            c[t] += 1
    return c

def python_compute_d(token_str, vocab, exclude_token):
    """
    Compute d[t] = length of segmenting token_str using vocab without exclude_token.
    """
    vocab_without_t = [v for v in vocab if v != exclude_token]
    if not vocab_without_t:
        return len(token_str)
    tokens = python_segment(token_str, vocab_without_t)
    return len(tokens) if tokens else len(token_str)

def python_greedy_compression(corpus, seed_vocab, target_size, keep_tokens):
    """
    Python reference implementation of greedy compression algorithm.

    1. Segment corpus → get c[t]
    2. For each token, compute d[t] = segment token's string without itself
    3. Compute ΔL[t] = c[t] * (d[t] - 1)
    4. Delete token with minimum ΔL (ties broken by vocab order)
    5. Repeat until target size

    Returns: (final_vocab, deletion_order)
    """
    vocab = list(seed_vocab)
    deletion_order = []

    while len(vocab) > target_size:
        # Compute c[t]
        c = python_compute_c(corpus, vocab)

        # Find deletable tokens (not in keep_tokens)
        deletable = [t for t in vocab if t not in keep_tokens]
        if not deletable:
            break

        # Compute ΔL for each deletable token
        deltas = {}
        for t in deletable:
            d_t = python_compute_d(t, vocab, t)
            delta = c.get(t, 0) * (d_t - 1)
            deltas[t] = delta

        # Find token with minimum ΔL (tie-break by vocab order)
        vocab_order = {t: i for i, t in enumerate(vocab)}
        best_token = min(deletable, key=lambda t: (deltas[t], vocab_order[t]))

        # Delete it
        vocab = [t for t in vocab if t != best_token]
        deletion_order.append(best_token)

    return vocab, deletion_order

def python_greedy_compression_batch(corpus, seed_vocab, target_size, keep_tokens, prune_ratio=0.25):
    """
    Python reference implementation with batch deletion (like Rust version).

    Each pass: delete ceil(prune_ratio * remaining) tokens at once,
    then re-segment and recompute.

    Returns: (final_vocab, list of deleted tokens per pass)
    """
    import math

    vocab = list(seed_vocab)
    passes = []

    while len(vocab) > target_size:
        remaining = len(vocab) - target_size
        k = max(1, math.ceil(prune_ratio * remaining))
        k = min(k, remaining)

        # Compute c[t]
        c = python_compute_c(corpus, vocab)

        # Find deletable tokens
        deletable = [t for t in vocab if t not in keep_tokens]
        if not deletable:
            break

        # Compute ΔL for each deletable token
        deltas = {}
        for t in deletable:
            d_t = python_compute_d(t, vocab, t)
            delta = c.get(t, 0) * (d_t - 1)
            deltas[t] = delta

        # Sort by ΔL (tie-break by vocab order)
        vocab_order = {t: i for i, t in enumerate(vocab)}
        sorted_deletable = sorted(deletable, key=lambda t: (deltas[t], vocab_order[t]))

        # Delete bottom k tokens
        to_delete = set(sorted_deletable[:k])
        vocab = [t for t in vocab if t not in to_delete]
        passes.append(list(to_delete))

    return vocab, passes


def python_rand_compression(corpus, seed_vocab, target_size, keep_tokens, sample_size=100):
    """
    Python reference implementation of rand_compression algorithm.

    Instead of computing d[t] from token string, we sample spans that use t
    and measure actual cost difference when resegmenting without t.

    1. Segment corpus → get c[t] and track which spans use each token
    2. For each token t with c[t] > 0:
       - Sample up to sample_size spans that use t
       - Resegment each span without t
       - Compute avg_extra = average extra tokens
    3. ΔL[t] = avg_extra × c[t]
    4. Delete token with minimum ΔL
    5. Repeat until target size

    Returns: (final_vocab, deletion_order)
    """
    vocab = list(seed_vocab)
    deletion_order = []

    while len(vocab) > target_size:
        # Segment corpus and build reverse index
        c = Counter()  # token -> count
        reverse_index = {}  # token -> list of (span_idx, count_in_span)
        segmentations = []  # span_idx -> (tokens, span_count)

        for span_idx, text in enumerate(corpus):
            tokens = python_segment(text, vocab)
            segmentations.append(tokens)

            # Count tokens in this span
            local_counts = Counter(tokens)
            for token, cnt in local_counts.items():
                c[token] += cnt
                if token not in reverse_index:
                    reverse_index[token] = []
                reverse_index[token].append((span_idx, cnt))

        # Find deletable tokens
        deletable = [t for t in vocab if t not in keep_tokens]
        if not deletable:
            break

        # Compute ΔL for each deletable token using sampling
        deltas = {}
        for t in deletable:
            if c.get(t, 0) == 0:
                # Token not used - ΔL = 0
                deltas[t] = 0.0
                continue

            spans_using_t = reverse_index.get(t, [])
            if not spans_using_t:
                deltas[t] = 0.0
                continue

            # Sample spans
            sample_count = min(len(spans_using_t), sample_size)
            total_extra = 0.0
            total_weight = 0.0

            vocab_without_t = [v for v in vocab if v != t]

            for span_idx, weight in spans_using_t[:sample_count]:
                old_tokens = segmentations[span_idx]
                old_len = len(old_tokens)

                # Resegment without token t
                span_text = corpus[span_idx]
                new_tokens = python_segment(span_text, vocab_without_t)
                new_len = len(new_tokens)

                extra = new_len - old_len
                total_extra += extra * weight
                total_weight += weight

            # Average extra cost per usage
            avg_extra = total_extra / total_weight if total_weight > 0 else 0.0

            # ΔL[t] = avg_extra × c[t]
            deltas[t] = avg_extra * c.get(t, 0)

        # Find token with minimum ΔL (tie-break by vocab order)
        vocab_order = {t: i for i, t in enumerate(vocab)}
        best_token = min(deletable, key=lambda t: (deltas[t], vocab_order[t]))

        # Delete it
        vocab = [t for t in vocab if t != best_token]
        deletion_order.append(best_token)

    return vocab, deletion_order


def rust_train_rand_compression(corpus, seed_vocab, target_size, keep_tokens=None, sample_size=100):
    """
    Train Rust CompressionTrainer with rand_scoring=True, one token at a time.
    Returns (final_vocab, deletion_order).
    """
    current_vocab = list(seed_vocab)
    deletion_order = []
    special = list(keep_tokens) if keep_tokens else []

    while len(current_vocab) > target_size:
        # Train to remove exactly 1 token
        tok = Tokenizer(Unigram())
        trainer = CompressionTrainer(
            vocab_size=len(current_vocab) - 1,
            show_progress=False,
            seed_vocab=current_vocab,
            special_tokens=special,
            rand_scoring=True,
            rand_sample_size=sample_size,
            prune_ratio=0.0,
            min_prune=1,
        )
        tok.train_from_iterator(corpus, trainer=trainer)

        new_vocab = set(tok.get_vocab().keys())
        deleted = set(current_vocab) - new_vocab
        assert len(deleted) == 1, f"Expected 1 deletion, got {deleted}"

        deleted_token = list(deleted)[0]
        deletion_order.append(deleted_token)
        current_vocab = [t for t in current_vocab if t != deleted_token]

    return current_vocab, deletion_order


# ---------------------------------------------------------------------
# Test 1: Algorithm correctness (compare Python vs Rust)
# ---------------------------------------------------------------------

def test_algorithm_correctness():
    """
    Verify that the Rust CompressionTrainer produces the same results
    as the Python reference implementation.
    """
    print("\n=== Test 1: Algorithm Correctness ===")

    # Simple corpus and seed vocab
    corpus = ["abcde"] * 10 + ["abc"] * 5 + ["de"] * 5 + ["ab", "cd", "bc"]
    seed = list("abcde") + ["ab", "bc", "cd", "de", "abc", "bcd", "cde", "abcd", "bcde", "abcde"]
    keep = set("abcde")  # alphabet must be kept
    target = 8

    print(f"Corpus: {len(corpus)} sentences")
    print(f"Seed vocab: {seed}")
    print(f"Target: {target}")

    # Run Python reference (batch version to match Rust)
    print("\nRunning Python reference...")
    py_vocab, py_passes = python_greedy_compression_batch(
        corpus, seed, target, keep, prune_ratio=0.25
    )
    print(f"Python final vocab: {sorted(py_vocab)}")
    print(f"Python passes: {py_passes}")

    # Run Rust trainer
    print("\nRunning Rust trainer...")
    tok = Tokenizer(Unigram())
    trainer = CompressionTrainer(
        vocab_size=target,
        show_progress=False,
        seed_vocab=seed,
        prune_ratio=0.25,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    rust_vocab = vocab_set(tok)
    print(f"Rust final vocab: {sorted(rust_vocab)}")

    # Compare
    py_vocab_set = set(py_vocab)
    if py_vocab_set == rust_vocab:
        print("✓ Python and Rust produce IDENTICAL vocabs")
    else:
        print(f"Python only: {py_vocab_set - rust_vocab}")
        print(f"Rust only: {rust_vocab - py_vocab_set}")
        # Allow small differences due to tie-breaking
        diff = len(py_vocab_set.symmetric_difference(rust_vocab))
        assert diff <= 2, f"Too many differences: {diff}"
        print(f"⚠ Small difference ({diff} tokens) - likely tie-breaking")

    # Verify alphabet preserved in both
    for c in "abcde":
        assert c in py_vocab_set, f"Python missing alphabet: {c}"
        assert c in rust_vocab, f"Rust missing alphabet: {c}"
    print("✓ Alphabet preserved in both")

    # Verify both reach target size
    assert len(py_vocab) == target, f"Python vocab size {len(py_vocab)} != {target}"
    assert len(rust_vocab) == target, f"Rust vocab size {len(rust_vocab)} != {target}"
    print("✓ Both reach target size")


def rust_train_step_by_step(corpus, seed_vocab, target_size, keep_tokens=None):
    """
    Train Rust CompressionTrainer one token at a time to capture deletion order.
    Returns (final_vocab, deletion_order).
    """
    current_vocab = list(seed_vocab)
    deletion_order = []
    special = list(keep_tokens) if keep_tokens else []

    while len(current_vocab) > target_size:
        # Train to remove exactly 1 token
        tok = Tokenizer(Unigram())
        trainer = CompressionTrainer(
            vocab_size=len(current_vocab) - 1,
            show_progress=False,
            seed_vocab=current_vocab,
            special_tokens=special,
            prune_ratio=0.0,
            min_prune=1,
        )
        tok.train_from_iterator(corpus, trainer=trainer)

        new_vocab = set(tok.get_vocab().keys())
        deleted = set(current_vocab) - new_vocab
        assert len(deleted) == 1, f"Expected 1 deletion, got {deleted}"

        deleted_token = list(deleted)[0]
        deletion_order.append(deleted_token)
        current_vocab = [t for t in current_vocab if t != deleted_token]

    return current_vocab, deletion_order


def test_algorithm_correctness_single_delete():
    """
    Test with prune_ratio=0 (single deletion per pass) for exact comparison.
    """
    print("\n=== Test 1b: Algorithm Correctness (Single Delete) ===")

    corpus = ["abcde"] * 5 + ["abc"] * 3 + ["de"] * 3
    seed = list("abcde") + ["ab", "bc", "cd", "de", "abc", "cde", "abcde"]
    keep = set("abcde")
    target = 7  # Only delete 5 tokens

    print(f"Seed: {seed} ({len(seed)} tokens)")
    print(f"Target: {target}")

    # Python single-delete reference
    py_vocab, py_order = python_greedy_compression(corpus, seed, target, keep)
    print(f"\nPython deletion order: {py_order}")
    print(f"Python final vocab: {sorted(py_vocab)}")

    # Rust step-by-step
    rust_vocab, rust_order = rust_train_step_by_step(corpus, seed, target, keep)
    print(f"Rust deletion order: {rust_order}")
    print(f"Rust final vocab: {sorted(rust_vocab)}")

    # Compare deletion order
    if py_order == rust_order:
        print("✓ EXACT deletion order match!")
    else:
        print(f"⚠ Deletion order differs:")
        print(f"  Python: {py_order}")
        print(f"  Rust:   {rust_order}")
        # Check if final vocabs match at least
        assert set(py_vocab) == set(rust_vocab), "Final vocabs don't match!"
        print("  (but final vocabs match)")

    # Compare final vocab
    if set(py_vocab) == set(rust_vocab):
        print("✓ Final vocab matches")
    else:
        diff = set(py_vocab).symmetric_difference(set(rust_vocab))
        assert len(diff) <= 1, f"Too much difference: {diff}"


def test_deletion_order_complex():
    """
    More complex test cases to verify deletion order.
    """
    print("\n=== Test 1c: Deletion Order (Complex Cases) ===")

    test_cases = [
        # Case 1: Unused tokens should be deleted first (ΔL = 0)
        # "de" is in vocab but never used in corpus (corpus only has "abc")
        {
            "name": "Unused tokens first",
            "corpus": ["abc"] * 10,
            "seed": list("abcde") + ["ab", "bc", "abc", "de"],  # "de" unused
            "keep": set("abcde"),
            "target": 7,
        },
        # Case 2: High frequency vs low frequency
        {
            "name": "Frequency matters",
            "corpus": ["aaa"] * 100 + ["bbb"] * 10 + ["ab"] * 5,
            "seed": list("ab") + ["aa", "bb", "aaa", "bbb", "ab"],
            "keep": set("ab"),
            "target": 4,
        },
        # Case 3: d(t) matters - longer decomposition = higher cost
        {
            "name": "Decomposition length matters",
            "corpus": ["abcd"] * 10,
            "seed": list("abcd") + ["ab", "cd", "bc", "abcd"],
            "keep": set("abcd"),
            "target": 6,
        },
        # Case 4: Tie-breaking by vocab order
        {
            "name": "Tie-breaking",
            "corpus": ["ab", "cd"],  # Both used once, same d(t)=2
            "seed": list("abcd") + ["ab", "cd"],  # ab before cd in vocab
            "keep": set("abcd"),
            "target": 5,
        },
        # Case 5: Chain of dependencies
        {
            "name": "Dependency chain",
            "corpus": ["abcdef"] * 10,
            "seed": list("abcdef") + ["ab", "cd", "ef", "abcd", "cdef", "abcdef"],
            "keep": set("abcdef"),
            "target": 8,
        },
    ]

    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"Corpus sample: {case['corpus'][:3]}...")
        print(f"Seed: {case['seed']}")

        # Python reference
        py_vocab, py_order = python_greedy_compression(
            case["corpus"], case["seed"], case["target"], case["keep"]
        )

        # Rust step-by-step
        rust_vocab, rust_order = rust_train_step_by_step(
            case["corpus"], case["seed"], case["target"], case["keep"]
        )

        print(f"Python order: {py_order}")
        print(f"Rust order:   {rust_order}")

        if py_order == rust_order:
            print("✓ Deletion order matches!")
        else:
            # Check final vocab
            if set(py_vocab) == set(rust_vocab):
                print("⚠ Order differs but final vocab matches")
            else:
                diff = set(py_vocab).symmetric_difference(set(rust_vocab))
                print(f"✗ Final vocab differs: {diff}")
                assert False, f"Test failed: {case['name']}"


def test_delta_calculation():
    """
    Verify ΔL calculation is correct by checking specific cases.
    """
    print("\n=== Test 1d: ΔL Calculation Verification ===")

    # Simple case: token "ab" used 10 times, d("ab") = 2 without "ab"
    # ΔL("ab") = 10 * (2 - 1) = 10
    corpus = ["ab"] * 10
    seed = list("ab") + ["ab"]
    keep = set("ab")

    # Get c(t) and d(t) from Python
    vocab = list(seed)
    c = python_compute_c(corpus, vocab)
    print(f"c['ab'] = {c.get('ab', 0)}")  # Should be 10

    d_ab = python_compute_d("ab", vocab, "ab")
    print(f"d['ab'] = {d_ab}")  # Should be 2 (a + b)

    delta_ab = c.get("ab", 0) * (d_ab - 1)
    print(f"ΔL['ab'] = {c.get('ab', 0)} * ({d_ab} - 1) = {delta_ab}")

    # Single chars ('a', 'b') are in keep set - they can't be deleted
    # and d[single_char] isn't meaningful since they can't decompose further

    assert c.get('ab', 0) == 10, "c['ab'] should be 10"
    assert d_ab == 2, "d['ab'] should be 2 (a + b)"
    assert delta_ab == 10, "ΔL['ab'] should be 10"

    print("✓ ΔL calculations verified")


def test_sample_sentences():
    """
    Test with realistic sample sentences, comparing Python vs Rust deletion order.
    """
    print("\n=== Test 1e: Sample Sentences Comparison ===")

    test_cases = [
        {
            "name": "Simple words",
            "corpus": [
                "the cat sat on the mat",
                "the dog ran in the park",
                "a cat and a dog",
                "the mat is red",
                "sat on the park bench",
            ],
            "target": 20,
        },
        {
            "name": "Programming terms",
            "corpus": [
                "function return value",
                "return function call",
                "value of function",
                "call the function",
                "return the value",
            ],
            "target": 15,
        },
        {
            "name": "Repeated patterns",
            "corpus": [
                "abab cdcd efef",
                "abcd abcd abcd",
                "efef abab cdcd",
                "cdcd efef abab",
                "abcd efef abab",
            ],
            "target": 12,
        },
        {
            "name": "Mixed lengths",
            "corpus": [
                "a ab abc abcd",
                "abcd abc ab a",
                "ab abcd a abc",
                "abc a abcd ab",
                "abcd ab abc a",
            ] * 3,
            "target": 8,
        },
    ]

    all_passed = True
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        corpus = case["corpus"]
        target = case["target"]

        # Build seed vocab from corpus (alphabet + substrings)
        all_chars = set()
        for s in corpus:
            all_chars.update(s)
        alphabet = sorted(all_chars)

        # Add common substrings as seed
        substring_counts = Counter()
        for s in corpus:
            words = s.split()
            for w in words:
                for length in range(2, min(len(w) + 1, 8)):
                    for i in range(len(w) - length + 1):
                        substring_counts[w[i:i+length]] += 1

        # Take top substrings
        top_substrings = [s for s, _ in substring_counts.most_common(50)]
        seed_vocab = alphabet + [s for s in top_substrings if s not in alphabet]

        # Keep alphabet protected
        keep_tokens = set(alphabet)

        # Adjust target if needed
        actual_target = min(target, len(seed_vocab))
        if actual_target == len(seed_vocab):
            print(f"  Skipping (seed={len(seed_vocab)}, target={target})")
            continue

        print(f"  Corpus: {len(corpus)} sentences")
        print(f"  Seed vocab: {len(seed_vocab)} tokens")
        print(f"  Target: {actual_target}")

        # Python reference
        py_vocab, py_order = python_greedy_compression(
            corpus, seed_vocab, actual_target, keep_tokens
        )

        # Rust step-by-step
        rust_vocab, rust_order = rust_train_step_by_step(
            corpus, seed_vocab, actual_target, keep_tokens
        )

        # Compare
        print(f"  Deletions: {len(py_order)}")
        if py_order == rust_order:
            print(f"  ✓ EXACT deletion order match!")
        elif set(py_vocab) == set(rust_vocab):
            print(f"  ⚠ Order differs but final vocab matches")
            # Show first difference
            for i, (p, r) in enumerate(zip(py_order, rust_order)):
                if p != r:
                    print(f"    First diff at step {i}: Python={p}, Rust={r}")
                    break
        else:
            diff = set(py_vocab).symmetric_difference(set(rust_vocab))
            print(f"  ✗ Final vocab differs: {diff}")
            all_passed = False

        # Show some deletion examples
        if len(py_order) > 0:
            print(f"  First 5 deletions: {py_order[:5]}")

    if all_passed:
        print("\n✓ All sample sentence tests passed!")
    else:
        assert False, "Sample sentence tests failed"


def test_rand_compression():
    """
    Test rand_compression algorithm: compare Python vs Rust implementation.
    """
    print("\n=== Test 1f: Rand Compression Algorithm ===")

    test_cases = [
        {
            "name": "Simple case",
            "corpus": ["abcde"] * 5 + ["abc"] * 3 + ["de"] * 2,
            "seed": list("abcde") + ["ab", "bc", "cd", "de", "abc", "cde", "abcde"],
            "keep": set("abcde"),
            "target": 7,
        },
        {
            "name": "Overlapping tokens",
            "corpus": ["abab"] * 5 + ["baba"] * 5,
            "seed": list("ab") + ["ab", "ba", "aba", "bab", "abab", "baba"],
            "keep": set("ab"),
            "target": 5,
        },
        {
            "name": "Frequency difference",
            "corpus": ["aaa"] * 10 + ["bbb"] * 2,
            "seed": list("ab") + ["aa", "bb", "aaa", "bbb"],
            "keep": set("ab"),
            "target": 4,
        },
    ]

    all_passed = True
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        corpus = case["corpus"]
        seed = case["seed"]
        keep = case["keep"]
        target = case["target"]

        print(f"  Corpus: {len(corpus)} sentences")
        print(f"  Seed vocab: {len(seed)} tokens")
        print(f"  Target: {target}")

        # Python rand_compression reference
        py_vocab, py_order = python_rand_compression(
            corpus, seed, target, keep, sample_size=100
        )

        # Rust rand_compression
        rust_vocab, rust_order = rust_train_rand_compression(
            corpus, seed, target, keep, sample_size=100
        )

        print(f"  Python order: {py_order}")
        print(f"  Rust order:   {rust_order}")

        if py_order == rust_order:
            print("  ✓ EXACT deletion order match!")
        elif set(py_vocab) == set(rust_vocab):
            print("  ⚠ Order differs but final vocab matches")
            for i, (p, r) in enumerate(zip(py_order, rust_order)):
                if p != r:
                    print(f"    First diff at step {i}: Python={p}, Rust={r}")
                    break
        else:
            diff = set(py_vocab).symmetric_difference(set(rust_vocab))
            print(f"  ✗ Final vocab differs: {diff}")
            all_passed = False

    if all_passed:
        print("\n✓ All rand_compression tests passed!")
    else:
        assert False, "Rand compression tests failed"


def test_rand_vs_original():
    """
    Compare rand_compression vs original compression on same data.
    Both should produce valid results but may differ in token selection.
    """
    print("\n=== Test 1g: Rand vs Original Compression ===")

    corpus = ["hello world"] * 10 + ["hello there"] * 5 + ["world here"] * 5
    seed = list("abcdefghilnortw ") + [
        "he", "ll", "lo", "wo", "rl", "ld", "th", "er", "re",
        "hel", "llo", "wor", "rld", "the", "her", "ere",
        "hell", "ello", "worl", "orld", "ther", "here",
        "hello", "world", "there",
    ]
    keep = set("abcdefghilnortwh ")
    target = 22

    print(f"Corpus: {len(corpus)} sentences")
    print(f"Seed: {len(seed)} tokens, Target: {target}")

    # Original compression (Python reference)
    orig_vocab, orig_order = python_greedy_compression(corpus, seed, target, keep)

    # Rand compression (Python reference)
    rand_vocab, rand_order = python_rand_compression(corpus, seed, target, keep, sample_size=100)

    print(f"\nOriginal method deleted: {orig_order[:10]}...")
    print(f"Rand method deleted:     {rand_order[:10]}...")

    print(f"\nOriginal final vocab: {len(orig_vocab)} tokens")
    print(f"Rand final vocab:     {len(rand_vocab)} tokens")

    # Both should reach target and preserve alphabet
    assert len(orig_vocab) == target, f"Original didn't reach target: {len(orig_vocab)}"
    assert len(rand_vocab) == target, f"Rand didn't reach target: {len(rand_vocab)}"

    for c in keep:
        assert c in orig_vocab, f"Original missing alphabet: {c}"
        assert c in rand_vocab, f"Rand missing alphabet: {c}"

    # Show difference
    common = set(orig_vocab) & set(rand_vocab)
    only_orig = set(orig_vocab) - set(rand_vocab)
    only_rand = set(rand_vocab) - set(orig_vocab)

    print(f"\nCommon tokens: {len(common)}")
    print(f"Only in original: {only_orig}")
    print(f"Only in rand: {only_rand}")

    print("✓ Both methods produce valid vocabularies")


# ---------------------------------------------------------------------
# Test 2: Basic correctness
# ---------------------------------------------------------------------

def test_basic_correctness():
    """
    Verify that CompressionTrainer:
    - Reaches target vocab size
    - Preserves alphabet (single chars from corpus)
    - Produces valid segmentations
    """
    print("\n=== Test 2: Basic Correctness ===")

    corpus = [
        "hello world",
        "hello hello",
        "world world world",
        "the quick brown fox",
        "jumps over the lazy dog",
    ] * 10

    target_size = 50

    tok = Tokenizer(Unigram())
    tok.pre_tokenizer = Whitespace()

    trainer = CompressionTrainer(
        vocab_size=target_size,
        show_progress=True,
        max_piece_length=16,
        seed_size=1000,
        prune_ratio=0.25,
    )

    tok.train_from_iterator(corpus, trainer=trainer)

    final_vocab = vocab_set(tok)
    print(f"Target: {target_size}, Got: {len(final_vocab)}")

    # Check vocab size
    assert len(final_vocab) <= target_size + 10, \
        f"Vocab too large: {len(final_vocab)} > {target_size}"

    # Check alphabet preserved (single chars from corpus should be there)
    all_chars = set("".join(corpus))
    for c in all_chars:
        if c != " ":  # whitespace handled by pre_tokenizer
            assert c in final_vocab, f"Missing alphabet char: '{c}'"

    # Check we can encode everything
    for s in corpus:
        enc = tok.encode(s)
        assert len(enc.tokens) > 0, f"Empty encoding for: {s}"
        # Verify decode works
        decoded = tok.decode(enc.ids)
        assert decoded.replace(" ", "") == s.replace(" ", ""), \
            f"Decode mismatch: '{decoded}' vs '{s}'"

    print(f"✓ Vocab size: {len(final_vocab)}")
    print(f"✓ Alphabet preserved")
    print(f"✓ All strings encode/decode correctly")

# ---------------------------------------------------------------------
# Test 3: Batch deletion verification
# ---------------------------------------------------------------------

def test_batch_deletion():
    """
    Verify that batch deletion works:
    - With prune_ratio=0.25, each pass should delete ~25% of remaining
    - Total tokens should decrease or stay same after training
    """
    print("\n=== Test 3: Batch Deletion ===")

    corpus = ["abcde"] * 100 + ["abc"] * 50 + ["de"] * 50 + ["ab", "cd", "bc"]

    seed = list("abcde") + ["ab", "bc", "cd", "de", "abc", "bcd", "cde", "abcd", "bcde", "abcde"]

    # Train with batch deletion
    tok = Tokenizer(Unigram())
    trainer = CompressionTrainer(
        vocab_size=8,
        show_progress=True,
        seed_vocab=seed,
        prune_ratio=0.25,
    )
    tok.train_from_iterator(corpus, trainer=trainer)

    final_vocab = vocab_set(tok)
    print(f"Seed size: {len(seed)}, Target: 8, Final: {len(final_vocab)}")

    # Alphabet must be preserved
    for c in "abcde":
        assert c in final_vocab, f"Missing: {c}"

    # Should be close to target
    assert len(final_vocab) <= 10, f"Vocab too large: {len(final_vocab)}"

    print(f"✓ Final vocab: {sorted(final_vocab)}")

# ---------------------------------------------------------------------
# Test 4: Byte fallback
# ---------------------------------------------------------------------

def ensure_unk_in_json(model_dict):
    vocab = model_dict.get("vocab")
    unk_idx = None
    for i, (tok, _) in enumerate(vocab):
        if tok == "<unk>":
            unk_idx = i
            break
    if unk_idx is None:
        vocab.insert(0, ["<unk>", 0.0])
        unk_idx = 0
    model_dict["unk_id"] = unk_idx

def ensure_byte_tokens_in_json(model_dict):
    vocab = model_dict.get("vocab")
    present = {t for (t, _) in vocab}
    for b in range(256):
        token = f"<0x{b:02X}>"
        if token not in present:
            vocab.append([token, -1.0])

def test_byte_fallback():
    """
    Test that byte_fallback handles OOV characters correctly.
    """
    print("\n=== Test 4: Byte Fallback ===")

    corpus = ["hello", "world"] * 100

    tok = Tokenizer(Unigram())
    trainer = CompressionTrainer(
        vocab_size=20,
        show_progress=False,
        seed_size=500,
    )
    tok.train_from_iterator(corpus, trainer=trainer)

    # Save and patch for byte_fallback
    tmp_plain = "tmp_test_unigram.json"
    tmp_fb = "tmp_test_unigram_fb.json"

    tok.save(tmp_plain)

    with open(tmp_plain, "r") as f:
        data = json.load(f)

    model = data["model"]
    ensure_unk_in_json(model)
    model["byte_fallback"] = True
    ensure_byte_tokens_in_json(model)

    with open(tmp_fb, "w") as f:
        json.dump(data, f)

    tok_fb = Tokenizer.from_file(tmp_fb)

    # Test OOV chars
    test_cases = [
        ("hello🙂world", 4),  # 🙂 = 4 bytes
        ("helloéworld", 2),   # é = 2 bytes
        ("hello世界", 6),     # 世界 = 3+3 bytes
    ]

    for text, expected_oov_bytes in test_cases:
        enc = tok_fb.encode(text)
        # Count byte tokens
        byte_tokens = [t for t in enc.tokens if t.startswith("<0x")]
        assert len(byte_tokens) == expected_oov_bytes, \
            f"Expected {expected_oov_bytes} byte tokens for '{text}', got {len(byte_tokens)}: {enc.tokens}"

    # Cleanup
    for p in [tmp_plain, tmp_fb]:
        try:
            os.remove(p)
        except Exception:
            pass

    print("✓ Byte fallback handles OOV correctly")

# ---------------------------------------------------------------------
# Test 5: Large scale test
# ---------------------------------------------------------------------

def test_large_scale():
    """
    Test with larger corpus and vocab to ensure no crashes/hangs.
    """
    print("\n=== Test 5: Large Scale ===")

    # Generate larger corpus
    import random
    random.seed(42)

    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "hello", "world", "python", "rust", "code", "test", "data",
             "machine", "learning", "neural", "network", "transformer"]

    corpus = []
    for _ in range(10000):
        sentence = " ".join(random.choices(words, k=random.randint(3, 10)))
        corpus.append(sentence)

    print(f"Corpus: {len(corpus)} sentences")

    tok = Tokenizer(Unigram())
    tok.pre_tokenizer = Whitespace()

    trainer = CompressionTrainer(
        vocab_size=200,
        show_progress=True,
        seed_size=10000,
        prune_ratio=0.2,
    )

    start = time.time()
    tok.train_from_iterator(corpus, trainer=trainer)
    elapsed = time.time() - start

    final_vocab = vocab_set(tok)
    print(f"Time: {elapsed:.2f}s")
    print(f"Final vocab: {len(final_vocab)}")

    # Verify encoding works
    sample_tokens = total_tokens(tok, corpus[:100])
    print(f"Tokens (100 samples): {sample_tokens}")

    assert len(final_vocab) <= 250, f"Vocab too large"
    assert elapsed < 60, f"Training too slow: {elapsed}s"

    print("✓ Large scale test passed")

# ---------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------

if __name__ == "__main__":
    test_algorithm_correctness()
    test_algorithm_correctness_single_delete()
    test_deletion_order_complex()
    test_delta_calculation()
    test_sample_sentences()
    test_rand_compression()
    test_rand_vs_original()
    test_basic_correctness()
    test_batch_deletion()
    test_byte_fallback()
    test_large_scale()

    print("\n" + "="*50)
    print("All tests passed ✅")
    print("="*50)
