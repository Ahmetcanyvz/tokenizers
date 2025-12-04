# test_bpe_telemetry.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# --- Tiny toy corpus with repeat patterns to produce obvious merges ---
CORPUS = [
    "lorem ipsum dolor sit amet amet amet",
    "ipsum ipsum lorem lorem dolor",
    "data data science science data",
    "byte pair encoding encoding",
    "hello hello hello world world",
    "banana bandana banana band",
    "the quick brown fox jumps over the lazy dog",
    "tokenizers make tokenization fast and fun",
    "bpe exact approx count scoring options",
    "merge pairs improve likelihood",
] * 20  # repeat to increase frequencies


def train_variant(name: str,
                  score_by: str = "count",
                  vocab_size: int = 400,
                  min_freq: int = 2,
                  track_ll: bool = True,
                  snap_every: int = 10,
                  snap_size: int = 200):
    print(f"\n=== Training {name} ===")
    print(f"  score_by={score_by}, vocab_size={vocab_size}, min_freq={min_freq}, "
          f"track_ll={track_ll}, snapshot_every={snap_every}, sample_size={snap_size}")

    tok = Tokenizer(BPE())                     # fresh BPE model
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        show_progress=True,
        score_by=score_by,
        track_ll=track_ll,
        # telemetry knobs
        score_snapshot_every=snap_every,
        score_sample_size=snap_size,
    )

    # Train directly from iterator
    tok.train_from_iterator(CORPUS, trainer=trainer)

    # Pull telemetry back from the model
    bpe = tok.model  # PyBPE
    tel = bpe.telemetry()

    # Summaries
    ll = tel.get("ll", [])
    merges = tel.get("merge_trace", [])
    snaps = tel.get("score_snapshots", [])

    print(f"  merges accepted: {len(merges)}")
    print(f"  ll points      : {len(ll)} {'(tracked)' if track_ll else '(disabled)'}")
    print(f"  snapshots      : {len(snaps)}")

    # Show last 5 LL values (if tracked)
    if ll:
        tail = ll[-5:] if len(ll) > 5 else ll
        print("  LL tail        :", ", ".join(f"{x:.3f}" for x in tail))

    # Helper to decode pair ids back to tokens
    def decode_pair(pair):
        a, b = pair
        ta = bpe.id_to_token(a) or f"<{a}>"
        tb = bpe.id_to_token(b) or f"<{b}>"
        return ta, tb

    # Show first 5 merge events with decoded tokens
    show = merges[:5]
    if show:
        print("  first 5 merges:")
        for ev in show:
            ta, tb = decode_pair(ev["pair"])
            score = ev.get("score", None)
            dll = ev.get("delta_ll", None)
            extras = []
            if score is not None:
                extras.append(f"score={score:.3f}")
            if dll is not None:
                extras.append(f"ΔLL={dll:.3f}")
            extras_s = ", ".join(extras) if extras else ""
            print(f"    step={ev['step']:>3} pair=({ta!r}, {tb!r}) count={ev['count']}"
                  + (f"  {extras_s}" if extras_s else ""))

    # Show 2 snapshots (early & late) with a tiny sample of items
    if snaps:
        print("  sample snapshots (at most 2 printed):")
        for s in (snaps[:1] + snaps[-1:]):
            step = s["step"]
            items = s.get("items", [])[:5]  # print small head
            print(f"    step {step}: {len(s.get('items', []))} items (showing 5)")
            for it in items:
                ta, tb = decode_pair(it["pair"])
                print(f"      ({ta!r}, {tb!r}) score={it['score']:.3f} count={it['count']}")

    return tok, tel


if __name__ == "__main__":
    # Classic BPE by pair frequency
    tok_count, tel_count = train_variant("bpe_count", score_by="count", track_ll=True)

    # Greedy by exact ΔLL
    tok_exact, tel_exact = train_variant("bpe_exact_ll", score_by="exact_ll", track_ll=True)

    # Greedy by approximate ΔLL
    tok_approx, tel_approx = train_variant("bpe_approx_ll", score_by="approx_ll", track_ll=True)

    print("\nDone.")