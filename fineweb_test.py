# -*- coding: utf-8 -*-
"""
Train three byte-level BPE tokenizers (count, exact_ll, approx_ll) to 50k vocab
on a line-per-sentence text file. Save models & videos:
  - delta_ll_hist.mp4 (ΔLL per-merge windows)
  - snapshot_scores.mp4 (ONE histogram per snapshot; one frame per snapshot)

Prereqs:
  - Your local `tokenizers` wheel/extension is installed (with telemetry).
  - Python deps: numpy, matplotlib, imageio  (auto-installed if missing).

Outputs (per run):
  results/<run_name>/
    <run_name>-tokenizer.json
    <run_name>-model-files.json    # lists vocab/merges saved by model.save()
    <run_name>-telemetry.json      # {"ll": [...], "merge_trace": [...], "score_snapshots": [...]}
    delta_ll_frames/               # PNG frames for ΔLL video
    delta_ll_hist.mp4
    snapshot_frames/               # PNG frames for snapshot-score video
    snapshot_scores.mp4
    snapshot_plots/                # a few candidate-score snapshot histograms (static)
"""
import os
import json
import math
import time
import shutil
from typing import Iterable, List, Optional

# ---------- CONFIG ----------
DATASET_PATH = "/Users/ahmetcanyavuz/Developer/tokenizers/fineweb_data/fineweb_en_sentences.txt"
RESULTS_DIR  = "./results_100k_fixed"
VOCAB_SIZE   = 100_000
MIN_FREQ     = 2

# Telemetry / visualization
SNAPSHOT_EVERY        = 1000   # take candidate-score snapshots every N accepted merges
SNAPSHOT_SAMPLE_SIZE  = 100_000 # sample up to this many pairs per snapshot (from trainer)
TRACK_LL              = True

# ΔLL video (windowed, not snapshots)
DLL_WINDOW            = 1000   # merges per histogram frame
VIDEO_FPS             = 4

# Snapshot-score video options
SNAPSHOT_BINS         = 100
SNAPSHOT_QCLIP        = (0.01, 0.99)  # (kept for reference; not used when plotting raw values)
MAX_SNAPSHOT_FRAMES   = None          # None = use all snapshots; or put an int to cap
# ----------------------------

# Optional: install small deps for plotting/video if missing
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import imageio.v3 as iio
except ImportError as e:
    import subprocess, sys
    pkgs = ["numpy", "matplotlib", "imageio"]
    print(f"⚠️  Missing {e.name}. Installing: {', '.join(pkgs)} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U"] + pkgs)
    import numpy as np
    import matplotlib.pyplot as plt
    import imageio.v3 as iio

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.decoders import ByteLevel as ByteLevelDec


def line_iterator(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


def ensure_clean_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_tokenizer(tokenizer: Tokenizer, out_dir: str, name: str):
    ensure_clean_dir(out_dir)
    tok_json = os.path.join(out_dir, f"{name}-tokenizer.json")
    tokenizer.save(tok_json)
    # Also save model-native files (vocab.json / merges.txt)
    try:
        saved = tokenizer.model.save(out_dir, name)
        with open(os.path.join(out_dir, f"{name}-model-files.json"), "w") as fout:
            json.dump([str(p) for p in saved], fout, indent=2)
    except Exception as e:
        print(f"ℹ️  Model.save skipped for {name}: {e}")


def make_delta_ll_video(telemetry: dict, out_dir: str, run_name: str,
                        window: int = 1000, fps: int = 4):
    merges: List[dict] = telemetry.get("merge_trace", [])
    if not merges:
        print(f"⚠️  [{run_name}] No merge_trace; skipping ΔLL video.")
        return

    dll: List[float] = []
    # Our telemetry puts exact delta under "delta_ll"; if absent, skip
    for ev in merges:
        val = ev.get("delta_ll", None)
        if val is None:
            continue
        dll.append(float(val))

    if not dll:
        print(f"⚠️  [{run_name}] No delta_ll values; skipping ΔLL video.")
        return

    frames_dir = os.path.join(out_dir, "delta_ll_frames")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    arr = np.array(dll, dtype=np.float64)
    # RAW RANGE (no clipping)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        hi = lo + 1e-6
    bins = np.linspace(lo, hi, 50)

    frames_paths: List[str] = []
    total = len(arr)
    n_frames = max(1, math.ceil(total / window))

    for k in range(n_frames):
        start = k * window
        end = min((k + 1) * window, total)
        chunk = arr[start:end]

        fig = plt.figure(figsize=(8, 5), dpi=120)
        plt.hist(chunk, bins=bins)
        plt.xlabel("ΔLL (exact)")
        plt.ylabel("count")
        plt.title(f"{run_name}: ΔLL distribution — merges {start}..{end-1} (window={window})")
        plt.tight_layout()
        frame_path = os.path.join(frames_dir, f"frame_{k:05d}.png")
        fig.savefig(frame_path)
        plt.close(fig)
        frames_paths.append(frame_path)

    mp4_path = os.path.join(out_dir, "delta_ll_hist.mp4")
    try:
        frames = [iio.imread(p) for p in frames_paths]
        iio.imwrite(mp4_path, frames, fps=fps, codec="libx264", quality=8)
        print(f"🎬 [{run_name}] ΔLL histogram video: {mp4_path} [{len(frames_paths)} frames]")
    except Exception as e:
        gif_path = os.path.join(out_dir, "delta_ll_hist.gif")
        iio.imwrite(gif_path, [iio.imread(p) for p in frames_paths], duration=1.0 / fps)
        print(f"🎬 [{run_name}] ΔLL histogram GIF: {gif_path} [{len(frames_paths)} frames]. Reason: {e}")


def make_snapshot_score_video(telemetry: dict, out_dir: str, run_name: str,
                              fps: int = 4, bins: int = 60,
                              q_clip=(0.01, 0.99), max_frames: Optional[int] = None):
    """Build a video where EACH FRAME is ONE SNAPSHOT histogram of candidate scores."""
    snaps: List[dict] = telemetry.get("score_snapshots", [])
    if not snaps:
        print(f"ℹ️  [{run_name}] No score_snapshots; skipping snapshot-score video.")
        return

    # Sort by step, keep only snapshots that have items
    snaps = [s for s in sorted(snaps, key=lambda s: s.get("step", 0)) if s.get("items")]
    if not snaps:
        print(f"ℹ️  [{run_name}] Empty snapshots; skipping snapshot-score video.")
        return

    if max_frames is not None:
        snaps = snaps[:max_frames]

    # Gather all scores across snapshots to fix a global bin range (stable axes)
    all_scores = []
    per_snapshot_scores = []
    steps = []
    for s in snaps:
        items = s["items"]
        sc = [float(it.get("score", 0.0)) for it in items]
        if sc:
            arr = np.asarray(sc, dtype=np.float64)
            per_snapshot_scores.append(arr)
            steps.append(int(s.get("step", 0)))
            all_scores.append(arr)

    if not per_snapshot_scores:
        print(f"ℹ️  [{run_name}] No scores in snapshots; skipping video.")
        return

    # RAW RANGE (no clipping)
    all_concat = np.concatenate(all_scores)
    lo, hi = float(all_concat.min()), float(all_concat.max())
    if hi <= lo:
        hi = lo + 1e-6
    bin_edges = np.linspace(lo, hi, bins)

    # Precompute a common y-limit for consistent animation scale
    ymax = 0
    for arr in per_snapshot_scores:
        counts, _ = np.histogram(arr, bins=bin_edges)
        ymax = max(ymax, int(counts.max()))
    if ymax == 0:
        ymax = 1

    frames_dir = os.path.join(out_dir, "snapshot_frames")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    frame_paths: List[str] = []
    for idx, arr in enumerate(per_snapshot_scores):
        step = steps[idx]

        fig = plt.figure(figsize=(8, 5), dpi=120)
        plt.hist(arr, bins=bin_edges)
        plt.xlim(lo, hi)
        plt.ylim(0, ymax * 1.05)
        plt.xlabel("Candidate score")
        plt.ylabel("count")
        plt.title(f"{run_name}: snapshot at step={step} (n={arr.size})")
        plt.tight_layout()
        frame_path = os.path.join(frames_dir, f"frame_{idx:05d}.png")
        fig.savefig(frame_path)
        plt.close(fig)
        frame_paths.append(frame_path)

    out_path = os.path.join(out_dir, "snapshot_scores.mp4")
    try:
        frames = [iio.imread(p) for p in frame_paths]
        iio.imwrite(out_path, frames, fps=fps, codec="libx264", quality=8)
        print(f"🎬 [{run_name}] Snapshot score video: {out_path} [{len(frame_paths)} frames]")
    except Exception as e:
        gif_path = os.path.join(out_dir, "snapshot_scores.gif")
        iio.imwrite(gif_path, [iio.imread(p) for p in frame_paths], duration=1.0 / fps)
        print(f"🎬 [{run_name}] Snapshot score GIF: {gif_path} [{len(frame_paths)} frames]. Reason: {e}")


def make_snapshot_histograms(telemetry: dict, out_dir: str, run_name: str, max_plots: int = 6):
    """Optional: save a handful of static snapshot histograms."""
    snaps: List[dict] = telemetry.get("score_snapshots", [])
    if not snaps:
        print(f"ℹ️  [{run_name}] No score_snapshots; skipping snapshot plots.")
        return

    snaps = [s for s in sorted(snaps, key=lambda s: s.get("step", 0)) if s.get("items")]
    if not snaps:
        print(f"ℹ️  [{run_name}] Empty snapshots; skipping snapshot plots.")
        return

    snaps_dir = os.path.join(out_dir, "snapshot_plots")
    if os.path.exists(snaps_dir):
        shutil.rmtree(snaps_dir)
    os.makedirs(snaps_dir, exist_ok=True)

    idxs = np.linspace(0, len(snaps) - 1, num=min(max_plots, len(snaps)), dtype=int).tolist()
    for i in idxs:
        snap = snaps[i]
        step = snap.get("step", i)
        items = snap.get("items", [])
        scores = np.array([float(it.get("score", 0.0)) for it in items], dtype=np.float64)
        if scores.size == 0:
            continue

        # RAW histogram (per-snapshot)
        fig = plt.figure(figsize=(8, 5), dpi=120)
        plt.hist(scores, bins=SNAPSHOT_BINS)
        plt.xlabel("Candidate score (policy)")
        plt.ylabel("count")
        plt.title(f"{run_name}: candidate-score snapshot at step={step} (n={len(scores)})")
        plt.tight_layout()
        out_path = os.path.join(snaps_dir, f"snapshot_step_{int(step):06d}.png")
        fig.savefig(out_path)
        plt.close(fig)

    print(f"🖼  [{run_name}] Saved {len(idxs)} score snapshot histograms under {snaps_dir}")


def train_bpe_variant(dataset_path: str, out_root: str, run_name: str,
                      score_by: str, stop_by: str = "vocab_size"):
    """
    Train a byte-level BPE variant with telemetry enabled.
    score_by: "count" | "exact_ll" | "approx_ll"
    stop_by : "vocab_size" | "delta_ll_exact" | "delta_ll_approx"
    """
    print(f"\n=== Training {run_name} ===")
    out_dir = os.path.join(out_root, run_name)
    ensure_clean_dir(out_dir)

    model = BPE(byte_fallback=True)  # byte coverage without UNK
    tok = Tokenizer(model)
    tok.pre_tokenizer = ByteLevelPre(add_prefix_space=True)
    tok.decoder = ByteLevelDec()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        show_progress=True,
        score_by=score_by,
        stop_by=stop_by,
        track_ll=TRACK_LL,
        score_snapshot_every=SNAPSHOT_EVERY,
        score_sample_size=SNAPSHOT_SAMPLE_SIZE,
    )

    t0 = time.time()
    tok.train_from_iterator(line_iterator(dataset_path), trainer=trainer)
    t1 = time.time()
    print(f"✅ {run_name} trained in {t1-t0:.1f}s")

    # Save tokenizer + model files
    save_tokenizer(tok, out_dir, run_name)

    # Save telemetry
    telemetry = tok.model.telemetry()
    tel_path = os.path.join(out_dir, f"{run_name}-telemetry.json")
    with open(tel_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"🧪 [{run_name}] Saved telemetry: {tel_path}")

    # ΔLL histogram video (per-merge windows; RAW values)
    make_delta_ll_video(telemetry, out_dir, run_name, window=DLL_WINDOW, fps=VIDEO_FPS)
    # Snapshot-score histogram video (ONE snapshot → ONE frame; RAW values)
    make_snapshot_score_video(
        telemetry, out_dir, run_name,
        fps=VIDEO_FPS, bins=SNAPSHOT_BINS, q_clip=SNAPSHOT_QCLIP, max_frames=MAX_SNAPSHOT_FRAMES
    )
    # Optional: a handful of static snapshot images
    make_snapshot_histograms(telemetry, out_dir, run_name)


def main():
    if not os.path.isfile(DATASET_PATH):
        raise SystemExit(f"❌ File not found: {DATASET_PATH}")

    ensure_clean_dir(RESULTS_DIR)

    # Run 0: classic (count-based) BPE
    train_bpe_variant(
        dataset_path=DATASET_PATH,
        out_root=RESULTS_DIR,
        run_name="bpe_count",
        score_by="count",
        stop_by="vocab_size",
    )

    # Run 1: exact LL scoring
    train_bpe_variant(
        dataset_path=DATASET_PATH,
        out_root=RESULTS_DIR,
        run_name="bpe_exact_ll",
        score_by="exact_ll",
        stop_by="vocab_size",
    )

    # Run 2: approximate LL scoring
    train_bpe_variant(
        dataset_path=DATASET_PATH,
        out_root=RESULTS_DIR,
        run_name="bpe_approx_ll",
        score_by="approx_ll",
        stop_by="vocab_size",
    )

    print("\n✅ All done. Artifacts under ./results/:")
    print("  - results/bpe_count")
    print("  - results/bpe_exact_ll")
    print("  - results/bpe_approx_ll")


if __name__ == "__main__":
    main()