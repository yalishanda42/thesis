"""Phase 0 exploratory validation — full scan of the E-GMD *train* split.

Runs the note-level analysis the metadata EDA can't (per-pitch velocity
distributions, inter-onset-gap histogram) in parallel over every train file, and
writes the artifacts that feed the final feature spec (design spec §3):

  docs/phase0/aggregates.npz        raw mergeable aggregates (reproducible)
  docs/phase0/pitch_velocity.csv    per-GM-pitch count + velocity stats
  docs/phase0/summary.json          decided hyperparameters + audit numbers
  docs/phase0/fig_pitch_counts.png  pitch usage bar chart
  docs/phase0/fig_velocity_dists.png velocity histograms per used voice
  docs/phase0/fig_gap_hist.png      inter-onset-gap histogram + valley

Usage:
    .venv/bin/python scripts/phase0_analysis.py [--limit N] [--workers K]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from drum_dynamics.midi import drum_name
from drum_dynamics.phase0 import (
    GAP_EDGES,
    GAP_MS_EDGES,
    FileScan,
    gap_bin_centers,
    merge_scans,
    scan_file,
    velocity_stats,
    window_min,
)

#: No valley exists in the near-zero (<~0.05 beat) region — the density decays
#: monotonically — so the simultaneity tolerance cannot be data-derived from a
#: valley (see docs/phase0/results.md). This is a documented fixed hyperparameter:
#: notes within this beat-fraction are treated as one "instant" / chord. Expressed
#: in beats (not ms) so it is tempo-robust; ~0.02 beat ≈ 11 ms at the median 110 bpm
#: and stays safely below a 16th note even at the fastest tempos.
RECOMMENDED_SIMULTANEITY_TOL_BEATS = 0.02

EGMD_BASE = os.path.join("data", "e-gmd", "e-gmd-v1.0.0")
EGMD_CSV = os.path.join(EGMD_BASE, "e-gmd-v1.0.0.csv")
OUT_DIR = os.path.join("docs", "phase0")


def _worker(arg):
    """Top-level so it is picklable by multiprocessing."""
    path, bpm = arg
    return scan_file(path, bpm)


def scan_train_split(df: pd.DataFrame, workers: int) -> tuple[FileScan, int, int]:
    """Parallel-scan every train file; return merged aggregate + (n_files, n_failed)."""
    train = df[df["split"] == "train"].reset_index(drop=True)
    args = [
        (os.path.join(EGMD_BASE, row.midi_filename), float(row.bpm))
        for row in train.itertuples(index=False)
    ]
    n = len(args)
    print(f"scanning {n} train files on {workers} workers ...")
    t0 = time.time()
    scans: list[FileScan] = []
    with Pool(workers) as pool:
        for i, s in enumerate(pool.imap_unordered(_worker, args, chunksize=64), 1):
            scans.append(s)
            if i % 2000 == 0 or i == n:
                rate = i / (time.time() - t0)
                print(f"  {i}/{n}  ({rate:.0f} files/s)")
    n_failed = sum(1 for s in scans if not s.ok)
    return merge_scans(scans), n, n_failed


def per_pitch_table(pv_matrix: np.ndarray) -> pd.DataFrame:
    """One row per GM pitch that actually appears, with velocity stats."""
    rows = []
    total_notes = int(pv_matrix.sum())
    for pitch in range(pv_matrix.shape[0]):
        counts = pv_matrix[pitch]
        n = int(counts.sum())
        if n == 0:
            continue
        s = velocity_stats(counts)
        rows.append(
            {
                "pitch": pitch,
                "gm_name": drum_name(pitch),
                "count": n,
                "share_pct": round(100.0 * n / total_notes, 4),
                "vel_mean": round(s.mean, 2),
                "vel_std": round(s.std, 2),
                "vel_median": s.median,
                "vel_p10": s.p10,
                "vel_p90": s.p90,
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def categorical_audit(df: pd.DataFrame) -> dict:
    """Cardinality / skew of the conditioning categoricals on the train split (§3.3, §5)."""
    train = df[df["split"] == "train"]
    ts = train["time_signature"].value_counts()
    n = len(train)
    return {
        "n_train_files": int(n),
        "style_nunique": int(train["style"].nunique()),
        "style_top_genre_nunique": int(train["style"].str.split("/").str[0].nunique()),
        "style_top10": train["style"].value_counts().head(10).to_dict(),
        "time_signature": ts.to_dict(),
        "time_signature_44_pct": round(100.0 * ts.get("4-4", 0) / n, 3),
        "time_signature_non44_pct": round(100.0 * (n - ts.get("4-4", 0)) / n, 3),
        "beat_type": train["beat_type"].value_counts().to_dict(),
        "bpm": {
            "min": float(train["bpm"].min()),
            "max": float(train["bpm"].max()),
            "mean": round(float(train["bpm"].mean()), 2),
            "std": round(float(train["bpm"].std()), 2),
        },
    }


def _plot_pitch_counts(tbl: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [f"{r.pitch} {r.gm_name}" for r in tbl.itertuples()]
    ax.barh(range(len(tbl)), tbl["count"], color="#4C72B0")
    ax.set_yticks(range(len(tbl)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("note count (log scale)")
    ax.set_title("E-GMD train split — pitch usage")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_velocity_dists(pv_matrix: np.ndarray, tbl: pd.DataFrame, path: str) -> None:
    used = tbl.sort_values("count", ascending=False)
    k = len(used)
    ncol = 4
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.2 * nrow), squeeze=False)
    for i, r in enumerate(used.itertuples()):
        ax = axes[i // ncol][i % ncol]
        ax.bar(np.arange(128), pv_matrix[r.pitch], width=1.0, color="#55A868")
        ax.set_title(f"{r.gm_name}\n(n={r.count:,}, μ={r.vel_mean})", fontsize=8)
        ax.set_xlim(0, 127)
        ax.tick_params(labelsize=6)
    for j in range(k, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Velocity distribution per used voice (E-GMD train)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_gap_hist(gap_counts: np.ndarray, tol: float, subdiv: float, path: str) -> None:
    centers = gap_bin_centers()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, xmax, title in ((axes[0], 0.2, "zoom: 0–0.2 beat"), (axes[1], 1.0, "0–1 beat")):
        mask = centers <= xmax
        ax.bar(centers[mask], gap_counts[: centers.size][mask],
               width=np.diff(GAP_EDGES[: centers.size + 1])[mask] * 0.9, color="#C44E52")
        ax.axvline(tol, color="green", ls="--", lw=1.2,
                   label=f"SIMULTANEITY_TOL = {tol:.3f} beat (fixed)")
        ax.axvline(subdiv, color="black", ls=":", lw=1.2,
                   label=f"subdivision valley ≈ {subdiv:.3f} beat")
        ax.set_xlabel("inter-onset gap (beats)")
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Inter-onset gap histogram (beats): no near-zero valley; "
                 "clear valley before the 16th-note grid", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_gap_hist_ms(gap_ms_counts: np.ndarray, subdiv_ms: float, path: str) -> None:
    centers = gap_bin_centers(GAP_MS_EDGES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, xmax, title in ((axes[0], 50.0, "zoom: 0–50 ms"), (axes[1], 250.0, "0–250 ms")):
        mask = centers <= xmax
        ax.bar(centers[mask], gap_ms_counts[: centers.size][mask],
               width=np.diff(GAP_MS_EDGES[: centers.size + 1])[mask] * 0.9, color="#8172B3")
        ax.axvline(subdiv_ms, color="black", ls=":", lw=1.2,
                   label=f"subdivision valley ≈ {subdiv_ms:.0f} ms @ median bpm")
        ax.set_xlabel("inter-onset gap (ms)")
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Inter-onset gap histogram (ms): near-zero decay has no valley "
                 "(the ~1 ms comb is 480-PPQ tick quantization)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="scan only the first N train files (debug)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(EGMD_CSV)
    if args.limit:
        # keep only the first N train files for a quick smoke run
        train_head = df[df["split"] == "train"].head(args.limit)
        df = pd.concat([df[df["split"] != "train"], train_head])

    merged, n_files, n_failed = scan_train_split(df, args.workers)
    print(f"parsed {n_files - n_failed}/{n_files} files ({n_failed} failed); "
          f"{merged.n_notes:,} notes total")

    tbl = per_pitch_table(merged.pv_matrix)
    # Two boundaries live in the gap distribution (see results.md):
    #  (1) simultaneity: the near-zero (<~0.05 beat) region decays monotonically —
    #      NO valley — so SIMULTANEITY_TOL is a fixed hyperparameter, not derived.
    #  (2) subdivision valley: a clear, broad trough separates the sub-16th ornament
    #      mass from the 16th-note grid. We locate it as the density minimum in the
    #      window between the near-zero peak and the 16th cluster.
    subdiv_valley_beats = window_min(merged.gap_counts, GAP_EDGES, lo=0.05, hi=0.22, smooth=7)
    subdiv_valley_ms = window_min(merged.gap_ms_counts, GAP_MS_EDGES, lo=25.0, hi=95.0, smooth=9)
    cats = categorical_audit(df)

    # --- persist artifacts ---
    np.savez_compressed(
        os.path.join(OUT_DIR, "aggregates.npz"),
        pv_matrix=merged.pv_matrix,
        gap_counts=merged.gap_counts,
        gap_edges=GAP_EDGES,
        gap_ms_counts=merged.gap_ms_counts,
        gap_ms_edges=GAP_MS_EDGES,
        n_notes=merged.n_notes,
    )
    tbl.to_csv(os.path.join(OUT_DIR, "pitch_velocity.csv"), index=False)

    median_bpm = float(df[df["split"] == "train"]["bpm"].median())
    tol = RECOMMENDED_SIMULTANEITY_TOL_BEATS
    summary = {
        "n_train_files": n_files,
        "n_failed": n_failed,
        "n_notes": int(merged.n_notes),
        "n_pitches_used": int(len(tbl)),
        "median_train_bpm": median_bpm,
        "simultaneity_tol": {
            # near-zero region has no valley -> fixed, tempo-robust, documented
            "near_zero_valley_exists": False,
            "recommended_tol_beats": tol,
            "recommended_tol_ms_at_median_bpm": round(tol * 60.0 / median_bpm * 1000.0, 1),
        },
        "subdivision_valley": {
            # data-derived; separates sub-16th ornaments from the 16th-note grid
            "beats": round(subdiv_valley_beats, 4),
            "ms_at_median_bpm": round(subdiv_valley_ms, 1),
            "note": "ornament vs 16th-grid boundary; supports the no-grid design (§2)",
        },
        "categorical_audit": cats,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    _plot_pitch_counts(tbl, os.path.join(OUT_DIR, "fig_pitch_counts.png"))
    _plot_velocity_dists(merged.pv_matrix, tbl, os.path.join(OUT_DIR, "fig_velocity_dists.png"))
    _plot_gap_hist(merged.gap_counts, tol, subdiv_valley_beats,
                   os.path.join(OUT_DIR, "fig_gap_hist.png"))
    _plot_gap_hist_ms(merged.gap_ms_counts, subdiv_valley_ms,
                      os.path.join(OUT_DIR, "fig_gap_hist_ms.png"))

    print("\n=== per-pitch (top rows) ===")
    print(tbl.to_string(index=False))
    print(f"\nSIMULTANEITY_TOL (fixed)   = {tol} beat "
          f"(≈ {summary['simultaneity_tol']['recommended_tol_ms_at_median_bpm']} ms @ median bpm)")
    print(f"subdivision valley         = {subdiv_valley_beats:.4f} beat "
          f"(≈ {subdiv_valley_ms:.0f} ms @ median bpm)")
    print(f"4/4 share on train = {cats['time_signature_44_pct']}%  "
          f"(non-4/4 = {cats['time_signature_non44_pct']}%)")
    print(f"\nartifacts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
