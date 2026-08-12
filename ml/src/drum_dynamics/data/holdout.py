"""Drummer-disjoint re-partitioning for the robustness experiment (Plan D).

The official E-GMD split shares every drummer across train/val/test, so its
metrics measure *in-distribution* performance. To test generalization to an
unseen player, this re-partitions the already-featurized rows so that the
held-out drummers form the test split and never appear in train/val.

Torch-free (pandas only) — safe to import without triggering the lazy torch
imports in ``drum_dynamics/__init__``.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def build_drummer_holdout(frames: Iterable[pd.DataFrame], holdout_drummers,
                          val_frac: float = 0.1, seed: int = 42):
    """Re-partition featurized frames into a drummer-disjoint split.

    Parameters
    ----------
    frames : iterable of DataFrames
        The original (train/val/test) featurized frames to pool. Each must
        carry ``file_id``, ``drummer`` and ``genre`` columns.
    holdout_drummers : set-like
        Drummer ids that become the *test* split. Must not be the unique
        source of any genre, or that genre would vanish from training.
    val_frac : float
        Fraction of the remaining *files* (per genre) to route to validation.
    seed : int
        Seed for the deterministic per-genre file shuffle.

    Returns
    -------
    (train_df, val_df, test_df) : tuple of DataFrames
        A partition of the pooled rows. Splitting is by ``file_id`` so no file
        leaks across splits; per-genre stratification keeps every genre present
        in train (a genre's single file is never sent to val).
    """
    df = pd.concat(list(frames), ignore_index=True)
    hold = set(holdout_drummers)

    test = df[df["drummer"].isin(hold)].copy()
    rest = df[~df["drummer"].isin(hold)].copy()

    files = rest[["file_id", "genre"]].drop_duplicates("file_id")
    rng = np.random.RandomState(seed)
    val_ids: set = set()
    for _, sub in files.groupby("genre"):
        ids = sub["file_id"].to_numpy()
        ids = ids[rng.permutation(len(ids))]           # deterministic shuffle
        # keep >=1 file per genre in train: only split when >1 file exists
        n_val = int(round(len(ids) * val_frac)) if len(ids) > 1 else 0
        n_val = min(n_val, len(ids) - 1)               # never empty a genre from train
        val_ids.update(ids[:n_val].tolist())

    is_val = rest["file_id"].isin(val_ids)
    val = rest[is_val].copy()
    train = rest[~is_val].copy()
    return train, val, test
