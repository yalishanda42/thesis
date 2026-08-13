"""Persist the MDN inference-time artifacts (vocab + bpm stats) not in the ckpt."""
from __future__ import annotations

from ..data.seqdata import build_genre_vocab, bpm_stats


def build_mdn_meta(train_df) -> dict:
    """Reproduce the train-time genre vocab + bpm normalization for inference."""
    bpm_mean, bpm_std = bpm_stats(train_df)
    return {
        "genre_vocab": build_genre_vocab(train_df),
        "bpm_mean": bpm_mean,
        "bpm_std": bpm_std,
        "head": "mdn",
    }
