"""Section B event-sequence tensors for the velocity Transformer (design §5).

Turns the cached Section A parquet (per-note rows) into padded, windowed token
tensors. STRUCTURAL ONLY — velocity is the target, never a feature (design §1.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .voicemap import CANONICAL_VOICES

_VOICE_IDX = {v: i for i, v in enumerate(CANONICAL_VOICES)}

# fixed tensor-column order for the numeric feature block
NUMERIC_FEATURES = [
    "sin_beat", "cos_beat", "sin_bar", "cos_bar",
    "bpm_z", "log_time_to_prev", "same_onset",
]
MAX_LEN = 512                      # tokens per window
_DELTA_CLIP_BEATS = 8.0            # matches Plan A TIME_DELTA_CLIP_BEATS


def build_genre_vocab(train_df: pd.DataFrame) -> dict:
    """Sorted train genres -> 1..G; index 0 is reserved for <unk>."""
    genres = sorted(train_df["genre"].astype(str).unique())
    return {g: i + 1 for i, g in enumerate(genres)}


def bpm_stats(train_df: pd.DataFrame) -> tuple:
    """Train-split bpm mean/std; std floored at 1.0."""
    mean = float(train_df["bpm"].mean())
    std = float(train_df["bpm"].std())
    return mean, (std if std > 1e-6 else 1.0)
