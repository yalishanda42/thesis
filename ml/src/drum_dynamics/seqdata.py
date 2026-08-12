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


def build_split_tensors(df, genre_vocab, bpm_mean, bpm_std, max_len=MAX_LEN):
    """Per-note rows -> padded windowed token tensors (design §5). No leakage."""
    df = df.reset_index(drop=True)
    row_all = np.arange(len(df))
    voice_all = df["voice"].map(_VOICE_IDX).to_numpy()
    genre_all = df["genre"].astype(str).map(lambda g: genre_vocab.get(g, 0)).to_numpy()
    bpm_z_all = ((df["bpm"].to_numpy() - bpm_mean) / bpm_std)
    onset_all = df["onset_sec"].to_numpy(dtype=float)
    bpm_all = df["bpm"].to_numpy(dtype=float)
    vel_all = df["velocity"].to_numpy(dtype=float)
    sin_beat = df["sin_beat"].to_numpy(dtype=float)
    cos_beat = df["cos_beat"].to_numpy(dtype=float)
    sin_bar = df["sin_bar"].to_numpy(dtype=float)
    cos_bar = df["cos_bar"].to_numpy(dtype=float)

    windows = []   # list of (positions[np.int64], log_time_to_prev[np.float32], same_onset[np.float32])
    for _, idx in df.groupby("file_id", sort=False).groups.items():
        pos = np.asarray(idx, dtype=np.int64)
        order = np.lexsort((voice_all[pos], onset_all[pos]))   # sort by onset, then voice
        pos = pos[order]
        onset_beats = onset_all[pos] / (60.0 / bpm_all[pos])
        delta = np.empty(len(pos), dtype=float)
        delta[0] = _DELTA_CLIP_BEATS
        if len(pos) > 1:
            delta[1:] = onset_beats[1:] - onset_beats[:-1]
        ltp = np.log1p(np.clip(delta, 0.0, _DELTA_CLIP_BEATS)).astype(np.float32)
        same_onset = np.zeros(len(pos), dtype=np.float32)
        if len(pos) > 1:
            same_onset[1:] = (delta[1:] == 0.0).astype(np.float32)
        for s in range(0, len(pos), max_len):
            sl = slice(s, s + max_len)
            windows.append((pos[sl], ltp[sl], same_onset[sl]))

    n = len(windows)
    voice_t = np.zeros((n, max_len), dtype=np.int64)
    genre_t = np.zeros((n, max_len), dtype=np.int64)
    num_t = np.zeros((n, max_len, len(NUMERIC_FEATURES)), dtype=np.float32)
    target_t = np.zeros((n, max_len), dtype=np.float32)
    pad_t = np.ones((n, max_len), dtype=bool)
    row_t = np.full((n, max_len), -1, dtype=np.int64)

    for i, (pos, ltp, so) in enumerate(windows):
        L = len(pos)
        voice_t[i, :L] = voice_all[pos]
        genre_t[i, :L] = genre_all[pos]
        num_t[i, :L, 0] = sin_beat[pos]
        num_t[i, :L, 1] = cos_beat[pos]
        num_t[i, :L, 2] = sin_bar[pos]
        num_t[i, :L, 3] = cos_bar[pos]
        num_t[i, :L, 4] = bpm_z_all[pos]
        num_t[i, :L, 5] = ltp
        num_t[i, :L, 6] = so
        target_t[i, :L] = vel_all[pos]
        pad_t[i, :L] = False
        row_t[i, :L] = row_all[pos]

    return {
        "voice_idx": torch.from_numpy(voice_t),
        "genre_idx": torch.from_numpy(genre_t),
        "num_feats": torch.from_numpy(num_t),
        "target": torch.from_numpy(target_t),
        "pad_mask": torch.from_numpy(pad_t),
        "row_idx": torch.from_numpy(row_t),
    }


def scatter_predictions(row_idx, preds, pad_mask, n_rows) -> np.ndarray:
    """Place each non-pad token's prediction at its original row position."""
    out = np.zeros(n_rows, dtype=np.float64)
    ri = row_idx.reshape(-1).cpu().numpy()
    pr = preds.reshape(-1).detach().cpu().numpy().astype(np.float64)
    keep = ri >= 0
    out[ri[keep]] = pr[keep]
    return out
