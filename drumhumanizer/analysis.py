"""Interpretability & error-analysis helpers (Plan E).

Pure, torch-free functions over a note DataFrame that carries the ground-truth
``velocity`` and a model prediction column (default ``pred``). Kept separate from
plotting/model code so the aggregation logic is unit-testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ON_BEAT_TOL = 0.1                 # |phase_beat - nearest pulse| below this = "on-beat"


def add_metrical_cols(df: pd.DataFrame, beats_per_bar: int = 4) -> pd.DataFrame:
    """Derive beat position from the phase features (E-GMD test data is all 4/4).

    Adds: ``beat_in_bar`` (0-indexed), ``is_downbeat`` (beat 1), ``is_backbeat``
    (beats 2 & 4), ``metrical_class`` ("on-beat"/"off-beat" by proximity of
    ``phase_beat`` to an integer pulse).
    """
    out = df.copy()
    beat = np.floor(out["phase_bar"].to_numpy() * beats_per_bar).astype(int)
    beat = np.clip(beat, 0, beats_per_bar - 1)
    out["beat_in_bar"] = beat
    out["is_downbeat"] = beat == 0
    out["is_backbeat"] = np.isin(beat, [1, 3])            # 0-indexed beats 2 & 4
    pb = out["phase_beat"].to_numpy()
    dist = np.minimum(pb, 1.0 - pb)                       # distance to nearest pulse
    out["metrical_class"] = np.where(dist < ON_BEAT_TOL, "on-beat", "off-beat")
    return out


def residual_table(df: pd.DataFrame, by, pred_col: str = "pred") -> pd.DataFrame:
    """Per-group error decomposition. Returns a frame indexed by the group key with
    n, true_mean, pred_mean, true_std, pred_std, bias (pred-true), and mae."""
    tmp = df.assign(_ae=(df[pred_col] - df["velocity"]).abs())
    g = tmp.groupby(by)
    res = pd.DataFrame({
        "n": g.size(),
        "true_mean": g["velocity"].mean(),
        "pred_mean": g[pred_col].mean(),
        "true_std": g["velocity"].std(),
        "pred_std": g[pred_col].std(),
        "mae": g["_ae"].mean(),
    })
    res["bias"] = res["pred_mean"] - res["true_mean"]
    return res


def dynamic_level_table(df: pd.DataFrame, pred_col: str = "pred", n_bins: int = 8,
                        lo: float = 0.0, hi: float = 127.0) -> pd.DataFrame:
    """Bin notes by TRUE velocity and report mean prediction + bias per bin.

    Exposes regression-to-the-mean: a point estimator predicts soft notes too loud
    (bias > 0) and loud notes too soft (bias < 0)."""
    edges = np.linspace(lo, hi, n_bins + 1)
    b = np.clip(np.digitize(df["velocity"].to_numpy(), edges[1:-1]), 0, n_bins - 1)
    tmp = df.assign(_bin=b, _ae=(df[pred_col] - df["velocity"]).abs())
    g = tmp.groupby("_bin")
    res = pd.DataFrame({
        "n": g.size(),
        "true_mean": g["velocity"].mean(),
        "pred_mean": g[pred_col].mean(),
        "mae": g["_ae"].mean(),
    })
    res["bias"] = res["pred_mean"] - res["true_mean"]
    res["bin_lo"] = edges[:-1][res.index]
    res["bin_hi"] = edges[1:][res.index]
    return res


def embedding_2d(weight, n_components: int = 2) -> np.ndarray:
    """PCA-project embedding rows to 2D via mean-centered SVD (deterministic).

    weight: array [n, d]. Returns [n, n_components]."""
    W = np.asarray(weight, dtype=float)
    Wc = W - W.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
    return Wc @ Vt[:n_components].T
