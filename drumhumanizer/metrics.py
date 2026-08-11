"""Velocity-model evaluation metrics (design §7). MSE alone is insufficient."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def _mean_group_corr(df, y_pred, group_cols, fn, min_n):
    y = np.asarray(y_pred, dtype=float)
    vals = []
    for _, idx in df.groupby(group_cols).groups.items():
        pos = df.index.get_indexer(idx)
        t = df["velocity"].to_numpy()[pos]
        p = y[pos]
        if len(t) < min_n or np.std(t) == 0 or np.std(p) == 0:
            continue
        vals.append(fn(t, p))
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(df: pd.DataFrame, y_pred) -> dict:
    df = df.reset_index(drop=True)
    y = np.asarray(y_pred, dtype=float)
    t = df["velocity"].to_numpy(dtype=float)
    per_genre = {g: mae(sub["velocity"], y[df.index.get_indexer(sub.index)])
                 for g, sub in df.groupby("genre")}
    return {
        "mae": mae(t, y),
        "rmse": rmse(t, y),
        "per_track_pearson": _mean_group_corr(df, y, "file_id",
                                              lambda a, b: pearsonr(a, b)[0], 2),
        "per_track_spearman": _mean_group_corr(df, y, "file_id",
                                               lambda a, b: spearmanr(a, b)[0], 2),
        "within_bar_spearman": _mean_group_corr(df, y, ["file_id", "bar_index"],
                                                lambda a, b: spearmanr(a, b)[0], 3),
        "mean_abs_std_diff": float(np.mean([
            abs(np.std(t[df.index.get_indexer(sub.index)]) -
                np.std(y[df.index.get_indexer(sub.index)]))
            for _, sub in df.groupby("file_id")])),
        "global_std_ratio": float(np.std(y) / np.std(t)) if np.std(t) else float("nan"),
        "per_genre_mae": per_genre,
    }
