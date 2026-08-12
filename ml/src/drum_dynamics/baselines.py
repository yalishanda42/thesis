"""Reference "dumb humanizer" baselines the learned model must beat (design §7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import N_PHASE_BINS


def _phase_bin(phase_beat: pd.Series) -> pd.Series:
    return np.minimum((phase_beat * N_PHASE_BINS).astype(int), N_PHASE_BINS - 1)


class GlobalMeanBaseline:
    def fit(self, df: pd.DataFrame):
        self.mean_ = float(df["velocity"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mean_, dtype=float)


class LookupTableBaseline:
    """Mean velocity per (voice, genre, phase_bin), backing off to coarser keys."""

    def fit(self, df: pd.DataFrame):
        d = df.copy()
        d["phase_bin"] = _phase_bin(d["phase_beat"])
        self.global_ = float(d["velocity"].mean())
        self.by_voice_ = d.groupby("voice")["velocity"].mean().to_dict()
        self.by_vg_ = d.groupby(["voice", "genre"])["velocity"].mean().to_dict()
        self.by_vgp_ = d.groupby(["voice", "genre", "phase_bin"])["velocity"].mean().to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pb = _phase_bin(df["phase_beat"]).to_numpy()
        v = df["voice"].to_numpy()
        g = df["genre"].to_numpy()
        out = np.empty(len(df), dtype=float)
        for i in range(len(df)):
            out[i] = self.by_vgp_.get((v[i], g[i], pb[i]),
                     self.by_vg_.get((v[i], g[i]),
                     self.by_voice_.get(v[i], self.global_)))
        return out
