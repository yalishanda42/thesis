# Plan B — Event-Sequence Transformer & Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a non-autoregressive event-sequence Transformer (design §5/§6 model 2) that predicts every drum note's velocity in a single parallel pass from structural features only, and evaluate it on the E-GMD test split against Plan A's baselines and LightGBM using the shared metrics harness.

**Architecture:** Reuse the cached Section A parquet from Plan A as the source of truth. A new `drumhumanizer.seqdata` module turns per-note rows into padded, windowed token tensors (Section B feature subset: drum-part + genre embeddings, sin/cos metrical phase, z-scored bpm, log delta-time-to-prev, same-onset flag) while tracking each token's original parquet row so predictions can be scattered back for evaluation. A new `drumhumanizer.model` holds the `VelocityTransformer` (input projection → sinusoidal positional encoding → `nn.TransformerEncoder` → `Linear(d,1)` head). One script trains it, evaluates with `drumhumanizer.metrics.evaluate`, and writes results/figures under `docs/plan_b/`.

**Tech Stack:** Python 3.12, PyTorch 2.13 (Apple **MPS** backend — no CUDA on this machine; falls back to CPU), `pandas`/`numpy`, `pyarrow` (read parquet), `matplotlib`, and the existing `drumhumanizer` package (`voicemap`, `metrics`).

## Global Constraints

- **No velocity leakage (design §1.1):** velocity is the *target only*. No note's velocity may be a feature for any other note. Section B features are structure/timing only.
- **Non-autoregressive (design §1.1):** the model sees the full structural window and emits all velocities in one pass. No causal mask, no ordering hacks. Padding is masked out of both attention and loss.
- **Section B feature set (design §5):** per token = drum-part embedding, genre embedding, `sin/cos(2π·phase_beat)`, `sin/cos(2π·phase_bar)`, z-scored `bpm`, `log1p` delta-time-to-previous-event (in beats), and a `same_onset_as_previous` flag. **Hand-engineered window features are dropped** (attention learns context): no `log_same_voice_*`, no `log_time_to_next`, no simultaneity multi-hot, no `simult_count`, no `density_1beat`. **Time-signature is dropped** (Phase 0: 98.78% 4/4).
- **Fit statistics on train only:** genre vocabulary, `bpm` mean/std — all computed on the `train` split and applied unchanged to `validation`/`test`.
- **Simultaneous ordering (design §5):** within a file, tokens are ordered by `(onset_sec, voice_index)` (voice index is the pitch-order proxy — raw pitch is not in the parquet); the `same_onset_as_previous` flag marks tokens sharing the previous token's onset.
- **Reproducibility:** seed everything with `42` (`torch.manual_seed`, `numpy`). Model + scaler config is deterministic.
- **Device:** use `mps` if available, else `cpu`. Tests run on CPU.
- **Inputs:** `data/processed/egmd_tabular_{train,validation,test}.parquet` (built by Plan A Task 5). Baseline reference: `docs/plan_a/metrics.json`.
- **Run commands** use the repo venv: `.venv/bin/python`, `.venv/bin/python -m pytest`.

---

## File Structure

**Create:**
- `drumhumanizer/seqdata.py` — Section B tensor pipeline: genre vocab, bpm stats, windowing, padded tensor assembly, prediction scatter-back.
- `drumhumanizer/model.py` — `VelocityTransformer` (`nn.Module`) + sinusoidal positional encoding.
- `scripts/train_transformer.py` — train + evaluate + figures + `metrics.json`.
- `tests/test_seqdata.py`, `tests/test_model.py`.
- `docs/plan_b/` — results (`metrics.json`, figures, `results.md`) written by the script/worker.

**Modify:**
- `drumhumanizer/__init__.py` — export the new public helpers.

---

### Task 1: Sequence vocab & normalization stats

**Files:**
- Create: `drumhumanizer/seqdata.py` (first portion)
- Test: `tests/test_seqdata.py`
- Modify: `drumhumanizer/__init__.py`

**Interfaces:**
- Produces (consumed by Task 2 and the training script):
  - `NUMERIC_FEATURES: list[str]` — the 7 numeric per-token feature names, in fixed tensor-column order: `["sin_beat", "cos_beat", "sin_bar", "cos_bar", "bpm_z", "log_time_to_prev", "same_onset"]`.
  - `MAX_LEN: int = 512` — window length in tokens.
  - `build_genre_vocab(train_df: pandas.DataFrame) -> dict[str, int]` — sorted train genres mapped to `1..G`; index `0` reserved for `<unk>` (unseen genres in val/test).
  - `bpm_stats(train_df: pandas.DataFrame) -> tuple[float, float]` — `(mean, std)`; std floored at `1.0` to avoid divide-by-zero.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seqdata.py
import numpy as np
import pandas as pd

from drumhumanizer.seqdata import (
    NUMERIC_FEATURES,
    MAX_LEN,
    build_genre_vocab,
    bpm_stats,
)


def test_numeric_features_order_and_no_leakage():
    assert NUMERIC_FEATURES == [
        "sin_beat", "cos_beat", "sin_bar", "cos_bar",
        "bpm_z", "log_time_to_prev", "same_onset",
    ]
    assert not [c for c in NUMERIC_FEATURES if "vel" in c.lower()]
    assert MAX_LEN == 512


def test_build_genre_vocab_reserves_zero_for_unk():
    df = pd.DataFrame({"genre": ["rock", "funk", "rock", "jazz"]})
    vocab = build_genre_vocab(df)
    assert 0 not in vocab.values()          # 0 is <unk>
    assert set(vocab) == {"funk", "jazz", "rock"}
    assert vocab["funk"] == 1 and vocab["jazz"] == 2 and vocab["rock"] == 3  # sorted


def test_bpm_stats_floors_std():
    df = pd.DataFrame({"bpm": [120.0, 120.0, 120.0]})   # zero variance
    mean, std = bpm_stats(df)
    assert mean == 120.0
    assert std == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_seqdata.py -q`
Expected: FAIL (module `drumhumanizer.seqdata` not found).

- [ ] **Step 3: Write the implementation**

```python
# drumhumanizer/seqdata.py
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
```

- [ ] **Step 4: Export from the package**

In `drumhumanizer/__init__.py`, add after the `.metrics` import:

```python
from .seqdata import NUMERIC_FEATURES, MAX_LEN, build_genre_vocab, bpm_stats
```

and add `"NUMERIC_FEATURES"`, `"MAX_LEN"`, `"build_genre_vocab"`, `"bpm_stats"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_seqdata.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add drumhumanizer/seqdata.py tests/test_seqdata.py drumhumanizer/__init__.py
git commit -m "feat: add sequence genre vocab and bpm normalization stats"
```

---

### Task 2: Windowed tensor builder + prediction scatter

**Files:**
- Modify: `drumhumanizer/seqdata.py` (append)
- Modify: `drumhumanizer/__init__.py`
- Test: `tests/test_seqdata.py` (append)

**Interfaces:**
- Consumes: `NUMERIC_FEATURES`, `MAX_LEN`, `_VOICE_IDX`, `_DELTA_CLIP_BEATS` from Task 1; the parquet columns `file_id, onset_sec, velocity, voice, genre, sin_beat, cos_beat, sin_bar, cos_bar, bpm`.
- Produces:
  - `build_split_tensors(df, genre_vocab, bpm_mean, bpm_std, max_len=MAX_LEN) -> dict[str, torch.Tensor]` with keys and shapes (`W` = number of windows):
    - `voice_idx`: `LongTensor [W, max_len]`
    - `genre_idx`: `LongTensor [W, max_len]`
    - `num_feats`: `FloatTensor [W, max_len, len(NUMERIC_FEATURES)]`
    - `target`: `FloatTensor [W, max_len]` (raw velocity 0–127)
    - `pad_mask`: `BoolTensor [W, max_len]` (`True` = padding)
    - `row_idx`: `LongTensor [W, max_len]` (original 0-based row position in `df`; `-1` for padding)
  - `scatter_predictions(row_idx, preds, pad_mask, n_rows) -> np.ndarray` — flatten and place each non-pad token's prediction at its `row_idx`; returns a `float64` array of length `n_rows`.
- Delta-time and `same_onset` are **recomputed inside the window from `onset_sec`/`bpm`** after the `(onset_sec, voice_index)` sort, so they are consistent with the emitted token order (design §5). First token of each file gets `log1p(_DELTA_CLIP_BEATS)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seqdata.py  (append)
import torch

from drumhumanizer.seqdata import build_split_tensors, scatter_predictions


def _toy_df():
    # file "a": 3 notes, two simultaneous at t=0 (kick+snare), one at t=0.5
    # file "b": 2 notes at t=0, t=1
    return pd.DataFrame({
        "file_id": ["a", "a", "a", "b", "b"],
        "onset_sec": [0.0, 0.0, 0.5, 0.0, 1.0],
        "velocity": [100, 40, 70, 88, 55],
        "voice": ["kick", "snare", "closed-hh", "kick", "snare"],
        "genre": ["funk", "funk", "funk", "rock", "rock"],
        "sin_beat": [0.0, 0.0, 1.0, 0.0, 0.0],
        "cos_beat": [1.0, 1.0, 0.0, 1.0, 1.0],
        "sin_bar": [0.0, 0.0, 0.5, 0.0, 0.5],
        "cos_bar": [1.0, 1.0, 0.5, 1.0, 0.5],
        "bpm": [120.0, 120.0, 120.0, 120.0, 120.0],
    })


def test_build_split_tensors_shapes_and_padding():
    df = _toy_df()
    vocab = build_genre_vocab(df)
    t = build_split_tensors(df, vocab, 120.0, 1.0, max_len=4)
    # 2 files -> 2 windows (each file <= max_len)
    assert t["voice_idx"].shape == (2, 4)
    assert t["num_feats"].shape == (2, 4, len(NUMERIC_FEATURES))
    # file "a" has 3 real tokens -> 1 pad; file "b" has 2 -> 2 pads
    assert t["pad_mask"].sum().item() == 1 + 2


def test_build_split_tensors_same_onset_flag():
    df = _toy_df()
    vocab = build_genre_vocab(df)
    t = build_split_tensors(df, vocab, 120.0, 1.0, max_len=4)
    so_col = NUMERIC_FEATURES.index("same_onset")
    # find file "a" window (the one with 3 real tokens)
    real_counts = (~t["pad_mask"]).sum(dim=1)
    wa = int((real_counts == 3).nonzero()[0])
    same_onset = t["num_feats"][wa, :3, so_col]
    # ordered by (onset, voice_idx): kick(0), snare(0), hh(0.5)
    # token0 first -> 0 ; token1 same onset as token0 -> 1 ; token2 new onset -> 0
    assert same_onset.tolist() == [0.0, 1.0, 0.0]


def test_scatter_predictions_round_trips_targets():
    df = _toy_df()
    vocab = build_genre_vocab(df)
    t = build_split_tensors(df, vocab, 120.0, 1.0, max_len=4)
    # feeding the targets through the scatter must reconstruct df velocity exactly
    out = scatter_predictions(t["row_idx"], t["target"], t["pad_mask"], len(df))
    assert np.allclose(out, df["velocity"].to_numpy())


def test_build_split_tensors_no_velocity_in_features():
    df = _toy_df()
    vocab = build_genre_vocab(df)
    t = build_split_tensors(df, vocab, 120.0, 1.0, max_len=4)
    # velocity only lives in the target tensor; num_feats never equals it by construction
    assert "target" in t and t["num_feats"].shape[-1] == len(NUMERIC_FEATURES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_seqdata.py -k "build_split or scatter" -q`
Expected: FAIL (`build_split_tensors` not defined).

- [ ] **Step 3: Write the implementation (append to `drumhumanizer/seqdata.py`)**

```python
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
```

- [ ] **Step 4: Export from the package** — add `build_split_tensors` and `scatter_predictions` to the `.seqdata` import line and to `__all__` in `drumhumanizer/__init__.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_seqdata.py -q`
Expected: PASS (all seqdata tests).

- [ ] **Step 6: Commit**

```bash
git add drumhumanizer/seqdata.py tests/test_seqdata.py drumhumanizer/__init__.py
git commit -m "feat: build windowed sequence tensors with row-index scatter-back"
```

---

### Task 3: The VelocityTransformer model

**Files:**
- Create: `drumhumanizer/model.py`
- Test: `tests/test_model.py`
- Modify: `drumhumanizer/__init__.py`

**Interfaces:**
- Consumes: `NUMERIC_FEATURES` (length = numeric input width), `CANONICAL_VOICES` (voice vocab size = 14).
- Produces:
  - `VelocityTransformer(n_genres, n_numeric=len(NUMERIC_FEATURES), n_voices=14, d_model=128, n_heads=8, n_layers=4, dim_ff=256, dropout=0.1, voice_emb=8, genre_emb=16, max_len=512)` — `nn.Module`.
  - `forward(voice_idx, genre_idx, num_feats, pad_mask) -> FloatTensor [B, L]` — per-token predicted velocity (raw scale). `pad_mask` (`True`=pad) is passed to attention as `src_key_padding_mask`, so real-token outputs are invariant to the number of padding tokens.
  - `n_genres` is `len(genre_vocab) + 1` (the `<unk>` slot at index 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import torch

from drumhumanizer.model import VelocityTransformer
from drumhumanizer.seqdata import NUMERIC_FEATURES


def _tiny_model():
    torch.manual_seed(42)
    return VelocityTransformer(n_genres=5, d_model=32, n_heads=4, n_layers=2,
                               dim_ff=64, max_len=16).eval()


def test_forward_output_shape():
    m = _tiny_model()
    B, L = 2, 6
    voice = torch.randint(0, 14, (B, L))
    genre = torch.randint(0, 5, (B, L))
    num = torch.randn(B, L, len(NUMERIC_FEATURES))
    pad = torch.zeros(B, L, dtype=torch.bool)
    y = m(voice, genre, num, pad)
    assert y.shape == (B, L)


def test_padding_invariance():
    # Real-token predictions must not change when extra padding is appended.
    m = _tiny_model()
    L = 3
    voice = torch.randint(0, 14, (1, L))
    genre = torch.randint(0, 5, (1, L))
    num = torch.randn(1, L, len(NUMERIC_FEATURES))
    pad = torch.zeros(1, L, dtype=torch.bool)

    pad_extra = torch.cat([voice, torch.zeros(1, 2, dtype=torch.long)], dim=1)
    genre_extra = torch.cat([genre, torch.zeros(1, 2, dtype=torch.long)], dim=1)
    num_extra = torch.cat([num, torch.zeros(1, 2, len(NUMERIC_FEATURES))], dim=1)
    mask_extra = torch.tensor([[False, False, False, True, True]])

    with torch.no_grad():
        y1 = m(voice, genre, num, pad)
        y2 = m(pad_extra, genre_extra, num_extra, mask_extra)
    assert torch.allclose(y1[0, :L], y2[0, :L], atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_model.py -q`
Expected: FAIL (module `drumhumanizer.model` not found).

- [ ] **Step 3: Write the implementation**

```python
# drumhumanizer/model.py
"""Non-autoregressive event-sequence Transformer for velocity (design §5/§6)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .seqdata import NUMERIC_FEATURES
from .voicemap import CANONICAL_VOICES


def _sinusoidal_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe                                   # [max_len, d_model]


class VelocityTransformer(nn.Module):
    def __init__(self, n_genres, n_numeric=len(NUMERIC_FEATURES), n_voices=len(CANONICAL_VOICES),
                 d_model=128, n_heads=8, n_layers=4, dim_ff=256, dropout=0.1,
                 voice_emb=8, genre_emb=16, max_len=512):
        super().__init__()
        self.voice_emb = nn.Embedding(n_voices, voice_emb)
        self.genre_emb = nn.Embedding(n_genres, genre_emb)
        self.input_proj = nn.Linear(voice_emb + genre_emb + n_numeric, d_model)
        self.register_buffer("pos_enc", _sinusoidal_pos_enc(max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, voice_idx, genre_idx, num_feats, pad_mask):
        v = self.voice_emb(voice_idx)                     # [B, L, voice_emb]
        g = self.genre_emb(genre_idx)                     # [B, L, genre_emb]
        x = torch.cat([v, g, num_feats], dim=-1)          # [B, L, in]
        x = self.input_proj(x)                            # [B, L, d]
        x = x + self.pos_enc[: x.size(1)].unsqueeze(0)    # add positional encoding
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(x).squeeze(-1)                   # [B, L]
```

- [ ] **Step 4: Export from the package** — add `from .model import VelocityTransformer` and `"VelocityTransformer"` to `__all__` in `drumhumanizer/__init__.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_model.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add drumhumanizer/model.py tests/test_model.py drumhumanizer/__init__.py
git commit -m "feat: add non-autoregressive VelocityTransformer model"
```

---

### Task 4: Train, evaluate, and compare against Plan A

**Files:**
- Create: `scripts/train_transformer.py`
- Create (output): `docs/plan_b/metrics.json`, `docs/plan_b/fig_training_curve.png`, `docs/plan_b/fig_pred_vs_true.png`, `docs/plan_b/fig_per_genre.png`, `docs/plan_b/results.md`

**Interfaces:**
- Consumes: the parquet from Plan A Task 5; `build_genre_vocab`, `bpm_stats`, `build_split_tensors`, `scatter_predictions`, `VelocityTransformer`, `drumhumanizer.metrics.evaluate`; `docs/plan_a/metrics.json` for the baseline comparison.
- Produces: a printed comparison table (transformer vs Plan A's global_mean / lookup_table / lightgbm) and `docs/plan_b/metrics.json` with keys `transformer` (an `evaluate()` dict), `transformer_best_epoch`, `transformer_val_mae_curve`, and `baselines` (the copied Plan A dicts for convenience).

- [ ] **Step 1: Write the script**

```python
# scripts/train_transformer.py
"""Train the event-sequence Transformer on the Section B features and evaluate
on the test split against Plan A's baselines.

Usage: .venv/bin/python scripts/train_transformer.py [--epochs N] [--batch-size B]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
import torch                      # noqa: E402
from torch.utils.data import DataLoader, TensorDataset   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drumhumanizer.metrics import evaluate                                       # noqa: E402
from drumhumanizer.model import VelocityTransformer                              # noqa: E402
from drumhumanizer.seqdata import (                                              # noqa: E402
    build_genre_vocab, bpm_stats, build_split_tensors, scatter_predictions,
)

PROC = os.path.join("data", "processed")
OUT = os.path.join("docs", "plan_b")
PLAN_A = os.path.join("docs", "plan_a", "metrics.json")
SEED = 42
KEYS = ["voice_idx", "genre_idx", "num_feats", "target", "pad_mask", "row_idx"]


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load(split):
    return pd.read_parquet(os.path.join(PROC, f"egmd_tabular_{split}.parquet"))


def _loader(tensors, batch_size, shuffle):
    ds = TensorDataset(*[tensors[k] for k in KEYS])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _masked_mae(pred, target, pad_mask):
    m = ~pad_mask
    return (pred[m] - target[m]).abs().mean()


def _predict(model, loader, device, n_rows):
    model.eval()
    preds = np.zeros(n_rows, dtype=np.float64)
    with torch.no_grad():
        for voice, genre, num, target, pad, row in loader:
            y = model(voice.to(device), genre.to(device), num.to(device), pad.to(device))
            preds += scatter_predictions(row, y.cpu(), pad, n_rows)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = _device()
    print(f"device: {device}")

    train, val, test = _load("train"), _load("validation"), _load("test")
    genre_vocab = build_genre_vocab(train)
    bpm_mean, bpm_std = bpm_stats(train)
    print(f"genres: {len(genre_vocab)}  bpm mean/std: {bpm_mean:.1f}/{bpm_std:.1f}")

    t0 = time.time()
    tr_t = build_split_tensors(train, genre_vocab, bpm_mean, bpm_std)
    va_t = build_split_tensors(val, genre_vocab, bpm_mean, bpm_std)
    te_t = build_split_tensors(test, genre_vocab, bpm_mean, bpm_std)
    print(f"windows train/val/test: {tr_t['target'].shape[0]}/"
          f"{va_t['target'].shape[0]}/{te_t['target'].shape[0]}  "
          f"({time.time()-t0:.0f}s to build)")

    tr_loader = _loader(tr_t, args.batch_size, shuffle=True)
    va_loader = _loader(va_t, args.batch_size, shuffle=False)
    te_loader = _loader(te_t, args.batch_size, shuffle=False)

    model = VelocityTransformer(n_genres=len(genre_vocab) + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_val, best_epoch, bad, curve = float("inf"), -1, 0, []
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        te0 = time.time()
        for voice, genre, num, target, pad, row in tr_loader:
            opt.zero_grad()
            y = model(voice.to(device), genre.to(device), num.to(device), pad.to(device))
            loss = _masked_mae(y, target.to(device), pad.to(device))
            loss.backward()
            opt.step()
        # validation MAE
        model.eval()
        with torch.no_grad():
            num_sum, den = 0.0, 0
            for voice, genre, num, target, pad, row in va_loader:
                y = model(voice.to(device), genre.to(device), num.to(device), pad.to(device))
                m = ~pad.to(device)
                num_sum += (y[m] - target.to(device)[m]).abs().sum().item()
                den += int(m.sum().item())
            val_mae = num_sum / den
        curve.append(val_mae)
        print(f"epoch {epoch:2d}  val_mae {val_mae:.4f}  ({time.time()-te0:.0f}s)")
        if val_mae < best_val - 1e-4:
            best_val, best_epoch, bad = val_mae, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # evaluate on test with the shared harness
    pred = _predict(model, te_loader, device, len(test))
    results = {
        "transformer": evaluate(test, pred),
        "transformer_best_epoch": best_epoch,
        "transformer_val_mae_curve": curve,
    }
    if os.path.exists(PLAN_A):
        with open(PLAN_A) as fh:
            a = json.load(fh)
        results["baselines"] = {k: a[k] for k in ("global_mean", "lookup_table", "lightgbm") if k in a}

    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # comparison table
    rows = []
    if "baselines" in results:
        for name in ("global_mean", "lookup_table", "lightgbm"):
            if name in results["baselines"]:
                rows.append((name, results["baselines"][name]))
    rows.append(("transformer", results["transformer"]))
    print(f"\n{'model':14} {'MAE':>7} {'RMSE':>7} {'trk_r':>7} {'wbar_rho':>9} {'std_ratio':>9}")
    for name, m in rows:
        print(f"{name:14} {m['mae']:7.3f} {m['rmse']:7.3f} {m['per_track_pearson']:7.3f} "
              f"{m['within_bar_spearman']:9.3f} {m['global_std_ratio']:9.3f}")

    # figures
    plt.figure(figsize=(6, 4)); plt.plot(range(1, len(curve) + 1), curve, marker="o")
    plt.xlabel("epoch"); plt.ylabel("val MAE"); plt.title("Transformer validation MAE")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_training_curve.png"), dpi=120); plt.close()

    yte = test["velocity"].to_numpy()
    idx = np.random.RandomState(SEED).choice(len(yte), size=min(20000, len(yte)), replace=False)
    plt.figure(figsize=(5, 5)); plt.hexbin(yte[idx], pred[idx], gridsize=60, cmap="viridis")
    plt.plot([0, 127], [0, 127], "r--", lw=1); plt.xlabel("true"); plt.ylabel("pred")
    plt.title("Transformer: predicted vs true velocity")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_pred_vs_true.png"), dpi=120); plt.close()

    gs = pd.Series(results["transformer"]["per_genre_mae"]).sort_values()
    gs.plot.barh(figsize=(7, 8)); plt.xlabel("MAE"); plt.title("Transformer per-genre MAE")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_per_genre.png"), dpi=120); plt.close()

    print(f"\nwrote results to {OUT}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the whole pipeline on a tiny slice**

Run a 1-epoch sanity check on a subsampled parquet to confirm the pipeline runs end-to-end before the full run:
```bash
.venv/bin/python -c "
import pandas as pd, torch
from drumhumanizer.seqdata import build_genre_vocab, bpm_stats, build_split_tensors, scatter_predictions
from drumhumanizer.model import VelocityTransformer
d = pd.read_parquet('data/processed/egmd_tabular_validation.parquet').head(5000)
v = build_genre_vocab(d); mean,std = bpm_stats(d)
t = build_split_tensors(d, v, mean, std)
m = VelocityTransformer(n_genres=len(v)+1)
y = m(t['voice_idx'][:2], t['genre_idx'][:2], t['num_feats'][:2], t['pad_mask'][:2])
print('forward ok', y.shape)
p = scatter_predictions(t['row_idx'], t['target'], t['pad_mask'], len(d))
print('scatter exact:', bool((abs(p - d['velocity'].to_numpy())<1e-6).all()))
"
```
Expected: `forward ok torch.Size([2, 512])` and `scatter exact: True`.

- [ ] **Step 3: Run full training**

Run: `.venv/bin/python scripts/train_transformer.py`
Expected: prints per-epoch val MAE, a comparison table, and writes `docs/plan_b/metrics.json` + three figures. On MPS this is roughly 1–3 min/epoch; early stopping usually triggers within ~10 epochs.

- [ ] **Step 4: Verify the model clears the bar**

Confirm in the printed table that `transformer` has **lower MAE and RMSE than `lookup_table`** and a **higher `per_track_pearson` and `within_bar_spearman`** than `lookup_table` — beating the "dumb humanizer" is the real bar (design §7). Also report whether it beats `lightgbm` (the honest "does the transformer earn its complexity?" question, design §6). If the transformer does **not** beat the lookup table, stop and report — do not tune blindly beyond the defaults.

- [ ] **Step 5: Write a short results note**

Create `docs/plan_b/results.md` summarizing: the comparison table (copy the printed numbers), the validation-MAE training curve (best epoch), whether the transformer beats the lookup table and how it compares to LightGBM, the per-genre MAE spread vs Plan A's, and a note on the std-ratio (does the transformer flatten dynamics less or more than the trees?). Reference the three figures. Note deferred items: probabilistic heads (Plan C), held-out-drummer check, listening test.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_transformer.py docs/plan_b/
git commit -m "feat: train and evaluate event-sequence velocity Transformer"
```

---

## Self-Review notes

- **Spec §5 coverage:** drum-part embedding + genre embedding (Task 3), sin/cos phase_beat & phase_bar (Task 2 features), z-scored bpm on train stats (Task 1/2), log1p delta-time-to-prev in beats (Task 2), same_onset flag (Task 2), simultaneous ordering by (onset, voice) (Task 2), window features dropped + time-signature dropped (Global Constraints / NUMERIC_FEATURES). ✓
- **Spec §6 model 2:** non-autoregressive Transformer encoder, deterministic `Linear(d,1)` head, MAE loss (Task 3/4). ✓
- **Spec §1.1 no leakage:** velocity only in `target`; asserted in `test_build_split_tensors_no_velocity_in_features` and enforced by `NUMERIC_FEATURES` (Task 1/2). ✓
- **Spec §7 evaluation:** provided splits, reuse of `evaluate` (MAE/RMSE + per-track corr + std match + within-bar ranking + per-genre), compared against both baselines and LightGBM (Task 4). ✓ *Held-out-drummer check and listening test deferred (drummer column carried in parquet).*
- **Non-autoregressive / padding:** enforced by `src_key_padding_mask`; asserted in `test_padding_invariance` (Task 3). ✓
- **Train-only statistics:** genre vocab + bpm stats fit on train, `<unk>` slot for unseen genres (Task 1). ✓
- **Placeholder scan:** no TBD/TODO; all steps carry runnable code. ✓
- **Type consistency:** `NUMERIC_FEATURES` (len 7) drives both `num_feats` last dim (Task 2) and model `n_numeric` (Task 3); `n_genres = len(genre_vocab)+1` consistent between Task 3 interface and Task 4 construction; tensor dict keys match `KEYS` in the script. ✓
```
