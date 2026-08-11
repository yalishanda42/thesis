# Plan A — Data Pipeline & Tabular Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the per-note tabular feature pipeline (Section A) for E-GMD and a LightGBM velocity-prediction baseline that is evaluated against the "dumb humanizer" baselines.

**Architecture:** Reusable, unit-tested feature code lives in the `drumhumanizer` package; two scripts drive it — one builds the cached parquet dataset over all splits, one trains + evaluates the model. All structural features only (no velocity leakage, per design §1.1). Phase 0 decisions are locked in as constants.

**Tech Stack:** Python 3.12, `partitura` (MIDI → note arrays), `pandas`/`numpy`, `lightgbm` (gradient-boosted trees, native categorical + gain importances), `scikit-learn` (metrics), `scipy.stats` (rank correlations), `pyarrow` (parquet), `matplotlib`.

## Global Constraints

- **No velocity leakage (design §1.1):** no note's velocity may be a feature for any other note. Every feature is derived from structure/timing only. The *only* place `velocity` appears is the target column `velocity`.
- **Phase 0 locked decisions** (see `docs/phase0/results.md`):
  - `SIMULTANEITY_TOL = 0.02` beat (fixed).
  - Canonical **14-voice** grouping (Task 2 table); edge hi-hats (22, 26) and electric snare (40) separate; toms merged.
  - Time-signature: **kept** in the tabular model (cheap for trees) even though it will be dropped for the transformer.
- **Splits:** use E-GMD's own `split` column (`train`/`validation`/`test`). Fit everything (baselines, model, categorical levels, any statistics) on **train only**.
- **Reproducibility:** all randomness seeded with `random_state=42`. Processed datasets cached under `data/processed/` (already gitignored).
- **sklearn 1.6+:** use `sklearn.metrics.root_mean_squared_error` (the `squared=` arg is gone).
- **Dataset locations:** MIDI base `data/e-gmd/e-gmd-v1.0.0`, metadata CSV `data/e-gmd/e-gmd-v1.0.0/e-gmd-v1.0.0.csv`.
- **Run commands** use the repo venv: `.venv/bin/python`, `.venv/bin/python -m pytest`.

---

## File Structure

**Create:**
- `drumhumanizer/voicemap.py` — canonical voice grouping (pitch → voice), locked from Phase 0.
- `drumhumanizer/features.py` — Section A feature extraction (metrical + structural + conditioning + target) → per-file DataFrame.
- `drumhumanizer/baselines.py` — `GlobalMeanBaseline`, `LookupTableBaseline` ("dumb humanizer").
- `drumhumanizer/metrics.py` — evaluation metrics (MAE/RMSE, per-track corr, std match, within-bar ranking, per-genre).
- `scripts/build_dataset.py` — parallel feature extraction over all splits → parquet.
- `scripts/train_tabular.py` — fit baselines + LightGBM, evaluate, write results/figures.
- `tests/test_voicemap.py`, `tests/test_features.py`, `tests/test_baselines.py`, `tests/test_metrics.py`.
- `docs/plan_a/` — results (`metrics.json`, figures) written by the training script.

**Modify:**
- `requirements.txt` — add `lightgbm`, `pyarrow`.
- `drumhumanizer/__init__.py` — export the new public helpers.

---

### Task 1: Dependencies

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Produces: importable `lightgbm` and `pyarrow` in the venv (later tasks depend on both).

- [ ] **Step 1: Add dependencies to `requirements.txt`**

Add these two lines under the existing scikit-learn line:

```
lightgbm>=4.0         # gradient-boosted trees: native categorical + gain importances
pyarrow>=15           # parquet I/O for the cached tabular dataset
```

- [ ] **Step 2: Install**

Run: `.venv/bin/pip install "lightgbm>=4.0" "pyarrow>=15"`
Note (macOS): LightGBM needs OpenMP at runtime. If import fails with a `libomp`/`libgomp` error, run `brew install libomp` and reinstall.

- [ ] **Step 3: Verify imports**

Run: `.venv/bin/python -c "import lightgbm, pyarrow; print(lightgbm.__version__, pyarrow.__version__)"`
Expected: two version numbers, no error.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: add lightgbm and pyarrow for Plan A tabular baseline"
```

---

### Task 2: Canonical voice map

**Files:**
- Create: `drumhumanizer/voicemap.py`
- Test: `tests/test_voicemap.py`
- Modify: `drumhumanizer/__init__.py`

**Interfaces:**
- Produces:
  - `CANONICAL_VOICES: list[str]` — the 14 voices, in fixed order (defines multi-hot column order).
  - `PITCH_TO_VOICE: dict[int, str]`
  - `voice_of(pitch: int) -> str` — unknown pitch → `"aux-perc"`.
  - `voice_index(voice: str) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_voicemap.py
from drumhumanizer.voicemap import CANONICAL_VOICES, voice_of, voice_index


def test_canonical_voices_are_14_unique():
    assert len(CANONICAL_VOICES) == 14
    assert len(set(CANONICAL_VOICES)) == 14


def test_voice_of_locked_from_phase0():
    assert voice_of(36) == "kick"
    assert voice_of(38) == "snare"
    assert voice_of(40) == "snare-accent"          # distinct hot articulation
    assert voice_of(22) == "closed-hh-edge"        # Roland edge hit kept separate
    assert voice_of(26) == "open-hh"
    assert voice_of(48) == "tom" and voice_of(43) == "tom"   # toms merged
    assert voice_of(51) == "ride" and voice_of(59) == "ride"


def test_voice_of_unknown_falls_back():
    assert voice_of(3) == "aux-perc"


def test_voice_index_matches_order():
    assert voice_index("kick") == 0
    assert voice_index(CANONICAL_VOICES[-1]) == 13
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_voicemap.py -q`
Expected: FAIL (module `drumhumanizer.voicemap` not found).

- [ ] **Step 3: Write the implementation**

```python
# drumhumanizer/voicemap.py
"""Canonical drum voice grouping, locked from Phase 0 (docs/phase0/results.md).

Merges rare/indistinguishable GM pitches and keeps common, distributionally
distinct articulations separate (Roland hi-hat edge hits 22/26, electric snare 40).
The order of CANONICAL_VOICES is fixed — it defines the simultaneity multi-hot
column order in the feature table.
"""

from __future__ import annotations

CANONICAL_VOICES = [
    "kick",
    "snare",
    "snare-accent",
    "side-stick",
    "closed-hh",
    "closed-hh-edge",
    "pedal-hh",
    "open-hh",
    "ride",
    "ride-bell",
    "tom",
    "crash",
    "aux-cymbal",
    "aux-perc",
]

PITCH_TO_VOICE = {
    36: "kick",
    38: "snare",
    40: "snare-accent",
    37: "side-stick",
    42: "closed-hh",
    22: "closed-hh-edge",
    44: "pedal-hh",
    46: "open-hh", 26: "open-hh",
    51: "ride", 59: "ride",
    53: "ride-bell",
    41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
    49: "crash", 57: "crash",
    52: "aux-cymbal", 55: "aux-cymbal",
    54: "aux-perc", 58: "aux-perc", 39: "aux-perc", 56: "aux-perc",
}

_VOICE_TO_INDEX = {v: i for i, v in enumerate(CANONICAL_VOICES)}


def voice_of(pitch: int) -> str:
    """Canonical voice for a MIDI pitch; unknown pitches fall back to 'aux-perc'."""
    return PITCH_TO_VOICE.get(int(pitch), "aux-perc")


def voice_index(voice: str) -> int:
    return _VOICE_TO_INDEX[voice]
```

- [ ] **Step 4: Export from the package**

In `drumhumanizer/__init__.py`, add after the `.midi` import block:

```python
from .voicemap import CANONICAL_VOICES, PITCH_TO_VOICE, voice_of, voice_index
```

and add `"CANONICAL_VOICES"`, `"PITCH_TO_VOICE"`, `"voice_of"`, `"voice_index"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_voicemap.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add drumhumanizer/voicemap.py tests/test_voicemap.py drumhumanizer/__init__.py
git commit -m "feat: add canonical 14-voice map locked from Phase 0"
```

---

### Task 3: Metrical & derived features

**Files:**
- Create: `drumhumanizer/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces (all consumed by Task 4):
  - Constants: `SIMULTANEITY_TOL_BEATS = 0.02`, `TIME_DELTA_CLIP_BEATS = 8.0`, `N_PHASE_BINS = 16`, `SUBDIVISIONS: dict[str, int]`.
  - `beats_per_bar(time_signature: str) -> int` — `"4-4" -> 4`.
  - `metrical_phase(onset_sec: np.ndarray, bpm: float, bpb: int) -> tuple[np.ndarray, np.ndarray]` returns `(phase_beat, phase_bar)`, each in `[0, 1)`.
  - `swing_ratio(phase_beat: np.ndarray) -> np.ndarray` — 0 straight, ~1 hard shuffle.
  - `nearest_subdivision(phase_beat: np.ndarray) -> np.ndarray` of dtype `object` (strings from `SUBDIVISIONS`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import numpy as np

from drumhumanizer.features import (
    SIMULTANEITY_TOL_BEATS,
    beats_per_bar,
    metrical_phase,
    nearest_subdivision,
    swing_ratio,
)


def test_simultaneity_tol_locked():
    assert SIMULTANEITY_TOL_BEATS == 0.02


def test_beats_per_bar():
    assert beats_per_bar("4-4") == 4
    assert beats_per_bar("3-4") == 3


def test_metrical_phase_120bpm_44():
    # bpm=120 -> beat=0.5s, bar=2.0s. Onsets at 0, 0.5, 0.75, 2.0 s.
    onset = np.array([0.0, 0.5, 0.75, 2.0])
    pb, pbar = metrical_phase(onset, bpm=120, bpb=4)
    assert np.allclose(pb, [0.0, 0.0, 0.5, 0.0])
    assert np.allclose(pbar, [0.0, 0.25, 0.375, 0.0])


def test_swing_ratio_straight_vs_triplet():
    # straight 8th (phase 0.5) -> 0 ; triplet offbeat (2/3) -> ~1 ; onbeat -> 0
    sr = swing_ratio(np.array([0.0, 0.5, 2.0 / 3.0]))
    assert np.isclose(sr[0], 0.0)
    assert np.isclose(sr[1], 0.0)
    assert np.isclose(sr[2], 1.0, atol=0.02)


def test_nearest_subdivision_picks_closest_grid():
    # 0.25 is exactly a 16th; 1/3 is an 8th-triplet
    got = nearest_subdivision(np.array([0.25, 1.0 / 3.0]))
    assert got[0] == "16th"
    assert got[1] == "8th-triplet"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_features.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Write the implementation**

```python
# drumhumanizer/features.py  (Task 3 portion — Task 4 appends build_note_features)
"""Section A tabular features for the drum-velocity model (design §4).

STRUCTURAL ONLY — no note's velocity is ever used as a feature (design §1.1).
"""

from __future__ import annotations

import numpy as np

SIMULTANEITY_TOL_BEATS = 0.02      # Phase 0: fixed (no near-zero valley)
TIME_DELTA_CLIP_BEATS = 8.0        # clip inter-onset deltas before log1p
N_PHASE_BINS = 16                  # phase_beat bins for the lookup-table baseline

# candidate subdivision grids: name -> divisions per beat
SUBDIVISIONS = {
    "8th": 2,
    "16th": 4,
    "32nd": 8,
    "8th-triplet": 3,
    "quintuplet": 5,
}


def beats_per_bar(time_signature: str) -> int:
    """Beats per bar from an E-GMD time-signature string like '4-4' -> 4."""
    return int(str(time_signature).split("-")[0])


def metrical_phase(onset_sec: np.ndarray, bpm: float, bpb: int):
    """Continuous metrical phase within the beat and within the bar, each in [0, 1)."""
    onset_sec = np.asarray(onset_sec, dtype=float)
    beat_dur = 60.0 / float(bpm)
    bar_dur = beat_dur * bpb
    phase_beat = np.mod(onset_sec, beat_dur) / beat_dur
    phase_bar = np.mod(onset_sec, bar_dur) / bar_dur
    return phase_beat, phase_bar


def swing_ratio(phase_beat: np.ndarray) -> np.ndarray:
    """How far an offbeat is pushed toward the triplet position.

    0 at the straight 8th (phase 0.5), 1 at the 8th-note-triplet (phase 2/3).
    Defined only in the offbeat region [0.4, 0.8]; 0 elsewhere (onbeats etc.).
    """
    phase_beat = np.asarray(phase_beat, dtype=float)
    out = np.zeros_like(phase_beat)
    region = (phase_beat >= 0.4) & (phase_beat <= 0.8)
    out[region] = (phase_beat[region] - 0.5) / (2.0 / 3.0 - 0.5)
    return out


def nearest_subdivision(phase_beat: np.ndarray) -> np.ndarray:
    """For each onset, the candidate grid whose nearest gridline it is closest to."""
    phase_beat = np.asarray(phase_beat, dtype=float)
    names = list(SUBDIVISIONS)
    # distance to nearest gridline for each grid (phase is circular on [0,1))
    dists = np.empty((len(names), phase_beat.size))
    for i, name in enumerate(names):
        d = SUBDIVISIONS[name]
        scaled = phase_beat * d
        dists[i] = np.abs(scaled - np.round(scaled)) / d
    return np.array(names, dtype=object)[np.argmin(dists, axis=0)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_features.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add drumhumanizer/features.py tests/test_features.py
git commit -m "feat: add metrical phase, swing ratio, nearest-subdivision features"
```

---

### Task 4: Structural context + per-file assembler

**Files:**
- Modify: `drumhumanizer/features.py` (append)
- Modify: `drumhumanizer/__init__.py`
- Test: `tests/test_features.py` (append)

**Interfaces:**
- Consumes: everything from Task 3, `drumhumanizer.voicemap.voice_of/CANONICAL_VOICES`, `drumhumanizer.midi.load_note_array`.
- Produces: `build_note_features(note_array: np.ndarray, meta: Mapping) -> pandas.DataFrame` — one row per note. `meta` must have keys `id`, `drummer`, `split`, `bpm`, `time_signature`, `style`, `beat_type`. Output columns (exact):
  - keys: `file_id, drummer, split, onset_sec, bar_index`
  - target: `velocity`
  - categorical: `voice, genre, style, time_signature, beat_type, nearest_subdiv`
  - metrical numeric: `phase_beat, phase_bar, sin_beat, cos_beat, sin_bar, cos_bar, swing_ratio`
  - context numeric: `log_time_to_prev, log_time_to_next, log_same_voice_prev, log_same_voice_next, simult_count, density_1beat`
  - conditioning numeric: `bpm`
  - multi-hot: `simult_<voice>` for each voice in `CANONICAL_VOICES` (14 columns, 0/1)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py  (append)
import os
import pandas as pd
from drumhumanizer.features import build_note_features
from drumhumanizer.midi import load_note_array
from drumhumanizer.voicemap import CANONICAL_VOICES

EGMD_BASE = os.path.join("data", "e-gmd", "e-gmd-v1.0.0")


def _synthetic_note_array():
    # kick(36) & snare(38) simultaneous at t=0, hat(42) at t=0.5s; bpm 120 -> beat 0.5s
    import numpy as np
    dt = np.dtype([("onset_sec", float), ("duration_sec", float),
                   ("onset_tick", int), ("duration_tick", int),
                   ("pitch", int), ("velocity", int),
                   ("track", int), ("channel", int), ("id", object)])
    rows = [(0.0, 0.1, 0, 48, 36, 100, 0, 9, "n0"),
            (0.0, 0.1, 0, 48, 38, 40, 0, 9, "n1"),
            (0.5, 0.1, 240, 48, 42, 70, 0, 9, "n2")]
    return np.array(rows, dtype=dt)


def _meta(**kw):
    base = dict(id="drummerX/s/1", drummer="drummerX", split="train",
                bpm=120, time_signature="4-4", style="funk/groove1", beat_type="beat")
    base.update(kw)
    return base


def test_build_note_features_columns_and_no_leakage():
    df = build_note_features(_synthetic_note_array(), _meta())
    assert len(df) == 3
    # velocity is the only velocity-derived column
    assert "velocity" in df.columns
    assert not [c for c in df.columns if "vel" in c.lower() and c != "velocity"]
    for c in ["voice", "genre", "phase_beat", "sin_beat", "swing_ratio",
              "log_time_to_prev", "simult_count", "density_1beat", "bar_index"]:
        assert c in df.columns
    for v in CANONICAL_VOICES:
        assert f"simult_{v}" in df.columns


def test_build_note_features_simultaneity_and_voice():
    df = build_note_features(_synthetic_note_array(), _meta())
    kick = df[df["voice"] == "kick"].iloc[0]
    # kick fires simultaneously with snare -> count 2, both multi-hots set, self set
    assert kick["simult_count"] == 2
    assert kick["simult_kick"] == 1 and kick["simult_snare"] == 1
    assert kick["simult_closed-hh"] == 0
    assert kick["genre"] == "funk"


def test_build_note_features_on_real_file():
    df = pd.read_csv(os.path.join(EGMD_BASE, "e-gmd-v1.0.0.csv"))
    row = df[df["beat_type"] == "beat"].iloc[0]
    na = load_note_array(os.path.join(EGMD_BASE, row["midi_filename"]))
    out = build_note_features(na, row.to_dict())
    assert len(out) == len(na)
    assert out["velocity"].between(0, 127).all()
    assert out["phase_beat"].between(0, 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_features.py -k build_note_features -q`
Expected: FAIL (`build_note_features` not defined).

- [ ] **Step 3: Write the implementation (append to `drumhumanizer/features.py`)**

```python
import pandas as pd

from .voicemap import CANONICAL_VOICES, voice_of


def _log_clip_beats(delta_beats: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(delta_beats, 0.0, TIME_DELTA_CLIP_BEATS))


def build_note_features(note_array, meta) -> pd.DataFrame:
    """One structural feature row per note (design §4). No velocity leakage."""
    order = np.argsort(note_array["onset_sec"], kind="stable")
    na = note_array[order]
    onset = na["onset_sec"].astype(float)
    pitch = na["pitch"].astype(int)
    n = len(na)

    bpm = float(meta["bpm"])
    beat_dur = 60.0 / bpm
    bpb = beats_per_bar(meta["time_signature"])
    onset_beats = onset / beat_dur

    phase_beat, phase_bar = metrical_phase(onset, bpm, bpb)
    voices = np.array([voice_of(p) for p in pitch], dtype=object)

    # global consecutive deltas (any voice), in beats
    to_prev = np.full(n, TIME_DELTA_CLIP_BEATS)
    to_next = np.full(n, TIME_DELTA_CLIP_BEATS)
    if n > 1:
        d = np.diff(onset_beats)
        to_prev[1:] = d
        to_next[:-1] = d

    # same-voice consecutive deltas, in beats
    sv_prev = np.full(n, TIME_DELTA_CLIP_BEATS)
    sv_next = np.full(n, TIME_DELTA_CLIP_BEATS)
    for v in set(voices):
        idx = np.where(voices == v)[0]
        if idx.size > 1:
            dv = np.diff(onset_beats[idx])
            sv_prev[idx[1:]] = dv
            sv_next[idx[:-1]] = dv

    # simultaneity multi-hot + count, and ±1-beat density (vectorized via searchsorted)
    lo = np.searchsorted(onset_beats, onset_beats - SIMULTANEITY_TOL_BEATS, side="left")
    hi = np.searchsorted(onset_beats, onset_beats + SIMULTANEITY_TOL_BEATS, side="right")
    dlo = np.searchsorted(onset_beats, onset_beats - 1.0, side="left")
    dhi = np.searchsorted(onset_beats, onset_beats + 1.0, side="right")
    simult_count = hi - lo
    density = dhi - dlo
    multihot = {f"simult_{v}": np.zeros(n, dtype=np.int8) for v in CANONICAL_VOICES}
    for i in range(n):
        for j in range(lo[i], hi[i]):
            multihot[f"simult_{voices[j]}"][i] = 1

    style = str(meta["style"])
    out = pd.DataFrame({
        "file_id": str(meta["id"]),
        "drummer": str(meta["drummer"]),
        "split": str(meta["split"]),
        "onset_sec": onset,
        "bar_index": np.floor(onset / (beat_dur * bpb)).astype(int),
        "velocity": na["velocity"].astype(int),
        "voice": voices,
        "genre": style.split("/")[0],
        "style": style,
        "time_signature": str(meta["time_signature"]),
        "beat_type": str(meta["beat_type"]),
        "nearest_subdiv": nearest_subdivision(phase_beat),
        "phase_beat": phase_beat,
        "phase_bar": phase_bar,
        "sin_beat": np.sin(2 * np.pi * phase_beat),
        "cos_beat": np.cos(2 * np.pi * phase_beat),
        "sin_bar": np.sin(2 * np.pi * phase_bar),
        "cos_bar": np.cos(2 * np.pi * phase_bar),
        "swing_ratio": swing_ratio(phase_beat),
        "log_time_to_prev": _log_clip_beats(to_prev),
        "log_time_to_next": _log_clip_beats(to_next),
        "log_same_voice_prev": _log_clip_beats(sv_prev),
        "log_same_voice_next": _log_clip_beats(sv_next),
        "simult_count": simult_count.astype(int),
        "density_1beat": density.astype(int),
        "bpm": bpm,
    })
    for name, col in multihot.items():
        out[name] = col
    return out
```

- [ ] **Step 4: Export from the package**

In `drumhumanizer/__init__.py` add `from .features import build_note_features` and add `"build_note_features"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_features.py -q`
Expected: PASS (all feature tests).

- [ ] **Step 6: Commit**

```bash
git add drumhumanizer/features.py tests/test_features.py drumhumanizer/__init__.py
git commit -m "feat: assemble per-note structural feature table (Section A)"
```

---

### Task 5: Dataset builder script

**Files:**
- Create: `scripts/build_dataset.py`

**Interfaces:**
- Consumes: `build_note_features`, `drumhumanizer.midi.load_note_array`.
- Produces: `data/processed/egmd_tabular_{train,validation,test}.parquet` — the concatenated feature rows for each split.

- [ ] **Step 1: Write the script**

```python
# scripts/build_dataset.py
"""Build the Section A tabular dataset for every E-GMD split -> parquet.

Usage: .venv/bin/python scripts/build_dataset.py [--limit N] [--workers K]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drumhumanizer.features import build_note_features   # noqa: E402
from drumhumanizer.midi import load_note_array           # noqa: E402

EGMD_BASE = os.path.join("data", "e-gmd", "e-gmd-v1.0.0")
EGMD_CSV = os.path.join(EGMD_BASE, "e-gmd-v1.0.0.csv")
OUT_DIR = os.path.join("data", "processed")


def _worker(record):
    path, meta = record
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            na = load_note_array(path)
        if len(na) == 0:
            return None
        return build_note_features(na, meta)
    except Exception:
        return None


def build_split(df, split, workers):
    sub = df[df["split"] == split]
    records = [(os.path.join(EGMD_BASE, r.midi_filename), r._asdict())
               for r in sub.itertuples(index=False)]
    print(f"[{split}] {len(records)} files on {workers} workers ...")
    t0 = time.time()
    frames, failed = [], 0
    with Pool(workers) as pool:
        for i, out in enumerate(pool.imap_unordered(_worker, records, chunksize=32), 1):
            if out is None:
                failed += 1
            else:
                frames.append(out)
            if i % 5000 == 0 or i == len(records):
                print(f"  [{split}] {i}/{len(records)}  ({i/(time.time()-t0):.0f} files/s)")
    result = pd.concat(frames, ignore_index=True)
    print(f"[{split}] {len(result):,} rows, {failed} files failed")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(EGMD_CSV)
    if args.limit:
        df = df.groupby("split", group_keys=False).head(args.limit)

    for split in ("train", "validation", "test"):
        out = build_split(df, split, args.workers)
        path = os.path.join(OUT_DIR, f"egmd_tabular_{split}.parquet")
        out.to_parquet(path, index=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on a small limit**

Run: `.venv/bin/python scripts/build_dataset.py --limit 50 --workers 4`
Expected: writes three parquet files under `data/processed/`, non-zero row counts, 0 (or near-0) failures.

- [ ] **Step 3: Verify parquet loads and has no leakage columns**

Run:
```bash
.venv/bin/python -c "import pandas as pd; d=pd.read_parquet('data/processed/egmd_tabular_train.parquet'); print(d.shape); print([c for c in d.columns if 'vel' in c.lower()])"
```
Expected: a shape is printed and the only velocity column is `['velocity']`.

- [ ] **Step 4: Build the full dataset**

Run: `.venv/bin/python scripts/build_dataset.py --workers 9`
Expected: three parquet files for the full splits (train ≈ 11 M rows). This takes a few minutes.

- [ ] **Step 5: Commit** (parquet is gitignored under `data/`, so only the script is committed)

```bash
git add scripts/build_dataset.py
git commit -m "feat: build cached Section A tabular dataset over all splits"
```

---

### Task 6: Baselines ("dumb humanizer")

**Files:**
- Create: `drumhumanizer/baselines.py`
- Test: `tests/test_baselines.py`
- Modify: `drumhumanizer/__init__.py`

**Interfaces:**
- Consumes: `N_PHASE_BINS` from `drumhumanizer.features`.
- Produces:
  - `GlobalMeanBaseline().fit(df) -> self`; `.predict(df) -> np.ndarray` (constant train-mean velocity).
  - `LookupTableBaseline().fit(df) -> self`; `.predict(df) -> np.ndarray` — mean velocity per `(voice, genre, phase_bin)` with fallback `(voice, genre) -> (voice) -> global`. `phase_bin = min(int(phase_beat * N_PHASE_BINS), N_PHASE_BINS - 1)`.
  - Both `fit` use only `df.velocity` from the **train** frame; `predict` never reads velocity.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines.py
import numpy as np
import pandas as pd

from drumhumanizer.baselines import GlobalMeanBaseline, LookupTableBaseline


def _df():
    return pd.DataFrame({
        "voice": ["kick", "kick", "snare", "snare"],
        "genre": ["funk", "funk", "funk", "funk"],
        "phase_beat": [0.0, 0.0, 0.5, 0.5],
        "velocity": [100, 110, 40, 50],
    })


def test_global_mean_predicts_constant():
    b = GlobalMeanBaseline().fit(_df())
    pred = b.predict(_df())
    assert np.allclose(pred, 75.0)


def test_lookup_table_uses_group_mean():
    b = LookupTableBaseline().fit(_df())
    pred = b.predict(_df())
    assert np.allclose(pred, [105, 105, 45, 45])


def test_lookup_table_falls_back_for_unseen_key():
    b = LookupTableBaseline().fit(_df())
    unseen = pd.DataFrame({"voice": ["tom"], "genre": ["jazz"], "phase_beat": [0.3]})
    pred = b.predict(unseen)               # no matching key -> global mean
    assert np.isclose(pred[0], 75.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baselines.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Write the implementation**

```python
# drumhumanizer/baselines.py
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
```

- [ ] **Step 4: Export** — add `from .baselines import GlobalMeanBaseline, LookupTableBaseline` and both names to `__all__` in `drumhumanizer/__init__.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_baselines.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add drumhumanizer/baselines.py tests/test_baselines.py drumhumanizer/__init__.py
git commit -m "feat: add global-mean and lookup-table baselines"
```

---

### Task 7: Evaluation metrics

**Files:**
- Create: `drumhumanizer/metrics.py`
- Test: `tests/test_metrics.py`
- Modify: `drumhumanizer/__init__.py`

**Interfaces:**
- Produces:
  - `mae(y_true, y_pred) -> float`, `rmse(y_true, y_pred) -> float`.
  - `evaluate(df, y_pred) -> dict` where `df` has columns `velocity, file_id, bar_index, genre`. Returns keys: `mae, rmse, per_track_pearson, per_track_spearman, mean_abs_std_diff, global_std_ratio, within_bar_spearman, per_genre_mae` (`per_genre_mae` is a `dict`).
  - Correlation/ranking aggregates ignore groups with < 2 (corr) or < 3 (within-bar) notes and skip zero-variance groups (NaN-safe means).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import numpy as np
import pandas as pd

from drumhumanizer.metrics import evaluate, mae, rmse


def test_mae_rmse():
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0
    assert np.isclose(rmse([0, 0], [1, 1]), 1.0)


def test_evaluate_perfect_prediction():
    df = pd.DataFrame({
        "velocity": [10, 20, 30, 40, 50, 60],
        "file_id": ["a", "a", "a", "b", "b", "b"],
        "bar_index": [0, 0, 0, 0, 0, 0],
        "genre": ["funk", "funk", "funk", "rock", "rock", "rock"],
    })
    m = evaluate(df, np.array(df["velocity"], dtype=float))
    assert m["mae"] == 0.0
    assert np.isclose(m["per_track_pearson"], 1.0)
    assert np.isclose(m["within_bar_spearman"], 1.0)
    assert set(m["per_genre_mae"]) == {"funk", "rock"}
    assert m["per_genre_mae"]["funk"] == 0.0


def test_evaluate_std_ratio_detects_flattening():
    df = pd.DataFrame({
        "velocity": [0, 50, 100, 0, 50, 100],
        "file_id": ["a"] * 6, "bar_index": [0] * 6, "genre": ["funk"] * 6,
    })
    flat = np.full(6, 50.0)                 # predicts the mean -> no spread
    m = evaluate(df, flat)
    assert m["global_std_ratio"] < 0.1     # pred std / true std ~ 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Write the implementation**

```python
# drumhumanizer/metrics.py
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
```

- [ ] **Step 4: Export** — add `from .metrics import mae, rmse, evaluate` and the three names to `__all__` in `drumhumanizer/__init__.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add drumhumanizer/metrics.py tests/test_metrics.py drumhumanizer/__init__.py
git commit -m "feat: add velocity evaluation metrics (corr, std match, ranking)"
```

---

### Task 8: Train & evaluate the LightGBM baseline

**Files:**
- Create: `scripts/train_tabular.py`
- Create (output): `docs/plan_a/metrics.json`, `docs/plan_a/fig_importance.png`, `docs/plan_a/fig_pred_vs_true.png`, `docs/plan_a/fig_per_genre.png`

**Interfaces:**
- Consumes: the parquet from Task 5; `GlobalMeanBaseline`, `LookupTableBaseline`, `evaluate`, `CANONICAL_VOICES`.
- Produces: a printed comparison table + `docs/plan_a/metrics.json` with keys `global_mean`, `lookup_table`, `lightgbm`, each mapping to an `evaluate()` dict, plus `lightgbm_best_iteration`.

- [ ] **Step 1: Write the script**

```python
# scripts/train_tabular.py
"""Fit baselines + LightGBM on the Section A dataset and evaluate on the test split.

Usage: .venv/bin/python scripts/train_tabular.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
import lightgbm as lgb            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drumhumanizer.baselines import GlobalMeanBaseline, LookupTableBaseline  # noqa: E402
from drumhumanizer.metrics import evaluate                                    # noqa: E402

PROC = os.path.join("data", "processed")
OUT = os.path.join("docs", "plan_a")
CAT = ["voice", "genre", "style", "time_signature", "beat_type", "nearest_subdiv"]
DROP = ["file_id", "drummer", "split", "onset_sec", "bar_index", "velocity"]
SEED = 42


def _load(split):
    return pd.read_parquet(os.path.join(PROC, f"egmd_tabular_{split}.parquet"))


def _xy(df, cat_dtypes=None):
    X = df.drop(columns=DROP)
    for c in CAT:
        X[c] = X[c].astype("category")
        if cat_dtypes is not None:      # align val/test categories to train's
            X[c] = X[c].cat.set_categories(cat_dtypes[c])
    return X, df["velocity"].astype(float)


def main():
    os.makedirs(OUT, exist_ok=True)
    train, val, test = _load("train"), _load("validation"), _load("test")

    # baselines (fit on train, evaluate on test)
    gm = GlobalMeanBaseline().fit(train)
    lut = LookupTableBaseline().fit(train)
    results = {
        "global_mean": evaluate(test, gm.predict(test)),
        "lookup_table": evaluate(test, lut.predict(test)),
    }

    # LightGBM
    Xtr, ytr = _xy(train)
    cat_dtypes = {c: Xtr[c].cat.categories for c in CAT}
    Xval, yval = _xy(val, cat_dtypes)
    Xte, yte = _xy(test, cat_dtypes)
    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=2000, learning_rate=0.05,
        num_leaves=255, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric="l1",
              categorical_feature=CAT,
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    pred = model.predict(Xte, num_iteration=model.best_iteration_)
    results["lightgbm"] = evaluate(test, pred)
    results["lightgbm_best_iteration"] = int(model.best_iteration_)

    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # comparison table
    print(f"\n{'model':14} {'MAE':>7} {'RMSE':>7} {'trk_r':>7} {'wbar_rho':>9} {'std_ratio':>9}")
    for name in ("global_mean", "lookup_table", "lightgbm"):
        m = results[name]
        print(f"{name:14} {m['mae']:7.3f} {m['rmse']:7.3f} {m['per_track_pearson']:7.3f} "
              f"{m['within_bar_spearman']:9.3f} {m['global_std_ratio']:9.3f}")

    # figures
    imp = pd.Series(model.feature_importances_, index=Xtr.columns).sort_values()
    ax = imp.tail(20).plot.barh(figsize=(8, 7)); ax.set_title("LightGBM gain importance (top 20)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_importance.png"), dpi=120); plt.close()

    idx = np.random.RandomState(SEED).choice(len(yte), size=min(20000, len(yte)), replace=False)
    plt.figure(figsize=(5, 5)); plt.hexbin(yte.to_numpy()[idx], pred[idx], gridsize=60, cmap="viridis")
    plt.plot([0, 127], [0, 127], "r--", lw=1); plt.xlabel("true"); plt.ylabel("pred")
    plt.title("LightGBM: predicted vs true velocity")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_pred_vs_true.png"), dpi=120); plt.close()

    genres = results["lightgbm"]["per_genre_mae"]
    gs = pd.Series(genres).sort_values()
    gs.plot.barh(figsize=(7, 8)); plt.xlabel("MAE"); plt.title("LightGBM per-genre MAE")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_per_genre.png"), dpi=120); plt.close()

    print(f"\nwrote results to {OUT}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run training on the full dataset**

Run: `.venv/bin/python scripts/train_tabular.py`
Expected: prints a 3-row comparison table; writes `docs/plan_a/metrics.json` and three figures.

- [ ] **Step 3: Verify the model beats the baselines**

Confirm in the printed table that `lightgbm` has **lower MAE and RMSE** than both `global_mean` and `lookup_table`, and a **higher `per_track_pearson`** and `within_bar_spearman`. Beating the `lookup_table` is the real bar (design §7). Also confirm `lightgbm` `global_std_ratio` is closer to 1.0 than `global_mean` (which is ~0). If LightGBM does **not** beat the lookup table, stop and report — do not tune blindly.

- [ ] **Step 4: Write a short results note**

Create `docs/plan_a/results.md` summarizing: the comparison table (copy the printed numbers), the top-10 feature importances, the per-genre MAE spread (does funk get more dynamic range than pop? design §7 hypothesis), and whether the model beats the lookup table. Reference the three figures.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_tabular.py docs/plan_a/
git commit -m "feat: train and evaluate LightGBM tabular velocity baseline"
```

---

## Self-Review notes

- **Spec §4 coverage:** drum part (Task 2 voice), metrical position sin/cos + raw (Task 3/4), swing ratio (Task 3), nearest-subdivision (Task 3), conditioning style/genre/bpm/time_sig/beat_type (Task 4), time_to_prev/next & same-voice intervals (Task 4), simultaneity multi-hot + count (Task 4), local density (Task 4). ✓
- **Spec §6 model 1 (tabular):** LightGBM native categorical, MAE/MSE (Task 8). ✓
- **Spec §7 evaluation:** provided splits (Global Constraints), both baselines (Task 6), MAE/RMSE + per-track corr + std match + within-bar ranking + per-genre (Task 7/8). ✓ *Held-out-drummer secondary check and the optional listening test are deferred (drummer column is carried in the dataset so it can be added later without a rebuild).*
- **Leakage:** enforced by construction and asserted in `test_build_note_features_columns_and_no_leakage` and the parquet check (Task 5 Step 3). ✓
- **Phase 0 decisions:** `SIMULTANEITY_TOL=0.02` (Task 3), 14-voice map (Task 2), time-sig kept for trees (Task 8 `CAT`). ✓
