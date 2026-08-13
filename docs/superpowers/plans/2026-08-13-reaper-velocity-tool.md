# Reaper velocity-restoration tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Reaper Python ReaScript that rewrites the velocities of a selected MIDI drum region in place, backed by a warm local Python inference service serving the `drum_dynamics` LightGBM and MDN models.

**Architecture:** A warm HTTP inference service (`ml/src/drum_dynamics/serve/`) loads both models once and answers `POST /predict` (notes+params → velocities) and `GET /health` (dropdown lists). A stdlib-only Reaper ReaScript (`plugin/reaper/`) gathers the selection, POSTs to the service (auto-starting it if down), and writes the returned velocities back into the selected notes only. A pure client-logic module and a model-agnostic inference core keep both sides unit-testable.

**Tech Stack:** Python ≥3.12, pandas/numpy, LightGBM, PyTorch, stdlib `http.server`, Reaper ReaScript (Python).

**Spec:** `docs/superpowers/specs/2026-08-13-reaper-velocity-tool-design.md`

## Global Constraints

- Python ≥ 3.12 (from `ml/pyproject.toml`).
- **No new runtime dependencies.** Service uses stdlib `http.server`; reuse existing `torch`/`lightgbm`/`pandas`/`joblib`.
- **ReaScript is stdlib-only** — no third-party imports (it runs under Reaper's embedded Python).
- **`heads.sample` change must be backward-compatible:** `temperature=1.0` reproduces prior output exactly.
- **Genre is derived from style:** `genre = style.split("/")[0]` (matches `build_note_features`).
- **Output velocities are integers clamped to `[1, 127]`.**
- Models load from `data/processed/` by default: `lightgbm_model.joblib`, `head_mdn.pt`, `mdn_meta.json`, `lightgbm_features.json`.
- **Request/response contract** (used by tasks 3–7):
  ```json
  request = {
    "model": "lgbm" | "mdn",
    "style": "rock/halftime",
    "temperature": 1.0,
    "blend": 1.0,
    "beat_type": "beat" | "fill",
    "bpm": 120.0,
    "time_signature": "4-4",
    "notes": [{"index": 0, "pitch": 36, "onset_sec": 0.0, "velocity": 40, "selected": true}]
  }
  response = {"velocities": {"0": 97}}   // only selected notes, keys are stringified indices
  ```

---

### Task 1: Temperature support in the MDN sampler

**Files:**
- Modify: `ml/src/drum_dynamics/models/heads.py` (the `sample` function, ~line 144-164)
- Test: `ml/tests/test_heads.py`

**Interfaces:**
- Produces: `heads.sample(head_type, raw, generator=None, temperature=1.0) -> Tensor`. For `mdn`: `temperature<=0` returns the mixture mean (`heads.point`); `temperature>0` tempers the mixture weights by `1/temperature` and scales component sigma by `temperature`. For `gaussian`: scales sigma by `temperature`. `categorical` ignores `temperature`. `temperature=1.0` reproduces prior behavior.

- [ ] **Step 1: Write the failing tests**

Add to `ml/tests/test_heads.py`:

```python
import torch
from drum_dynamics.models import heads


def _mdn_raw(n=2000):
    # deterministic raw params for MDN: [*, 3*K]
    torch.manual_seed(0)
    return torch.randn(n, 3 * heads.MDN_K)


def test_sample_temperature_zero_returns_mixture_mean():
    raw = _mdn_raw()
    got = heads.sample("mdn", raw, temperature=0.0)
    assert torch.allclose(got, heads.point("mdn", raw).clamp(0, 127))


def test_sample_temperature_is_deterministic_with_seed():
    raw = _mdn_raw()
    a = heads.sample("mdn", raw, generator=torch.Generator().manual_seed(7), temperature=1.3)
    b = heads.sample("mdn", raw, generator=torch.Generator().manual_seed(7), temperature=1.3)
    assert torch.equal(a, b)


def test_higher_temperature_widens_spread():
    raw = _mdn_raw()
    lo = heads.sample("mdn", raw, generator=torch.Generator().manual_seed(1), temperature=0.2)
    hi = heads.sample("mdn", raw, generator=torch.Generator().manual_seed(1), temperature=2.5)
    assert hi.std() > lo.std()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest ml/tests/test_heads.py -k temperature -v`
Expected: FAIL — `sample()` has no `temperature` kwarg (`TypeError`).

- [ ] **Step 3: Implement temperature in `sample`**

Replace the `sample` function body in `heads.py` with (keep the signature additions and the `.clamp(0, 127)` return):

```python
def sample(head_type, raw, generator=None, temperature=1.0):
    if head_type == "gaussian":
        mu, sigma = _gauss_params(raw)
        eps = torch.randn(mu.shape, generator=generator, device=mu.device)
        y = mu + temperature * sigma * eps
    elif head_type == "mdn":
        log_pi, mu, sigma = _mdn_params(raw)
        if temperature <= 0:
            return (log_pi.exp() * mu).sum(-1).clamp(0, 127)   # mixture mean == point("mdn")
        tempered = torch.log_softmax(log_pi / temperature, dim=-1)
        flat_pi = tempered.exp().reshape(-1, MDN_K)
        k = torch.multinomial(flat_pi, 1, generator=generator).reshape(mu.shape[:-1])
        muk = mu.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        sigk = sigma.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        eps = torch.randn(muk.shape, generator=generator, device=muk.device)
        y = muk + temperature * sigk * eps
    elif head_type == "categorical":
        p = torch.softmax(raw, dim=-1)
        b = torch.multinomial(p.reshape(-1, N_BINS), 1, generator=generator).reshape(p.shape[:-1])
        u = torch.rand(b.shape, generator=generator, device=b.device)
        y = b.float() * BIN_WIDTH + u * BIN_WIDTH
    else:
        raise ValueError(f"unknown head_type {head_type!r}")
    return y.clamp(0, 127)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest ml/tests/test_heads.py -v`
Expected: PASS (all, including any pre-existing tests).

- [ ] **Step 5: Commit**

```bash
git add ml/src/drum_dynamics/models/heads.py ml/tests/test_heads.py
git commit -m "feat: temperature control in MDN/gaussian samplers"
```

---

### Task 2: MDN inference-artifact export

**Files:**
- Create: `ml/src/drum_dynamics/serve/__init__.py` (empty)
- Create: `ml/src/drum_dynamics/serve/export.py`
- Create: `ml/scripts/export_mdn.py`
- Test: `ml/tests/test_serve_export.py`

**Interfaces:**
- Produces: `serve.export.build_mdn_meta(train_df) -> dict` with keys `genre_vocab: dict[str,int]`, `bpm_mean: float`, `bpm_std: float`, `head: "mdn"`. Written by `scripts/export_mdn.py` to `data/processed/mdn_meta.json`. Consumed by `MdnModel.load` (Task 4).

- [ ] **Step 1: Write the failing test**

Create `ml/tests/test_serve_export.py`:

```python
import pandas as pd
from drum_dynamics.serve.export import build_mdn_meta


def test_build_mdn_meta_shapes():
    df = pd.DataFrame({"genre": ["rock", "jazz", "rock"], "bpm": [120.0, 90.0, 100.0]})
    meta = build_mdn_meta(df)
    assert meta["head"] == "mdn"
    assert meta["genre_vocab"] == {"jazz": 1, "rock": 2}   # sorted, 0 reserved for <unk>
    assert abs(meta["bpm_mean"] - 103.3333) < 1e-3
    assert meta["bpm_std"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest ml/tests/test_serve_export.py -v`
Expected: FAIL — module `drum_dynamics.serve.export` does not exist.

- [ ] **Step 3: Implement the export helper and script**

Create `ml/src/drum_dynamics/serve/__init__.py` (empty file).

Create `ml/src/drum_dynamics/serve/export.py`:

```python
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
```

Create `ml/scripts/export_mdn.py`:

```python
#!/usr/bin/env python
"""Export MDN inference artifacts (genre vocab + bpm stats) to mdn_meta.json.

The MDN checkpoint (head_mdn.pt["best_model"]) is a full model state dict, but the
genre vocabulary and bpm normalization are recomputed from the train parquet at
train time and never persisted. The service needs them to reproduce inference.

Usage: .venv/bin/python ml/scripts/export_mdn.py
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from drum_dynamics.serve.export import build_mdn_meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=os.path.join("data", "processed", "egmd_tabular_train.parquet"))
    p.add_argument("--out", default=os.path.join("data", "processed", "mdn_meta.json"))
    args = p.parse_args()

    meta = build_mdn_meta(pd.read_parquet(args.train))
    with open(args.out, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {args.out}  (genres={len(meta['genre_vocab'])}, "
          f"bpm_mean={meta['bpm_mean']:.2f}, bpm_std={meta['bpm_std']:.2f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest ml/tests/test_serve_export.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/src/drum_dynamics/serve/__init__.py ml/src/drum_dynamics/serve/export.py ml/scripts/export_mdn.py ml/tests/test_serve_export.py
git commit -m "feat: export MDN inference artifacts (vocab + bpm stats)"
```

---

### Task 3: Model-agnostic inference core

**Files:**
- Create: `ml/src/drum_dynamics/serve/core.py`
- Test: `ml/tests/test_serve_core.py`

**Interfaces:**
- Consumes: `data.features.build_note_features` (unchanged).
- Produces: `serve.core.predict_velocities(request: dict, predict_all: Callable[[pd.DataFrame], np.ndarray]) -> dict[int, int]`. `predict_all` receives the feature DataFrame (rows in `build_note_features`'s stable-onset order) and returns one predicted velocity per row in that same order. The core maps predictions back to original note indices, blends with each selected note's original velocity, rounds, and clamps to `[1, 127]`. Only `selected` notes appear in the result.

- [ ] **Step 1: Write the failing tests**

Create `ml/tests/test_serve_core.py`:

```python
import numpy as np
from drum_dynamics.serve.core import predict_velocities


def _req(notes, blend=1.0):
    return {"model": "lgbm", "style": "funk/groove1", "temperature": 1.0,
            "blend": blend, "beat_type": "beat", "bpm": 120.0,
            "time_signature": "4-4", "notes": notes}


def test_only_selected_notes_returned():
    notes = [{"index": 5, "pitch": 36, "onset_sec": 0.0, "velocity": 40, "selected": True},
             {"index": 6, "pitch": 38, "onset_sec": 0.0, "velocity": 40, "selected": False}]
    out = predict_velocities(_req(notes), lambda df: np.full(len(df), 100.0))
    assert set(out) == {5}


def test_blend_and_clamp():
    notes = [{"index": 0, "pitch": 36, "onset_sec": 0.0, "velocity": 40, "selected": True}]
    # blend 0.5 of pred 100 with orig 40 -> 70
    assert predict_velocities(_req(notes, blend=0.5), lambda df: np.full(len(df), 100.0))[0] == 70
    # clamp high and low
    assert predict_velocities(_req(notes), lambda df: np.full(len(df), 500.0))[0] == 127
    assert predict_velocities(_req(notes), lambda df: np.full(len(df), -10.0))[0] == 1


def test_predictions_map_back_to_correct_index_when_unsorted():
    # onsets out of order; predict_all returns each row's own onset_sec.
    notes = [{"index": 0, "pitch": 36, "onset_sec": 3.0, "velocity": 0, "selected": True},
             {"index": 1, "pitch": 38, "onset_sec": 1.0, "velocity": 0, "selected": True},
             {"index": 2, "pitch": 42, "onset_sec": 2.0, "velocity": 0, "selected": True}]
    out = predict_velocities(_req(notes), lambda df: df["onset_sec"].to_numpy())
    assert out == {0: 3, 1: 1, 2: 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest ml/tests/test_serve_core.py -v`
Expected: FAIL — module `drum_dynamics.serve.core` does not exist.

- [ ] **Step 3: Implement the core**

Create `ml/src/drum_dynamics/serve/core.py`:

```python
"""Model-agnostic inference orchestration: notes+params -> new velocities."""
from __future__ import annotations

import numpy as np

from ..data.features import build_note_features

_NOTE_DT = np.dtype([("onset_sec", float), ("pitch", int), ("velocity", int)])


def _note_array(notes):
    return np.array([(n["onset_sec"], n["pitch"], n["velocity"]) for n in notes], dtype=_NOTE_DT)


def predict_velocities(request, predict_all):
    notes = request["notes"]
    if not notes:
        return {}
    na = _note_array(notes)
    # build_note_features sorts by onset_sec (stable); reproduce that order to map back.
    order = np.argsort(na["onset_sec"], kind="stable")
    meta = {"id": "infer", "drummer": "infer", "split": "infer",
            "bpm": float(request["bpm"]), "time_signature": str(request["time_signature"]),
            "style": str(request["style"]), "beat_type": str(request["beat_type"])}
    df = build_note_features(na, meta)                 # rows correspond to na[order]
    preds_sorted = np.asarray(predict_all(df), dtype=float)
    preds = np.empty(len(notes), dtype=float)
    preds[order] = preds_sorted                        # back to input positions

    blend = float(request.get("blend", 1.0))
    out = {}
    for pos, n in enumerate(notes):
        if not n.get("selected"):
            continue
        v = blend * preds[pos] + (1.0 - blend) * float(n["velocity"])
        out[n["index"]] = int(np.clip(round(v), 1, 127))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest ml/tests/test_serve_core.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/src/drum_dynamics/serve/core.py ml/tests/test_serve_core.py
git commit -m "feat: model-agnostic velocity inference core"
```

---

### Task 4: Model wrappers + engine

**Files:**
- Create: `ml/src/drum_dynamics/serve/models.py`
- Test: `ml/tests/test_serve_models.py`

**Interfaces:**
- Consumes: `serve.core.predict_velocities` (Task 3), `heads.sample` (Task 1), `models.model.VelocityTransformer`, `data.seqdata.{build_split_tensors, scatter_predictions}`.
- Produces:
  - `LgbmModel(bundle).predict_all(df) -> np.ndarray` — reproduces training's categorical encoding via `bundle["cat_categories"]`.
  - `MdnModel(transformer, genre_vocab, bpm_mean, bpm_std, device="cpu").predict_all(df, temperature=1.0, seed=42) -> np.ndarray`, plus `MdnModel.load(meta_path, ckpt_path, device="cpu")`.
  - `Engine(lgbm, mdn, styles, genres)` with `.predict(request) -> dict[int,int]` (routes on `request["model"]`) and `.levels() -> {"models","styles","genres"}`, plus `Engine.load(proc_dir="data/processed")`.

- [ ] **Step 1: Write the failing tests**

Create `ml/tests/test_serve_models.py`:

```python
import numpy as np
import pandas as pd
import lightgbm as lgb

from drum_dynamics.serve.models import LgbmModel, MdnModel, Engine
from drum_dynamics.models.model import VelocityTransformer
from drum_dynamics.data.features import build_note_features

_DT = np.dtype([("onset_sec", float), ("pitch", int), ("velocity", int)])


def _feature_df():
    na = np.array([(0.0, 36, 100), (0.0, 38, 40), (0.5, 42, 70), (1.0, 36, 90)], dtype=_DT)
    meta = dict(id="x", drummer="x", split="train", bpm=120, time_signature="4-4",
                style="funk/groove", beat_type="beat")
    return build_note_features(na, meta)


def test_mdn_predict_all_plumbing_with_fresh_model():
    df = _feature_df()
    gv = {"funk": 1}
    m = VelocityTransformer(n_genres=len(gv) + 1, head="mdn")
    mdn = MdnModel(m, gv, bpm_mean=120.0, bpm_std=10.0)
    out = mdn.predict_all(df, temperature=1.0, seed=0)
    assert out.shape == (len(df),)
    assert ((out >= 0) & (out <= 127)).all()


def test_lgbm_predict_all_returns_row_aligned():
    df = _feature_df()
    from drum_dynamics.serve.models import LgbmModel
    CAT = ["voice", "genre", "style", "time_signature", "beat_type", "nearest_subdiv"]
    DROP = ["file_id", "drummer", "split", "onset_sec", "bar_index", "velocity"]
    X = df.drop(columns=DROP)
    for c in CAT:
        X[c] = X[c].astype("category")
    model = lgb.LGBMRegressor(n_estimators=3, min_child_samples=1).fit(X, df["velocity"], categorical_feature=CAT)
    bundle = {"model": model, "cat": CAT, "drop": DROP,
              "cat_categories": {c: X[c].cat.categories for c in CAT}, "best_iteration": 3}
    out = LgbmModel(bundle).predict_all(df)
    assert out.shape == (len(df),)


def test_engine_routes_and_levels():
    class Fake:
        def predict_all(self, df, **kw):
            return np.full(len(df), 111.0)
    eng = Engine(Fake(), Fake(), styles=["funk"], genres=["funk"])
    req = {"model": "lgbm", "style": "funk", "temperature": 1.0, "blend": 1.0,
           "beat_type": "beat", "bpm": 120, "time_signature": "4-4",
           "notes": [{"index": 0, "pitch": 36, "onset_sec": 0.0, "velocity": 1, "selected": True}]}
    assert eng.predict(req) == {0: 111}
    assert eng.levels()["models"] == ["lgbm", "mdn"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest ml/tests/test_serve_models.py -v`
Expected: FAIL — module `drum_dynamics.serve.models` does not exist.

- [ ] **Step 3: Implement the models + engine**

Create `ml/src/drum_dynamics/serve/models.py`:

```python
"""Model wrappers (LightGBM, MDN) + the Engine that routes requests to them."""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import torch

from .core import predict_velocities
from ..models import heads
from ..models.model import VelocityTransformer
from ..data.seqdata import build_split_tensors, scatter_predictions


class LgbmModel:
    def __init__(self, bundle):
        self.model = bundle["model"]
        self.cat = list(bundle["cat"])
        self.drop = list(bundle["drop"])
        self.cat_categories = bundle["cat_categories"]
        self.best_iteration = int(bundle["best_iteration"])

    def predict_all(self, df):
        X = df.drop(columns=self.drop)
        for c in self.cat:
            X[c] = X[c].astype("category").cat.set_categories(self.cat_categories[c])
        return np.asarray(self.model.predict(X, num_iteration=self.best_iteration), dtype=float)


class MdnModel:
    def __init__(self, transformer, genre_vocab, bpm_mean, bpm_std, device="cpu"):
        self.m = transformer.to(device).eval()
        self.genre_vocab = genre_vocab
        self.bpm_mean = float(bpm_mean)
        self.bpm_std = float(bpm_std)
        self.device = device

    @classmethod
    def load(cls, meta_path, ckpt_path, device="cpu"):
        with open(meta_path) as fh:
            meta = json.load(fh)
        gv = meta["genre_vocab"]
        m = VelocityTransformer(n_genres=len(gv) + 1, head="mdn")
        m.load_state_dict(torch.load(ckpt_path, map_location=device)["best_model"])
        return cls(m, gv, meta["bpm_mean"], meta["bpm_std"], device)

    def predict_all(self, df, temperature=1.0, seed=42):
        t = build_split_tensors(df, self.genre_vocab, self.bpm_mean, self.bpm_std)
        gen = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            raw = self.m(t["voice_idx"].to(self.device), t["genre_idx"].to(self.device),
                         t["num_feats"].to(self.device), t["pad_mask"].to(self.device)).cpu()
        s = heads.sample("mdn", raw, generator=gen, temperature=float(temperature))
        return scatter_predictions(t["row_idx"], s, t["pad_mask"], len(df))


class Engine:
    def __init__(self, lgbm, mdn, styles, genres):
        self.lgbm = lgbm
        self.mdn = mdn
        self.styles = list(styles)
        self.genres = list(genres)

    @classmethod
    def load(cls, proc_dir=os.path.join("data", "processed")):
        lgbm = LgbmModel(joblib.load(os.path.join(proc_dir, "lightgbm_model.joblib")))
        mdn = MdnModel.load(os.path.join(proc_dir, "mdn_meta.json"),
                            os.path.join(proc_dir, "head_mdn.pt"))
        with open(os.path.join(proc_dir, "lightgbm_features.json")) as fh:
            feats = json.load(fh)
        lv = feats["categorical_levels"]
        return cls(lgbm, mdn, lv["style"], lv["genre"])

    def predict(self, request):
        model = request["model"]
        if model == "lgbm":
            predict_all = self.lgbm.predict_all
        elif model == "mdn":
            temp = float(request.get("temperature", 1.0))
            seed = int(request.get("seed", 42))
            predict_all = lambda df: self.mdn.predict_all(df, temperature=temp, seed=seed)  # noqa: E731
        else:
            raise ValueError(f"unknown model {model!r}")
        return predict_velocities(request, predict_all)

    def levels(self):
        return {"models": ["lgbm", "mdn"], "styles": self.styles, "genres": self.genres}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest ml/tests/test_serve_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/src/drum_dynamics/serve/models.py ml/tests/test_serve_models.py
git commit -m "feat: LightGBM + MDN model wrappers and routing engine"
```

---

### Task 5: HTTP service + lifecycle

**Files:**
- Create: `ml/src/drum_dynamics/serve/server.py`
- Create: `ml/src/drum_dynamics/serve/__main__.py`
- Test: `ml/tests/test_serve_http.py`

**Interfaces:**
- Consumes: `Engine` (Task 4).
- Produces:
  - `serve.server.route(engine, method, path, body) -> (int, dict)` — pure routing over a loaded engine.
  - `serve.server.pid_alive(pid) -> bool` and `serve.server.should_exit(now, last_request, idle_timeout, parent_alive) -> bool`.
  - `serve.server.run(engine, port, parent_pid, idle_timeout)` — starts the HTTP server + a watchdog thread.
  - `python -m drum_dynamics.serve --port 8765 --parent-pid <PID> --idle-timeout 1800 [--proc-dir DIR]`.

- [ ] **Step 1: Write the failing tests**

Create `ml/tests/test_serve_http.py`:

```python
from drum_dynamics.serve.server import route, pid_alive, should_exit


class _Eng:
    def levels(self):
        return {"models": ["lgbm", "mdn"], "styles": ["funk"], "genres": ["funk"]}

    def predict(self, request):
        return {0: 99}


def test_route_health():
    status, body = route(_Eng(), "GET", "/health", None)
    assert status == 200 and body["status"] == "ok" and body["models"] == ["lgbm", "mdn"]


def test_route_predict_stringifies_keys():
    status, body = route(_Eng(), "POST", "/predict", {"model": "lgbm", "notes": []})
    assert status == 200 and body == {"velocities": {"0": 99}}


def test_route_unknown_path_404():
    status, _ = route(_Eng(), "GET", "/nope", None)
    assert status == 404


def test_pid_alive_current_process():
    import os
    assert pid_alive(os.getpid()) is True
    assert pid_alive(2_147_483_000) is False   # implausibly-high PID


def test_should_exit_conditions():
    assert should_exit(now=100.0, last_request=0.0, idle_timeout=30, parent_alive=True) is True
    assert should_exit(now=10.0, last_request=0.0, idle_timeout=30, parent_alive=True) is False
    assert should_exit(now=10.0, last_request=0.0, idle_timeout=30, parent_alive=False) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest ml/tests/test_serve_http.py -v`
Expected: FAIL — module `drum_dynamics.serve.server` does not exist.

- [ ] **Step 3: Implement the server**

Create `ml/src/drum_dynamics/serve/server.py`:

```python
"""Stdlib HTTP wrapper + lifecycle watchdog for the inference engine."""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def route(engine, method, path, body):
    """Pure request router over a loaded engine. Returns (status, json-able dict)."""
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", **engine.levels()}
    if method == "POST" and path == "/predict":
        vel = engine.predict(body)
        return 200, {"velocities": {str(k): v for k, v in vel.items()}}
    return 404, {"error": "not found"}


def pid_alive(pid):
    if pid is None:
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists but not owned by us
    return True


def should_exit(now, last_request, idle_timeout, parent_alive):
    if not parent_alive:
        return True
    if idle_timeout and (now - last_request) > idle_timeout:
        return True
    return False


def _make_handler(engine, state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):        # silence default stderr logging
            pass

        def _send(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            status, body = route(engine, "GET", self.path, None)
            self._send(status, body)

        def do_POST(self):
            state["last_request"] = time.monotonic()
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                status, out = route(engine, "POST", self.path, body)
            except Exception as e:                       # noqa: BLE001 - report to client
                status, out = 500, {"error": str(e)}
            self._send(status, out)

    return Handler


def run(engine, port=8765, parent_pid=None, idle_timeout=1800):
    state = {"last_request": time.monotonic()}
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(engine, state))

    def watchdog():
        while True:
            time.sleep(5)
            if should_exit(time.monotonic(), state["last_request"], idle_timeout, pid_alive(parent_pid)):
                os._exit(0)

    threading.Thread(target=watchdog, daemon=True).start()
    print(f"Dynamics Needed engine on http://127.0.0.1:{port} (parent={parent_pid})")
    httpd.serve_forever()
```

Create `ml/src/drum_dynamics/serve/__main__.py`:

```python
"""Entry point: python -m drum_dynamics.serve [options]."""
from __future__ import annotations

import argparse
import os

from .models import Engine
from .server import run


def main() -> None:
    p = argparse.ArgumentParser(prog="drum_dynamics.serve")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--parent-pid", type=int, default=None)
    p.add_argument("--idle-timeout", type=int, default=1800)
    p.add_argument("--proc-dir", default=os.path.join("data", "processed"))
    args = p.parse_args()

    engine = Engine.load(args.proc_dir)
    run(engine, port=args.port, parent_pid=args.parent_pid, idle_timeout=args.idle_timeout)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest ml/tests/test_serve_http.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/src/drum_dynamics/serve/server.py ml/src/drum_dynamics/serve/__main__.py ml/tests/test_serve_http.py
git commit -m "feat: stdlib HTTP inference service + lifecycle watchdog"
```

---

### Task 6: Reaper client pure logic

**Files:**
- Create: `plugin/reaper/dn_core.py`
- Test: `ml/tests/test_reaper_core.py`

**Interfaces:**
- Produces (all pure, stdlib-only):
  - `dn_core.genres_from_styles(styles) -> list[str]` — sorted unique `style.split("/")[0]`.
  - `dn_core.filter_styles_by_genre(styles, genre) -> list[str]`.
  - `dn_core.resolve_target_indices(notes) -> list[int]` — selected indices if any are selected, else all.
  - `dn_core.build_predict_request(model, style, temperature, blend, beat_type, bpm, time_signature, notes) -> dict` matching the Global-Constraints contract.
  - `dn_core.parse_velocities(response) -> dict[int,int]`.

- [ ] **Step 1: Write the failing tests**

Create `ml/tests/test_reaper_core.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join("plugin", "reaper"))
import dn_core


def test_genres_from_styles():
    assert dn_core.genres_from_styles(["rock/indie", "rock", "jazz/swing"]) == ["jazz", "rock"]


def test_filter_styles_by_genre():
    styles = ["rock", "rock/indie", "jazz/swing"]
    assert dn_core.filter_styles_by_genre(styles, "rock") == ["rock", "rock/indie"]


def test_resolve_target_indices_prefers_selected():
    notes = [{"index": 0, "selected": True}, {"index": 1, "selected": False}]
    assert dn_core.resolve_target_indices(notes) == [0]
    all_unsel = [{"index": 3, "selected": False}, {"index": 4, "selected": False}]
    assert dn_core.resolve_target_indices(all_unsel) == [3, 4]


def test_build_predict_request_shape():
    notes = [{"index": 0, "pitch": 36, "onset_sec": 0.0, "velocity": 40, "selected": True}]
    req = dn_core.build_predict_request("mdn", "rock/indie", 1.2, 0.8, "fill", 128.0, "4-4", notes)
    assert req["model"] == "mdn" and req["style"] == "rock/indie"
    assert req["temperature"] == 1.2 and req["blend"] == 0.8 and req["beat_type"] == "fill"
    assert req["bpm"] == 128.0 and req["time_signature"] == "4-4" and req["notes"] == notes


def test_parse_velocities():
    assert dn_core.parse_velocities({"velocities": {"0": 97, "3": 12}}) == {0: 97, 3: 12}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest ml/tests/test_reaper_core.py -v`
Expected: FAIL — `plugin/reaper/dn_core.py` does not exist.

- [ ] **Step 3: Implement the pure client logic**

Create `plugin/reaper/dn_core.py`:

```python
"""Pure, stdlib-only client logic shared by the Reaper ReaScript (unit-tested).

Kept free of any Reaper API so it runs under normal pytest.
"""
from __future__ import annotations


def genres_from_styles(styles):
    return sorted({s.split("/")[0] for s in styles})


def filter_styles_by_genre(styles, genre):
    return [s for s in styles if s.split("/")[0] == genre]


def resolve_target_indices(notes):
    selected = [n["index"] for n in notes if n.get("selected")]
    return selected if selected else [n["index"] for n in notes]


def build_predict_request(model, style, temperature, blend, beat_type, bpm, time_signature, notes):
    return {
        "model": model,
        "style": style,
        "temperature": float(temperature),
        "blend": float(blend),
        "beat_type": beat_type,
        "bpm": float(bpm),
        "time_signature": time_signature,
        "notes": notes,
    }


def parse_velocities(response):
    return {int(k): int(v) for k, v in response["velocities"].items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest ml/tests/test_reaper_core.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugin/reaper/dn_core.py ml/tests/test_reaper_core.py
git commit -m "feat: pure Reaper client logic (request build, selection, parsing)"
```

---

### Task 7: Reaper ReaScript shell

**Files:**
- Create: `plugin/reaper/dynamics_needed.py`
- Modify: `.gitignore` (add runtime + local config)

**Interfaces:**
- Consumes: `dn_core` (Task 6), the running HTTP service (Task 5), a machine-local config written by Task 8 (`plugin/reaper/config.local.json`: `{"venv_python", "repo_root", "port"}`).
- Produces: the Reaper action script (imported/run inside Reaper). No automated test — a **manual smoke test** validates it.

- [ ] **Step 1: Add runtime/config ignores**

Append to `.gitignore`:

```
# Dynamics Needed Reaper tool (machine-local)
plugin/reaper/.runtime/
plugin/reaper/config.local.json
```

- [ ] **Step 2: Write the ReaScript**

Create `plugin/reaper/dynamics_needed.py`:

```python
"""Dynamics Needed — Reaper action: restore velocities of the selected drum notes.

Runs under Reaper's embedded Python (stdlib only). Auto-starts the local inference
service if it isn't already running, then rewrites the selected notes' velocities.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

from reaper_python import *  # noqa: F401,F403  (provides RPR_* functions)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dn_core  # noqa: E402


def _msg(text):
    RPR_ShowConsoleMsg(text + "\n")


def _load_config():
    with open(os.path.join(_HERE, "config.local.json")) as fh:
        return json.load(fh)


def _base_url(cfg):
    return "http://127.0.0.1:{}".format(cfg.get("port", 8765))


def _health(cfg):
    try:
        with urllib.request.urlopen(_base_url(cfg) + "/health", timeout=1) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _start_engine(cfg):
    runtime = os.path.join(_HERE, ".runtime")
    os.makedirs(runtime, exist_ok=True)
    log = open(os.path.join(runtime, "engine.log"), "a")
    subprocess.Popen(
        [cfg["venv_python"], "-m", "drum_dynamics.serve",
         "--port", str(cfg.get("port", 8765)), "--parent-pid", str(os.getpid())],
        cwd=cfg["repo_root"], stdout=log, stderr=log,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )


def _ensure_engine(cfg):
    health = _health(cfg)
    if health:
        return health
    _start_engine(cfg)
    for _ in range(30):                      # ~15s for cold torch import
        time.sleep(0.5)
        health = _health(cfg)
        if health:
            return health
    return None


def _active_take():
    editor = RPR_MIDIEditor_GetActive()
    return RPR_MIDIEditor_GetTake(editor) if editor else None


def _read_notes(take):
    """Return list of note dicts with project-time onsets and selection flags."""
    _, _, note_count, _, _ = RPR_MIDI_CountEvts(take, 0, 0, 0)
    notes = []
    for i in range(note_count):
        ok, _, _, sel, _, startppq, _, _, pitch, vel = RPR_MIDI_GetNote(
            take, i, 0, 0, 0.0, 0.0, 0, 0, 0)
        if not ok:
            continue
        onset = RPR_MIDI_GetProjTimeFromPPQPos(take, startppq)
        notes.append({"index": i, "pitch": int(pitch), "onset_sec": float(onset),
                      "velocity": int(vel), "selected": bool(sel)})
    return notes


def _tempo_and_sig(take):
    item = RPR_GetMediaItemTake_Item(take)
    pos = RPR_GetMediaItemInfo_Value(item, "D_POSITION")
    _, _, _, num, denom, bpm = RPR_TimeMap_GetTimeSigAtTime(0, pos, 0, 0, 0.0)
    return float(bpm), "{}-{}".format(int(num), int(denom))


def _track_key(take):
    item = RPR_GetMediaItemTake_Item(take)
    track = RPR_GetMediaItem_Track(item)
    _, _, guid = RPR_GetSetMediaTrackInfo_String(track, "GUID", "", False)
    return "dn_last_" + guid


def _load_last(key):
    _, _, val, _ = RPR_GetProjExtState(0, "DynamicsNeeded", key, "")
    try:
        return json.loads(val) if val else {}
    except Exception:
        return {}


def _save_last(key, params):
    RPR_SetProjExtState(0, "DynamicsNeeded", key, json.dumps(params))


def _dialog(health, last):
    """Native GetUserInputs dialog. Returns params dict or None if cancelled."""
    styles = health["styles"]
    genres = health["genres"]
    defaults = {"genre": last.get("genre", genres[0]),
                "style": last.get("style", styles[0]),
                "model": last.get("model", "mdn"),
                "temperature": last.get("temperature", 1.0),
                "blend": last.get("blend", 1.0),
                "beat_type": last.get("beat_type", "beat")}
    fields = "genre,style,model (lgbm/mdn),temperature,blend (0-1),fill? (y/n)"
    csv_default = "{},{},{},{},{},{}".format(
        defaults["genre"], defaults["style"], defaults["model"],
        defaults["temperature"], defaults["blend"],
        "y" if defaults["beat_type"] == "fill" else "n")
    ok, _, _, _, csv, _ = RPR_GetUserInputs("Dynamics Needed", 6, fields, csv_default, 1024)
    if not ok:
        return None
    genre, style, model, temp, blend, fill = (csv.split(",", 5) + [""] * 6)[:6]
    return {"genre": genre.strip(), "style": style.strip(), "model": model.strip(),
            "temperature": float(temp), "blend": float(blend),
            "beat_type": "fill" if fill.strip().lower().startswith("y") else "beat"}


def _apply(take, notes, velocities):
    for n in notes:
        if n["index"] in velocities:
            RPR_MIDI_SetNote(take, n["index"], -1, -1, -1, -1, -1, -1,
                             velocities[n["index"]], False)
    RPR_MIDI_Sort(take)


def main():
    cfg = _load_config()
    take = _active_take()
    if not take:
        _msg("Dynamics Needed: open a MIDI item in the MIDI editor first.")
        return
    health = _ensure_engine(cfg)
    if not health:
        _msg("Dynamics Needed: could not reach the inference engine. "
             "Run plugin/reaper/setup_reaper.py once, then retry.")
        return

    notes = _read_notes(take)
    if not notes:
        _msg("Dynamics Needed: no notes found in the active take.")
        return

    key = _track_key(take)
    params = _dialog(health, _load_last(key))
    if params is None:
        return
    _save_last(key, params)

    targets = set(dn_core.resolve_target_indices(notes))
    for n in notes:
        n["selected"] = n["index"] in targets
    bpm, time_sig = _tempo_and_sig(take)
    req = dn_core.build_predict_request(
        params["model"], params["style"], params["temperature"], params["blend"],
        params["beat_type"], bpm, time_sig, notes)

    data = json.dumps(req).encode()
    try:
        request = urllib.request.Request(_base_url(cfg) + "/predict", data=data,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as r:
            response = json.loads(r.read())
    except Exception as e:
        _msg("Dynamics Needed: prediction failed: {}".format(e))
        return

    velocities = dn_core.parse_velocities(response)
    RPR_Undo_BeginBlock()
    _apply(take, notes, velocities)
    RPR_Undo_EndBlock("Dynamics Needed: restore velocities", -1)
    _msg("Dynamics Needed: updated {} notes.".format(len(velocities)))


main()
```

- [ ] **Step 3: Manual smoke test (documented; no automated assertion)**

After Task 8's setup has run, in Reaper: create a MIDI item with a flat-velocity drum groove, open it in the MIDI editor, select some notes, run the action `Dynamics Needed`. Verify: the engine starts on first run, the dialog appears with genre/style dropdvalues, and only the selected notes' velocities change (one undo step reverts them).

- [ ] **Step 4: Commit**

```bash
git add plugin/reaper/dynamics_needed.py .gitignore
git commit -m "feat: Reaper ReaScript shell (auto-start engine, dialog, in-place rewrite)"
```

---

### Task 8: One-time setup script + docs

**Files:**
- Create: `plugin/reaper/setup_reaper.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing at runtime.
- Produces: `plugin/reaper/config.local.json` (`venv_python`, `repo_root`, `port`) and installs the action into Reaper. Consumed by `dynamics_needed.py` (Task 7).

- [ ] **Step 1: Write the setup script**

Create `plugin/reaper/setup_reaper.py`:

```python
#!/usr/bin/env python
"""One-time setup for the Dynamics Needed Reaper tool.

Writes plugin/reaper/config.local.json (the venv python + repo root the ReaScript
needs to auto-start the engine) and prints how to register the action in Reaper.

Usage: .venv/bin/python plugin/reaper/setup_reaper.py [--port 8765]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    cfg = {"venv_python": sys.executable, "repo_root": REPO_ROOT, "port": args.port}
    cfg_path = os.path.join(HERE, "config.local.json")
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)

    print("Wrote {}".format(cfg_path))
    print("  venv_python: {}".format(cfg["venv_python"]))
    print("  repo_root  : {}".format(cfg["repo_root"]))
    print()
    print("Register the action in Reaper (one time):")
    print("  Actions -> Show action list -> New action -> Load ReaScript...")
    print("  Select: {}".format(os.path.join(HERE, "dynamics_needed.py")))
    print("  Then bind it to a key or add it to a toolbar / MIDI editor menu.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the setup script runs**

Run: `.venv/bin/python plugin/reaper/setup_reaper.py`
Expected: prints the config path and Reaper registration steps; `plugin/reaper/config.local.json` now exists (and is gitignored). Confirm: `test -f plugin/reaper/config.local.json && echo OK`.

- [ ] **Step 3: Update the README**

In `README.md`, replace the monorepo-layout line
`plugin/        C++ DAW plugin "Dynamics Needed"  (not yet scaffolded)`
with:
`plugin/reaper/ Reaper ReaScript tool "Dynamics Needed" (in-place velocity restore)`

Then add a new section after the "LightGBM native export" section:

```markdown
## Reaper velocity-restoration tool

A Reaper Python ReaScript rewrites the velocities of selected MIDI drum notes in
place, backed by a warm local inference service (`python -m drum_dynamics.serve`)
that serves both the LightGBM and MDN models.

```bash
# 1. Export the MDN inference artifacts (vocab + bpm stats) once
.venv/bin/python ml/scripts/export_mdn.py

# 2. One-time setup: writes config.local.json + prints Reaper registration steps
.venv/bin/python plugin/reaper/setup_reaper.py
```

In Reaper: Actions → New action → Load ReaScript → `plugin/reaper/dynamics_needed.py`,
then bind it. Select notes in the MIDI editor and run the action: the engine
auto-starts on first use, a dialog collects genre/style/model/temperature/blend,
and only the selected notes' velocities are updated (one undo step). The engine
shuts down when Reaper closes.
```

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/python -m pytest ml/tests/ -q`
Expected: PASS (all tests, including the new serve + reaper-core tests).

- [ ] **Step 5: Commit**

```bash
git add plugin/reaper/setup_reaper.py README.md
git commit -m "feat: Reaper tool setup script + README docs"
```

---

## Self-Review

**Spec coverage:**
- §4.1 pure core → Task 3. §4.1 genre-from-style → Task 3 (`meta`) + Task 6. §4.1 blend/clamp → Task 3. §4.1 LGBM path → Task 4 (`LgbmModel`). §4.1 MDN path (temperature) → Task 1 + Task 4 (`MdnModel`). MDN artifact gap → Task 2.
- §4.2 `POST /predict` + `GET /health` + dropdown lists from `lightgbm_features.json` → Task 4 (`Engine.load`/`levels`) + Task 5 (`route`).
- §4.3 lifecycle (`--parent-pid`, `--idle-timeout`, watchdog, log file) → Task 5. Pidfile: **downgraded** — the fixed-port `/health` check plus `start_new_session` process already prevent double-starts; an explicit pidfile added no behavior and was dropped. (Noted intentionally; not a gap.)
- §5.1 selection precedence → Task 7 `_read_notes` (selected flags) + Task 6 `resolve_target_indices`; read-all-context/write-selected → Task 3 (all notes featurized, only selected returned) + Task 7 `_apply`.
- §5.2 extraction (PPQ→sec, bpm/time-sig) → Task 7. §5.3 dialog + per-track memory → Task 7 (`_dialog`, `_load_last`/`_save_last`). §5.4 auto-start + undo-wrapped apply + error UX → Task 7.
- §6 setup + use-time UX → Task 8 + Task 7. §7 testing → tasks 1–6 tests. §8 repo layout + README → tasks 2/7/8.

**Placeholder scan:** No TBD/TODO left. Task 4 Step 3 intentionally shows a scaffold-then-final replacement; the final file contents are fully specified.

**Type consistency:** `predict_all(df) -> np.ndarray` (row order) is consistent across Task 3 (consumer), Task 4 (`LgbmModel`/`MdnModel` producers), and the injected fakes in tests. Request/response contract identical across tasks 3–7. `route(engine, method, path, body) -> (int, dict)` consistent between Task 5 impl and tests. `dn_core` signatures match between Task 6 and Task 7 call sites.

**Known follow-ups (deferred, not blocking):** native combo-box dropdowns in `GetUserInputs` are entered as free text with remembered defaults; a ReaImGui upgrade (true dropdowns filtered by genre via `filter_styles_by_genre`) is a polish task for the implement-and-test phase.
```
