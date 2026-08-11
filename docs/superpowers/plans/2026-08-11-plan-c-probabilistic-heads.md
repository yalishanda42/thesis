# Plan C — Probabilistic Velocity Heads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three probabilistic output heads (Gaussian, MDN, Categorical) on Plan B's transformer backbone so the model predicts a *distribution* over velocity, train each by NLL (warm-started from Plan B), and evaluate whether sampling restores the dynamic spread that point estimation flattens.

**Architecture:** Keep Plan B's `VelocityTransformer` backbone unchanged and parameterize it by a `head` argument. A new, stateless, unit-tested `drumhumanizer/heads.py` interprets the raw per-token head output as distribution parameters and provides masked NLL, sampling, deterministic point-readout, and a comparable per-bin log-probability. One training script trains any head (warm-start + resumable, reusing Plan B's machinery) and evaluates with three metric families (NLL, deterministic-readout metrics, sampled-distribution match).

**Tech Stack:** Python 3.12, PyTorch 2.13 (Apple MPS), `numpy`/`pandas`, `scipy` (Wasserstein), `matplotlib`, existing `drumhumanizer` package (`seqdata`, `model`, `metrics`, `voicemap`). `torch.distributions` is used **in tests only** as an oracle.

## Global Constraints

- **Same backbone, swap head + loss only (design §3):** reuse Plan B `VelocityTransformer` submodules (`voice_emb`, `genre_emb`, `input_proj`, `pos_enc`, `dropout`, `encoder`) with identical names so Plan B checkpoints load. Only `self.head` changes.
- **Back-compatible default:** `head="deterministic"` must reproduce Plan B exactly — `self.head = nn.Linear(d, 1)` and `forward` returns `[B, L]` (squeezed). Existing `scripts/train_transformer.py`, `notebooks/audition_models.ipynb`, and `data/processed/transformer_best.pt` must keep working.
- **Binning (design §2):** `N_BINS = 32` uniform bins over velocity `[0, 128)`, `BIN_WIDTH = 4`. `bin(v) = clamp(v // 4, 0, 31)`; `center(b) = b*4 + 2`.
- **MDN (design §2/§3):** `MDN_K = 5`; `σ = softplus(σ_raw) + 1.0`. Gaussian: `σ = softplus(σ_raw) + 1e-3`. All mixture math in log-space via `logsumexp`.
- **Losses:** masked NLL over non-pad tokens only (`pad_mask` True = pad).
- **Warm-start (default on):** initialize backbone from `data/processed/transformer_best.pt` (`best_model` state, `head.*` keys removed, `strict=False`); fine-tune all params with the new head.
- **Reproducibility:** seed `42` everywhere including sampling. Sampling runs on **CPU** with a seeded `torch.Generator` (MPS generator support is incomplete) — move head outputs to CPU before sampling.
- **Device:** `mps` if available else `cpu`. Tests run on CPU.
- **Inputs:** the Plan A parquet (`data/processed/egmd_tabular_{train,validation,test}.parquet`), the transformer sidecar (`data/processed/transformer_meta.json`), and the Plan B checkpoint (`data/processed/transformer_best.pt`). Baselines for the combined table: `docs/plan_a/metrics.json`, `docs/plan_b/metrics.json`.
- **Run commands** use the repo venv: `.venv/bin/python`, `.venv/bin/python -m pytest`.

---

## File Structure

**Create:**
- `drumhumanizer/heads.py` — binning + Gaussian/MDN/Categorical parameter parsing, `logprob`, `nll`, `bin_logprob`, `point`, `sample`, and dispatch + `head_output_dim`.
- `scripts/train_head.py` — train one probabilistic head (warm-start, resumable), evaluate, write `docs/plan_c/<head>/`.
- `tests/test_heads.py`, `tests/test_prob_metrics.py`.
- `docs/plan_c/` — per-head results + combined `results.md`.

**Modify:**
- `drumhumanizer/model.py` — `VelocityTransformer(head=...)` + `warm_start_backbone(...)`.
- `drumhumanizer/metrics.py` — add `wasserstein1d`, `hist_intersection`.
- `drumhumanizer/__init__.py` — lazily export `warm_start_backbone` (torch path) and eagerly export the two new metrics.

---

### Task 1: Probabilistic heads module

**Files:**
- Create: `drumhumanizer/heads.py`
- Test: `tests/test_heads.py`

**Interfaces:**
- Produces (consumed by Tasks 2 & 4):
  - Constants `N_BINS=32`, `BIN_WIDTH=4`, `MDN_K=5`, `HEAD_OUTPUT_DIM: dict[str,int]`.
  - `head_output_dim(head_type) -> int`.
  - `bin_centers(device=None) -> Tensor[N_BINS]`; `velocity_to_bin(y) -> LongTensor`.
  - `logprob(head_type, raw, y) -> Tensor` (per-token; the head's own log-density/log-mass).
  - `nll(head_type, raw, y, pad_mask) -> scalar` (masked mean of `-logprob`).
  - `bin_logprob(head_type, raw, y) -> Tensor` (per-token log-prob of the true velocity's 32-bin — comparable across heads).
  - `point(head_type, raw) -> Tensor` (deterministic readout).
  - `sample(head_type, raw, generator=None) -> Tensor` (velocities clamped to `[0,127]`).
  - Head raw shapes: gaussian `[...,2]`, mdn `[...,3K]` (layout `[π_logits(K), μ(K), σ_raw(K)]`), categorical `[...,32]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_heads.py
import math
import torch
import torch.distributions as D

from drumhumanizer import heads


def test_output_dims():
    assert heads.head_output_dim("deterministic") == 1
    assert heads.head_output_dim("gaussian") == 2
    assert heads.head_output_dim("mdn") == 3 * heads.MDN_K
    assert heads.head_output_dim("categorical") == heads.N_BINS


def test_velocity_to_bin_and_centers():
    y = torch.tensor([0.0, 3.9, 4.0, 127.0, 200.0])
    assert heads.velocity_to_bin(y).tolist() == [0, 0, 1, 31, 31]
    c = heads.bin_centers()
    assert c.shape == (heads.N_BINS,)
    assert c[0].item() == 2.0 and c[-1].item() == 126.0


def test_gaussian_logprob_matches_torch_distribution():
    raw = torch.tensor([[2.5, 0.7]])          # mu=2.5, sigma=softplus(0.7)+1e-3
    y = torch.tensor([3.0])
    mu, sigma = heads._gauss_params(raw)
    oracle = D.Normal(mu, sigma).log_prob(y)
    assert torch.allclose(heads.logprob("gaussian", raw, y), oracle, atol=1e-6)
    assert torch.allclose(heads.point("gaussian", raw), mu)


def test_mdn_logprob_matches_mixture_oracle():
    K = heads.MDN_K
    torch.manual_seed(0)
    raw = torch.randn(4, K * 3)
    y = torch.rand(4) * 127
    log_pi, mu, sigma = heads._mdn_params(raw)
    mix = D.MixtureSameFamily(D.Categorical(logits=log_pi), D.Normal(mu, sigma))
    assert torch.allclose(heads.logprob("mdn", raw, y), mix.log_prob(y), atol=1e-5)
    # point readout is the mixture mean
    assert torch.allclose(heads.point("mdn", raw), (log_pi.exp() * mu).sum(-1), atol=1e-5)


def test_categorical_nll_matches_cross_entropy():
    torch.manual_seed(0)
    raw = torch.randn(6, heads.N_BINS)
    y = torch.rand(6) * 127
    b = heads.velocity_to_bin(y)
    import torch.nn.functional as F
    pad = torch.zeros(6, dtype=torch.bool)
    assert torch.allclose(heads.nll("categorical", raw, y, pad),
                          F.cross_entropy(raw, b), atol=1e-6)


def test_bin_logprob_is_comparable_and_bounded():
    y = torch.tensor([50.0])
    for raw, ht in [(torch.tensor([[50.0, 0.5]]), "gaussian"),
                    (torch.randn(1, 3 * heads.MDN_K), "mdn"),
                    (torch.randn(1, heads.N_BINS), "categorical")]:
        lp = heads.bin_logprob(ht, raw, y)
        assert lp.shape == (1,)
        assert (lp <= 1e-6).all()            # log of a probability mass <= 0


def test_nll_masks_padding():
    raw = torch.zeros(1, 3, 2)               # gaussian
    y = torch.tensor([[50.0, 50.0, 9999.0]])
    pad = torch.tensor([[False, False, True]])
    # padded token (absurd y) must not affect the loss
    loss_masked = heads.nll("gaussian", raw, y, pad)
    loss_first_two = heads.nll("gaussian", raw[:, :2], y[:, :2], pad[:, :2])
    assert torch.allclose(loss_masked, loss_first_two, atol=1e-6)


def test_sample_is_in_range_and_reproducible():
    torch.manual_seed(0)
    for ht, raw in [("gaussian", torch.randn(50, 2)),
                    ("mdn", torch.randn(50, 3 * heads.MDN_K)),
                    ("categorical", torch.randn(50, heads.N_BINS))]:
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        s1 = heads.sample(ht, raw, generator=g1)
        s2 = heads.sample(ht, raw, generator=g2)
        assert s1.shape == (50,)
        assert (s1 >= 0).all() and (s1 <= 127).all()
        assert torch.allclose(s1, s2)        # seeded -> reproducible
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_heads.py -q`
Expected: FAIL (module `drumhumanizer.heads` not found).

- [ ] **Step 3: Write the implementation**

```python
# drumhumanizer/heads.py
"""Probabilistic output heads for the velocity model (design Plan C §3-§5).

Each head interprets the raw per-token output of ``VelocityTransformer`` as the
parameters of a distribution over velocity. Functions are stateless and keyed by
``head_type``; the backbone stays head-agnostic. Losses are masked NLL.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

N_BINS = 32
BIN_WIDTH = 128 // N_BINS            # 4
MDN_K = 5
_SIGMA_FLOOR_MDN = 1.0
_SIGMA_FLOOR_GAUSS = 1e-3
_LOG_2PI = math.log(2 * math.pi)
_SQRT2 = math.sqrt(2.0)

HEAD_OUTPUT_DIM = {"deterministic": 1, "gaussian": 2, "mdn": 3 * MDN_K, "categorical": N_BINS}


def head_output_dim(head_type: str) -> int:
    return HEAD_OUTPUT_DIM[head_type]


def bin_centers(device=None) -> torch.Tensor:
    return torch.arange(N_BINS, dtype=torch.float, device=device) * BIN_WIDTH + BIN_WIDTH / 2


def velocity_to_bin(y: torch.Tensor) -> torch.Tensor:
    return torch.clamp((y / BIN_WIDTH).long(), 0, N_BINS - 1)


# ── Gaussian ────────────────────────────────────────────────────────────────
def _gauss_params(raw):
    mu = raw[..., 0]
    sigma = F.softplus(raw[..., 1]) + _SIGMA_FLOOR_GAUSS
    return mu, sigma


def _normal_logprob(y, mu, sigma):
    return -0.5 * _LOG_2PI - torch.log(sigma) - 0.5 * ((y - mu) / sigma) ** 2


def _normal_cdf(x, mu, sigma):
    return 0.5 * (1.0 + torch.erf((x - mu) / (sigma * _SQRT2)))


# ── MDN ─────────────────────────────────────────────────────────────────────
def _mdn_params(raw):
    r = raw.view(*raw.shape[:-1], 3, MDN_K)
    log_pi = torch.log_softmax(r[..., 0, :], dim=-1)
    mu = r[..., 1, :]
    sigma = F.softplus(r[..., 2, :]) + _SIGMA_FLOOR_MDN
    return log_pi, mu, sigma


# ── per-token log-probability (each head's own measure) ──────────────────────
def logprob(head_type, raw, y):
    if head_type == "gaussian":
        mu, sigma = _gauss_params(raw)
        return _normal_logprob(y, mu, sigma)
    if head_type == "mdn":
        log_pi, mu, sigma = _mdn_params(raw)
        comp = _normal_logprob(y.unsqueeze(-1), mu, sigma)      # [..., K]
        return torch.logsumexp(log_pi + comp, dim=-1)
    if head_type == "categorical":
        logp = torch.log_softmax(raw, dim=-1)
        b = velocity_to_bin(y).unsqueeze(-1)
        return logp.gather(-1, b).squeeze(-1)
    raise ValueError(f"unknown head_type {head_type!r}")


def nll(head_type, raw, y, pad_mask):
    lp = logprob(head_type, raw, y)
    keep = ~pad_mask
    return -(lp[keep]).mean()


def bin_logprob(head_type, raw, y):
    """log P(true velocity's 32-bin) under the head — comparable across heads."""
    if head_type == "categorical":
        return logprob("categorical", raw, y)
    b = velocity_to_bin(y)
    lo = (b * BIN_WIDTH).float()
    hi = lo + BIN_WIDTH
    if head_type == "gaussian":
        mu, sigma = _gauss_params(raw)
        mass = _normal_cdf(hi, mu, sigma) - _normal_cdf(lo, mu, sigma)
    elif head_type == "mdn":
        log_pi, mu, sigma = _mdn_params(raw)
        cdf_hi = _normal_cdf(hi.unsqueeze(-1), mu, sigma)
        cdf_lo = _normal_cdf(lo.unsqueeze(-1), mu, sigma)
        mass = (log_pi.exp() * (cdf_hi - cdf_lo)).sum(-1)
    else:
        raise ValueError(f"unknown head_type {head_type!r}")
    return torch.log(mass.clamp_min(1e-12))


def point(head_type, raw):
    if head_type == "deterministic":
        return raw.squeeze(-1) if raw.dim() > 2 else raw
    if head_type == "gaussian":
        return _gauss_params(raw)[0]
    if head_type == "mdn":
        log_pi, mu, _ = _mdn_params(raw)
        return (log_pi.exp() * mu).sum(-1)
    if head_type == "categorical":
        p = torch.softmax(raw, dim=-1)
        return (p * bin_centers(raw.device)).sum(-1)
    raise ValueError(f"unknown head_type {head_type!r}")


def sample(head_type, raw, generator=None):
    if head_type == "gaussian":
        mu, sigma = _gauss_params(raw)
        eps = torch.randn(mu.shape, generator=generator, device=mu.device)
        y = mu + sigma * eps
    elif head_type == "mdn":
        log_pi, mu, sigma = _mdn_params(raw)
        flat_pi = log_pi.exp().reshape(-1, MDN_K)
        k = torch.multinomial(flat_pi, 1, generator=generator).reshape(mu.shape[:-1])
        muk = mu.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        sigk = sigma.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        eps = torch.randn(muk.shape, generator=generator, device=muk.device)
        y = muk + sigk * eps
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

Run: `.venv/bin/python -m pytest tests/test_heads.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add drumhumanizer/heads.py tests/test_heads.py
git commit -m "feat: add probabilistic velocity heads (gaussian, mdn, categorical)"
```

---

### Task 2: Head-parameterized model + warm-start

**Files:**
- Modify: `drumhumanizer/model.py`
- Modify: `drumhumanizer/__init__.py`
- Test: `tests/test_model.py` (append)

**Interfaces:**
- Consumes: `drumhumanizer.heads.head_output_dim`.
- Produces:
  - `VelocityTransformer(..., head="deterministic")` — new `head` kwarg; `self.head = nn.Linear(d_model, head_output_dim(head))`, `self.head_type = head`. `forward` returns `[B, L]` for `deterministic` (unchanged), else `[B, L, head_output_dim(head)]`.
  - `warm_start_backbone(model, ckpt_path) -> (missing, unexpected)` — load a Plan-B-style checkpoint's `best_model` backbone weights (dropping `head.*`) into `model` with `strict=False`.

- [ ] **Step 1: Write the failing test (append to `tests/test_model.py`)**

```python
# tests/test_model.py  (append)
import os
import tempfile

from drumhumanizer.heads import head_output_dim, N_BINS, MDN_K
from drumhumanizer.model import warm_start_backbone


def test_head_output_shapes():
    B, L = 2, 5
    voice = torch.randint(0, 14, (B, L)); genre = torch.randint(0, 5, (B, L))
    num = torch.randn(B, L, len(NUMERIC_FEATURES)); pad = torch.zeros(B, L, dtype=torch.bool)
    assert VelocityTransformer(n_genres=5, d_model=32, n_heads=4, n_layers=1, dim_ff=64,
                               max_len=16, head="deterministic")(voice, genre, num, pad).shape == (B, L)
    assert VelocityTransformer(n_genres=5, d_model=32, n_heads=4, n_layers=1, dim_ff=64,
                               max_len=16, head="gaussian")(voice, genre, num, pad).shape == (B, L, 2)
    assert VelocityTransformer(n_genres=5, d_model=32, n_heads=4, n_layers=1, dim_ff=64,
                               max_len=16, head="mdn")(voice, genre, num, pad).shape == (B, L, 3 * MDN_K)
    assert VelocityTransformer(n_genres=5, d_model=32, n_heads=4, n_layers=1, dim_ff=64,
                               max_len=16, head="categorical")(voice, genre, num, pad).shape == (B, L, N_BINS)


def test_warm_start_loads_backbone_drops_head():
    torch.manual_seed(0)
    det = VelocityTransformer(n_genres=5, d_model=32, n_heads=4, n_layers=1, dim_ff=64, max_len=16)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ck.pt")
        torch.save({"best_model": det.state_dict()}, p)
        gauss = VelocityTransformer(n_genres=5, d_model=32, n_heads=4, n_layers=1, dim_ff=64,
                                    max_len=16, head="gaussian")
        missing, unexpected = warm_start_backbone(gauss, p)
        # backbone weights transferred exactly; only the head is fresh/mismatched
        assert torch.equal(gauss.input_proj.weight, det.input_proj.weight)
        assert torch.equal(gauss.voice_emb.weight, det.voice_emb.weight)
        assert all(k.startswith("head.") for k in missing)      # only head missing
        assert unexpected == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_model.py -k "head_output_shapes or warm_start" -q`
Expected: FAIL (`head` kwarg / `warm_start_backbone` not defined).

- [ ] **Step 3: Modify `drumhumanizer/model.py`**

Change the import block and `__init__`/`forward`, and add `warm_start_backbone`:

```python
# at top, add:
from .heads import head_output_dim
```

Replace the head construction in `__init__` (`self.head = nn.Linear(d_model, 1)`) with:

```python
        self.head_type = head
        self.head = nn.Linear(d_model, head_output_dim(head))
```

Add `head="deterministic"` as the **last** parameter of `__init__` (after `max_len=512`).

Replace the final line of `forward` (`return self.head(x).squeeze(-1)`) with:

```python
        out = self.head(x)                       # [B, L, out_dim]
        if self.head_type == "deterministic":
            return out.squeeze(-1)               # [B, L] — Plan B behavior
        return out                               # [B, L, out_dim]
```

Append at module end:

```python
def warm_start_backbone(model, ckpt_path):
    """Load Plan-B backbone weights (dropping the head) into ``model``.

    Returns ``(missing, unexpected)`` from ``load_state_dict(strict=False)``;
    ``missing`` should be exactly the new head's parameters.
    """
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck.get("best_model", ck)
    backbone = {k: v for k, v in state.items() if not k.startswith("head.")}
    return model.load_state_dict(backbone, strict=False)
```

- [ ] **Step 4: Export from the package**

In `drumhumanizer/__init__.py`, add `"warm_start_backbone"` to the `_LAZY` dict mapping to `"model"`, and add `"warm_start_backbone"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_model.py -q`
Expected: PASS (all model tests, including the original Plan B `test_forward_output_shape` and `test_padding_invariance`).

- [ ] **Step 6: Commit**

```bash
git add drumhumanizer/model.py drumhumanizer/__init__.py tests/test_model.py
git commit -m "feat: parameterize VelocityTransformer by head + warm-start helper"
```

---

### Task 3: Distribution-match metrics

**Files:**
- Modify: `drumhumanizer/metrics.py`
- Modify: `drumhumanizer/__init__.py`
- Test: `tests/test_prob_metrics.py`

**Interfaces:**
- Produces:
  - `wasserstein1d(a, b) -> float` — 1-D earth-mover distance between two velocity sample sets (`scipy.stats.wasserstein_distance`).
  - `hist_intersection(a, b, bins=32, lo=0, hi=128) -> float` — histogram-intersection similarity in `[0,1]` (1 = identical distributions) over fixed bins.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prob_metrics.py
import numpy as np

from drumhumanizer.metrics import wasserstein1d, hist_intersection


def test_wasserstein_zero_for_identical():
    a = np.array([10.0, 20, 30, 40])
    assert wasserstein1d(a, a) == 0.0
    assert wasserstein1d(a, a + 5) == 5.0        # constant shift -> EMD = shift


def test_hist_intersection_bounds():
    rng = np.random.RandomState(0)
    a = rng.uniform(0, 128, 5000)
    assert np.isclose(hist_intersection(a, a), 1.0)          # identical -> 1
    disjoint = hist_intersection(np.zeros(100), np.full(100, 127.0))
    assert disjoint < 0.05                                    # no overlap -> ~0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_prob_metrics.py -q`
Expected: FAIL (`wasserstein1d` not defined).

- [ ] **Step 3: Add to `drumhumanizer/metrics.py`**

Add the import at the top (next to the existing scipy import):

```python
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
```

Append:

```python
def wasserstein1d(a, b) -> float:
    """1-D earth-mover distance between two velocity sample sets."""
    return float(wasserstein_distance(np.asarray(a, float), np.asarray(b, float)))


def hist_intersection(a, b, bins: int = 32, lo: float = 0.0, hi: float = 128.0) -> float:
    """Histogram-intersection similarity in [0, 1] over fixed bins (1 = identical)."""
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(np.asarray(a, float), bins=edges, density=False)
    pb, _ = np.histogram(np.asarray(b, float), bins=edges, density=False)
    pa = pa / max(pa.sum(), 1)
    pb = pb / max(pb.sum(), 1)
    return float(np.minimum(pa, pb).sum())
```

- [ ] **Step 4: Export** — add `from .metrics import ... wasserstein1d, hist_intersection` to the existing metrics import line in `drumhumanizer/__init__.py` and both names to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_prob_metrics.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add drumhumanizer/metrics.py drumhumanizer/__init__.py tests/test_prob_metrics.py
git commit -m "feat: add wasserstein + histogram-intersection distribution metrics"
```

---

### Task 4: Train & evaluate each probabilistic head

**Files:**
- Create: `scripts/train_head.py`
- Create (output): `docs/plan_c/<head>/metrics.json` + figures for `head in {gaussian, mdn, categorical}`, and `docs/plan_c/results.md`.

**Interfaces:**
- Consumes: `seqdata` tensors, `VelocityTransformer(head=...)`, `warm_start_backbone`, `heads.*`, `metrics.evaluate/wasserstein1d/hist_intersection`; Plan A/B metrics for the combined table.
- Produces per head: `metrics.json` with keys `native_nll`, `discretized_nll`, `deterministic_readout` (an `evaluate()` dict on the point readout), `sampled` (an `evaluate()` dict on a sampled prediction), `sampled_wasserstein`, `sampled_hist_intersection`, `best_epoch`, `val_nll_curve`.

- [ ] **Step 1: Write the script**

```python
# scripts/train_head.py
"""Train one probabilistic head on the Plan B backbone (warm-start, resumable)
and evaluate: NLL, deterministic-readout metrics, and sampled-distribution match.

Usage: .venv/bin/python scripts/train_head.py --head {gaussian,mdn,categorical}
       [--epochs N] [--run-epochs N] [--resume] [--eval-only] [--no-warm-start]
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
from drumhumanizer import heads                                                   # noqa: E402
from drumhumanizer.metrics import evaluate, wasserstein1d, hist_intersection      # noqa: E402
from drumhumanizer.model import VelocityTransformer, warm_start_backbone          # noqa: E402
from drumhumanizer.seqdata import (                                               # noqa: E402
    build_genre_vocab, bpm_stats, build_split_tensors, scatter_predictions,
)

PROC = os.path.join("data", "processed")
OUT = os.path.join("docs", "plan_c")
PLAN_A = os.path.join("docs", "plan_a", "metrics.json")
PLAN_B = os.path.join("docs", "plan_b", "metrics.json")
BACKBONE = os.path.join(PROC, "transformer_best.pt")
SEED = 42
KEYS = ["voice_idx", "genre_idx", "num_feats", "target", "pad_mask", "row_idx"]


def _device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _free(device):
    if device.type == "mps":
        torch.mps.empty_cache()


def _load(split):
    return pd.read_parquet(os.path.join(PROC, f"egmd_tabular_{split}.parquet"))


def _loader(t, bs, shuffle):
    return DataLoader(TensorDataset(*[t[k] for k in KEYS]), batch_size=bs, shuffle=shuffle)


def _val_nll(model, loader, head, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for voice, genre, num, target, pad, row in loader:
            raw = model(voice.to(device), genre.to(device), num.to(device), pad.to(device))
            keep = (~pad.to(device))
            lp = heads.logprob(head, raw, target.to(device))
            tot += (-lp[keep]).sum().item()
            n += int(keep.sum().item())
    return tot / n


def _predict(model, loader, head, device, n_rows, seed):
    """Return (point_pred, sampled_pred) aligned to df rows."""
    model.eval()
    point = np.zeros(n_rows); samp = np.zeros(n_rows)
    gen = torch.Generator().manual_seed(seed)      # CPU generator for reproducible sampling
    with torch.no_grad():
        for voice, genre, num, target, pad, row in loader:
            raw = model(voice.to(device), genre.to(device), num.to(device), pad.to(device)).cpu()
            point += scatter_predictions(row, heads.point(head, raw), pad, n_rows)
            samp += scatter_predictions(row, heads.sample(head, raw, generator=gen), pad, n_rows)
    return point, samp


def _test_nlls(model, loader, head, device):
    model.eval()
    nat_tot = dis_tot = 0.0; n = 0
    with torch.no_grad():
        for voice, genre, num, target, pad, row in loader:
            raw = model(voice.to(device), genre.to(device), num.to(device), pad.to(device))
            keep = (~pad.to(device))
            y = target.to(device)
            nat_tot += (-heads.logprob(head, raw, y)[keep]).sum().item()
            dis_tot += (-heads.bin_logprob(head, raw, y)[keep]).sum().item()
            n += int(keep.sum().item())
    return nat_tot / n, dis_tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, choices=["gaussian", "mdn", "categorical"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--run-epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-warm-start", action="store_true")
    args = ap.parse_args()

    head = args.head
    out_dir = os.path.join(OUT, head)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(PROC, f"head_{head}.pt")
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = _device(); print(f"device: {device}  head: {head}")

    train, val, test = _load("train"), _load("validation"), _load("test")
    genre_vocab = build_genre_vocab(train)
    bpm_mean, bpm_std = bpm_stats(train)
    t0 = time.time()
    tr = build_split_tensors(train, genre_vocab, bpm_mean, bpm_std)
    va = build_split_tensors(val, genre_vocab, bpm_mean, bpm_std)
    te = build_split_tensors(test, genre_vocab, bpm_mean, bpm_std)
    print(f"windows tr/va/te: {tr['target'].shape[0]}/{va['target'].shape[0]}/{te['target'].shape[0]} "
          f"({time.time()-t0:.0f}s)")
    tr_l = _loader(tr, args.batch_size, True)
    va_l = _loader(va, args.batch_size, False)
    te_l = _loader(te, args.batch_size, False)

    model = VelocityTransformer(n_genres=len(genre_vocab) + 1, head=head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_epoch, curve = -1, []
    if args.eval_only:
        model.load_state_dict(torch.load(ckpt, map_location=device)["best_model"])
    else:
        best_val, bad, start, best_state = float("inf"), 0, 0, None
        if args.resume and os.path.exists(ckpt):
            ck = torch.load(ckpt, map_location=device)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            best_val, best_epoch = ck["best_val"], ck["best_epoch"]
            start, curve, bad, best_state = ck["epochs_done"], list(ck["curve"]), ck["bad"], ck["best_model"]
            print(f"resumed at epoch {start} (best_val {best_val:.4f})")
        elif not args.no_warm_start and os.path.exists(BACKBONE):
            missing, unexpected = warm_start_backbone(model, BACKBONE)
            print(f"warm-started backbone from {BACKBONE} (fresh: {list(missing)})")

        ran = 0
        for epoch in range(start + 1, args.epochs + 1):
            if ran >= args.run_epochs:
                print(f"run-epochs budget reached"); break
            model.train(); te0 = time.time()
            for voice, genre, num, target, pad, row in tr_l:
                opt.zero_grad()
                raw = model(voice.to(device), genre.to(device), num.to(device), pad.to(device))
                loss = heads.nll(head, raw, target.to(device), pad.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # MDN stability
                opt.step()
            _free(device)
            val_nll = _val_nll(model, va_l, head, device); _free(device)
            curve.append(val_nll); ran += 1
            print(f"epoch {epoch:2d}  val_nll {val_nll:.4f}  ({time.time()-te0:.0f}s)")
            if val_nll < best_val - 1e-4:
                best_val, best_epoch, bad = val_nll, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            torch.save({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "opt": opt.state_dict(), "best_model": best_state, "best_val": best_val,
                        "best_epoch": best_epoch, "epochs_done": epoch, "curve": curve, "bad": bad}, ckpt)
            if bad >= args.patience:
                print(f"early stopping at epoch {epoch}"); break
        if best_state is not None:
            model.load_state_dict(best_state)

    # ── evaluate ──────────────────────────────────────────────────────────
    native_nll, discretized_nll = _test_nlls(model, te_l, head, device)
    point, samp = _predict(model, te_l, head, device, len(test), SEED)
    true = test["velocity"].to_numpy(float)
    results = {
        "native_nll": native_nll,
        "discretized_nll": discretized_nll,
        "deterministic_readout": evaluate(test, point),
        "sampled": evaluate(test, samp),
        "sampled_wasserstein": wasserstein1d(samp, true),
        "sampled_hist_intersection": hist_intersection(samp, true),
        "best_epoch": best_epoch,
        "val_nll_curve": curve,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n[{head}] native_nll {native_nll:.4f}  discretized_nll {discretized_nll:.4f}")
    print(f"  point   : MAE {results['deterministic_readout']['mae']:.3f}  "
          f"std_ratio {results['deterministic_readout']['global_std_ratio']:.3f}")
    print(f"  sampled : MAE {results['sampled']['mae']:.3f}  "
          f"std_ratio {results['sampled']['global_std_ratio']:.3f}  "
          f"W1 {results['sampled_wasserstein']:.3f}  hist∩ {results['sampled_hist_intersection']:.3f}")

    # figures: true vs sampled velocity histogram; val NLL curve
    plt.figure(figsize=(7, 4))
    plt.hist(true, bins=32, range=(0, 128), alpha=0.5, density=True, label="true")
    plt.hist(samp, bins=32, range=(0, 128), alpha=0.5, density=True, label=f"{head} sampled")
    plt.xlabel("velocity"); plt.ylabel("density"); plt.legend()
    plt.title(f"{head}: sampled vs true velocity distribution")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig_hist.png"), dpi=120); plt.close()

    if curve:
        plt.figure(figsize=(6, 4)); plt.plot(range(1, len(curve) + 1), curve, marker="o")
        plt.xlabel("epoch"); plt.ylabel("val NLL"); plt.title(f"{head}: validation NLL")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig_val_nll.png"), dpi=120); plt.close()

    print(f"wrote results to {out_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the pipeline (1 epoch, tiny)**

Run: `.venv/bin/python scripts/train_head.py --head gaussian --epochs 1 --run-epochs 1`
Expected: warm-start message, one `val_nll` line, an evaluation block, and files under `docs/plan_c/gaussian/`. (This is a full-data epoch on MPS, ~5-7 min; if that is too long for a smoke check, temporarily point `_load` at `.head(200000)` — revert before the real runs.)

- [ ] **Step 3: Train all three heads**

Warm-started from the Plan B backbone, each head should converge in a few epochs. Run each (resume after any interruption — every epoch is checkpointed):

```bash
.venv/bin/python scripts/train_head.py --head gaussian
.venv/bin/python scripts/train_head.py --head mdn
.venv/bin/python scripts/train_head.py --head categorical
```

If a run is killed, re-run the same command with `--resume` (loses at most one epoch). Each is ~5-7 min/epoch on MPS.

- [ ] **Step 4: Verify the sampling payoff**

For each head confirm in `metrics.json` / stdout that the **sampled** readout has a **`global_std_ratio` much closer to 1.0** than both the head's own **point** readout and the Plan B deterministic model (~0.69) — i.e. sampling restores dynamic spread — and a high `sampled_hist_intersection`. Point-readout MAE is expected to be similar to Plan B (mean-like), while sampled MAE is a little higher but with far better spread/histogram match. If sampling does **not** improve std ratio / histogram intersection over the deterministic model, stop and report.

- [ ] **Step 5: Write the combined results note**

Create `docs/plan_c/results.md` comparing deterministic (Plan B) / Gaussian / MDN / Categorical across: native & discretized NLL (categorical vs the CDF-integrated continuous heads), deterministic-readout MAE/corr, and — the payoff — sampled std ratio, Wasserstein, and histogram intersection. State which head best reproduces the true velocity distribution and whether the probabilistic heads earn their complexity over the deterministic transformer and LightGBM. Reference each head's `fig_hist.png`.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_head.py docs/plan_c/
git commit -m "feat: train and evaluate probabilistic velocity heads (Plan C)"
```

---

### Task 5 (optional): Audition sampled humanizations

**Files:**
- Modify: `notebooks/audition_models.ipynb`

**Interfaces:**
- Consumes: `heads.point`/`heads.sample`, a `head_*.pt` checkpoint.

- [ ] **Step 1: Add a probabilistic-model branch**

Extend the notebook's model-loading cell with `MODEL in {"gaussian","mdn","categorical"}`: load `data/processed/head_<MODEL>.pt` into `VelocityTransformer(head=MODEL)`, and define `predict(feats)` to return **both** a point readout and a sampled readout (build the sequence tensor as for the transformer, run the model, apply `heads.point` / `heads.sample`). Add a `READOUT = "sample"` config flag (`"sample"` or `"mean"`).

- [ ] **Step 2: Verify it renders**

Execute the notebook for one probabilistic head from `notebooks/` (fresh kernel) and confirm audio + rolls appear with no errors. A *sampled* audition should sound more dynamically varied than the deterministic one.

- [ ] **Step 3: Commit**

```bash
git add notebooks/audition_models.ipynb
git commit -m "feat: audition sampled humanizations from probabilistic heads"
```

---

## Self-Review notes

- **Design §2 coverage:** 32-bin categorical, K=5 MDN with σ floor (Task 1). ✓
- **Design §3 (same backbone, pluggable head, warm-start):** Task 2 (`head` kwarg keeps deterministic default + checkpoint compatibility; `warm_start_backbone` drops `head.*`, `strict=False`). ✓
- **Design §4 losses:** Gaussian NLL, MDN logsumexp mixture NLL, categorical cross-entropy — masked (Task 1 `nll`). ✓
- **Design §5 readouts:** `point` (μ / mixture-mean / expected-bin) and `sample` (all clamped 0-127, reproducible seeded CPU generator) (Task 1). ✓
- **Design §6 evaluation:** native NLL, comparable discretized NLL (CDF bin mass for continuous heads), deterministic-readout `evaluate()`, sampled `evaluate()` + Wasserstein + histogram intersection (Tasks 3-4). ✓
- **Back-compat:** deterministic default preserves Plan B forward shape and checkpoint loading; original `test_model.py` tests must still pass (Task 2). ✓
- **Compute:** warm-start + resumable per-epoch checkpoint (`head_<head>.pt`) + grad clipping for MDN stability (Task 4). ✓
- **Type consistency:** `head_output_dim` drives both the model head size (Task 2) and the raw-output layout parsed in `heads.py` (Task 1); `KEYS`/tensor dict match `seqdata.build_split_tensors`. ✓
- **Placeholder scan:** none; all steps carry runnable code. ✓
