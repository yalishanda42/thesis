# Design: Plan C — Probabilistic Velocity Heads

**Date:** 2026-08-11
**Status:** Approved design; ready for implementation planning
**Builds on:** [drum-velocity design](2026-08-11-drum-velocity-humanization-design.md) §6 (model progression) and §9 (deferred to Plan C); Plan A (tabular) and Plan B (deterministic transformer), both merged.

---

## 1. Motivation

Plan B's deterministic transformer minimizes MAE by predicting one velocity per note and, like any point estimator, is pulled toward the conditional **mean** — flattening dynamics (Plan B: global std ratio 0.69–0.90; the mean of a ghost/accent note ≈ 60 is musically wrong). Humanization is one-to-many: the *variance* is the humanity. Plan C keeps the exact same backbone and swaps only the **output head + loss** so the model predicts a **distribution** `p(velocity | context)` and we **sample** from it to restore realistic, multimodal dynamics.

All heads are trained by maximum likelihood (minimize NLL = cross-entropy to the true conditional), which differ only in how flexible `q(y|x)` is allowed to be:

| head | `q(y\|x)` | params/token | loss | multimodal? |
|------|-----------|--------------|------|-------------|
| Deterministic (Plan B) | Gaussian, σ≡1 | 1 (μ) | MAE/MSE | no |
| Gaussian | one Gaussian | 2 (μ, σ) | Gaussian NLL | no |
| MDN | K Gaussians | 3K (π, μ, σ), K=5 | mixture NLL | yes (continuous) |
| Categorical | binned pmf | B (B=32 bins) | cross-entropy | yes (non-parametric) |

The three probabilistic heads are **mutually-exclusive model variants** (not an ensemble): train each separately, compare, use one at inference.

## 2. Scope

Implement **all three** heads (Gaussian, MDN, Categorical) on the shared Plan B backbone, train each, and evaluate against the deterministic baseline and Plans A/B.

**Categorical bins:** **32 uniform bins** over velocity `[0, 128)` (width 4), not 128. With 11M notes, 32 bins keep resolution while respecting ordinal locality and needing less per-class data. Bin of velocity `v` = `min(v // 4, 31)`; bin center = `bin*4 + 2`.

**MDN components:** **K = 5**. σ constrained by `softplus + floor` (floor = 1.0 velocity unit) to prevent variance collapse / NaNs.

## 3. Architecture — same backbone, pluggable head

Reuse Plan B's `VelocityTransformer` **exactly** (voice/genre embeddings → input projection → sinusoidal positional encoding → `TransformerEncoder`), keeping submodule names identical so Plan B checkpoints load. Parameterize the model by `head`:

- `head="deterministic"` → `nn.Linear(d, 1)` (unchanged; preserves Plan B behavior and checkpoint compatibility — default).
- `head="gaussian"` → `nn.Linear(d, 2)` → `(μ, σ_raw)`; `σ = softplus(σ_raw) + 1e-3`.
- `head="mdn"` → `nn.Linear(d, 3K)` → split `(π_logits, μ, σ_raw)`; `π = softmax(π_logits)`, `σ = softplus(σ_raw) + 1.0`.
- `head="categorical"` → `nn.Linear(d, 32)` → bin logits.

`forward()` returns the raw head output (shape depends on head). Loss, sampling, and point-readout live in a separate, unit-tested `drumhumanizer/heads.py` keyed by head type — the backbone stays agnostic.

**Warm-start (default on):** initialize the backbone from Plan B's best checkpoint (`data/processed/transformer_best.pt`) — load its `state_dict` with the `head.*` keys removed and `strict=False`, so all backbone weights transfer and only the new head is fresh. The backbone already learned good structural representations; fine-tuning the whole model with the new head converges in far fewer epochs (important given ~2h/run MPS cost). Fallback: train from scratch.

## 4. Losses (masked over non-pad tokens)

- **Gaussian NLL:** `0.5·log(2π) + log σ + (y-μ)²/(2σ²)`.
- **MDN mixture NLL:** `-logsumexp_k( log π_k + logN(y; μ_k, σ_k) )` — always in log-space via `logsumexp`; never multiply components.
- **Categorical:** cross-entropy between the 32 bin logits and the true velocity's bin index.

## 5. Readouts

Two readouts per probabilistic head:

- **Deterministic point** (for MAE/RMSE parity with Plans A/B):
  - Gaussian → `μ`
  - MDN → mixture mean `Σ_k π_k μ_k`
  - Categorical → expected value `Σ_b p_b · center_b`
- **Sample** (the payoff — restores dynamics):
  - Gaussian → `y ~ N(μ, σ)`
  - MDN → `k ~ Categorical(π)`, then `y ~ N(μ_k, σ_k)`
  - Categorical → `b ~ Categorical(p)`, then `y ~ Uniform[b*4, b*4+4)`
  - all samples clipped/rounded to integer `[0, 127]`

## 6. Evaluation

The core question: **does modelling the distribution + sampling restore dynamics that point estimation flattens?** Three families of metrics on the **test** split:

1. **Held-out NLL** — the proper probabilistic score (lower = better). Continuous heads (Gaussian, MDN) are directly comparable to each other; the categorical NLL is discrete over bins and not directly comparable to continuous NLL.
   - **Comparable (discretized) NLL:** to compare *all three* fairly, also report the NLL of the true velocity's **32-bin** under each head — for the continuous heads compute the bin's probability *mass* by CDF integration (`Φ((hi-μ)/σ) - Φ((lo-μ)/σ)`, mixture-weighted for MDN); for categorical it is the bin softmax prob. This puts all heads on the same discrete measure.

2. **Deterministic-readout metrics** — reuse `drumhumanizer.metrics.evaluate` on the point readout: MAE, RMSE, per-track Pearson/Spearman, within-bar ranking, per-genre. Apples-to-apples with LightGBM (Plan A) and the deterministic transformer (Plan B).

3. **Sampled-distribution match** — the payoff, on a sampled prediction:
   - `evaluate()` on the sampled output — especially **std ratio** and per-track std error (does sampling recover the natural spread?).
   - **Global velocity-histogram distance** between sampled and true velocities (e.g. 1-D Wasserstein / earth-mover distance and/or histogram-intersection over the 32 bins).
   - Note the expectation: sampling should *lower* point MAE slightly (samples scatter) but *dramatically improve* std ratio and histogram match vs the deterministic readout — that trade-off is the thesis point.

Reproducibility: seed 42 everywhere (incl. sampling RNG); resumable training reused from Plan B.

## 7. Deliverables

- `drumhumanizer/heads.py` — head modules + per-head `nll`, `sample`, `point` (readout), and bin helpers. Unit-tested.
- `drumhumanizer/model.py` — `VelocityTransformer` extended with a `head` argument (deterministic default → back-compatible).
- `drumhumanizer/metrics.py` (or a new `prob_metrics.py`) — `discretized_nll`, `hist_distance`. Unit-tested.
- `scripts/train_head.py` — train one probabilistic head (warm-start, resumable), evaluate (all three metric families), write `docs/plan_c/<head>/`.
- `docs/plan_c/` — per-head `metrics.json` + figures (true-vs-sampled velocity histograms, per-genre, calibration/reliability), and a combined `results.md` comparing deterministic / Gaussian / MDN / Categorical.
- **Optional:** extend `notebooks/audition_models.ipynb` to audition a *sampled* humanization from a probabilistic head (hear the restored dynamics).

Deferred (unchanged from the master design §7/§8): held-out-drummer robustness check, formal listening test.

## 8. Self-review

- **Spec §6 model 3 coverage:** Gaussian / MDN / Categorical heads, same backbone, swap head+loss only (§3). ✓
- **No leakage / non-autoregressive:** inherited from the Plan B backbone (structural features only, single parallel pass). ✓
- **Fair comparison:** deterministic-readout metrics reuse the identical `evaluate` harness as Plans A/B; discretized NLL makes continuous and categorical heads comparable (§6). ✓
- **Categorical bins:** 32 (decision), not 128 (§2). ✓
- **MDN stability:** σ floor + logsumexp (§3/§4). ✓
- **Compute:** warm-start from Plan B backbone to keep training tractable on MPS; resumable checkpointing reused (§3). ✓
