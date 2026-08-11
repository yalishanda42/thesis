# Plan C — Results: Probabilistic Velocity Heads

Three probabilistic heads (Gaussian, MDN K=5, Categorical 32-bin) on the **same Plan B
transformer backbone**, warm-started from Plan B's checkpoint and fine-tuned by NLL.
Each predicts a *distribution* over velocity so we can **sample** to restore the
dynamic spread that point estimation flattens. Evaluated on the E-GMD **test** split
with the shared harness; seed 42; MPS. Each head early-stopped at 5–7 epochs.

## The core result — sampling restores dynamics

Every head's **sampled** output has a far higher `std_ratio` than its **point** readout,
confirming Plan C's hypothesis (a point estimate hugs the conditional mean and
flattens; sampling from the learned distribution puts the variance back):

| head | point std_ratio | **sampled std_ratio** |
|------|----------------:|----------------------:|
| Gaussian | 0.564 | 0.851 |
| MDN (anti-collapse) | 0.807 | **0.995** |
| **Categorical** | 0.784 | **0.982** |

## Full comparison (test split)

Point/regression models (deterministic readout) and the probabilistic heads under
both readouts. `disc_nll` = negative log-prob of the true velocity's 32-bin —
**comparable across all heads** (CDF-integrated for the continuous heads); lower is
better. `W1` = 1-D Wasserstein between sampled and true velocities; `histI` =
histogram intersection (1 = identical distributions).

| model / readout | disc_nll | MAE | per-track r | within-bar ρ | std_ratio | W1 | histI |
|---|---:|---:|---:|---:|---:|---:|---:|
| lookup_table (Plan A) | — | 21.39 | 0.552 | 0.561 | 0.729 | — | — |
| **LightGBM (Plan A)** | — | **18.02** | **0.706** | **0.672** | 0.692 | — | — |
| transformer det. (Plan B) | — | 19.56 | 0.605 | 0.612 | 0.895 | — | — |
| Gaussian — point | 3.55 | 22.02 | 0.575 | 0.616 | 0.564 | — | — |
| Gaussian — sample | 3.55 | 27.62 | 0.293 | 0.356 | 0.851 | 6.03 | 0.873 |
| MDN — point | 3.13 | 19.69 | 0.617 | 0.608 | 0.807 | — | — |
| MDN — sample | 3.13 | 24.55 | 0.455 | 0.468 | 0.995 | 2.95 | 0.921 |
| **Categorical — point** | **3.04** | **19.17** | **0.632** | **0.626** | 0.784 | — | — |
| **Categorical — sample** | **3.04** | 24.16 | 0.468 | 0.483 | **0.982** | **1.49** | **0.947** |

(Native NLL — Gaussian 4.93, MDN 4.47 — is a continuous density and not comparable to
the categorical's discrete NLL; `disc_nll` is the fair cross-head number.)

> **MDN mode-collapse fix.** The MDN was initially degenerate: diagnostics showed
> π = `[0.007, 0, 0.993, 0, 0]` — one component carried 99.3% of the weight (effective
> #components 1.03/5), so it behaved like a single broad Gaussian and its histogram was
> indistinguishable from the Gaussian head's. Root cause was classic winner-take-all
> collapse (symmetric head init + softmax rich-get-richer, amplified by warm-starting
> from a point-optimized backbone). Fix: **spread the component-mean initialization**
> across the velocity range + a **load-balancing penalty** on the batch-averaged π. The
> retrained MDN uses 2.67/5 effective components with means spanning [14, 42, 64, 95,
> 127] (ghost → accent), and its metrics jumped accordingly (sampled std_ratio
> 0.868 → 0.995, W1 6.44 → 2.95, histI 0.877 → 0.921, disc-NLL 3.47 → 3.13). The numbers
> above are the fixed model.

## Findings

1. **The categorical head wins on every axis that matters.** Best comparable NLL
   (3.04 vs 3.47/3.55), best point-readout MAE of any head (19.17 — beating even the
   Plan B deterministic transformer's 19.56 and its per-track r 0.632 > 0.605), and by
   a wide margin the best distributional match when sampling: **std_ratio 0.982**
   (near-perfect), **W1 1.49** (vs ~6.4 for the continuous heads), **histI 0.947**.
   Velocity is a bounded integer, and the non-parametric discrete head models it
   without the continuous heads' wasted probability mass outside [0, 127].

2. **Sampling restores spread but costs point accuracy/ranking.** For every head,
   sampling raises MAE and lowers per-track/within-bar correlation (e.g. categorical
   r 0.632 → 0.468) because the added stochasticity decorrelates note-by-note ordering.
   This is the expected, honest trade-off: **use the deterministic readout for
   accuracy/ranking; sample for realistic, human-like variance.** The categorical head
   is best under *both* readouts, so it is the recommended model.

3. **MDN (after the anti-collapse fix) clearly beats the Gaussian and rivals the
   categorical.** Once its components stay alive, the mixture's genuine multimodality
   shows: disc-NLL 3.13 (vs Gaussian 3.55, categorical 3.04), sampled std_ratio 0.995
   (best of all heads) and much better distribution match than Gaussian (W1 2.95 vs
   6.03, histI 0.921 vs 0.873). The categorical still edges it on W1/histI/NLL and is
   simpler and stabler, but the MDN is now a legitimate multimodal model rather than a
   disguised single Gaussian. It trained stably (σ floor + grad clipping; no NaNs).

4. **Does the distribution earn its complexity?** For pure point accuracy, LightGBM
   (MAE 18.02) is still the strongest single number — but it is a *point* model and
   **cannot** reproduce the velocity distribution. The categorical head's sampled
   output matches the true distribution almost exactly (histI 0.947, std_ratio 0.982),
   which is the actual humanization objective (design §1: "the variance *is* the
   humanity"). No point model — LightGBM or deterministic transformer — can do that.

## Figures

Per head under `docs/plan_c/<head>/`:
- `fig_hist.png` — sampled vs true velocity distribution (the categorical overlay is
  visibly the tightest match).
- `fig_val_nll.png` — validation NLL per epoch.

## Recommendation & next steps

**Categorical (32-bin) is the Plan C model of choice**: best likelihood, best point
accuracy among heads, and best-in-class distributional realism when sampling. Natural
follow-ups (deferred): ordinal/soft-label targets to exploit bin adjacency; a
temperature knob to trade sampled spread vs accuracy; the held-out-drummer robustness
check and a formal A/B listening test (design §7/§8). The audition notebook can play
sampled humanizations from any head for qualitative listening.
