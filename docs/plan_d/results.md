# Plan D — Results: Held-out-Drummer Robustness

**Question.** All Plan A–C numbers use E-GMD's official split, which shares *every*
drummer across train/val/test. So they measure **in-distribution** performance: for
every test file the model saw that same drummer's style during training. Does the
humanizer generalize to a **player it has never heard**?

**Design.** Re-partition the featurized rows so two drummers form a drummer-disjoint
test set and never appear in train/val (`ml/src/drum_dynamics/data/holdout.py`,
`scripts/build_holdout_split.py`). We hold out **drummer3 + drummer8**. Neither is the
unique source of any genre, so **all 17 genres remain in training** (genre
differentiation — more important than any single drummer — is preserved). drummer1 and
drummer5 *cannot* be held out: they uniquely own dance/gospel/punk and middleeastern.

The remaining 7 drummers are split into train/val by file (genre-stratified, seed 42).
We retrain the Plan B backbone and the recommended Plan C **categorical** head from
scratch on this split (same hyper-parameters), then evaluate on the held-out drummers.

Held-out test = **134 files / 2.07 M notes (14.4 %)** across 9 genres, each still
covered by ≥1 training drummer.

## Headline: a real generalization gap

| model / readout | split | disc_nll | MAE | per-track r | within-bar ρ | std_ratio | W1 | histI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| backbone (deterministic) | in-distribution (Plan B) | — | 19.56 | 0.605 | 0.612 | 0.895 | — | — |
| backbone (deterministic) | **held-out d3+d8** | — | **27.65** | 0.478 | 0.491 | 0.837 | — | — |
| categorical — point | in-distribution (Plan C) | 3.04 | 19.17 | 0.632 | 0.626 | 0.784 | — | — |
| categorical — point | **held-out d3+d8** | 3.31 | **27.98** | 0.490 | 0.489 | 0.706 | — | — |
| categorical — sample | in-distribution (Plan C) | 3.04 | 24.16 | 0.468 | — | 0.982 | **1.49** | **0.947** |
| categorical — sample | **held-out d3+d8** | 3.31 | 31.96 | 0.309 | — | 0.921 | **16.82** | 0.788 |

Point MAE jumps **≈19 → ≈28** (+45 %) on unseen drummers, and the sampled Wasserstein
distance explodes **1.49 → 16.82**.

**It is not a data-loss artifact.** During retraining, the **validation MAE on *seen*
drummers reached 19.05**, matching Plan B's in-distribution ~19.5. So the model trained
to full in-distribution quality on the reduced data — the test-set jump is specifically
the *unseen-drummer* effect, not the 14 % smaller training set.

## Root cause: absolute loudness is an unlearnable per-player trait

The W1 blow-up with an intact `std_ratio` (0.92) is the tell — **right shape, wrong
location.** The held-out drummers simply hit *harder* than the training pool:

| group | mean velocity | std |
|---|---:|---:|
| training drummers (pool) | 63.2 | 33.7 |
| **held-out d3 + d8 (true)** | **84.6** | 36.2 |
| — drummer3 | 91.6 | 31.5 |
| — drummer8 | 82.7 | 37.1 |
| **model prediction (point)** | **67.7** | 25.5 |

The model predicts a mean of **67.7** — essentially the *training* distribution (pool
mean 63.2, nudged up by genre/context) — while d3+d8 actually play at **84.6**. The
resulting **−16.8 velocity bias ≈ the 16.8 sampled W1**: the sampled distribution is the
learned one, shifted low, not distorted. A drummer's overall dynamic level is idiosyncratic
and **cannot be inferred from structure/timing alone** for a player never seen; the model
defaults to what it learned.

Decomposing the point-MAE gap (27.98 vs in-distribution 19.17):
- **bias-corrected MAE = 24.20** (after removing the global mean offset) — so ≈ **3.8 MAE
  is pure absolute-level offset**, and the remaining ≈ **5 MAE is genuine loss of
  within-track predictive structure**.

## What *does* generalize

Relative dynamics partially transfer: held-out **per-track r 0.49** and **within-bar ρ
0.49** stay clearly positive (in-distribution 0.63 / 0.63). The model still ranks accents
vs ghost notes and reproduces within-bar accent patterning for a new player — it just
mis-calibrates the absolute level. Sampling still restores spread (std_ratio 0.92),
so the *character* of the dynamics survives even as absolute MAE degrades.

## Caveats (honest scope)

1. **This is a hard/adversarial holdout.** drummer3 (91.6) and drummer8 (82.7) are among
   the *loudest* players in E-GMD, while the training pool is dragged low by the dominant
   drummer1 (64.9) and drummer5 (54.3). A holdout of average-loudness drummers would show
   a smaller gap. A fairer point estimate would average several leave-one-drummer-out folds
   (deferred — each fold is a full retrain).
2. **Only two held-out drummers**, and several held-out genres have very few files
   (neworleans/pop = 1, afrobeat/latin = 3), so per-genre held-out numbers are noisy; the
   aggregate is carried by rock/funk/hiphop/jazz.

## Thesis implications & next steps

The core finding is a clean, defensible limitation: **the humanizer generalizes the
*relative* expressive shape of a performance to unseen players but not their *absolute*
dynamic level.** Three ways to act on it, in increasing ambition:

1. **Report relative-dynamics metrics as primary** (per-track r, within-bar ρ,
   std_ratio) — they generalize; absolute MAE conflates player loudness with model skill.
2. **Condition on a loudness cue** — feed a per-performance target-loudness scalar (or a
   few calibration notes) so the model can offset to a new player's level.
3. **Predict per-track-normalized velocity** (z-scored within performance) and rescale at
   playback — removes absolute level from the learning problem entirely.

## Figures

- `fig_velocity_shift.png` — train vs held-out true velocity distributions with the model's
  predicted mean; the shift *is* the gap.
- `docs/plan_c_holdout/categorical/fig_hist.png` — sampled vs true held-out velocity
  (visibly shifted low relative to the true held-out distribution).

## Reproduce

```bash
.venv/bin/python ml/scripts/build_holdout_split.py                       # drummer3+8 -> test
.venv/bin/python ml/scripts/train_transformer.py --tag holdout           # retrain backbone
.venv/bin/python ml/scripts/train_head.py --head categorical --tag holdout  # retrain head + eval
```
