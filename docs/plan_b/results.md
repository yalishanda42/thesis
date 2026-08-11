# Plan B — Results: Event-Sequence Transformer

Non-autoregressive Transformer (design §5/§6 model 2) predicting every note's
velocity in a single parallel pass from **structural features only** (drum-part +
genre embeddings, sin/cos metrical phase, z-scored bpm, log delta-time-to-prev,
same-onset flag). Trained on E-GMD `train`, early-checkpointed on `validation`,
evaluated on the held-out `test` split with the shared `drumhumanizer.metrics`
harness — the same one used for Plan A, so the numbers are directly comparable.

Config: `d_model=128`, 4 layers, 8 heads, window `MAX_LEN=512`, AdamW `lr=3e-4`,
masked-L1 loss, seed 42, Apple MPS. Best epoch **14** (val MAE 18.99); 15-epoch cap.

## Comparison table (test split)

| model          |    MAE |   RMSE | per-track r | per-track ρ | within-bar ρ | std ratio | Δstd/track |
|----------------|-------:|-------:|------------:|------------:|-------------:|----------:|-----------:|
| global_mean    | 29.668 | 34.638 |         nan |         nan |          nan |     0.000 |          — |
| lookup_table   | 21.393 | 28.380 |       0.552 |       0.512 |        0.561 |     0.729 |      8.281 |
| **lightgbm**   | **18.020** | **24.072** |   **0.706** |   **0.665** |    **0.672** |     0.692 |      9.021 |
| transformer    | 19.564 | 26.820 |       0.605 |       0.556 |        0.612 | **0.895** |  **5.470** |

(`Δstd/track` = mean absolute difference between predicted and true per-track velocity std — lower is better; measures the mean-regression / flattening failure directly, design §7.)

## Verdict — the transformer clears the bar, but the trees still win on accuracy

The **real bar (design §7) is the lookup-table "dumb humanizer,"** and the
transformer beats it on every axis: lower MAE (19.56 vs 21.39) and RMSE (26.82 vs
28.38), higher per-track correlation (0.605 vs 0.552) and within-bar ranking (0.612
vs 0.561). ✅

It does **not** beat the LightGBM tabular baseline on point accuracy: LightGBM has
lower MAE/RMSE and higher correlations. On this dataset, with minimal tuning, the
gradient-boosted trees on the hand-engineered Section A window features are the
stronger predictor — an honest answer to the design's "does the transformer earn its
complexity?" question (§6): **not yet, on MAE.**

## Where the transformer is actually better: it doesn't flatten the dynamics

The one axis where the transformer clearly wins is exactly the one MSE-style accuracy
hides — **spread preservation**:

- **std ratio 0.895** (vs LightGBM 0.692, global-mean 0.000): the transformer's
  predicted velocities retain ~90% of the true dynamic range; LightGBM only ~69%.
- **per-track std error 5.47** (vs LightGBM 9.02): the transformer's per-track
  loudness variation is far closer to the real performance.

This is the mean-regression failure the design called out (§1.2, §7): a model can win
on MAE by hugging the conditional mean and quietly compressing the dynamics. LightGBM
does more of that; the transformer keeps the peaks and valleys. For a *humanization*
task, where the variance **is** the humanity, this is a meaningful qualitative edge
even at a slightly worse MAE — and it is a strong motivation for the Plan C
probabilistic heads (Gaussian/MDN/categorical), which target the distribution rather
than a point estimate.

## Per-genre MAE — transformer vs LightGBM

| genre      | transformer | lightgbm |     | genre  | transformer | lightgbm |
|------------|------------:|---------:|-----|--------|------------:|---------:|
| punk       |       12.82 |    12.18 |     | pop    |   **19.94** |    24.11 |
| neworleans |       14.58 |    13.94 |     | gospel |       20.25 | **11.83** |
| reggae     |       16.13 |    14.44 |     | funk   |       20.48 |    18.76 |
| country    |       16.74 |    15.81 |     | hiphop |       20.87 |    19.18 |
| latin      |       16.80 |    16.33 |     | rock   |       22.60 |    19.65 |
| jazz       |       17.20 |    15.70 |     | soul   |       22.66 |    19.88 |

LightGBM is better on most genres, but the two models disagree sharply on the
extremes: the transformer is **much better on pop** (19.94 vs 24.11 — pop was
LightGBM's *hardest* genre) yet **much worse on gospel** (20.25 vs 11.83 — gospel was
LightGBM's *easiest*). The trees appear to exploit strong genre-local velocity
regularities (gospel accents are highly learnable by a lookup/tree), while the
transformer generalizes more uniformly and does not over-fit the easy genres.

## Figures

- `fig_training_curve.png` — validation MAE per epoch (45.8 → 19.0, best at epoch 14).
- `fig_pred_vs_true.png` — predicted vs true velocity (hexbin, 20k test sample); the
  wider vertical spread vs Plan A's LightGBM plot reflects the higher retained variance.
- `fig_per_genre.png` — transformer per-genre MAE.

## Honest caveats & next steps

- **Minimal tuning:** 15 epochs, default hyperparameters, single non-overlapping
  512-token window scheme, no LR schedule, MPS (no CUDA). The gap to LightGBM is
  plausibly narrowable with tuning (harmonics on phase, overlapping windows, longer
  training, LR warmup/cosine).
- **Deferred (design §7/§8):** held-out-drummer robustness check and the qualitative
  A/B listening test (the `drummer` column is carried in the dataset, so the former
  needs no rebuild).
- **Plan C:** probabilistic heads on this same backbone. The transformer's superior
  spread preservation is the strongest signal that modelling the velocity
  *distribution* (Gaussian/MDN/categorical), not a point estimate, is the right
  direction for humanization quality.
