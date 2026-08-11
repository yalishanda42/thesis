# Plan A — Results: Tabular Velocity Baseline

Structural-only (no velocity leakage) per-note velocity prediction on E-GMD, using
the Section A feature table (design §4). Fit on the `train` split, evaluated on the
held-out `test` split. Reproducible: `random_state=42`, LightGBM early-stopped on
`validation` (best iteration **206**).

## Comparison table (test split)

| model          |    MAE |   RMSE | per-track r | per-track ρ | within-bar ρ | std ratio |
|----------------|-------:|-------:|------------:|------------:|-------------:|----------:|
| global_mean    | 29.668 | 34.638 |         nan |         nan |          nan |     0.000 |
| lookup_table   | 21.393 | 28.380 |       0.552 |       0.512 |        0.561 |     0.729 |
| **lightgbm**   | **18.020** | **24.072** |   **0.706** |   **0.665** |    **0.672** |     0.692 |

**The learned model beats both baselines**, and — the real bar (design §7) — beats
the "dumb humanizer" lookup table on every quality axis: lower MAE (−3.4) and RMSE
(−4.3), and markedly higher correlation with the true dynamics both per track
(Pearson 0.71 vs 0.55) and *within the bar* (Spearman 0.67 vs 0.56). The global-mean
predictor produces no spread at all (std ratio 0.000); LightGBM restores most of the
per-track dynamic range (std ratio 0.69, mean abs per-track std diff 9.0 velocity units).

> Caveat: the lookup table's global std ratio (0.729) is nominally closer to 1.0 than
> LightGBM's (0.692), but it achieves that spread with *worse-placed* variation — its
> within-bar and per-track rank correlations are far lower. Matching the amount of
> spread matters less than putting it on the right notes.

## Top feature importances (LightGBM gain)

1. `style` (7745) — genre/sub-style is the single strongest conditioner of dynamics
2. `bpm` (5414)
3. `log_same_voice_prev` (4446) — time since the same voice last struck
4. `log_same_voice_next` (4007)
5. `voice` (3945) — which drum piece
6. `log_time_to_next` (3086)
7. `log_time_to_prev` (2907)
8. `genre` (2098)
9. `sin_bar` (1961) — metrical position within the bar
10. `density_1beat` (1926)

Same-voice inter-onset timing (features 3–4) is more informative than global
neighbour timing (6–7): how long since *this* drum last played predicts its loudness
better than how long since *any* drum played — consistent with per-voice dynamic
patterns (e.g. hi-hat accent cycles, ghost-note snares). Metrical position (`sin_bar`,
`cos_bar`, `cos_beat`) and local density round out the top features. See
`fig_importance.png`.

## Per-genre MAE spread

Sorted best → worst (test MAE, velocity units):

| easiest |  MAE  | hardest |  MAE  |
|---------|------:|---------|------:|
| gospel     | 11.83 | pop     | 24.11 |
| punk       | 12.18 | soul    | 19.88 |
| neworleans | 13.94 | rock    | 19.65 |
| reggae     | 14.44 | hiphop  | 19.18 |
| afrocuban  | 15.03 | funk    | 18.76 |

There is a ~2× spread in difficulty across genres (11.8 → 24.1). The design §7
hypothesis was that more dynamically expressive genres (e.g. funk) would show larger
errors than flat/programmed ones (e.g. pop). The data pushes back: **pop is the
*hardest* genre**, not the flattest — its velocities are the least predictable from
structure alone, while funk sits mid-pack (18.8). Genres with strong, idiomatic
dynamic conventions (gospel, punk, New Orleans, reggae) are the *easiest*, plausibly
because their accent patterns are more regular and thus more learnable from metrical
position + voice + timing. See `fig_per_genre.png`.

## Figures

- `fig_importance.png` — LightGBM gain importance, top 20.
- `fig_pred_vs_true.png` — predicted vs true velocity (hexbin, 20k test sample).
- `fig_per_genre.png` — per-genre MAE.

## Verdict

The LightGBM tabular baseline clears the Plan A success criterion: it beats the
lookup-table "dumb humanizer" on MAE, RMSE, per-track correlation, and within-bar
ranking. This establishes the tabular reference point the sequence model (Plan B)
must improve upon. Held-out-drummer generalization and a listening test are deferred;
the `drummer` column is carried in the dataset so the former can be run without a rebuild.
