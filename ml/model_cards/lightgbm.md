---
license: mit
library_name: drum_dynamics
tags:
  - drums
  - midi
  - velocity
  - dynamics
  - music
  - lightgbm
  - tabular
datasets:
  - e-gmd
metrics:
  - mae
  - rmse
---

# Dynamics Needed — LightGBM velocity baseline

**Version:** {{version}}

Gradient-boosted-tree baseline that predicts per-note velocities ("dynamics")
for MIDI drum tracks from tabular note features. Trained on the Expanded Groove
MIDI Dataset (E-GMD). Part of the *Dynamics Needed* thesis project.

## Intended use

Given a MIDI drum track with flat/undynamic velocities, predict a
"best-fitting" velocity per note to restore human-like dynamics. This is the
tabular baseline; see the MDN transformer model for the probabilistic variant.

## Training data

[E-GMD](https://magenta.tensorflow.org/datasets/e-gmd) (Expanded Groove MIDI
Dataset), evaluated on the held-out test split.

## Metrics (test split)

| model         | MAE                  | RMSE                  |
|---------------|----------------------|-----------------------|
| Global-mean   | {{global_mean.mae}}  | {{global_mean.rmse}}  |
| Lookup table  | {{lookup_table.mae}} | {{lookup_table.rmse}} |
| **LightGBM**  | {{lightgbm.mae}}     | {{lightgbm.rmse}}     |

Per-track Pearson (LightGBM): {{lightgbm.per_track_pearson}}

## Limitations

This is a **point** (single-value) predictor and a **baseline** — published for
reproducibility and comparison, not as a final production model.

- **Flattens dynamics.** It regresses toward the conditional mean (std ratio
  ~0.69), so it cannot reproduce the full velocity distribution / ghost-note
  tails. Restoring that spread is the job of the probabilistic transformer heads.
- **Absolute-loudness generalization gap.** On drummers unseen in training,
  point MAE degrades (~19 → ~28); a player's overall loudness is not inferable
  from structure alone. Relative dynamics (per-track / within-bar ranking)
  transfer better than absolute level.
- **Known data artifact.** E-GMD's multi-kit rendering remaps pads to different
  voices per kit, which biases some per-voice results; a single-kit rebuild is a
  pending fix. See the project's `docs/methodology/kit-remapping-artifact.md`.
- **No listening test yet.** Numbers here are offline metrics; perceptual
  A/B validation is future work.

## License

Set to `mit` by default — change to match the thesis's chosen license before
publishing.
