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

## License

Set to `mit` by default — change to match the thesis's chosen license before
publishing.
