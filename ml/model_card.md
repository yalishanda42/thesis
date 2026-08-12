---
license: mit
library_name: drum_dynamics
tags:
  - drums
  - midi
  - velocity
  - dynamics
  - music
datasets:
  - e-gmd
metrics:
  - mae
  - rmse
---

# Dynamics Needed — drum velocity model

Predicts per-note velocities ("dynamics") for MIDI drum tracks, trained on the
Expanded Groove MIDI Dataset (E-GMD). Part of the *Dynamics Needed* thesis
project.

## Intended use

Given a MIDI drum track with flat/undynamic velocities, predict a
"best-fitting" velocity per note to restore human-like dynamics.

## Training data

[E-GMD](https://magenta.tensorflow.org/datasets/e-gmd) (Expanded Groove MIDI
Dataset).

## Metrics

| metric | value                    |
|--------|--------------------------|
| MAE    | _(fill in when final)_   |
| RMSE   | _(fill in when final)_   |

## License

Set to `mit` by default — change to match the thesis's chosen license before
publishing.
