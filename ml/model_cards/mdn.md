---
license: mit
library_name: drum_dynamics
tags:
  - drums
  - midi
  - velocity
  - dynamics
  - music
  - transformer
  - mixture-density-network
datasets:
  - e-gmd
metrics:
  - mae
  - rmse
  - nll
---

# Dynamics Needed — MDN transformer velocity model

Transformer with a mixture-density-network (MDN) head that predicts a
*distribution* over per-note velocities ("dynamics") for MIDI drum tracks.
Trained on the Expanded Groove MIDI Dataset (E-GMD). Part of the
*Dynamics Needed* thesis project.

## Intended use

Given a MIDI drum track with flat/undynamic velocities, sample or read out a
"best-fitting" velocity per note to restore human-like dynamics. Unlike the
LightGBM baseline, this model captures velocity *uncertainty*.

## Training data

[E-GMD](https://magenta.tensorflow.org/datasets/e-gmd) (Expanded Groove MIDI
Dataset), evaluated on the held-out test split.

## Metrics (test split)

| metric                            | value                            |
|-----------------------------------|----------------------------------|
| Native NLL                        | {{native_nll}}                   |
| Discretized NLL                   | {{discretized_nll}}              |
| Deterministic readout — MAE       | {{deterministic_readout.mae}}    |
| Deterministic readout — RMSE      | {{deterministic_readout.rmse}}   |
| Sampled — Wasserstein-1           | {{sampled_wasserstein}}          |
| Sampled — histogram intersection  | {{sampled_hist_intersection}}    |

## License

Set to `mit` by default — change to match the thesis's chosen license before
publishing.
