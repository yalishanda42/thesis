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

**Version:** {{version}}

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

Use the **deterministic readout** for accuracy/ranking; **sample** from the
predicted distribution to restore human-like variance.

## Limitations

Published as a versioned research artifact, not a final production model.

- **Point-vs-sample trade-off.** Sampling restores the dynamic spread that a
  point estimate flattens, but raises MAE and lowers note-by-note ranking
  correlation. Pick the readout to match the use case.
- **Absolute-loudness generalization gap.** On unseen drummers, MAE degrades
  (~19 → ~28) and the sampled distribution shifts low — a player's overall
  loudness is not inferable from structure alone. Relative dynamics transfer;
  absolute level does not.
- **Known data artifact.** E-GMD's multi-kit rendering remaps pads to different
  voices per kit, biasing some per-voice results; a single-kit rebuild is a
  pending fix. See the project's `docs/methodology/kit-remapping-artifact.md`.
- **No listening test yet.** Numbers here are offline metrics; perceptual
  A/B validation is future work.

## License

Set to `mit` by default — change to match the thesis's chosen license before
publishing.
