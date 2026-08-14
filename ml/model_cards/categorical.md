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
  - categorical
datasets:
  - e-gmd
metrics:
  - mae
  - rmse
  - nll
---

# Dynamics Needed — categorical transformer velocity model

**Version:** {{version}}

Transformer with a categorical (softmax-over-velocity-bins) head that predicts a
*distribution* over per-note velocities ("dynamics") for MIDI drum tracks.
Shares the same backbone as the [MDN model](https://huggingface.co/yalishanda/dynamics-needed-mdn);
the head models velocity as a 128-way classification rather than a mixture of
Gaussians. Trained on the Expanded Groove MIDI Dataset (E-GMD). Part of the
*Dynamics Needed* thesis project.

## Files

- `backbone.pt` — shared transformer backbone (identical to the MDN repo).
- `categorical_head.pt` — the categorical output head checkpoint.

## Intended use

Given a MIDI drum track with flat/undynamic velocities, sample or read out a
"best-fitting" velocity per note to restore human-like dynamics. Like the MDN
model this captures velocity *uncertainty*, but via a discrete softmax over
velocity bins; sampling is not temperature-controlled.

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

Because the head is already discrete, the native and discretized NLL coincide.
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
