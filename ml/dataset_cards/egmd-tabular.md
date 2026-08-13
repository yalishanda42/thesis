---
license: cc-by-4.0
pretty_name: Dynamics Needed — E-GMD Section-A tabular features
tags:
  - music
  - midi
  - drums
  - velocity
  - dynamics
  - tabular
size_categories:
  - 10M<n<100M
configs:
  - config_name: default
    data_files:
      - split: train
        path: egmd_tabular_train.parquet
      - split: validation
        path: egmd_tabular_validation.parquet
      - split: test
        path: egmd_tabular_test.parquet
---

# Dynamics Needed — E-GMD Section-A tabular features

Per-note **structural features + target velocity** for every drum note in the
Expanded Groove MIDI Dataset (E-GMD v1.0.0), used to train the *Dynamics Needed*
drum-dynamics models. One row per note; the learning task is to predict `velocity`
("dynamics") from structure/timing features **without velocity leakage**.

Derived from the raw MIDI mirror `yalishanda/e-gmd-v1.0.0-midi` via the feature
extraction in the project repo (`drum_dynamics`, `ml/scripts/build_dataset.py`).

## Splits (E-GMD official partition)

| split | rows (notes) |
|-------|-------------:|
| train | 11,070,345 |
| validation | 1,696,507 |
| test | 1,560,251 |

## Columns (40)

- **Target:** `velocity` (0–127).
- **Identity/keys (drop before training):** `file_id`, `drummer`, `split`,
  `onset_sec`, `bar_index`.
- **Categorical:** `voice` (drum piece), `genre`, `style`, `time_signature`,
  `beat_type`, `nearest_subdiv`.
- **Metrical phase:** `phase_beat`, `phase_bar`, `sin_beat`, `cos_beat`, `sin_bar`,
  `cos_bar`, `swing_ratio`.
- **Timing:** `log_time_to_prev`, `log_time_to_next`, `log_same_voice_prev`,
  `log_same_voice_next`.
- **Density / simultaneity:** `simult_count`, `density_1beat`, `bpm`, and
  per-voice `simult_*` co-occurrence flags.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("yalishanda/dynamics-needed-egmd-tabular")
```

## Known artifact

E-GMD's multi-kit rendering remaps pads to different voices per kit, which can
scramble the `voice` label across kits and bias per-voice statistics. See the
project's `docs/methodology/kit-remapping-artifact.md`. A single-kit rebuild is a
pending fix.

## License & attribution

Derived from E-GMD, which is licensed **CC BY 4.0** by Google LLC; this derived
dataset is released under the same license. You must attribute the original authors.

```bibtex
@misc{callender2020improving,
    title={Improving Perceptual Quality of Drum Transcription with the Expanded Groove MIDI Dataset},
    author={Lee Callender and Curtis Hawthorne and Jesse Engel},
    year={2020},
    eprint={2004.00188},
    archivePrefix={arXiv},
    primaryClass={cs.SD}
}
```
