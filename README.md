# Dynamics Needed

Master's thesis project: **given a MIDI drum track, predict each note's
"best-fitting" velocity** — restoring/​predicting drum *dynamics* (as opposed to
tempo/timing humanization). Focus dataset: the
[Expanded Groove MIDI Dataset (E-GMD)](https://magenta.tensorflow.org/datasets/e-gmd).

## Monorepo layout

```
ml/            Python: ML models + research + training (package: drum_dynamics)
  src/drum_dynamics/  core/ data/ models/ eval/ viz/ research/
  scripts/            training + dataset builders + publish_model.py
  notebooks/          EDA
  tests/
plugin/reaper/ Reaper ReaScript tool "Dynamics Needed" (in-place velocity restore)
web/           React landing page                 (not yet scaffolded)
manuscript/    LaTeX/PDF thesis
docs/          research notes & results
data/  sf/     dataset + soundfonts (gitignored)
```

## Setup

```bash
# 1. Python environment + editable install of the ml package
python -m venv .venv
.venv/bin/pip install -e 'ml/'               # add [notebooks] for jupyter: -e 'ml/[notebooks]'

# 2. Native FluidSynth library (for playback)
brew install fluid-synth                      # macOS   (Debian: apt-get install fluidsynth)

# 3. Dataset (~100 MB) -> data/e-gmd/  (gitignored)
mkdir -p data
curl -fL -o data/e-gmd-v1.0.0-midi.zip \
  https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0-midi.zip
unzip -nq data/e-gmd-v1.0.0-midi.zip -d data/e-gmd

# 4. General MIDI soundfont (~148 MB) -> sf/big/  (gitignored)
mkdir -p sf/big
curl -fL -o sf/big/FluidR3_GM.sf2 \
  'https://raw.githubusercontent.com/urish/cinto/master/media/FluidR3%20GM.sf2'
```

> The original keymusician01 S3 soundfont link is dead (404); the `urish/cinto`
> mirror above is the same FluidR3_GM bank.

## Verify

Run from the repo root:

```bash
.venv/bin/python -m pytest ml/tests/ -v
```

## Publishing models to Hugging Face

The source lives in this monorepo; trained weights are published to separate
HF **model** repos (git+LFS on the Hub) — one repo per model, HF holds
artifacts, not the codebase. `publish_model.py` uploads the artifact and a model
card, auto-filling the card's `{{dotted.key}}` placeholders from a training
`metrics.json` (`--metrics`) and the `{{version}}` from `--version` (which also
tags the repo `v<version>`, so `hf download <repo> --revision v0.1.0` works).

```bash
.venv/bin/hf auth login                       # once (run via `! hf auth login`)

# LightGBM baseline -> its own repo, card auto-filled + versioned
.venv/bin/python ml/scripts/publish_model.py \
  --repo <namespace>/dynamics-needed-lgbm \
  --artifact data/processed/lightgbm_model.joblib --path-in-repo model.joblib \
  --card ml/model_cards/lightgbm.md --metrics docs/plan_a/metrics.json --version 0.1.0
# add the C++-native artifacts to the same repo (see below)
.venv/bin/python ml/scripts/export_lightgbm.py
.venv/bin/hf upload <namespace>/dynamics-needed-lgbm data/processed/lightgbm_model.txt    model.txt     --repo-type model
.venv/bin/hf upload <namespace>/dynamics-needed-lgbm data/processed/lightgbm_features.json features.json --repo-type model

# MDN transformer -> its own repo (needs BOTH the backbone and the MDN head)
.venv/bin/python ml/scripts/publish_model.py \
  --repo <namespace>/dynamics-needed-mdn \
  --artifact data/processed/head_mdn.pt --path-in-repo mdn_head.pt \
  --card ml/model_cards/mdn.md --metrics docs/plan_c/mdn/metrics.json --version 0.1.0
.venv/bin/hf upload <namespace>/dynamics-needed-mdn data/processed/transformer_best.pt backbone.pt --repo-type model
```

The `metrics.json` files are produced by `ml/scripts/train_tabular.py` and
`ml/scripts/train_head.py`. Load weights back with `hf download <namespace>/<repo>`
(add `--revision v0.1.0` for a pinned version).

### Datasets on the Hub

E-GMD was not on the Hub, so this project publishes two dataset repos (both
CC BY 4.0, cards under `ml/dataset_cards/`):

- **[`yalishanda/e-gmd-v1.0.0-midi`](https://huggingface.co/datasets/yalishanda/e-gmd-v1.0.0-midi)**
  — unofficial MIDI-only mirror of the Expanded Groove MIDI Dataset (original
  `e-gmd-v1.0.0-midi.zip` + CSV + LICENSE), attributed to Callender/Hawthorne/Engel.
- **[`yalishanda/dynamics-needed-egmd-tabular`](https://huggingface.co/datasets/yalishanda/dynamics-needed-egmd-tabular)**
  — our derived per-note Section-A feature table (parquet, official splits),
  loadable via `load_dataset(...)`.

### LightGBM native export (for C++ / the DAW plugin)

`ml/scripts/export_lightgbm.py` converts the joblib model to LightGBM's native
text format for cross-language inference:

- `lightgbm_model.txt` — load in C++ via `LGBM_BoosterCreateFromModelfile`, predict
  with `LGBM_BoosterPredictForMat` (link `lib_lightgbm`). Verified to reproduce the
  Python model's predictions exactly.
- `lightgbm_features.json` — the exact feature order + each categorical feature's
  ordered level list. Build the C++ feature vector in `feature_names` order and pass
  each categorical as its **index** in `categorical_levels[col]` (the integer code
  the model splits on). Note: LightGBM is the *point* model (flattens dynamics), good
  for bootstrapping the plugin's inference/feature pipeline before adding the
  transformer.

## Reaper velocity-restoration tool

A Reaper Python ReaScript rewrites the velocities of selected MIDI drum notes in
place, backed by a warm local inference service (`python -m drum_dynamics.serve`)
that serves both the LightGBM and MDN models.

```bash
# 1. Export the MDN inference artifacts (vocab + bpm stats) once
.venv/bin/python ml/scripts/export_mdn.py

# 2. One-time setup: writes config.local.json + prints Reaper registration steps
.venv/bin/python plugin/reaper/setup_reaper.py
```

In Reaper: Actions → New action → Load ReaScript → `plugin/reaper/dynamics_needed.py`,
then bind it. Select notes in the MIDI editor and run the action: the engine
auto-starts on first use, a dialog collects genre/style/model/temperature/blend,
and only the selected notes' velocities are updated (one undo step). The engine
shuts down when Reaper closes.

### Troubleshooting: OpenMP/libomp segfault

If the service or `pytest` crashes with a segmentation fault, PyTorch and
LightGBM are loading two different OpenMP runtimes. Unify them by pointing
torch's bundled libomp at the one LightGBM uses (Homebrew's):

```bash
cd .venv/lib/python3.12/site-packages/torch/lib
cp -n libomp.dylib libomp.dylib.orig      # backup
ln -sf "$(brew --prefix libomp)/lib/libomp.dylib" libomp.dylib
```

A torch reinstall/upgrade reverts this — re-apply if the segfault returns.
