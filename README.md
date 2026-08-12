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
plugin/        C++ DAW plugin "Dynamics Needed"  (not yet scaffolded)
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

The source lives in this monorepo; trained weights are published to a separate
HF **model** repo (git+LFS on the Hub) — HF holds artifacts, not the codebase.

```bash
.venv/bin/hf auth login                       # once
.venv/bin/python ml/scripts/publish_model.py \
  --repo <namespace>/dynamics-needed \
  --artifact data/processed/transformer_best.pt \
  --path-in-repo model.pt
```

Load weights back in code / from the plugin backend with `hf download
<namespace>/dynamics-needed`. E-GMD already exists on the Hub — link it, don't
re-upload.
