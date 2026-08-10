# Humanizing Drum Dynamics

Master's thesis project: **given a MIDI drum track, predict each note's
"best-fitting" velocity** — i.e. *humanizing dynamics*, analogous to the
tempo/timing humanization that already exists. Focus dataset: the
[Expanded Groove MIDI Dataset (E-GMD)](https://magenta.tensorflow.org/datasets/e-gmd).

## Layout

```
drumhumanizer/        Reusable helpers (extracted from the notebooks)
  midi.py             MidiNote/Idx wrappers, GM drum map, load_note_array
  viz.py              piano_roll / drums_roll
  playback.py         In-notebook audio via FluidSynth
notebooks/eda.ipynb   Drums EDA of E-GMD (imports drumhumanizer)
tests/                Smoke tests against a real E-GMD file
legacy_notebooks/     Original proof-of-concept (poc.ipynb: piano + drums LSTM); superseded

```

## Setup

```bash
# 1. Python environment
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Native FluidSynth library (for playback)
brew install fluid-synth          # macOS   (Debian: apt-get install fluidsynth)

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

> The soundfont link in the original notebook (keymusician01 S3) is dead (404);
> the `urish/cinto` mirror above is the same FluidR3_GM bank.

## Verify

```bash
.venv/bin/python -m pytest tests/ -v            # 6 smoke tests
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=thesis-venv notebooks/eda.ipynb
```

Register the venv kernel once with:
`.venv/bin/python -m ipykernel install --user --name thesis-venv`.
