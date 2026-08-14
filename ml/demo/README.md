---
title: Dynamics Needed
emoji: 🥁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
python_version: "3.12"
short_description: Humanize drum dynamics
startup_duration_timeout: 30m
---

# Dynamics Needed — demo Space

Web demo for structure-driven drum-velocity prediction: upload a drum-MIDI
groove, pick a model and its musical context, and get back a version whose note
velocities are predicted from **structure and timing alone** (never from the
note's own loudness), plus a before/after velocity plot and an audio preview.

Three models, all trained on [E-GMD](https://magenta.tensorflow.org/datasets/e-gmd):

- **LightGBM** — fast, deterministic gradient-boosted trees.
- **Transformer · MDN** — mixture-density head, temperature-controllable.
- **Transformer · Categorical** — softmax-over-bins head.

## This directory is the source of truth

The demo source lives in the thesis monorepo at `ml/demo/`; the Hugging Face
Space is a deploy target, not a second repo. Deploy with:

```bash
./deploy.sh <namespace>/<space-name>
```

`deploy.sh` assembles a self-contained bundle (this app + a vendored copy of the
`drum_dynamics` package + the six ready-to-load model files from
`data/processed/` + the FluidR3_GM SoundFont) and pushes it with
`hf upload … --repo-type space`. Model weights and the SoundFont are **not**
committed to the monorepo; they are copied in at deploy time.

## Hardware

Runs on ZeroGPU (the only free option for a Gradio Space on a non-PRO account).
The models are tiny and CPU-only, so inference never requests a GPU — a single
no-op `@spaces.GPU` function satisfies the ZeroGPU requirement without burning
visitor quota.

## Credits

Example grooves are clips from the E-GMD dataset (Google Magenta, CC-BY 4.0).
