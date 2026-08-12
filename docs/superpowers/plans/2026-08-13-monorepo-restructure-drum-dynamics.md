# Monorepo Restructure + `drum_dynamics` Package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the repo into a monorepo (Python `ml/`, `plugin/`, `web/`, `manuscript/`), rename the flat `drumhumanizer` package to an installable, family-organized `drum_dynamics` package under `ml/src/`, and wire up (but do not run) Hugging Face model publishing.

**Architecture:** Pure move + rename + packaging with **zero change to existing code behavior**, plus additive files (packaging, HF plumbing, empty component homes). The existing 12-file pytest suite is the safety net: it must pass identically before and after. The work lands in two green, independently-reviewable core commits — (1) relocate+rename the package flat, (2) reorganize modules into family subpackages — followed by additive tasks.

**Tech Stack:** Python 3.12, setuptools (src-layout + PEP 660 editable install), pytest, LightGBM, PyTorch (lazy-loaded), partitura, huggingface_hub. C++/JUCE and React are *not* built here — only empty directory homes.

## Global Constraints

- Package import name is exactly `drum_dynamics` (snake_case). Brand/product name is "Dynamics Needed" (human-facing only; not an import name).
- `requires-python >=3.12`.
- **Torch-free light path must be preserved:** importing `drum_dynamics` or its light modules (`core`, `data.features`, `eval.metrics`, `viz`) must NOT import torch. On this macOS box, loading torch's OpenMP before LightGBM segfaults. Only `data.seqdata` and `models.model` may import torch, and only lazily.
- All subpackage `__init__.py` files stay **empty** (no eager imports).
- `data/`, `sf/`, `manuscript/`, `docs/` stay at the repo root. Scripts/tests/notebooks continue to run **from the repo root**.
- Historical docs under `docs/superpowers/plans/` and `docs/superpowers/specs/` are NOT edited (except adding this plan). All other `drumhumanizer` references get updated.
- Do NOT delete any `.gitkeep` placeholder files.
- Frequent commits: one per task. Run the suite before committing each core task.

**Spec:** `docs/superpowers/specs/2026-08-13-monorepo-restructure-drum-dynamics-design.md`

---

### Task 0: Branch + baseline

**Files:** none (setup only)

- [ ] **Step 1: Create a work branch**

```bash
git checkout -b refactor/monorepo-drum-dynamics
```

- [ ] **Step 2: Capture the baseline test result (the target to preserve)**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: all tests pass. Record the passed count (should be the full existing suite across 12 test files). This exact pass count must hold after every core task below.

---

### Task 1: Relocate + rename the package (flat), add packaging, editable install

Moves the whole Python tree under `ml/`, renames `drumhumanizer` → `drum_dynamics` (modules stay **flat** at this stage), adds real packaging, and switches from `sys.path` hacks to an editable install. The flat `from drum_dynamics import <symbol>` and `from drum_dynamics.<module>` paths remain valid because modules are still flat.

**Files:**
- Create: `ml/pyproject.toml`
- Move (git mv): `drumhumanizer/` → `ml/src/drum_dynamics/`; `scripts/` → `ml/scripts/`; `tests/` → `ml/tests/`; `notebooks/` → `ml/notebooks/`; `legacy_notebooks/` → `ml/legacy_notebooks/`
- Modify: every `.py` under `ml/scripts/` and `ml/tests/` (drop `sys.path` hack + `# noqa: E402`, rename import); `ml/src/drum_dynamics/__init__.py` and `ml/src/drum_dynamics/holdout.py` (name in docstrings via global rename)
- Delete: `requirements.txt` (folded into pyproject)

**Interfaces:**
- Produces: an installed `drum_dynamics` package importable from anywhere; flat symbols (`from drum_dynamics import mae, VelocityTransformer, ...`) and flat module paths (`drum_dynamics.midi`, `drum_dynamics.model`, ...) all resolve. Later tasks rely on this install existing.

- [ ] **Step 1: Move the Python tree into `ml/` with git mv (preserves history)**

```bash
mkdir -p ml/src
git mv drumhumanizer ml/src/drum_dynamics
git mv scripts ml/scripts
git mv tests ml/tests
git mv notebooks ml/notebooks
git mv legacy_notebooks ml/legacy_notebooks
```

- [ ] **Step 2: Create `ml/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "drum_dynamics"
version = "0.1.0"
description = "Predicting/restoring drum-note dynamics (velocities) — the Dynamics Needed thesis"
requires-python = ">=3.12"
dependencies = [
    "partitura>=1.9",   # MIDI performance/score parsing -> note arrays
    "music21",          # music theory toolkit (early EDA)
    "numpy",
    "pandas",
    "scikit-learn>=1.6", # use root_mean_squared_error (squared= arg removed in 1.6+)
    "lightgbm>=4.0",    # gradient-boosted trees: native categorical + gain importances
    "pyarrow>=15",      # parquet I/O for the cached tabular dataset
    "torch",
    "matplotlib",
    "pyfluidsynth",     # needs native FluidSynth (brew install fluid-synth)
    "huggingface_hub",  # model publishing (see ml/scripts/publish_model.py)
]

[project.optional-dependencies]
notebooks = ["jupyter", "ipykernel", "nbconvert"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Global-rename the import name across the moved Python code (flat, safe token)**

`drumhumanizer` is a unique token; a global text replace is safe while modules are still flat.

```bash
grep -rl "drumhumanizer" ml/scripts ml/tests ml/src/drum_dynamics --include="*.py" \
  | xargs sed -i '' 's/drumhumanizer/drum_dynamics/g'
```

(macOS `sed -i ''`. Verify with `grep -rn "drumhumanizer" ml/ --include="*.py"` → no matches.)

- [ ] **Step 4: Remove the now-dead `sys.path` hacks from every script**

In each of `ml/scripts/build_dataset.py`, `phase0_analysis.py`, `analyze_model.py`, `build_holdout_split.py`, `train_head.py`, `train_transformer.py`, `train_tabular.py`: delete the line

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

and delete every trailing `# noqa: E402` on the `from drum_dynamics...` import lines that followed it. Leave existing `import os` / `import sys` lines in place (they are still used for data-path construction). Do the same conceptually in `ml/scripts/refactor_eda_notebook.py` only if it contains a live `sys.path.insert` at module scope (its emitted-string version is handled in Task 6).

```bash
sed -i '' '/sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))/d' \
  ml/scripts/build_dataset.py ml/scripts/phase0_analysis.py ml/scripts/analyze_model.py \
  ml/scripts/build_holdout_split.py ml/scripts/train_head.py ml/scripts/train_transformer.py \
  ml/scripts/train_tabular.py
sed -i '' 's/[[:space:]]*# noqa: E402//' \
  ml/scripts/build_dataset.py ml/scripts/phase0_analysis.py ml/scripts/analyze_model.py \
  ml/scripts/build_holdout_split.py ml/scripts/train_head.py ml/scripts/train_transformer.py \
  ml/scripts/train_tabular.py
```

- [ ] **Step 5: Delete `requirements.txt` (folded into pyproject)**

```bash
git rm requirements.txt
```

- [ ] **Step 6: Editable-install into the existing root venv**

```bash
.venv/bin/pip install -e ml/
```

Expected: builds and installs `drum_dynamics 0.1.0` (editable). No errors.

- [ ] **Step 7: Run the suite from the repo root — must match baseline**

```bash
.venv/bin/python -m pytest ml/tests/ -q
```

Expected: same pass count as Task 0, Step 2. (cwd is the repo root, so `data/` and `sf/` still resolve; `import drum_dynamics` now resolves via the editable install.)

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: relocate + rename drumhumanizer -> drum_dynamics under ml/ (flat, installable)"
```

---

### Task 2: Reorganize modules into family subpackages

Splits the flat package into `core/ data/ models/ eval/ viz/ research/`, updates the package's public API and lazy-import map, fixes the 6 intra-package relative imports, and migrates the absolute submodule imports in scripts/tests. Torch stays out of the light path.

**Files:**
- Create: `ml/src/drum_dynamics/{core,data,models,eval,viz,research}/__init__.py` (empty)
- Move (git mv): the 13 modules into their families (mapping below)
- Modify: `ml/src/drum_dynamics/__init__.py` (import block + `_LAZY`); the 6 modules with intra-package relative imports; all `.py` under `ml/scripts/` and `ml/tests/` that import submodules

**Interfaces:**
- Produces final public paths. Flat convenience API unchanged: `from drum_dynamics import Idx, MidiNote, DRUM_MIDI_NAME, midi_number_to_tone, load_note_array, CANONICAL_VOICES, PITCH_TO_VOICE, voice_of, voice_index, build_note_features, GlobalMeanBaseline, LookupTableBaseline, mae, rmse, evaluate, wasserstein1d, hist_intersection, piano_roll, drums_roll, play_midi_file, play_midi_notes, set_soundfont, get_soundfont` still work, plus lazy `NUMERIC_FEATURES, MAX_LEN, build_genre_vocab, bpm_stats, build_split_tensors, scatter_predictions, VelocityTransformer, warm_start_backbone`. Deep module paths become `drum_dynamics.core.midi`, `drum_dynamics.core.voicemap`, `drum_dynamics.data.features`, `drum_dynamics.data.seqdata`, `drum_dynamics.data.holdout`, `drum_dynamics.models.baselines`, `drum_dynamics.models.model`, `drum_dynamics.models.heads`, `drum_dynamics.eval.metrics`, `drum_dynamics.viz.viz`, `drum_dynamics.viz.playback`, `drum_dynamics.research.analysis`, `drum_dynamics.research.phase0`.

Family mapping:

| Module | Family dir |
|---|---|
| midi.py, voicemap.py | core/ |
| features.py, seqdata.py, holdout.py | data/ |
| baselines.py, model.py, heads.py | models/ |
| metrics.py | eval/ |
| viz.py, playback.py | viz/ |
| analysis.py, phase0.py | research/ |

- [ ] **Step 1: Create empty family packages and move the modules**

```bash
P=ml/src/drum_dynamics
for d in core data models eval viz research; do mkdir -p "$P/$d"; : > "$P/$d/__init__.py"; done
git add $P/core/__init__.py $P/data/__init__.py $P/models/__init__.py \
        $P/eval/__init__.py $P/viz/__init__.py $P/research/__init__.py
git mv $P/midi.py $P/voicemap.py $P/core/
git mv $P/features.py $P/seqdata.py $P/holdout.py $P/data/
git mv $P/baselines.py $P/model.py $P/heads.py $P/models/
git mv $P/metrics.py $P/eval/
git mv $P/viz.py $P/viz/viz.py
git mv $P/playback.py $P/viz/playback.py
git mv $P/analysis.py $P/phase0.py $P/research/
```

- [ ] **Step 2: Fix the 6 intra-package relative imports (convert cross-family to `..<fam>.`)**

Apply these exact edits:

- `ml/src/drum_dynamics/models/baselines.py`:
  `from .features import N_PHASE_BINS` → `from ..data.features import N_PHASE_BINS`
- `ml/src/drum_dynamics/data/features.py`:
  `from .voicemap import CANONICAL_VOICES, voice_of` → `from ..core.voicemap import CANONICAL_VOICES, voice_of`
- `ml/src/drum_dynamics/models/model.py` (three lines):
  `from .heads import head_output_dim, init_mdn_head` → **unchanged** (same family)
  `from .seqdata import NUMERIC_FEATURES` → `from ..data.seqdata import NUMERIC_FEATURES`
  `from .voicemap import CANONICAL_VOICES` → `from ..core.voicemap import CANONICAL_VOICES`
- `ml/src/drum_dynamics/viz/playback.py`:
  `from .midi import MidiNote, load_note_array` → `from ..core.midi import MidiNote, load_note_array`
- `ml/src/drum_dynamics/data/seqdata.py`:
  `from .voicemap import CANONICAL_VOICES` → `from ..core.voicemap import CANONICAL_VOICES`
- `ml/src/drum_dynamics/viz/viz.py`:
  `from .midi import DRUM_MIDI_NAME, MidiNote, midi_number_to_tone` → `from ..core.midi import DRUM_MIDI_NAME, MidiNote, midi_number_to_tone`

- [ ] **Step 3: Update the package `__init__.py` import block**

Replace the light-import block (the `from .midi ...` through `from .playback ...` lines) with:

```python
from .core.midi import (
    Idx,
    MidiNote,
    DRUM_MIDI_NAME,
    EGMD_EXTRA_MIDI_NAME,
    TONES_FORMAT,
    drum_name,
    midi_number_to_tone,
    load_note_array,
)
from .core.voicemap import CANONICAL_VOICES, PITCH_TO_VOICE, voice_of, voice_index
from .data.features import build_note_features
from .models.baselines import GlobalMeanBaseline, LookupTableBaseline
from .eval.metrics import mae, rmse, evaluate, wasserstein1d, hist_intersection
from .viz.viz import piano_roll, drums_roll
from .viz.playback import play_midi_file, play_midi_notes, set_soundfont, get_soundfont
```

- [ ] **Step 4: Update the `_LAZY` map to the new torch-module paths**

Replace the `_LAZY` dict with:

```python
_LAZY = {
    "NUMERIC_FEATURES": "data.seqdata",
    "MAX_LEN": "data.seqdata",
    "build_genre_vocab": "data.seqdata",
    "bpm_stats": "data.seqdata",
    "build_split_tensors": "data.seqdata",
    "scatter_predictions": "data.seqdata",
    "VelocityTransformer": "models.model",
    "warm_start_backbone": "models.model",
}
```

(`__getattr__` already does `importlib.import_module(f".{_LAZY[name]}", __name__)`, which resolves `.data.seqdata` / `.models.model` correctly — no change needed there.)

- [ ] **Step 5: Migrate the absolute submodule imports in scripts + tests**

Run this deterministic, word-boundary-safe rewrite (handles both `drum_dynamics.<mod>` dotted paths and the `from drum_dynamics import <moved-module>` attribute forms):

```bash
.venv/bin/python - <<'PY'
import re, pathlib
FAM = {"midi":"core","voicemap":"core","features":"data","seqdata":"data",
       "holdout":"data","baselines":"models","model":"models","heads":"models",
       "metrics":"eval","viz":"viz","playback":"viz","analysis":"research","phase0":"research"}
files = [p for d in ("ml/scripts","ml/tests") for p in pathlib.Path(d).glob("*.py")]
for f in files:
    s = f.read_text(); orig = s
    # combined attribute import first
    s = s.replace("from drum_dynamics import analysis, heads",
                  "from drum_dynamics.research import analysis\nfrom drum_dynamics.models import heads")
    # remaining single attribute imports of moved modules
    for m, fam in FAM.items():
        s = re.sub(rf"from drum_dynamics import {m}\b", f"from drum_dynamics.{fam} import {m}", s)
    # dotted submodule paths (word-boundary so .model does not match .models)
    for m, fam in FAM.items():
        s = re.sub(rf"\bdrum_dynamics\.{m}\b", f"drum_dynamics.{fam}.{m}", s)
    if s != orig:
        f.write_text(s); print("updated", f)
PY
```

Note: the flat convenience imports (e.g. `from drum_dynamics import mae, VelocityTransformer` in `test_smoke.py`) are intentionally left untouched — they resolve through the package `__init__`.

- [ ] **Step 6: Reinstall (pick up new subpackages) and run the suite**

```bash
.venv/bin/pip install -e ml/
.venv/bin/python -m pytest ml/tests/ -q
```

Expected: same pass count as Task 0.

- [ ] **Step 7: Verify the torch-free light path still holds**

```bash
.venv/bin/python -c "import sys, drum_dynamics; from drum_dynamics.core import midi, voicemap; from drum_dynamics.data import features; from drum_dynamics.eval import metrics; from drum_dynamics.viz import viz, playback; assert 'torch' not in sys.modules, 'torch leaked into the light path'; print('light path torch-free OK')"
```

Expected: prints `light path torch-free OK` (no assertion error).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: organize drum_dynamics into core/data/models/eval/viz/research subpackages"
```

---

### Task 3: Empty component homes for the C++ plugin and React frontend

**Files:**
- Create: `plugin/.gitkeep`, `web/.gitkeep`

- [ ] **Step 1: Create the two homes**

```bash
mkdir -p plugin web
: > plugin/.gitkeep
: > web/.gitkeep
git add plugin/.gitkeep web/.gitkeep
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: add empty plugin/ and web/ component homes"
```

---

### Task 4: Expand the root `.gitignore` for all three toolchains

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Append the toolchain blocks**

Append to `.gitignore` (keep all existing rules above):

```gitignore

# Python — packaging & tooling caches
build/
dist/
*.egg-info/
.eggs/
*.egg
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
.tox/

# C++ / CMake (plugin/)
plugin/build/
plugin/cmake-build-*/
CMakeCache.txt
CMakeFiles/
compile_commands.json
*.o
*.obj
*.a
*.so
*.dylib
*.dll

# Node / React (web/)
node_modules/
web/dist/
web/build/
.vite/
.next/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*
.pnpm-store/

# Env / secrets
.env
.env.*
!.env.example
```

- [ ] **Step 2: Confirm no currently-tracked file becomes ignored, and .gitkeeps survive**

```bash
git ls-files -i -c --exclude-standard   # tracked files matching ignore rules
git check-ignore manuscript/build/.gitkeep plugin/.gitkeep web/.gitkeep && echo "UNEXPECTED: gitkeep ignored" || echo "gitkeeps NOT ignored (correct)"
```

Expected: the first command lists nothing (no tracked file is newly ignored — `.gitkeep` files are tracked and stay tracked even inside ignored dirs); the second prints `gitkeeps NOT ignored (correct)`.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore Python/C++/Node build + env artifacts at repo root"
```

---

### Task 5: Hugging Face model-publishing plumbing (no upload)

Adds a thin publisher script and a model-card template. `huggingface_hub` is already in `pyproject.toml` deps from Task 1. Nothing is uploaded; only the tooling is created and smoke-tested offline.

**Files:**
- Create: `ml/scripts/publish_model.py`
- Create: `ml/model_card.md`
- Test: `ml/tests/test_publish_model.py`

**Interfaces:**
- Produces: `publish_model.py` CLI — `--repo <id> --artifact <path> --path-in-repo <name> [--private] [--no-card]`; exits non-zero with `artifact not found: <path>` when the artifact is missing.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/test_publish_model.py
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "publish_model.py"


def test_help_runs():
    r = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "--repo" in r.stdout


def test_missing_artifact_exits_nonzero(tmp_path):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", "x/y",
         "--artifact", str(tmp_path / "nope.pt"), "--path-in-repo", "model.pt"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "artifact not found" in (r.stderr + r.stdout)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest ml/tests/test_publish_model.py -q
```

Expected: FAIL (script does not exist yet).

- [ ] **Step 3: Write `ml/scripts/publish_model.py`**

```python
#!/usr/bin/env python
"""Publish a trained drum_dynamics model artifact to the Hugging Face Hub.

Example:
    python ml/scripts/publish_model.py \\
        --repo <namespace>/dynamics-needed \\
        --artifact data/processed/transformer_best.pt \\
        --path-in-repo model.pt

Auth: run `hf auth login` first, or set HF_TOKEN. No namespace is hardcoded.
On the first run the model card (ml/model_card.md) is uploaded as the repo README.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO_TYPE = "model"
MODEL_CARD = Path(__file__).resolve().parent.parent / "model_card.md"


def publish(repo_id: str, artifact: str, path_in_repo: str, *,
            private: bool, upload_card: bool) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=artifact,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=REPO_TYPE,
    )
    if upload_card and MODEL_CARD.is_file():
        api.upload_file(
            path_or_fileobj=str(MODEL_CARD),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Publish a drum_dynamics model artifact to the Hugging Face Hub."
    )
    p.add_argument("--repo", required=True, help="HF repo id, e.g. user/dynamics-needed")
    p.add_argument("--artifact", required=True, help="local path to the weight file")
    p.add_argument("--path-in-repo", required=True, help="destination filename in the repo")
    p.add_argument("--private", action="store_true", help="create the repo as private")
    p.add_argument("--no-card", action="store_true",
                   help="skip uploading model_card.md as the repo README")
    args = p.parse_args()

    if not Path(args.artifact).is_file():
        raise SystemExit(f"artifact not found: {args.artifact}")

    publish(args.repo, args.artifact, args.path_in_repo,
            private=args.private, upload_card=not args.no_card)
    print(f"published {args.artifact} -> {args.repo}:{args.path_in_repo}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `ml/model_card.md`**

```markdown
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
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest ml/tests/test_publish_model.py -q
```

Expected: PASS (both tests). Confirms the CLI parses and guards a missing artifact — no network used.

- [ ] **Step 6: Commit**

```bash
git add ml/scripts/publish_model.py ml/model_card.md ml/tests/test_publish_model.py
git commit -m "feat: add HF model publishing script + model card template"
```

---

### Task 6: Update README, manuscript, active docs, and the notebook

Repoints all remaining `drumhumanizer` references (except historical `docs/superpowers/plans` + `specs`) and fixes the moved notebook.

**Files:**
- Rewrite: `README.md`
- Modify: `manuscript/chapters/06-realizatsiya.tex` (1 line)
- Modify: `docs/methodology/kit-remapping-artifact.md`, `docs/phase0/results.md`, `docs/plan_b/results.md`, `docs/plan_d/results.md`, `docs/plan_e/results.md`
- Modify: `ml/notebooks/eda.ipynb`, `ml/scripts/refactor_eda_notebook.py`

- [ ] **Step 1: Rewrite `README.md`**

Replace the entire file with:

````markdown
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
.venv/bin/pip install -e ml/                 # add [notebooks] for jupyter: -e ml/[notebooks]

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
````

- [ ] **Step 2: Update the manuscript reference (escape the LaTeX underscore)**

In `manuscript/chapters/06-realizatsiya.tex`, change `\texttt{drumhumanizer}` to `\texttt{drum\_dynamics}`:

```bash
sed -i '' 's/\\texttt{drumhumanizer}/\\texttt{drum\\_dynamics}/' manuscript/chapters/06-realizatsiya.tex
grep -n "drum" manuscript/chapters/06-realizatsiya.tex   # confirm \texttt{drum\_dynamics}
```

Do not touch any other manuscript prose.

- [ ] **Step 3: Update active-docs references (path + import forms)**

Apply these exact string replacements across `docs/methodology/kit-remapping-artifact.md`, `docs/phase0/results.md`, `docs/plan_b/results.md`, `docs/plan_d/results.md`, `docs/plan_e/results.md`:

- `drumhumanizer/features.py` → `ml/src/drum_dynamics/data/features.py`
- `drumhumanizer/voicemap.py` → `ml/src/drum_dynamics/core/voicemap.py`
- `drumhumanizer/phase0.py` → `ml/src/drum_dynamics/research/phase0.py`
- `drumhumanizer/holdout.py` → `ml/src/drum_dynamics/data/holdout.py`
- `drumhumanizer/analysis.py` → `ml/src/drum_dynamics/research/analysis.py`
- `drumhumanizer.midi import load_note_array` → `drum_dynamics.core.midi import load_note_array`
- `drumhumanizer.voicemap import voice_of` → `drum_dynamics.core.voicemap import voice_of`
- `drumhumanizer.midi.drum_name` → `drum_dynamics.core.midi.drum_name`
- `drumhumanizer.metrics` → `drum_dynamics.eval.metrics`

```bash
cd "$(git rev-parse --show-toplevel)"
for f in docs/methodology/kit-remapping-artifact.md docs/phase0/results.md \
         docs/plan_b/results.md docs/plan_d/results.md docs/plan_e/results.md; do
  sed -i '' \
    -e 's#drumhumanizer/features.py#ml/src/drum_dynamics/data/features.py#g' \
    -e 's#drumhumanizer/voicemap.py#ml/src/drum_dynamics/core/voicemap.py#g' \
    -e 's#drumhumanizer/phase0.py#ml/src/drum_dynamics/research/phase0.py#g' \
    -e 's#drumhumanizer/holdout.py#ml/src/drum_dynamics/data/holdout.py#g' \
    -e 's#drumhumanizer/analysis.py#ml/src/drum_dynamics/research/analysis.py#g' \
    -e 's#drumhumanizer\.midi import load_note_array#drum_dynamics.core.midi import load_note_array#g' \
    -e 's#drumhumanizer\.voicemap import voice_of#drum_dynamics.core.voicemap import voice_of#g' \
    -e 's#drumhumanizer\.midi\.drum_name#drum_dynamics.core.midi.drum_name#g' \
    -e 's#drumhumanizer\.metrics#drum_dynamics.eval.metrics#g' \
    "$f"
done
grep -rn "drumhumanizer" docs/methodology docs/phase0 docs/plan_b docs/plan_d docs/plan_e || echo "active docs clean"
```

Expected: prints `active docs clean`.

- [ ] **Step 4: Fix the moved notebook (`ml/notebooks/eda.ipynb`)**

The notebook uses the flat API, so only the package name + the two relative-path depths (it now sits one level deeper than before, while `data/`/`sf/` stay at root) need fixing:

```bash
python3 - <<'PY'
import pathlib
f = pathlib.Path("ml/notebooks/eda.ipynb")
s = f.read_text()
s = s.replace("drumhumanizer", "drum_dynamics")
s = s.replace("os.path.join('..', 'data'", "os.path.join('..', '..', 'data'")
s = s.replace("../sf/big", "../../sf/big")
f.write_text(s)
print("eda.ipynb updated")
PY
```

(The `sys.path.insert(0, '..')` cell is now redundant given the editable install; it is harmless and left in place.)

- [ ] **Step 5: Update the notebook generator to match**

In `ml/scripts/refactor_eda_notebook.py`, apply the same intent so a regenerated notebook is correct: replace emitted `drumhumanizer` → `drum_dynamics`, `os.path.join('..', 'data'` → `os.path.join('..', '..', 'data'`, `../sf` → `../../sf`, and repoint any hardcoded output path from `notebooks/eda.ipynb` to `ml/notebooks/eda.ipynb`.

```bash
sed -i '' \
  -e 's/drumhumanizer/drum_dynamics/g' \
  -e "s#os.path.join('..', 'data'#os.path.join('..', '..', 'data'#g" \
  -e 's#\.\./sf#../../sf#g' \
  -e 's#notebooks/eda.ipynb#ml/notebooks/eda.ipynb#g' \
  ml/scripts/refactor_eda_notebook.py
```

- [ ] **Step 6: Confirm no stray references remain outside historical docs**

```bash
grep -rn "drumhumanizer" . --exclude-dir=.venv --exclude-dir=.git \
  | grep -v "docs/superpowers/plans/" | grep -v "docs/superpowers/specs/" \
  && echo "FOUND stray references (fix them)" || echo "clean (only historical superpowers docs remain)"
```

Expected: `clean (only historical superpowers docs remain)`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "docs: update README, manuscript, active docs, and notebook for drum_dynamics monorepo"
```

---

### Task 7: Final verification, memory updates, finish branch

**Files:**
- Update memory: `project-env-setup.md`, `MEMORY.md` (+ new monorepo-layout note)

- [ ] **Step 1: Full suite + torch-free check from a clean root**

```bash
.venv/bin/pip install -e ml/
.venv/bin/python -m pytest ml/tests/ -v
.venv/bin/python -c "import sys, drum_dynamics; from drum_dynamics.core import midi; from drum_dynamics.eval import metrics; assert 'torch' not in sys.modules; print('OK: tests + torch-free light path')"
```

Expected: full suite green (baseline count + the 2 new publish tests), torch-free check passes.

- [ ] **Step 2: Update memory notes**

Update `project-env-setup` memory: package is now `drum_dynamics` under `ml/src/`, installed via `.venv/bin/pip install -e ml/` (no more `sys.path` hacks), tests at `ml/tests/`, `data/`/`sf/` still at root. Add a short `monorepo-layout` memory (ml/plugin/web/manuscript) and update the `MEMORY.md` index line(s). Keep `.venv` at repo root as documented.

- [ ] **Step 3: Finish the branch**

Use the superpowers:finishing-a-development-branch skill (per the repo's branch-finish workflow: merge to `main` and push).

---

## Self-Review

**Spec coverage:**
- Repo top-level layout (ml/plugin/web/manuscript/docs/data/sf) → Tasks 1, 3; data/sf/manuscript/docs stay at root → Tasks 1, 6.
- `drumhumanizer` → `drum_dynamics`, src-layout, family subpackages → Tasks 1, 2.
- pyproject + editable install, drop requirements.txt + sys.path hacks → Task 1.
- Empty `__init__` files + lazy torch map + torch-free guarantee → Task 2 (Steps 1, 3, 4, 7).
- Import-site migration (deep paths + attribute forms + intra-package) → Task 2 (Steps 2, 5).
- plugin/ + web/ homes (.gitkeep only) → Task 3.
- Expanded .gitignore for all toolchains + .gitkeep survival → Task 4.
- HF publishing (script + model card + huggingface_hub dep, no upload) → Tasks 1 (dep), 5.
- README rewrite, manuscript 1-line, active docs, notebook + generator, leave historical superpowers docs → Task 6.
- Verification (tests pass, grep clean) + memory updates + finish → Tasks 0, 7.

**Placeholder scan:** No "TBD/TODO" in plan steps. The only `_(fill in when final)_` markers are intentional *template content* inside `ml/model_card.md` (a stub by design), not plan gaps.

**Type/name consistency:** Deep module paths in Task 2's Interfaces block match the family mapping, the `_LAZY` values (`data.seqdata`, `models.model`), the migration script's `FAM` dict, and the doc replacements in Task 6. `publish()`/`main()` signatures in Task 5's test match the implementation. `drum_dynamics` used everywhere (no `drumhumanizer` survives outside historical docs, asserted in Task 6 Step 6).
