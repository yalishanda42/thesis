# Monorepo restructure + `drum_dynamics` package — design

**Date:** 2026-08-13
**Status:** approved (pending spec review)
**Type:** refactor (move + rename + packaging + additive HF-publishing plumbing;
zero change to existing code behavior)

## Motivation

Two problems with the current repo:

1. **Name.** The Python package is `drumhumanizer`. "Humanize" conventionally
   refers to *tempo/timing* humanization; people expect that, not dynamics. The
   project is about predicting/restoring drum-note **dynamics** (velocities), so
   the name misleads. (Consistent with the thesis terminology decision to avoid
   the „хуманизация'' word family.)
2. **Flat, mixed package.** `drumhumanizer/` is a flat directory holding modules
   of different *kinds* — reusable library code, one-off research code,
   visualization utilities — with no separation.

Additionally, the repo will grow to house four heterogeneous components:
a Python ML package (+ research), the LaTeX/PDF thesis manuscript, a C++ DAW
plugin, and a React.js landing page. The layout must accommodate all four.

## Naming

- **Brand / product name:** **"Dynamics Needed"** — human-facing name for the
  DAW plugin, the landing page, the repo README headline. Separate from code.
- **Python import package:** **`drum_dynamics`** — clean snake_case identifier.
- The brand and the import name are deliberately distinct (cf. "PyTorch" the
  brand vs `torch` the import).

## Repo top-level layout (single git monorepo)

The repo stays one git repository. One top-level directory per component, each
self-contained with its own tooling, plus a thin top-level README that routes to
each.

```
thesis/                          (repo root — local dir name & git repo unchanged)
├── README.md                    rewritten: "Dynamics Needed" brand + routes to each component
├── ml/                          Python: ML models + research + training (see below)
├── plugin/                      C++ DAW plugin "Dynamics Needed"  — empty, .gitkeep only
├── web/                         React landing page                — empty, .gitkeep only
├── manuscript/                  LaTeX/PDF thesis                  — UNCHANGED (does not move)
├── docs/                        research notes/results            — stays at root (cross-component)
├── data/                        E-GMD dataset (gitignored)        — stays at root
└── sf/                          soundfonts (gitignored)           — stays at root
```

Rationale:

- `data/` and `sf/` stay at root so scripts run from the repo root keep resolving
  `data/…` / `sf/…` paths with no change.
- `manuscript/` does not move — avoids churning the VS Code / tectonic config
  (`latex-workshop.latex.search.rootFiles.include: manuscript/main.tex`).
- `docs/` stays at root — it is cross-component project documentation and moving
  it would churn historical plan/spec path references.
- `plugin/` and `web/` are **directory homes only** for now: an empty directory
  with a `.gitkeep`. No build scaffolding, no README. Scaffolding C++/JUCE and
  React/Vite are deferred to their own future tasks.

## Inside `ml/` — src-layout, installable package

```
ml/
├── pyproject.toml               NEW — name=drum_dynamics; deps migrated from requirements.txt
├── src/drum_dynamics/
│   ├── __init__.py              flat convenience re-exports + lazy torch loading (preserved)
│   ├── core/                    midi.py, voicemap.py
│   │   └── __init__.py          empty
│   ├── data/                    features.py, seqdata.py, holdout.py
│   │   └── __init__.py          empty
│   ├── models/                  baselines.py, model.py, heads.py
│   │   └── __init__.py          empty
│   ├── eval/                    metrics.py
│   │   └── __init__.py          empty
│   ├── viz/                     viz.py, playback.py
│   │   └── __init__.py          empty
│   └── research/                analysis.py, phase0.py   (one-off; walled off from reusable core)
│       └── __init__.py          empty
├── scripts/                     train_tabular.py, train_transformer.py, train_head.py,
│                                build_dataset.py, build_holdout_split.py, phase0_analysis.py,
│                                analyze_model.py, refactor_eda_notebook.py
├── notebooks/                   (moved from repo-root notebooks/)
├── legacy_notebooks/            (moved from repo-root legacy_notebooks/)
└── tests/                       (moved from repo-root tests/)
```

### Module → family mapping

| Current (`drumhumanizer/`) | New (`drum_dynamics/`)        |
|----------------------------|-------------------------------|
| `midi.py`                  | `core/midi.py`                |
| `voicemap.py`              | `core/voicemap.py`            |
| `features.py`              | `data/features.py`            |
| `seqdata.py`               | `data/seqdata.py`             |
| `holdout.py`               | `data/holdout.py`             |
| `baselines.py`             | `models/baselines.py`         |
| `model.py`                 | `models/model.py`             |
| `heads.py`                 | `models/heads.py`             |
| `metrics.py`               | `eval/metrics.py`             |
| `viz.py`                   | `viz/viz.py`                  |
| `playback.py`              | `viz/playback.py`             |
| `analysis.py`              | `research/analysis.py`        |
| `phase0.py`                | `research/phase0.py`          |

Note: `eval/` shadows the builtin `eval` name only *within the package
namespace* — harmless and conventional in ML repos.

## Import mechanism — the one workflow change

Today nothing is installed: every script does
`sys.path.insert(0, <repo-root>)` and imports `drumhumanizer`; tests rely on
`python -m pytest` putting cwd (repo root) on `sys.path`. A `src/` layout is not
importable without an install, so:

1. Add `ml/pyproject.toml`.
2. One-time: `.venv/bin/pip install -e ml/` into the **existing root venv**
   (venv stays at repo root, gitignored).
3. After that, `import drum_dynamics` resolves from anywhere.
4. **Delete every `sys.path.insert(...)` hack** from scripts, plus the
   `# noqa: E402` guards that existed only to appease the post-insert imports.
5. Fold `requirements.txt` dependencies into `pyproject.toml`
   `[project.dependencies]`; **remove `requirements.txt`** (single source of
   truth). Preserve the curated-not-frozen intent as a comment.

Scripts, tests, and notebooks still *run from the repo root* exactly as before —
only the import name and the resolution mechanism change.

**Data-path safety:** scripts resolve data via cwd-relative paths (e.g.
`PROC = os.path.join("data", "processed")`), **not** `__file__`-relative. The
only `__file__` use in scripts is the `sys.path` hack being deleted. Therefore
moving scripts into `ml/scripts/` does not affect data resolution, provided they
continue to be run from the repo root.

### `pyproject.toml` sketch

- Build backend: setuptools (`[build-system]`).
- `[project]` name `drum_dynamics`, `requires-python >=3.12`.
- `[project.dependencies]`: partitura, music21, numpy, pandas,
  scikit-learn>=1.6, lightgbm>=4.0, pyarrow>=15, torch, matplotlib,
  pyfluidsynth, huggingface_hub. (Notebook tooling —
  jupyter/ipykernel/nbconvert — goes in an optional
  `[project.optional-dependencies].notebooks` group.)
- `[tool.setuptools.packages.find]` with `where = ["src"]`.

## Import-site migration (mechanical — all references)

`drumhumanizer.<mod>` → `drum_dynamics.<family>.<mod>`. Examples:

- `drumhumanizer.midi`     → `drum_dynamics.core.midi`
- `drumhumanizer.voicemap` → `drum_dynamics.core.voicemap`
- `drumhumanizer.features` → `drum_dynamics.data.features`
- `drumhumanizer.seqdata`  → `drum_dynamics.data.seqdata`
- `drumhumanizer.holdout`  → `drum_dynamics.data.holdout`
- `drumhumanizer.baselines`→ `drum_dynamics.models.baselines`
- `drumhumanizer.model`    → `drum_dynamics.models.model`
- `drumhumanizer.heads`    → `drum_dynamics.models.heads`
- `drumhumanizer.metrics`  → `drum_dynamics.eval.metrics`
- `drumhumanizer.viz`      → `drum_dynamics.viz.viz`
- `drumhumanizer.playback` → `drum_dynamics.viz.playback`
- `drumhumanizer.analysis` → `drum_dynamics.research.analysis`
- `drumhumanizer.phase0`   → `drum_dynamics.research.phase0`

Attribute-style imports like `from drumhumanizer import analysis, heads` are
preserved via the root `__init__.py` re-exports (see below), becoming
`from drum_dynamics import analysis, heads`.

Affected files (from `grep`): all of `ml/scripts/*.py`, all of `ml/tests/*.py`,
`ml/src/drum_dynamics/__init__.py`, `ml/src/drum_dynamics/data/holdout.py`
(self-reference), and doc/manuscript references (below).

## Torch-free guarantee preserved

The macOS libomp constraint — loading torch's OpenMP before LightGBM segfaults —
must stay honored. The current `__init__.py` keeps torch out of the light path
via a lazy `__getattr__`. Preserved as follows:

- **Subpackage `__init__.py` files stay empty** — no eager imports, so
  `from drum_dynamics.data import features` never pulls in `seqdata` (torch).
- **Root `__init__.py`** keeps eager imports of the light symbols and the lazy
  `__getattr__` for torch-dependent symbols. The `_LAZY` map updates to the new
  module paths:
  - `NUMERIC_FEATURES`, `MAX_LEN`, `build_genre_vocab`, `bpm_stats`,
    `build_split_tensors`, `scatter_predictions` → `data.seqdata`
  - `VelocityTransformer`, `warm_start_backbone` → `models.model`
- The flat `__all__` convenience API (for notebooks/interactive use) is kept
  intact, sourced from the new subpaths.

## Ancillary updates

- **README.md** — rewritten for the monorepo + "Dynamics Needed" brand + new
  setup (`pip install -e ml/`, per-component layout, verification commands with
  `ml/tests/`).
- **.gitignore** — add `web/node_modules`, `web/dist`, `plugin/build`,
  `plugin/cmake-build-*`. Existing python/data/sf/mac rules unchanged
  (`.venv`, `data`, `**/sf/big`, etc. still apply at root).
- **manuscript** — update the single `\texttt{drumhumanizer}` →
  `\texttt{drum\_dynamics}` in `manuscript/chapters/06-realizatsiya.tex`.
  **No other manuscript prose is touched.**
- **docs** — update path/name references in *active* docs
  (`docs/methodology/`, any results docs). **Leave `docs/superpowers/plans/`
  and `docs/superpowers/specs/` untouched** — they are a historical record of
  past work and describe the state at the time.
- **`ml/scripts/refactor_eda_notebook.py`** — regenerate the notebook cells it
  emits so they use `drum_dynamics` (flat import) and drop the
  `sys.path.insert(0, '..')` cell (editable install makes it unnecessary).

## Model publishing to Hugging Face (structure only — no upload during refactor)

HF Hub repos are **artifact publishing targets**, not source homes: a model repo
(`git`+LFS on hf.co) holds only weights + config + a model card, and is
populated by pushing files up (`hf upload`). It does **not** mirror this
monorepo. Therefore the source stays a single monorepo (the thesis's single
artifact); HF is just a distribution endpoint we push *to*. No splitting the
source into multiple repos.

Wire up the plumbing now so publishing isn't ad-hoc later:

- **`ml/scripts/publish_model.py`** — thin wrapper over `hf upload` (via the
  `huggingface_hub` Python API or shelling to the `hf` CLI). Reads a trained
  artifact from the existing gitignored `data/processed/` location (no new
  checkpoint convention introduced) and uploads it to a `model`-type repo, e.g.
  `hf upload <namespace>/dynamics-needed <artifact> <path-in-repo> --type model`.
  Repo id and artifact path are CLI args / env vars — no hardcoded namespace.
  Adds `huggingface_hub` to `pyproject.toml` deps.
- **`ml/model_card.md`** — model-card template with YAML front-matter
  (license, `library_name`, `pipeline_tag`, `tags`, `datasets: [e-gmd]`,
  metrics placeholders) + prose stub. Uploaded as the model repo's `README.md`.
- **README** — short "Publishing models to Hugging Face" section documenting
  `hf auth login`, `publish_model.py`, and `hf download` for the reverse path
  (loading weights in code / the future plugin backend).

Not done during this refactor: no actual `hf upload`, no HF repo creation, no
auth. Just the script + template + docs, ready to run when a model is final.

## Out of scope (deferred to their own tasks)

- Actual C++/JUCE plugin scaffolding.
- Actual React/Vite frontend scaffolding.
- CI configuration.
- Any model, feature, or logic change. This refactor is **move + rename +
  packaging only**, with **zero behavior change**.

## Verification

Run from the repo root:

```bash
.venv/bin/pip install -e ml/
.venv/bin/python -m pytest ml/tests/ -v
```

Success criteria: the editable install succeeds, and **all existing tests pass**
(same count as before the refactor), confirming the move/rename introduced no
behavior change. A grep for `drumhumanizer` across code returns only the
historical `docs/superpowers/` artifacts intentionally left untouched.

## Post-refactor follow-ups (not part of this task)

- Update memory notes: `project-env-setup` (new package name, editable install,
  `ml/` layout), and add a note on the monorepo layout.
- Re-register the Jupyter kernel note in README if the venv path assumptions
  changed (they do not — venv stays at root).
