# Reaper velocity-restoration tool — Design

- **Date:** 2026-08-13
- **Status:** Approved for planning
- **Component of:** *Dynamics Needed* (drum-dynamics thesis)

## 1. Context

The `drum_dynamics` package predicts a "best-fitting" velocity per drum note
from purely structural features (voice, metrical phase, timing deltas,
simultaneity, density, bpm, genre/style) — never using velocity as an input.
Two trained models exist: a **LightGBM** point regressor (deterministic
baseline) and an **MDN transformer** (probabilistic, temperature-controlled,
restores the full dynamic spread).

This spec covers the **DAW-facing tool** that lets a user apply those
predictions to a MIDI drum selection inside a DAW.

### 1.1 Why this shape (decisions already made)

- **No real-time plugin.** The model needs musically-complete context
  (bar/beat phase, time-to-prev/next, simultaneity windows) and processing
  time; feeding it arbitrary mid-stream chunks violates its input assumptions.
  Real-time VST/AU MIDI processing is therefore ruled out.
- **DAW scripting, not a portable plugin.** The desired UX — select notes →
  invoke → rewrite velocities *in place* — is impossible for a cross-DAW
  VST3/AU plugin (a plugin only sees a live stream, never the stored clip).
  It *is* possible via DAW scripting. Among the user's hosts (Reaper, Ableton,
  Logic), **only Reaper** supports it cleanly (Python ReaScript, full MIDI
  note read/write API). Logic's Scripter is real-time-only; Ableton needs Max
  for Live. **Target: Reaper only.**
- **Shared inference service.** Inference runs in a warm local Python service
  reused by both the Reaper script and a future Streamlit demo. This avoids
  wiring `torch`/`lightgbm` into Reaper's embedded Python (the ReaScript stays
  stdlib-only) and keeps the model loaded once.
- **Both models user-selectable** (LGBM + MDN).
- **Distribution: thesis-demo grade.** No consumer installer/PyInstaller
  bundle. A one-time setup script installs the ReaScript and records the venv
  path; at use-time the script auto-starts the engine — no terminal.

## 2. Goals / Non-goals

**Goals**
- In-place velocity rewrite of a selected MIDI drum region in Reaper.
- Per-invocation dialog for the inputs the MIDI lacks: genre, style, model,
  temperature (MDN), blend, fill-flag.
- Warm local inference engine, auto-started and auto-terminated, no terminal
  at use-time.
- A pure, unit-testable inference core reused by a future Streamlit frontend.

**Non-goals (out of scope)**
- Real-time / streaming processing.
- Ableton, Logic, or any non-Reaper host.
- C++ plugin and the LightGBM native export (`lightgbm_model.txt`) — unused by
  this path; kept only for a hypothetical future C++ revival.
- The Streamlit app itself (separate later effort; it will reuse this service).
- A consumer installer / frozen executable.

## 3. Architecture

```
┌─────────────────────┐     HTTP POST /predict      ┌──────────────────────────┐
│  Reaper ReaScript    │ ──────────────────────────> │  Inference service       │
│  (Python, stdlib)    │ <────────────────────────── │  (Python, warm)          │
│  gather → POST →     │     {velocities: {...}}      │  loads LGBM + MDN once,  │
│  write selected notes│                              │  localhost:8765          │
└─────────────────────┘                              └──────────────────────────┘
        (future Streamlit app is a second client of the same service)
```

Two parts to build: the **inference service** and the **Reaper ReaScript**.

## 4. Component 1 — Inference core + service

Location: `ml/src/drum_dynamics/serve/`.

### 4.1 Pure core (no HTTP, unit-testable)

`predict_velocities(request) -> dict[int, int]`

- **Input** (`request`):
  - `model`: `"lgbm"` | `"mdn"`
  - `style`: one of the 61 known styles (e.g. `"rock/halftime"`); `genre` is
    **derived** as `style.split("/")[0]` (matches `build_note_features`).
  - `temperature`: float, used only by MDN (ignored by LGBM).
  - `blend`: float in `[0, 1]` (0 = keep original, 1 = full prediction).
  - `beat_type`: `"beat"` | `"fill"`.
  - `bpm`: float; `time_signature`: E-GMD string (e.g. `"4-4"`).
  - `notes`: list of `{index:int, pitch:int, onset_sec:float,
    velocity:int, selected:bool}`.
- **Processing:**
  1. Build features for **all** notes (context correctness) via
     `build_note_features`, injecting `style`/`beat_type`/`bpm`/`time_signature`
     into `meta`.
  2. LGBM path → `model.predict(df)`. MDN path → build the note sequence
     (reuse `data/seqdata.py` + `models/`), sample with `temperature`.
  3. **Blend + clamp**, only for `selected` notes:
     `new = clamp(round(blend*pred + (1-blend)*orig), 1, 127)`.
- **Output:** `{note_index: new_velocity}` for **selected notes only**.

### 4.2 HTTP wrapper

- `POST /predict` — body = the request above; returns `{"velocities": {index: vel}}`.
- `GET /health` — returns `{"status":"ok","models":[...],"styles":[...],
  "genres":[...]}`. The genre/style lists are read from
  `lightgbm_features.json` (`categorical_levels`) — the single source of truth
  for the dialog dropdowns.
- Framework: FastAPI or Flask (implementer's choice; keep it minimal).

### 4.3 Lifecycle

Started as `python -m drum_dynamics.serve` with args:

- `--port 8765` (fixed default).
- `--parent-pid <PID>` — the engine polls `os.kill(pid, 0)` and **exits when
  that process (Reaper) is gone**. Primary termination path.
- `--idle-timeout 1800` — exits after N seconds with no `/predict`. Backstop.

On startup the engine writes a **pidfile** and logs to a **log file** (paths
under a project-local runtime dir, e.g. `plugin/reaper/.runtime/`). Models load
once at startup and stay resident. There is **no** manual stop action.

## 5. Component 2 — Reaper ReaScript

Location: `plugin/reaper/`. Language: Python, **stdlib only** (`urllib`, `json`,
`subprocess`, `os`).

### 5.1 Selection semantics

Precedence for "the selection":
1. Selected notes in the active MIDI take, else
2. Notes inside the current time selection, else
3. All notes in the selected media item(s).

**Read the entire take for feature context, but only overwrite the velocities
of the notes in the resolved selection set.**

### 5.2 Extraction

- Enumerate notes via `MIDI_CountEvts` / `MIDI_GetNote`.
- Convert note start PPQ → seconds via `TimeMap` helpers (project-relative
  onset in seconds is what the core needs).
- `bpm` and `time_signature` via `TimeMap_GetTimeSigAtTime` at the take start.

### 5.3 Dialog (per invocation)

Native `GetUserInputs` with combo-box fields (ReaImGui is an optional future
upgrade, not required):

- **genre** (dropdown, filters styles) → **style** (dropdown, filtered)
- **model** (`lgbm` / `mdn`)
- **temperature** (relevant to MDN)
- **blend** (0–1)
- **fill?** checkbox → sets `beat_type` = `"fill"` (default `"beat"`)

Last-used values are **remembered per track** via project ext-state.

### 5.4 Auto-start + apply

1. `GET /health`. If it fails, `subprocess.Popen` the engine detached
   (`start_new_session=True`, stdin closed, stdout/stderr → log file), launching
   **the project venv's python** (path from the setup-written config), passing
   `--parent-pid <this Reaper PID>`. Poll `/health` until green (show a
   "Starting engine…" message; timeout ~15s).
2. Build the request (all notes + selection flags + dialog values), `POST
   /predict`.
3. Apply returned velocities with `MIDI_SetNote` on selected notes only, then
   `MIDI_Sort`, all wrapped in `Undo_BeginBlock` / `Undo_EndBlock` (one clean
   undo step).
4. **Error UX:** if the engine can't be reached/started, a dialog explains how
   to run the one-time setup.

## 6. Setup & use-time UX

- **One-time setup** (`plugin/reaper/setup` script, documented in README):
  copies the ReaScript into Reaper's `Scripts/` dir, registers the action, and
  writes the venv python path into a small config the script reads.
- **Use-time:** click the Reaper menu action → first click boots the engine
  (few-second wait) → dialog → notes rewritten in place → engine stays warm →
  dies when Reaper closes. No terminal, ever.

## 7. Testing strategy

- **Core (`predict_velocities`):** unit tests for blend math + clamp,
  genre-derived-from-style, both model paths, "only selected notes change,"
  and "context uses all notes." TDD, reusing `ml/tests` conventions.
- **Service:** one contract test — start the app, POST a sample groove, assert
  response shape and index alignment.
- **ReaScript:** factor pure logic (PPQ→sec, selection gather, request build,
  response apply) into a plain module importable **outside** Reaper and unit-test
  it; keep raw Reaper API calls in a thin, untested shell. Plus a manual Reaper
  smoke test on a flat-velocity groove.

## 8. Repo layout changes

```
ml/src/drum_dynamics/serve/     NEW  inference core + HTTP service + __main__
plugin/reaper/                  NEW  ReaScript (thin client) + pure logic module + setup
plugin/reaper/.runtime/         NEW  pidfile + logs (gitignored)
README.md                            update "C++ plugin" note → Reaper script + service
```

## 9. Risks / notes

- **MDN inference shape.** The MDN path must reconstruct the sequence
  representation the model expects (`seqdata.py`); getting per-note alignment
  right is the main implementation risk. LGBM is straightforward.
- **Cold start.** First `torch` import can take several seconds; mitigated by the
  warm engine (paid once per Reaper session).
- **`beat_type` is a guess.** The user's MIDI has no beat/fill annotation;
  default `beat` + optional fill flag is a pragmatic approximation.
- **Absolute-loudness gap.** Per the model card, absolute velocity level
  generalizes worse than relative dynamics; the blend control mitigates
  surprising output.
```
