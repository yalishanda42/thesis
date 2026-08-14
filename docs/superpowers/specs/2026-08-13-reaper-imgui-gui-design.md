# Dynamics Needed — ReaImGui GUI (design)

**Date:** 2026-08-13
**Status:** REVISED — pivoted to Lua (see "Pivot to Lua" below); the Python panel was implemented, failed at runtime, and is abandoned.
**Supersedes:** the single-row `RPR_GetUserInputs` CSV dialog in `plugin/reaper/dynamics_needed.py`

## Pivot to Lua (2026-08-13, post-implementation)

The Python panel was built and passed all static review, but on the first real
run in REAPER it failed with `name 'RPR_ImGui_CreateContext' is not defined`.

**Root cause (verified on disk):** REAPER's shipped `reaper_python.py`
(`~/Applications/REAPER.app/Contents/Plugins/`) is a *static* file that defines a
`def RPR_<Name>` wrapper only for the *core* ReaScript API (each looks up a
pointer in the `_ft` table and calls it via ctypes). It contains **no
`RPR_ImGui_*` wrappers** and has **no `__getattr__`**, so `from reaper_python
import *` can never expose ReaImGui's functions — installing ReaImGui does not
help. The only Python route is hand-writing a ctypes `CFUNCTYPE` per ImGui call,
which is impractical for a full GUI. This is the exact risk the walking-skeleton
gate was meant to catch; the gate was skipped.

**Decision:** rewrite the panel in **Lua**, where `reaper.ImGui_*` (or
`require 'imgui'`) is the native, ergonomic, well-trodden path. Consequences that
override the Python decisions below:

- **No threads in Lua.** The background predict worker and the non-blocking
  "starting" state are replaced by single-threaded, debounced, blocking calls on
  the defer loop (accepted latency tradeoff).
- **HTTP via `curl`** through `reaper.ExecProcess` (POST body via temp file);
  **JSON via a bundled `json.lua`** (Lua has neither in stdlib).
- **Engine autostart** via `reaper.ExecProcess` (detached spawn) + per-frame
  short-timeout `curl` health polling while showing "Starting engine...".
- `dn_core` logic is **ported to `dn_core.lua`** and self-tested with the local
  `lua` interpreter (`~/homebrew/bin/lua`, 5.4). The Python modules
  (`dynamics_needed.py`, `engine_client.py`, `predict_worker.py`, `dn_core.py`)
  and their pytest files are **removed** — superseded, not used by the Lua panel.
- The Python **inference engine** (`drum_dynamics.serve`) and `setup_reaper.py`
  are unchanged.
- ReaImGui must be installed via ReaPack (it currently is not).

Everything below this section is the original Python design, retained for
history; the ImGui *feature* behavior (controls, live preview, apply-with-undo,
per-track persistence, status line) still holds — only the language/runtime
changes.

## Goal

Replace the cramped CSV dialog with a modern **ReaImGui** panel that offers
proper controls (dropdowns, sliders, toggle) **plus a live velocity preview**:
before/after bars for the target notes that update as you drag temperature/blend,
so you see the predicted humanization shape before committing.

## Decisions (from brainstorming)

- **Vision:** Controls + live preview (not a full persistent dashboard).
- **Stack:** **Pure Python** ReaImGui panel driving Reaper's embedded Python.
  Chosen over a Lua panel because live preview is latency-sensitive: Python
  calls the engine **in-process** via `urllib` (single-digit ms), whereas Lua
  has no in-process HTTP and would fork a `curl` process per preview
  (~10–40 ms spawn overhead — often larger than the inference itself). Python
  also keeps `dn_core.py` as the single unit-tested source of truth and allows a
  background predict thread (Lua has no threads).
- **Dependency:** ReaImGui (installed by the user via ReaPack). Required — it is
  the only route to a modern panel in Reaper.
- **Platform:** macOS-targeted (the thesis environment). No curl dependency now
  that transport is in-process `urllib`, so this is not a hard constraint.
- **Known risk:** driving ImGui's frame/defer loop from Reaper's Python runtime
  is less-trodden than Lua. Resolved by a **walking-skeleton first step** (§7)
  that is a go/no-go gate before any UI is built. Fallback if it fails: the Lua
  panel design, accepting curl latency.

## 1. Architecture & files

The Python inference engine (`drum_dynamics.serve` on `127.0.0.1:8765`) is
**unchanged**. The Reaper action is rewritten as an ImGui panel. Engine
lifecycle and HTTP move out of the (Reaper-coupled, untestable) ReaScript into
an importable, unit-tested module.

```
plugin/reaper/
  dynamics_needed.py   REWRITE — ImGui context + defer loop + Reaper glue (the registered action)
  dn_core.py           KEEP    — pure request/response logic, unit-tested (extend as needed)
  engine_client.py     NEW     — in-process engine lifecycle + HTTP (health/ensure/predict), unit-tested
  setup_reaper.py      KEEP    — unchanged; the config it writes is reused as-is
```

Responsibility split:

- **`dn_core.py`** (pure, tested): `build_predict_request`, `resolve_target_indices`,
  `parse_velocities`, `genres_from_styles`, `filter_styles_by_genre`. No Reaper,
  no I/O.
- **`engine_client.py`** (I/O, tested by mocking `urllib`/`subprocess`):
  `base_url(cfg)`, `health(cfg)`, `ensure_engine(cfg)` (start if down + poll
  `/health`), `predict(cfg, request) -> response`. Extracted from today's inline
  `_health` / `_start_engine` / `_ensure_engine`.
- **`dynamics_needed.py`** (Reaper-coupled, thin, not unit-tested): config load,
  ImGui context lifecycle, the defer loop, reading notes / tempo / time-sig,
  per-track persistence, applying velocities with undo, and the predict worker.

## 2. Concurrency & the predict worker

To keep the panel at frame rate while predicting, predict runs on a **background
thread**. Hard rule: **the worker touches no Reaper (`RPR_*`) API** — those are
main-thread only. The worker only does pure logic + `urllib`.

- Main (defer) thread snapshots the notes it read from the take and the current
  params, and hands `(seq, params, notes_snapshot)` to the worker.
- The worker **debounces** (~150 ms) and coalesces: it always computes the
  *latest* requested `seq`, discarding stale ones. It stores `(seq, velocities)`
  under a lock.
- The main thread reads the latest result each frame; if its `seq` is current,
  it updates the preview bars. Apply uses **this previewed result** — never a
  fresh predict — so "apply what you saw" holds even though MDN is stochastic.

## 3. Data flow

1. **On open:** load config (`RPR_GetResourcePath()/dynamics_needed_config.json`,
   unchanged); `engine_client.ensure_engine(cfg)` → populate genre/style/model
   dropdowns from `/health`; load last-used params for the active track.
2. **Each frame:** read the active take's notes (throttled/hash-guarded so a new
   predict is only queued when the note set or params actually change);
   determine target notes (selected, else all); draw controls + preview.
3. **On param change (live) or `[Preview]`:** queue a predict via the worker.
   `[Preview]` forces an immediate predict even if params are unchanged — this
   doubles as **reroll** for the stochastic MDN model.
4. **On `[Apply]`:** write the previewed velocities to the take inside
   `RPR_Undo_BeginBlock`/`EndBlock`, rewriting **only** velocity per note
   (re-reading each note's other fields first — the existing correctness fix is
   preserved).

## 4. UI & interaction

Matches the approved preview:

```
┌ Dynamics Needed ─────────────────┐
│ Genre [rock ▼] Style [straight ▼] │
│ Model ( ) LGBM  (•) MDN           │
│ Temp [===o--] 1.0 Blend [==o-] .8 │
│ Is a fill? [ ]        [x] Live    │
│ ─ velocities (N target notes) ─── │
│  ▁▃▅█▃▅▂▇▃▅▁█  predicted (color)  │
│  ▂▂▂▂▂▂▂▂▂▂▂▂  current (faint)    │
│ status: ready                     │
│        [ Preview ]  [ Apply ]     │
└───────────────────────────────────┘
```

- **Genre ▼** filters **Style ▼** (`filter_styles_by_genre`). **Model** radio
  LGBM/MDN. **Temp** slider 0–2 (def 1.0). **Blend** slider 0–1 (def 0.8).
  **"Is a fill?"** checkbox (default **off** — beat is the default `beat_type`).
  **Live** checkbox (default on) toggles auto-predict
  on change; when off, only `[Preview]` predicts.
- **Target notes** = selected notes, else all notes (unchanged rule). Count shown.
- **Velocity lane** drawn with the ImGui draw list: current velocities as faint
  bars, predicted as colored bars, both scaled 0–127, in note (time) order.
- **Persistence:** last-used params saved per track via `ProjExtState` keyed on
  track GUID (unchanged behavior).

## 5. Error handling

A colored status line at the panel bottom reflects engine state:

- **starting** — while `ensure_engine` polls (cold torch import can take ~15 s).
- **ready** — engine reachable.
- **error + `[Retry]`** — engine unreachable; Retry re-runs `ensure_engine`.
- **no take** — no active MIDI editor/take; `[Apply]`/`[Preview]` disabled.
- **predict failed** — request errored; keep the last preview, show the message.

## 6. Testing

- **`dn_core.py`** — existing pytest kept and extended for any new helpers.
- **`engine_client.py`** — pytest with `urllib`/`subprocess` mocked (or a stub
  HTTP server): `base_url`, `health` parsing, `ensure_engine` start-and-poll
  (down→up transition, timeout), `predict` request/response round-trip.
- **`dynamics_needed.py`** — the ImGui loop + Reaper glue is thin and Reaper-only;
  verified by the walking-skeleton smoke test (§7) and manual testing, not unit
  tests.

## 7. First step — walking skeleton (go/no-go)

Before any UI: a minimal `dynamics_needed.py` that
(1) creates an ImGui context and runs a stable defer loop drawing an empty
window, (2) reads the active take's note count, and (3) calls
`engine_client.ensure_engine` and shows the live `/health` payload. This proves
the only real unknowns end-to-end — Python defer-loop stability, ImGui
rendering, in-process `urllib` to the engine, note reading — before UI is built
on top. If it fails, fall back to the Lua panel design.

## Out of scope (YAGNI)

Per-hit drag-to-edit velocities, presets, applied-pass history, dockable
persistent dashboard — deferred to a possible later "Full dashboard" iteration.
