# Dynamics Needed ReaImGui GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-row `RPR_GetUserInputs` CSV dialog with a modern ReaImGui panel offering dropdowns/sliders/toggle plus a live before/after velocity preview.

**Architecture:** The Python inference engine (`drum_dynamics.serve`) is unchanged. Engine lifecycle + HTTP move from the ReaScript into an importable, unit-tested `engine_client.py`. A pure, unit-tested `predict_worker.py` runs predictions on a background thread (debounced + coalesced) so the panel stays at frame rate. `dynamics_needed.py` becomes a thin, Reaper-coupled ImGui panel that consumes both plus the existing pure `dn_core.py`.

**Tech Stack:** Python (Reaper embedded runtime), ReaImGui (Dear ImGui bindings via ReaPack), stdlib `urllib`/`subprocess`/`threading`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-13-reaper-imgui-gui-design.md`

## Global Constraints

- **ReaScript ASCII-only:** every file Reaper loads (`dynamics_needed.py` and any module it imports at runtime) must be pure ASCII — Reaper loads Python under an ASCII locale. No smart quotes, em dashes, or non-ASCII in strings/comments.
- **No `__file__` in the ReaScript:** Reaper defines neither `__file__` nor a Python `get_action_context()`. Locate config via `RPR_GetResourcePath()/dynamics_needed_config.json` only.
- **`RPR_*` calls return tuples:** every `RPR_*` function returns a tuple whose first elements echo the in/out params; unpack accordingly (see existing `dynamics_needed.py` for the pattern).
- **Reaper API is main-thread only:** the background predict worker must NEVER call any `RPR_*` function. It receives plain data snapshots and returns plain data.
- **Config shape (unchanged):** `{"venv_python": str, "repo_root": str, "port": int}` written by `setup_reaper.py` to the Reaper resource path.
- **Engine contract (unchanged):** `GET /health` -> `{"status":"ok","models":["lgbm","mdn"],"styles":[...],"genres":[...]}`; `POST /predict` (body = request dict) -> `{"velocities": {"<index>": <int velocity>}}`.
- **Tests:** run from repo root with `python -m pytest`. New pure-Python tests live in `ml/tests/` and prepend `plugin/reaper` to `sys.path` (mirror `ml/tests/test_reaper_core.py`). These modules import only stdlib, so no torch/LightGBM/libomp is loaded.
- **Defaults:** Temperature slider 0.0-2.0 (default 1.0); Blend slider 0.0-1.0 (default 0.8); "Is a fill?" default off (`beat_type="beat"`); Live preview default on. Per-track saved params override these defaults when present.

---

### Task 1: Extract `engine_client.py` (engine lifecycle + HTTP)

Move the inline `_base_url` / `_health` / `_start_engine` / `_ensure_engine` logic out of the ReaScript into a testable module. Pure stdlib; no Reaper API.

**Files:**
- Create: `plugin/reaper/engine_client.py`
- Test: `ml/tests/test_engine_client.py`

**Interfaces:**
- Consumes: config dict `{"venv_python","repo_root","port"}`.
- Produces:
  - `base_url(cfg) -> str`
  - `health(cfg, timeout=1.0) -> dict | None` (GET `/health`; `None` on any failure)
  - `start_engine(cfg) -> None` (spawns `venv_python -m drum_dynamics.serve --port <port> --parent-pid <pid>` detached, cwd=`repo_root`)
  - `ensure_engine(cfg, tries=30, delay=0.5, sleep=time.sleep) -> dict | None` (returns health if up, else starts and polls; `sleep` injectable for tests)

- [ ] **Step 1: Write the failing tests**

```python
# ml/tests/test_engine_client.py
import os
import sys

sys.path.insert(0, os.path.join("plugin", "reaper"))
import engine_client


def test_base_url_uses_port():
    assert engine_client.base_url({"port": 9000}) == "http://127.0.0.1:9000"


def test_base_url_defaults_to_8765():
    assert engine_client.base_url({}) == "http://127.0.0.1:8765"


def test_health_returns_none_on_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("refused")
    monkeypatch.setattr(engine_client.urllib.request, "urlopen", boom)
    assert engine_client.health({"port": 8765}) is None


def test_ensure_engine_returns_immediately_when_up(monkeypatch):
    monkeypatch.setattr(engine_client, "health", lambda cfg, **k: {"status": "ok"})
    started = []
    monkeypatch.setattr(engine_client, "start_engine", lambda cfg: started.append(True))
    out = engine_client.ensure_engine({"port": 8765}, sleep=lambda s: None)
    assert out == {"status": "ok"}
    assert started == []  # never started, it was already up


def test_ensure_engine_starts_then_polls_until_ready(monkeypatch):
    calls = {"n": 0}
    def fake_health(cfg, **k):
        calls["n"] += 1
        return {"status": "ok"} if calls["n"] >= 3 else None
    started = []
    monkeypatch.setattr(engine_client, "health", fake_health)
    monkeypatch.setattr(engine_client, "start_engine", lambda cfg: started.append(True))
    out = engine_client.ensure_engine({"port": 8765}, tries=5, sleep=lambda s: None)
    assert out == {"status": "ok"}
    assert started == [True]  # started exactly once


def test_ensure_engine_gives_up_and_returns_none(monkeypatch):
    monkeypatch.setattr(engine_client, "health", lambda cfg, **k: None)
    monkeypatch.setattr(engine_client, "start_engine", lambda cfg: None)
    assert engine_client.ensure_engine({"port": 8765}, tries=3, sleep=lambda s: None) is None


def test_start_engine_spawns_expected_argv(monkeypatch, tmp_path):
    seen = {}
    def fake_popen(argv, **kwargs):
        seen["argv"] = argv
        seen["cwd"] = kwargs.get("cwd")
        class P: pass
        return P()
    monkeypatch.setattr(engine_client.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(engine_client.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(engine_client, "open", lambda *a, **k: open(os.devnull, "a"), raising=False)
    cfg = {"venv_python": "/venv/py", "repo_root": str(tmp_path), "port": 8765}
    engine_client.start_engine(cfg)
    assert seen["argv"][:4] == ["/venv/py", "-m", "drum_dynamics.serve", "--port"]
    assert "8765" in seen["argv"]
    assert seen["cwd"] == str(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest ml/tests/test_engine_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine_client'`.

- [ ] **Step 3: Write `engine_client.py`**

```python
# plugin/reaper/engine_client.py
"""Engine lifecycle + HTTP for the Dynamics Needed panel.

Pure stdlib, no Reaper API, so it runs under normal pytest. Extracted from the
old ReaScript so the network/subprocess logic is unit-tested.
"""
import json
import os
import subprocess
import time
import urllib.request


def base_url(cfg):
    return "http://127.0.0.1:{}".format(cfg.get("port", 8765))


def health(cfg, timeout=1.0):
    try:
        with urllib.request.urlopen(base_url(cfg) + "/health", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def start_engine(cfg):
    runtime = os.path.join(cfg["repo_root"], "plugin", "reaper", ".runtime")
    os.makedirs(runtime, exist_ok=True)
    log = open(os.path.join(runtime, "engine.log"), "a")
    subprocess.Popen(
        [cfg["venv_python"], "-m", "drum_dynamics.serve",
         "--port", str(cfg.get("port", 8765)), "--parent-pid", str(os.getpid())],
        cwd=cfg["repo_root"], stdout=log, stderr=log,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )


def predict(cfg, request, timeout=60.0):
    data = json.dumps(request).encode()
    req = urllib.request.Request(base_url(cfg) + "/predict", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ensure_engine(cfg, tries=30, delay=0.5, sleep=time.sleep):
    h = health(cfg)
    if h:
        return h
    start_engine(cfg)
    for _ in range(tries):
        sleep(delay)
        h = health(cfg)
        if h:
            return h
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest ml/tests/test_engine_client.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add plugin/reaper/engine_client.py ml/tests/test_engine_client.py
git commit -m "feat(reaper): extract engine_client with lifecycle + HTTP, unit-tested"
```

---

### Task 2: Predict worker (debounced, coalesced, background thread)

A pure module that runs predictions off the UI thread and always computes the
*latest* request, discarding superseded ones. Deterministically testable via
`_process_once()`; the thread just calls that on a debounce.

**Files:**
- Create: `plugin/reaper/predict_worker.py`
- Test: `ml/tests/test_predict_worker.py`

**Interfaces:**
- Consumes: a `predict_fn(request_dict) -> velocities_dict` callable (in the panel this is `lambda req: dn_core.parse_velocities(engine_client.predict(cfg, req))`).
- Produces: `PredictWorker` with
  - `submit(seq: int, request: dict) -> None` (overwrites any un-processed pending request)
  - `_process_once() -> None` (processes the latest pending; no-op if none)
  - `result() -> tuple[int, dict] | None` (latest `(seq, velocities)`)
  - `last_error() -> str | None`
  - `start() -> None` / `stop() -> None` (thread lifecycle; thread calls `_process_once` after `debounce` seconds)

- [ ] **Step 1: Write the failing tests**

```python
# ml/tests/test_predict_worker.py
import os
import sys

sys.path.insert(0, os.path.join("plugin", "reaper"))
from predict_worker import PredictWorker


def test_no_pending_is_noop():
    w = PredictWorker(lambda req: {0: 1})
    w._process_once()
    assert w.result() is None


def test_processes_latest_and_coalesces():
    calls = []
    w = PredictWorker(lambda req: (calls.append(req), {0: req["t"]})[1])
    w.submit(1, {"t": 1.0})
    w.submit(2, {"t": 1.5})   # supersedes seq 1 before processing
    w._process_once()
    assert len(calls) == 1 and calls[0]["t"] == 1.5
    assert w.result() == (2, {0: 1.5})


def test_error_keeps_last_good_result():
    w = PredictWorker(lambda req: {0: 42})
    w.submit(1, {"t": 1.0})
    w._process_once()
    assert w.result() == (1, {0: 42})

    def boom(req):
        raise RuntimeError("predict failed")
    w._predict_fn = boom
    w.submit(2, {"t": 2.0})
    w._process_once()
    assert w.result() == (1, {0: 42})          # unchanged
    assert "predict failed" in w.last_error()


def test_success_clears_previous_error():
    w = PredictWorker(lambda req: {0: 7})
    w._predict_fn = lambda req: (_ for _ in ()).throw(RuntimeError("x"))
    w.submit(1, {"t": 1.0}); w._process_once()
    assert w.last_error() is not None
    w._predict_fn = lambda req: {0: 7}
    w.submit(2, {"t": 1.0}); w._process_once()
    assert w.last_error() is None
    assert w.result() == (2, {0: 7})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest ml/tests/test_predict_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'predict_worker'`.

- [ ] **Step 3: Write `predict_worker.py`**

```python
# plugin/reaper/predict_worker.py
"""Background predict worker: debounced, coalescing, pure (no Reaper API).

The UI thread calls submit() on every change; the worker thread runs the most
recent request after a short debounce, discarding superseded ones, and stashes
the result for the UI thread to read. Deterministic via _process_once().
"""
import threading
import time


class PredictWorker:
    def __init__(self, predict_fn, debounce=0.15):
        self._predict_fn = predict_fn
        self._debounce = debounce
        self._lock = threading.Lock()
        self._pending = None          # (seq, request) or None
        self._result = None           # (seq, velocities) or None
        self._error = None            # str or None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def submit(self, seq, request):
        with self._lock:
            self._pending = (seq, request)
        self._wake.set()

    def _take_pending(self):
        with self._lock:
            p = self._pending
            self._pending = None
            return p

    def _process_once(self):
        p = self._take_pending()
        if p is None:
            return
        seq, request = p
        try:
            velocities = self._predict_fn(request)
        except Exception as e:            # keep last good result on failure
            with self._lock:
                self._error = str(e)
            return
        with self._lock:
            self._result = (seq, velocities)
            self._error = None

    def result(self):
        with self._lock:
            return self._result

    def last_error(self):
        with self._lock:
            return self._error

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                break
            time.sleep(self._debounce)     # debounce; later submits coalesce
            self._process_once()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest ml/tests/test_predict_worker.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add plugin/reaper/predict_worker.py ml/tests/test_predict_worker.py
git commit -m "feat(reaper): add debounced coalescing predict worker, unit-tested"
```

---

### Task 3: Walking skeleton (ImGui loop + engine health) — GO/NO-GO GATE

Prove the Python defer loop + ImGui + in-process engine call work in Reaper
before building any UI. **This is a manual checkpoint** — there is no pytest for
Reaper glue. If the defer loop or ImGui bindings do not work from Reaper's
Python, STOP and escalate: the fallback is the Lua panel from the spec.

**Files:**
- Create: `plugin/reaper/dynamics_needed.py` (start fresh; the old CSV-dialog version is replaced over Tasks 3-6)

**Interfaces:**
- Consumes: `engine_client.ensure_engine`, `engine_client.health`; config at `RPR_GetResourcePath()/dynamics_needed_config.json`.
- Produces: a registered Reaper action that opens a persistent ImGui window.

> **ImGui API note:** exact ReaImGui function names/arities are version-specific. In Reaper, run the action **"ReaImGui: ReaScript documentation"** and open the **ReaImGui demo** to confirm signatures for the installed version. From Python they are exposed with the `RPR_` prefix (e.g. `RPR_ImGui_CreateContext`). The Python defer pattern is `RPR_defer("loop()")` (Reaper eval's the string in the script namespace — you cannot pass a Python callable). Confirming these three things (context creation, Begin/End, defer) is the whole point of this task.

- [ ] **Step 1: Write the skeleton**

```python
# plugin/reaper/dynamics_needed.py
"""Dynamics Needed - ReaImGui panel (walking skeleton).

ASCII only. No __file__. RPR_* return tuples. Runs under Reaper's embedded
Python. Requires ReaImGui (install via ReaPack).
"""
import json
import os
import sys

from reaper_python import *  # noqa: F401,F403


def _load_config():
    cfg_path = os.path.join(RPR_GetResourcePath(), "dynamics_needed_config.json")
    with open(cfg_path) as fh:
        return json.load(fh)


def _init():
    cfg = _load_config()
    sys.path.insert(0, os.path.join(cfg["repo_root"], "plugin", "reaper"))
    import engine_client
    state = {
        "cfg": cfg,
        "engine_client": engine_client,
        "ctx": RPR_ImGui_CreateContext("Dynamics Needed"),
        "health": engine_client.ensure_engine(cfg),
        "open": True,
    }
    return state


STATE = None


def _active_take_note_count():
    editor = RPR_MIDIEditor_GetActive()
    take = RPR_MIDIEditor_GetTake(editor) if editor else None
    if not take:
        return None
    _, _, note_count, _, _ = RPR_MIDI_CountEvts(take, 0, 0, 0)
    return note_count


def loop():
    ctx = STATE["ctx"]
    visible, STATE["open"] = RPR_ImGui_Begin(ctx, "Dynamics Needed", True)
    if visible:
        n = _active_take_note_count()
        RPR_ImGui_Text(ctx, "Notes in active take: {}".format("none" if n is None else n))
        h = STATE["health"]
        RPR_ImGui_Text(ctx, "Engine: {}".format("ready" if h else "unreachable"))
        if h:
            RPR_ImGui_Text(ctx, "Styles: {}".format(len(h.get("styles", []))))
        RPR_ImGui_End(ctx)
    if STATE["open"]:
        RPR_defer("loop()")


def main():
    global STATE
    try:
        STATE = _init()
    except Exception as e:
        RPR_ShowConsoleMsg("Dynamics Needed: init failed: {}\n".format(e))
        return
    RPR_defer("loop()")


main()
```

- [ ] **Step 2: Register and run in Reaper (manual)**

1. Ensure ReaImGui is installed (ReaPack -> browse packages -> ReaImGui) and config exists (`.venv/bin/python plugin/reaper/setup_reaper.py`).
2. Actions -> Show action list -> New action -> Load ReaScript -> select `plugin/reaper/dynamics_needed.py`.
3. Open a MIDI item in the MIDI editor, then run the action.

Expected: a window titled "Dynamics Needed" opens and stays open, showing the
active take's note count and "Engine: ready" (styles count > 0). Dragging the
window and closing it (title-bar X) behaves normally; closing stops the defer
loop (no runaway console spam).

- [ ] **Step 3: GO/NO-GO decision**

- **GO:** window renders, updates live (change MIDI selection -> note count reflects it), engine reachable. Proceed to Task 4.
- **NO-GO:** defer loop errors, ImGui names differ irreconcilably, or the window won't render. STOP. Record the failure in the spec's "Known risk" section and escalate to switch to the Lua design.

- [ ] **Step 4: Commit (only on GO)**

```bash
git add plugin/reaper/dynamics_needed.py
git commit -m "feat(reaper): ImGui walking skeleton (engine health + take read)"
```

---

### Task 4: Controls, genre/style wiring, per-track persistence

Build the real controls onto the skeleton. Reuses the already-tested
`dn_core.genres_from_styles` / `filter_styles_by_genre`. Reaper glue -> manual
verification.

**Files:**
- Modify: `plugin/reaper/dynamics_needed.py`

**Interfaces:**
- Consumes: `dn_core.genres_from_styles(styles)`, `dn_core.filter_styles_by_genre(styles, genre)`; health `styles`/`models`.
- Produces: a `params` dict in `STATE` = `{"genre","style","model","temperature","blend","beat_type"}`, persisted per track.

- [ ] **Step 1: Add persistence + defaults helpers**

Add to `dynamics_needed.py` (import `dn_core` in `_init` alongside `engine_client`):

```python
def _active_take():
    editor = RPR_MIDIEditor_GetActive()
    return RPR_MIDIEditor_GetTake(editor) if editor else None


def _track_key(take):
    item = RPR_GetMediaItemTake_Item(take)
    track = RPR_GetMediaItem_Track(item)
    _, _, _, guid, _ = RPR_GetSetMediaTrackInfo_String(track, "GUID", "", False)
    return "dn_last_" + guid


def _load_last(key):
    _, _, _, _, val, _ = RPR_GetProjExtState(0, "DynamicsNeeded", key, "", 4096)
    try:
        return json.loads(val) if val else {}
    except Exception:
        return {}


def _save_last(key, params):
    RPR_SetProjExtState(0, "DynamicsNeeded", key, json.dumps(params))


def _default_params(health, last):
    styles = health.get("styles", []) or ["rock"]
    genres = health.get("genres", []) or ["rock"]
    return {
        "genre": last.get("genre", genres[0]),
        "style": last.get("style", styles[0]),
        "model": last.get("model", "mdn"),
        "temperature": float(last.get("temperature", 1.0)),
        "blend": float(last.get("blend", 0.8)),
        "beat_type": last.get("beat_type", "beat"),
    }
```

- [ ] **Step 2: Render controls in `loop()`**

Replace the skeleton body inside `if visible:` with the controls. Use the ImGui
signatures confirmed in Task 2 (Combo/RadioButton/SliderDouble/Checkbox each
return `(changed, new_value)`-style tuples in Python — unpack per the demo).

```python
        p = STATE["params"]
        h = STATE["health"]
        genres = h.get("genres", []) or ["rock"]
        styles_for_genre = dn_core.filter_styles_by_genre(h.get("styles", []), p["genre"]) or [p["style"]]

        # Genre combo
        gi = genres.index(p["genre"]) if p["genre"] in genres else 0
        changed, gi = RPR_ImGui_Combo(ctx, "Genre", gi, "\x00".join(genres) + "\x00", len(genres))
        if changed:
            p["genre"] = genres[gi]
            new_styles = dn_core.filter_styles_by_genre(h.get("styles", []), p["genre"])
            p["style"] = new_styles[0] if new_styles else p["style"]
            STATE["dirty"] = True

        # Style combo (filtered by genre)
        si = styles_for_genre.index(p["style"]) if p["style"] in styles_for_genre else 0
        changed, si = RPR_ImGui_Combo(ctx, "Style", si, "\x00".join(styles_for_genre) + "\x00", len(styles_for_genre))
        if changed:
            p["style"] = styles_for_genre[si]; STATE["dirty"] = True

        # Model radio
        if RPR_ImGui_RadioButton(ctx, "LGBM", p["model"] == "lgbm"):
            p["model"] = "lgbm"; STATE["dirty"] = True
        RPR_ImGui_SameLine(ctx)
        if RPR_ImGui_RadioButton(ctx, "MDN", p["model"] == "mdn"):
            p["model"] = "mdn"; STATE["dirty"] = True

        # Sliders
        changed, p["temperature"] = RPR_ImGui_SliderDouble(ctx, "Temp", p["temperature"], 0.0, 2.0)
        if changed: STATE["dirty"] = True
        changed, p["blend"] = RPR_ImGui_SliderDouble(ctx, "Blend", p["blend"], 0.0, 1.0)
        if changed: STATE["dirty"] = True

        # Is a fill?
        changed, is_fill = RPR_ImGui_Checkbox(ctx, "Is a fill?", p["beat_type"] == "fill")
        if changed:
            p["beat_type"] = "fill" if is_fill else "beat"; STATE["dirty"] = True

        # Live toggle
        _, STATE["live"] = RPR_ImGui_Checkbox(ctx, "Live", STATE["live"])
```

In `_init`, initialize `STATE["params"]`, `STATE["live"] = True`, `STATE["dirty"] = False`, and load per-track params:

```python
    take = None
    editor = RPR_MIDIEditor_GetActive()
    if editor:
        take = RPR_MIDIEditor_GetTake(editor)
    last = _load_last(_track_key(take)) if take else {}
    state["params"] = _default_params(state["health"] or {}, last)
    state["live"] = True
    state["dirty"] = False
```

- [ ] **Step 3: Verify in Reaper (manual)**

Reload the action. Expected: Genre/Style/Model/Temp/Blend/"Is a fill?"/Live
controls render; changing Genre repopulates Style; "Is a fill?" starts unchecked;
Temp starts 1.0, Blend 0.8. (Persistence is exercised in Task 6 once Apply saves.)

- [ ] **Step 4: Confirm pure-logic tests still pass**

Run: `python -m pytest ml/tests/test_reaper_core.py -v`
Expected: PASS (unchanged `dn_core` tests).

- [ ] **Step 5: Commit**

```bash
git add plugin/reaper/dynamics_needed.py
git commit -m "feat(reaper): panel controls, genre/style wiring, per-track params"
```

---

### Task 5: Live preview (worker wiring + velocity lane)

Wire the predict worker to the controls and draw the before/after velocity lane.

**Files:**
- Modify: `plugin/reaper/dynamics_needed.py`

**Interfaces:**
- Consumes: `predict_worker.PredictWorker`, `engine_client.predict`, `dn_core.build_predict_request` / `resolve_target_indices` / `parse_velocities`; `RPR_ImGui_GetWindowDrawList` + `RPR_ImGui_DrawList_AddRectFilled`.
- Produces: `STATE["preview"]` = `{index: velocity}` predicted map for the target notes; consumed by Task 6's Apply.

- [ ] **Step 1: Add note reading + tempo/sig helpers**

```python
def _read_notes(take):
    _, _, note_count, _, _ = RPR_MIDI_CountEvts(take, 0, 0, 0)
    notes = []
    for i in range(note_count):
        ok, _, _, sel, _, startppq, _, _, pitch, vel = RPR_MIDI_GetNote(
            take, i, 0, 0, 0.0, 0.0, 0, 0, 0)
        if not ok:
            continue
        onset = RPR_MIDI_GetProjTimeFromPPQPos(take, startppq)
        notes.append({"index": i, "pitch": int(pitch), "onset_sec": float(onset),
                      "velocity": int(vel), "selected": bool(sel)})
    return notes


def _tempo_and_sig(take):
    item = RPR_GetMediaItemTake_Item(take)
    pos = RPR_GetMediaItemInfo_Value(item, "D_POSITION")
    _, _, num, denom, bpm = RPR_TimeMap_GetTimeSigAtTime(0, pos, 0, 0, 0.0)
    return float(bpm), "{}-{}".format(int(num), int(denom))
```

- [ ] **Step 2: Create the worker in `_init`**

```python
    import predict_worker
    ec, cfgd = state["engine_client"], state["cfg"]
    state["worker"] = predict_worker.PredictWorker(
        lambda req: dn_core.parse_velocities(ec.predict(cfgd, req)))
    state["worker"].start()
    state["seq"] = 0
    state["preview"] = {}          # index -> predicted velocity
    state["notes"] = []            # last snapshot read on the main thread
```

- [ ] **Step 3: Queue predictions + read results in `loop()`**

After the controls block, still inside `if visible:`:

```python
        take = _active_take()
        notes = _read_notes(take) if take else []
        STATE["notes"] = notes
        targets = set(dn_core.resolve_target_indices(notes))
        for nt in notes:
            nt["selected"] = nt["index"] in targets

        # Build a change signature so we only re-predict on real changes.
        sig = (json.dumps(STATE["params"], sort_keys=True),
               tuple((nt["index"], nt["pitch"], round(nt["onset_sec"], 4)) for nt in notes if nt["selected"]))
        force = STATE.pop("force_preview", False)
        if take and (force or (STATE["live"] and sig != STATE.get("last_sig"))):
            STATE["last_sig"] = sig
            bpm, time_sig = _tempo_and_sig(take)
            req = dn_core.build_predict_request(
                STATE["params"]["model"], STATE["params"]["style"],
                STATE["params"]["temperature"], STATE["params"]["blend"],
                STATE["params"]["beat_type"], bpm, time_sig, notes)
            STATE["seq"] += 1
            STATE["worker"].submit(STATE["seq"], req)

        res = STATE["worker"].result()
        if res is not None:
            STATE["preview"] = res[1]

        # [Preview] button forces a fresh predict (also rerolls stochastic MDN)
        if RPR_ImGui_Button(ctx, "Preview"):
            STATE["force_preview"] = True
```

- [ ] **Step 4: Draw the velocity lane**

```python
        target_notes = [nt for nt in notes if nt["selected"]]
        RPR_ImGui_Text(ctx, "velocities ({} target notes)".format(len(target_notes)))
        dl = RPR_ImGui_GetWindowDrawList(ctx)
        x, y = RPR_ImGui_GetCursorScreenPos(ctx)
        w = 260.0
        lane_h = 60.0
        n = max(1, len(target_notes))
        bw = w / n
        cur_col = 0x8080807F       # faint gray (RGBA)
        pred_col = 0x33CCFFFF      # cyan
        for i, nt in enumerate(target_notes):
            bx = x + i * bw
            cur = nt["velocity"] / 127.0
            pred = STATE["preview"].get(nt["index"], nt["velocity"]) / 127.0
            RPR_ImGui_DrawList_AddRectFilled(dl, bx, y + lane_h * (1 - cur), bx + bw - 1, y + lane_h, cur_col)
            RPR_ImGui_DrawList_AddRectFilled(dl, bx, y + lane_h * (1 - pred), bx + bw - 1, y + lane_h, pred_col)
        RPR_ImGui_Dummy(ctx, w, lane_h)   # reserve layout space under the drawing
```

- [ ] **Step 5: Verify in Reaper (manual)**

Reload. Select notes; expected: two overlaid bar rows (faint = current, cyan =
predicted). Dragging Temp/Blend updates the cyan row ~150ms after you stop
(panel stays responsive). Toggling Live off freezes auto-updates; `[Preview]`
forces a refresh and rerolls MDN. Verify the worker never calls `RPR_*` (only
`engine_client.predict` + `dn_core`).

- [ ] **Step 6: Commit**

```bash
git add plugin/reaper/dynamics_needed.py
git commit -m "feat(reaper): live velocity preview via background worker + lane render"
```

---

### Task 6: Apply with undo, status line, cleanup

Write previewed velocities to the take, add the colored status line + Retry, save
per-track params on Apply, and remove the last remnants of the old dialog.

**Files:**
- Modify: `plugin/reaper/dynamics_needed.py`
- Delete: any leftover CSV-dialog helpers (there should be none if Task 3 started fresh — verify)

**Interfaces:**
- Consumes: `STATE["preview"]`, `STATE["notes"]`, `RPR_Undo_BeginBlock`/`EndBlock`, `RPR_MIDI_GetNote`/`SetNote`/`Sort`.
- Produces: applied velocities on the take; saved per-track params.

- [ ] **Step 1: Add apply helper (velocity-only rewrite, preserves the correctness fix)**

```python
def _apply(take, velocities):
    # This binding always writes every field, so re-read each note and rewrite
    # ONLY velocity (leave selection/mute/position/pitch untouched).
    for idx, vel in velocities.items():
        ok, _, _, sel, muted, startppq, endppq, chan, pitch, _ = RPR_MIDI_GetNote(
            take, idx, 0, 0, 0.0, 0.0, 0, 0, 0)
        if ok:
            RPR_MIDI_SetNote(take, idx, sel, muted, startppq, endppq, chan, pitch,
                             int(vel), True)   # noSort; one sort after the loop
    RPR_MIDI_Sort(take)
```

- [ ] **Step 2: Add status line + Apply/Retry buttons in `loop()`**

Before `RPR_ImGui_End(ctx)`:

```python
        # Status line
        if STATE["health"] is None:
            RPR_ImGui_TextColored(ctx, 0xFF4040FF, "Engine unreachable")
            RPR_ImGui_SameLine(ctx)
            if RPR_ImGui_Button(ctx, "Retry"):
                STATE["health"] = STATE["engine_client"].ensure_engine(STATE["cfg"])
        elif STATE["worker"].last_error():
            RPR_ImGui_TextColored(ctx, 0xFFAA40FF, "Predict failed: " + STATE["worker"].last_error())
        elif not _active_take():
            RPR_ImGui_TextColored(ctx, 0xAAAAAAFF, "Open a MIDI item to edit")
        else:
            RPR_ImGui_TextColored(ctx, 0x40FF40FF, "ready")

        # Apply
        can_apply = bool(_active_take()) and bool(STATE["preview"])
        if not can_apply:
            RPR_ImGui_BeginDisabled(ctx, True)
        if RPR_ImGui_Button(ctx, "Apply") and can_apply:
            take = _active_take()
            RPR_Undo_BeginBlock()
            _apply(take, STATE["preview"])
            RPR_Undo_EndBlock("Dynamics Needed: restore velocities", -1)
            _save_last(_track_key(take), STATE["params"])
        if not can_apply:
            RPR_ImGui_EndDisabled(ctx)
```

- [ ] **Step 3: Clean shutdown**

When the window closes, stop the worker. At the end of `loop()`:

```python
    if not STATE["open"]:
        STATE["worker"].stop()
        return
    RPR_defer("loop()")
```

- [ ] **Step 4: ASCII audit**

Run: `python -c "open('plugin/reaper/dynamics_needed.py','rb').read().decode('ascii')"`
Expected: no error (file is pure ASCII). Repeat for `engine_client.py` and `predict_worker.py`.

- [ ] **Step 5: Verify end-to-end in Reaper (manual)**

Reload. Select drum notes -> preview shows -> Apply writes velocities (single
undo step; only velocities change, positions/pitches intact). Reopen the panel
on the same track -> last params restored. Engine-down -> red status + Retry
recovers. Close window -> defer loop stops (no console spam), worker thread ends.

- [ ] **Step 6: Full suite + commit**

Run: `python -m pytest ml/tests/test_engine_client.py ml/tests/test_predict_worker.py ml/tests/test_reaper_core.py -v`
Expected: PASS.

```bash
git add plugin/reaper/dynamics_needed.py
git commit -m "feat(reaper): apply with undo, status line, per-track save, cleanup"
```

---

## Self-Review

**Spec coverage:**
- Architecture/files (spec §1): `engine_client.py` (Task 1), `dn_core.py` kept (Tasks 4-5 consume it), `dynamics_needed.py` rewrite (Tasks 3-6). ✓
- Concurrency/worker (spec §2): Task 2 (worker, coalescing, "apply what you saw" via `STATE["preview"]` used by Apply in Task 6, not a fresh predict). ✓
- Data flow (spec §3): open->ensure_engine+dropdowns (Tasks 3-4); per-frame read+target+draw (Task 5); param change/Preview reroll (Task 5); Apply with undo velocity-only (Task 6). ✓
- UI (spec §4): controls + genre/style filter + defaults + Live + per-track persistence (Task 4); velocity lane (Task 5). ✓
- Error handling (spec §5): status line states + Retry + disabled Apply (Task 6); predict failure keeps last preview (Task 2 + Task 6 status). ✓
- Testing (spec §6): engine_client + predict_worker pytest (Tasks 1-2); dn_core tests kept (Task 4); ImGui glue manual (Tasks 3,5,6). ✓
- Walking skeleton go/no-go (spec §7): Task 3. ✓

**Placeholder scan:** No TBD/TODO; every code step has concrete code; manual Reaper steps give exact expected outcomes. The one deliberate flag is the ImGui-signature confirmation in Task 3 — inherent to a version-specific binding, and the skeleton task exists precisely to pin it down.

**Type consistency:** `base_url/health/start_engine/ensure_engine/predict` (Task 1) used identically in Tasks 3/5. `PredictWorker.submit(seq,request)/result()->(seq,vel)/last_error()/start()/stop()` (Task 2) used consistently in Tasks 5/6. `STATE["preview"]` is `{index: velocity}` produced in Task 5, consumed by `_apply` in Task 6. `params` keys match `dn_core.build_predict_request` argument order.
