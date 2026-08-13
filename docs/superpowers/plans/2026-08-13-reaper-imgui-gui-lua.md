# Dynamics Needed ReaImGui GUI — Lua Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Rewrite the abandoned Python ImGui panel as a Lua ReaImGui panel (the Python path is impossible — REAPER's `reaper_python.py` never exposes `RPR_ImGui_*`; see the spec's "Pivot to Lua").

**Architecture:** The Python inference engine (`drum_dynamics.serve`) and `setup_reaper.py` are unchanged. The registered action becomes `dynamics_needed.lua`, using `reaper.ImGui_*` via `require 'imgui'`. Lua has no threads, no HTTP, no JSON: predictions are single-threaded, debounced, blocking `curl` calls via `reaper.ExecProcess`; JSON via a bundled `json.lua`; pure request logic in `dn_core.lua` (self-tested with the local `lua` interpreter). The Python plugin files and their pytest are removed.

**Tech Stack:** Lua 5.4 (REAPER embedded), ReaImGui (ReaPack), `curl` (`/usr/bin/curl`), `reaper.ExecProcess`, local `lua` at `~/homebrew/bin/lua` for tests.

**Spec:** `docs/superpowers/specs/2026-08-13-reaper-imgui-gui-design.md` (see "Pivot to Lua").

## Global Constraints

- **ReaImGui must be installed** via ReaPack (it currently is NOT). Nothing runs in REAPER until it is. Version pin: `local ImGui = require 'imgui' '0.9'` (a floor; newer installs still satisfy it).
- **ReaImGui Begin/End rule:** call `ImGui.End(ctx)` ONLY when `Begin` returned visible. Loop shape:
  ```lua
  local function loop()
    local visible, open = ImGui.Begin(ctx, 'Dynamics Needed', true)
    if visible then
      -- draw ...
      ImGui.End(ctx)          -- only when visible
    end
    if open then reaper.defer(loop) else <cleanup> end
  end
  reaper.defer(loop)
  ```
- **No `__file__`-style assumptions:** locate config via `reaper.GetResourcePath() .. '/dynamics_needed_config.json'`. Config shape: `{"venv_python":str,"repo_root":str,"port":int}`.
- **`reaper.ExecProcess(cmd, timeout_ms)`** returns a string `"<exitcode>\n<stdout>"` (or nil). Parse: split on the FIRST newline; before = exit code, after = stdout body.
- **HTTP:** `curl` at `/usr/bin/curl`. Health: `curl -s -m 1 <base>/health`. Predict: POST with `--data-binary @<tempfile>` and `-H "Content-Type: application/json"`. `base = "http://127.0.0.1:" .. port`.
- **Engine autostart (detached, sets cwd):**
  `/bin/bash -lc "cd '<repo_root>' && nohup '<venv_python>' -m drum_dynamics.serve --port <port> >> '<repo_root>/plugin/reaper/.runtime/engine.log' 2>&1 &"` via ExecProcess with a short timeout (~200ms; the `&` detaches).
- **Single-threaded debounce:** track `last_change = reaper.time_precise()`; fire a predict only when dirty AND `reaper.time_precise() - last_change > 0.15`. `[Preview]` forces immediately (and rerolls stochastic MDN).
- **Defaults:** Temp 0.0-2.0 (default 1.0); Blend 0.0-1.0 (default 0.8); "Is a fill?" default OFF (`beat_type="beat"`); Live default ON. Per-track saved params (ProjExtState, keyed on track GUID) override defaults.
- **Engine contract (unchanged):** `GET /health` -> `{"status":"ok","models":["lgbm","mdn"],"styles":[...],"genres":[...]}`; `POST /predict` (request dict) -> `{"velocities":{"<index>":<int>}}`.
- **Lua tests:** run with `~/homebrew/bin/lua plugin/reaper/test_dn_core.lua` (exit 0, prints "ok").
- **ImGui widget return shapes (Lua):** `changed, idx = ImGui.Combo(ctx, label, idx0based, items_nullsep)`; `changed, v = ImGui.SliderDouble(ctx, label, v, min, max)`; `changed, b = ImGui.Checkbox(ctx, label, b)`; `clicked = ImGui.RadioButton(ctx, label, active)`; `clicked = ImGui.Button(ctx, label)`; `ImGui.Text(ctx, s)`; `ImGui.TextColored(ctx, 0xRRGGBBAA, s)`; `ImGui.SameLine(ctx)`; `ImGui.BeginDisabled(ctx, disabled)` / `ImGui.EndDisabled(ctx)`; `dl = ImGui.GetWindowDrawList(ctx)`; `ImGui.DrawList_AddRectFilled(dl, x1,y1,x2,y2, 0xRRGGBBAA)`; `x,y = ImGui.GetCursorScreenPos(ctx)`; `ImGui.Dummy(ctx, w, h)`. Implementers MUST confirm exact arities against the installed version via the "ReaImGui: ReaScript documentation" action / bundled demo; Task L2's skeleton is the validation point.

---

### Task L1: `json.lua` + `dn_core.lua` (pure) + self-tests

**Files:**
- Create: `plugin/reaper/json.lua` (minimal JSON encode/decode)
- Create: `plugin/reaper/dn_core.lua` (pure request/response logic)
- Create: `plugin/reaper/test_dn_core.lua` (assert-based self-test, runs under `lua`)

**Interfaces produced:**
- `json.encode(value) -> string`, `json.decode(string) -> value` (objects->tables keyed by string; arrays->1-based lists; numbers->number; true/false/null).
- `dn_core.genres_from_styles(styles)`, `.filter_styles_by_genre(styles, genre)`, `.resolve_target_indices(notes)`, `.build_predict_request(model,style,temperature,blend,beat_type,bpm,time_signature,notes)`, `.parse_velocities(response)`.

- [ ] **Step 1: Write `json.lua`** — reproduce the well-known rxi/json.lua (MIT) OR a compact equivalent. It must round-trip: nested objects, arrays, strings (with escaping), integers/floats, booleans, null. Keep it dependency-free.

- [ ] **Step 2: Write `dn_core.lua`**

```lua
local M = {}

local function genre_of(style) return style:match('^[^/]+') end

function M.genres_from_styles(styles)
  local set = {}
  for _, s in ipairs(styles) do set[genre_of(s)] = true end
  local out = {}
  for g in pairs(set) do out[#out + 1] = g end
  table.sort(out)
  return out
end

function M.filter_styles_by_genre(styles, genre)
  local out = {}
  for _, s in ipairs(styles) do
    if genre_of(s) == genre then out[#out + 1] = s end
  end
  return out
end

function M.resolve_target_indices(notes)
  local sel = {}
  for _, n in ipairs(notes) do if n.selected then sel[#sel + 1] = n.index end end
  if #sel > 0 then return sel end
  local all = {}
  for _, n in ipairs(notes) do all[#all + 1] = n.index end
  return all
end

function M.build_predict_request(model, style, temperature, blend, beat_type, bpm, time_signature, notes)
  local blend_c = blend
  if blend_c < 0.0 then blend_c = 0.0 elseif blend_c > 1.0 then blend_c = 1.0 end
  local temp_c = temperature
  if temp_c < 0.0 then temp_c = 0.0 end
  return {
    model = model, style = style,
    temperature = temp_c, blend = blend_c,
    beat_type = beat_type, bpm = bpm + 0.0,
    time_signature = time_signature, notes = notes,
  }
end

function M.parse_velocities(response)
  local out = {}
  for k, v in pairs(response.velocities) do out[tonumber(k)] = math.floor(v) end
  return out
end

return M
```

- [ ] **Step 3: Write `test_dn_core.lua`** — mirror the Python pytest cases (`ml/tests/test_reaper_core.py`):

```lua
package.path = package.path .. ';' .. (arg[0]:gsub('[^/]+$', '')) .. '?.lua'
local dn = require 'dn_core'
local json = require 'json'
local function eq(a, b, msg) assert(a == b, (msg or '') .. ' expected ' .. tostring(b) .. ' got ' .. tostring(a)) end

-- genres_from_styles
local g = dn.genres_from_styles({'rock/indie', 'rock', 'jazz/swing'})
eq(#g, 2); eq(g[1], 'jazz'); eq(g[2], 'rock')

-- filter_styles_by_genre
local f = dn.filter_styles_by_genre({'rock', 'rock/indie', 'jazz/swing'}, 'rock')
eq(#f, 2); eq(f[1], 'rock'); eq(f[2], 'rock/indie')

-- resolve_target_indices prefers selected, else all
local sel = dn.resolve_target_indices({{index=0, selected=true}, {index=1, selected=false}})
eq(#sel, 1); eq(sel[1], 0)
local all = dn.resolve_target_indices({{index=3, selected=false}, {index=4, selected=false}})
eq(#all, 2); eq(all[1], 3); eq(all[2], 4)

-- build_predict_request shape + clamps
local req = dn.build_predict_request('mdn', 'rock/indie', 1.2, 0.8, 'fill', 128.0, '4-4', {})
eq(req.model, 'mdn'); eq(req.style, 'rock/indie'); eq(req.temperature, 1.2)
eq(req.blend, 0.8); eq(req.beat_type, 'fill'); eq(req.bpm, 128.0); eq(req.time_signature, '4-4')
local c = dn.build_predict_request('mdn', 'rock', -0.5, 2.0, 'beat', 120.0, '4-4', {})
eq(c.temperature, 0.0); eq(c.blend, 1.0)
local c2 = dn.build_predict_request('mdn', 'rock', 1.0, -1.0, 'beat', 120.0, '4-4', {})
eq(c2.blend, 0.0)

-- parse_velocities
local pv = dn.parse_velocities({velocities = {['0'] = 97, ['3'] = 12}})
eq(pv[0], 97); eq(pv[3], 12)

-- json round-trip
local rt = json.decode(json.encode({a = 1, b = {'x', 'y'}, c = true}))
eq(rt.a, 1); eq(rt.b[1], 'x'); eq(rt.b[2], 'y'); eq(rt.c, true)

print('ok')
```

- [ ] **Step 4: Run** `~/homebrew/bin/lua plugin/reaper/test_dn_core.lua` → prints `ok`, exit 0.
- [ ] **Step 5: Commit** `feat(reaper): lua dn_core + json + self-tests`

---

### Task L2: Lua walking skeleton — GO/NO-GO GATE (must actually run in REAPER)

Prove `require 'imgui'`, the defer loop, config+JSON read, and a `curl` health round-trip all work. **Do NOT skip the human run this time.**

**Files:** Create `plugin/reaper/dynamics_needed.lua`

- [ ] **Step 1: Write the skeleton**

```lua
-- Dynamics Needed - ReaImGui panel (Lua walking skeleton)
local ImGui = require 'imgui' '0.9'

local sep = package.config:sub(1, 1)
local script_dir = ({reaper.get_action_context()})[2]:match('^(.*[/\\])')
package.path = package.path .. ';' .. script_dir .. '?.lua'
local json = require 'json'

local function read_config()
  local path = reaper.GetResourcePath() .. sep .. 'dynamics_needed_config.json'
  local fh = io.open(path, 'r'); if not fh then return nil end
  local body = fh:read('*a'); fh:close()
  local ok, cfg = pcall(json.decode, body)
  if ok then return cfg end
  return nil
end

local function exec(cmd, timeout_ms)
  local out = reaper.ExecProcess(cmd, timeout_ms)
  if not out then return nil, nil end
  local nl = out:find('\n')
  if not nl then return out, '' end
  return out:sub(1, nl - 1), out:sub(nl + 1)
end

local cfg = read_config()
local base = cfg and ('http://127.0.0.1:' .. tostring(cfg.port or 8765)) or nil

local function health()
  if not base then return nil end
  local code, body = exec('/usr/bin/curl -s -m 1 ' .. base .. '/health', 1500)
  if code == '0' and body and #body > 0 then
    local ok, h = pcall(json.decode, body)
    if ok and h and h.status == 'ok' then return h end
  end
  return nil
end

local ctx = ImGui.CreateContext('Dynamics Needed')
local H = health()

local function active_take_note_count()
  local editor = reaper.MIDIEditor_GetActive()
  local take = editor and reaper.MIDIEditor_GetTake(editor) or nil
  if not take then return nil end
  local _, notecnt = reaper.MIDI_CountEvts(take)
  return notecnt
end

local function loop()
  local visible, open = ImGui.Begin(ctx, 'Dynamics Needed', true)
  if visible then
    local n = active_take_note_count()
    ImGui.Text(ctx, 'Notes in active take: ' .. (n and tostring(n) or 'none'))
    ImGui.Text(ctx, 'Engine: ' .. (H and 'ready' or 'unreachable'))
    if H then ImGui.Text(ctx, 'Styles: ' .. tostring(#(H.styles or {}))) end
    ImGui.End(ctx)
  end
  if open then reaper.defer(loop) end
end

if not cfg then
  reaper.ShowConsoleMsg('Dynamics Needed: no config. Run setup_reaper.py once.\n')
else
  reaper.defer(loop)
end
```

- [ ] **Step 2 (HUMAN):** Install ReaImGui (ReaPack -> Browse packages -> ReaImGui -> install; restart REAPER). Register `dynamics_needed.lua` (Actions -> New action -> Load ReaScript). Open a MIDI item, run it.
- [ ] **Step 3 (HUMAN) GO/NO-GO:** GO = window renders, note count updates live with selection, "Engine: ready" + styles > 0. NO-GO = capture the exact ReaScript console error.
- [ ] **Step 4: Commit (on GO)** `feat(reaper): lua ImGui walking skeleton`

Note: static checks the implementer CAN run before the human step: `~/homebrew/bin/lua -e "assert(loadfile('plugin/reaper/dynamics_needed.lua'))"` (syntax-parses; will not execute REAPER APIs).

---

### Task L3: Engine autostart + non-blocking-ish "starting" status

**Files:** Modify `plugin/reaper/dynamics_needed.lua`

- [ ] **Step 1:** Add detached spawn + per-frame poll. Replace the one-shot `H = health()` with state that starts the engine if down and polls each frame with a short curl timeout while showing "Starting engine...".

```lua
local engine = { ready = false, health = nil, started = false, unreachable = false }

local function start_engine()
  local log = cfg.repo_root .. sep .. 'plugin' .. sep .. 'reaper' .. sep .. '.runtime' .. sep .. 'engine.log'
  local cmd = "/bin/bash -lc \"cd '" .. cfg.repo_root .. "' && nohup '" .. cfg.venv_python ..
    "' -m drum_dynamics.serve --port " .. tostring(cfg.port or 8765) ..
    " >> '" .. log .. "' 2>&1 &\""
  reaper.ExecProcess(cmd, 200)
end

local start_time = reaper.time_precise()
local function poll_engine()
  if engine.ready then return end
  local h = health()                       -- short-timeout curl (blocks ~<=1s)
  if h then engine.ready = true; engine.health = h; return end
  if not engine.started then start_engine(); engine.started = true end
  if reaper.time_precise() - start_time > 25 then engine.unreachable = true end
end
```

Call `poll_engine()` at the top of `loop()`; render "Starting engine..." until `engine.ready`, then the (skeleton's) health info, else if `engine.unreachable` show unreachable. Reset (`engine.ready=false, started=false, unreachable=false, start_time=now`) on a future Retry button.

- [ ] **Step 2 (HUMAN):** cold start shows "Starting engine..." then "ready" within ~15s without a hard freeze (UI updates a few times/sec).
- [ ] **Step 3: Commit** `feat(reaper): lua engine autostart + starting status`

---

### Task L4: Controls, genre/style wiring, per-track persistence

**Files:** Modify `plugin/reaper/dynamics_needed.lua`

- [ ] **Step 1:** Add per-track persistence + defaults.

```lua
local function active_take()
  local ed = reaper.MIDIEditor_GetActive(); return ed and reaper.MIDIEditor_GetTake(ed) or nil
end
local function track_key(take)
  local item = reaper.GetMediaItemTake_Item(take)
  local track = reaper.GetMediaItem_Track(item)
  local _, guid = reaper.GetSetMediaTrackInfo_String(track, 'GUID', '', false)
  return 'dn_last_' .. guid
end
local function load_last(key)
  local _, val = reaper.GetProjExtState(0, 'DynamicsNeeded', key)
  if val and #val > 0 then local ok, t = pcall(json.decode, val); if ok then return t end end
  return {}
end
local function save_last(key, params) reaper.SetProjExtState(0, 'DynamicsNeeded', key, json.encode(params)) end

local function default_params(h, last)
  local styles = h.styles or {'rock'}
  local genres = h.genres or {'rock'}
  local genre = last.genre or genres[1]
  local sfg = dn.filter_styles_by_genre(styles, genre)
  local style = last.style or (sfg[1] or styles[1])
  return { genre = genre, style = style, model = last.model or 'mdn',
           temperature = last.temperature or 1.0, blend = last.blend or 0.8,
           beat_type = last.beat_type or 'beat' }
end
```

Build `params` once when `engine.ready` and `params == nil` (health carries the genres — apply the "true fallback" only if `h.genres` empty: `if (not h.genres or #h.genres == 0) and h.styles then h.genres = dn.genres_from_styles(h.styles) end`).

- [ ] **Step 2:** Render controls (only when `engine.ready` and params built). Use the widget shapes in Global Constraints. Genre Combo filters Style Combo; changing genre resets style to `filter_styles_by_genre(styles, genre)[1]`. Model radio LGBM/MDN. Temp/Blend sliders. "Is a fill?" checkbox (maps beat_type). "Live" checkbox (default true). Any change sets `dirty=true; last_change=reaper.time_precise()`. Combo items: `table.concat(list, '\0') .. '\0'`; indices are 0-based (convert to/from Lua 1-based).

- [ ] **Step 3 (HUMAN):** controls render; genre filters style; defaults correct (Temp 1.0, Blend 0.8, fill off).
- [ ] **Step 4: Commit** `feat(reaper): lua panel controls + genre/style + per-track params`

---

### Task L5: Live preview (debounced blocking predict + velocity lane)

**Files:** Modify `plugin/reaper/dynamics_needed.lua`

- [ ] **Step 1:** Note reading + tempo/sig (main thread only).

```lua
local function read_notes(take)
  local _, notecnt = reaper.MIDI_CountEvts(take)
  local notes = {}
  for i = 0, notecnt - 1 do
    local ok, sel, _, startppq, _, _, pitch, vel = reaper.MIDI_GetNote(take, i)
    if ok then
      local onset = reaper.MIDI_GetProjTimeFromPPQPos(take, startppq)
      notes[#notes + 1] = { index = i, pitch = pitch, onset_sec = onset, velocity = vel, selected = sel }
    end
  end
  return notes
end
local function tempo_and_sig(take)
  local item = reaper.GetMediaItemTake_Item(take)
  local pos = reaper.GetMediaItemInfo_Value(item, 'D_POSITION')
  local _, _, _, num, denom, bpm = reaper.TimeMap_GetTimeSigAtTime(0, pos)
  return bpm, tostring(math.floor(num)) .. '-' .. tostring(math.floor(denom))
end
```
(Confirm `MIDI_GetNote` / `TimeMap_GetTimeSigAtTime` return arity against the ReaScript docs — Lua returns booleans + out-params.)

- [ ] **Step 2:** Predict via curl POST (blocking, debounced).

```lua
local function predict(request)
  local tmp = reaper.GetResourcePath() .. sep .. 'dn_predict_req.json'
  local fh = io.open(tmp, 'w'); fh:write(json.encode(request)); fh:close()
  local code, body = exec('/usr/bin/curl -s -m 5 -X POST -H "Content-Type: application/json" --data-binary @"' ..
    tmp .. '" ' .. base .. '/predict', 6000)
  if code == '0' and body and #body > 0 then
    local ok, resp = pcall(json.decode, body)
    if ok and resp.velocities then return dn.parse_velocities(resp), nil end
  end
  return nil, 'predict failed'
end
```

In `loop()` (when ready): read notes, mark selected via `resolve_target_indices`, compute a change signature (params + selected note index/pitch/onset). On change set `dirty`/`last_change`. When `dirty and time_precise()-last_change > 0.15` (or a `force` flag from [Preview]): build request via `dn.build_predict_request`, call `predict`, store `preview` ({index->vel}) or `predict_error`; clear dirty. `[Preview]` sets force (also rerolls MDN).

- [ ] **Step 3:** Draw the velocity lane with the draw list — faint current bars under colored predicted bars, scaled 0-127, in note order (guard `n = math.max(1, #targets)`; use `preview[idx] or note.velocity`).

- [ ] **Step 4 (HUMAN):** cyan predicted bars over faint current; dragging Temp/Blend updates ~150ms after release; `[Preview]` rerolls; Live off freezes auto-update.
- [ ] **Step 5: Commit** `feat(reaper): lua live velocity preview + lane`

---

### Task L6: Apply with undo, status line, cleanup, remove Python files

**Files:** Modify `plugin/reaper/dynamics_needed.lua`; DELETE the obsolete Python plugin + tests.

- [ ] **Step 1:** Apply (velocity-only, undo-wrapped).

```lua
local function apply(take, velocities)
  reaper.Undo_BeginBlock()
  for idx, vel in pairs(velocities) do
    local ok, sel, muted, startppq, endppq, chan, pitch = reaper.MIDI_GetNote(take, idx)
    if ok then
      reaper.MIDI_SetNote(take, idx, sel, muted, startppq, endppq, chan, pitch, math.floor(vel), true)
    end
  end
  reaper.MIDI_Sort(take)
  reaper.Undo_EndBlock('Dynamics Needed: restore velocities', -1)
end
```
(Confirm `MIDI_GetNote` return order and `MIDI_SetNote` arg order against ReaScript docs; rewrite ONLY velocity, preserve sel/muted/positions/pitch.)

- [ ] **Step 2:** Status line (green ready / orange predict_error / gray no-take / red unreachable + `[Retry]` which re-arms `poll_engine`). `[Apply]` guarded by `active_take() and next(preview) ~= nil`; wrap with `BeginDisabled/EndDisabled`; on click apply `preview` and `save_last(track_key(take), params)`. Apply uses `preview` (apply-what-you-saw), never a fresh predict.

- [ ] **Step 3:** Delete obsolete Python files and their tests:
  `git rm plugin/reaper/dynamics_needed.py plugin/reaper/engine_client.py plugin/reaper/predict_worker.py plugin/reaper/dn_core.py ml/tests/test_engine_client.py ml/tests/test_predict_worker.py ml/tests/test_reaper_core.py`

- [ ] **Step 4:** Static + Lua checks: `~/homebrew/bin/lua -e "assert(loadfile('plugin/reaper/dynamics_needed.lua'))"`; `~/homebrew/bin/lua plugin/reaper/test_dn_core.lua` prints `ok`. Confirm the full pytest suite still passes for the REMAINING ml tests (the drum_dynamics tests are untouched): `python -m pytest ml/tests -q` (the three deleted reaper tests are gone; the rest must stay green).

- [ ] **Step 5 (HUMAN):** end-to-end: select notes -> preview -> Apply (single undo, positions/pitch intact) -> reopen restores params -> kill engine -> red + Retry recovers -> close window stops cleanly.
- [ ] **Step 6: Commit** `feat(reaper): lua apply with undo, status line, cleanup; remove python panel`

---

## Self-Review

- Spec "Pivot to Lua" coverage: json/dn_core/tests (L1); skeleton+config+health (L2); autostart+starting (L3); controls+persistence (L4); debounced blocking preview+lane (L5); apply+status+cleanup+python removal (L6). ✓
- No-threads tradeoff honored: debounce via `time_precise`, blocking curl, per-frame health poll (L3/L5). ✓
- "Apply what you saw": Apply consumes `preview`, never re-predicts (L6). ✓
- Velocity-only + undo preserved (L6). ✓
- Known residual risk: exact ReaImGui/ReaScript Lua arities (Combo/SliderDouble/MIDI_GetNote/TimeMap_GetTimeSigAtTime) must be confirmed against the installed ReaImGui docs — L2 is the validation gate before UI is built on top.
