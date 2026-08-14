# plugin/reaper/dynamics_needed.py
"""Dynamics Needed - ReaImGui panel (Python).

Restores humanized velocities of drum notes with a live preview. Runs under
Reaper's embedded Python. Requires ReaImGui (install via ReaPack) -- its API is
reached through the official Python shim ``imgui.py`` (ReaImGui does NOT expose
RPR_ImGui_* in reaper_python; the shim provides module-level imgui.* functions).

ASCII only. No __file__. RPR_* (core API) return tuples; imgui.* out-params are
returned as tuples too (e.g. Begin -> (visible, open), Combo -> (changed, idx)).
"""
import json
import os
import sys

from reaper_python import *  # noqa: F401,F403

try:
    sys.path.append(os.path.join(
        RPR_GetResourcePath(), "Scripts", "ReaTeam Extensions", "API"))
    import imgui  # ReaImGui official Python binding shim
except Exception as e:
    RPR_ShowConsoleMsg(
        "Dynamics Needed: ReaImGui Python binding not found "
        "(install ReaImGui via ReaPack): {}\n".format(e))
    raise


def _resource_path():
    return RPR_GetResourcePath()


def _module_dir(cfg, resource_path):
    # Dev config points at the repo; installed build ships helpers beside the script.
    if cfg and cfg.get("repo_root"):
        return os.path.join(cfg["repo_root"], "plugin", "reaper")
    # Hardcode path (must match dn_paths.script_dir); cannot import dn_paths here
    # because sys.path.insert happens AFTER this runs (bootstrap ordering).
    return os.path.join(resource_path, "Scripts", "Dynamics Needed", "MIDI Editor")


def _read_config_if_any(resource_path):
    cfg_path = os.path.join(resource_path, "dynamics_needed_config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


def _active_take():
    editor = RPR_MIDIEditor_GetActive()
    return RPR_MIDIEditor_GetTake(editor) if editor else None


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


def _default_params(health, last, dn_core):
    styles = health.get("styles", []) or ["rock"]
    genres = health.get("genres", []) or ["rock"]
    default_genre = last.get("genre", genres[0])
    styles_for_default = dn_core.filter_styles_by_genre(styles, default_genre) or styles
    default_style = last.get("style", styles_for_default[0] if styles_for_default else styles[0])
    return {
        "genre": default_genre,
        "style": default_style,
        "model": last.get("model", "mdn"),
        "temperature": float(last.get("temperature", 1.0)),
        "blend": float(last.get("blend", 0.8)),
        "beat_type": last.get("beat_type", "beat"),
    }


def _start_engine_async(state):
    state["engine_ready"] = False
    state["health"] = None
    def worker():
        h = state["engine_client"].ensure_engine(state["cfg"])
        if h and not h.get("genres") and h.get("styles"):
            h["genres"] = state["dn_core"].genres_from_styles(h["styles"])
        with state["engine_lock"]:
            state["health"] = h
            state["engine_ready"] = True
    import threading
    threading.Thread(target=worker, daemon=True).start()


def _init():
    resource_path = _resource_path()
    cfg = _read_config_if_any(resource_path)

    # Make helper modules importable (installed location, or the dev repo).
    sys.path.insert(0, _module_dir(cfg, resource_path))
    import dn_paths
    import bootstrap
    import engine_client
    import dn_core
    import predict_worker
    import threading

    # Engine bootstrap: download on first run, and re-download when the pinned
    # engine version changes. Dev config (repo_root/venv_python) bypasses this.
    is_dev = bool(cfg and cfg.get("repo_root"))
    if not is_dev:
        version = bootstrap.PINNED_ENGINE_VERSION
        needs_engine = (cfg is None
                        or cfg.get("engine_version") != version
                        or not bootstrap.is_installed(resource_path, version))
        if needs_engine:
            if not bootstrap.is_installed(resource_path, version):
                ok = RPR_ShowMessageBox(
                    "Dynamics Needed needs to download its engine (~250 MB). Download now?",
                    "Dynamics Needed", 4)  # 4 = Yes/No; 6 = Yes
                if ok != 6:
                    raise RuntimeError("engine download declined")
                try:
                    cfg = bootstrap.install(resource_path, version, dn_paths.platform_key())
                except bootstrap.BootstrapError as e:
                    raise RuntimeError("engine install failed: {}".format(e))
            else:
                # Already unpacked on disk (e.g. config missing/stale) -> (re)write config.
                cfg = bootstrap.write_config(resource_path, version)

    state = {
        "cfg": cfg,
        "engine_client": engine_client,
        "dn_core": dn_core,
        "ctx": imgui.CreateContext("Dynamics Needed"),
        "engine_lock": threading.Lock(),
        "engine_ready": False,
        "health": None,
        "open": True,
    }
    _start_engine_async(state)
    take = None
    editor = RPR_MIDIEditor_GetActive()
    if editor:
        take = RPR_MIDIEditor_GetTake(editor)
    state["last_params"] = _load_last(_track_key(take)) if take else {}
    state["params"] = None
    state["live"] = True
    ec, cfgd = state["engine_client"], state["cfg"]
    state["worker"] = predict_worker.PredictWorker(
        lambda req: dn_core.parse_velocities(ec.predict(cfgd, req)))
    state["worker"].start()
    state["seq"] = 0
    state["preview"] = {}
    return state


STATE = None


def loop():
    ctx = STATE["ctx"]
    dn_core = STATE["dn_core"]
    visible, STATE["open"] = imgui.Begin(ctx, "Dynamics Needed", True)
    if visible:
        with STATE["engine_lock"]:
            engine_ready = STATE["engine_ready"]
            h = STATE["health"]

        if not engine_ready:
            imgui.Text(ctx, "Starting engine...")
            imgui.End(ctx)
        elif h is None:
            imgui.TextColored(ctx, 0xFF4040FF, "Engine unreachable")
            imgui.SameLine(ctx)
            if imgui.Button(ctx, "Retry"):
                _start_engine_async(STATE)
            imgui.End(ctx)
        else:
            if STATE["params"] is None:
                STATE["params"] = _default_params(h, STATE["last_params"], dn_core)

            p = STATE["params"]
            genres = h.get("genres", []) or ["rock"]
            styles_for_genre = dn_core.filter_styles_by_genre(h.get("styles", []), p["genre"]) or [p["style"]]

            # Genre combo
            gi = genres.index(p["genre"]) if p["genre"] in genres else 0
            changed, gi = imgui.Combo(ctx, "Genre", gi, "\x00".join(genres) + "\x00")
            if changed:
                p["genre"] = genres[gi]
                new_styles = dn_core.filter_styles_by_genre(h.get("styles", []), p["genre"])
                p["style"] = new_styles[0] if new_styles else p["style"]

            # Style combo (filtered by genre). Only LGBM uses style; the
            # transformer heads (MDN, Categorical) condition on genre only.
            style_disabled = p["model"] != "lgbm"
            si = styles_for_genre.index(p["style"]) if p["style"] in styles_for_genre else 0
            if style_disabled:
                imgui.BeginDisabled(ctx, True)
            changed, si = imgui.Combo(ctx, "Style", si, "\x00".join(styles_for_genre) + "\x00")
            if changed:
                p["style"] = styles_for_genre[si]
            if style_disabled:
                imgui.EndDisabled(ctx)
                imgui.SameLine(ctx)
                imgui.Text(ctx, "(LGBM only)")

            # Model radio
            if imgui.RadioButton(ctx, "LGBM", p["model"] == "lgbm"):
                p["model"] = "lgbm"
            imgui.SameLine(ctx)
            if imgui.RadioButton(ctx, "MDN", p["model"] == "mdn"):
                p["model"] = "mdn"
            imgui.SameLine(ctx)
            if imgui.RadioButton(ctx, "Categorical", p["model"] == "categorical"):
                p["model"] = "categorical"

            # Sliders. Temp only affects MDN sampling (LGBM is deterministic and
            # Categorical sampling ignores temperature), so enable it only for MDN.
            temp_disabled = p["model"] != "mdn"
            if temp_disabled:
                imgui.BeginDisabled(ctx, True)
            changed, p["temperature"] = imgui.SliderDouble(ctx, "Temp", p["temperature"], 0.0, 2.0)
            if temp_disabled:
                imgui.EndDisabled(ctx)
                imgui.SameLine(ctx)
                imgui.Text(ctx, "(MDN only)")
            changed, p["blend"] = imgui.SliderDouble(ctx, "Blend", p["blend"], 0.0, 1.0)

            # Is a fill?
            changed, is_fill = imgui.Checkbox(ctx, "Is a fill?", p["beat_type"] == "fill")
            if changed:
                p["beat_type"] = "fill" if is_fill else "beat"

            take = _active_take()
            notes = _read_notes(take) if take else []
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

            # Live toggle + manual Preview on one row; Preview is hidden while
            # Live is on (auto-predict makes it redundant).
            _, STATE["live"] = imgui.Checkbox(ctx, "Live Preview", STATE["live"])
            if not STATE["live"]:
                imgui.SameLine(ctx)
                if imgui.Button(ctx, "Preview"):   # forces a fresh predict
                    STATE["force_preview"] = True

            target_notes = [nt for nt in notes if nt["selected"]]
            imgui.Text(ctx, "velocities ({} target notes)".format(len(target_notes)))
            dl = imgui.GetWindowDrawList(ctx)
            x, y = imgui.GetCursorScreenPos(ctx)
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
                imgui.DrawList_AddRectFilled(dl, bx, y + lane_h * (1 - cur), bx + bw - 1, y + lane_h, cur_col)
                imgui.DrawList_AddRectFilled(dl, bx, y + lane_h * (1 - pred), bx + bw - 1, y + lane_h, pred_col)
            imgui.Dummy(ctx, w, lane_h)   # reserve layout space under the drawing

            # Status line
            if STATE["worker"].last_error():
                imgui.TextColored(ctx, 0xFFAA40FF, "Predict failed: " + STATE["worker"].last_error())
            elif not _active_take():
                imgui.TextColored(ctx, 0xAAAAAAFF, "Open a MIDI item to edit")
            else:
                imgui.TextColored(ctx, 0x40FF40FF, "ready")

            # Apply
            can_apply = bool(_active_take()) and bool(STATE["preview"])
            if not can_apply:
                imgui.BeginDisabled(ctx, True)
            if imgui.Button(ctx, "Apply") and can_apply:
                take = _active_take()
                RPR_Undo_BeginBlock()
                _apply(take, STATE["preview"])
                RPR_Undo_EndBlock("Dynamics Needed: restore velocities", -1)
                _save_last(_track_key(take), STATE["params"])
            if not can_apply:
                imgui.EndDisabled(ctx)

            imgui.End(ctx)
    if not STATE["open"]:
        STATE["worker"].stop()
        return
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
