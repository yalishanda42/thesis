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


def _init():
    cfg = _load_config()
    sys.path.insert(0, os.path.join(cfg["repo_root"], "plugin", "reaper"))
    import engine_client
    import dn_core
    health = engine_client.ensure_engine(cfg)
    if health and not health.get("genres") and health.get("styles"):
        health["genres"] = dn_core.genres_from_styles(health["styles"])
    state = {
        "cfg": cfg,
        "engine_client": engine_client,
        "dn_core": dn_core,
        "ctx": RPR_ImGui_CreateContext("Dynamics Needed"),
        "health": health,
        "open": True,
    }
    take = None
    editor = RPR_MIDIEditor_GetActive()
    if editor:
        take = RPR_MIDIEditor_GetTake(editor)
    last = _load_last(_track_key(take)) if take else {}
    state["params"] = _default_params(state["health"] or {}, last)
    state["live"] = True
    state["dirty"] = False
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
    dn_core = STATE["dn_core"]
    visible, STATE["open"] = RPR_ImGui_Begin(ctx, "Dynamics Needed", True)
    if visible:
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
