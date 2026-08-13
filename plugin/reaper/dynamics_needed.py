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
