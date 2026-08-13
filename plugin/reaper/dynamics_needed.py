"""Dynamics Needed - Reaper action: restore velocities of the selected drum notes.

Runs under Reaper's embedded Python (stdlib only). Auto-starts the local inference
service if it isn't already running, then rewrites the selected notes' velocities.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

from reaper_python import *  # noqa: F401,F403  (provides RPR_* functions)

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # Reaper doesn't define __file__ for ReaScripts
    _HERE = os.path.dirname(RPR_get_action_context()[1])
sys.path.insert(0, _HERE)
import dn_core  # noqa: E402


def _msg(text):
    RPR_ShowConsoleMsg(text + "\n")


def _load_config():
    with open(os.path.join(_HERE, "config.local.json")) as fh:
        return json.load(fh)


def _base_url(cfg):
    return "http://127.0.0.1:{}".format(cfg.get("port", 8765))


def _health(cfg):
    try:
        with urllib.request.urlopen(_base_url(cfg) + "/health", timeout=1) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _start_engine(cfg):
    runtime = os.path.join(_HERE, ".runtime")
    os.makedirs(runtime, exist_ok=True)
    log = open(os.path.join(runtime, "engine.log"), "a")
    subprocess.Popen(
        [cfg["venv_python"], "-m", "drum_dynamics.serve",
         "--port", str(cfg.get("port", 8765)), "--parent-pid", str(os.getpid())],
        cwd=cfg["repo_root"], stdout=log, stderr=log,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )


def _ensure_engine(cfg):
    health = _health(cfg)
    if health:
        return health
    _start_engine(cfg)
    for _ in range(30):                      # ~15s for cold torch import
        time.sleep(0.5)
        health = _health(cfg)
        if health:
            return health
    return None


def _active_take():
    editor = RPR_MIDIEditor_GetActive()
    return RPR_MIDIEditor_GetTake(editor) if editor else None


def _read_notes(take):
    """Return list of note dicts with project-time onsets and selection flags."""
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
    _, _, _, num, denom, bpm = RPR_TimeMap_GetTimeSigAtTime(0, pos, 0, 0, 0.0)
    return float(bpm), "{}-{}".format(int(num), int(denom))


def _track_key(take):
    item = RPR_GetMediaItemTake_Item(take)
    track = RPR_GetMediaItem_Track(item)
    _, _, guid = RPR_GetSetMediaTrackInfo_String(track, "GUID", "", False)
    return "dn_last_" + guid


def _load_last(key):
    _, _, val, _ = RPR_GetProjExtState(0, "DynamicsNeeded", key, "")
    try:
        return json.loads(val) if val else {}
    except Exception:
        return {}


def _save_last(key, params):
    RPR_SetProjExtState(0, "DynamicsNeeded", key, json.dumps(params))


def _dialog(health, last):
    """Native GetUserInputs dialog. Returns params dict or None if cancelled."""
    styles = health["styles"]
    genres = health["genres"]
    defaults = {"genre": last.get("genre", genres[0]),
                "style": last.get("style", styles[0]),
                "model": last.get("model", "mdn"),
                "temperature": last.get("temperature", 1.0),
                "blend": last.get("blend", 1.0),
                "beat_type": last.get("beat_type", "beat")}
    fields = "genre,style,model (lgbm/mdn),temperature,blend (0-1),fill? (y/n)"
    csv_default = "{},{},{},{},{},{}".format(
        defaults["genre"], defaults["style"], defaults["model"],
        defaults["temperature"], defaults["blend"],
        "y" if defaults["beat_type"] == "fill" else "n")
    ok, _, _, _, csv, _ = RPR_GetUserInputs("Dynamics Needed", 6, fields, csv_default, 1024)
    if not ok:
        return None
    genre, style, model, temp, blend, fill = (csv.split(",", 5) + [""] * 6)[:6]
    try:
        temperature = float(temp)
        blend_val = float(blend)
    except ValueError:
        _msg("Dynamics Needed: temperature and blend must be numbers.")
        return None
    return {"genre": genre.strip(), "style": style.strip(), "model": model.strip(),
            "temperature": temperature, "blend": blend_val,
            "beat_type": "fill" if fill.strip().lower().startswith("y") else "beat"}


def _apply(take, notes, velocities):
    for n in notes:
        if n["index"] in velocities:
            RPR_MIDI_SetNote(take, n["index"], -1, -1, -1, -1, -1, -1,
                             velocities[n["index"]], False)
    RPR_MIDI_Sort(take)


def main():
    try:
        cfg = _load_config()
    except Exception:
        _msg("Dynamics Needed: no config found. Run "
             "'python plugin/reaper/setup_reaper.py' once, then retry.")
        return
    take = _active_take()
    if not take:
        _msg("Dynamics Needed: open a MIDI item in the MIDI editor first.")
        return
    health = _ensure_engine(cfg)
    if not health:
        _msg("Dynamics Needed: could not reach the inference engine. "
             "Run 'python plugin/reaper/setup_reaper.py' once, then retry. "
             "If it persists, check plugin/reaper/.runtime/engine.log")
        return

    notes = _read_notes(take)
    if not notes:
        _msg("Dynamics Needed: no notes found in the active take.")
        return

    key = _track_key(take)
    params = _dialog(health, _load_last(key))
    if params is None:
        return
    _save_last(key, params)

    targets = set(dn_core.resolve_target_indices(notes))
    for n in notes:
        n["selected"] = n["index"] in targets
    bpm, time_sig = _tempo_and_sig(take)
    req = dn_core.build_predict_request(
        params["model"], params["style"], params["temperature"], params["blend"],
        params["beat_type"], bpm, time_sig, notes)

    data = json.dumps(req).encode()
    try:
        request = urllib.request.Request(_base_url(cfg) + "/predict", data=data,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as r:
            response = json.loads(r.read())
    except Exception as e:
        _msg("Dynamics Needed: prediction failed: {}".format(e))
        return

    velocities = dn_core.parse_velocities(response)
    RPR_Undo_BeginBlock()
    _apply(take, notes, velocities)
    RPR_Undo_EndBlock("Dynamics Needed: restore velocities", -1)
    _msg("Dynamics Needed: updated {} notes.".format(len(velocities)))


main()
