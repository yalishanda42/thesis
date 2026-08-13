"""Model-agnostic inference orchestration: notes+params -> new velocities."""
from __future__ import annotations

import numpy as np

from ..data.features import build_note_features

_NOTE_DT = np.dtype([("onset_sec", float), ("pitch", int), ("velocity", int)])


def _note_array(notes):
    return np.array([(n["onset_sec"], n["pitch"], n["velocity"]) for n in notes], dtype=_NOTE_DT)


def predict_velocities(request, predict_all):
    notes = request["notes"]
    if not notes:
        return {}
    na = _note_array(notes)
    # build_note_features sorts by onset_sec (stable); reproduce that order to map back.
    order = np.argsort(na["onset_sec"], kind="stable")
    meta = {"id": "infer", "drummer": "infer", "split": "infer",
            "bpm": float(request["bpm"]), "time_signature": str(request["time_signature"]),
            "style": str(request["style"]), "beat_type": str(request["beat_type"])}
    df = build_note_features(na, meta)                 # rows correspond to na[order]
    preds_sorted = np.asarray(predict_all(df), dtype=float)
    preds = np.empty(len(notes), dtype=float)
    preds[order] = preds_sorted                        # back to input positions

    blend = float(request.get("blend", 1.0))
    out = {}
    for pos, n in enumerate(notes):
        if not n.get("selected"):
            continue
        v = blend * preds[pos] + (1.0 - blend) * float(n["velocity"])
        out[n["index"]] = int(np.clip(round(v), 1, 127))
    return out
