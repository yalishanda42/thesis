"""Pure, stdlib-only client logic shared by the Reaper ReaScript (unit-tested).

Kept free of any Reaper API so it runs under normal pytest.
"""
from __future__ import annotations


def genres_from_styles(styles):
    return sorted({s.split("/")[0] for s in styles})


def filter_styles_by_genre(styles, genre):
    return [s for s in styles if s.split("/")[0] == genre]


def resolve_target_indices(notes):
    selected = [n["index"] for n in notes if n.get("selected")]
    return selected if selected else [n["index"] for n in notes]


def build_predict_request(model, style, temperature, blend, beat_type, bpm, time_signature, notes):
    return {
        "model": model,
        "style": style,
        "temperature": max(0.0, float(temperature)),
        "blend": min(1.0, max(0.0, float(blend))),
        "beat_type": beat_type,
        "bpm": float(bpm),
        "time_signature": time_signature,
        "notes": notes,
    }


def parse_velocities(response):
    return {int(k): int(v) for k, v in response["velocities"].items()}
