import os
import sys

sys.path.insert(0, os.path.join("plugin", "reaper"))
import dn_core


def test_genres_from_styles():
    assert dn_core.genres_from_styles(["rock/indie", "rock", "jazz/swing"]) == ["jazz", "rock"]


def test_filter_styles_by_genre():
    styles = ["rock", "rock/indie", "jazz/swing"]
    assert dn_core.filter_styles_by_genre(styles, "rock") == ["rock", "rock/indie"]


def test_resolve_target_indices_prefers_selected():
    notes = [{"index": 0, "selected": True}, {"index": 1, "selected": False}]
    assert dn_core.resolve_target_indices(notes) == [0]
    all_unsel = [{"index": 3, "selected": False}, {"index": 4, "selected": False}]
    assert dn_core.resolve_target_indices(all_unsel) == [3, 4]


def test_build_predict_request_shape():
    notes = [{"index": 0, "pitch": 36, "onset_sec": 0.0, "velocity": 40, "selected": True}]
    req = dn_core.build_predict_request("mdn", "rock/indie", 1.2, 0.8, "fill", 128.0, "4-4", notes)
    assert req["model"] == "mdn" and req["style"] == "rock/indie"
    assert req["temperature"] == 1.2 and req["blend"] == 0.8 and req["beat_type"] == "fill"
    assert req["bpm"] == 128.0 and req["time_signature"] == "4-4" and req["notes"] == notes


def test_parse_velocities():
    assert dn_core.parse_velocities({"velocities": {"0": 97, "3": 12}}) == {0: 97, 3: 12}


def test_build_predict_request_clamps_blend_and_temperature():
    req = dn_core.build_predict_request("mdn", "rock", -0.5, 2.0, "beat", 120.0, "4-4", [])
    assert req["temperature"] == 0.0
    assert req["blend"] == 1.0
    req2 = dn_core.build_predict_request("mdn", "rock", 1.0, -1.0, "beat", 120.0, "4-4", [])
    assert req2["blend"] == 0.0
