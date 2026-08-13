import numpy as np
from drum_dynamics.serve.core import predict_velocities


def _req(notes, blend=1.0):
    return {"model": "lgbm", "style": "funk/groove1", "temperature": 1.0,
            "blend": blend, "beat_type": "beat", "bpm": 120.0,
            "time_signature": "4-4", "notes": notes}


def test_only_selected_notes_returned():
    notes = [{"index": 5, "pitch": 36, "onset_sec": 0.0, "velocity": 40, "selected": True},
             {"index": 6, "pitch": 38, "onset_sec": 0.0, "velocity": 40, "selected": False}]
    out = predict_velocities(_req(notes), lambda df: np.full(len(df), 100.0))
    assert set(out) == {5}


def test_blend_and_clamp():
    notes = [{"index": 0, "pitch": 36, "onset_sec": 0.0, "velocity": 40, "selected": True}]
    # blend 0.5 of pred 100 with orig 40 -> 70
    assert predict_velocities(_req(notes, blend=0.5), lambda df: np.full(len(df), 100.0))[0] == 70
    # clamp high and low
    assert predict_velocities(_req(notes), lambda df: np.full(len(df), 500.0))[0] == 127
    assert predict_velocities(_req(notes), lambda df: np.full(len(df), -10.0))[0] == 1


def test_predictions_map_back_to_correct_index_when_unsorted():
    # onsets out of order; predict_all returns each row's own onset_sec.
    notes = [{"index": 0, "pitch": 36, "onset_sec": 3.0, "velocity": 0, "selected": True},
             {"index": 1, "pitch": 38, "onset_sec": 1.0, "velocity": 0, "selected": True},
             {"index": 2, "pitch": 42, "onset_sec": 2.0, "velocity": 0, "selected": True}]
    out = predict_velocities(_req(notes), lambda df: df["onset_sec"].to_numpy())
    assert out == {0: 3, 1: 1, 2: 2}
