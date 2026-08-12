import numpy as np

from drum_dynamics.features import (
    SIMULTANEITY_TOL_BEATS,
    beats_per_bar,
    metrical_phase,
    nearest_subdivision,
    swing_ratio,
)


def test_simultaneity_tol_locked():
    assert SIMULTANEITY_TOL_BEATS == 0.02


def test_beats_per_bar():
    assert beats_per_bar("4-4") == 4
    assert beats_per_bar("3-4") == 3


def test_metrical_phase_120bpm_44():
    # bpm=120 -> beat=0.5s, bar=2.0s. Onsets at 0, 0.5, 0.75, 2.0 s.
    onset = np.array([0.0, 0.5, 0.75, 2.0])
    pb, pbar = metrical_phase(onset, bpm=120, bpb=4)
    assert np.allclose(pb, [0.0, 0.0, 0.5, 0.0])
    assert np.allclose(pbar, [0.0, 0.25, 0.375, 0.0])


def test_swing_ratio_straight_vs_triplet():
    # straight 8th (phase 0.5) -> 0 ; triplet offbeat (2/3) -> ~1 ; onbeat -> 0
    sr = swing_ratio(np.array([0.0, 0.5, 2.0 / 3.0]))
    assert np.isclose(sr[0], 0.0)
    assert np.isclose(sr[1], 0.0)
    assert np.isclose(sr[2], 1.0, atol=0.02)


def test_nearest_subdivision_picks_closest_grid():
    # 0.25 is exactly a 16th; 1/3 is an 8th-triplet
    got = nearest_subdivision(np.array([0.25, 1.0 / 3.0]))
    assert got[0] == "16th"
    assert got[1] == "8th-triplet"


import os

import pandas as pd

from drum_dynamics.features import build_note_features
from drum_dynamics.midi import load_note_array
from drum_dynamics.voicemap import CANONICAL_VOICES

EGMD_BASE = os.path.join("data", "e-gmd", "e-gmd-v1.0.0")


def _synthetic_note_array():
    # kick(36) & snare(38) simultaneous at t=0, hat(42) at t=0.5s; bpm 120 -> beat 0.5s
    dt = np.dtype([("onset_sec", float), ("duration_sec", float),
                   ("onset_tick", int), ("duration_tick", int),
                   ("pitch", int), ("velocity", int),
                   ("track", int), ("channel", int), ("id", object)])
    rows = [(0.0, 0.1, 0, 48, 36, 100, 0, 9, "n0"),
            (0.0, 0.1, 0, 48, 38, 40, 0, 9, "n1"),
            (0.5, 0.1, 240, 48, 42, 70, 0, 9, "n2")]
    return np.array(rows, dtype=dt)


def _meta(**kw):
    base = dict(id="drummerX/s/1", drummer="drummerX", split="train",
                bpm=120, time_signature="4-4", style="funk/groove1", beat_type="beat")
    base.update(kw)
    return base


def test_build_note_features_columns_and_no_leakage():
    df = build_note_features(_synthetic_note_array(), _meta())
    assert len(df) == 3
    # velocity is the only velocity-derived column
    assert "velocity" in df.columns
    assert not [c for c in df.columns if "vel" in c.lower() and c != "velocity"]
    for c in ["voice", "genre", "phase_beat", "sin_beat", "swing_ratio",
              "log_time_to_prev", "simult_count", "density_1beat", "bar_index"]:
        assert c in df.columns
    for v in CANONICAL_VOICES:
        assert f"simult_{v}" in df.columns


def test_build_note_features_simultaneity_and_voice():
    df = build_note_features(_synthetic_note_array(), _meta())
    kick = df[df["voice"] == "kick"].iloc[0]
    # kick fires simultaneously with snare -> count 2, both multi-hots set, self set
    assert kick["simult_count"] == 2
    assert kick["simult_kick"] == 1 and kick["simult_snare"] == 1
    assert kick["simult_closed-hh"] == 0
    assert kick["genre"] == "funk"


def test_build_note_features_on_real_file():
    df = pd.read_csv(os.path.join(EGMD_BASE, "e-gmd-v1.0.0.csv"))
    row = df[df["beat_type"] == "beat"].iloc[0]
    na = load_note_array(os.path.join(EGMD_BASE, row["midi_filename"]))
    out = build_note_features(na, row.to_dict())
    assert len(out) == len(na)
    assert out["velocity"].between(0, 127).all()
    assert out["phase_beat"].between(0, 1).all()
