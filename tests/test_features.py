import numpy as np

from drumhumanizer.features import (
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
