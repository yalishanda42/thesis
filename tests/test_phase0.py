"""Unit tests for the pure Phase 0 aggregation helpers.

These exercise the math on hand-constructed inputs (no MIDI needed) so the
train-split scan can be trusted. Run with:  .venv/bin/python -m pytest tests/ -v
"""

import numpy as np

from drumhumanizer.phase0 import (
    GAP_EDGES,
    GAP_MS_EDGES,
    N_MIDI,
    FileScan,
    find_simultaneity_valley,
    gap_bin_centers,
    gap_histogram,
    inter_onset_gaps_beats,
    inter_onset_gaps_ms,
    merge_scans,
    pitch_velocity_matrix,
    velocity_stats,
    window_min,
)


def test_gap_edges_have_uniform_bin_widths():
    # regression: np.arange(start, stop, step) accumulated float error and made
    # some bins narrower -> spurious count dips. Edges must be evenly spaced
    # within each resolution band.
    fine = GAP_EDGES[GAP_EDGES < 0.20]
    widths = np.diff(fine)
    assert np.allclose(widths, 0.002, atol=1e-9)


def test_inter_onset_gaps_ms():
    # onsets 0.0, 0.010, 0.030 s -> gaps of 10 ms and 20 ms
    gaps = inter_onset_gaps_ms(np.array([0.0, 0.010, 0.030]))
    assert np.allclose(gaps, [10.0, 20.0])
    assert inter_onset_gaps_ms(np.array([0.5])).size == 0


def test_inter_onset_gaps_beats_basic():
    # bpm=120 -> beat_dur=0.5s. Onsets 0.0, 0.5, 1.0s -> gaps of 1.0 beat each.
    gaps = inter_onset_gaps_beats(np.array([0.0, 0.5, 1.0]), bpm=120)
    assert np.allclose(gaps, [1.0, 1.0])


def test_inter_onset_gaps_sorts_and_handles_simultaneous():
    # unsorted input with two simultaneous onsets -> a 0-beat gap appears
    gaps = inter_onset_gaps_beats(np.array([1.0, 0.0, 0.0]), bpm=120)
    assert gaps.size == 2
    assert gaps.min() == 0.0


def test_inter_onset_gaps_too_few():
    assert inter_onset_gaps_beats(np.array([0.5]), bpm=120).size == 0
    assert inter_onset_gaps_beats(np.array([]), bpm=120).size == 0


def test_pitch_velocity_matrix_counts_and_bounds():
    m = pitch_velocity_matrix([38, 38, 42], [100, 100, 20])
    assert m.shape == (N_MIDI, N_MIDI)
    assert m[38, 100] == 2
    assert m[42, 20] == 1
    assert m.sum() == 3
    # out-of-range values are dropped, not clamped
    m2 = pitch_velocity_matrix([200, -1], [10, 10])
    assert m2.sum() == 0


def test_velocity_stats_exact():
    counts = np.zeros(N_MIDI, dtype=np.int64)
    counts[[10, 20]] = [1, 1]  # values {10, 20}
    s = velocity_stats(counts)
    assert s.count == 2
    assert s.mean == 15.0
    assert s.std == 5.0
    assert s.median in (10.0, 20.0)  # searchsorted picks a real value


def test_velocity_stats_empty():
    s = velocity_stats(np.zeros(N_MIDI, dtype=np.int64))
    assert s.count == 0
    assert np.isnan(s.mean)


def test_gap_histogram_edges_and_length():
    counts = gap_histogram(np.array([0.001, 0.001, 0.5, 100.0]))
    assert counts.shape[0] == len(GAP_EDGES) - 1
    assert counts.sum() == 4
    # first bin [0, 0.002) catches the two ~0 gaps
    assert counts[0] == 2


def test_find_valley_between_two_clusters():
    # simultaneity spike at bin 0, a valley, then a subdivision cluster
    centers = gap_bin_centers()
    counts = np.zeros(centers.size, dtype=np.int64)
    counts[0] = 1000                          # simultaneity cluster near 0
    sub = int(np.argmin(np.abs(centers - 0.15)))
    counts[sub] = 800                          # a real subdivision at ~0.15 beat
    valley, found = find_simultaneity_valley(counts, smooth=1)
    assert found
    assert 0.0 < valley < 0.15


def test_find_valley_reports_not_found_when_monotonic():
    centers = gap_bin_centers()
    counts = np.linspace(1000, 0, centers.size)  # strictly decreasing
    valley, found = find_simultaneity_valley(counts, smooth=1)
    assert not found


def test_window_min_locates_trough_in_window():
    centers = gap_bin_centers()
    counts = np.ones(centers.size, dtype=float) * 100
    trough = int(np.argmin(np.abs(centers - 0.15)))
    counts[trough] = 1  # a deep trough at ~0.15 beat
    got = window_min(counts, GAP_EDGES, lo=0.05, hi=0.22, smooth=1)
    assert abs(got - 0.15) < 0.01
    # a trough outside the window is ignored
    assert not np.isclose(window_min(counts, GAP_EDGES, lo=0.30, hi=0.50, smooth=1), 0.15)


def test_merge_scans_sums_and_skips_failures():
    ms = gap_histogram(inter_onset_gaps_ms(np.array([0.0, 0.5])), GAP_MS_EDGES)
    a = FileScan(pv_matrix=pitch_velocity_matrix([38], [100]),
                 gap_counts=gap_histogram(np.array([0.5])),
                 gap_ms_counts=ms, n_notes=1, ok=True)
    b = FileScan(pv_matrix=pitch_velocity_matrix([38], [100]),
                 gap_counts=gap_histogram(np.array([0.5])),
                 gap_ms_counts=ms, n_notes=1, ok=True)
    bad = FileScan(pv_matrix=np.zeros((N_MIDI, N_MIDI), dtype=np.int64),
                   gap_counts=np.zeros(len(GAP_EDGES) - 1, dtype=np.int64),
                   gap_ms_counts=np.zeros(len(GAP_MS_EDGES) - 1, dtype=np.int64),
                   n_notes=0, ok=False)
    total = merge_scans([a, b, bad])
    assert total.n_notes == 2
    assert total.pv_matrix[38, 100] == 2
    assert total.gap_counts.sum() == 2
