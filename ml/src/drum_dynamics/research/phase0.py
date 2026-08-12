"""Phase 0 — exploratory validation over the E-GMD *train* split.

The design spec (``docs/superpowers/specs/2026-08-11-drum-velocity-humanization-design.md``
§3) makes several feature-spec decisions *data-driven* rather than guessed:

1. **Pitch usage & velocity distributions** — per GM pitch, a count and a velocity
   histogram. Drives the final **voice grouping**.
2. **Simultaneity tolerance** — histogram of very-small inter-onset gaps, in beats,
   whose valley separates "meant to be simultaneous" from "genuinely sequential."
   The valley sets ``SIMULTANEITY_TOL``.
3. **Categorical cardinality / skew** — style / time_signature / bpm / beat_type
   (mostly derivable from the metadata CSV).
4. **Time-signature audit** — how much of the corpus is 4/4 (§5 gate).

This module holds the *pure, testable* pieces. The full scan is driven by
``scripts/phase0_analysis.py``. Everything here works on compact, **mergeable**
aggregates (a ``(128, 128)`` pitch×velocity matrix and a fixed-edge gap histogram)
so the train split can be processed in parallel and summed across workers without
holding raw per-note data in memory.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

import partitura

# --- histogram grids ---------------------------------------------------------

#: MIDI values are 0..127, so a (128, 128) matrix indexed [pitch, velocity]
#: stores the *exact* per-pitch velocity distribution (velocities are integers).
N_MIDI = 128

#: Fixed bin *edges* for the inter-onset-gap histogram, in **beats**. Built as
#: ``integer * width`` (then rounded) rather than ``np.arange(start, stop, width)``
#: — the latter accumulates step error and produces uneven bin widths (spurious
#: count dips at ~0.05/0.10/0.15/0.20). Edges are fixed so per-worker histograms
#: can be summed. Fine near zero, coarser out to one beat, with an overflow bin.
GAP_EDGES = np.round(
    np.concatenate(
        [
            np.arange(0, 100) * 0.002,      # 0.000 .. 0.198 beat, 0.002-beat bins
            np.arange(20, 100) * 0.010,     # 0.20 .. 0.99 beat, 0.010-beat bins
            [1.0, np.inf],                  # [1.0, inf) overflow
        ]
    ),
    6,
)

#: Fixed bin edges for the gap histogram in **milliseconds** — the physically
#: principled unit for "were these two hits meant to be simultaneous?" (a fixed
#: neuromuscular/recording window, independent of tempo). Very fine near zero.
GAP_MS_EDGES = np.round(
    np.concatenate(
        [
            np.arange(0, 100) * 0.5,        # 0 .. 49.5 ms, 0.5-ms bins
            np.arange(50, 100) * 1.0,       # 50 .. 99 ms, 1-ms bins
            np.arange(20, 51) * 5.0,        # 100 .. 250 ms, 5-ms bins
            [np.inf],                       # [250, inf) overflow
        ]
    ),
    6,
)


def gap_bin_centers(edges: np.ndarray = GAP_EDGES) -> np.ndarray:
    """Finite bin centers for ``edges`` (the last, infinite, bin is dropped)."""
    finite = edges[np.isfinite(edges)]
    return (finite[:-1] + finite[1:]) / 2.0


# --- pure computations -------------------------------------------------------


def inter_onset_gaps_beats(onset_sec: np.ndarray, bpm: float) -> np.ndarray:
    """Consecutive inter-onset gaps (any voice), in beats.

    Onsets are sorted, differenced, and converted seconds→beats via the nominal
    ``bpm`` (``beat_dur = 60 / bpm``). Simultaneous hits produce a gap of ~0, which
    is exactly the mass Phase 0 needs to locate the simultaneity valley. Returns an
    empty array for fewer than two onsets.
    """
    onset_sec = np.asarray(onset_sec, dtype=float)
    if onset_sec.size < 2:
        return np.empty(0, dtype=float)
    beat_dur = 60.0 / float(bpm)
    ordered = np.sort(onset_sec)
    return np.diff(ordered) / beat_dur


def inter_onset_gaps_ms(onset_sec: np.ndarray) -> np.ndarray:
    """Consecutive inter-onset gaps (any voice), in **milliseconds**.

    Tempo-independent, so this is the principled unit for the simultaneity
    threshold. Returns an empty array for fewer than two onsets.
    """
    onset_sec = np.asarray(onset_sec, dtype=float)
    if onset_sec.size < 2:
        return np.empty(0, dtype=float)
    return np.diff(np.sort(onset_sec)) * 1000.0


def pitch_velocity_matrix(pitch: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """A ``(128, 128)`` int64 matrix ``M[pitch, velocity]`` of note counts."""
    pitch = np.asarray(pitch, dtype=np.int64)
    velocity = np.asarray(velocity, dtype=np.int64)
    ok = (pitch >= 0) & (pitch < N_MIDI) & (velocity >= 0) & (velocity < N_MIDI)
    m = np.zeros((N_MIDI, N_MIDI), dtype=np.int64)
    np.add.at(m, (pitch[ok], velocity[ok]), 1)
    return m


def gap_histogram(gaps_beats: np.ndarray, edges: np.ndarray = GAP_EDGES) -> np.ndarray:
    """Counts of ``gaps_beats`` into the fixed ``edges`` (length ``len(edges) - 1``)."""
    counts, _ = np.histogram(np.asarray(gaps_beats, dtype=float), bins=edges)
    return counts.astype(np.int64)


@dataclass
class VelocityStats:
    count: int
    mean: float
    std: float
    median: float
    p10: float
    p90: float


def velocity_stats(velocity_counts: np.ndarray) -> VelocityStats:
    """Exact distribution stats from a length-128 histogram over velocities 0..127."""
    counts = np.asarray(velocity_counts, dtype=np.int64)
    total = int(counts.sum())
    values = np.arange(counts.size)
    if total == 0:
        return VelocityStats(0, float("nan"), float("nan"),
                             float("nan"), float("nan"), float("nan"))
    mean = float((values * counts).sum() / total)
    var = float((counts * (values - mean) ** 2).sum() / total)
    cum = np.cumsum(counts)

    def quantile(q: float) -> float:
        # smallest value whose cumulative count reaches q of the total
        target = q * total
        return float(values[np.searchsorted(cum, target, side="left")])

    return VelocityStats(
        count=total,
        mean=mean,
        std=float(np.sqrt(var)),
        median=quantile(0.5),
        p10=quantile(0.10),
        p90=quantile(0.90),
    )


def find_simultaneity_valley(
    gap_counts: np.ndarray,
    edges: np.ndarray = GAP_EDGES,
    search_max: float = 0.2,
    smooth: int = 3,
):
    """Locate the *first* valley after the near-zero peak — the boundary between
    "meant to be simultaneous" and "genuinely sequential."

    Returns ``(valley, found)``. ``found`` is False when the histogram just
    decreases monotonically (no bimodal structure) within ``[0, search_max)``; the
    returned value is then the search-window minimum as a fallback and the caller
    should treat it as advisory. Counts are lightly box-smoothed (width ``smooth``)
    before the search so single-bin noise doesn't masquerade as a valley.
    """
    centers = gap_bin_centers(edges)
    counts = np.asarray(gap_counts[: centers.size], dtype=float)
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        counts = np.convolve(counts, kernel, mode="same")
    window = centers < search_max
    idx = np.where(window)[0]
    if idx.size == 0:
        return float(centers[0]), False
    counts = counts[idx]
    peak = int(np.argmax(counts))
    # first index after the peak where the count stops falling (a local minimum)
    for i in range(peak + 1, counts.size - 1):
        if counts[i] <= counts[i - 1] and counts[i] < counts[i + 1]:
            return float(centers[idx[i]]), True
    # no rebound -> monotonic decrease; report the window min as advisory
    return float(centers[idx[int(np.argmin(counts))]]), False


def window_min(
    counts: np.ndarray,
    edges: np.ndarray,
    lo: float,
    hi: float,
    smooth: int = 5,
) -> float:
    """Center of the (box-smoothed) minimum bin within ``[lo, hi)`` — used to locate
    the broad, flat trough between the sub-16th ornament mass and the subdivision
    grid. Unlike :func:`find_simultaneity_valley` this makes no bimodality claim; it
    simply reports where the density bottoms out in a caller-chosen window.
    """
    centers = gap_bin_centers(edges)
    c = np.asarray(counts[: centers.size], dtype=float)
    if smooth > 1:
        c = np.convolve(c, np.ones(smooth) / smooth, mode="same")
    idx = np.where((centers >= lo) & (centers < hi))[0]
    if idx.size == 0:
        return float("nan")
    return float(centers[idx[int(np.argmin(c[idx]))]])


# --- per-file scan (parallel worker) -----------------------------------------


@dataclass
class FileScan:
    pv_matrix: np.ndarray      # (128, 128) pitch × velocity counts
    gap_counts: np.ndarray     # len(GAP_EDGES) - 1, gaps in beats
    gap_ms_counts: np.ndarray  # len(GAP_MS_EDGES) - 1, gaps in milliseconds
    n_notes: int
    ok: bool                   # False if the file failed to parse


def _empty_scan(ok: bool) -> FileScan:
    return FileScan(
        pv_matrix=np.zeros((N_MIDI, N_MIDI), dtype=np.int64),
        gap_counts=np.zeros(len(GAP_EDGES) - 1, dtype=np.int64),
        gap_ms_counts=np.zeros(len(GAP_MS_EDGES) - 1, dtype=np.int64),
        n_notes=0,
        ok=ok,
    )


def scan_file(path: str, bpm: float) -> FileScan:
    """Parse one E-GMD MIDI file into compact, mergeable Phase 0 aggregates.

    Never raises: a parse failure returns ``FileScan(ok=False)`` so the runner can
    count failures instead of aborting the whole scan.
    """
    try:
        # partitura warns once per note-off (note_on velocity=0); E-GMD has
        # millions of them, so silence it — it dominates runtime otherwise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            na = partitura.load_performance_midi(path).performedparts[0].note_array()
    except Exception:
        return _empty_scan(ok=False)
    scan = _empty_scan(ok=True)
    scan.n_notes = int(len(na))
    if scan.n_notes == 0:
        return scan
    scan.pv_matrix = pitch_velocity_matrix(na["pitch"], na["velocity"])
    scan.gap_counts = gap_histogram(inter_onset_gaps_beats(na["onset_sec"], bpm))
    scan.gap_ms_counts = gap_histogram(inter_onset_gaps_ms(na["onset_sec"]), GAP_MS_EDGES)
    return scan


def merge_scans(scans) -> FileScan:
    """Sum a sequence of :class:`FileScan` into one aggregate."""
    total = _empty_scan(ok=True)
    n_ok = 0
    for s in scans:
        if not s.ok:
            continue
        n_ok += 1
        total.pv_matrix += s.pv_matrix
        total.gap_counts += s.gap_counts
        total.gap_ms_counts += s.gap_ms_counts
        total.n_notes += s.n_notes
    total.ok = n_ok > 0
    return total
