"""Section A tabular features for the drum-velocity model (design §4).

STRUCTURAL ONLY — no note's velocity is ever used as a feature (design §1.1).
"""

from __future__ import annotations

import numpy as np

SIMULTANEITY_TOL_BEATS = 0.02      # Phase 0: fixed (no near-zero valley)
TIME_DELTA_CLIP_BEATS = 8.0        # clip inter-onset deltas before log1p
N_PHASE_BINS = 16                  # phase_beat bins for the lookup-table baseline

# candidate subdivision grids: name -> divisions per beat
SUBDIVISIONS = {
    "8th": 2,
    "16th": 4,
    "32nd": 8,
    "8th-triplet": 3,
    "quintuplet": 5,
}


def beats_per_bar(time_signature: str) -> int:
    """Beats per bar from an E-GMD time-signature string like '4-4' -> 4."""
    return int(str(time_signature).split("-")[0])


def metrical_phase(onset_sec: np.ndarray, bpm: float, bpb: int):
    """Continuous metrical phase within the beat and within the bar, each in [0, 1)."""
    onset_sec = np.asarray(onset_sec, dtype=float)
    beat_dur = 60.0 / float(bpm)
    bar_dur = beat_dur * bpb
    phase_beat = np.mod(onset_sec, beat_dur) / beat_dur
    phase_bar = np.mod(onset_sec, bar_dur) / bar_dur
    return phase_beat, phase_bar


def swing_ratio(phase_beat: np.ndarray) -> np.ndarray:
    """How far an offbeat is pushed toward the triplet position.

    0 at the straight 8th (phase 0.5), 1 at the 8th-note-triplet (phase 2/3).
    Defined only in the offbeat region [0.4, 0.8]; 0 elsewhere (onbeats etc.).
    """
    phase_beat = np.asarray(phase_beat, dtype=float)
    out = np.zeros_like(phase_beat)
    region = (phase_beat >= 0.4) & (phase_beat <= 0.8)
    out[region] = (phase_beat[region] - 0.5) / (2.0 / 3.0 - 0.5)
    return out


def nearest_subdivision(phase_beat: np.ndarray) -> np.ndarray:
    """For each onset, the candidate grid whose nearest gridline it is closest to."""
    phase_beat = np.asarray(phase_beat, dtype=float)
    names = list(SUBDIVISIONS)
    # distance to nearest gridline for each grid (phase is circular on [0,1))
    dists = np.empty((len(names), phase_beat.size))
    for i, name in enumerate(names):
        d = SUBDIVISIONS[name]
        scaled = phase_beat * d
        dists[i] = np.abs(scaled - np.round(scaled)) / d
    return np.array(names, dtype=object)[np.argmin(dists, axis=0)]
