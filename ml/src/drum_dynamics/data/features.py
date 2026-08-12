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


import pandas as pd

from ..core.voicemap import CANONICAL_VOICES, voice_of


def _log_clip_beats(delta_beats: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(delta_beats, 0.0, TIME_DELTA_CLIP_BEATS))


def build_note_features(note_array, meta) -> pd.DataFrame:
    """One structural feature row per note (design §4). No velocity leakage."""
    order = np.argsort(note_array["onset_sec"], kind="stable")
    na = note_array[order]
    onset = na["onset_sec"].astype(float)
    pitch = na["pitch"].astype(int)
    n = len(na)

    bpm = float(meta["bpm"])
    beat_dur = 60.0 / bpm
    bpb = beats_per_bar(meta["time_signature"])
    onset_beats = onset / beat_dur

    phase_beat, phase_bar = metrical_phase(onset, bpm, bpb)
    voices = np.array([voice_of(p) for p in pitch], dtype=object)

    # global consecutive deltas (any voice), in beats
    to_prev = np.full(n, TIME_DELTA_CLIP_BEATS)
    to_next = np.full(n, TIME_DELTA_CLIP_BEATS)
    if n > 1:
        d = np.diff(onset_beats)
        to_prev[1:] = d
        to_next[:-1] = d

    # same-voice consecutive deltas, in beats
    sv_prev = np.full(n, TIME_DELTA_CLIP_BEATS)
    sv_next = np.full(n, TIME_DELTA_CLIP_BEATS)
    for v in set(voices):
        idx = np.where(voices == v)[0]
        if idx.size > 1:
            dv = np.diff(onset_beats[idx])
            sv_prev[idx[1:]] = dv
            sv_next[idx[:-1]] = dv

    # simultaneity multi-hot + count, and ±1-beat density (vectorized via searchsorted)
    lo = np.searchsorted(onset_beats, onset_beats - SIMULTANEITY_TOL_BEATS, side="left")
    hi = np.searchsorted(onset_beats, onset_beats + SIMULTANEITY_TOL_BEATS, side="right")
    dlo = np.searchsorted(onset_beats, onset_beats - 1.0, side="left")
    dhi = np.searchsorted(onset_beats, onset_beats + 1.0, side="right")
    simult_count = hi - lo
    density = dhi - dlo
    multihot = {f"simult_{v}": np.zeros(n, dtype=np.int8) for v in CANONICAL_VOICES}
    for i in range(n):
        for j in range(lo[i], hi[i]):
            multihot[f"simult_{voices[j]}"][i] = 1

    style = str(meta["style"])
    out = pd.DataFrame({
        "file_id": str(meta["id"]),
        "drummer": str(meta["drummer"]),
        "split": str(meta["split"]),
        "onset_sec": onset,
        "bar_index": np.floor(onset / (beat_dur * bpb)).astype(int),
        "velocity": na["velocity"].astype(int),
        "voice": voices,
        "genre": style.split("/")[0],
        "style": style,
        "time_signature": str(meta["time_signature"]),
        "beat_type": str(meta["beat_type"]),
        "nearest_subdiv": nearest_subdivision(phase_beat),
        "phase_beat": phase_beat,
        "phase_bar": phase_bar,
        "sin_beat": np.sin(2 * np.pi * phase_beat),
        "cos_beat": np.cos(2 * np.pi * phase_beat),
        "sin_bar": np.sin(2 * np.pi * phase_bar),
        "cos_bar": np.cos(2 * np.pi * phase_bar),
        "swing_ratio": swing_ratio(phase_beat),
        "log_time_to_prev": _log_clip_beats(to_prev),
        "log_time_to_next": _log_clip_beats(to_next),
        "log_same_voice_prev": _log_clip_beats(sv_prev),
        "log_same_voice_next": _log_clip_beats(sv_next),
        "simult_count": simult_count.astype(int),
        "density_1beat": density.astype(int),
        "bpm": bpm,
    })
    for name, col in multihot.items():
        out[name] = col
    return out
