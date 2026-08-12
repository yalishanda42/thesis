"""Smoke tests for the drum_dynamics package.

These run against a real E-GMD file (the first one in the metadata CSV) to
verify the extracted helpers behave the same as the original inline notebook
code. Run with:  .venv/bin/python -m pytest tests/ -v
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless: never open a window during tests

import numpy as np
import pandas as pd

from drum_dynamics import (
    Idx,
    MidiNote,
    DRUM_MIDI_NAME,
    midi_number_to_tone,
    load_note_array,
    piano_roll,
    drums_roll,
    play_midi_notes,
    get_soundfont,
)
from drum_dynamics.viz import _build_roll

EGMD_BASE = os.path.join("data", "e-gmd", "e-gmd-v1.0.0")
EGMD_CSV = os.path.join(EGMD_BASE, "e-gmd-v1.0.0.csv")


def _first_drum_file():
    df = pd.read_csv(EGMD_CSV)
    # pick the first "beat" (a full groove, not a fill) for a representative roll
    row = df[df["beat_type"] == "beat"].iloc[0]
    return os.path.join(EGMD_BASE, row["midi_filename"])


def test_midi_number_to_tone():
    assert midi_number_to_tone(60) == "C4"
    assert midi_number_to_tone(69).startswith("A")


def test_drum_map_known_pieces():
    assert DRUM_MIDI_NAME[36] == "Bass Drum 1"
    assert DRUM_MIDI_NAME[38] == "Acoustic Snare"
    assert DRUM_MIDI_NAME[42] == "Closed Hi Hat"


def test_drum_name_covers_egmd_extras():
    from drum_dynamics import drum_name

    # General MIDI pitch resolves via the GM map
    assert drum_name(36) == "Bass Drum 1"
    # E-GMD Roland hi-hat edge articulations (outside GM) resolve too
    assert "Edge" in drum_name(22)
    assert "Edge" in drum_name(26)
    # unknown pitch degrades gracefully
    assert drum_name(3) == "(unmapped 3)"


def test_load_note_array_and_midinote():
    notes = load_note_array(_first_drum_file())
    assert len(notes) > 0
    n = MidiNote(notes[0])
    assert 0 <= n.velocity <= 127
    assert 0 <= n.pitch <= 127
    assert n.end_secs >= n.begin_secs
    # channel 9 is the GM drum channel
    assert n.channel == 9


def test_build_roll_shape():
    notes = load_note_array(_first_drum_file())
    roll, min_pitch = _build_roll(notes)
    assert roll.ndim == 2
    assert roll.max() <= 1.0 and roll.min() >= 0.0
    # velocities present -> some non-zero cells
    assert roll.max() > 0.0


def test_viz_runs_headless():
    notes = load_note_array(_first_drum_file())
    piano_roll(notes, is_drums=True, show=False)
    drums_roll(notes, show=False)


def test_playback_returns_audio():
    notes = load_note_array(_first_drum_file())
    assert os.path.exists(get_soundfont()), f"soundfont missing: {get_soundfont()}"
    audio = play_midi_notes(notes, is_drums=True)
    # IPython Audio stores rendered PCM on .data
    assert audio.data is not None and len(audio.data) > 0
