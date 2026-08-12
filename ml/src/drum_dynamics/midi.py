"""MIDI note-access helpers built on top of ``partitura`` note arrays.

``partitura`` returns a structured ``ndarray`` where each note is a tuple of
nine fields. Rather than remembering positional indices everywhere, we wrap a
single note in :class:`MidiNote` and expose named properties, and we name the
tuple positions in the :class:`Idx` enum.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import partitura


class Idx(Enum):
    """Positional indices into a partitura performance note tuple."""

    ONSET_SECS = 0
    DURATION_SECS = 1
    ONSET_TICKS = 2
    DURATION_TICKS = 3
    PITCH = 4
    VELOCITY = 5
    TRACK = 6
    CHANNEL = 7
    ID = 8


class MidiNote:
    """Convenience wrapper around one partitura performance-note tuple."""

    def __init__(self, notetuple):
        self.notetuple = notetuple

    @property
    def begin_secs(self):
        return self.notetuple[Idx.ONSET_SECS.value]

    @property
    def duration_secs(self):
        return self.notetuple[Idx.DURATION_SECS.value]

    @property
    def begin_ticks(self):
        return self.notetuple[Idx.ONSET_TICKS.value]

    @property
    def duration_ticks(self):
        return self.notetuple[Idx.DURATION_TICKS.value]

    @property
    def pitch(self):
        return self.notetuple[Idx.PITCH.value]

    @property
    def velocity(self):
        return self.notetuple[Idx.VELOCITY.value]

    @property
    def track(self):
        return self.notetuple[Idx.TRACK.value]

    @property
    def channel(self):
        return self.notetuple[Idx.CHANNEL.value]

    @property
    def id(self):
        return self.notetuple[Idx.ID.value]

    @property
    def end_secs(self):
        return self.begin_secs + self.duration_secs

    @property
    def end_ticks(self):
        return self.begin_ticks + self.duration_ticks

    def __repr__(self):
        return f"MidiNote({self.notetuple})"


# General MIDI percussion key map (MIDI note number -> drum piece name).
DRUM_MIDI_NAME = {
    35: "Acoustic Bass Drum",
    36: "Bass Drum 1",
    37: "Side Stick",
    38: "Acoustic Snare",
    39: "Hand Clap",
    40: "Electric Snare",
    41: "Low Floor Tom",
    42: "Closed Hi Hat",
    43: "High Floor Tom",
    44: "Pedal Hi-Hat",
    45: "Low Tom",
    46: "Open Hi-Hat",
    47: "Low-Mid Tom",
    48: "Hi-Mid Tom",
    49: "Crash Cymbal 1",
    50: "High Tom",
    51: "Ride Cymbal 1",
    52: "Chinese Cymbal",
    53: "Ride Bell",
    54: "Tambourine",
    55: "Splash Cymbal",
    56: "Cowbell",
    57: "Crash Cymbal 2",
    58: "Vibra Slap",
    59: "Ride Cymbal 2",
    60: "High Bongo",
    61: "Low Bongo",
    62: "Mute High Conga",
    63: "Open High Conga",
    64: "Low Conga",
    65: "High Timbale",
    66: "Low Timbale",
    67: "High Agogo",
    68: "Low Agogo",
    69: "Cabasa",
    70: "Maracas",
    71: "Short Whistle",
    72: "Long Whistle",
    73: "Short Guiro",
    74: "Long Guiro",
    75: "Claves",
    76: "High Wood Block",
    77: "Low Wood Block",
    78: "Mute Cuica",
    79: "Open Cuica",
    80: "Mute Triangle",
    81: "Open Triangle",
}


# E-GMD is recorded on a Roland TD-series electronic kit, which emits a few
# articulation pitches *outside* the General MIDI percussion range (notably the
# hi-hat "edge" hits). These appear in the data but are not in DRUM_MIDI_NAME.
EGMD_EXTRA_MIDI_NAME = {
    22: "Hi-Hat Closed (Edge)",
    26: "Hi-Hat Open (Edge)",
}


def drum_name(midi_number: int) -> str:
    """Human-readable drum name for a pitch, covering GM *and* E-GMD extras."""
    if midi_number in DRUM_MIDI_NAME:
        return DRUM_MIDI_NAME[midi_number]
    if midi_number in EGMD_EXTRA_MIDI_NAME:
        return EGMD_EXTRA_MIDI_NAME[midi_number]
    return f"(unmapped {midi_number})"


TONES_FORMAT = [
    "C{octave}",
    "C#{octave}/Db{octave}",
    "D{octave}",
    "D#{octave}/Eb{octave}",
    "E{octave}",
    "F{octave}",
    "F#{octave}/Gb{octave}",
    "G{octave}",
    "G#{octave}/Ab{octave}",
    "A{octave}",
    "A#{octave}/Bb{octave}",
    "B{octave}",
]


def midi_number_to_tone(midi_number: int) -> str:
    """Convert a MIDI note number to a tone name, e.g. ``60 -> 'C4'``."""
    if midi_number < 0 or midi_number > 127:
        raise ValueError("MIDI number must be in [0, 127].")
    return TONES_FORMAT[midi_number % 12].format(octave=midi_number // 12 - 1)


def load_note_array(path: str, part: int = 0) -> np.ndarray:
    """Load a performance MIDI file and return one part's note array.

    Every file in E-GMD contains a single performed part, so ``part=0`` is the
    sensible default.
    """
    performance = partitura.load_performance_midi(path)
    return performance.performedparts[part].note_array()
