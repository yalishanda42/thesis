"""drumhumanizer — utilities for the drum-velocity "humanization" thesis.

This package extracts the helper code that was previously duplicated inline
across the exploratory notebooks (``poc.ipynb`` and ``notebooks/eda.ipynb``):

* :mod:`drumhumanizer.midi`     — MIDI note access helpers (``MidiNote``, ``Idx``),
  the General MIDI percussion map, and loading helpers built on ``partitura``.
* :mod:`drumhumanizer.viz`      — piano-roll / drum-roll visualization.
* :mod:`drumhumanizer.playback` — in-notebook audio rendering via FluidSynth.

The goal of the wider project is to predict a "best-fitting" velocity for every
note in a MIDI *drum* track — i.e. humanizing dynamics, analogous to existing
tempo humanization. We focus on the Expanded Groove MIDI Dataset (E-GMD).
"""

from .midi import (
    Idx,
    MidiNote,
    DRUM_MIDI_NAME,
    EGMD_EXTRA_MIDI_NAME,
    TONES_FORMAT,
    drum_name,
    midi_number_to_tone,
    load_note_array,
)
from .voicemap import CANONICAL_VOICES, PITCH_TO_VOICE, voice_of, voice_index
from .viz import piano_roll, drums_roll
from .playback import play_midi_file, play_midi_notes, set_soundfont, get_soundfont

__all__ = [
    "Idx",
    "MidiNote",
    "DRUM_MIDI_NAME",
    "EGMD_EXTRA_MIDI_NAME",
    "TONES_FORMAT",
    "drum_name",
    "midi_number_to_tone",
    "load_note_array",
    "CANONICAL_VOICES",
    "PITCH_TO_VOICE",
    "voice_of",
    "voice_index",
    "piano_roll",
    "drums_roll",
    "play_midi_file",
    "play_midi_notes",
    "set_soundfont",
    "get_soundfont",
]
