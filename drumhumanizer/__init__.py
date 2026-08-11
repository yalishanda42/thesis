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

import importlib

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
from .features import build_note_features
from .baselines import GlobalMeanBaseline, LookupTableBaseline
from .metrics import mae, rmse, evaluate, wasserstein1d, hist_intersection
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
    "build_note_features",
    "GlobalMeanBaseline",
    "LookupTableBaseline",
    "mae",
    "rmse",
    "evaluate",
    "wasserstein1d",
    "hist_intersection",
    "NUMERIC_FEATURES",
    "MAX_LEN",
    "build_genre_vocab",
    "bpm_stats",
    "build_split_tensors",
    "scatter_predictions",
    "VelocityTransformer",
    "warm_start_backbone",
    "piano_roll",
    "drums_roll",
    "play_midi_file",
    "play_midi_notes",
    "set_soundfont",
    "get_soundfont",
]

# Torch-dependent symbols are imported lazily: importing this package (or its
# light submodules like `features`/`midi`) must NOT pull in torch. On macOS,
# loading torch's OpenMP runtime before LightGBM's segfaults, so the tabular
# path must stay torch-free. Accessing these names (or importing the
# `.model`/`.seqdata` submodules directly) loads torch on demand.
_LAZY = {
    "NUMERIC_FEATURES": "seqdata",
    "MAX_LEN": "seqdata",
    "build_genre_vocab": "seqdata",
    "bpm_stats": "seqdata",
    "build_split_tensors": "seqdata",
    "scatter_predictions": "seqdata",
    "VelocityTransformer": "model",
    "warm_start_backbone": "model",
}


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
