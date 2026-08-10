"""In-notebook MIDI playback via FluidSynth.

Renders a partitura note array to audio using a General MIDI SoundFont and
returns an ``IPython.display.Audio`` widget. Requires the ``fluidsynth`` system
library (``brew install fluid-synth`` / ``apt-get install fluidsynth``), the
``pyfluidsynth`` Python bindings, and a ``.sf2`` SoundFont.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from ctypes.util import find_library

import numpy as np

from .midi import MidiNote, load_note_array


def _ensure_fluidsynth_discoverable() -> None:
    """Best-effort: help pyfluidsynth find libfluidsynth on macOS.

    pyfluidsynth locates the native library via ``ctypes.util.find_library`` and,
    as a fallback, ``$HOMEBREW_PREFIX/lib/libfluidsynth.dylib``. Homebrew installed
    in a non-standard prefix (e.g. ``~/homebrew``) isn't on the default search
    path, so we set ``HOMEBREW_PREFIX`` from ``brew --prefix`` if needed. Must run
    before ``import fluidsynth``.
    """
    if sys.platform != "darwin":
        return
    if find_library("fluidsynth") or os.getenv("HOMEBREW_PREFIX"):
        return
    brew = shutil.which("brew")
    if not brew:
        return
    try:
        prefix = subprocess.check_output([brew, "--prefix"], text=True).strip()
    except Exception:
        return
    if os.path.exists(os.path.join(prefix, "lib", "libfluidsynth.dylib")):
        os.environ["HOMEBREW_PREFIX"] = prefix

# Default SoundFont location. Override with the DRUMHUMANIZER_SF2 env var or
# :func:`set_soundfont`. FluidR3_GM is a General MIDI bank that includes a
# percussion set on bank 128.
_DEFAULT_SF_PATH = os.environ.get(
    "DRUMHUMANIZER_SF2",
    os.path.join("sf", "big", "FluidR3_GM.sf2"),
)

_sf_path = _DEFAULT_SF_PATH


def set_soundfont(path: str) -> None:
    """Set the SoundFont (``.sf2``) file used for playback."""
    global _sf_path
    _sf_path = path


def get_soundfont() -> str:
    """Return the currently configured SoundFont path."""
    return _sf_path


def play_midi_file(path: str, is_drums: bool = False, sr: int = 44100):
    """Load a MIDI file's first part and render it to notebook audio."""
    notes = load_note_array(path)
    return play_midi_notes(notes, is_drums=is_drums, sr=sr)


def play_midi_notes(notes: np.ndarray, is_drums: bool = False, sr: int = 44100):
    """Render a partitura note array to an ``IPython.display.Audio`` widget.

    Parameters
    ----------
    notes:
        A partitura note array.
    is_drums:
        Route to the GM percussion set (channel 9, bank 128) when ``True``,
        otherwise the melodic piano preset (channel 0, bank 0).
    sr:
        Sample rate in Hz.
    """
    # Imported lazily so the rest of the package works without the native lib.
    _ensure_fluidsynth_discoverable()
    import fluidsynth
    from IPython.display import Audio

    fl = fluidsynth.Synth(samplerate=sr)
    sfid = fl.sfload(_sf_path)
    channel = 9 if is_drums else 0
    bank = 128 if is_drums else 0
    preset = 0
    fl.program_select(channel, sfid, bank, preset)

    # Build a time-ordered timeline of note-on / note-off events.
    events: dict[float, tuple[str, MidiNote]] = {}
    events.update({MidiNote(n).begin_secs: ("on", MidiNote(n)) for n in notes})
    events.update({MidiNote(n).end_secs: ("off", MidiNote(n)) for n in notes})
    sorted_event_times = sorted(events.keys())

    audio_data = []
    for curr_time, next_time in zip(sorted_event_times[:-1], sorted_event_times[1:]):
        event_type, note = events[curr_time]

        if event_type == "on":
            fl.noteon(note.channel, note.pitch, note.velocity)
        elif event_type == "off":
            fl.noteoff(note.channel, note.pitch)

        interval_samples = fl.get_samples(int((next_time - curr_time) * sr))
        # get_samples returns interleaved stereo; keep the left channel (mono).
        audio_data.extend(interval_samples[::2])

    fl.delete()

    audio_data = np.array(audio_data, dtype="int16") / (2**15 - 1)
    return Audio(audio_data, rate=sr)
