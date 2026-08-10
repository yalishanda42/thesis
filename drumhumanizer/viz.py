"""Piano-roll / drum-roll visualization for partitura note arrays.

A piano-roll is a ``(pitch, time)`` matrix whose cell value encodes note
velocity (normalized to ``[0, 1]``), giving a compact overview of *what* plays
*when* and *how hard*. Time is measured in MIDI ticks (integers) rather than
seconds to keep the matrix index-aligned.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .midi import DRUM_MIDI_NAME, MidiNote, midi_number_to_tone


def _build_roll(notes):
    """Return ``(roll, min_pitch)`` for a note array.

    ``roll`` has shape ``(n_pitches, n_ticks)`` and holds velocity/127.
    """
    start_tick = min(MidiNote(n).begin_ticks for n in notes)
    end_tick = max(MidiNote(n).end_ticks for n in notes)

    max_pitch = max(MidiNote(n).pitch for n in notes)
    min_pitch = min(MidiNote(n).pitch for n in notes)

    roll = np.zeros((max_pitch - min_pitch + 1, int(end_tick - start_tick)))

    for n in notes:
        mn = MidiNote(n)
        pitch_idx = mn.pitch - min_pitch
        start_idx = int(mn.begin_ticks - start_tick)
        end_idx = int(mn.end_ticks - start_tick)
        roll[pitch_idx, start_idx:end_idx] = mn.velocity / 127

    return roll, min_pitch


def piano_roll(notes, is_drums=False, figsize=(15, 12), show=True):
    """Plot a piano-roll (or drum-roll) with a velocity colorbar.

    Parameters
    ----------
    notes:
        A partitura note array (iterable of note tuples).
    is_drums:
        If ``True``, the y-axis is labelled with General MIDI drum-piece names
        instead of tone names.
    figsize:
        Matplotlib figure size.
    show:
        Call ``plt.show()`` before returning (set ``False`` to compose figures).
    """
    roll, min_pitch = _build_roll(notes)

    plt.figure(figsize=figsize)
    plt.imshow(roll, aspect="auto", origin="lower", interpolation="none", cmap="Reds")
    plt.xlabel("time (MIDI ticks)")
    plt.ylabel("drum kit piece" if is_drums else "pitch")

    if is_drums:
        plt.yticks(
            range(roll.shape[0]),
            [DRUM_MIDI_NAME.get(min_pitch + i, min_pitch + i) for i in range(roll.shape[0])],
        )
    else:
        plt.yticks(
            range(roll.shape[0]),
            [midi_number_to_tone(min_pitch + i) for i in range(roll.shape[0])],
        )

    cbar = plt.colorbar()
    cbar.set_label("velocity")
    ticks_count = 16
    cbar.set_ticks(np.linspace(0, 1, ticks_count))
    cbar.set_ticklabels(np.linspace(0, 127, ticks_count, dtype=int))

    if show:
        plt.show()


def drums_roll(notes, **kwargs):
    """Convenience wrapper: :func:`piano_roll` with drum-piece y-axis labels."""
    return piano_roll(notes, is_drums=True, **kwargs)
