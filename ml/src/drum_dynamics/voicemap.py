"""Canonical drum voice grouping, locked from Phase 0 (docs/phase0/results.md).

Merges rare/indistinguishable GM pitches and keeps common, distributionally
distinct articulations separate (Roland hi-hat edge hits 22/26, electric snare 40).
The order of CANONICAL_VOICES is fixed — it defines the simultaneity multi-hot
column order in the feature table.
"""

from __future__ import annotations

CANONICAL_VOICES = [
    "kick",
    "snare",
    "snare-accent",
    "side-stick",
    "closed-hh",
    "closed-hh-edge",
    "pedal-hh",
    "open-hh",
    "ride",
    "ride-bell",
    "tom",
    "crash",
    "aux-cymbal",
    "aux-perc",
]

PITCH_TO_VOICE = {
    36: "kick",
    38: "snare",
    40: "snare-accent",
    37: "side-stick",
    42: "closed-hh",
    22: "closed-hh-edge",
    44: "pedal-hh",
    46: "open-hh", 26: "open-hh",
    51: "ride", 59: "ride",
    53: "ride-bell",
    41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
    49: "crash", 57: "crash",
    52: "aux-cymbal", 55: "aux-cymbal",
    54: "aux-perc", 58: "aux-perc", 39: "aux-perc", 56: "aux-perc",
}

_VOICE_TO_INDEX = {v: i for i, v in enumerate(CANONICAL_VOICES)}


def voice_of(pitch: int) -> str:
    """Canonical voice for a MIDI pitch; unknown pitches fall back to 'aux-perc'."""
    return PITCH_TO_VOICE.get(int(pitch), "aux-perc")


def voice_index(voice: str) -> int:
    return _VOICE_TO_INDEX[voice]
