from drum_dynamics.voicemap import CANONICAL_VOICES, voice_of, voice_index


def test_canonical_voices_are_14_unique():
    assert len(CANONICAL_VOICES) == 14
    assert len(set(CANONICAL_VOICES)) == 14


def test_voice_of_locked_from_phase0():
    assert voice_of(36) == "kick"
    assert voice_of(38) == "snare"
    assert voice_of(40) == "snare-accent"          # distinct hot articulation
    assert voice_of(22) == "closed-hh-edge"        # Roland edge hit kept separate
    assert voice_of(26) == "open-hh"
    assert voice_of(48) == "tom" and voice_of(43) == "tom"   # toms merged
    assert voice_of(51) == "ride" and voice_of(59) == "ride"


def test_voice_of_unknown_falls_back():
    assert voice_of(3) == "aux-perc"


def test_voice_index_matches_order():
    assert voice_index("kick") == 0
    assert voice_index(CANONICAL_VOICES[-1]) == 13
