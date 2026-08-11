import numpy as np
import pandas as pd

from drumhumanizer.seqdata import (
    NUMERIC_FEATURES,
    MAX_LEN,
    build_genre_vocab,
    bpm_stats,
)


def test_numeric_features_order_and_no_leakage():
    assert NUMERIC_FEATURES == [
        "sin_beat", "cos_beat", "sin_bar", "cos_bar",
        "bpm_z", "log_time_to_prev", "same_onset",
    ]
    assert not [c for c in NUMERIC_FEATURES if "vel" in c.lower()]
    assert MAX_LEN == 512


def test_build_genre_vocab_reserves_zero_for_unk():
    df = pd.DataFrame({"genre": ["rock", "funk", "rock", "jazz"]})
    vocab = build_genre_vocab(df)
    assert 0 not in vocab.values()          # 0 is <unk>
    assert set(vocab) == {"funk", "jazz", "rock"}
    assert vocab["funk"] == 1 and vocab["jazz"] == 2 and vocab["rock"] == 3  # sorted


def test_bpm_stats_floors_std():
    df = pd.DataFrame({"bpm": [120.0, 120.0, 120.0]})   # zero variance
    mean, std = bpm_stats(df)
    assert mean == 120.0
    assert std == 1.0
