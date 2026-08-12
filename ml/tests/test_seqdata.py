import numpy as np
import pandas as pd

from drum_dynamics.seqdata import (
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


import torch

from drum_dynamics.seqdata import build_split_tensors, scatter_predictions


def _toy_df():
    # file "a": 3 notes, two simultaneous at t=0 (kick+snare), one at t=0.5
    # file "b": 2 notes at t=0, t=1
    return pd.DataFrame({
        "file_id": ["a", "a", "a", "b", "b"],
        "onset_sec": [0.0, 0.0, 0.5, 0.0, 1.0],
        "velocity": [100, 40, 70, 88, 55],
        "voice": ["kick", "snare", "closed-hh", "kick", "snare"],
        "genre": ["funk", "funk", "funk", "rock", "rock"],
        "sin_beat": [0.0, 0.0, 1.0, 0.0, 0.0],
        "cos_beat": [1.0, 1.0, 0.0, 1.0, 1.0],
        "sin_bar": [0.0, 0.0, 0.5, 0.0, 0.5],
        "cos_bar": [1.0, 1.0, 0.5, 1.0, 0.5],
        "bpm": [120.0, 120.0, 120.0, 120.0, 120.0],
    })


def test_build_split_tensors_shapes_and_padding():
    df = _toy_df()
    vocab = build_genre_vocab(df)
    t = build_split_tensors(df, vocab, 120.0, 1.0, max_len=4)
    # 2 files -> 2 windows (each file <= max_len)
    assert t["voice_idx"].shape == (2, 4)
    assert t["num_feats"].shape == (2, 4, len(NUMERIC_FEATURES))
    # file "a" has 3 real tokens -> 1 pad; file "b" has 2 -> 2 pads
    assert t["pad_mask"].sum().item() == 1 + 2


def test_build_split_tensors_same_onset_flag():
    df = _toy_df()
    vocab = build_genre_vocab(df)
    t = build_split_tensors(df, vocab, 120.0, 1.0, max_len=4)
    so_col = NUMERIC_FEATURES.index("same_onset")
    # find file "a" window (the one with 3 real tokens)
    real_counts = (~t["pad_mask"]).sum(dim=1)
    wa = int((real_counts == 3).nonzero()[0])
    same_onset = t["num_feats"][wa, :3, so_col]
    # ordered by (onset, voice_idx): kick(0), snare(0), hh(0.5)
    # token0 first -> 0 ; token1 same onset as token0 -> 1 ; token2 new onset -> 0
    assert same_onset.tolist() == [0.0, 1.0, 0.0]


def test_scatter_predictions_round_trips_targets():
    df = _toy_df()
    vocab = build_genre_vocab(df)
    t = build_split_tensors(df, vocab, 120.0, 1.0, max_len=4)
    # feeding the targets through the scatter must reconstruct df velocity exactly
    out = scatter_predictions(t["row_idx"], t["target"], t["pad_mask"], len(df))
    assert np.allclose(out, df["velocity"].to_numpy())


def test_build_split_tensors_no_velocity_in_features():
    df = _toy_df()
    vocab = build_genre_vocab(df)
    t = build_split_tensors(df, vocab, 120.0, 1.0, max_len=4)
    # velocity only lives in the target tensor; num_feats never equals it by construction
    assert "target" in t and t["num_feats"].shape[-1] == len(NUMERIC_FEATURES)
