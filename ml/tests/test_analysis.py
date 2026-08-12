import numpy as np
import pandas as pd

from drum_dynamics import analysis


def test_add_metrical_cols_beat_and_backbeat():
    df = pd.DataFrame({
        "phase_bar": [0.0, 0.25, 0.5, 0.75, 0.99],
        "phase_beat": [0.0, 0.02, 0.5, 0.0, 0.5],
    })
    out = analysis.add_metrical_cols(df, beats_per_bar=4)
    assert out["beat_in_bar"].tolist() == [0, 1, 2, 3, 3]
    assert out["is_downbeat"].tolist() == [True, False, False, False, False]
    # backbeats are beats 2 & 4 -> 0-indexed 1 & 3
    assert out["is_backbeat"].tolist() == [False, True, False, True, True]
    # on-beat when phase_beat is near an integer pulse, else off-beat
    assert out["metrical_class"].tolist() == [
        "on-beat", "on-beat", "off-beat", "on-beat", "off-beat"]


def test_residual_table_stats():
    df = pd.DataFrame({
        "voice": ["kick", "kick", "snare", "snare"],
        "velocity": [100.0, 80.0, 40.0, 60.0],
        "pred": [90.0, 90.0, 50.0, 50.0],
    })
    r = analysis.residual_table(df, "kick" if False else "voice")
    assert r.loc["kick", "n"] == 2
    assert r.loc["kick", "true_mean"] == 90.0
    assert r.loc["kick", "pred_mean"] == 90.0
    assert r.loc["kick", "bias"] == 0.0
    # MAE for kick: |90-100|+|90-80| over 2 = 10
    assert r.loc["kick", "mae"] == 10.0
    assert r.loc["snare", "bias"] == 0.0  # pred_mean 50 == true_mean 50
    assert r.loc["snare", "mae"] == 10.0


def test_dynamic_level_table_reveals_regression_to_mean():
    # true velocities spread 0..127; predictions pulled toward the overall mean
    rng = np.random.RandomState(0)
    true = rng.uniform(0, 127, 4000)
    mean = true.mean()
    pred = mean + 0.4 * (true - mean)          # classic shrink toward the mean
    df = pd.DataFrame({"velocity": true, "pred": pred})
    t = analysis.dynamic_level_table(df, n_bins=8)
    # softest bin: predicted too LOUD (bias > 0); loudest bin: too SOFT (bias < 0)
    assert t["bias"].iloc[0] > 0
    assert t["bias"].iloc[-1] < 0
    # bias decreases monotonically from soft to loud
    assert (np.diff(t["bias"].to_numpy()) < 0).all()


def test_embedding_2d_shape_and_variance_order():
    # points vary mostly along axis 0 -> PC1 should capture that spread
    W = np.zeros((10, 5))
    W[:, 0] = np.linspace(-5, 5, 10)
    W[:, 1] = np.linspace(-1, 1, 10) * 0.1
    xy = analysis.embedding_2d(W)
    assert xy.shape == (10, 2)
    # PC1 spread >> PC2 spread
    assert xy[:, 0].std() > 5 * xy[:, 1].std()
