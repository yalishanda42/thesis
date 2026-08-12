import numpy as np
import pandas as pd

from drum_dynamics.eval.metrics import evaluate, mae, rmse


def test_mae_rmse():
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0
    assert np.isclose(rmse([0, 0], [1, 1]), 1.0)


def test_evaluate_perfect_prediction():
    df = pd.DataFrame({
        "velocity": [10, 20, 30, 40, 50, 60],
        "file_id": ["a", "a", "a", "b", "b", "b"],
        "bar_index": [0, 0, 0, 0, 0, 0],
        "genre": ["funk", "funk", "funk", "rock", "rock", "rock"],
    })
    m = evaluate(df, np.array(df["velocity"], dtype=float))
    assert m["mae"] == 0.0
    assert np.isclose(m["per_track_pearson"], 1.0)
    assert np.isclose(m["within_bar_spearman"], 1.0)
    assert set(m["per_genre_mae"]) == {"funk", "rock"}
    assert m["per_genre_mae"]["funk"] == 0.0


def test_evaluate_std_ratio_detects_flattening():
    df = pd.DataFrame({
        "velocity": [0, 50, 100, 0, 50, 100],
        "file_id": ["a"] * 6, "bar_index": [0] * 6, "genre": ["funk"] * 6,
    })
    flat = np.full(6, 50.0)                 # predicts the mean -> no spread
    m = evaluate(df, flat)
    assert m["global_std_ratio"] < 0.1     # pred std / true std ~ 0
