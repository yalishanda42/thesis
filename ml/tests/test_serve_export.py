import pandas as pd
from drum_dynamics.serve.export import build_mdn_meta


def test_build_mdn_meta_shapes():
    df = pd.DataFrame({"genre": ["rock", "jazz", "rock"], "bpm": [120.0, 90.0, 100.0]})
    meta = build_mdn_meta(df)
    assert meta["head"] == "mdn"
    assert meta["genre_vocab"] == {"jazz": 1, "rock": 2}   # sorted, 0 reserved for <unk>
    assert abs(meta["bpm_mean"] - 103.3333) < 1e-3
    assert meta["bpm_std"] > 0
