import numpy as np
import pandas as pd
import lightgbm as lgb

from drum_dynamics.serve.models import LgbmModel, MdnModel, Engine
from drum_dynamics.models.model import VelocityTransformer
from drum_dynamics.data.features import build_note_features

_DT = np.dtype([("onset_sec", float), ("pitch", int), ("velocity", int)])


def _feature_df():
    na = np.array([(0.0, 36, 100), (0.0, 38, 40), (0.5, 42, 70), (1.0, 36, 90)], dtype=_DT)
    meta = dict(id="x", drummer="x", split="train", bpm=120, time_signature="4-4",
                style="funk/groove", beat_type="beat")
    return build_note_features(na, meta)


def test_mdn_predict_all_plumbing_with_fresh_model():
    df = _feature_df()
    gv = {"funk": 1}
    m = VelocityTransformer(n_genres=len(gv) + 1, head="mdn")
    mdn = MdnModel(m, gv, bpm_mean=120.0, bpm_std=10.0)
    out = mdn.predict_all(df, temperature=1.0, seed=0)
    assert out.shape == (len(df),)
    assert ((out >= 0) & (out <= 127)).all()


def test_lgbm_predict_all_returns_row_aligned():
    df = _feature_df()
    from drum_dynamics.serve.models import LgbmModel
    CAT = ["voice", "genre", "style", "time_signature", "beat_type", "nearest_subdiv"]
    DROP = ["file_id", "drummer", "split", "onset_sec", "bar_index", "velocity"]
    X = df.drop(columns=DROP)
    for c in CAT:
        X[c] = X[c].astype("category")
    model = lgb.LGBMRegressor(n_estimators=3, min_child_samples=1).fit(X, df["velocity"], categorical_feature=CAT)
    bundle = {"model": model, "cat": CAT, "drop": DROP,
              "cat_categories": {c: X[c].cat.categories for c in CAT}, "best_iteration": 3}
    out = LgbmModel(bundle).predict_all(df)
    assert out.shape == (len(df),)


def test_engine_routes_and_levels():
    class Fake:
        def predict_all(self, df, **kw):
            return np.full(len(df), 111.0)
    eng = Engine(Fake(), Fake(), styles=["funk"], genres=["funk"])
    req = {"model": "lgbm", "style": "funk", "temperature": 1.0, "blend": 1.0,
           "beat_type": "beat", "bpm": 120, "time_signature": "4-4",
           "notes": [{"index": 0, "pitch": 36, "onset_sec": 0.0, "velocity": 1, "selected": True}]}
    assert eng.predict(req) == {0: 111}
    assert eng.levels()["models"] == ["lgbm", "mdn"]
