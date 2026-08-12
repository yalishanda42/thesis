import numpy as np
import pandas as pd

from drum_dynamics.baselines import GlobalMeanBaseline, LookupTableBaseline


def _df():
    return pd.DataFrame({
        "voice": ["kick", "kick", "snare", "snare"],
        "genre": ["funk", "funk", "funk", "funk"],
        "phase_beat": [0.0, 0.0, 0.5, 0.5],
        "velocity": [100, 110, 40, 50],
    })


def test_global_mean_predicts_constant():
    b = GlobalMeanBaseline().fit(_df())
    pred = b.predict(_df())
    assert np.allclose(pred, 75.0)


def test_lookup_table_uses_group_mean():
    b = LookupTableBaseline().fit(_df())
    pred = b.predict(_df())
    assert np.allclose(pred, [105, 105, 45, 45])


def test_lookup_table_falls_back_for_unseen_key():
    b = LookupTableBaseline().fit(_df())
    unseen = pd.DataFrame({"voice": ["tom"], "genre": ["jazz"], "phase_beat": [0.3]})
    pred = b.predict(unseen)               # no matching key -> global mean
    assert np.isclose(pred[0], 75.0)
