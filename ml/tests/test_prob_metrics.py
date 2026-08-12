import numpy as np

from drum_dynamics.metrics import wasserstein1d, hist_intersection


def test_wasserstein_zero_for_identical():
    a = np.array([10.0, 20, 30, 40])
    assert wasserstein1d(a, a) == 0.0
    assert wasserstein1d(a, a + 5) == 5.0        # constant shift -> EMD = shift


def test_hist_intersection_bounds():
    rng = np.random.RandomState(0)
    a = rng.uniform(0, 128, 5000)
    assert np.isclose(hist_intersection(a, a), 1.0)          # identical -> 1
    disjoint = hist_intersection(np.zeros(100), np.full(100, 127.0))
    assert disjoint < 0.05                                    # no overlap -> ~0
