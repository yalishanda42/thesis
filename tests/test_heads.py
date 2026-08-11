import math

import torch
import torch.distributions as D

from drumhumanizer import heads


def test_output_dims():
    assert heads.head_output_dim("deterministic") == 1
    assert heads.head_output_dim("gaussian") == 2
    assert heads.head_output_dim("mdn") == 3 * heads.MDN_K
    assert heads.head_output_dim("categorical") == heads.N_BINS


def test_velocity_to_bin_and_centers():
    y = torch.tensor([0.0, 3.9, 4.0, 127.0, 200.0])
    assert heads.velocity_to_bin(y).tolist() == [0, 0, 1, 31, 31]
    c = heads.bin_centers()
    assert c.shape == (heads.N_BINS,)
    assert c[0].item() == 2.0 and c[-1].item() == 126.0


def test_gaussian_logprob_matches_torch_distribution():
    raw = torch.tensor([[2.5, 0.7]])          # mu=2.5, sigma=softplus(0.7)+1e-3
    y = torch.tensor([3.0])
    mu, sigma = heads._gauss_params(raw)
    oracle = D.Normal(mu, sigma).log_prob(y)
    assert torch.allclose(heads.logprob("gaussian", raw, y), oracle, atol=1e-6)
    assert torch.allclose(heads.point("gaussian", raw), mu)


def test_mdn_logprob_matches_mixture_oracle():
    K = heads.MDN_K
    torch.manual_seed(0)
    raw = torch.randn(4, K * 3)
    y = torch.rand(4) * 127
    log_pi, mu, sigma = heads._mdn_params(raw)
    mix = D.MixtureSameFamily(D.Categorical(logits=log_pi), D.Normal(mu, sigma))
    assert torch.allclose(heads.logprob("mdn", raw, y), mix.log_prob(y), atol=1e-5)
    # point readout is the mixture mean
    assert torch.allclose(heads.point("mdn", raw), (log_pi.exp() * mu).sum(-1), atol=1e-5)


def test_categorical_nll_matches_cross_entropy():
    import torch.nn.functional as F
    torch.manual_seed(0)
    raw = torch.randn(6, heads.N_BINS)
    y = torch.rand(6) * 127
    b = heads.velocity_to_bin(y)
    pad = torch.zeros(6, dtype=torch.bool)
    assert torch.allclose(heads.nll("categorical", raw, y, pad),
                          F.cross_entropy(raw, b), atol=1e-6)


def test_bin_logprob_is_comparable_and_bounded():
    y = torch.tensor([50.0])
    for raw, ht in [(torch.tensor([[50.0, 0.5]]), "gaussian"),
                    (torch.randn(1, 3 * heads.MDN_K), "mdn"),
                    (torch.randn(1, heads.N_BINS), "categorical")]:
        lp = heads.bin_logprob(ht, raw, y)
        assert lp.shape == (1,)
        assert (lp <= 1e-6).all()            # log of a probability mass <= 0


def test_nll_masks_padding():
    raw = torch.zeros(1, 3, 2)               # gaussian
    y = torch.tensor([[50.0, 50.0, 9999.0]])
    pad = torch.tensor([[False, False, True]])
    loss_masked = heads.nll("gaussian", raw, y, pad)
    loss_first_two = heads.nll("gaussian", raw[:, :2], y[:, :2], pad[:, :2])
    assert torch.allclose(loss_masked, loss_first_two, atol=1e-6)


def test_sample_is_in_range_and_reproducible():
    torch.manual_seed(0)
    for ht, raw in [("gaussian", torch.randn(50, 2)),
                    ("mdn", torch.randn(50, 3 * heads.MDN_K)),
                    ("categorical", torch.randn(50, heads.N_BINS))]:
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        s1 = heads.sample(ht, raw, generator=g1)
        s2 = heads.sample(ht, raw, generator=g2)
        assert s1.shape == (50,)
        assert (s1 >= 0).all() and (s1 <= 127).all()
        assert torch.allclose(s1, s2)        # seeded -> reproducible
