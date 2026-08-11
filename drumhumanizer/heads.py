"""Probabilistic output heads for the velocity model (design Plan C §3-§5).

Each head interprets the raw per-token output of ``VelocityTransformer`` as the
parameters of a distribution over velocity. Functions are stateless and keyed by
``head_type``; the backbone stays head-agnostic. Losses are masked NLL.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

N_BINS = 32
BIN_WIDTH = 128 // N_BINS            # 4
MDN_K = 5
_SIGMA_FLOOR_MDN = 1.0
_SIGMA_FLOOR_GAUSS = 1e-3
_LOG_2PI = math.log(2 * math.pi)
_SQRT2 = math.sqrt(2.0)

HEAD_OUTPUT_DIM = {"deterministic": 1, "gaussian": 2, "mdn": 3 * MDN_K, "categorical": N_BINS}


def head_output_dim(head_type: str) -> int:
    return HEAD_OUTPUT_DIM[head_type]


def bin_centers(device=None) -> torch.Tensor:
    return torch.arange(N_BINS, dtype=torch.float, device=device) * BIN_WIDTH + BIN_WIDTH / 2


def velocity_to_bin(y: torch.Tensor) -> torch.Tensor:
    return torch.clamp((y / BIN_WIDTH).long(), 0, N_BINS - 1)


# ── Gaussian ────────────────────────────────────────────────────────────────
def _gauss_params(raw):
    mu = raw[..., 0]
    sigma = F.softplus(raw[..., 1]) + _SIGMA_FLOOR_GAUSS
    return mu, sigma


def _normal_logprob(y, mu, sigma):
    return -0.5 * _LOG_2PI - torch.log(sigma) - 0.5 * ((y - mu) / sigma) ** 2


def _normal_cdf(x, mu, sigma):
    return 0.5 * (1.0 + torch.erf((x - mu) / (sigma * _SQRT2)))


# ── MDN ─────────────────────────────────────────────────────────────────────
def _mdn_params(raw):
    r = raw.view(*raw.shape[:-1], 3, MDN_K)
    log_pi = torch.log_softmax(r[..., 0, :], dim=-1)
    mu = r[..., 1, :]
    sigma = F.softplus(r[..., 2, :]) + _SIGMA_FLOOR_MDN
    return log_pi, mu, sigma


# ── per-token log-probability (each head's own measure) ──────────────────────
def logprob(head_type, raw, y):
    if head_type == "gaussian":
        mu, sigma = _gauss_params(raw)
        return _normal_logprob(y, mu, sigma)
    if head_type == "mdn":
        log_pi, mu, sigma = _mdn_params(raw)
        comp = _normal_logprob(y.unsqueeze(-1), mu, sigma)      # [..., K]
        return torch.logsumexp(log_pi + comp, dim=-1)
    if head_type == "categorical":
        logp = torch.log_softmax(raw, dim=-1)
        b = velocity_to_bin(y).unsqueeze(-1)
        return logp.gather(-1, b).squeeze(-1)
    raise ValueError(f"unknown head_type {head_type!r}")


def nll(head_type, raw, y, pad_mask):
    lp = logprob(head_type, raw, y)
    keep = ~pad_mask
    return -(lp[keep]).mean()


def bin_logprob(head_type, raw, y):
    """log P(true velocity's 32-bin) under the head — comparable across heads."""
    if head_type == "categorical":
        return logprob("categorical", raw, y)
    b = velocity_to_bin(y)
    lo = (b * BIN_WIDTH).float()
    hi = lo + BIN_WIDTH
    if head_type == "gaussian":
        mu, sigma = _gauss_params(raw)
        mass = _normal_cdf(hi, mu, sigma) - _normal_cdf(lo, mu, sigma)
    elif head_type == "mdn":
        log_pi, mu, sigma = _mdn_params(raw)
        cdf_hi = _normal_cdf(hi.unsqueeze(-1), mu, sigma)
        cdf_lo = _normal_cdf(lo.unsqueeze(-1), mu, sigma)
        mass = (log_pi.exp() * (cdf_hi - cdf_lo)).sum(-1)
    else:
        raise ValueError(f"unknown head_type {head_type!r}")
    return torch.log(mass.clamp_min(1e-12))


def point(head_type, raw):
    if head_type == "deterministic":
        return raw.squeeze(-1) if raw.dim() > 2 else raw
    if head_type == "gaussian":
        return _gauss_params(raw)[0]
    if head_type == "mdn":
        log_pi, mu, _ = _mdn_params(raw)
        return (log_pi.exp() * mu).sum(-1)
    if head_type == "categorical":
        p = torch.softmax(raw, dim=-1)
        return (p * bin_centers(raw.device)).sum(-1)
    raise ValueError(f"unknown head_type {head_type!r}")


def sample(head_type, raw, generator=None):
    if head_type == "gaussian":
        mu, sigma = _gauss_params(raw)
        eps = torch.randn(mu.shape, generator=generator, device=mu.device)
        y = mu + sigma * eps
    elif head_type == "mdn":
        log_pi, mu, sigma = _mdn_params(raw)
        flat_pi = log_pi.exp().reshape(-1, MDN_K)
        k = torch.multinomial(flat_pi, 1, generator=generator).reshape(mu.shape[:-1])
        muk = mu.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        sigk = sigma.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        eps = torch.randn(muk.shape, generator=generator, device=muk.device)
        y = muk + sigk * eps
    elif head_type == "categorical":
        p = torch.softmax(raw, dim=-1)
        b = torch.multinomial(p.reshape(-1, N_BINS), 1, generator=generator).reshape(p.shape[:-1])
        u = torch.rand(b.shape, generator=generator, device=b.device)
        y = b.float() * BIN_WIDTH + u * BIN_WIDTH
    else:
        raise ValueError(f"unknown head_type {head_type!r}")
    return y.clamp(0, 127)
