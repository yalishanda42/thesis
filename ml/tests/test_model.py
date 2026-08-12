import torch

from drum_dynamics.models.model import VelocityTransformer
from drum_dynamics.data.seqdata import NUMERIC_FEATURES


def _tiny_model():
    torch.manual_seed(42)
    return VelocityTransformer(n_genres=5, d_model=32, n_heads=4, n_layers=2,
                               dim_ff=64, max_len=16).eval()


def test_forward_output_shape():
    m = _tiny_model()
    B, L = 2, 6
    voice = torch.randint(0, 14, (B, L))
    genre = torch.randint(0, 5, (B, L))
    num = torch.randn(B, L, len(NUMERIC_FEATURES))
    pad = torch.zeros(B, L, dtype=torch.bool)
    y = m(voice, genre, num, pad)
    assert y.shape == (B, L)


def test_padding_invariance():
    # Real-token predictions must not change when extra padding is appended.
    m = _tiny_model()
    L = 3
    voice = torch.randint(0, 14, (1, L))
    genre = torch.randint(0, 5, (1, L))
    num = torch.randn(1, L, len(NUMERIC_FEATURES))
    pad = torch.zeros(1, L, dtype=torch.bool)

    pad_extra = torch.cat([voice, torch.zeros(1, 2, dtype=torch.long)], dim=1)
    genre_extra = torch.cat([genre, torch.zeros(1, 2, dtype=torch.long)], dim=1)
    num_extra = torch.cat([num, torch.zeros(1, 2, len(NUMERIC_FEATURES))], dim=1)
    mask_extra = torch.tensor([[False, False, False, True, True]])

    with torch.no_grad():
        y1 = m(voice, genre, num, pad)
        y2 = m(pad_extra, genre_extra, num_extra, mask_extra)
    assert torch.allclose(y1[0, :L], y2[0, :L], atol=1e-4)


import os
import tempfile

from drum_dynamics.models.heads import head_output_dim, N_BINS, MDN_K
from drum_dynamics.models.model import warm_start_backbone


def _mk(head):
    return VelocityTransformer(n_genres=5, d_model=32, n_heads=4, n_layers=1, dim_ff=64,
                               max_len=16, head=head)


def test_head_output_shapes():
    B, L = 2, 5
    voice = torch.randint(0, 14, (B, L)); genre = torch.randint(0, 5, (B, L))
    num = torch.randn(B, L, len(NUMERIC_FEATURES)); pad = torch.zeros(B, L, dtype=torch.bool)
    assert _mk("deterministic")(voice, genre, num, pad).shape == (B, L)
    assert _mk("gaussian")(voice, genre, num, pad).shape == (B, L, 2)
    assert _mk("mdn")(voice, genre, num, pad).shape == (B, L, 3 * MDN_K)
    assert _mk("categorical")(voice, genre, num, pad).shape == (B, L, N_BINS)


def test_warm_start_loads_backbone_drops_head():
    torch.manual_seed(0)
    det = _mk("deterministic")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ck.pt")
        torch.save({"best_model": det.state_dict()}, p)
        gauss = _mk("gaussian")
        missing, unexpected = warm_start_backbone(gauss, p)
        assert torch.equal(gauss.input_proj.weight, det.input_proj.weight)
        assert torch.equal(gauss.voice_emb.weight, det.voice_emb.weight)
        assert all(k.startswith("head.") for k in missing)      # only head missing
        assert list(unexpected) == []
