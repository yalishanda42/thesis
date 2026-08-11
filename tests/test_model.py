import torch

from drumhumanizer.model import VelocityTransformer
from drumhumanizer.seqdata import NUMERIC_FEATURES


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
