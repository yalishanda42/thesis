"""Non-autoregressive event-sequence Transformer for velocity (design §5/§6)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .heads import head_output_dim, init_mdn_head
from ..data.seqdata import NUMERIC_FEATURES
from ..core.voicemap import CANONICAL_VOICES


def _sinusoidal_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe                                   # [max_len, d_model]


class VelocityTransformer(nn.Module):
    def __init__(self, n_genres, n_numeric=len(NUMERIC_FEATURES), n_voices=len(CANONICAL_VOICES),
                 d_model=128, n_heads=8, n_layers=4, dim_ff=256, dropout=0.1,
                 voice_emb=8, genre_emb=16, max_len=512, head="deterministic"):
        super().__init__()
        self.head_type = head
        self.voice_emb = nn.Embedding(n_voices, voice_emb)
        self.genre_emb = nn.Embedding(n_genres, genre_emb)
        self.input_proj = nn.Linear(voice_emb + genre_emb + n_numeric, d_model)
        self.register_buffer("pos_enc", _sinusoidal_pos_enc(max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        # enable_nested_tensor=False: the nested-tensor fast path calls
        # aten::_nested_tensor_from_mask_left_aligned, which is unimplemented on
        # MPS. It is only a padding-skip optimization; disabling it keeps outputs
        # identical and works on every device.
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Linear(d_model, head_output_dim(head))
        if head == "mdn":
            init_mdn_head(self.head)          # spread component means to avoid collapse

    def forward(self, voice_idx, genre_idx, num_feats, pad_mask):
        v = self.voice_emb(voice_idx)                     # [B, L, voice_emb]
        g = self.genre_emb(genre_idx)                     # [B, L, genre_emb]
        x = torch.cat([v, g, num_feats], dim=-1)          # [B, L, in]
        x = self.input_proj(x)                            # [B, L, d]
        x = x + self.pos_enc[: x.size(1)].unsqueeze(0)    # add positional encoding
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        out = self.head(x)                                # [B, L, out_dim]
        if self.head_type == "deterministic":
            return out.squeeze(-1)                        # [B, L] — Plan B behavior
        return out                                        # [B, L, out_dim]


def warm_start_backbone(model, ckpt_path):
    """Load Plan-B backbone weights (dropping the head) into ``model``.

    Returns ``(missing, unexpected)`` from ``load_state_dict(strict=False)``;
    ``missing`` should be exactly the new head's parameters.
    """
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck.get("best_model", ck)
    backbone = {k: v for k, v in state.items() if not k.startswith("head.")}
    return model.load_state_dict(backbone, strict=False)
