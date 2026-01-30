# model_condition.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import LayerNorm, MultiheadAttention


class Conditional(nn.Module):
    """
    Cross-modal conditional module (paper-consistent):
    - Music semantic embeddings act as conditioning signals.
    - Time-resolved fNIRS representations are modulated via cross-attention.

    Inputs:
        f_fnirs: [B, N, T_f, D_fnirs]   (e.g., N=48 channels)
        f_music: [B, 1, T_m, D_music]

    Returns:
        x_condition:     [B, N, T_f, H]     (time-resolved conditional states)
        attn_mean:       [B, T_f, T_m]      (mean attention weights across channels)
        attention_loss:  scalar Tensor      (entropy-based regularization; engineering extension)
    """

    def __init__(
        self,
        music_feat_dim: int = 1024,
        fnirs_feat_dim: int = 1024,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        # Project both modalities into the same hidden space (with LayerNorm for stability).
        self.pj_music = nn.Sequential(
            nn.Linear(music_feat_dim, hidden_dim),
            LayerNorm(hidden_dim),
        )
        self.pj_fnirs = nn.Sequential(
            nn.Linear(fnirs_feat_dim, hidden_dim),
            LayerNorm(hidden_dim),
        )

        # Cross-attention:
        # Paper-consistent semantics:
        #   Query   = fNIRS (neural states to be modulated)
        #   Key/Val = Music (conditioning signals)
        self.cross_attn = MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout_rate)

        # NOTE: You had a LayerNorm here but commented it out in forward.
        # We keep the module minimal and do not apply it unless you explicitly want it.
        self.norm_out = LayerNorm(hidden_dim)

    @staticmethod
    def attention_entropy(attn: Tensor, eps: float = 1e-6) -> Tensor:
        """
        Compute mean entropy of attention distributions.

        Args:
            attn:
                Either [B, N, T_q, T_k] or [B*N, T_q, T_k].
                Values are expected to be probabilities (as returned by PyTorch attention).
            eps: numerical stability.

        Returns:
            Scalar entropy (higher -> more uniform attention).
        """
        if attn.ndim == 3:
            # [B*N, T_q, T_k] -> treat as a batch
            p = attn.clamp(min=eps)
            entropy = -(p * p.log()).sum(dim=-1).mean()
            return entropy

        if attn.ndim == 4:
            # [B, N, T_q, T_k]
            p = attn.clamp(min=eps)
            entropy = -(p * p.log()).sum(dim=-1).mean()
            return entropy

        raise ValueError(f"Unsupported attention shape: {tuple(attn.shape)}")

    def forward(self, f_fnirs: Tensor, f_music: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Args:
            f_fnirs: [B, N, T_f, D_fnirs]
            f_music: [B, 1, T_m, D_music]

        Returns:
            x_condition: [B, N, T_f, H]
            attn_mean:   [B, T_f, T_m]
            attention_loss: scalar
        """
        if f_fnirs.ndim != 4 or f_music.ndim != 4:
            raise ValueError(
                "Expected f_fnirs and f_music to be 4D tensors: "
                f"got f_fnirs={tuple(f_fnirs.shape)}, f_music={tuple(f_music.shape)}"
            )

        B, N, T_f, D_f = f_fnirs.shape
        Bm, Nm, T_m, D_m = f_music.shape
        if Bm != B or Nm != 1:
            raise ValueError(
                "Expected f_music shape [B, 1, T_m, D_music]. "
                f"Got {tuple(f_music.shape)} while f_fnirs batch is B={B}."
            )

        # Expand music embeddings along channel dimension so each channel can attend to the same music sequence.
        # [B, 1, T_m, D_m] -> [B, N, T_m, D_m]
        f_music = f_music.expand(B, N, T_m, D_m)

        # Flatten channels into batch for efficient attention computation:
        # f_fnirs_flat: [B*N, T_f, D_f]
        # f_music_flat: [B*N, T_m, D_m]
        f_fnirs_flat = f_fnirs.reshape(B * N, T_f, D_f)
        f_music_flat = f_music.reshape(B * N, T_m, D_m)

        # Project to shared hidden space.
        q_fnirs = self.pj_fnirs(f_fnirs_flat)   # [B*N, T_f, H]
        kv_music = self.pj_music(f_music_flat)  # [B*N, T_m, H]

        # Cross-attention (paper-consistent):
        #   Query   = fNIRS (to be modulated)
        #   Key/Val = Music (conditioning)
        x_attn, attn_weights = self.cross_attn(
            query=q_fnirs,
            key=kv_music,
            value=kv_music,
            need_weights=True,
        )
        # x_attn:       [B*N, T_f, H]
        # attn_weights: [B*N, T_f, T_m]  (already averaged over heads by PyTorch by default)

        # Dropout as in your original implementation.
        x_out = self.dropout(x_attn)

        # Reshape back to [B, N, T_f, H].
        x_condition = x_out.view(B, N, T_f, self.hidden_dim)

        # You explicitly said the time flip is correct and should be kept.
        x_condition = torch.flip(x_condition, dims=[2])

        # Attention summary for modulation intensity analysis:
        # mean over channels -> [B, T_f, T_m]
        attn_weights_4d = attn_weights.view(B, N, T_f, T_m)
        attn_mean = attn_weights_4d.mean(dim=1)

        # Engineering extension retained:
        # Encourage "sharper" (lower entropy) attention by maximizing negative entropy,
        # i.e., attention_loss = -Entropy.
        attention_loss = -1.0 * self.attention_entropy(attn_weights_4d)

        return x_condition, attn_mean, attention_loss
