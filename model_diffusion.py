# model_diffusion.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Paper-consistent conditional module (music -> modulates fNIRS via cross-attention)
from model_condition import Conditional


class ApproxTopologicalLoss(nn.Module):
    """
    Engineering extension: approximate topology-sensitive loss.

    This implements a differentiable proxy of graph connectivity evolution across
    a sweep of distance thresholds (a "connectivity curve"), using the 2nd smallest
    eigenvalue (Fiedler value) of a normalized Laplacian as a connectivity indicator.

    NOTE:
    - This is not strictly required by the paper's core DCBCD concept, but can be kept
      as an engineering extension if you want stronger structural constraints.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        thresholds: int = 20,
        temperature: float = 10.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.eps = float(eps)

        # Thresholds for the connectivity curve in "distance" space.
        self.register_buffer("thresholds", torch.linspace(0.0, 2.0, int(thresholds)))

    def _compute_connectivity_curve(self, dist_matrix: Tensor) -> Tensor:
        """
        Args:
            dist_matrix: [B, N, N] distance-like matrix

        Returns:
            curve: [B, K] where K = number of thresholds
        """
        if dist_matrix.ndim != 3:
            raise ValueError(f"dist_matrix must be [B, N, N], got {tuple(dist_matrix.shape)}")

        B, N, _ = dist_matrix.shape
        curves: List[Tensor] = []

        eyeN = torch.eye(N, device=dist_matrix.device, dtype=dist_matrix.dtype).unsqueeze(0)

        for eps_thr in self.thresholds:
            # Soft adjacency from distances (higher similarity for smaller distances).
            adj = torch.sigmoid((eps_thr - dist_matrix) * self.temperature)

            # Enforce symmetry (undirected graph).
            adj = torch.maximum(adj, adj.transpose(1, 2))

            # Normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
            deg = adj.sum(dim=-1)  # [B, N]
            deg_inv_sqrt = deg.clamp(min=self.eps).pow(-0.5)  # [B, N]
            D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)  # [B, N, N]

            lap = eyeN - D_inv_sqrt @ adj @ D_inv_sqrt
            lap = lap + eyeN * self.eps  # numerical stability

            # Eigenvalues in ascending order; second smallest approximates global connectivity.
            eigvals = torch.linalg.eigvalsh(lap)  # [B, N]
            second_smallest = eigvals[:, 1]  # assumes at least weak connectivity
            curves.append(second_smallest)

        return torch.stack(curves, dim=1)  # [B, K]

    def forward(self, pred_corr: Tensor, true_corr: Tensor) -> Tensor:
        """
        Args:
            pred_corr: [B, N, N] predicted correlation/adjacency-like matrix
            true_corr: [B, N, N] target matrix

        Returns:
            scalar loss
        """
        if pred_corr.ndim == 2:
            pred_corr = pred_corr.unsqueeze(0)
        if true_corr.ndim == 2:
            true_corr = true_corr.unsqueeze(0)

        if pred_corr.shape != true_corr.shape:
            raise ValueError(
                f"Shape mismatch: pred_corr={tuple(pred_corr.shape)}, true_corr={tuple(true_corr.shape)}"
            )

        # Convert correlation-like values to a distance proxy.
        pred_dist = torch.sqrt(1.0 - torch.clamp(pred_corr, -1.0 + self.eps, 1.0 - self.eps))
        true_dist = torch.sqrt(1.0 - torch.clamp(true_corr, -1.0 + self.eps, 1.0 - self.eps))

        pred_curve = self._compute_connectivity_curve(pred_dist)
        true_curve = self._compute_connectivity_curve(true_dist)

        topo_loss = F.l1_loss(pred_curve, true_curve)
        mse_loss = F.mse_loss(pred_corr, true_corr)

        return (1.0 - self.alpha) * mse_loss + self.alpha * topo_loss


class TimeEmbedding(nn.Module):
    """
    Standard sinusoidal timestep embedding, used to condition the diffusion predictor on t.

    Output:
        emb: [B, D]
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        half_dim = embedding_dim // 2
        if half_dim == 0:
            raise ValueError("embedding_dim is too small for sinusoidal embedding")

        inv_freq = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        self.register_buffer("inv_freq", inv_freq)
        self.embedding_dim = int(embedding_dim)

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: [B] or [B, 1] integer diffusion steps

        Returns:
            emb: [B, D]
        """
        t = t.view(-1).float().unsqueeze(-1)  # [B, 1]
        freqs = t * self.inv_freq  # [B, half_dim]
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # [B, 2*half_dim]

        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))  # make it [B, D] when D is odd

        return emb


class DiffusionEdgePredictor(nn.Module):
    """
    Predict diffusion noise for each edge in the adjacency matrix.

    Paper-consistent structure:
    - A conditional module produces time-resolved neural modulation states driven by music.
    - We align diffusion steps t with temporal indexing into the conditional sequence.
    - We condition edge prediction on:
        (i) node identity embeddings
        (ii) current noisy adjacency value
        (iii) pre-stimulation anchor graph value (g_pre)
        (iv) conditional state at time t (from Conditional module)
        (v) time embedding of diffusion step
    """

    def __init__(
        self,
        node_feature_dim: int,
        node_emb_dim: int,
        time_emb_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.node_proj = nn.Linear(node_feature_dim, node_emb_dim)
        self.time_embed = TimeEmbedding(time_emb_dim)

        # Conditional fuser: music semantics -> modulates fNIRS via cross-attention (paper-consistent).
        # We set hidden_dim=hidden_dim to match the downstream concatenation dimensionality.
        self.condition_fuser = Conditional(hidden_dim=hidden_dim)

        # Edge-level MLP over pairwise node features + conditioning.
        in_dim = 2 * node_emb_dim + 1 + 1 + hidden_dim + time_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        noisy_adj: Tensor,
        node_feats: Tensor,
        feat_fnirs: Tensor,
        feat_music: Tensor,
        t: Tensor,
        g_pre: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            noisy_adj:  [B, N, N]     noisy adjacency at diffusion step t
            node_feats: [B, N, F]     node features (e.g., one-hot node ids)
            feat_fnirs: [B, N, T_f, D_fnirs]   fNIRS time series features
            feat_music: [B, 1, T_m, D_music]   music time series features
            t:          [B, 1] or [B] diffusion step indices (Long)
            g_pre:      [B, N, N] pre-stimulation anchor adjacency

        Returns:
            noise_pred:      [B, N, N]
            attn_weights:    [B, T_f, T_m]  (mean across channels)
            attention_loss:  scalar Tensor
        """
        if noisy_adj.ndim != 3:
            raise ValueError(f"noisy_adj must be [B, N, N], got {tuple(noisy_adj.shape)}")
        if node_feats.ndim != 3:
            raise ValueError(f"node_feats must be [B, N, F], got {tuple(node_feats.shape)}")
        if g_pre.ndim != 3:
            raise ValueError(f"g_pre must be [B, N, N], got {tuple(g_pre.shape)}")

        B, N, _ = noisy_adj.shape

        # Project node features.
        h = self.node_proj(node_feats)  # [B, N, node_emb_dim]

        # Build pairwise representations for edges (i, j).
        h_i = h.unsqueeze(2).expand(-1, N, N, -1)  # [B, N, N, node_emb_dim]
        h_j = h.unsqueeze(1).expand(-1, N, N, -1)  # [B, N, N, node_emb_dim]

        # Scalar edge inputs.
        adj_val = noisy_adj.unsqueeze(-1)  # [B, N, N, 1]
        gpre_val = g_pre.unsqueeze(-1)     # [B, N, N, 1]

        # Conditional fusion (paper-consistent): music conditions fNIRS via cross-attention.
        # condition_fused: [B, N, T_f, H]
        condition_fused, attn_weights, attention_loss = self.condition_fuser(feat_fnirs, feat_music)

        # Align diffusion step with temporal index into condition_fused.
        # We assume diffusion steps are aligned with the conditional timeline as described in the paper.
        # Index tensor must be Long.
        if t.ndim == 2:
            t_idx = t.view(B)  # [B]
        else:
            t_idx = t.view(B)

        t_idx = t_idx.long().clamp(min=0, max=condition_fused.size(2) - 1)  # safe indexing

        # Gather conditional state at time t for each channel.
        # condition_fused: [B, N, T_f, H]
        # index:           [B, N, 1, H] with t broadcast
        idx = t_idx.view(B, 1, 1, 1).expand(-1, N, 1, condition_fused.size(-1))
        cond_t = torch.gather(condition_fused, dim=2, index=idx).squeeze(2)  # [B, N, H]
        cond_t = cond_t.unsqueeze(1).expand(-1, N, N, -1)                    # [B, N, N, H]

        # Time embedding at diffusion step t.
        time_emb = self.time_embed(t_idx)  # [B, time_emb_dim]
        time_emb = time_emb.unsqueeze(1).unsqueeze(1).expand(-1, N, N, -1)  # [B, N, N, time_emb_dim]

        # Concatenate all edge inputs.
        x = torch.cat([h_i, h_j, adj_val, gpre_val, cond_t, time_emb], dim=-1)  # [B, N, N, in_dim]

        # Predict noise for each edge.
        noise_pred = self.mlp(x).squeeze(-1)  # [B, N, N]

        return noise_pred, attn_weights, attention_loss


class GraphDiffusionModel(nn.Module):
    """
    Dynamic Controlled Brain-network Conditional Diffusion (DCBCD) model.

    Core paper-consistent components:
    - Diffusion process over graph adjacency matrices.
    - Conditional control via cross-modal attention between music semantics and fNIRS dynamics.
    - Anchor constraint via pre-stimulation graph (g_pre).
    - Training objective: diffusion noise prediction + structural/topological regularization.

    Notes:
    - This implementation keeps your original engineering extensions (topological loss, zero-condition path),
      but uses clearer structure and paper-aligned documentation.
    """

    def __init__(
        self,
        T: int,
        beta_start: float,
        beta_end: float,
        node_feature_dim: int,
        node_emb_dim: int,
        time_emb_dim: int,
        hidden_dim: int,
        use_zero_condition: bool = True,
    ) -> None:
        super().__init__()

        if T <= 1:
            raise ValueError("T must be > 1 for diffusion")
        self.T = int(T)
        self.use_zero_condition = bool(use_zero_condition)

        # Linear beta schedule.
        betas = torch.linspace(float(beta_start), float(beta_end), self.T)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)

        # Edge noise predictor.
        self.edge_predictor = DiffusionEdgePredictor(
            node_feature_dim=node_feature_dim,
            node_emb_dim=node_emb_dim,
            time_emb_dim=time_emb_dim,
            hidden_dim=hidden_dim,
        )

        # Losses.
        self.topo_loss_fn = ApproxTopologicalLoss()
        self.mse = nn.MSELoss()

        # Default node identity features for N=48 channels (paper setting).
        # If your N differs, consider making this configurable.
        node_ids = torch.arange(48)
        self.register_buffer("node_features", F.one_hot(node_ids, num_classes=48).float())

    def forward(
        self,
        g_pre: Tensor,
        g_pst: Tensor,
        feat_fnirs: Tensor,
        feat_music: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Training forward pass.

        Args:
            g_pre:      [B, N, N] pre-stimulation graph (anchor / initial state)
            g_pst:      [B, N, N] post-stimulation graph (target state)
            feat_fnirs: [B, N, T_f, D_fnirs]
            feat_music: [B, 1, T_m, D_music]

        Returns:
            total_loss: scalar
            diff_loss:  scalar (noise prediction)
            topo_loss:  scalar (structural regularization)
            attn_f2m:   [B, T_f, T_m] (attention weights summary)
        """
        if g_pre.ndim != 3 or g_pst.ndim != 3:
            raise ValueError("g_pre and g_pst must be [B, N, N] tensors")

        device = g_pre.device
        B, N, _ = g_pre.shape

        # Node identity features (one-hot).
        # NOTE: By default this is built for N=48. If your N != 48, adapt accordingly.
        node_features = self.node_features.unsqueeze(0).repeat(B, 1, 1).to(device)

        x0 = g_pre  # diffusion starts from pre-stimulation graph as the anchor state

        # Sample a diffusion step t uniformly for each batch element.
        t = torch.randint(0, self.T, (B, 1), device=device).long()
        t_idx = t.view(B)  # [B]

        alpha_bar = self.alphas_bar[t_idx].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        sqrt_ab = alpha_bar.sqrt()
        sqrt_1_ab = (1.0 - alpha_bar).sqrt()

        # Forward diffusion: x_t = sqrt(ab)*x0 + sqrt(1-ab)*noise
        noise = torch.randn_like(x0)
        noisy_adj = sqrt_ab * x0 + sqrt_1_ab * noise

        # Predict noise with conditional control.
        noise_pred, attn_f2m, attention_loss = self.edge_predictor(
            noisy_adj=noisy_adj,
            node_feats=node_features,
            feat_fnirs=feat_fnirs,
            feat_music=feat_music,
            t=t,
            g_pre=g_pre,
        )
        diff_loss = self.mse(noise_pred, noise)

        # Estimate x0 from predicted noise.
        x0_pred = (noisy_adj - sqrt_1_ab * noise_pred) / sqrt_ab

        # Topology/structure loss relative to post-stimulation target.
        topo_loss_c = self.topo_loss_fn(x0_pred, g_pst)

        # Optional zero-condition branch (engineering extension; conceptually akin to a control/baseline).
        if self.use_zero_condition:
            def zeros_like(x: Tensor) -> Tensor:
                return torch.zeros_like(x)

            noise_pred_0c, _, _ = self.edge_predictor(
                noisy_adj=noisy_adj,
                node_feats=node_features,
                feat_fnirs=zeros_like(feat_fnirs),
                feat_music=zeros_like(feat_music),
                t=t,
                g_pre=g_pre,
            )
            x0_pred_0c = (noisy_adj - sqrt_1_ab * noise_pred_0c) / sqrt_ab
            topo_loss_0c = self.topo_loss_fn(x0_pred_0c, g_pre)

            # Keep your original weighting.
            topo_loss = 0.75 * topo_loss_c + 0.25 * topo_loss_0c
        else:
            topo_loss = topo_loss_c

        # Total objective (keep your original sign/weighting).
        total_loss = diff_loss + topo_loss - 0.01 * attention_loss

        return total_loss, diff_loss, topo_loss, attn_f2m

    @torch.no_grad()
    def sample_trajectory(
        self,
        g_0: Tensor,
        feat_fnirs: Tensor,
        feat_music: Tensor,
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Generate a graph evolution trajectory under the conditional diffusion model.

        Args:
            g_0:       [B, N, N] initial graph (anchor)
            feat_fnirs: [B, N, T_f, D_fnirs]
            feat_music: [B, 1, T_m, D_music]
            num_steps: number of diffusion steps to simulate (default: self.T)

        Returns:
            trajectory: [B, num_steps, N, N]
        """
        if num_steps is None:
            num_steps = self.T
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")

        device = g_0.device
        B, N, _ = g_0.shape

        # Node identity features.
        node_features = self.node_features.unsqueeze(0).repeat(B, 1, 1).to(device)

        trajectory: List[Tensor] = [g_0]
        x_t = g_0

        # NOTE: This function preserves your original sampling logic.
        # It is a simple trajectory reconstruction based on predicting noise at each step.
        for step in range(1, num_steps):
            t_tensor = torch.full((B, 1), step, device=device, dtype=torch.long)

            alpha_bar = self.alphas_bar[step].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, 1, 1]
            sqrt_ab = alpha_bar.sqrt()
            sqrt_1_ab = (1.0 - alpha_bar).sqrt()

            noise = torch.randn_like(x_t)

            # Preserve original behavior: perturb g_0 (anchor) rather than x_t.
            noisy_adj = sqrt_ab * g_0 + sqrt_1_ab * noise

            noise_pred, _, _ = self.edge_predictor(
                noisy_adj=noisy_adj,
                node_feats=node_features,
                feat_fnirs=feat_fnirs,
                feat_music=feat_music,
                t=t_tensor,
                g_pre=g_0,
            )

            x_t = (noisy_adj - sqrt_1_ab * noise_pred) / sqrt_ab
            trajectory.append(x_t)

        return torch.stack(trajectory, dim=1)  # [B, num_steps, N, N]
