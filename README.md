# Dynamic Controlled Brain-network Conditional Diffusion (DCBCD)

This repository provides a PyTorch implementation of the **Dynamic Controlled Brain-network Conditional Diffusion (DCBCD)** model proposed in:

> **Decoding Affective Neural Modulation by Musical Features: Insights from the Language of Music**

The DCBCD model is designed to mechanistically model **music-induced neural modulation** by simulating **controlled diffusion processes over functional brain networks**, conditioned on cross-modal interactions between music semantics and fNIRS-derived neural dynamics.

---

## Overview

Music-based stimulation (MS) induces dynamic and progressive reorganization of brain functional networks. Rather than directly predicting clinical outcomes, DCBCD formulates music-induced neural modulation as a **controllable graph-editing problem**, where:

* **Graphs** represent brain functional connectivity.
* **Diffusion processes** model gradual network reconfiguration.
* **Cross-modal attention** captures how music semantics modulate neural dynamics over time.
* **Pre-stimulation networks** act as anchor states constraining the diffusion trajectory.

This implementation follows the conceptual framework described in the manuscript while providing a clean, reproducible, and extensible engineering realization.

---

## Model Architecture

### Core Components

1. **Conditional Cross-Modal Module (`Conditional`)**

   * Implements *paper-consistent* cross-attention:

     * **Query**: fNIRS-derived neural representations
     * **Key / Value**: music semantic embeddings
   * Produces time-resolved conditional neural states.
   * Attention weights are used to quantify *modulation intensity*.

2. **Graph Diffusion Model (`GraphDiffusionModel`)**

   * Implements a DDPM-style diffusion process over adjacency matrices.
   * Learns to transform pre-stimulation brain networks toward post-stimulation states.
   * Conditioned on:

     * Node identity embeddings
     * Noisy adjacency values
     * Pre-stimulation anchor graph
     * Conditional neural states
     * Diffusion timestep embeddings

3. **Edge Noise Predictor (`DiffusionEdgePredictor`)**

   * Predicts diffusion noise for each graph edge.
   * Aligns diffusion steps with temporal indices of neural modulation, consistent with the manuscript.

---

## Paper-Consistent Design Choices

The following design decisions strictly follow the manuscript:

* Music semantics modulate neural dynamics via **cross-modal attention**
* Conditioning is **time-resolved**, aligned with diffusion steps
* Pre-stimulation graphs are used as **anchors**
* Attention weights are interpreted as **modulation intensity**

These components correspond directly to the DCBCD conceptual framework described in the paper.

---

## Engineering Extensions (Clearly Marked)

For stability and practical training, several **engineering extensions** are included. These do **not alter the conceptual claims** of the paper and can be removed or ablated if desired:

* **Approximate topological loss**

  * Uses a differentiable connectivity curve based on Laplacian eigenvalues
  * Encourages structurally consistent graph evolution
* **Attention entropy regularization**

  * Promotes sharper, interpretable cross-modal attention
* **Zero-condition branch**

  * Acts as a control/baseline pathway analogous to classifier-free guidance

These extensions are explicitly documented in the code to ensure transparency and reviewer clarity.

---

## Repository Structure

```
.
├── model_condition.py     # Cross-modal conditional attention module
├── model_diffusion.py     # Graph diffusion model and training logic
├── README.md              # This file
```

---

## Input / Output Specifications

### Inputs

* `g_pre`: `[B, N, N]`
  Pre-stimulation functional connectivity graphs
* `g_pst`: `[B, N, N]`
  Post-stimulation functional connectivity graphs
* `feat_fnirs`: `[B, N, T_f, D_fnirs]`
  Time-resolved fNIRS neural features
* `feat_music`: `[B, 1, T_m, D_music]`
  Music semantic embeddings (e.g., from MU-LLaMA)

### Outputs

* Total training loss
* Diffusion noise prediction loss
* Structural/topological regularization loss
* Cross-modal attention weights (for modulation analysis)

---

## Usage Example (Training Step)

```python
model = GraphDiffusionModel(
    T=1000,
    beta_start=1e-4,
    beta_end=0.02,
    node_feature_dim=48,
    node_emb_dim=64,
    time_emb_dim=64,
    hidden_dim=64,
)

total_loss, diff_loss, topo_loss, attn = model(
    g_pre, g_pst, feat_fnirs, feat_music
)
```

---

## Reproducibility Notes

* All modules include explicit shape checks.
* Cross-modal attention semantics are fixed and documented.
* Engineering extensions are clearly separated from core mechanisms.
* The code is written to be directly open-source and reviewer-auditable.

---

## Citation

If you use this code, please cite:

```bibtex
@article{zhu2024music,
  title={Decoding Affective Neural Modulation by Musical Features: Insights from the Language of Music},
  author={Zhu, Yue and Zhang, Haifeng and others},
  journal={},
  year={2024}
}
```

---

## Disclaimer

This repository provides a research implementation intended for scientific reproducibility and methodological transparency. Clinical conclusions should be drawn from the experimental results reported in the associated manuscript, not from the code alone.
