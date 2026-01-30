# Decoding Affective Neural Modulation by Musical Features (Core Model Code)

This repository contains the **core deep learning model implementation** of **DCBCD (Dynamic Controlled Brain-network Conditional Diffusion)** proposed in:

**“Decoding Affective Neural Modulation by Musical Features: Insights from the Language of Music”**
Yue Zhu#, Haifeng Zhang#, Yao Xiao, …, Chonghui Song*, Fei Wang*.

## What this repo is for

Music-based stimulation (MS) is widely used for affect regulation, yet its clinical deployment remains largely empirical. In our work, we treat **music as a structured, language-like temporal system** and model **music-induced brain network reconfiguration** as a **controllable graph-editing** problem.

This repo provides the **core modeling code** used to couple:

* **music semantic embeddings** (from a Music Understanding LLM, e.g., MU-LLaMA),
* **temporally resolved fNIRS-derived neural dynamics** (time-series encoding, e.g., Tiny Time Mixers / TTM),
* and **brain functional connectivity (FC) graphs** (Yeo 7-network based),

to simulate the **transition from pre-stimulation brain networks to post-stimulation networks** via a **conditional diffusion process**, and to quantify **time-varying modulation intensity** via **cross-modal attention weights**.

In short: **given music (as “language”) + neural time series**, DCBCD learns **how brain networks are progressively reorganized under stimulation**, and supports **interpretable temporal segmentation** into **high- vs low-modulation music segments (“effective music”)**.

---

## Key ideas implemented

### 1) Brain network transition as conditional diffusion (DCBCD)

We represent functional brain connectivity as graphs and learn a **progressive editing trajectory** from:

* **Pre-stimulation FC**  →  **During-stimulation dynamic FC**  →  **Post-stimulation FC**

The diffusion framework provides an iterative, gradual transformation mechanism that matches how stimulation effects accumulate over time.

### 2) Music–brain coupling via cross-modal attention

We explicitly model how **high-level music semantics** regulate neural responses using a **cross-modal cross-attention** module:

* Query/Key/Value are constructed across **music embeddings** and **neural time representations**
* The resulting attention map gives both:

  * a **control signal** for network editing, and
  * a **time-resolved importance score** (average attention weights) used for modulation profiling.

### 3) Model-informed temporal segmentation of “effective music”

We compute a modulation intensity time series from attention weights and segment the session into:

* **High-modulation segments** (median + 1.645 × MAD threshold)
* **Low-modulation segments**

To reduce confounds from session onset transient dynamics, we exclude early transient phases via a **data-driven HMM segmentation** on a global connectivity trajectory (e.g., Fiedler-based).

This supports analyses reported in the paper, including:

* high-modulation segments inducing **hierarchical hypoconnectivity patterns**
* and their association with long-term depression improvement.

---

## What’s included (core modules)

* `dc_bcd/`

  * `diffusion.py` — conditional diffusion backbone for FC graph editing
  * `conditioner.py` — music–brain conditional module (cross-modal attention)
  * `graph_ops.py` — graph construction & edit operators (FC adjacency processing)
  * `encoders/` — neural time-series encoder (e.g., TTM) + music embedding adapter
* `configs/` — reproducible experiment configs (training/eval/inference)
* `scripts/` — training / evaluation / inference entry points
* `utils/` — logging, metrics, checkpointing, seed control

> Note: this repo focuses on **model code**. Clinical data (fNIRS/PHQ-9) may require controlled access and is not redistributed here.

---

## Minimal usage

### Training (example)

1. Prepare:

   * pre-stimulation FC graphs
   * post-1st-stimulation FC graphs (anchor)
   * during-stimulation neural time-series features
   * aligned music embeddings per time window

2. Train:

```bash
python scripts/train.py --config configs/dcbcd.yaml
```

### Inference (estimate modulation intensity and segment effective music)

```bash
python scripts/infer.py --ckpt <path_to_ckpt> --out <out_dir>
```

Outputs typically include:

* predicted post-stimulation FC graph
* time-wise modulation curve (attention-derived importance score)
* high/low-modulation time masks

---

## Reproducibility & reporting

We emphasize **mechanism-oriented modeling** rather than directly predicting clinical scores. Reported outcomes are typically framed as:

* similarity between predicted and observed post-stimulation brain networks,
* pathway-level FC changes (e.g., unimodal → transmodal hierarchy),
* interpretability via cross-modal attention and time segmentation,
* downstream association between modulation-sensitive ΔFC and symptom improvement (e.g., ΔPHQ-9).

Random seeds and config snapshots are logged for deterministic reruns when possible.

---

## Data & ethics

* This repository **does not ship raw participant data**.
* fNIRS and clinical measures were collected under IRB approval and informed consent (see paper for registry and ethics).
* If you are an academic collaborator and require data access, please follow the contact route described in the manuscript.

---

## Citation

---

## Why this repo may be useful to the community

Most music-therapy ML pipelines stop at:

* handcrafted acoustic features, or
* end-to-end clinical score prediction.

DCBCD provides a third path:

* **explicit network transition modeling**
* **temporally localized “effective music” discovery**
* and **interpretable music–brain coupling** via cross-modal attention.

This can support both:

* mechanistic neuroscience questions, and
* mechanism-driven stimulus design (including AI-guided music generation prompts).

