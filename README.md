# Awesome-Learnable-Q

<div align="center">

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![Learnable Query](https://img.shields.io/badge/Topic-Learnable%20Query-8A2BE2.svg)](#)
[![License](https://img.shields.io/badge/License-CC0_1.0-blue.svg)](LICENSE.txt)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**🔎 A Curated List of Awesome Works on Learnable Queries and Query-based Models across AI.**  
*From object queries and mask queries to track queries, BEV queries, intention queries, label queries, multimodal query tokens, and domain-specific scientific queries.*

<p align="center">
  <img src="https://github.com/lingxitong/Awesome-Learnable-Q/blob/main/Logo.png" alt="Awesome Learnable Q" width="100%" style="border-radius: 15px; box-shadow: 0 4px 24px rgba(0,0,0,.1); margin: 5px 0;">
</p>

</div>

---

## 🚩 News & Updates

_Major updates and repository announcements are shown below._

🚧 **[Ongoing] Cross-domain Query Map** — This repository is being built as a one-stop resource for models that use **explicit learnable / dynamic / semantic / geometric queries** as a core modeling component.

💡 **[Ongoing] Contributions Welcome** — Missing an important query-based paper? Feel free to open a PR.

📌 **[Scope] Beyond DETR** — We include not only DETR-family object queries, but also mask, track, BEV, motion, label, instance, latent, multimodal, audio, medical, and action queries.

---

## Overview

- 🎯 [Aim of the Project](#aim-of-the-project)
- 🧠 [What Counts as a Query-based Model?](#what-counts-as-a-query-based-model)
- 🧭 [Query Taxonomy](#query-taxonomy)
- 🧱 [Foundational Query Architectures](#foundational-query-architectures)
- 📦 [Object Detection](#object-detection)
- 🎨 [Image Segmentation](#image-segmentation)
- 🎞️ [Object Tracking](#object-tracking)
- ⏱️ [Video Understanding and Temporal Grounding](#video-understanding-and-temporal-grounding)
- 🚗 [3D Perception, BEV, and Motion Prediction](#3d-perception-bev-and-motion-prediction)
- 🧍 [Human-Object Interaction and Pose](#human-object-interaction-and-pose)
- 🏷️ [Classification and Label Queries](#classification-and-label-queries)
- 💬 [Vision-Language and Multimodal Models](#vision-language-and-multimodal-models)
- 🔤 [Natural Language Processing and Information Extraction](#natural-language-processing-and-information-extraction)
- 🔊 [Audio and Speech](#audio-and-speech)
- 🩺 [Medical Imaging and Computational Pathology](#medical-imaging-and-computational-pathology)
- 🤖 [Robotics, Embodied AI, and Action Prediction](#robotics-embodied-ai-and-action-prediction)
- ⚡ [Query Mechanisms, Efficiency, and Analysis](#query-mechanisms-efficiency-and-analysis)
- 🔭 [Future Trends and Hot Topics](#future-trends-and-hot-topics)
- 🤝 [Contributing](#contributing)
- 🙏 [Acknowledgements](#acknowledgements)
- 📝 [Citation](#citation)

---

## Aim of the Project

The **query** has become one of the most reusable architectural primitives in modern deep learning.

Starting from the fixed learnable **object queries** in DETR, query-based modeling has rapidly expanded to segmentation, tracking, 3D perception, autonomous driving, temporal grounding, motion prediction, multimodal learning, information extraction, audio analysis, medical imaging, computational pathology, and robotics.

This repository aims to:

- 🔍 **Organize** representative query-based papers across different AI communities.
- 🗺️ **Map** how query design evolves from static learnable embeddings to dynamic, geometric, semantic, temporal, and multimodal queries.
- 🧩 **Connect** seemingly different architectures through a common query-centric view.
- 📚 **Provide** a compact reading list for researchers entering query-based modeling.
- 💡 **Inspire** query designs that can transfer from one domain to another.
- 🔬 **Track** papers studying query initialization, supervision, propagation, specialization, sparsity, semantics, and interpretability.

---

## What Counts as a Query-based Model?

This repository uses **query** in a broader architectural sense than the ordinary `Q` matrix in standard self-attention.

A model is considered query-based when it contains an explicit compact representation that actively **retrieves, binds, aggregates, decodes, or propagates information** from a larger feature bank.

A canonical pattern is:

```text
Learnable / Dynamic Queries Q
          │
          ▼
 Cross-Attention / Matching
          │
          ├──────── Image / Video / Text / Point Cloud / WSI / Audio / ...
          ▼
      Updated Queries
          │
          ▼
 Box / Mask / Track / Label / Trajectory / Concept / Action / Answer
```

### ✅ Included

- Learnable object / mask / track / label / task / action queries.
- Reference-point, anchor-box, or geometry-aware queries.
- Queries propagated between frames or decoder layers.
- Query tokens used as a bottleneck between modalities.
- Semantic, text, class, anatomical, pathological, or intention queries.
- Papers explicitly studying query initialization, selection, matching, routing, adaptation, or utilization.

### ❌ Usually Not Included

- Standard Transformer self-attention merely because it internally computes Q/K/V.
- A vanilla `[CLS]` token used only for global pooling.
- Prompt tokens that do not explicitly query or bind to another feature set.
- Attention pooling methods with no explicit query-centric formulation.

---

## Query Taxonomy

| Query Type | Typical Form | Main Role | Representative Works |
|---|---|---|---|
| **Free Learnable Query** | `nn.Embedding(Nq, d)` | Discover task-specific entities | DETR, MaskFormer |
| **Geometric Query** | point / box / reference coordinate | Inject spatial prior | DAB-DETR, DETR3D |
| **Dynamic Query** | sample-conditioned query | Adapt query bank to input | FastInst, PaQ-DETR |
| **Temporal / Persistent Query** | previous-frame query state | Carry identity or memory | TrackFormer, MOTR |
| **Semantic / Label Query** | label/text embedding | Retrieve class-specific evidence | Query2Label, OneFormer |
| **Latent Bottleneck Query** | small latent token bank | Compress large inputs | Perceiver, Flamingo |
| **Output Query** | output-specific query | Decode structured outputs | Perceiver IO |
| **Intention Query** | trajectory-mode query | Represent future behavior modes | MTR, MTR++ |
| **Domain Query** | anatomy / pathology / annotator query | Encode scientific or clinical prior | TAB, QuCCeS |

---


## Curation Policy

<em>The main list prioritizes peer-reviewed papers accepted by major conferences and journals. Preprints and workshop papers are retained only when they introduce a particularly query-centric idea and are explicitly labeled.</em>

- ⭐ **Primary venues** — CVPR, ICCV, ECCV, NeurIPS, ICLR, ICML, AAAI, ACL, EMNLP, IJCAI, MICCAI, CoRL, RSS, ICRA, and major IEEE/ACM journals.
- 🧪 **Preprints / Workshops** — included selectively and clearly marked as **Preprint** or **Workshop**.
- 🔎 **Query relevance** — an explicit query representation must participate in information retrieval, binding, aggregation, decoding, matching, temporal propagation, or task conditioning.
- 📝 **Entry description** — every item should explain what the query represents rather than merely stating that the model uses a Transformer.

---

# Papers

## Foundational Query Architectures

<em>General architectures that established learned seeds, slots, latent queries, or output queries as reusable neural representations.</em>

- **Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks** — introduces learnable seed vectors in Pooling by Multihead Attention (PMA), an early query-like set aggregation mechanism. [![Paper](https://img.shields.io/badge/Paper-ICML%202019-1f77b4.svg)](https://proceedings.mlr.press/v97/lee19d.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/juho-lee/set_transformer)

- **Object-Centric Learning with Slot Attention** — iteratively updates a fixed set of learned slots that compete to bind to objects, closely related to iterative learnable-query reasoning. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202020-1f77b4.svg)](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/google-deepmind/deepmind-research/tree/master/slot_attention)

- **Perceiver: General Perception with Iterative Attention** — uses a compact learnable latent array as queries to repeatedly cross-attend to arbitrarily large inputs. [![Paper](https://img.shields.io/badge/Paper-ICML%202021-1f77b4.svg)](https://proceedings.mlr.press/v139/jaegle21a.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver)

- **Perceiver IO: A General Architecture for Structured Inputs & Outputs** — extends Perceiver with explicit output queries, allowing the same architecture to decode arbitrary structured outputs. [![Paper](https://img.shields.io/badge/Paper-ICLR%202022-1f77b4.svg)](https://arxiv.org/abs/2107.14795) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver)

- **DETR: End-to-End Object Detection with Transformers** — popularizes a fixed bank of learnable object queries as task-level prediction slots and makes set prediction the canonical query-decoding paradigm. [![Paper](https://img.shields.io/badge/Paper-ECCV%202020-1f77b4.svg)](https://arxiv.org/abs/2005.12872) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/facebookresearch/detr)

- **Sparse R-CNN: End-to-End Object Detection with Learnable Proposals** — replaces dense anchors with a sparse set of learnable proposal boxes and proposal features, an important bridge between proposal-based detection and learnable-query reasoning. [![Paper](https://img.shields.io/badge/Paper-CVPR%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Sparse_R-CNN_End-to-End_Object_Detection_With_Learnable_Proposals_CVPR_2021_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/PeizeSun/SparseR-CNN)

- **SOLQ: Segmenting Objects by Learning Queries** — extends a unified object query to encode class, location, and mask information simultaneously, illustrating how query slots can represent multiple structured outputs. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202021-1f77b4.svg)](https://proceedings.neurips.cc/paper/2021/hash/bc4e356fee1972242c8f7eabf4dff517-Abstract.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/megvii-research/SOLQ)

- **Learned Queries for Efficient Local Attention** — introduces QnA, where shared learned queries aggregate local windows and provide a query-centric alternative to expensive local self-attention. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Arar_Learned_Queries_for_Efficient_Local_Attention_CVPR_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/moabarar/qna)

- **Flamingo: a Visual Language Model for Few-Shot Learning** — the Perceiver Resampler uses a compact bank of learned latent queries to transform arbitrarily many visual features into a fixed-size interface for a language model. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202022-1f77b4.svg)](https://arxiv.org/abs/2204.14198)

- **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** — Q-Former establishes learnable query tokens as a reusable information bottleneck between frozen foundation models. [![Paper](https://img.shields.io/badge/Paper-ICML%202023-1f77b4.svg)](https://proceedings.mlr.press/v202/li23q.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/salesforce/LAVIS)

- **BoQ: A Place is Worth a Bag of Learnable Queries** — learns a bank of global queries that consistently probe local image features through cross-attention, showing that learned queries can also serve as global representation aggregators. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Ali-bey_BoQ_A_Place_is_Worth_a_Bag_of_Learnable_Queries_CVPR_2024_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/amaralibey/Bag-of-Queries)

---

## Object Detection

<em>The main lineage of learnable object queries: from static DETR queries to geometric, denoised, dense, hybrid-supervised, and image-adaptive queries.</em>

- **DETR: End-to-End Object Detection with Transformers** — formulates detection as set prediction using a fixed small set of learned object queries and bipartite matching. [![Paper](https://img.shields.io/badge/Paper-ECCV%202020-1f77b4.svg)](https://arxiv.org/abs/2005.12872) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/facebookresearch/detr)

- **Deformable DETR: Deformable Transformers for End-to-End Object Detection** — associates object queries with reference points and sparse deformable cross-attention over multi-scale features. [![Paper](https://img.shields.io/badge/Paper-ICLR%202021-1f77b4.svg)](https://arxiv.org/abs/2010.04159) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/fundamentalvision/Deformable-DETR)

- **Conditional DETR for Fast Training Convergence** — decomposes decoder queries into content and spatial components, improving localization and convergence. [![Paper](https://img.shields.io/badge/Paper-ICCV%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2021/html/Meng_Conditional_DETR_for_Fast_Training_Convergence_ICCV_2021_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/Atten4Vis/ConditionalDETR)

- **Anchor DETR: Query Design for Transformer-Based Detector** — directly uses learnable reference points as anchor queries and explicitly studies query spatial design. [![Paper](https://img.shields.io/badge/Paper-AAAI%202022-1f77b4.svg)](https://arxiv.org/abs/2109.07107) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/megvii-research/AnchorDETR)

- **DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR** — parameterizes each query as a dynamically refined 4D anchor box. [![Paper](https://img.shields.io/badge/Paper-ICLR%202022-1f77b4.svg)](https://arxiv.org/abs/2201.12329) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/IDEA-Research/DAB-DETR)

- **DN-DETR: Accelerate DETR Training by Introducing Query DeNoising** — adds noisy ground-truth queries during training to stabilize bipartite matching. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Li_DN-DETR_Accelerate_DETR_Training_by_Introducing_Query_DeNoising_CVPR_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/IDEA-Research/DN-DETR)

- **AdaMixer: A Fast-Converging Query-Based Object Detector** — treats query decoding as adaptive feature sampling and mixing, emphasizing query-specific feature extraction. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Gao_AdaMixer_A_Fast-Converging_Query-Based_Object_Detector_CVPR_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/MCG-NJU/AdaMixer)

- **DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection** — combines mixed query selection, anchor-box formulation, and contrastive denoising. [![Paper](https://img.shields.io/badge/Paper-ICLR%202023-1f77b4.svg)](https://arxiv.org/abs/2203.03605) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/IDEA-Research/DINO)

- **DETRs with Hybrid Matching (H-DETR)** — adds an auxiliary one-to-many matching branch so more queries receive positive supervision during training. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Jia_DETRs_With_Hybrid_Matching_CVPR_2023_paper.html)

- **Dense Distinct Query for End-to-End Object Detection (DDQ)** — densely initializes queries but explicitly filters them into distinct candidates before one-to-one assignment. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Dense_Distinct_Query_for_End-to-End_Object_Detection_CVPR_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/jshilong/DDQ)

- **Group DETR: Fast DETR Training with Group-Wise One-to-Many Assignment** — trains multiple groups of object queries with group-wise matching, effectively augmenting query supervision. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://arxiv.org/abs/2207.13085) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/Atten4Vis/GroupDETR)

- **Co-DETR: DETRs with Collaborative Hybrid Assignments Training** — improves decoder queries through collaborative one-to-one and one-to-many auxiliary assignment heads. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Zong_DETRs_with_Collaborative_Hybrid_Assignments_Training_ICCV_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/Sense-X/Co-DETR)

- **RT-DETR: DETRs Beat YOLOs on Real-time Object Detection** — introduces uncertainty-minimal / IoU-aware query selection for real-time DETR decoding. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/lyuwenyu/RT-DETR)

- **PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection** — learns shared latent patterns and dynamically composes image-specific queries, explicitly targeting query utilization imbalance. [![Paper](https://img.shields.io/badge/Paper-CVPR%202026-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_PaQ-DETR_Learning_Pattern_and_Quality-Aware_Dynamic_Queries_for_Object_Detection_CVPR_2026_paper.html)

- **UP-DETR: Unsupervised Pre-training for Object Detection with Transformers** — pre-trains DETR by making object queries retrieve randomly cropped image patches, directly supervising query-to-region matching before detection fine-tuning. [![Paper](https://img.shields.io/badge/Paper-CVPR%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2021/html/Dai_UP-DETR_Unsupervised_Pre-Training_for_Object_Detection_With_Transformers_CVPR_2021_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/dddzg/up-detr)

- **SMCA: Fast Convergence of DETR with Spatially Modulated Co-Attention** — couples object queries with spatial Gaussian priors so each query focuses cross-attention around a predicted object location. [![Paper](https://img.shields.io/badge/Paper-ICCV%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2021/html/Gao_Fast_Convergence_of_DETR_With_Spatially_Modulated_Co-Attention_ICCV_2021_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/gaopengcuhk/SMCA-DETR)

- **Dynamic DETR: End-to-End Object Detection with Dynamic Attention** — improves query decoding through dynamic convolution and region-focused attention, making query-feature interaction explicitly adaptive. [![Paper](https://img.shields.io/badge/Paper-ICCV%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2021/html/Dai_Dynamic_DETR_End-to-End_Object_Detection_With_Dynamic_Attention_ICCV_2021_paper.html)

- **Sparse R-CNN: End-to-End Object Detection with Learnable Proposals** — uses learnable proposal boxes and proposal features as sparse detection slots instead of dense hand-designed anchors. [![Paper](https://img.shields.io/badge/Paper-CVPR%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Sparse_R-CNN_End-to-End_Object_Detection_With_Learnable_Proposals_CVPR_2021_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/PeizeSun/SparseR-CNN)

- **Sparse DETR: Efficient End-to-End Object Detection with Learnable Sparsity** — learns which encoder tokens are useful for object queries and lets decoder queries attend only to selected sparse features. [![Paper](https://img.shields.io/badge/Paper-ICLR%202022-1f77b4.svg)](https://openreview.net/forum?id=gZ9hCDWe6ke) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/kakaobrain/sparse-detr)

- **SAM-DETR: Accelerating DETR Convergence via Semantic-Aligned Matching** — makes each object query search for semantically aligned key points before decoder cross-attention, improving query initialization and matching. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Accelerating_DETR_Convergence_via_Semantic-Aligned_Matching_CVPR_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/ZhangGongjie/SAM-DETR)

- **Enhanced Training of Query-Based Object Detection via Selective Query Recollection (SQR)** — recollects selected intermediate decoder queries and reuses them as additional training queries, strengthening supervision without changing inference cost. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Enhanced_Training_of_Query-Based_Object_Detection_via_Selective_Query_Recollection_CVPR_2023_paper.html)

- **DEQDet: Deep Equilibrium Object Detection** — interprets query refinement as a fixed-point process and repeatedly refines object queries toward an equilibrium representation. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Deep_Equilibrium_Object_Detection_ICCV_2023_paper.html)

- **Focus-DETR: Less is More — Focus Attention for Efficient DETR** — selects foreground-relevant encoder tokens so object queries perform cross-attention over a compact, informative feature subset. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Less_is_More_Focus_Attention_for_Efficient_DETR_ICCV_2023_paper.html)

- **Salience DETR: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement** — stabilizes query selection by progressively filtering tokens according to hierarchical salience scores. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Salience_DETR_Enhancing_Detection_Transformer_with_Hierarchical_Salience_Filtering_Refinement_CVPR_2024_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/xiuqhou/Salience-DETR)

- **MS-DETR: Efficient DETR Training with Mixed Supervision** — applies auxiliary one-to-many supervision directly to the same primary decoder object queries, improving optimization while preserving one-to-one inference. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_MS-DETR_Efficient_DETR_Training_with_Mixed_Supervision_CVPR_2024_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/Atten4Vis/MS-DETR)

---

## Image Segmentation

<em>Learnable mask, class, kernel, and instance queries for semantic, instance, panoptic, and universal segmentation.</em>

- **Segmenter: Transformer for Semantic Segmentation** — uses learnable class embeddings that interact with patch embeddings to decode class masks. [![Paper](https://img.shields.io/badge/Paper-ICCV%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2021/html/Strudel_Segmenter_Transformer_for_Semantic_Segmentation_ICCV_2021_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/rstrudel/segmenter)

- **MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation** — recasts segmentation as mask classification using a fixed set of learned mask queries. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202021-1f77b4.svg)](https://arxiv.org/abs/2107.06278) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/facebookresearch/MaskFormer)

- **QueryInst: Parallelly Supervised Mask R-CNN** — represents each instance with query features and performs parallel query-driven box and mask refinement. [![Paper](https://img.shields.io/badge/Paper-ICCV%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2021/html/Fang_Instances_As_Queries_ICCV_2021_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/hustvl/QueryInst)

- **K-Net: Towards Unified Image Segmentation** — uses learnable kernels as dynamic object/region representations, providing a query-like unified segmentation formulation. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202021-1f77b4.svg)](https://arxiv.org/abs/2106.14855) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/ZwwWayne/K-Net)

- **Mask2Former: Masked-Attention Mask Transformer for Universal Image Segmentation** — iteratively refines learnable mask queries with masked cross-attention. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/facebookresearch/Mask2Former)

- **Mask DINO: Towards a Unified Transformer-based Framework for Object Detection and Segmentation** — unifies box and mask prediction with DINO-style query embeddings and denoising training. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Mask_DINO_Towards_a_Unified_Transformer-Based_Framework_for_Object_Detection_CVPR_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/IDEA-Research/MaskDINO)

- **OneFormer: One Transformer To Rule Universal Image Segmentation** — combines task-conditioned queries, object queries, and text queries in a single universal model. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Jain_OneFormer_One_Transformer_To_Rule_Universal_Image_Segmentation_CVPR_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/SHI-Labs/OneFormer)

- **X-Decoder: Generalized Decoding for Pixel, Image, and Language** — combines generic latent queries and text queries for open-world segmentation and language tasks. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://arxiv.org/abs/2212.11270) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/microsoft/X-Decoder)

- **SEEM: Segment Everything Everywhere All at Once** — unifies learnable visual queries with interactive spatial and language prompts. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202023-1f77b4.svg)](https://arxiv.org/abs/2304.06718) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)

- **FastInst: A Simple Query-Based Model for Real-Time Instance Segmentation** — dynamically initializes instance queries from high-semantic pixel embeddings to reduce decoder iterations. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/He_FastInst_A_Simple_Query-Based_Model_for_Real-Time_Instance_Segmentation_CVPR_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/junjiehe96/FastInst)

- **MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers** — predicts a fixed set of panoptic masks with learned mask slots, bringing set prediction and query-style decoding to panoptic segmentation. [![Paper](https://img.shields.io/badge/Paper-CVPR%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_MaX-DeepLab_End-to-End_Panoptic_Segmentation_With_Mask_Transformers_CVPR_2021_paper.html)

- **SOLQ: Segmenting Objects by Learning Queries** — lets a single object query encode category, localization, and mask information, providing a unified query representation for instance segmentation. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202021-1f77b4.svg)](https://proceedings.neurips.cc/paper/2021/hash/bc4e356fee1972242c8f7eabf4dff517-Abstract.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/megvii-research/SOLQ)

- **CMT-DeepLab: Clustering Mask Transformers for Panoptic Segmentation** — treats mask prediction as cluster assignment between pixel features and Transformer-learned object-centric mask embeddings. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_CMT-DeepLab_Clustering_Mask_Transformers_for_Panoptic_Segmentation_CVPR_2022_paper.html)

- **Panoptic SegFormer: Delving Deeper into Panoptic Segmentation with Transformers** — explicitly decouples thing and stuff query behaviors and improves query-based panoptic decoding. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Panoptic_SegFormer_Delving_Deeper_Into_Panoptic_Segmentation_With_Transformers_CVPR_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/zhiqi-li/Panoptic-SegFormer)

- **kMaX-DeepLab: k-means Mask Transformer** — updates object queries using a k-means-like cross-attention formulation, making query-mask clustering more explicit. [![Paper](https://img.shields.io/badge/Paper-ECCV%202022-1f77b4.svg)](https://arxiv.org/abs/2207.04044) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/bytedance/kmax-deeplab)

- **MP-Former: Mask-Piloted Transformer for Image Segmentation** — feeds mask-piloted auxiliary queries into the decoder during training to reduce inconsistency between query features and predicted masks. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_MP-Former_Mask-Piloted_Transformer_for_Image_Segmentation_CVPR_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/IDEA-Research/MP-Former)

- **CoMasTRe: Continual Segmentation with Disentangled Objectness Learning and Class Recognition** — leverages the class-agnostic objectness encoded by query-based masks to mitigate forgetting in continual segmentation. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Cermelli_Continual_Segmentation_with_Disentangled_Objectness_Learning_and_Class_Recognition_CVPR_2024_paper.html)

- **SimCIS: Rethinking Query-Based Transformer for Continual Image Segmentation** — directly assigns selected visual features to queries and introduces visual-query replay across incremental learning stages. [![Paper](https://img.shields.io/badge/Paper-CVPR%202025-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Rethinking_Query-based_Transformer_for_Continual_Image_Segmentation_CVPR_2025_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/SooLab/SimCIS)

---

## Object Tracking

<em>Queries evolve from static detection slots into persistent temporal states that carry object identity through time.</em>

- **TrackFormer: Multi-Object Tracking with Transformers** — introduces identity-preserving track queries that are propagated autoregressively across frames while static queries initialize new tracks. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Meinhardt_TrackFormer_Multi-Object_Tracking_With_Transformers_CVPR_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/timmeinhardt/trackformer)

- **MOTR: End-to-End Multiple-Object Tracking with Transformer** — each track query models an entire object track and is transferred and updated frame-by-frame. [![Paper](https://img.shields.io/badge/Paper-ECCV%202022-1f77b4.svg)](https://arxiv.org/abs/2105.03247) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/megvii-research/MOTR)

- **MOTRv2: Bootstrapping End-to-End Multi-Object Tracking by Pretrained Object Detectors** — uses detector proposals as anchor-form track queries, injecting strong detection priors into MOTR. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_MOTRv2_Bootstrapping_End-to-End_Multi-Object_Tracking_by_Pretrained_Object_Detectors_CVPR_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/megvii-research/MOTRv2)

- **MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking** — augments track-query representations with long-term memory to improve temporal identity modeling. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://arxiv.org/abs/2307.15700) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/MCG-NJU/MeMOTR)

- **MUTR3D: A Multi-Camera Tracking Framework via 3D-to-2D Queries** — introduces persistent 3D track queries whose reference locations and features are updated across cameras and frames. [![Paper](https://img.shields.io/badge/Paper-CVPRW%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022W/WAD/html/Zhang_MUTR3D_A_Multi-Camera_Tracking_Framework_via_3D-to-2D_Queries_CVPRW_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/a1600012888/mutr3d)

- **ADA-Track: End-to-End Multi-Camera 3D Multi-Object Tracking with Alternating Detection and Association** — separates track and detection queries and alternates query-to-image detection with query-to-query association. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_ADA-Track_End-to-End_Multi-Camera_3D_Multi-Object_Tracking_with_Alternating_Detection_and_CVPR_2024_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/dsx0511/ADA-Track)

- **CO-MOT: Boosting End-to-End Transformer-Based Multi-Object Tracking via Coopetition Label Assignment and Shadow Sets** — improves end-to-end track-query learning with richer query assignment and duplicate-aware shadow tracking. [![Paper](https://img.shields.io/badge/Paper-ICLR%202025-1f77b4.svg)](https://openreview.net/forum?id=Q9gW3Z2e7M) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/BingfengYan/CO-MOT)

- **SambaMOTR: Synchronized Set-of-Sequences Modeling for Multiple Object Tracking** — models trajectories as synchronized query sequences and autoregressively predicts future track-query states. [![Paper](https://img.shields.io/badge/Paper-ICLR%202025-1f77b4.svg)](https://openreview.net/forum?id=FvYtpfQyG7) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/mattiasegu/sambamotr)

- **MOTIP: Multiple Object Tracking as ID Prediction** — reframes association as direct ID decoding from trajectory representations, providing a modern Transformer baseline adjacent to persistent-query tracking. [![Paper](https://img.shields.io/badge/Paper-CVPR%202025-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/MCG-NJU/MOTIP)

---

## Video Understanding and Temporal Grounding

<em>Queries represent temporal moments, object tubes, referred instances, or persistent video objects.</em>

- **VisTR: End-to-End Video Instance Segmentation with Transformers** — uses instance-sequence queries to jointly predict video object sequences and masks. [![Paper](https://img.shields.io/badge/Paper-CVPR%202021-1f77b4.svg)](https://arxiv.org/abs/2011.14503) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/Epiphqny/VisTR)

- **Moment-DETR / QVHighlights** — formulates video moment retrieval as set prediction with learnable temporal moment queries. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202021-1f77b4.svg)](https://arxiv.org/abs/2107.09609) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/jayleicn/moment_detr)

- **TubeDETR: Spatio-Temporal Video Grounding with Transformers** — uses language-conditioned queries to decode referred object tubes across space and time. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_TubeDETR_Spatio-Temporal_Video_Grounding_With_Transformers_CVPR_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/antoyang/TubeDETR)

- **SeqFormer: Sequential Transformer for Video Instance Segmentation** — propagates and aggregates instance queries over frames to construct robust video-level object representations. [![Paper](https://img.shields.io/badge/Paper-ECCV%202022-1f77b4.svg)](https://arxiv.org/abs/2112.08275) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/wjf5203/SeqFormer)

- **ReferFormer: Language as Queries for Referring Video Object Segmentation** — conditions object queries on language so that queries directly search for the referred object and remain linked across frames. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://arxiv.org/abs/2201.00487) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/wjn922/ReferFormer)

- **IFC: Video Instance Segmentation using Inter-Frame Communication Transformers** — uses a small set of memory tokens / object-centric representations to exchange information across frames for video instance segmentation. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202021-1f77b4.svg)](https://proceedings.neurips.cc/paper/2021/hash/6f3ef77ac0e3619e98159e9b6febf557-Abstract.html)

- **EfficientVIS: Efficient Video Instance Segmentation via Tracklet Query and Proposal** — introduces tracklet queries that jointly encode instance identity across video frames and are refined from tracklet proposals. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Efficient_Video_Instance_Segmentation_via_Tracklet_Query_and_Proposal_CVPR_2022_paper.html)

- **TeViT: Temporally Efficient Vision Transformer for Video Instance Segmentation** — augments a query-based VIS head with spatiotemporal query interaction so object queries remain coherent over time. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Temporally_Efficient_Vision_Transformer_for_Video_Instance_Segmentation_CVPR_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/hustvl/TeViT)

- **TubeFormer-DeepLab: Video Mask Transformer** — predicts temporally consistent tube masks using a fixed set of Transformer mask queries. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_TubeFormer-DeepLab_Video_Mask_Transformer_CVPR_2022_paper.html)

- **Video K-Net: A Simple, Strong, and Unified Baseline for Video Segmentation** — propagates and updates object kernels / query-like instance representations across frames for multiple video segmentation tasks. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Video_K-Net_A_Simple_Strong_and_Unified_Baseline_for_Video_Segmentation_CVPR_2022_paper.html)

- **VITA: Video Instance Segmentation via Object Token Association** — constructs frame-level object tokens and associates them into video-level object queries/tokens for global temporal reasoning. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202022-1f77b4.svg)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ebf9205ec6eb848613f0f83f0676f1d1-Abstract-Conference.html)

- **MinVIS: A Minimal Video Instance Segmentation Framework without Video-Based Training** — associates query embeddings predicted independently on adjacent frames, revealing the temporal identity information already encoded by image-trained queries. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202022-1f77b4.svg)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/14c2b4f14e18f6ef6deae2dbb4653ba9-Abstract-Conference.html)

- **MDQE: Mining Discriminative Query Embeddings to Segment Occluded Instances on Challenging Videos** — redesigns object-query initialization and representation so nearby/occluded instances obtain more discriminative temporal query embeddings. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Li_MDQE_Mining_Discriminative_Query_Embeddings_To_Segment_Occluded_Instances_on_CVPR_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/MinghanLi/MDQE_CVPR2023)

- **GenVIS: A Generalized Framework for Video Instance Segmentation** — trains query-based VIS over long videos using generalized target assignment so instance queries preserve object identity across clips. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Heo_A_Generalized_Framework_for_Video_Instance_Segmentation_CVPR_2023_paper.html)

- **Look Before You Match: Instance Understanding Matters in Video Object Segmentation** — adds a query-based instance-segmentation branch so object-level query understanding complements memory matching. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Look_Before_You_Match_Instance_Understanding_Matters_in_Video_Object_CVPR_2023_paper.html)

- **Cutie: Putting the Object Back into Video Object Segmentation** — maintains a small adaptive set of object queries as high-level target summaries and iteratively interacts them with pixel memory. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Putting_the_Object_Back_into_Video_Object_Segmentation_CVPR_2024_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/hkchengrex/Cutie)

- **Action Detail Matters: Refining Video Recognition with Local Action Queries** — learns action queries that selectively retrieve action-relevant local regions and propagates them progressively through the video. [![Paper](https://img.shields.io/badge/Paper-CVPR%202025-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Action_Detail_Matters_Refining_Video_Recognition_with_Local_Action_Queries_CVPR_2025_paper.html)

---

## 3D Perception, BEV, and Motion Prediction

<em>Queries become 3D anchors, BEV grid cells, object-centric temporal memories, and trajectory-intention modes.</em>

- **3DETR: An End-to-End Transformer Model for 3D Object Detection** — transfers DETR-style set prediction to point clouds using 3D query points sampled from input geometry. [![Paper](https://img.shields.io/badge/Paper-ICCV%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2021/html/Misra_An_End-to-End_Transformer_Model_for_3D_Object_Detection_ICCV_2021_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/facebookresearch/3detr)

- **DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries** — learnable 3D object queries project reference points into multiple cameras to retrieve image features. [![Paper](https://img.shields.io/badge/Paper-CoRL%202021-1f77b4.svg)](https://arxiv.org/abs/2110.06922) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/WangYueFt/detr3d)

- **PETR: Position Embedding Transformation for Multi-View 3D Object Detection** — uses 3D-aware object queries with position-aware image features in a DETR-style decoder. [![Paper](https://img.shields.io/badge/Paper-ECCV%202022-1f77b4.svg)](https://arxiv.org/abs/2203.05625) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/megvii-research/PETR)

- **BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers** — represents the world with a dense set of predefined learnable BEV queries. [![Paper](https://img.shields.io/badge/Paper-ECCV%202022-1f77b4.svg)](https://arxiv.org/abs/2203.17270) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/fundamentalvision/BEVFormer)

- **Motion Transformer (MTR)** — introduces a small set of learnable motion / intention query pairs, each responsible for one future motion mode. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202022-1f77b4.svg)](https://papers.neurips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/sshaoshuai/MTR)

- **PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images** — extends position-aware object queries to temporal and multi-task 3D perception. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://arxiv.org/abs/2206.01256) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/megvii-research/PETR)

- **StreamPETR: Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection** — propagates historical object queries through a memory queue for streaming 3D detection. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://arxiv.org/abs/2303.11926) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/exiawsh/StreamPETR)

- **SparseBEV: High-Performance Sparse 3D Object Detection from Multi-Camera Videos** — each sparse query explicitly represents 3D translation, size, rotation, and velocity and drives adaptive spatio-temporal sampling. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://arxiv.org/abs/2308.09244) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/MCG-NJU/SparseBEV)

- **PARQ: Pixel-Aligned Recurrent Queries for Multi-View 3D Object Detection** — recurrently updates query position and appearance using pixel-aligned multi-view information. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Xie_Pixel-Aligned_Recurrent_Queries_for_Multi-View_3D_Object_Detection_ICCV_2023_paper.html)

- **MTR++: Multi-Agent Motion Prediction with Symmetric Scene Modeling and Guided Intention Querying** — extends learnable intention queries to interacting multi-agent future prediction with mutually guided querying. [![Paper](https://img.shields.io/badge/Paper-TPAMI%202024-1f77b4.svg)](https://arxiv.org/abs/2306.17770) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/sshaoshuai/MTR)

- **SPFormer: Superpoint Transformer for 3D Scene Instance Segmentation** — predicts 3D object instances directly using query vectors that attend to superpoint features. [![Paper](https://img.shields.io/badge/Paper-AAAI%202023-1f77b4.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/25344) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/sunjiahao1999/SPFormer)

- **Mask3D: Mask Transformer for 3D Instance Segmentation** — represents each candidate 3D object with an instance query and predicts masks by query-to-point feature interaction. [![Paper](https://img.shields.io/badge/Paper-ICRA%202023-1f77b4.svg)](https://arxiv.org/abs/2210.03105) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/JonasSchult/Mask3D)

- **QueryFormer: Query Refinement Transformer for 3D Instance Segmentation** — improves 3D instance queries through stronger initialization, iterative refinement, and background-query suppression. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Lu_QueryFormer_Query_Refinement_Transformer_for_3D_Instance_Segmentation_ICCV_2023_paper.html)

- **Query6DoF: Learning Sparse Queries as Implicit Shape Prior for Category-Level 6DoF Pose Estimation** — learns category-specific sparse queries in which each query represents a shape component and dynamically retrieves corresponding point-cloud evidence. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Query6DoF_Learning_Sparse_Queries_as_Implicit_Shape_Prior_for_Category-Level_ICCV_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/hustvl/Query6DoF)

- **Object as Query: Lifting Any 2D Object Detector to 3D Detection** — turns 2D detections into object queries that retrieve and lift multi-view image evidence into 3D object predictions. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Object_as_Query_Lifting_Any_2D_Object_Detector_to_3D_ICCV_2023_paper.html)

- **OneFormer3D: One Transformer for Unified Point Cloud Segmentation** — unifies semantic, instance, and panoptic point-cloud segmentation with query-based decoding and task-shared representations. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Kolodiazhnyi_OneFormer3D_One_Transformer_for_Unified_Point_Cloud_Segmentation_CVPR_2024_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/filaPro/oneformer3d)

- **ADA-Track: End-to-End Multi-Camera 3D Multi-Object Tracking with Alternating Detection and Association** — combines detection queries and persistent track queries through alternating query-to-image and query-to-query attention. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_ADA-Track_End-to-End_Multi-Camera_3D_Multi-Object_Tracking_with_Alternating_Detection_and_CVPR_2024_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/dsx0511/ADA-Track)

- **UniAD: Planning-Oriented Autonomous Driving** — uses a unified query design as task interfaces across tracking, mapping, motion forecasting, occupancy, and planning, making query representations carriers of upstream knowledge to the planner. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/OpenDriveLab/UniAD)

---

## Human-Object Interaction and Pose

<em>Queries represent human-object pairs, interactions, humans, or keypoints.</em>

- **HOTR: End-to-End Human-Object Interaction Detection with Transformers** — introduces interaction queries that directly decode human-object interaction triplets. [![Paper](https://img.shields.io/badge/Paper-CVPR%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2021/html/Kim_HOTR_End-to-End_Human-Object_Interaction_Detection_With_Transformers_CVPR_2021_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/kakaobrain/HOTR)

- **QPIC: Query-Based Pairwise Human-Object Interaction Detection** — designs each query to capture at most one human-object pair and its interaction. [![Paper](https://img.shields.io/badge/Paper-CVPR%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2021/html/Tamura_QPIC_Query-Based_Pairwise_Human-Object_Interaction_Detection_With_Image-Wide_Contextual_Information_CVPR_2021_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/hitachi-rd-cv/qpic)

- **Poseur: Direct Human Pose Regression with Transformers** — uses keypoint queries with deformable attention to directly regress human keypoint coordinates. [![Paper](https://img.shields.io/badge/Paper-ECCV%202022-1f77b4.svg)](https://arxiv.org/abs/2201.07412) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/aim-uofa/Poseur)

- **ED-Pose: Explicit Box Detection Unifies End-to-End Multi-Person Pose Estimation** — builds human queries and expands them into human-keypoint queries in a unified DETR-like framework. [![Paper](https://img.shields.io/badge/Paper-ICLR%202023-1f77b4.svg)](https://openreview.net/forum?id=s4WVupnJjmX) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/IDEA-Research/ED-Pose)

- **CDN: Mining the Benefits of Two-Stage and One-Stage HOI Detection** — uses cascaded Transformer decoders so one query stream focuses on human-object pair detection and a second stage focuses on interaction classification. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202021-1f77b4.svg)](https://proceedings.neurips.cc/paper/2021/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/YueLiao/CDN)

- **PETR: End-to-End Multi-Person Pose Estimation with Transformers** — formulates pose estimation as hierarchical set prediction with human-level and joint-level query representations. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Shi_End-to-End_Multi-Person_Pose_Estimation_With_Transformers_CVPR_2022_paper.html)

- **GEN-VLKT: Simplify Association and Enhance Interaction Understanding for HOI Detection** — uses independent human/object query sets plus interaction queries generated from instance decoder outputs. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Liao_GEN-VLKT_Simplify_Association_and_Enhance_Interaction_Understanding_for_HOI_Detection_CVPR_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/YueLiao/gen-vlkt)

- **Category Query Learning for Human-Object Interaction Classification** — explicitly associates learnable queries with interaction categories and converts them into image-specific category representations through a decoder. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Xie_Category_Query_Learning_for_Human-Object_Interaction_Classification_CVPR_2023_paper.html)

- **HOICLIP: Efficient Knowledge Transfer for HOI Detection with Vision-Language Models** — uses a query-driven interaction decoder to retrieve CLIP visual regions and fuse interaction knowledge into HOI detection. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Ning_HOICLIP_Efficient_Knowledge_Transfer_for_HOI_Detection_With_Vision-Language_Models_CVPR_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/Artanic30/HOICLIP)

- **Group Pose: A Simple Baseline for End-to-End Multi-Person Pose Estimation** — represents every human by one instance query plus K keypoint queries and explicitly structures query interaction within and across persons. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Group_Pose_A_Simple_Baseline_for_End-to-End_Multi-Person_Pose_Estimation_ICCV_2023_paper.html)

- **QAHOI: Query-Based Anchors for Human-Object Interaction Detection** — uses query-based anchors to predict all components of an HOI instance across multi-scale features. [![Paper](https://img.shields.io/badge/Paper-MVA%202023-1f77b4.svg)](https://www.ieice.org/publications/proceedings/summary.php?expandable=18&iconf=MVA&number=O2-1-1&session_num=O2-1&year=2023) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/cjw2021/QAHOI)

---

## Classification, Recognition, and Retrieval Queries

<em>Queries correspond to labels or label groups and retrieve category-specific visual evidence.</em>

- **Query2Label: A Simple Transformer Way to Multi-Label Classification** — uses one learnable label query per category to retrieve class-specific evidence from image features. [![Paper](https://img.shields.io/badge/Paper-Preprint%202021-777777.svg)](https://arxiv.org/abs/2107.10834) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/SlongLiu/query2labels)

- **ML-Decoder: Scalable and Versatile Classification Head** — uses group-decoding queries and supports learnable, random, or language-derived queries for scalable classification and zero-shot learning. [![Paper](https://img.shields.io/badge/Paper-WACV%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/WACV2023/html/Ridnik_ML-Decoder_Scalable_and_Versatile_Classification_Head_WACV_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/Alibaba-MIIL/ML_Decoder)

- **Segmenter: Transformer for Semantic Segmentation** — although primarily a segmentation model, its learnable class embeddings provide an early example of label queries that retrieve category-specific evidence from image tokens. [![Paper](https://img.shields.io/badge/Paper-ICCV%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2021/html/Strudel_Segmenter_Transformer_for_Semantic_Segmentation_ICCV_2021_paper.html)

- **Learned Queries for Efficient Local Attention** — learned local queries pool neighborhood information efficiently and can be used as a general visual representation operator for recognition backbones. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Arar_Learned_Queries_for_Efficient_Local_Attention_CVPR_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/moabarar/qna)

- **BoQ: A Place is Worth a Bag of Learnable Queries** — uses learnable global queries to aggregate local descriptors into a compact visual-place representation for large-scale retrieval. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Ali-bey_BoQ_A_Place_is_Worth_a_Bag_of_Learnable_Queries_CVPR_2024_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/amaralibey/Bag-of-Queries)

- **Category Query Learning for Human-Object Interaction Classification** — treats interaction categories themselves as queries and turns them into sample-conditioned category representations for classification. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Xie_Category_Query_Learning_for_Human-Object_Interaction_Classification_CVPR_2023_paper.html)

- **Query6DoF: Learning Sparse Queries as Implicit Shape Prior for Category-Level 6DoF Pose Estimation** — learns category-specific sparse shape queries for recognition and pose estimation of unseen object instances. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Query6DoF_Learning_Sparse_Queries_as_Implicit_Shape_Prior_for_Category-Level_ICCV_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/hustvl/Query6DoF)

- **Action Detail Matters: Refining Video Recognition with Local Action Queries** — learns action-category-aware local queries that discover discriminative action regions without region-level supervision. [![Paper](https://img.shields.io/badge/Paper-CVPR%202025-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Action_Detail_Matters_Refining_Video_Recognition_with_Local_Action_Queries_CVPR_2025_paper.html)

- **Learning Interpretable Queries for Explainable Image Classification with Information Pursuit** — learns an interpretable query dictionary in vision-language feature space and classifies images through sequentially selected semantic queries. [![Paper](https://img.shields.io/badge/Paper-ICCV%202025-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Learning_Interpretable_Queries_for_Explainable_Image_Classification_with_Information_Pursuit_ICCV_2025_paper.html)

---

## Vision-Language and Multimodal Models

<em>Queries serve as cross-modal grounding variables or compact bottlenecks between vision encoders and language models.</em>

- **MDETR: Modulated Detection for End-to-End Multi-Modal Understanding** — conditions DETR-style object queries on text for phrase grounding and multimodal detection. [![Paper](https://img.shields.io/badge/Paper-ICCV%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2021/html/Kamath_MDETR_Modulated_Detection_for_End-to-End_Multi-Modal_Understanding_ICCV_2021_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/ashkamath/mdetr)

- **Flamingo: a Visual Language Model for Few-Shot Learning** — uses a Perceiver Resampler with learned latent queries to compress variable-size visual features into a fixed number of visual tokens. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202022-1f77b4.svg)](https://arxiv.org/abs/2204.14198)

- **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** — introduces Q-Former, where a fixed set of learnable query tokens extracts vision information needed by a frozen LLM. [![Paper](https://img.shields.io/badge/Paper-ICML%202023-1f77b4.svg)](https://proceedings.mlr.press/v202/li23q.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/salesforce/LAVIS)

- **InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning** — makes Q-Former instruction-aware so query tokens selectively extract task-relevant visual content. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202023-1f77b4.svg)](https://arxiv.org/abs/2305.06500) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/salesforce/LAVIS)

- **RefFormer: Referencing Where to Focus — Improving Visual Grounding with Referential Query** — adapts CLIP features to generate target-aware referential queries instead of using randomly initialized decoder queries. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202024-1f77b4.svg)](https://papers.nips.cc/paper_files/paper/2024/hash/54c67d3db2df24a31cf045525f9460b9-Abstract-Conference.html)

- **Vision-Language Transformer and Query Generation for Referring Segmentation** — generates multiple language-conditioned query sets that guide visual decoding toward the referred object. [![Paper](https://img.shields.io/badge/Paper-ICCV%202021-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2021/html/Ding_Vision-Language_Transformer_and_Query_Generation_for_Referring_Segmentation_ICCV_2021_paper.html)

- **GLIP: Grounded Language-Image Pre-Training** — unifies object detection and phrase grounding so language expressions provide semantic queries for visual localization at scale. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/microsoft/GLIP)

- **OV-DETR: Open-Vocabulary DETR with Conditional Matching** — conditions the detector on text or image query embeddings, enabling query-conditioned open-vocabulary object matching. [![Paper](https://img.shields.io/badge/Paper-ECCV%202022-1f77b4.svg)](https://arxiv.org/abs/2203.11876) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/yuhangzang/OV-DETR)

- **OneFormer: One Transformer To Rule Universal Image Segmentation** — jointly uses task queries, object queries, and text embeddings to condition one model on semantic, instance, or panoptic segmentation. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Jain_OneFormer_One_Transformer_To_Rule_Universal_Image_Segmentation_CVPR_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/SHI-Labs/OneFormer)

- **X-Decoder: Generalized Decoding for Pixel, Image, and Language** — uses latent queries together with text queries as a universal decoder interface for segmentation, retrieval, captioning, and referring tasks. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://arxiv.org/abs/2212.11270) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/microsoft/X-Decoder)

- **SEEM: Segment Everything Everywhere All at Once** — combines learnable visual queries with spatial, visual, and language prompts in a unified multimodal segmentation decoder. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202023-1f77b4.svg)](https://arxiv.org/abs/2304.06718) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)

- **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection** — combines language-guided query selection with learnable decoder content queries to connect text concepts and object localization. [![Paper](https://img.shields.io/badge/Paper-ECCV%202024-1f77b4.svg)](https://arxiv.org/abs/2303.05499) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/IDEA-Research/GroundingDINO)

- **Querying as Prompt: Parameter-Efficient Learning for Multimodal Language Model** — uses a learnable querying prompt simultaneously as a modality-retrieval query and as a prompt injected into the language model. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_Querying_as_Prompt_Parameter-Efficient_Learning_for_Multimodal_Language_Model_CVPR_2024_paper.html)

- **Segment and Caption Anything** — employs a lightweight query-based feature mixer to extract region-level representations and align segmentation outputs with a language decoder. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Segment_and_Caption_Anything_CVPR_2024_paper.html)

- **CrossMAE: Cross-Modality Masked Autoencoders for Region-Aware Audio-Visual Pre-Training** — introduces modality-specific learnable queries that retrieve aligned visual regions and spectrogram locations during cross-modal reconstruction. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_CrossMAE_Cross-Modality_Masked_Autoencoders_for_Region-Aware_Audio-Visual_Pre-Training_CVPR_2024_paper.html)

---

## Natural Language Processing and Information Extraction

<em>DETR-style set prediction is transferred from visual objects to entities, relations, and multimodal named entities.</em>

- **Joint Entity and Relation Extraction with Set Prediction Networks (SPN)** — directly predicts an unordered set of relational triples using non-autoregressive decoder queries and bipartite matching. [![Paper](https://img.shields.io/badge/Paper-TNNLS%202023-1f77b4.svg)](https://arxiv.org/abs/2011.01675) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/DianboWork/SPN4RE)

- **PIQN: Parallel Instance Query Network for Named Entity Recognition** — introduces global learnable instance queries, with each query predicting one entity in parallel. [![Paper](https://img.shields.io/badge/Paper-ACL%202022-1f77b4.svg)](https://aclanthology.org/2022.acl-long.67/)

- **BiSPN: Generating Entity Set and Relation Set Coherently in One Pass** — jointly generates entity and relation sets in parallel with bipartite set prediction and consistency objectives. [![Paper](https://img.shields.io/badge/Paper-Findings%20EMNLP%202023-1f77b4.svg)](https://aclanthology.org/2023.findings-emnlp.136/)

- **MQSPN: Multi-Grained Query-Guided Set Prediction Network for Grounded Multimodal Named Entity Recognition** — combines type-grained and learnable entity-grained queries to align textual entities with visual regions. [![Paper](https://img.shields.io/badge/Paper-AAAI%202025-1f77b4.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/34711) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/tangjielong928/mqspn)

- **Query-Based Instance Discrimination Network for Relational Triple Extraction** — represents relation candidates with query slots and discriminates multiple relational triples in parallel rather than autoregressively. [![Paper](https://img.shields.io/badge/Paper-EMNLP%202022-1f77b4.svg)](https://aclanthology.org/2022.emnlp-main.659/)

- **Propose-and-Refine: A Two-Stage Set Prediction Network for Nested Named Entity Recognition** — first proposes entity queries and then refines their spans/types, avoiding sequential entity generation. [![Paper](https://img.shields.io/badge/Paper-IJCAI%202022-1f77b4.svg)](https://www.ijcai.org/proceedings/2022/0604.pdf)

- **SetSum: A Set Prediction Network for Extractive Summarization** — uses a fixed set of learnable sentence queries and bipartite matching to predict summary sentences without autoregressive ordering. [![Paper](https://img.shields.io/badge/Paper-Findings%20ACL%202023-1f77b4.svg)](https://aclanthology.org/2023.findings-acl.595/)

- **QueryForm: A Simple Zero-Shot Form Entity Query Framework** — composes schema/entity-type information into semantic queries that retrieve corresponding entities from visually rich documents. [![Paper](https://img.shields.io/badge/Paper-Findings%20ACL%202023-1f77b4.svg)](https://aclanthology.org/2023.findings-acl.699/)

---

## Audio and Speech

<em>Queries represent audio events, speakers, sources, instruments, sounding objects, or cross-modal acoustic targets. Because explicit learnable-query audio work is a smaller literature than detection/segmentation, this section keeps a few influential preprints but marks them separately.</em>

- **Sound Event Detection Transformer (SEDT)** — adapts 1D-DETR to sound-event detection and introduces an audio-query branch with set-based event prediction. [![Paper](https://img.shields.io/badge/Paper-Preprint%202021-777777.svg)](https://arxiv.org/abs/2110.02011) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/Anaesthesiaye/sound_event_detection_transformer)

- **DiFormer: End-to-End Speaker Diarization with Transformer** — predicts a set of speaker masks, speaker vectors, and activity indicators from learnable query embeddings. [![Paper](https://img.shields.io/badge/Paper-Preprint%202021-777777.svg)](https://arxiv.org/abs/2112.07463)

- **Zero-Shot Audio Source Separation through Query-Based Learning from Weakly-Labeled Data** — encodes target-source queries to control which sound source a universal separator should extract. [![Paper](https://img.shields.io/badge/Paper-AAAI%202022-1f77b4.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/20366)

- **iQuery: Instruments As Queries for Audio-Visual Sound Separation** — learns instrument queries as audio prototypes and can add new query embeddings for unseen instruments while freezing the backbone. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_iQuery_Instruments_As_Queries_for_Audio-Visual_Sound_Separation_CVPR_2023_paper.html)

- **AQFormer: Discovering Sounding Objects by Audio Queries for Audio Visual Segmentation** — conditions object queries on audio so each query binds to a sounding object across video frames. [![Paper](https://img.shields.io/badge/Paper-IJCAI%202023-1f77b4.svg)](https://www.ijcai.org/proceedings/2023/0097.pdf)

- **Detect Any Sound: Open-Vocabulary Sound Event Detection with Multi-Modal Queries** — formulates sound-event detection as retrieval against text- or audio-derived query vectors for open-vocabulary generalization. [![Paper](https://img.shields.io/badge/Paper-Preprint%202025-777777.svg)](https://arxiv.org/abs/2507.16343)

- **CrossMAE: Cross-Modality Masked Autoencoders for Region-Aware Audio-Visual Pre-Training** — learns audio and visual query banks that retrieve aligned regions across spectrogram and image features. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_CrossMAE_Cross-Modality_Masked_Autoencoders_for_Region-Aware_Audio-Visual_Pre-Training_CVPR_2024_paper.html)

- **Audio-Visual Instance Segmentation** — initializes learnable video queries that aggregate object-centric audio-visual tokens and decode sounding-object classes and masks. [![Paper](https://img.shields.io/badge/Paper-CVPR%202025-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Audio-Visual_Instance_Segmentation_CVPR_2025_paper.html)

- **CASP: Consistency-Aware Audio-Induced Saliency Prediction Model for Omnidirectional Video** — uses learnable audio queries to dynamically align acoustic cues with panoramic visual regions and predict temporally coherent saliency. [![Paper](https://img.shields.io/badge/Paper-CVPR%202025-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Wan_CASP_Consistency-aware_Audio-induced_Saliency_Prediction_Model_for_Omnidirectional_Video_CVPR_2025_paper.html)

- **iQuery: Instruments as Queries for Audio-Visual Sound Separation** — represents instrument categories with trainable query prototypes that retrieve and separate corresponding audio-visual sources. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_iQuery_Instruments_As_Queries_for_Audio-Visual_Sound_Separation_CVPR_2023_paper.html)

---

## Medical Imaging and Computational Pathology

<em>Query-based modeling is particularly suitable for anatomy-, lesion-, annotator-, tissue-, phenotype-, and biomarker-aware medical learning.</em>

- **DT-MIL: Deformable Transformer for Multi-instance Learning on Histopathological Image** — uses learnable query embeddings in a deformable-transformer MIL aggregator for WSI representation learning. [![Paper](https://img.shields.io/badge/Paper-MICCAI%202021-1f77b4.svg)](https://miccai2021.org/openaccess/paperlinks/2021/09/01/160-Paper0828.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/yfzon/DT-MIL)

- **Transformer-based Annotation Bias-aware Medical Image Segmentation (TAB)** — introduces annotator-specific learnable preference queries, each supervised by one annotator's segmentation style. [![Paper](https://img.shields.io/badge/Paper-MICCAI%202023-1f77b4.svg)](https://conferences.miccai.org/2023/papers/667-Paper0211.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/Merrical/TAB)

- **Learnable Query Initialization for Surgical Instrument Instance Segmentation** — proposes a Query Proposal Network / Decoder to generate better decoder-query initialization for highly occluded surgical instruments. [![Paper](https://img.shields.io/badge/Paper-MICCAI%202023-1f77b4.svg)](https://conferences.miccai.org/2023/papers/369-Paper3610.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/AINeurosurgery/Learnable-QPD-for-maskDINO)

- **PathM3: A Multimodal Multi-Task Multiple Instance Learning Framework for Whole Slide Image Classification and Captioning** — adapts a query-based transformer to align WSI visual representations with diagnostic captions. [![Paper](https://img.shields.io/badge/Paper-MICCAI%202024-1f77b4.svg)](https://papers.miccai.org/miccai-2024/593-Paper3991.html)

- **QuCCeS: Query-guided Generalizable Medical Image Segmentation** — uses semantically driven learnable queries as flexible prototypes that adapt to each test sample under cross-center distribution shift. [![Paper](https://img.shields.io/badge/Paper-PRL%202024-1f77b4.svg)](https://www.sciencedirect.com/science/article/pii/S0167865524001752)

- **Learnable Context in Multiple Instance Learning for Whole Slide Image Classification and Segmentation** — prepends multiple learnable query/context vectors to WSI instance features to learn slide-level contextual representations. [![Paper](https://img.shields.io/badge/Paper-JIIM%202025-1f77b4.svg)](https://pubmed.ncbi.nlm.nih.gov/39495442/)

- **LiteMIL: A Computationally Efficient Cross-Attention MIL for Cancer Subtyping on Whole-Slide Images** — aggregates pathology patch features through configurable learnable query-based multi-head cross-attention. [![Paper](https://img.shields.io/badge/Paper-Preprint%202025-777777.svg)](https://doi.org/10.1101/2025.05.11.25327389) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/hkussaibi/LiteMIL)

- **CancerUniT: Towards a Single Unified Model for Detection, Segmentation, and Diagnosis of Eight Major Cancers** — decomposes medical class queries into organ/shared, tumor-detection, and diagnosis queries and explicitly builds a detection-to-diagnosis query hierarchy. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_CancerUniT_Towards_a_Single_Unified_Model_for_Effective_Detection_Segmentation_ICCV_2023_paper.html)

- **TransUNet: Rethinking the U-Net Architecture Design for Medical Image Segmentation through the Lens of Transformers** — reframes dense medical segmentation as mask classification with learnable queries refined by coarse-to-fine masked cross-attention. [![Paper](https://img.shields.io/badge/Paper-MedIA%202024-1f77b4.svg)](https://doi.org/10.1016/j.media.2024.103280)

- **APEx: Anatomy-Guided Pathology Segmentation** — decodes anatomy queries and pathology queries separately, then mixes anatomy-query representations into the pathology decoder as a clinically motivated structural prior. [![Paper](https://img.shields.io/badge/Paper-MICCAI%202024-1f77b4.svg)](https://papers.miccai.org/miccai-2024/079-Paper1464.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/alexanderjaus/APEx)

- **QueryNet: A Unified Framework for Accurate Polyp Segmentation and Detection** — treats object queries as cluster centers and reuses their instance information to couple polyp mask prediction with detection. [![Paper](https://img.shields.io/badge/Paper-MICCAI%202024-1f77b4.svg)](https://papers.miccai.org/miccai-2024/634-Paper1037.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/JiaxingChai/Query_Net)

- **IAUNet: Instance-Aware U-Net** — combines U-Net features with object-specific learnable queries and a lightweight Transformer decoder for biomedical instance segmentation. [![Paper](https://img.shields.io/badge/Paper-CVPRW%202025-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2025W/CVMI/html/Prytula_IAUNet_Instance-Aware_U-Net_CVPRW_2025_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/SlavkoPrytula/IAUNet)

- **STAN-LOC: Visual Query-Based Video Clip Localization for Fetal Ultrasound Sweep Videos** — formulates clinically relevant ultrasound clip retrieval around a visual query, transferring query-based localization to long medical sweep videos. [![Paper](https://img.shields.io/badge/Paper-MICCAI%202024-1f77b4.svg)](https://papers.miccai.org/miccai-2024/)

- **QMaxViT-Unet+: Query-Based Transformer Decoder for Medical Image Segmentation** — introduces a query-driven decoder on top of hybrid MaxViT/U-Net features for structure-aware medical mask prediction. [![Paper](https://img.shields.io/badge/Paper-CBIM%202025-1f77b4.svg)](https://www.sciencedirect.com/science/article/pii/S0010482525001429)

---

## Robotics, Embodied AI, and Action Prediction

<em>Queries can represent future action steps or provide compact latent interfaces between perception and control.</em>

- **ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware** — the Transformer decoder uses a fixed sequence of action-query positions to predict a chunk of future robot actions in parallel. [![Paper](https://img.shields.io/badge/Paper-RSS%202023-1f77b4.svg)](https://roboticsproceedings.org/rss19/p016.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/tonyzhaozh/act)

- **Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos** — uses learnable action queries during post-training to represent predicted action chunks for downstream manipulation. [![Paper](https://img.shields.io/badge/Paper-ICML%202026-1f77b4.svg)](https://research.beingbeyond.com/being-h0)

- **Perceiver-Actor (PerAct): A Multi-Task Transformer for Robotic Manipulation** — uses a small learned latent array to query millions of voxelized RGB-D features and decode discretized 6-DoF actions. [![Paper](https://img.shields.io/badge/Paper-CoRL%202022-1f77b4.svg)](https://proceedings.mlr.press/v205/shridhar23a.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/peract/peract)

- **Coarse-to-Fine Q-Attention: Efficient Learning for Visual Robotic Manipulation via Discretisation** — recursively queries increasingly fine spatial regions with Q-attention to localize manipulation actions efficiently. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/James_Coarse-to-Fine_Q-Attention_Efficient_Learning_for_Visual_Robotic_Manipulation_via_Discretisation_CVPR_2022_paper.html)

- **Query6DoF: Learning Sparse Queries as Implicit Shape Prior for Category-Level 6DoF Pose Estimation** — sparse category-specific queries act as compact object-shape priors and retrieve task-relevant 3D evidence for pose-aware manipulation. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Query6DoF_Learning_Sparse_Queries_as_Implicit_Shape_Prior_for_Category-Level_ICCV_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/hustvl/Query6DoF)

- **UniAD: Planning-Oriented Autonomous Driving** — connects perception, tracking, mapping, motion, occupancy, and planning through unified task queries that carry knowledge into the final planner. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/OpenDriveLab/UniAD)

- **MTR: Motion Transformer with Global Intention Localization and Local Movement Refinement** — represents distinct future behavior modes as learnable intention queries and refines each query into one trajectory hypothesis. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202022-1f77b4.svg)](https://papers.neurips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/sshaoshuai/MTR)

- **MTR++: Multi-Agent Motion Prediction with Symmetric Scene Modeling and Guided Intention Querying** — lets intention queries interact across multiple agents and guide one another toward socially coherent future modes. [![Paper](https://img.shields.io/badge/Paper-TPAMI%202024-1f77b4.svg)](https://arxiv.org/abs/2306.17770) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/sshaoshuai/MTR)

---

## Query Mechanisms, Efficiency, and Analysis

<em>Papers that study the query itself as the main methodological object rather than simply inheriting queries from a downstream architecture.</em>

- **Learned Queries for Efficient Local Attention** — introduces learned local aggregation queries in QnA, replacing expensive sliding-window self-attention with efficient query-and-attend operations. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Arar_Learned_Queries_for_Efficient_Local_Attention_CVPR_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/moabarar/qna)

- **Referencing Where to Focus: Improving Visual Grounding with Referential Query** — explicitly studies the weakness of vanilla randomly initialized queries and replaces them with target-aware referential queries. [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202024-1f77b4.svg)](https://papers.nips.cc/paper_files/paper/2024/hash/54c67d3db2df24a31cf045525f9460b9-Abstract-Conference.html)

- **PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection** — analyzes long-tail query activation and proposes pattern-based dynamic query generation to improve query utilization and optimization balance. [![Paper](https://img.shields.io/badge/Paper-CVPR%202026-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_PaQ-DETR_Learning_Pattern_and_Quality-Aware_Dynamic_Queries_for_Object_Detection_CVPR_2026_paper.html)

- **Learnable Query-Enhanced Pose Transformation** — uses learnable queries to fuse multi-scale source-image features for pose-guided human image generation. [![Paper](https://img.shields.io/badge/Paper-WACV%202026-1f77b4.svg)](https://openaccess.thecvf.com/content/WACV2026/html/Wang_Learnable_Query-Enhanced_Pose_Transformation_WACV_2026_paper.html)

- **Anchor DETR: Query Design for Transformer-Based Detector** — explicitly studies how spatial anchor points should parameterize object queries and demonstrates that query geometry is a major design axis. [![Paper](https://img.shields.io/badge/Paper-AAAI%202022-1f77b4.svg)](https://arxiv.org/abs/2109.07107) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/megvii-research/AnchorDETR)

- **DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR** — makes query position explicit as a dynamically refined box and separates content from geometry. [![Paper](https://img.shields.io/badge/Paper-ICLR%202022-1f77b4.svg)](https://arxiv.org/abs/2201.12329) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/IDEA-Research/DAB-DETR)

- **DN-DETR: Accelerate DETR Training by Introducing Query DeNoising** — directly supervises noisy ground-truth queries and shows query denoising can stabilize bipartite matching. [![Paper](https://img.shields.io/badge/Paper-CVPR%202022-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Li_DN-DETR_Accelerate_DETR_Training_by_Introducing_Query_DeNoising_CVPR_2022_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/IDEA-Research/DN-DETR)

- **Group DETR: Fast DETR Training with Group-Wise One-to-Many Assignment** — partitions queries into multiple groups and independently matches each group, increasing positive query supervision while retaining normal inference. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Group_DETR_Fast_DETR_Training_with_Group-Wise_One-to-Many_Assignment_ICCV_2023_paper.html)

- **DDQ: Dense Distinct Query for End-to-End Object Detection** — studies how to create many high-quality candidates but keep decoder queries distinct enough for one-to-one set prediction. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Dense_Distinct_Query_for_End-to-End_Object_Detection_CVPR_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/jshilong/DDQ)

- **SQR: Enhanced Training of Query-Based Object Detection via Selective Query Recollection** — reuses selected intermediate queries as training-only auxiliary queries and directly analyzes which decoder queries are worth recollecting. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Enhanced_Training_of_Query-Based_Object_Detection_via_Selective_Query_Recollection_CVPR_2023_paper.html)

- **DEQDet: Deep Equilibrium Object Detection** — turns repeated decoder-query refinement into an implicit equilibrium process, offering a different view of query depth and convergence. [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Deep_Equilibrium_Object_Detection_ICCV_2023_paper.html)

- **FastInst: A Simple Query-Based Model for Real-Time Instance Segmentation** — introduces instance-activation-guided query initialization and shows that content-adaptive query initialization can reduce decoder depth. [![Paper](https://img.shields.io/badge/Paper-CVPR%202023-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/He_FastInst_A_Simple_Query-Based_Model_for_Real-Time_Instance_Segmentation_CVPR_2023_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/junjiehe96/FastInst)

- **BoQ: A Place is Worth a Bag of Learnable Queries** — provides a clean study of globally shared learned queries as stable cross-attention aggregation operators rather than prediction slots. [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Ali-bey_BoQ_A_Place_is_Worth_a_Bag_of_Learnable_Queries_CVPR_2024_paper.html) [![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/amaralibey/Bag-of-Queries)

- **SimCIS: Rethinking Query-Based Transformer for Continual Image Segmentation** — studies query objectness, feature-to-query assignment, and visual-query replay under continual learning. [![Paper](https://img.shields.io/badge/Paper-CVPR%202025-1f77b4.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Rethinking_Query-based_Transformer_for_Continual_Image_Segmentation_CVPR_2025_paper.html)

---

## A Query-Centric Evolution

```text
                           DETR
                    Learnable Object Query
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   Spatial Query       Mask Query        Track Query
 DAB / Deformable    MaskFormer         MOTR / TrackFormer
        │                  │                  │
        ▼                  ▼                  ▼
 3D / BEV Query      Semantic Query     Persistent Memory
DETR3D / BEVFormer  Query2Label        StreamPETR
        │                  │                  │
        ├──────────────────┼──────────────────┤
        ▼                  ▼                  ▼
 Intention Query      Latent Query       Domain Query
      MTR          Perceiver / Q-Former  Medical / Science
                           │
                           ▼
                   Dynamic / Adaptive Query
                  FastInst / RefFormer /
                        PaQ-DETR
```

---

## Future Trends and Hot Topics

### 🧠 Dynamic and Input-Adaptive Queries
Moving from a globally shared static query bank toward **sample-specific query composition, query proposal, query routing, and query pruning**.

### 🧬 Semantically Grounded Queries
Replacing anonymous embeddings with explicit concepts such as **classes, anatomy, tissues, pathways, phenotypes, actions, or clinical biomarkers**.

### 🔁 Persistent Queries as Memory
Track queries demonstrate that queries can carry state through time. The same idea can extend to **long video, agents, world models, longitudinal medicine, and interactive systems**.

### 🧩 Queries as Universal Modality Interfaces
Perceiver, Flamingo, and Q-Former suggest a general architecture:

```text
Large Modality Encoder
        ↓
Small Query Bottleneck
        ↓
Another Foundation Model
```

### ⚡ Query-Efficient Computation
Future work may optimize **how many queries exist, when they activate, where they attend, how long they survive, and which queries can be merged or dropped**.

### 🔬 Scientific and Interpretable Queries
Medical and scientific AI can give queries explicit semantics, e.g.:

- tissue queries
- cell-type queries
- gene / pathway queries
- biomarker queries
- treatment-response queries
- anatomical queries
- molecular phenotype queries
- expert / hypothesis queries

### 🤖 Query-based Agents
A reasoning agent could maintain a bank of persistent learnable or dynamically generated queries corresponding to unresolved goals, hypotheses, tools, observations, or sub-tasks.

---

## Contributing

Contributions are welcome.

Please use the following format:

```markdown
- **Paper Title** — one-sentence description of what the query represents and how it is used. [![Paper](...)](...) [![Code](...)](...)
```

Before submitting a PR, please answer:

1. **What exactly is the query?**
2. **Is it learnable, dynamic, semantic, geometric, temporal, or input-derived?**
3. **What feature bank does it query?**
4. **What does the updated query represent?**
5. **Is query design a core contribution or inherited from a baseline?**

### Suggested Tags

```text
query:learnable
query:dynamic
query:geometric
query:semantic
query:temporal
query:latent
query:multimodal
query:medical
query:audio
query:nlp
query:robotics
```

---

## Acknowledgements

This repository follows the organization and visual style of **Awesome-AI4DigitalPathology** and is inspired by the open-source efforts behind DETR, Perceiver, MaskFormer / Mask2Former, MOTR, BEVFormer, MTR, BLIP-2, PIQN, and query-based medical AI.

We sincerely thank all authors and maintainers of the papers and repositories collected here.

---

## Citation

If this repository is helpful to your research, please consider citing or linking it:

```bibtex
@misc{awesome_learnable_q,
  title        = {Awesome-Learnable-Q: A Curated List of Learnable Query and Query-based Models across AI},
  author       = {Contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/lingxitong/Awesome-Learnable-Q}}
}
```


