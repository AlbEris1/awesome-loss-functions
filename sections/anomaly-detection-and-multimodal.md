# Loss Functions: Anomaly Detection & Multi-Modal Learning

> Comprehensive chronological catalog of loss functions used in anomaly detection and multi-modal learning (beyond CLIP). Each entry includes paper, authors, arXiv link, and PyTorch implementation.

---

## Part I — Anomaly Detection

---

**Autoencoder Reconstruction Loss** (2003→ongoing) — Measures pixel/feature-level reconstruction error (MSE/L1) between input and autoencoder output; anomalies yield high reconstruction error.
📄 [Reducing the Dimensionality of Data with Neural Networks](https://www.science.org/doi/10.1126/science.1127647) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (Science, 2006; concept dates to earlier work)
💻 [PyTorch Autoencoder for Anomaly Detection (anomalib)](https://github.com/open-edge-platform/anomalib)

---

**Deep SVDD Loss** (2018) — Maps data into a hypersphere of minimum volume; the loss minimizes the distance of all network outputs to the center, making outliers detectable as points far from the center.
📄 [Deep One-Class Classification](https://arxiv.org/abs/1802.04365) — Lukas Ruff, Robert A. Vandermeulen, Nico Görnitz, Lucas Deecke, Shoaib A. Siddiqui, Alexander Binder, Emmanuel Müller, Marius Kloft (ICML 2018)
💻 [lukasruff/Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch)

---

**DAGMM Loss** (2018) — Jointly optimizes a deep autoencoder reconstruction loss and a Gaussian Mixture Model energy-based loss for end-to-end density estimation and anomaly detection.
📄 [Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection](https://arxiv.org/abs/1809.04758) — Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Dae-ki Cho, Haifeng Chen (ICLR 2018)
💻 [mperezcarrasco/PyTorch-DAGMM](https://github.com/mperezcarrasco/PyTorch-DAGMM)

---

**f-AnoGAN Loss** (2019) — Combines image-space reconstruction loss and discriminator feature-space loss to score anomalies via a trained GAN encoder, enabling fast inference without iterative optimization.
📄 [f-AnoGAN: Fast Unsupervised Anomaly Detection with Generative Adversarial Networks](https://doi.org/10.1016/j.media.2019.01.010) — Thomas Schlegl, Philipp Seeböck, Sebastian M. Waldstein, Georg Langegger, Ursula Schmidt-Erfurth (Medical Image Analysis, 2019)
💻 [A03ki/f-AnoGAN](https://github.com/A03ki/f-AnoGAN)

---

**CutPaste Loss** (2021) — Self-supervised contrastive loss that learns representations by classifying images with cut-and-pasted rectangular patches from anomaly-free data; anomalies are detected via density estimation on learned embeddings.
📄 [CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](https://arxiv.org/abs/2104.04015) — Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon, Tomas Pfister (CVPR 2021)
💻 [LilitYolyan/CutPaste](https://github.com/LilitYolyan/CutPaste)

---

**DRAEM Loss** (2021) — Discriminatively trained reconstruction-anomaly embedding that combines a reconstructive sub-network loss (L2 + SSIM) with a discriminative segmentation loss (focal loss) on synthetically generated anomalies.
📄 [DRÆM – A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection](https://arxiv.org/abs/2108.07610) — Vitjan Zavrtanik, Matej Kristan, Danijel Skočaj (ICCV 2021)
💻 [VitjanZ/DRAEM](https://github.com/VitjanZ/DRAEM)

---

**ALIGN Loss** (2021) — Dual-encoder contrastive loss (normalized softmax) trained on 1.8B noisy image-alt-text pairs; scales CLIP-style image-text alignment without expensive data curation.
📄 [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918) — Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig (ICML 2021)
💻 [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) (includes ALIGN-style training)

---

**PatchCore** (2022) — Not a traditional loss; uses a coreset-subsampled memory bank of nominal patch-level features from a pretrained backbone, detecting anomalies via nearest-neighbor distance at test time.
📄 [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265) — Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler (CVPR 2022)
💻 [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)

---

**Reverse Distillation Loss** (2022) — Teacher-student knowledge distillation where a student decoder reconstructs multi-scale teacher encoder features; anomalies cause representation discrepancy between teacher and student outputs.
📄 [Anomaly Detection via Reverse Distillation from One-Class Embedding](https://arxiv.org/abs/2201.10703) — Hanqiu Deng, Xingyu Li (CVPR 2022)
💻 [hq-deng/RD4AD](https://github.com/hq-deng/RD4AD)

---

**BLIP Loss** (2022) — Multi-task vision-language loss combining image-text contrastive (ITC), image-text matching (ITM), and language modeling (LM) losses with a captioning-and-filtering (CapFilt) bootstrapping strategy.
📄 [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) — Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi (ICML 2022)
💻 [salesforce/BLIP](https://github.com/salesforce/BLIP)

---

**Classifier-Free Guidance** (2022) — Training-time technique that jointly trains conditional and unconditional diffusion models by randomly dropping conditioning; at inference, interpolates between both scores to trade off diversity and fidelity.
📄 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — Jonathan Ho, Tim Salimans (NeurIPS 2021 Workshop on DGMs)
💻 [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) (uses CFG at inference)

---

**LDM / Stable Diffusion Loss** (2022) — Denoising diffusion loss applied in a learned latent space (via a pretrained autoencoder), dramatically reducing computational cost while preserving synthesis quality; cross-attention conditions on text/layout.
📄 [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer (CVPR 2022)
💻 [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) / [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)

---

**DALL·E 2 (unCLIP) Loss** (2022) — Two-stage loss: a diffusion prior that generates CLIP image embeddings from text, plus a diffusion decoder that generates images conditioned on those embeddings; trained with MSE on CLIP embeddings and standard diffusion loss.
📄 [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125) — Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen (arXiv 2022)
💻 [lucidrains/DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch)

---

**Imagen Loss** (2022) — Cascaded diffusion loss using a frozen T5-XXL text encoder; trains a base 64×64 diffusion model plus two super-resolution diffusion models (64→256→1024) with standard ε-prediction denoising loss.
📄 [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487) — Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, et al. (NeurIPS 2022)
💻 [lucidrains/imagen-pytorch](https://github.com/lucidrains/imagen-pytorch)

---

**SimpleNet Loss** (2023) — Adds Gaussian noise to pretrained features to simulate anomalies, then trains a lightweight discriminator with binary cross-entropy to distinguish normal from anomalous feature distributions.
📄 [SimpleNet: A Simple Network for Image Anomaly Detection and Localization](https://arxiv.org/abs/2303.15140) — Zhikang Liu, Yiming Zhou, Yuansheng Xu, Zilei Wang (CVPR 2023)
💻 [DonaldRR/SimpleNet](https://github.com/DonaldRR/SimpleNet)

---

**BLIP-2 Loss** (2023) — Three-stage loss via a lightweight Q-Former: (1) image-text contrastive loss, (2) image-grounded text generation loss, and (3) image-text matching loss — bridging frozen image encoders and frozen LLMs.
📄 [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) — Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi (ICML 2023)
💻 [salesforce/LAVIS](https://github.com/salesforce/LAVIS)

---

**ImageBind Loss** (2023) — InfoNCE contrastive loss that binds six modalities (image, text, audio, depth, thermal, IMU) into a single embedding space using image-paired data only, leveraging CLIP's image-text alignment as an anchor.
📄 [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665) — Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra (CVPR 2023)
💻 [facebookresearch/ImageBind](https://github.com/facebookresearch/ImageBind)

---

## Quick Reference Table

| # | Loss / Method | Year | Domain | Key Idea |
|---|---|---|---|---|
| 1 | Autoencoder Reconstruction | ~2006 | AD | High reconstruction error = anomaly |
| 2 | Deep SVDD | 2018 | AD | Minimize hypersphere volume around normal data |
| 3 | DAGMM | 2018 | AD | Autoencoder + GMM density estimation |
| 4 | f-AnoGAN | 2019 | AD | GAN encoder + dual anomaly score |
| 5 | CutPaste | 2021 | AD | Self-supervised contrastive with synthetic defects |
| 6 | DRAEM | 2021 | AD | Reconstruction + discriminative segmentation |
| 7 | ALIGN | 2021 | MM | CLIP-style contrastive on 1.8B noisy pairs |
| 8 | PatchCore | 2022 | AD | Memory bank + nearest-neighbor scoring |
| 9 | Reverse Distillation | 2022 | AD | Teacher-student feature discrepancy |
| 10 | BLIP | 2022 | MM | ITC + ITM + LM with CapFilt bootstrap |
| 11 | Classifier-Free Guidance | 2022 | MM | Conditional/unconditional score interpolation |
| 12 | LDM / Stable Diffusion | 2022 | MM | Denoising loss in latent space |
| 13 | DALL·E 2 (unCLIP) | 2022 | MM | CLIP prior + diffusion decoder |
| 14 | Imagen | 2022 | MM | Cascaded diffusion + frozen T5 encoder |
| 15 | SimpleNet | 2023 | AD | Noise injection + feature discriminator |
| 16 | BLIP-2 | 2023 | MM | Q-Former bridging frozen encoders + LLMs |
| 17 | ImageBind | 2023 | MM | Six-modality InfoNCE via image anchor |

*AD = Anomaly Detection, MM = Multi-Modal Learning*
