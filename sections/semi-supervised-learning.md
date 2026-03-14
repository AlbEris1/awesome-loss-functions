# Semi-Supervised Learning & Self-Training Loss Functions

> A chronological catalog of the most important loss functions in semi-supervised learning (SSL) and self-training, from foundational pseudo-labeling to modern adaptive thresholding methods.

---

## 1. Pseudo-Label Loss (2013)

**Pseudo-Label Loss** (2013) — Trains on unlabeled data using the model's own highest-confidence class predictions as hard pseudo-labels via standard cross-entropy.

📄 [Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks](https://citeseerx.ist.psu.edu/document?doi=798d9840d2439a0e5d47bcf5d164aa46d5e7dc26) — Dong-Hyun Lee
💻 [microsoft/Semi-supervised-learning (USB)](https://github.com/microsoft/Semi-supervised-learning)

---

## 2. Π-Model / Temporal Ensemble Loss (2017)

**Π-Model & Temporal Ensemble Loss** (2017) — Enforces consistency by minimizing the MSE between predictions of the same input under different stochastic augmentations/dropout; Temporal Ensembling aggregates predictions over training epochs via exponential moving average.

📄 [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/abs/1610.02242) — Samuli Laine, Timo Aila
💻 [tensorfreitas/Temporal-Ensembling-for-Semi-Supervised-Learning](https://github.com/tensorfreitas/Temporal-Ensembling-for-Semi-Supervised-Learning)

---

## 3. Mean Teacher Loss (2017)

**Mean Teacher Loss** (2017) — Improves on Temporal Ensembling by maintaining an exponential moving average (EMA) of model weights as a "teacher" and minimizing the consistency loss (MSE) between student and teacher predictions.

📄 [Mean Teachers Are Better Role Models: Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results](https://arxiv.org/abs/1703.01780) — Antti Tarvainen, Harri Valpola
💻 [CuriousAI/mean-teacher](https://github.com/CuriousAI/mean-teacher)

---

## 4. Virtual Adversarial Training (VAT) Loss (2018)

**VAT Loss** (2018) — Regularizes the model by computing the KL divergence between predictions on clean inputs and adversarially perturbed inputs, smoothing the output distribution in the most sensitive directions without requiring labels.

📄 [Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning](https://arxiv.org/abs/1704.03976) — Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Shin Ishii
💻 [lyakaap/VAT-pytorch](https://github.com/lyakaap/VAT-pytorch)

---

## 5. MixMatch Loss (2019)

**MixMatch Loss** (2019) — Unifies consistency regularization, entropy minimization, and MixUp into a single holistic loss: guesses low-entropy pseudo-labels for augmented unlabeled data, mixes labeled and unlabeled examples, and combines supervised cross-entropy with an L2 consistency term.

📄 [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249) — David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, Colin Raffel
💻 [YU1ut/MixMatch-pytorch](https://github.com/YU1ut/MixMatch-pytorch)

---

## 6. Unsupervised Data Augmentation (UDA) Loss (2020)

**UDA Loss** (2020) — Enforces consistency between a model's prediction on an original unlabeled example and its prediction on a strongly augmented version (via RandAugment/back-translation), combined with Training Signal Annealing (TSA) to prevent overfitting on limited labels.

📄 [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848) — Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, Quoc V. Le
💻 [google-research/uda](https://github.com/google-research/uda)

---

## 7. ReMixMatch Loss (2020)

**ReMixMatch Loss** (2020) — Extends MixMatch with distribution alignment (matching the marginal class distribution of predictions to the true label distribution) and augmentation anchoring (anchoring strongly-augmented predictions to weakly-augmented ones), plus a self-supervised rotation loss.

📄 [ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring](https://arxiv.org/abs/1911.09785) — David Berthelot, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang, Colin Raffel
💻 [google-research/remixmatch](https://github.com/google-research/remixmatch)

---

## 8. FixMatch Loss (2020)

**FixMatch Loss** (2020) — Combines pseudo-labeling with consistency regularization in a simple pipeline: generates a hard pseudo-label from a weakly-augmented image (retained only above a fixed confidence threshold), then trains the model to predict that label on a strongly-augmented version via cross-entropy.

📄 [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685) — Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, Colin Raffel
💻 [kekmodel/FixMatch-pytorch](https://github.com/kekmodel/FixMatch-pytorch)

---

## 9. Noisy Student Loss (2020)

**Noisy Student Loss** (2020) — Iterative self-training where a teacher generates pseudo-labels on unlabeled data, and a noisy student (with dropout, stochastic depth, and RandAugment) is trained on the combined labeled + pseudo-labeled data using standard cross-entropy; the student then becomes the next teacher.

📄 [Self-Training with Noisy Student Improves ImageNet Classification](https://arxiv.org/abs/1911.04252) — Qizhe Xie, Minh-Thang Luong, Eduard Hovy, Quoc V. Le
💻 [huggingface/pytorch-image-models (timm — noisy-student)](https://github.com/huggingface/pytorch-image-models)

---

## 10. FlexMatch Loss (2021)

**FlexMatch Loss** (2021) — Introduces Curriculum Pseudo Labeling (CPL) to FixMatch: dynamically adjusts per-class confidence thresholds based on each class's learning status, allowing easier classes to contribute pseudo-labels earlier and harder classes to use lower thresholds.

📄 [FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling](https://arxiv.org/abs/2110.08263) — Bowen Zhang, Yidong Wang, Wenxin Hou, Hao Wu, Jindong Wang, Manabu Okumura, Takahiro Shinozaki
💻 [TorchSSL/TorchSSL](https://github.com/TorchSSL/TorchSSL)

---

## 11. FreeMatch Loss (2023)

**FreeMatch Loss** (2023) — Replaces fixed or curriculum-based thresholds with a fully self-adaptive global and per-class thresholding scheme derived from the model's own confidence EMA, plus a self-adaptive class fairness regularization penalty to encourage diverse predictions.

📄 [FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning](https://arxiv.org/abs/2205.07246) — Yidong Wang, Hao Chen, Qiang Heng, Wenxin Hou, Yue Fan, Zhen Wu, Jindong Wang, Marios Savvides, Takahiro Shinozaki, Bhiksha Raj, Bernt Schiele, Xing Xie
💻 [microsoft/Semi-supervised-learning (USB)](https://github.com/microsoft/Semi-supervised-learning)

---

## 12. SoftMatch Loss (2023)

**SoftMatch Loss** (2023) — Addresses the quantity-quality trade-off in pseudo-labeling by replacing hard thresholding with a truncated Gaussian weighting function over confidence scores, enabling soft sample weighting that retains more unlabeled data while down-weighting low-quality pseudo-labels, plus uniform alignment for class balance.

📄 [SoftMatch: Addressing the Quantity-Quality Trade-off in Semi-supervised Learning](https://arxiv.org/abs/2301.10921) — Hao Chen, Ran Tao, Yue Fan, Yidong Wang, Jindong Wang, Bernt Schiele, Xing Xie, Bhiksha Raj, Marios Savvides
💻 [Hhhhhhao/SoftMatch](https://github.com/Hhhhhhao/SoftMatch)

---

## Unified Implementations

All of the above methods (and more) are available in the **USB (Unified Semi-Supervised Learning Benchmark)** framework:

💻 [microsoft/Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning) — PyTorch-based unified codebase covering FixMatch, FlexMatch, FreeMatch, SoftMatch, MixMatch, ReMixMatch, UDA, Mean Teacher, Pseudo-Label, and more.

---

## Timeline Summary

| Year | Method | Key Innovation |
|------|--------|---------------|
| 2013 | Pseudo-Label | Hard pseudo-labels from model predictions |
| 2017 | Π-Model / Temporal Ensemble | Consistency regularization via self-ensembling |
| 2017 | Mean Teacher | EMA of weights as teacher for consistency |
| 2018 | VAT | Adversarial perturbation-based smoothness regularization |
| 2019 | MixMatch | Unified consistency + entropy minimization + MixUp |
| 2020 | UDA | Strong augmentation-based consistency training |
| 2020 | ReMixMatch | Distribution alignment + augmentation anchoring |
| 2020 | FixMatch | Simple threshold-based pseudo-label + strong augmentation |
| 2020 | Noisy Student | Iterative self-training with noise injection |
| 2021 | FlexMatch | Per-class adaptive thresholds (curriculum learning) |
| 2023 | FreeMatch | Self-adaptive thresholding from model confidence EMA |
| 2023 | SoftMatch | Truncated Gaussian soft weighting instead of hard thresholds |
