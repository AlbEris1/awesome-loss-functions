# Adversarial Robustness, Adversarial Training & Out-of-Distribution Detection — Loss Functions

> A chronological catalog of the most important loss functions and scoring methods used in adversarial robustness, adversarial training (AT), and out-of-distribution (OOD) detection.

---

## Part I — Adversarial Training & Adversarial Robustness

---

**1. FGSM Adversarial Training Loss** (2015) — Augments training with single-step adversarial examples generated via the fast gradient sign of the loss.
📄 [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) — Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy
💻 [Implementation (torchattacks)](https://github.com/Harry24k/adversarial-attacks-pytorch)

---

**2. PGD Adversarial Training Loss (Madry AT)** (2018) — Formulates adversarial training as robust optimization, using multi-step PGD to find worst-case perturbations.
📄 [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) — Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu
💻 [Implementation (Adversarial Training Pytorch)](https://github.com/LetheSec/Adversarial_Training_Pytorch)

---

**3. Adversarial Logit Pairing (ALP) Loss** (2018) — Penalizes the L2 distance between logits of clean and adversarial example pairs to encourage invariance.
📄 [Adversarial Logit Pairing](https://arxiv.org/abs/1803.06373) — Harini Kannan, Alexey Kurakin, Ian Goodfellow
💻 [Implementation (analysis repo)](https://github.com/labsix/adversarial-logit-pairing-analysis)

---

**4. TRADES Loss** (2019) — Decomposes robust error into natural error + boundary error; minimizes CE on clean data plus KL divergence between clean and adversarial predictions.
📄 [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/abs/1901.08573) — Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, Michael I. Jordan
💻 [Implementation (official)](https://github.com/yaodongyu/TRADES)

---

**5. Free Adversarial Training Loss** (2019) — Recycles gradient computations from the backward pass to simultaneously update model weights and craft adversarial perturbations at zero extra cost.
📄 [Adversarial Training for Free!](https://arxiv.org/abs/1904.12843) — Ali Shafahi, Mahyar Najibi, Amin Ghiasi, Zheng Xu, John Dickerson, Christoph Studer, Larry S. Davis, Gavin Taylor, Tom Goldstein
💻 [Implementation (PyTorch)](https://github.com/mahyarnajibi/FreeAdversarialTraining)

---

**6. Certified Robustness via Randomized Smoothing Loss** (2019) — Trains a base classifier under Gaussian noise to provide provable ℓ₂ robustness certificates via smoothing.
📄 [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/abs/1902.02918) — Jeremy M. Cohen, Elan Rosenfeld, J. Zico Kolter
💻 [Implementation (official)](https://github.com/locuslab/smoothing)

---

**7. MART Loss** (2020) — Explicitly differentiates misclassified and correctly classified examples during adversarial training via a boosted cross-entropy + KL regularization.
📄 [Improving Adversarial Robustness Requires Revisiting Misclassified Examples](https://openreview.net/forum?id=rklOg6EFwS) — Yisen Wang, Difan Zou, Jinfeng Yi, James Bailey, Xingjun Ma, Quanquan Gu
💻 [Implementation (official)](https://github.com/YisenWang/MART)

---

**8. AWP Loss (Adversarial Weight Perturbation)** (2020) — Applies a double perturbation: adversarial input perturbation + adversarial weight perturbation to flatten the weight loss landscape for better robust generalization.
📄 [Adversarial Weight Perturbation Helps Robust Generalization](https://arxiv.org/abs/2004.05884) — Dongxian Wu, Shu-Tao Xia, Yisen Wang
💻 [Implementation (official)](https://github.com/csdongxian/AWP)

---

**9. LBGAT Loss (Learnable Boundary Guided AT)** (2021) — Uses logits from a clean-trained model to guide the robust model's decision boundary, preserving natural accuracy while improving robustness.
📄 [Learnable Boundary Guided Adversarial Training](https://arxiv.org/abs/2011.11164) — Jiequan Cui, Shu Liu, Liwei Wang, Jiaya Jia
💻 [Implementation (official)](https://github.com/JIA-Lab-research/LBGAT)

---

**10. DKL Loss (Decoupled Kullback-Leibler Divergence)** (2024) — Proves KL divergence equals a decoupled form and proposes improved KL/DKL loss variants that achieve state-of-the-art adversarial robustness.
📄 [Decoupled Kullback-Leibler Divergence Loss](https://arxiv.org/abs/2305.13948) — Jiequan Cui, Zhuotao Tian, Zhisheng Zhong, Xiaojuan Qi, Bei Yu, Hanwang Zhang
💻 [Implementation (official)](https://github.com/jiequancui/DKL)

---

## Part II — Out-of-Distribution (OOD) Detection

---

**11. MSP Baseline (Maximum Softmax Probability)** (2017) — Uses the maximum softmax probability as a simple baseline score for detecting misclassified and OOD examples.
📄 [A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks](https://arxiv.org/abs/1610.02136) — Dan Hendrycks, Kevin Gimpel
💻 [Implementation (official)](https://github.com/hendrycks/error-detection)

---

**12. ODIN Score** (2018) — Enhances OOD detection by applying temperature scaling to softmax outputs and adding small input perturbations, without retraining.
📄 [Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks](https://arxiv.org/abs/1706.02690) — Shiyu Liang, Yixuan Li, R. Srikant
💻 [Implementation (official, PyTorch)](https://github.com/ShiyuLiang/odin-pytorch)

---

**13. Mahalanobis Distance-based OOD Score** (2018) — Fits class-conditional Gaussians to intermediate features and uses the Mahalanobis distance as a unified score for OOD and adversarial detection.
📄 [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/abs/1807.03888) — Kimin Lee, Kibok Lee, Honglak Lee, Jinwoo Shin
💻 [Implementation (official)](https://github.com/pokaxpoka/deep_Mahalanobis_detector)

---

**14. Outlier Exposure (OE) Loss** (2019) — Trains the model against an auxiliary dataset of outliers, forcing the network to produce uniform predictions on OOD data.
📄 [Deep Anomaly Detection with Outlier Exposure](https://arxiv.org/abs/1812.04606) — Dan Hendrycks, Mantas Mazeika, Thomas Dietterich
💻 [Implementation (official)](https://github.com/hendrycks/outlier-exposure)

---

**15. Energy-based OOD Loss** (2020) — Replaces softmax confidence with a theoretically grounded energy score derived from the log-sum-exp of logits; optionally fine-tunes with an energy-bounded margin loss.
📄 [Energy-based Out-of-distribution Detection](https://arxiv.org/abs/2010.03759) — Weitang Liu, Xiaoyun Wang, John Owens, Yixuan Li
💻 [Implementation (official)](https://github.com/wetliu/energy_ood)

---

**16. CSI Loss (Contrasting Shifted Instances)** (2020) — Combines contrastive learning with distributionally shifted augmentations to learn representations that separate in-distribution from OOD data.
📄 [CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances](https://arxiv.org/abs/2007.08176) — Jihoon Tack, Sangwoo Mo, Jongheon Jeong, Jinwoo Shin
💻 [Implementation (official)](https://github.com/alinlab/CSI)

---

**17. ATOM Loss (Adversarial Training with Outlier Mining)** (2021) — Mines the most informative outliers from an auxiliary dataset via adversarial selection, then trains an OOD detector that is robust to adversarial manipulation.
📄 [ATOM: Robustifying Out-of-distribution Detection Using Outlier Mining](https://arxiv.org/abs/2006.15207) — Jiefeng Chen, Yixuan Li, Xi Wu, Yingyu Liang, Somesh Jha
💻 [Implementation (official)](https://github.com/jfc43/informative-outlier-mining)

---

**18. SSD Loss (Self-Supervised Outlier Detection)** (2021) — Unifies self-supervised contrastive representation learning with Mahalanobis distance-based OOD scoring, requiring only unlabeled in-distribution data.
📄 [SSD: A Unified Framework for Self-Supervised Outlier Detection](https://arxiv.org/abs/2103.12051) — Vikash Sehwag, Mung Chiang, Prateek Mittal
💻 [Implementation (official)](https://github.com/inspire-group/SSD)

---

**19. ReAct (Rectified Activations)** (2021) — Truncates penultimate-layer activations at a threshold to reduce model overconfidence on OOD data, compatible with any post-hoc scoring function.
📄 [ReAct: Out-of-distribution Detection With Rectified Activations](https://proceedings.neurips.cc/paper/2021/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html) — Yiyou Sun, Chuan Guo, Yixuan Li
💻 [Implementation (official)](https://github.com/deeplearning-wisc/react)

---

**20. VOS Loss (Virtual Outlier Synthesis)** (2022) — Synthesizes virtual outliers in the feature space from class-conditional Gaussians and trains with an energy-bounded binary loss for OOD-aware learning without real OOD data.
📄 [VOS: Learning What You Don't Know by Virtual Outlier Synthesis](https://arxiv.org/abs/2202.01197) — Xuefeng Du, Zhaoning Wang, Mu Cai, Yixuan Li
💻 [Implementation (official)](https://github.com/deeplearning-wisc/vos)

---

**21. KNN-based OOD Score** (2022) — Uses the k-th nearest neighbor distance in the feature space as a non-parametric OOD score, avoiding Gaussian assumptions entirely.
📄 [Out-of-Distribution Detection with Deep Nearest Neighbors](https://arxiv.org/abs/2204.06507) — Yiyou Sun, Yifei Ming, Xiaojin Zhu, Yixuan Li
💻 [Implementation (official)](https://github.com/deeplearning-wisc/knn-ood)

---

**22. CIDER Loss (Compactness and Dispersion)** (2023) — Optimizes hyperspherical embeddings with a combined compactness loss (pulling ID samples to class prototypes) and dispersion loss (pushing prototypes apart) for OOD detection.
📄 [How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection](https://arxiv.org/abs/2203.04450) — Yifei Ming, Yiyou Sun, Ousmane Dia, Yixuan Li
💻 [Implementation (official)](https://github.com/deeplearning-wisc/cider)

---

## Unified Libraries

| Library | Description | Link |
|---------|-------------|------|
| **pytorch-ood** | Comprehensive OOD detection library (MSP, ODIN, Energy, Mahalanobis, etc.) | [kkirchheim/pytorch-ood](https://github.com/kkirchheim/pytorch-ood) |
| **Adversarial Training Pytorch** | Unified AT library (PGD-AT, TRADES, MART, LBGAT, AWP) | [LetheSec/Adversarial_Training_Pytorch](https://github.com/LetheSec/Adversarial_Training_Pytorch) |
| **torchattacks** | PyTorch adversarial attacks library (FGSM, PGD, CW, etc.) | [Harry24k/adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch) |
| **Adversarial Robustness Toolbox (ART)** | IBM's comprehensive robustness toolkit | [Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) |
