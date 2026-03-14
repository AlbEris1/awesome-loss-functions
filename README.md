# Awesome Loss Functions [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)](https://github.com/stabgan/awesome-loss-functions/issues)

A comprehensive, chronologically ordered collection of loss functions across all subdomains of deep learning and machine learning — with paper links, one-line descriptions, mathematical formulations, and implementation references.

**350+ loss functions. 25+ categories. Every subdomain of AI.**

> If this resource helps your research or engineering work, please consider giving it a ⭐

---

## What's New

- 🔊 **Audio, Music & Speech Generation** — WaveNet to Stable Audio, 19 losses
- 🎬 **Video Generation & Understanding** — VGAN to VideoPoet, 20 losses
- ⏳ **Time Series Forecasting** — Pinball Loss to TimesFM, 23 losses
- 🧠 **Continual & Lifelong Learning** — EWC to EASE, 18 methods
- ⚖️ **Calibration, Fairness & Bias Mitigation** — Brier Score to Group DRO, 18 losses
- 🛡️ **Adversarial Robustness & OOD Detection** — FGSM-AT to CIDER, 22 losses
- 🔍 **Anomaly Detection & Multi-Modal Learning** — Deep SVDD to ImageBind, 17 losses
- 🖼️ **Image-to-Image Translation** — Total Variation to DoveNet, 16 losses
- 📐 **Semi-Supervised Learning** — Pseudo-Label to SoftMatch, 12 losses
- 🎯 **Optical Flow, Video & Pose** — Horn-Schunck to SEA-RAFT, 33 losses

---

## Contents

**Core Categories (inline)**

- [Loss Selection Guide](#-loss-selection-guide)
- [Key Mathematical Formulations](#-key-mathematical-formulations)
- [Classification](#classification)
- [Regression](#regression)
- [Segmentation](#segmentation)
- [Object Detection (Bounding Box)](#object-detection-bounding-box)
- [Generative Models — GANs](#generative-models--gans)
- [Generative Models — VAEs](#generative-models--vaes)
- [Generative Models — Diffusion & Flow](#generative-models--diffusion--flow)
- [Reconstruction & Perceptual](#reconstruction--perceptual)
- [Image Super-Resolution & Restoration](#image-super-resolution--restoration)
- [Contrastive & Self-Supervised Learning](#contrastive--self-supervised-learning)
- [Metric Learning & Face Recognition](#metric-learning--face-recognition)
- [NLP & Language Modeling](#nlp--language-modeling)
- [LLM Alignment (RLHF / DPO)](#llm-alignment-rlhf--dpo)
- [Sequence-to-Sequence & Speech](#sequence-to-sequence--speech)
- [Reinforcement Learning](#reinforcement-learning)
- [Knowledge Distillation](#knowledge-distillation)
- [Regularization](#regularization)
- [3D Vision & Point Clouds](#3d-vision--point-clouds)
- [Depth Estimation](#depth-estimation)
- [Medical Imaging](#medical-imaging)
- [Graph Neural Networks](#graph-neural-networks)
- [Recommendation Systems](#recommendation-systems)
- [Multi-Task Learning](#multi-task-learning)
- [Uncertainty Estimation](#uncertainty-estimation)
- [Domain Adaptation](#domain-adaptation)

**Extended Categories (separate files)**

- [Audio, Music & Speech Generation](sections/audio-music-speech.md) — 19 losses
- [Video Generation & Understanding](sections/video-generation-understanding.md) — 20 losses
- [Time Series Forecasting](sections/time-series-forecasting.md) — 23 losses
- [Continual & Lifelong Learning](sections/continual-lifelong-learning.md) — 18 methods
- [Calibration, Fairness & Bias Mitigation](sections/calibration-fairness.md) — 18 losses
- [Adversarial Robustness & OOD Detection](sections/adversarial-robustness-ood.md) — 22 losses
- [Anomaly Detection & Multi-Modal Learning](sections/anomaly-detection-and-multimodal.md) — 17 losses
- [Image-to-Image Translation & Style Transfer](sections/image-to-image-translation.md) — 16 losses
- [Semi-Supervised Learning & Self-Training](sections/semi-supervised-learning.md) — 12 losses
- [Optical Flow, Video Prediction & Pose Estimation](sections/temporal-motion.md) — 33 losses

**Resources**

- [Survey Papers](#survey-papers)
- [Key Implementation Libraries](#key-implementation-libraries)

---

## 🧭 Loss Selection Guide

Not sure which loss to use? Here's a quick decision framework:

| Task | Default Choice | Class Imbalance | Noisy Labels | Need Calibration |
|------|---------------|-----------------|--------------|------------------|
| Binary Classification | BCE | Focal Loss | SCE / GCE | Focal + Temp. Scaling |
| Multi-class Classification | Cross-Entropy | Class-Balanced CE | Label Smoothing | Label Smoothing |
| Semantic Segmentation | CE + Dice | Focal Tversky | — | — |
| Object Detection (box) | Smooth L1 + Focal | Focal Loss | — | — |
| Object Detection (IoU) | CIoU / GIoU | — | — | — |
| Image Generation (GAN) | Hinge / Non-Saturating | — | — | — |
| Image Generation (Diffusion) | DDPM (ε-prediction) | — | — | — |
| Super-Resolution | L1 + Perceptual + GAN | — | — | — |
| Self-Supervised (vision) | InfoNCE / DINO | — | — | — |
| Face Recognition | ArcFace / AdaFace | Sub-center ArcFace | ElasticFace | — |
| Language Modeling | Cross-Entropy (NTP) | — | — | — |
| LLM Alignment | DPO / SimPO | — | — | — |
| Speech Recognition | CTC / RNN-T | — | — | — |
| RL (value-based) | DQN / Double DQN | — | — | — |
| RL (policy-based) | PPO | — | — | — |
| Regression | MSE / Huber | — | Huber | NLL w/ variance |
| Metric Learning | Triplet / Proxy Anchor | — | — | — |
| Medical Segmentation | Dice + Boundary | Tversky / Focal Tversky | — | — |
| 3D Reconstruction | Chamfer + Normal | — | — | — |
| Depth Estimation | Scale-Invariant | — | — | — |
| Time Series | MSE / Quantile | — | Huber | CRPS |
| Continual Learning | EWC / DER++ | — | — | — |
| Fairness | Group DRO | — | — | — |

---

## 📐 Key Mathematical Formulations

**Cross-Entropy Loss**

$$
\mathcal{L}_{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
$$

**Binary Cross-Entropy**

$$
\mathcal{L}_{BCE} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
$$

**Focal Loss**

$$
\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

**Dice Loss**

$$
\mathcal{L}_{Dice} = 1 - \frac{2 \sum_i p_i g_i}{\sum_i p_i + \sum_i g_i}
$$

**Triplet Loss**

$$
\mathcal{L}_{Triplet} = \max(0, \|f_a - f_p\|_2 - \|f_a - f_n\|_2 + \alpha)
$$

**InfoNCE / Contrastive Loss**

$$
\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

**KL Divergence**

$$
D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

**DDPM Loss (simplified)**

$$
\mathcal{L}_{DDPM} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

**DPO Loss**

$$
\mathcal{L}_{DPO} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right)
$$

**IoU Loss**

$$
\mathcal{L}_{IoU} = 1 - \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|}
$$

**ArcFace Loss**

$$
\mathcal{L}_{ArcFace} = -\log \frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cos \theta_j}}
$$

**Wasserstein Distance (WGAN)**

$$
\mathcal{L}_{WGAN} = \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]
$$

---

## Classification

**0/1 Loss** (1950) — The theoretical misclassification indicator; 1 if prediction ≠ label, 0 otherwise. Non-differentiable, foundational to learning theory.
📄 *Statistical Decision Functions* — Wald, A.

**Cross-Entropy Loss / Log Loss / Negative Log-Likelihood** (1948) — Measures divergence between predicted probability distribution and true labels; the default loss for multi-class classification.
📄 [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — Shannon, C.E.
💻 [`torch.nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

**Binary Cross-Entropy** (1958) — Cross-entropy specialized for two-class or multi-label problems; operates on each output independently.
📄 Derived from logistic regression — Cox, D.R. (1958)
💻 [`torch.nn.BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

**Hinge Loss / SVM Loss** (1995) — Maximizes the margin between classes; the core loss behind Support Vector Machines.
📄 [Support-Vector Networks](https://link.springer.com/article/10.1007/BF00994018) — Cortes, C. & Vapnik, V.
💻 [`torch.nn.MultiMarginLoss`](https://pytorch.org/docs/stable/generated/torch.nn.MultiMarginLoss.html)

**Knowledge Distillation Loss / Soft Cross-Entropy** (2015) — Trains a student network to mimic a teacher by matching softened output distributions.
📄 [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton, G., Vinyals, O. & Dean, J.
💻 [`torch.nn.KLDivLoss`](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)

**Large-Margin Softmax Loss (L-Softmax)** (2016) — Introduces angular margin constraints into softmax for intra-class compactness and inter-class separability.
📄 [Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/abs/1612.02295) — Liu, W., Wen, Y., Yu, Z. & Yang, M.
💻 [wy1iu/LargeMargin_Softmax_Loss](https://github.com/wy1iu/LargeMargin_Softmax_Loss)

**Center Loss** (2016) — Penalizes distance of features from learned class centers, improving discriminative feature learning.
📄 [A Discriminative Feature Learning Approach for Deep Face Recognition](https://kpzhang93.github.io/papers/eccv2016.pdf) — Wen, Y., Zhang, K., Li, Z. & Qiao, Y.
💻 [KaiyangZhou/pytorch-center-loss](https://github.com/KaiyangZhou/pytorch-center-loss)

**Label Smoothing** (2016) — Replaces hard one-hot targets with soft targets, preventing overconfident predictions and improving generalization.
📄 [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) — Szegedy, C. et al.
💻 [`torch.nn.CrossEntropyLoss(label_smoothing=...)`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

**Sparsemax Loss** (2016) — Sparse alternative to softmax that assigns exactly zero probability to irrelevant classes.
📄 [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068) — Martins, A.F.T. & Astudillo, R.F.
💻 [deep-spin/entmax](https://github.com/deep-spin/entmax)

**Focal Loss** (2017) — Down-weights well-classified examples to focus training on hard negatives; designed for extreme class imbalance.
📄 [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) — Lin, T.-Y., Goyal, P., Girshick, R., He, K. & Dollár, P.
💻 [AdeelH/pytorch-multi-class-focal-loss](https://github.com/AdeelH/pytorch-multi-class-focal-loss)

**Generalized Cross-Entropy (GCE)** (2018) — Noise-robust loss interpolating between MAE and cross-entropy via a tunable parameter q.
📄 [Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels](https://arxiv.org/abs/1805.07836) — Zhang, Z. & Sabuncu, M.R.
💻 [AlanChou/Truncated-Loss](https://github.com/AlanChou/Truncated-Loss)

**Complement Objective Training (COT)** (2019) — Augments cross-entropy with a complement objective that neutralizes non-target class probabilities.
📄 [Complement Objective Training](https://arxiv.org/abs/1903.01182) — Chen, H.-Y. et al.
💻 [henry8527/COT](https://github.com/henry8527/COT)

**Class-Balanced Loss** (2019) — Re-weights loss by the effective number of samples per class for long-tailed distributions.
📄 [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555) — Cui, Y. et al.
💻 [vandit15/Class-balanced-loss-pytorch](https://github.com/vandit15/Class-balanced-loss-pytorch)

**Symmetric Cross-Entropy (SCE)** (2019) — Combines standard CE with reverse CE for robustness to label noise.
📄 [Symmetric Cross Entropy for Robust Learning with Noisy Labels](https://arxiv.org/abs/1908.06112) — Wang, Y. et al.

**Bi-Tempered Logistic Loss** (2019) — Two temperature parameters bound the loss (handling mislabeled data) and produce heavy-tailed softmax (handling outliers).
📄 [Robust Bi-Tempered Logistic Loss Based on Bregman Divergences](https://arxiv.org/abs/1906.03361) — Amid, E. et al.
💻 [google/bi-tempered-loss](https://github.com/google/bi-tempered-loss)

**Taylor Cross-Entropy Loss** (2020) — Taylor series expansion of CE creating a noise-robust loss.
📄 [Can Cross Entropy Loss Be Robust to Label Noise?](https://www.ijcai.org/proceedings/2020/305) — Feng, L. et al.

**Asymmetric Loss (ASL)** (2021) — Different focusing levels for positive and negative samples in multi-label classification.
📄 [Asymmetric Loss For Multi-Label Classification](https://arxiv.org/abs/2009.14119) — Ben-Baruch, E. et al.
💻 [Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL)

**Poly Loss** (2022) — Views loss functions as polynomial expansions and adjusts leading coefficients; generalizes CE and focal loss.
📄 [PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions](https://arxiv.org/abs/2204.12511) — Leng, Z. et al.
💻 [abhuse/polyloss-pytorch](https://github.com/abhuse/polyloss-pytorch)

## Regression

**Mean Absolute Error (MAE) / L1 Loss** (~1757) — Penalizes absolute differences; robust to outliers but non-smooth gradient at zero.
📄 Attributed to Boscovich, R.J. (1757)
💻 [`torch.nn.L1Loss`](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)

**Mean Squared Error (MSE) / L2 Loss** (~1805) — Penalizes squared differences; sensitive to outliers. The method of least squares.
📄 Legendre, A.-M. (1805); Gauss, C.F. (1809)
💻 [`torch.nn.MSELoss`](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)

**Huber Loss** (1964) — MSE for small errors, MAE for large errors. Robust to outliers with smooth gradients near zero.
📄 [Robust Estimation of a Location Parameter](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full) — Huber, P.J.
💻 [`torch.nn.HuberLoss`](https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)

**Tukey's Biweight Loss** (1974) — Redescending M-estimator that completely rejects gross outliers beyond a threshold.
📄 [The Fitting of Power Series, Meaning Polynomials, Illustrated on Band-Spectroscopic Data](https://www.tandfonline.com/doi/abs/10.1080/00401706.1974.10489171) — Beaton, A.E. & Tukey, J.W.

**Quantile Loss / Pinball Loss** (1978) — Asymmetrically penalizes over/under-predictions for quantile regression and uncertainty estimation.
📄 [Regression Quantiles](https://people.eecs.berkeley.edu/~jordan/sail/readings/koenker-bassett.pdf) — Koenker, R. & Bassett, G.

**Smooth L1 Loss** (2015) — L2 for small errors, L1 for large errors (Huber with δ=1); standard for bounding box regression.
📄 [Fast R-CNN](https://arxiv.org/abs/1504.08083) — Girshick, R.
💻 [`torch.nn.SmoothL1Loss`](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)

**Wing Loss** (2018) — Amplifies small-to-medium range errors for facial landmark localization.
📄 [Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks](https://arxiv.org/abs/1711.06753) — Feng, Z.-H. et al.

**Balanced L1 Loss** (2019) — Rebalances inlier vs. outlier loss contributions in object detection regression.
📄 [Libra R-CNN: Towards Balanced Learning for Object Detection](https://arxiv.org/abs/1904.02701) — Pang, J. et al.
💻 [OceanPang/Libra_R-CNN](https://github.com/OceanPang/Libra_R-CNN)

**Adaptive Wing Loss** (2019) — Adapts curvature based on ground truth heatmap values for face alignment.
📄 [Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression](https://arxiv.org/abs/1904.07399) — Wang, X. et al.
💻 [protossw512/AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss)

**Log-Cosh Loss** (2022) — Approximates Huber loss using log(cosh(x)); twice differentiable everywhere.
📄 [Statistical Properties of the Log-Cosh Loss Function Used in Machine Learning](https://arxiv.org/abs/2208.04564) — Chen, K. et al.

## Segmentation

**Sensitivity-Specificity Loss** (2015) — Weighted combination of sensitivity and specificity for extreme class imbalance in lesion segmentation.
📄 [Deep Convolutional Encoder Networks for Multiple Sclerosis Lesion Segmentation](https://doi.org/10.1007/978-3-319-24574-4_1) — Brosch et al.

**Dice Loss** (2016) — Directly optimizes the Dice coefficient (F1 score); robust to class imbalance.
📄 [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797) — Milletari, F. et al.
💻 [JunMa11/SegLossOdyssey](https://github.com/JunMa11/SegLossOdyssey)

**Generalized Dice Loss (GDL)** (2017) — Per-class volume weighting for multi-class segmentation with highly imbalanced labels.
📄 [Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations](https://arxiv.org/abs/1707.03237) — Sudre, C.H. et al.

**Tversky Loss** (2017) — Tunable α/β parameters controlling the FP/FN trade-off; useful for small lesion segmentation.
📄 [Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks](https://arxiv.org/abs/1706.05721) — Salehi, S.S.M. et al.

**Lovász-Softmax Loss** (2018) — Tractable convex surrogate for directly optimizing the Jaccard index (IoU).
📄 [The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure](https://arxiv.org/abs/1705.08790) — Berman, M. et al.
💻 [bermanmaxim/LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax)

**Exponential Logarithmic Loss** (2018) — Combines exponentially weighted focal-style Dice and CE for very small structures.
📄 [3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes](https://arxiv.org/abs/1809.00076) — Wong et al.

**Asymmetric Similarity Loss** (2018) — Asymmetric Fβ-score-based similarity to balance precision and recall.
📄 [Asymmetric Loss Functions and Deep Densely Connected Networks for Highly Imbalanced Medical Image Segmentation](https://arxiv.org/abs/1803.11078) — Hashemi et al.

**Focal Tversky Loss** (2019) — Focal-style exponent on Tversky loss to focus on hard, misclassified regions.
📄 [A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation](https://arxiv.org/abs/1810.07842) — Abraham, N. & Khan, N.M.

**Boundary Loss** (2019) — Distance metric on contour space rather than region overlap; effective for highly unbalanced tasks.
📄 [Boundary Loss for Highly Unbalanced Segmentation](https://arxiv.org/abs/1812.07032) — Kervadec, H. et al.
💻 [LIVIAETS/boundary-loss](https://github.com/LIVIAETS/boundary-loss)

**Hausdorff Distance Loss** (2019) — Directly optimizes the Hausdorff distance between predicted and ground-truth boundaries.
📄 [Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks](https://arxiv.org/abs/1904.10030) — Karimi, D. & Salcudean, S.E.

**Combo Loss** (2019) — Weighted combination of modified CE and Dice loss for input and output class imbalance.
📄 [Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation](https://arxiv.org/abs/1805.02798) — Taghanaki et al.

**Region Mutual Information (RMI) Loss** (2019) — Maximizes mutual information between predicted and ground-truth label regions.
📄 [Region Mutual Information Loss for Semantic Segmentation](https://arxiv.org/abs/1910.12037) — Zhao et al.
💻 [ZJULearning/RMI](https://github.com/ZJULearning/RMI)

**Topological Loss** (2019) — Uses persistent homology to enforce correct topological structure in segmentation.
📄 [Topology-Preserving Deep Image Segmentation](https://arxiv.org/abs/1906.05404) — Hu et al.
💻 [HuXiaoling/TopoLoss](https://github.com/HuXiaoling/TopoLoss)

**Log-Cosh Dice Loss** (2020) — Log-cosh smoothing on Dice loss for smoother gradients and stable training.
📄 [A Survey of Loss Functions for Semantic Segmentation](https://arxiv.org/abs/2006.14822) — Jadon, S.

**clDice** (2021) — Topology-preserving loss for tubular structures; computes Dice on skeletonized centerlines.
📄 [clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://arxiv.org/abs/2003.07311) — Shit et al.
💻 [jocpae/clDice](https://github.com/jocpae/clDice)

**Unified Focal Loss** (2022) — Hierarchical framework generalizing Dice-based and CE-based losses with focal modulation.
📄 [Unified Focal Loss: Generalising Dice and Cross Entropy-Based Losses to Handle Class Imbalanced Medical Image Segmentation](https://arxiv.org/abs/2102.04525) — Yeung et al.
💻 [mlyg/unified-focal-loss](https://github.com/mlyg/unified-focal-loss)

## Object Detection (Bounding Box)

**Smooth L1 Loss** (2015) — Piecewise L2/L1 loss; standard for bounding box regression.
📄 [Fast R-CNN](https://arxiv.org/abs/1504.08083) — Girshick, R.

**IoU Loss** (2016) — Directly regresses Intersection-over-Union between predicted and ground-truth boxes.
📄 [UnitBox: An Advanced Object Detection Network](https://arxiv.org/abs/1608.01471) — Yu et al.

**Focal Loss** (2017) — Modulating factor (1−pₜ)^γ down-weights easy negatives in dense detection.
📄 [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) — Lin, T.-Y. et al.
💻 [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)

**Bounded IoU Loss** (2018) — Upper-bounds IoU change per coordinate for stable high-IoU refinement.
📄 [Improving Object Localization with Fitness NMS and Bounded IoU Loss](https://arxiv.org/abs/1711.00164) — Tychsen-Smith & Petersson

**GIoU Loss** (2019) — Extends IoU with a penalty based on the smallest enclosing box; enables gradient flow for non-overlapping boxes.
📄 [Generalized Intersection over Union](https://arxiv.org/abs/1902.09630) — Rezatofighi et al.

**DIoU Loss** (2020) — Adds normalized center-point distance penalty to IoU for faster convergence.
📄 [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287) — Zheng et al.
💻 [Zzh-tju/DIoU](https://github.com/Zzh-tju/DIoU)

**CIoU Loss** (2020) — Extends DIoU with aspect ratio consistency penalty for complete geometric alignment.
📄 [Distance-IoU Loss](https://arxiv.org/abs/1911.08287) — Zheng et al.

**Alpha-IoU Loss** (2021) — Power parameter α amplifies loss and gradient for high-quality anchors.
📄 [Alpha-IoU: A Family of Power Intersection over Union Losses](https://arxiv.org/abs/2110.13675) — He et al.
💻 [Jacobi93/Alpha-IoU](https://github.com/Jacobi93/Alpha-IoU)

**EIoU Loss** (2022) — Decomposes CIoU penalty into separate width/height terms.
📄 [Focal and Efficient IOU Loss for Accurate Bounding Box Regression](https://arxiv.org/abs/2101.08158) — Zhang et al.

**SIoU Loss** (2022) — Angle-aware penalty considering vector direction between predicted and target boxes.
📄 [SIoU Loss: More Powerful Learning for Bounding Box Regression](https://arxiv.org/abs/2205.12740) — Gevorgyan

**WIoU Loss** (2023) — Dynamic non-monotonic focusing mechanism based on outlier degree.
📄 [Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism](https://arxiv.org/abs/2301.10051) — Tong et al.
💻 [Instinct323/Wise-IoU](https://github.com/Instinct323/Wise-IoU)

**MPDIoU Loss** (2023) — Bounding box similarity via minimum point distances between corners.
📄 [MPDIoU: A Loss for Efficient and Accurate Bounding Box Regression](https://arxiv.org/abs/2307.07662) — Ma & Xu

**Inner-IoU Loss** (2023) — IoU through auxiliary inner bounding boxes with a scaling factor.
📄 [Inner-IoU: More Effective Intersection over Union Loss with Auxiliary Bounding Box](https://arxiv.org/abs/2311.02877) — Zhang et al.

## Generative Models — GANs

**Minimax / Original GAN Loss** (2014) — Discriminator maximizes, generator minimizes binary cross-entropy in a two-player minimax game.
📄 [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — Goodfellow et al.

**Non-Saturating GAN Loss** (2014) — Generator maximizes log(D(G(z))) instead of minimizing log(1−D(G(z))), providing stronger early gradients.
📄 [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — Goodfellow et al.

**Feature Matching Loss** (2016) — Generator matches expected feature statistics at an intermediate discriminator layer.
📄 [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) — Salimans et al.

**Least Squares GAN Loss (LSGAN)** (2017) — L2 objective minimizing Pearson χ² divergence for more stable training.
📄 [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076) — Mao et al.

**Wasserstein Loss (WGAN)** (2017) — Earth Mover's distance providing meaningful gradients even for non-overlapping distributions.
📄 [Wasserstein GAN](https://arxiv.org/abs/1701.07875) — Arjovsky, M. et al.
💻 [martinarjovsky/WassersteinGAN](https://github.com/martinarjovsky/WassersteinGAN)

**WGAN-GP** (2017) — Gradient penalty replacing weight clipping for better Lipschitz constraint enforcement.
📄 [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) — Gulrajani et al.

**Hinge Loss GAN** (2017) — Max-margin formulation with bounded gradients; used in BigGAN, SAGAN.
📄 [Geometric GAN](https://arxiv.org/abs/1705.02894) — Lim & Ye
📄 [Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — Miyato et al.

**Spectral Normalization** (2018) — Constrains spectral norm of weight matrices to stabilize discriminator training.
📄 [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957) — Miyato et al.

**R1 Regularization** (2018) — Zero-centered gradient penalty on real data for local convergence guarantees.
📄 [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406) — Mescheder et al.
💻 [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)

**Relativistic GAN Loss (RaGAN)** (2018) — Discriminator estimates probability that real data is more realistic than fake.
📄 [The Relativistic Discriminator](https://arxiv.org/abs/1807.00734) — Jolicoeur-Martineau, A.

**Mode Seeking Loss** (2019) — Maximizes image/latent distance ratio to encourage diverse mode exploration.
📄 [Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis](https://arxiv.org/abs/1903.05628) — Mao et al.

**Path Length Regularization** (2020) — Consistent Jacobian norm across latent space for smooth interpolations.
📄 [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — Karras et al.
💻 [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)

**LeCam Regularization** (2021) — LeCam divergence-based stabilization under limited data.
📄 [Regularizing Generative Adversarial Networks under Limited Data](https://arxiv.org/abs/2104.03310) — Tseng et al.
💻 [google/lecam-gan](https://github.com/google/lecam-gan)

**Projected GAN Loss** (2021) — Multi-scale discrimination in projected feature space from pretrained networks.
📄 [Projected GANs Converge Faster](https://arxiv.org/abs/2111.01007) — Sauer et al.

## Generative Models — VAEs

**ELBO / VAE Loss** (2013) — Reconstruction loss + KL divergence regularizer pushing posterior toward prior.
📄 [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Kingma, D.P. & Welling, M.
💻 [AntixK/PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)

**β-VAE Loss** (2017) — Upweights KL divergence (β > 1) for more disentangled latent representations.
📄 [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — Higgins et al.

**VQ-VAE Loss** (2017) — Reconstruction + vector quantization commitment loss + codebook loss for discrete latents.
📄 [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — van den Oord et al.

**WAE Loss** (2018) — Penalized Wasserstein distance using MMD or adversarial regularization on latent space.
📄 [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558) — Tolstikhin et al.

## Generative Models — Diffusion & Flow

**Denoising Score Matching** (2011) — Training a denoising autoencoder equals matching the score function of noise-perturbed data.
📄 [A Connection Between Score Matching and Denoising Autoencoders](https://doi.org/10.1162/NECO_a_00142) — Vincent, P.

**Score Matching with Langevin Dynamics (NCSN)** (2019) — Noise-conditional score network across multiple noise scales with annealed Langevin sampling.
📄 [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) — Song, Y. & Ermon, S.

**DDPM Loss** (2020) — Simplified variational bound: predict the noise added at each diffusion step via weighted MSE.
📄 [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho, J. et al.

**Variational Diffusion Loss** (2021) — Continuous-time variational lower bound with learnable noise schedule.
📄 [Variational Diffusion Models](https://arxiv.org/abs/2107.00630) — Kingma et al.

**v-prediction Loss** (2022) — Predicts velocity v = α·ε − σ·x for improved numerical stability and progressive distillation.
📄 [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) — Salimans, T. & Ho, J.

**Rectified Flow Loss** (2022) — Learns straight-line ODE trajectories between noise and data distributions.
📄 [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — Liu et al.

**Flow Matching Loss** (2023) — Simulation-free training for continuous normalizing flows; regresses vector fields of conditional probability paths.
📄 [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al.
💻 [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)

**Consistency Loss** (2023) — Self-consistency along the probability flow ODE for high-quality one-step generation.
📄 [Consistency Models](https://arxiv.org/abs/2303.01469) — Song et al.
💻 [OpenAI/consistency_models](https://github.com/openai/consistency_models)

## Reconstruction & Perceptual

**SSIM Loss** (2004) — Structural similarity using luminance, contrast, and structure comparisons; used as 1−SSIM.
📄 [Image Quality Assessment: From Error Visibility to Structural Similarity](https://ieeexplore.ieee.org/document/1284395) — Wang et al.
💻 [VainF/pytorch-msssim](https://github.com/VainF/pytorch-msssim)

**Style Loss (Gram Matrix)** (2015) — Matches Gram matrices of CNN feature maps for texture/style transfer.
📄 [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) — Gatys et al.

**Perceptual Loss / VGG Loss** (2016) — L2 distance between deep feature representations of generated and target images.
📄 [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) — Johnson et al.

**LPIPS** (2018) — Learned perceptual metric using calibrated deep features; correlates better with human perception than SSIM/PSNR.
📄 [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924) — Zhang et al.
💻 [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)

## Image Super-Resolution & Restoration

**Charbonnier Loss** (1994) — Differentiable approximation to L1 (√(x²+ε²)); robust to outliers, smooth at zero.
📄 [Two Deterministic Half-Quadratic Regularization Algorithms for Computed Imaging](https://ieeexplore.ieee.org/document/413553) — Charbonnier et al.

**MS-SSIM Loss** (2003) — Multi-scale SSIM evaluating structural similarity across multiple resolutions.
📄 [Multi-Scale Structural Similarity for Image Quality Assessment](https://ieeexplore.ieee.org/document/1292216) — Wang et al.

**SRGAN Loss** (2017) — Adversarial loss + VGG perceptual content loss for photo-realistic 4× super-resolution.
📄 [Photo-Realistic Single Image Super-Resolution Using a GAN](https://arxiv.org/abs/1609.04802) — Ledig et al.

**Contextual Loss** (2018) — Feature-level context matching without spatial alignment; enables training with non-aligned data.
📄 [The Contextual Loss for Image Transformation with Non-Aligned Data](https://arxiv.org/abs/1803.02077) — Mechrez et al.

**ESRGAN Loss** (2018) — Relativistic average discriminator + pre-activation VGG perceptual loss for enhanced texture recovery.
📄 [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219) — Wang et al.
💻 [xinntao/ESRGAN](https://github.com/xinntao/ESRGAN)

**Focal Frequency Loss** (2021) — Adaptively focuses on hard-to-synthesize frequencies in the Fourier domain.
📄 [Focal Frequency Loss for Image Reconstruction and Synthesis](https://arxiv.org/abs/2012.12821) — Jiang et al.
💻 [EndlessSora/focal-frequency-loss](https://github.com/EndlessSora/focal-frequency-loss)

## Contrastive & Self-Supervised Learning

**Contrastive Loss** (2005) — Pairwise loss pulling similar pairs together and pushing dissimilar pairs apart by a margin.
📄 [Learning a Similarity Metric Discriminatively, with Application to Face Verification](https://www.researchgate.net/publication/4156225) — Chopra, Hadsell, LeCun

**N-pair Loss** (2016) — Generalizes triplet loss by simultaneously pushing away negatives from N−1 classes.
📄 [Improved Deep Metric Learning with Multi-class N-pair Loss Objective](https://papers.nips.cc/paper/6200) — Sohn, K.

**InfoNCE / CPC Loss** (2018) — Noise-contrastive estimation maximizing mutual information between latent representations.
📄 [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) — van den Oord et al.
💻 [RElbers/info-nce-pytorch](https://github.com/RElbers/info-nce-pytorch)

**MoCo Loss** (2020) — InfoNCE with momentum-updated encoder and dynamic dictionary queue.
📄 [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) — He et al.
💻 [facebookresearch/moco](https://github.com/facebookresearch/moco)

**NT-Xent / SimCLR Loss** (2020) — Normalized temperature-scaled cross-entropy over cosine similarities of augmented pairs.
📄 [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) — Chen et al.

**BYOL Loss** (2020) — MSE between L2-normalized predictions and targets; learns without negative pairs via momentum teacher.
📄 [Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733) — Grill et al.

**SwAV Loss** (2020) — Swapped prediction contrasting cluster assignments from different augmented views.
📄 [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882) — Caron et al.
💻 [facebookresearch/swav](https://github.com/facebookresearch/swav)

**Supervised Contrastive Loss (SupCon)** (2020) — Extends self-supervised contrastive loss with label information to pull same-class embeddings together.
📄 [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) — Khosla et al.
💻 [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast)

**Barlow Twins Loss** (2021) — Cross-correlation matrix close to identity; reduces redundancy between embedding dimensions.
📄 [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) — Zbontar et al.
💻 [facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins)

**DINO Loss** (2021) — Self-distillation via cross-entropy between sharpened softmax outputs of student and momentum-teacher.
📄 [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) — Caron et al.
💻 [facebookresearch/dino](https://github.com/facebookresearch/dino)

**SimSiam Loss** (2021) — Negative cosine similarity with stop-gradient; no negatives, momentum, or large batches needed.
📄 [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566) — Chen & He

**CLIP Loss** (2021) — Symmetric cross-entropy over image-text cosine similarities aligning visual and language representations.
📄 [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — Radford et al.
💻 [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)

**VICReg Loss** (2022) — Variance + invariance + covariance regularization preventing collapse without negatives.
📄 [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906) — Bardes et al.
💻 [facebookresearch/vicreg](https://github.com/facebookresearch/vicreg)

**Decoupled Contrastive Loss** (2022) — Removes positive term from InfoNCE denominator, eliminating negative-positive coupling.
📄 [Decoupled Contrastive Learning](https://arxiv.org/abs/2110.06848) — Yeh et al.

**DINOv2 Loss** (2023) — DINO self-distillation + iBOT masked image modeling + Sinkhorn centering at scale.
📄 [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — Oquab et al.
💻 [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

**SigLIP Loss** (2023) — Pairwise sigmoid loss replacing softmax for efficient batch-parallel language-image pre-training.
📄 [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) — Zhai et al.

## Metric Learning & Face Recognition

**Triplet Loss** (2015) — Minimizes anchor-positive distance while maximizing anchor-negative distance by a margin.
📄 [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) — Schroff et al.
💻 [KevinMusgrave/pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

**Lifted Structured Loss** (2016) — Mines all positive and negative pairs in a batch simultaneously.
📄 [Deep Metric Learning via Lifted Structured Feature Embedding](https://arxiv.org/abs/1511.06452) — Oh Song et al.

**SphereFace / A-Softmax** (2017) — Multiplicative angular margin on a hypersphere for discriminative face features.
📄 [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063) — Liu et al.

**Proxy-NCA Loss** (2017) — Data-to-proxy comparisons with one learnable proxy per class; dramatically faster convergence.
📄 [No Fuss Distance Metric Learning Using Proxies](https://arxiv.org/abs/1703.07464) — Movshovitz-Attias et al.

**CosFace / LMCL** (2018) — Cosine margin penalty on target logit in normalized softmax.
📄 [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/abs/1801.09414) — Wang et al.
💻 [deepinsight/insightface](https://github.com/deepinsight/insightface)

**ArcFace** (2019) — Additive angular margin with clear geodesic distance interpretation.
📄 [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698) — Deng et al.
💻 [deepinsight/insightface](https://github.com/deepinsight/insightface)

**Multi-Similarity Loss** (2019) — Mines and weights pairs using self-similarity, relative similarity, and negative similarity.
📄 [Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](https://arxiv.org/abs/1904.06627) — Wang et al.

**SoftTriple Loss** (2019) — Multiple centers per class bridging proxy-based and triplet-based losses.
📄 [SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](https://arxiv.org/abs/1909.05235) — Qian et al.

**Circle Loss** (2020) — Unified pair similarity optimization with self-paced weighting.
📄 [Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/abs/2002.10857) — Sun et al.

**Proxy Anchor Loss** (2020) — Proxies as anchors associated with all batch data; fast convergence.
📄 [Proxy Anchor Loss for Deep Metric Learning](https://arxiv.org/abs/2003.13911) — Kim et al.
💻 [tjddus9597/Proxy-Anchor-CVPR2020](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020)

**Sub-center ArcFace** (2020) — Multiple sub-centers per class for noisy label handling.
📄 [Sub-center ArcFace: Boosting Face Recognition by Large-Scale Noisy Web Faces](https://arxiv.org/abs/1801.07698) — Deng et al.

**AdaFace** (2022) — Adaptive margin emphasizing hard or easy samples based on image quality.
📄 [AdaFace: Quality Adaptive Margin for Face Recognition](https://arxiv.org/abs/2204.00964) — Kim et al.
💻 [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)

**ElasticFace** (2022) — Random margin values from a normal distribution each iteration for flexible separability.
📄 [ElasticFace: Elastic Margin Loss for Deep Face Recognition](https://arxiv.org/abs/2109.09416) — Boutros et al.
💻 [fdbtrs/ElasticFace](https://github.com/fdbtrs/ElasticFace)

## NLP & Language Modeling

**Cross-Entropy / Next Token Prediction** — Standard autoregressive LM loss; foundation of GPT and all causal LMs.
📄 [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al. (GPT-2, 2019)

**Masked Language Model (MLM) Loss** (2019) — Masks 15% of tokens and predicts from bidirectional context. Introduced pre-train/fine-tune for NLU.
📄 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — Devlin et al.

**Replaced Token Detection (RTD)** (2020) — Discriminator classifies every token as original or replaced; loss defined over all tokens for better sample efficiency.
📄 [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — Clark et al.

**Sentence Order Prediction (SOP)** (2020) — Predicts whether two consecutive segments are in correct or swapped order.
📄 [ALBERT: A Lite BERT for Self-supervised Learning](https://arxiv.org/abs/1909.11942) — Lan et al.

**Span Corruption Loss** (2020) — Masks contiguous spans; encoder-decoder reconstructs only missing spans. All NLP tasks as text-to-text.
📄 [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — Raffel et al.

**Mixture of Denoisers (MoD)** (2022) — Unifies causal LM, prefix LM, and span corruption into a single pre-training objective.
📄 [UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131) — Tay et al.

## LLM Alignment (RLHF / DPO)

**PPO Loss / RLHF** (2017/2022) — Clipped surrogate objective for aligning LLMs with human preferences via a learned reward model.
📄 [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Schulman et al.
📄 [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Ouyang et al.
💻 [huggingface/trl](https://github.com/huggingface/trl)

**Reward Model Loss / Bradley-Terry** (2022) — Cross-entropy on pairwise human preferences for training scalar reward models.
📄 [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Ouyang et al.

**SLiC-HF Loss** (2023) — Contrastive ranking loss calibrating sequence likelihoods to human preferences.
📄 [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/abs/2305.10425) — Zhao et al.

**DPO Loss** (2023) — Closed-form policy optimization directly from preference pairs; no separate reward model or RL loop.
📄 [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al.
💻 [huggingface/trl — DPOTrainer](https://github.com/huggingface/trl)

**IPO Loss** (2023) — Squared loss on preference margins avoiding overfitting to Bradley-Terry assumption.
📄 [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) — Azar et al.

**CPO Loss** (2024) — Contrastive preference loss without reference model for machine translation.
📄 [Contrastive Preference Optimization](https://arxiv.org/abs/2401.08417) — Xu et al.

**KTO Loss** (2024) — Kahneman-Tversky prospect theory applied to alignment; works from binary (good/bad) feedback.
📄 [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — Ethayarajh et al.
💻 [huggingface/trl — KTOTrainer](https://github.com/huggingface/trl)

**GRPO Loss** (2024) — Group Relative Policy Optimization; estimates advantages from sampled output groups, eliminating the critic model.
📄 [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — Shao et al.

**ORPO Loss** (2024) — Odds-ratio penalty added to SFT loss; combines instruction tuning and preference alignment in one stage.
📄 [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — Hong et al.
💻 [huggingface/trl — ORPOTrainer](https://github.com/huggingface/trl)

**SimPO Loss** (2024) — Reference-free preference optimization using length-normalized average log probability as implicit reward.
📄 [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) — Meng et al.
💻 [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO)

**SPPO Loss** (2024) — Self-play preference optimization framing alignment as a two-player constant-sum game.
📄 [Self-Play Preference Optimization for Language Model Alignment](https://arxiv.org/abs/2405.00675) — Wu et al.

## Sequence-to-Sequence & Speech

**CTC Loss** (2006) — Marginalizes over all valid alignments between input and output sequences; foundational for ASR.
📄 [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — Graves et al.
💻 [`torch.nn.CTCLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)

**RNN-T Loss** (2012) — Extends CTC with a prediction network conditioning on previous outputs for streaming transduction.
📄 [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711) — Graves, A.
💻 [`torchaudio.transforms.RNNTLoss`](https://pytorch.org/audio/stable/generated/torchaudio.transforms.RNNTLoss.html)

**Scheduled Sampling Loss** (2015) — Gradually replaces ground-truth tokens with model predictions during training to mitigate exposure bias.
📄 [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/abs/1506.03099) — Bengio et al.

**Sequence-Level Training / MIXER** (2016) — Directly optimizes BLEU/ROUGE using REINFORCE.
📄 [Sequence Level Training with Recurrent Neural Networks](https://arxiv.org/abs/1511.06732) — Ranzato et al.

**Minimum Risk Training** (2016) — Minimizes expected task-level loss (e.g., 1−BLEU) via sampling.
📄 [Minimum Risk Training for Neural Machine Translation](https://arxiv.org/abs/1512.02433) — Shen et al.

**Mel-Spectrogram Reconstruction Loss** (2017) — L1/L2 between predicted and target mel-spectrograms; primary TTS training objective.
📄 [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135) — Wang et al.

**Multi-Resolution STFT Loss** (2020) — Spectral convergence + log-magnitude STFT at multiple FFT sizes for neural vocoder training.
📄 [Parallel WaveGAN](https://arxiv.org/abs/1910.11480) — Yamamoto et al.
💻 [csteinmetz1/auraloss](https://github.com/csteinmetz1/auraloss)

## Reinforcement Learning

**TD Loss / Temporal Difference** (1988) — Bootstrapped value estimation updating predictions toward reward + discounted next-state value.
📄 [Learning to Predict by the Methods of Temporal Differences](https://link.springer.com/article/10.1007/BF00115009) — Sutton, R.S.

**Q-Learning Loss** (1989) — Off-policy TD control bootstrapping with max Q-value over next actions.
📄 [Learning from Delayed Rewards](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf) — Watkins, C.J.C.H.

**REINFORCE / Policy Gradient** (1992) — Monte Carlo policy gradient weighted by returns.
📄 [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — Williams, R.J.

**DQN Loss** (2015) — Q-learning with deep networks, experience replay, and target networks.
📄 [Human-level Control through Deep Reinforcement Learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) — Mnih et al.
💻 [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

**Double DQN Loss** (2015) — Decouples action selection from evaluation to reduce overestimation bias.
📄 [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — van Hasselt et al.

**DDPG Loss** (2015) — Deterministic policy gradients for continuous control with experience replay.
📄 [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) — Lillicrap et al.

**GAE** (2015) — Exponentially-weighted multi-step TD errors for tunable bias-variance tradeoff.
📄 [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — Schulman et al.

**A3C / A2C Loss** (2016) — Actor-critic with policy gradient + value function baseline + entropy bonus.
📄 [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — Mnih et al.

**Distributional RL / C51 Loss** (2017) — Models full return distribution using categorical projection over fixed atoms.
📄 [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887) — Bellemare et al.

**PPO Clipped Surrogate Loss** (2017) — Clips probability ratio to prevent destructively large policy updates.
📄 [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Schulman et al.
💻 [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

**HER Loss** (2017) — Relabels failed trajectories with achieved goals for sample-efficient sparse-reward learning.
📄 [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) — Andrychowicz et al.

**QR-DQN Loss** (2018) — Quantile regression approximating the return distribution with learnable quantile locations.
📄 [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044) — Dabney et al.

**SAC Loss** (2018) — Maximum entropy actor-critic balancing exploration and exploitation automatically.
📄 [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) — Haarnoja et al.

**TD3 Loss** (2018) — Clipped double-Q learning + delayed policy updates + target policy smoothing.
📄 [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) — Fujimoto et al.

**V-trace Loss** (2018) — Importance-weighted off-policy correction for scalable distributed RL (IMPALA).
📄 [IMPALA: Scalable Distributed Deep-RL](https://arxiv.org/abs/1802.01561) — Espeholt et al.

**Decision Transformer Loss** (2021) — RL as sequence modeling; autoregressive transformer conditioned on returns, trained with supervised loss.
📄 [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) — Chen et al.
💻 [kzl/decision-transformer](https://github.com/kzl/decision-transformer)

## Knowledge Distillation

**Knowledge Distillation / KD Loss** (2015) — Student matches softened output distribution of teacher via KL divergence at elevated temperature.
📄 [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton, Vinyals, Dean

**FitNets / Hint Loss** (2015) — Student mimics intermediate feature representations of teacher.
📄 [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550) — Romero et al.

**Attention Transfer Loss** (2017) — Forces student to mimic spatial attention maps of teacher's intermediate layers.
📄 [Paying More Attention to Attention](https://arxiv.org/abs/1612.03928) — Zagoruyko & Komodakis
💻 [szagoruyko/attention-transfer](https://github.com/szagoruyko/attention-transfer)

**Born-Again Networks** (2018) — Self-distillation where identical-architecture student outperforms teacher.
📄 [Born Again Neural Networks](https://arxiv.org/abs/1805.04770) — Furlanello et al.

**PKT / Probabilistic KD** (2018) — Matches probability distributions in feature space rather than raw representations.
📄 [Learning Deep Representations with Probabilistic Knowledge Transfer](https://arxiv.org/abs/1803.10837) — Passalis & Tefas

**Relational KD / RKD** (2019) — Transfers mutual relations (distances and angles) between examples.
📄 [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068) — Park et al.

**Self-Distillation Loss** (2019) — Deeper layers supervise shallower classifiers within the same network.
📄 [Be Your Own Teacher](https://arxiv.org/abs/1905.08094) — Zhang et al.

**CRD / Contrastive Representation Distillation** (2020) — Maximizes mutual information between teacher and student via contrastive objective.
📄 [Contrastive Representation Distillation](https://arxiv.org/abs/1910.10699) — Tian et al.
💻 [HobbitLong/RepDistiller](https://github.com/HobbitLong/RepDistiller)

**ReviewKD** (2021) — Student's lower-level features guided by teacher's higher-level features through attention-based fusion.
📄 [Distilling Knowledge via Knowledge Review](https://arxiv.org/abs/2104.09044) — Chen et al.
💻 [dvlab-research/ReviewKD](https://github.com/dvlab-research/ReviewKD)

**DKD / Decoupled KD** (2022) — Decouples KD into target-class and non-target-class components for independent weighting.
📄 [Decoupled Knowledge Distillation](https://arxiv.org/abs/2203.08679) — Zhao et al.
💻 [megvii-research/mdistiller](https://github.com/megvii-research/mdistiller)

**DIST Loss** (2022) — Preserves inter-class relations and intra-class ranking rather than exact probability matching.
📄 [Knowledge Distillation from A Stronger Teacher](https://arxiv.org/abs/2205.10536) — Huang et al.
💻 [hunto/DIST_KD](https://github.com/hunto/DIST_KD)

## Regularization

**KL Divergence** (1951) — Measures information lost when approximating one distribution with another.
📄 [On Information and Sufficiency](https://doi.org/10.1214/aoms/1177729694) — Kullback & Leibler

**L2 Regularization / Weight Decay** (1970) — Penalizes sum of squared weights to prevent overfitting.
📄 [Ridge Regression](https://doi.org/10.1080/00401706.1970.10488634) — Hoerl & Kennard

**L1 Regularization / Lasso** (1996) — Penalizes sum of absolute weights, inducing sparsity.
📄 [Regression Shrinkage and Selection via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x) — Tibshirani, R.

**Elastic Net** (2005) — Combines L1 and L2 for sparsity + grouping of correlated features.
📄 [Regularization and Variable Selection via the Elastic Net](https://doi.org/10.1111/j.1467-9868.2005.00503.x) — Zou & Hastie

**Dropout** (2014) — Randomly zeroes activations; implicit ensemble of exponentially many sub-networks.
📄 [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) — Srivastava et al.

**Confidence Penalty** (2017) — Penalizes low-entropy (overconfident) output distributions.
📄 [Regularizing Neural Networks by Penalizing Confident Output Distributions](https://arxiv.org/abs/1701.06548) — Pereyra et al.

**Mixup Loss** (2018) — Trains on convex combinations of example pairs and their labels.
📄 [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) — Zhang et al.
💻 [facebookresearch/mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10)

**Manifold Mixup** (2019) — Extends Mixup to hidden representations at random intermediate layers.
📄 [Manifold Mixup: Better Representations by Interpolating Hidden States](https://arxiv.org/abs/1806.05236) — Verma et al.

**CutMix Loss** (2019) — Cuts and pastes rectangular patches between images while mixing labels proportionally.
📄 [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899) — Yun et al.

## 3D Vision & Point Clouds

**Chamfer Distance** (2017) — Average nearest-neighbor distance between two point sets; fast and widely used.
📄 [A Point Set Generation Network for 3D Object Reconstruction from a Single Image](https://arxiv.org/abs/1612.00603) — Fan et al.
💻 [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d)

**Earth Mover's Distance (EMD)** (2017) — Optimal transport distance with bijective matching; higher quality but more expensive than CD.
📄 [A Point Set Generation Network for 3D Object Reconstruction from a Single Image](https://arxiv.org/abs/1612.00603) — Fan et al.

**Normal Consistency Loss** (2018) — Penalizes inconsistency of surface normals between adjacent mesh faces.
📄 [Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images](https://arxiv.org/abs/1804.01654) — Wang et al.

**Mesh Laplacian Smoothing Loss** (2018) — Penalizes vertex deviation from neighbor centroid to prevent self-intersections.
📄 [Pixel2Mesh](https://arxiv.org/abs/1804.01654) — Wang et al.
💻 [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d)

**SDF Loss (DeepSDF)** (2019) — Regresses signed distance values; zero level-set defines the 3D surface.
📄 [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103) — Park et al.
💻 [facebookresearch/DeepSDF](https://github.com/facebookresearch/DeepSDF)

**Occupancy Loss** (2019) — Binary CE on predicted occupancy probabilities for 3D reconstruction.
📄 [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/abs/1812.03828) — Mescheder et al.

**NeRF Photometric Loss** (2020) — MSE between rendered and observed pixel colors via differentiable volume rendering.
📄 [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) — Mildenhall et al.

**3D Gaussian Splatting Loss** (2023) — L1 + D-SSIM for optimizing anisotropic 3D Gaussians for real-time radiance field rendering.
📄 [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — Kerbl et al.
💻 [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

## Depth Estimation

**Scale-Invariant Loss** (2014) — Log-space depth error minus mean shift; invariant to global scale ambiguity.
📄 [Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](https://arxiv.org/abs/1406.2283) — Eigen et al.

**Berhu Loss (Reverse Huber)** (2016) — L1 for small residuals, L2 for large; robust depth regression.
📄 [Deeper Depth Prediction with Fully Convolutional Residual Networks](https://arxiv.org/abs/1606.00373) — Laina et al.

**Photometric Consistency Loss** (2017) — Self-supervised SSIM + L1 with left-right disparity consistency for monocular depth.
📄 [Unsupervised Monocular Depth Estimation with Left-Right Consistency](https://arxiv.org/abs/1609.03677) — Godard et al.
💻 [nianticlabs/monodepth2](https://github.com/nianticlabs/monodepth2)

**Edge-Aware Smoothness Loss** (2017) — Locally smooth depth except at image edges, weighted by image gradients.
📄 [Unsupervised Monocular Depth Estimation with Left-Right Consistency](https://arxiv.org/abs/1609.03677) — Godard et al.

## Medical Imaging

**Deep Supervision Loss** (2015) — Auxiliary losses at intermediate layers providing direct gradient paths.
📄 [Deeply-Supervised Nets](https://arxiv.org/abs/1409.5185) — Lee et al.

**Dice Loss** (2016) — Directly optimizes Dice coefficient for volumetric medical image segmentation.
📄 [V-Net](https://arxiv.org/abs/1606.04797) — Milletari et al.

**Generalized Dice Loss** (2017) — Per-class volume weighting for highly imbalanced multi-class segmentation.
📄 [Generalised Dice Overlap as a Deep Learning Loss Function](https://arxiv.org/abs/1707.03237) — Sudre et al.

**Tversky Loss** (2017) — Tunable FP/FN trade-off for small lesion segmentation.
📄 [Tversky Loss Function for Image Segmentation](https://arxiv.org/abs/1706.05721) — Salehi et al.

**Attention-Gated Loss** (2018) — Learned attention gates suppress irrelevant regions in skip connections.
📄 [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999) — Oktay et al.

**Boundary / Surface Loss** (2019) — Distance metric on contour space for highly unbalanced medical segmentation.
📄 [Boundary Loss for Highly Unbalanced Segmentation](https://arxiv.org/abs/1812.07032) — Kervadec et al.
💻 [LIVIAETS/boundary-loss](https://github.com/LIVIAETS/boundary-loss)

**Distance Map Penalized CE** (2019) — Weights CE by distance transform maps to focus on boundary regions.
📄 [Distance Map Loss Penalty Term for Semantic Segmentation](https://arxiv.org/abs/1908.03679) — Caliva et al.

## Graph Neural Networks

**Variational Graph Auto-Encoder (VGAE) Loss** (2016) — Reconstruction BCE on adjacency matrix + KL divergence for unsupervised graph learning.
📄 [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) — Kipf & Welling

**Node Classification Loss** (2017) — Standard cross-entropy per-node in semi-supervised graph settings.
📄 [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Kipf & Welling
💻 [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric)

**Deep Graph Infomax (DGI) Loss** (2019) — Maximizes mutual information between local node and global graph representations.
📄 [Deep Graph Infomax](https://arxiv.org/abs/1809.10341) — Veličković et al.
💻 [PetarV-/DGI](https://github.com/PetarV-/DGI)

**Graph Matching Loss** (2019) — Attention-based cross-graph matching with margin-based pairwise loss.
📄 [Graph Matching Networks for Learning the Similarity of Graph Structured Objects](https://arxiv.org/abs/1904.12787) — Li et al.

**InfoGraph Loss** (2020) — Maximizes mutual information between graph-level and substructure-level representations.
📄 [InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning](https://arxiv.org/abs/1908.01000) — Sun et al.
💻 [sunfanyunn/InfoGraph](https://github.com/sunfanyunn/InfoGraph)

**GraphCL Loss** (2020) — NT-Xent contrastive loss on augmented graph views for self-supervised graph learning.
📄 [Graph Contrastive Learning with Augmentations](https://arxiv.org/abs/2010.13902) — You et al.
💻 [Shen-Lab/GraphCL](https://github.com/Shen-Lab/GraphCL)

**BGRL Loss** (2022) — Negative-sample-free self-supervised loss bootstrapping graph representations (inspired by BYOL).
📄 [Large-Scale Representation Learning on Graphs via Bootstrapping](https://arxiv.org/abs/2102.06514) — Thakoor et al.
💻 [nerdslab/bgrl](https://github.com/nerdslab/bgrl)

## Recommendation Systems

**ListNet Loss** (2007) — Listwise learning-to-rank using top-one probability distributions.
📄 [Learning to Rank: From Pairwise Approach to Listwise Approach](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf) — Cao et al.

**ListMLE Loss** (2008) — Listwise loss based on likelihood of ground-truth permutation under Plackett-Luce model.
📄 [Listwise Approach to Learning to Rank: Theory and Algorithm](https://dl.acm.org/doi/10.1145/1390156.1390306) — Xia et al.

**BPR Loss** (2009) — Pairwise loss maximizing posterior probability that user prefers observed over unobserved items.
📄 [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618) — Rendle et al.
💻 [guoyang9/BPR-pytorch](https://github.com/guoyang9/BPR-pytorch)

**Sampled Softmax Loss** (2015) — Approximates full softmax over large item vocabulary by sampling negatives.
📄 [On Using Very Large Target Vocabulary for Neural Machine Translation](https://arxiv.org/abs/1412.2007) — Jean et al.

**DirectAU Loss** (2022) — Directly optimizes alignment and uniformity on the hypersphere for collaborative filtering.
📄 [Towards Representation Alignment and Uniformity in Collaborative Filtering](https://arxiv.org/abs/2206.12811) — Wang et al.
💻 [THUwangcy/DirectAU](https://github.com/THUwangcy/DirectAU)

## Multi-Task Learning

**Uncertainty Weighting / Homoscedastic Uncertainty** (2018) — Learns task weights by modeling task-dependent uncertainty; noisy tasks auto-downweighted.
📄 [Multi-Task Learning Using Uncertainty to Weigh Losses](https://arxiv.org/abs/1705.07115) — Kendall et al.
💻 [median-research-group/LibMTL](https://github.com/median-research-group/LibMTL)

**GradNorm** (2018) — Dynamically normalizes gradient magnitudes across tasks to balance training rates.
📄 [GradNorm: Gradient Normalization for Adaptive Loss Balancing](https://arxiv.org/abs/1711.02257) — Chen et al.

**MGDA** (2018) — Multi-objective optimization finding Pareto-optimal descent direction via Frank-Wolfe on task gradients.
📄 [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650) — Sener & Koltun

**PCGrad** (2020) — Projects conflicting task gradients onto normal planes to reduce destructive interference.
📄 [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) — Yu et al.

**CAGrad** (2021) — Minimizes average loss while maximizing worst-case local improvement across tasks.
📄 [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/abs/2110.14048) — Liu et al.

**Nash-MTL** (2022) — Nash bargaining game where tasks negotiate a joint update direction.
📄 [Multi-Task Learning as a Bargaining Game](https://arxiv.org/abs/2202.01017) — Navon et al.
💻 [AvivNavon/nash-mtl](https://github.com/AvivNavon/nash-mtl)

## Uncertainty Estimation

**NLL with Learned Variance** (1994) — Network predicts mean and variance; NLL naturally trades off accuracy and calibration.
📄 [Estimating the Mean and Variance of the Target Probability Distribution](https://ieeexplore.ieee.org/document/374138) — Nix & Weigend

**MC Dropout** (2016) — Dropout at test time as approximate Bayesian inference for uncertainty estimation.
📄 [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142) — Gal & Ghahramani

**Deep Ensembles Loss** (2017) — Ensemble of networks with proper scoring rules + adversarial training for diversity.
📄 [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474) — Lakshminarayanan et al.

**Evidential Deep Learning Loss** (2018) — Dirichlet prior over class probabilities; Bayes risk + KL divergence regularizer.
📄 [Evidential Deep Learning to Quantify Classification Uncertainty](https://arxiv.org/abs/1806.01768) — Sensoy et al.

## Domain Adaptation

**Maximum Mean Discrepancy (MMD)** (2012) — Distribution distance in RKHS; aligns source and target features without adversarial training.
📄 [A Kernel Two-Sample Test](https://www.jmlr.org/papers/v13/gretton12a/gretton12a.pdf) — Gretton et al.
💻 [ZongxianLee/MMD_Loss.Pytorch](https://github.com/ZongxianLee/MMD_Loss.Pytorch)

**Domain Adversarial Loss / DANN** (2016) — Gradient reversal layer training domain classifier adversarially for domain-invariant features.
📄 [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818) — Ganin et al.
💻 [fungtion/DANN](https://github.com/fungtion/DANN)

**Deep CORAL Loss** (2016) — Aligns second-order statistics (covariance) of source and target deep features.
📄 [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/abs/1607.01719) — Sun & Saenko

**Wasserstein Distance for DA** (2018) — Earth Mover's Distance as domain discrepancy measure with gradient penalty.
📄 [Wasserstein Distance Guided Representation Learning for Domain Adaptation](https://arxiv.org/abs/1707.01217) — Shen et al.

**Contrastive Domain Discrepancy (CDD)** (2019) — Class-aware alignment maximizing inter-class and minimizing intra-class discrepancy across domains.
📄 [Contrastive Adaptation Network for Unsupervised Domain Adaptation](https://arxiv.org/abs/1901.00976) — Kang et al.

---

## Survey Papers

- 📄 [A Comprehensive Survey of Loss Functions and Metrics in Deep Learning](https://arxiv.org/abs/2307.02694) — Terven et al. (2025)
- 📄 [A Survey of Loss Functions for Semantic Segmentation](https://arxiv.org/abs/2006.14822) — Jadon (2020)
- 📄 [Loss Functions in the Era of Semantic Segmentation: A Survey and Outlook](https://arxiv.org/abs/2312.05391) — Azad et al. (2023)

## Key Implementation Libraries

| Library | Focus | Link |
|---------|-------|------|
| PyTorch (built-in) | CE, BCE, MSE, Huber, CTC, KLDiv, etc. | [pytorch.org](https://pytorch.org/docs/stable/nn.html#loss-functions) |
| pytorch-metric-learning | Triplet, Contrastive, ArcFace, ProxyNCA, etc. | [GitHub](https://github.com/KevinMusgrave/pytorch-metric-learning) |
| SegLossOdyssey | Dice, Tversky, Boundary, Hausdorff, etc. | [GitHub](https://github.com/JunMa11/SegLossOdyssey) |
| Hugging Face TRL | DPO, PPO, KTO, ORPO, SimPO, etc. | [GitHub](https://github.com/huggingface/trl) |
| Stable-Baselines3 | DQN, PPO, SAC, TD3, A2C, etc. | [GitHub](https://github.com/DLR-RM/stable-baselines3) |
| lightly | SimCLR, BYOL, MoCo, DINO, Barlow Twins, etc. | [GitHub](https://github.com/lightly-ai/lightly) |
| insightface | ArcFace, CosFace, Sub-center ArcFace | [GitHub](https://github.com/deepinsight/insightface) |
| open_clip | CLIP, SigLIP contrastive losses | [GitHub](https://github.com/mlfoundations/open_clip) |
| PyTorch3D | Chamfer, mesh losses, point cloud losses | [GitHub](https://github.com/facebookresearch/pytorch3d) |
| PyTorch Geometric | GNN losses, link prediction, node classification | [GitHub](https://github.com/pyg-team/pytorch_geometric) |
| LibMTL | Uncertainty weighting, GradNorm, PCGrad, Nash-MTL | [GitHub](https://github.com/median-research-group/LibMTL) |
| auraloss | Multi-Resolution STFT, mel losses | [GitHub](https://github.com/csteinmetz1/auraloss) |
| BasicSR | Perceptual, SSIM, Charbonnier, GAN losses for SR | [GitHub](https://github.com/XPixelGroup/BasicSR) |
| kornia | Focal, Dice, SSIM, and more | [GitHub](https://github.com/kornia/kornia) |
| anomalib | Anomaly detection losses and methods | [GitHub](https://github.com/open-edge-platform/anomalib) |
| Avalanche | Continual learning (EWC, SI, LwF, etc.) | [GitHub](https://github.com/ContinualAI/avalanche) |
| GluonTS | Time series forecasting losses | [GitHub](https://github.com/awslabs/gluonts) |
| audiocraft | Audio generation (EnCodec, MusicGen) | [GitHub](https://github.com/facebookresearch/audiocraft) |
| AIF360 | Fairness and bias mitigation | [GitHub](https://github.com/Trusted-AI/AIF360) |

---

## Star History

If you find this useful, please star the repo — it helps others discover it.

[![Star History Chart](https://api.star-history.com/svg?repos=stabgan/awesome-loss-functions&type=Date)](https://star-history.com/#stabgan/awesome-loss-functions&Date)
