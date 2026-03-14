# Calibration, Fairness & Bias Mitigation — Loss Functions

> A chronological catalog of loss functions for model calibration, fairness-aware learning, and algorithmic bias mitigation.

---

## Part I — Calibration

---

**1. Brier Score Loss** (1950) — Proper scoring rule measuring the mean squared error between predicted probabilities and binary outcomes, capturing both calibration and refinement.
📄 [Verification of Forecasts Expressed in Terms of Probability](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2) — Glenn W. Brier
💻 [Implementation (scikit-learn)](https://github.com/scikit-learn/scikit-learn)

---

**2. Platt Scaling Loss** (1999) — Post-hoc sigmoid calibration of classifier outputs via logistic regression on a held-out set.
📄 [Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods](https://home.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf) — John C. Platt
💻 [Implementation (scikit-learn CalibratedClassifierCV)](https://github.com/scikit-learn/scikit-learn)

---

**3. Temperature Scaling Loss** (2017) — Single scalar temperature parameter optimized on NLL for simple and effective post-hoc calibration of neural networks.
📄 [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) — Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger
💻 [Implementation (official)](https://github.com/gpleiss/temperature_scaling)

---

**4. Focal Loss for Calibration** (2017/2020) — Focal loss naturally improves calibration by down-weighting easy examples and reducing overconfidence on well-classified samples.
📄 [Calibrating Deep Neural Networks using Focal Loss](https://arxiv.org/abs/2002.09437) — Jishnu Mukhoti, Viveka Kulharia, Amartya Sanyal, Stuart Golodetz, Philip H.S. Torr, Puneet K. Dokania
💻 [Implementation (official)](https://github.com/torrvision/focal_calibration)

---

**5. MMCE Loss (Maximum Mean Calibration Error)** (2018) — Kernel-based differentiable calibration measure trainable end-to-end alongside the standard loss.
📄 [Trainable Calibration Measures for Neural Networks from Kernel Mean Embeddings](https://proceedings.mlr.press/v80/kumar18a.html) — Aviral Kumar, Sunita Sarawagi, Ujjwal Jain
💻 [Implementation (official)](https://github.com/aviralkumar2907/MMCE)

---

**6. Label Smoothing for Calibration** (2019) — Soft target distributions that prevent overconfident predictions and implicitly improve model calibration.
📄 [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) — Rafael Müller, Simon Kornblith, Geoffrey Hinton
💻 [Implementation (PyTorch built-in)](https://github.com/pytorch/pytorch)

---

**7. Mixup for Calibration** (2019) — Mixup data augmentation acts as an implicit calibration regularizer, improving predictive uncertainty estimates.
📄 [On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks](https://arxiv.org/abs/1905.11001) — Sunil Thulasidasan, Gopinath Chennupati, Jeff Bilmes, Tanmoy Bhattacharya, Sarah Michalak
💻 [Implementation (PyTorch Mixup)](https://github.com/facebookresearch/mixup-cifar10)

---

**8. Dirichlet Calibration Loss** (2019) — Extends temperature scaling to a full Dirichlet distribution-based multi-class calibration using a linear layer on log-probabilities.
📄 [Beyond Temperature Scaling: Obtaining Well-Calibrated Multiclass Probabilities with Dirichlet Calibration](https://arxiv.org/abs/1910.12656) — Meelis Kull, Miquel Perello Nieto, Markus Kängsepp, Telmo Silva Filho, Hao Song, Peter Flach
💻 [Implementation (official)](https://github.com/dirichletcal/dirichlet_python)

---

**9. Spline Calibration** (2020) — Post-hoc recalibration using natural cubic splines fitted on a held-out calibration set for flexible probability mapping.
📄 [Calibration of Neural Networks using Splines](https://arxiv.org/abs/2006.12800) — Kartik Gupta, Amir Rahimi, Thalaiyasingam Ajanthan, Thomas Mensink, Cristian Sminchisescu, Richard Hartley
💻 [Implementation (official)](https://github.com/kartikgupta-at-anu/spline-calibration)

---

**10. Focal Calibration Loss (AdaFocal)** (2022) — Calibration-aware adaptive focal loss that dynamically adjusts the focusing parameter using calibration feedback during training.
📄 [AdaFocal: Calibration-aware Adaptive Focal Loss](https://arxiv.org/abs/2211.11838) — Arindam Ghosh, Thomas Schaaf, Matt Gormley
💻 [Implementation (official)](https://github.com/3mcloud/AdaFocal)

---

## Part II — Fairness & Bias Mitigation

---

**11. Prejudice Remover Regularizer** (2012) — Regularization term penalizing mutual information between predictions and sensitive attributes to enforce fairness during training.
📄 [Fairness-Aware Classifier with Prejudice Remover Regularizer](https://doi.org/10.1007/978-3-642-33486-3_3) — Toshihiro Kamishima, Shotaro Akaho, Hideki Asoh, Jun Sakuma
💻 [Implementation (AIF360)](https://github.com/Trusted-AI/AIF360)

---

**12. Equalized Odds Post-Processing** (2016) — Adjusts classifier thresholds to equalize true positive and false positive rates across protected groups.
📄 [Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413) — Moritz Hardt, Eric Price, Nathan Srebro
💻 [Implementation (AIF360)](https://github.com/Trusted-AI/AIF360)

---

**13. Adversarial Debiasing Loss** (2018) — Adversarial training framework that removes sensitive attribute information from learned representations via a minimax objective.
📄 [Mitigating Unwanted Biases with Adversarial Learning](https://arxiv.org/abs/1801.07593) — Brian Hu Zhang, Blake Lemoine, Margaret Mitchell
💻 [Implementation (AIF360)](https://github.com/Trusted-AI/AIF360)

---

**14. Fairness Constraints as Lagrangian** (2019) — Constrained optimization framework enforcing non-differentiable fairness constraints via proxy-Lagrangian multipliers.
📄 [Optimization with Non-Differentiable Constraints with Applications to Fairness, Recall, Churn, and Other Goals](https://arxiv.org/abs/1809.04198) — Andrew Cotter, Heinrich Jiang, Maya Gupta, Serena Wang, Taman Narayan, Seungil You, Karthik Sridharan
💻 [Implementation (TFCO)](https://github.com/google-research/tensorflow_constrained_optimization)

---

**15. Group DRO Loss** (2020) — Distributionally robust optimization that minimizes the worst-case loss across predefined groups to improve tail-group performance.
📄 [Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization](https://arxiv.org/abs/1911.08731) — Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, Percy Liang
💻 [Implementation (official)](https://github.com/kohpangwei/group_DRO)

---

**16. FairMixup Loss** (2021) — Mixup-based data augmentation across sensitive groups with path-dependent regularization to enforce fairness constraints.
📄 [Fair Mixup: Fairness via Interpolation](https://arxiv.org/abs/2103.06503) — Ching-Yao Chuang, Youssef Mroueh
💻 [Implementation (official)](https://github.com/chingyaoc/fair-mixup)

---

**17. Just Train Twice (JTT) Loss** (2021) — Two-stage training that identifies misclassified minority examples in a first pass and upweights them in a second training run.
📄 [Just Train Twice: Improving Group Robustness without Training Group Information](https://arxiv.org/abs/2107.09044) — Evan Zheran Liu, Behzad Haghgoo, Annie S. Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, Chelsea Finn
💻 [Implementation (official)](https://github.com/anniesch/jtt)

---

**18. Contrastive Fairness Loss (FSCL)** (2022) — Fair Supervised Contrastive Loss that prevents encoder networks from encoding sensitive attribute information into learned representations.
📄 [Fair Contrastive Learning for Facial Attribute Classification](https://arxiv.org/abs/2203.16209) — Sungho Park, Jewook Lee, Pilhyeon Lee, Sunhee Hwang, Dohyung Kim, Hyeran Byun
💻 [Implementation (official)](https://github.com/sungho-CoolG/Fair-Supervised-Contrastive-Learning)

---

## Quick Reference

| # | Method | Year | Type | Key Idea |
|---|--------|------|------|----------|
| 1 | Brier Score Loss | 1950 | Calibration | Proper scoring rule (MSE of probabilities) |
| 2 | Platt Scaling | 1999 | Calibration | Sigmoid post-hoc recalibration |
| 3 | Temperature Scaling | 2017 | Calibration | Single-parameter softmax tempering |
| 4 | Focal Loss for Calibration | 2017/2020 | Calibration | Down-weight easy examples → less overconfidence |
| 5 | MMCE | 2018 | Calibration | Kernel-based differentiable calibration loss |
| 6 | Label Smoothing | 2019 | Calibration | Soft targets prevent overconfidence |
| 7 | Mixup for Calibration | 2019 | Calibration | Interpolation-based implicit regularizer |
| 8 | Dirichlet Calibration | 2019 | Calibration | Multi-class calibration via Dirichlet distributions |
| 9 | Spline Calibration | 2020 | Calibration | Cubic spline recalibration mapping |
| 10 | AdaFocal | 2022 | Calibration | Adaptive focal loss with calibration feedback |
| 11 | Prejudice Remover | 2012 | Fairness | Mutual information regularizer |
| 12 | Equalized Odds | 2016 | Fairness | Threshold adjustment for equal TPR/FPR |
| 13 | Adversarial Debiasing | 2018 | Fairness | Adversarial removal of sensitive info |
| 14 | Fairness via Lagrangian | 2019 | Fairness | Constrained optimization with multipliers |
| 15 | Group DRO | 2020 | Fairness | Minimize worst-group loss |
| 16 | FairMixup | 2021 | Fairness | Cross-group mixup regularization |
| 17 | JTT | 2021 | Fairness | Two-stage upweighting of minority errors |
| 18 | FSCL | 2022 | Fairness | Fair supervised contrastive learning |

---

## Unified Libraries

| Library | Description | Link |
|---------|-------------|------|
| **AIF360** | IBM's AI Fairness 360 toolkit — bias detection and mitigation algorithms (Prejudice Remover, Adversarial Debiasing, Equalized Odds, and more) | [Trusted-AI/AIF360](https://github.com/Trusted-AI/AIF360) |
| **Fairlearn** | Microsoft's fairness assessment and mitigation library with reductions-based algorithms and dashboard | [fairlearn/fairlearn](https://github.com/fairlearn/fairlearn) |
| **netcal** | Calibration framework for measuring and mitigating miscalibration (Platt, Temperature, Spline, Beta, and more) | [EFS-OpenSource/calibration-framework](https://github.com/EFS-OpenSource/calibration-framework) |
