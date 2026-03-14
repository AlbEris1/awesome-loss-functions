# Loss Functions for Optical Flow, Video Prediction, Temporal Modeling & Pose Estimation

A comprehensive, chronologically ordered reference of important loss functions used across optical flow estimation, video prediction, temporal modeling, and human pose estimation.

---

## 🔵 OPTICAL FLOW

**Brightness Constancy / Horn-Schunck Energy** (1981) — The foundational variational energy combining a photometric data term (brightness constancy) with a smoothness regularizer for dense optical flow.
📄 [Determining Optical Flow](https://www.sci.utah.edu/~gerig/CS6320-S2015/Materials/Horn-Schunck-1981-orig-article.pdf) — Berthold K.P. Horn, Brian G. Schunck
💻 [OpenCV `calcOpticalFlowHS`](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html)

---

**Endpoint Error (EPE) Loss** (2012) — The Euclidean distance between predicted and ground-truth flow vectors, averaged over all pixels; the standard supervised metric for optical flow.
📄 [A Database and Evaluation Methodology for Optical Flow](https://link.springer.com/article/10.1007/s11263-010-0390-2) — Simon Baker, Daniel Scharstein, J.P. Lewis, Stefan Roth, Michael J. Black, Richard Szeliski
💻 [NVIDIA FlowNet2 PyTorch (EPE)](https://github.com/NVIDIA/flownet2-pytorch)

---

**Charbonnier Penalty (Robust Flow Loss)** (2013) — A differentiable approximation to L1 (generalized Charbonnier) used as a robust penalty for both data and smoothness terms in optical flow, less sensitive to outliers than L2.
📄 [A Quantitative Analysis of Current Practices in Optical Flow Estimation and the Principles Behind Them](https://link.springer.com/article/10.1007/s11263-013-0644-x) — Deqing Sun, Stefan Roth, Michael J. Black
💻 [PWC-Net PyTorch (uses Charbonnier)](https://github.com/NVlabs/PWC-Net)

---

**Multi-Scale EPE Loss / FlowNet** (2015) — Supervised L2 (EPE) loss applied at multiple spatial resolutions of a coarse-to-fine decoder, enabling end-to-end learning of optical flow with CNNs.
📄 [FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852) — Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg, Philip Häusser, Caner Hazırbaş, Vladimir Golkov, Patrick van der Smagt, Daniel Cremers, Thomas Brox
💻 [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch)

---

**Photometric Loss (Brightness Constancy Loss)** (2016) — Unsupervised loss measuring pixel intensity difference between a warped frame (using predicted flow) and the target frame, enabling flow learning without ground truth.
📄 [Back to Basics: Unsupervised Learning of Optical Flow via Brightness Constancy and Motion Smoothness](https://arxiv.org/abs/1608.05842) — Jason J. Yu, Adam W. Harley, Konstantinos G. Derpanis
💻 [ily-R/Unsupervised-Optical-Flow (PyTorch)](https://github.com/ily-R/Unsupervised-Optical-Flow)

---

**Stacked Multi-Scale Loss / FlowNet 2.0** (2017) — Extends FlowNet's multi-scale EPE loss to stacked network architectures, with schedule-based loss weighting across cascaded refinement stages, reducing error by 50%+.
📄 [FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925) — Eddy Ilg, Nikolaus Mayer, Tonmoy Saikia, Margret Keuper, Alexey Dosovitskiy, Thomas Brox
💻 [NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch)

---

**Spatial Pyramid Loss / SPyNet** (2017) — Coarse-to-fine residual flow loss at each pyramid level, combining classical spatial pyramid formulation with deep learning in a compact model (96% smaller than FlowNet).
📄 [Optical Flow Estimation Using a Spatial Pyramid Network](https://arxiv.org/abs/1611.00850) — Anurag Ranjan, Michael J. Black
💻 [sniklaus/pytorch-spynet](https://github.com/sniklaus/pytorch-spynet)

---

**Census Transform Loss** (2018) — A robust photometric loss based on the census transform (ternary encoding of local neighborhoods), invariant to illumination changes, combined with occlusion-aware bidirectional flow estimation.
📄 [UnFlow: Unsupervised Learning of Optical Flow with a Bidirectional Census Loss](https://arxiv.org/abs/1711.07837) — Simon Meister, Junhwa Hur, Stefan Roth
💻 [simonmeister/UnFlow (TensorFlow)](https://github.com/simonmeister/UnFlow)

---

**Occlusion-Aware Flow Loss** (2018) — Explicitly models occlusion masks via forward-backward consistency checking, applying photometric and smoothness losses only in non-occluded regions for more accurate unsupervised flow.
📄 [Occlusion Aware Unsupervised Learning of Optical Flow](https://arxiv.org/abs/1711.05890) — Yang Wang, Yi Yang, Zhenheng Yang, Liang Zhao, Peng Wang, Wei Xu
💻 [coolbeam/UPFlow_pytorch](https://github.com/coolbeam/UPFlow_pytorch)

---

**Pyramid Warping Cost Volume Loss / PWC-Net** (2018) — Multi-scale L2 loss with robust Charbonnier penalty on a pyramid-warping-cost-volume architecture, fusing classical optical flow principles into an end-to-end trainable CNN.
📄 [PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371) — Deqing Sun, Xiaodong Yang, Ming-Yu Liu, Jan Kautz
💻 [NVlabs/PWC-Net](https://github.com/NVlabs/PWC-Net)

---

**Sequence Loss / RAFT** (2020) — Exponentially weighted sum of L1 distances between iterative flow predictions and ground truth across all recurrent update steps, supervising the full refinement trajectory.
📄 [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039) — Zachary Teed, Jia Deng
💻 [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) · [torchvision.models.optical_flow.raft](https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py)

---

**Mixture of Laplace Loss / SEA-RAFT** (2024) — Replaces standard L1 with a mixture of Laplace distribution likelihood loss, modeling per-pixel flow uncertainty and enabling more robust training with direct initial flow regression.
📄 [SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow](https://arxiv.org/abs/2405.14793) — Yihan Wang, Lahav Lipson, Jia Deng
💻 [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)

---

## 🟢 VIDEO PREDICTION & TEMPORAL MODELING

**Gradient Difference Loss (GDL)** (2016) — Penalizes differences in image gradients between predicted and target frames, encouraging sharp edges and reducing the blurriness inherent in MSE-based video prediction.
📄 [Deep Multi-Scale Video Prediction Beyond Mean Square Error](https://arxiv.org/abs/1511.05440) — Michaël Mathieu, Camille Couprie, Yann LeCun
💻 [dyelax/Adversarial_Video_Generation (TensorFlow)](https://github.com/dyelax/Adversarial_Video_Generation)

---

**Video Adversarial Loss** (2016) — Applies adversarial training (GAN) to video prediction, using a discriminator to distinguish real vs. generated frames, producing sharper and more realistic future frame predictions.
📄 [Deep Multi-Scale Video Prediction Beyond Mean Square Error](https://arxiv.org/abs/1511.05440) — Michaël Mathieu, Camille Couprie, Yann LeCun
💻 [dyelax/Adversarial_Video_Generation](https://github.com/dyelax/Adversarial_Video_Generation)

---

**Spatio-Temporal Adversarial Loss / VGAN** (2016) — A GAN loss with a spatio-temporal convolutional discriminator that evaluates both spatial realism and temporal coherence of generated video sequences.
📄 [Generating Videos with Scene Dynamics](https://arxiv.org/abs/1609.02612) — Carl Vondrick, Hamed Pirsiavash, Antonio Torralba
💻 [cvondrick/videogan](https://github.com/cvondrick/videogan)

---

**Predictive Coding Loss / PredNet** (2017) — Hierarchical prediction error loss inspired by neuroscience predictive coding theory, where each layer predicts the activity of the layer below and only forward-propagates the error.
📄 [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104) — William Lotter, Gabriel Kreiman, David Cox
💻 [coxlab/prednet (Keras)](https://github.com/coxlab/prednet)

---

**Motion-Content Decomposition Loss** (2017) — Separates video prediction into motion (optical flow / difference) and content (appearance) streams with independent reconstruction losses, improving long-term prediction quality.
📄 [Decomposing Motion and Content for Natural Video Sequence Prediction](https://arxiv.org/abs/1706.08033) — Ruben Villegas, Jimei Yang, Seunghoon Hong, Xunyu Lin, Honglak Lee
💻 [rubenvillegas/iclr2017mcnet](https://github.com/rubenvillegas/iclr2017mcnet)

---

**Temporal Adversarial Loss / MoCoGAN** (2018) — Decomposes video generation into motion and content codes with separate image and video discriminators; the temporal discriminator enforces realistic motion dynamics across frames.
📄 [MoCoGAN: Decomposing Motion and Content for Video Generation](https://arxiv.org/abs/1707.04993) — Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz
💻 [sergeytulyakov/mocogan](https://github.com/sergeytulyakov/mocogan)

---

**Temporal Consistency Loss (Short-term + Long-term)** (2018) — Flow-warped consistency loss that penalizes differences between consecutive output frames after optical-flow-based warping, with both short-term (adjacent) and long-term (distant) variants.
📄 [Learning Blind Video Temporal Consistency](https://arxiv.org/abs/1808.00449) — Wei-Sheng Lai, Jia-Bin Huang, Oliver Wang, Eli Shechtman, Ersin Yumer, Ming-Hsuan Yang
💻 [phoenix104104/fast_blind_video_consistency](https://github.com/phoenix104104/fast_blind_video_consistency)

---

**Warping Loss / vid2vid** (2018) — Measures the L1 distance between a generated frame and the previous generated frame warped by estimated optical flow, enforcing temporal smoothness in video-to-video synthesis.
📄 [Video-to-Video Synthesis](https://arxiv.org/abs/1808.06601) — Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu, Andrew Tao, Jan Kautz, Bryan Catanzaro
💻 [NVIDIA/vid2vid](https://github.com/NVIDIA/vid2vid)

---

**Temporal Self-Supervision Loss** (2019) — Self-supervised temporal coherence loss for GAN-based video tasks, using the network's own predictions across time as pseudo-labels to enforce frame-to-frame consistency without paired data.
📄 [Learning Temporal Coherence via Self-Supervision for GAN-based Video Generation](https://arxiv.org/abs/1811.09393) — Mengyu Chu, You Xie, Jonas Mayer, Laura Leal-Taixé, Nils Thuerey
💻 [thunil/TecoGAN](https://github.com/thunil/TecoGAN)

---

## 🟠 POSE ESTIMATION

**Heatmap MSE Loss** (2014) — Mean squared error between predicted and ground-truth Gaussian heatmaps for each joint, establishing the dominant paradigm for 2D pose estimation via heatmap regression.
📄 [Joint Training of a Convolutional Network and a Graphical Model for Human Pose Estimation](https://arxiv.org/abs/1406.2984) — Jonathan Tompson, Arjun Jain, Yann LeCun, Christoph Bregler
💻 [open-mmlab/mmpose (HeatmapLoss)](https://github.com/open-mmlab/mmpose)

---

**MPJPE Loss (Mean Per Joint Position Error)** (2014) — The Euclidean distance between predicted and ground-truth 3D joint positions, averaged over all joints; the standard metric and loss for 3D human pose estimation.
📄 [Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments](https://ieeexplore.ieee.org/document/6682899) — Catalin Ionescu, Dragos Papava, Vlad Olaru, Cristian Sminchisescu
💻 [cbsudux/Human-Pose-Estimation-101](https://github.com/cbsudux/Human-Pose-Estimation-101) · [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)

---

**OKS-based Loss (Object Keypoint Similarity)** (2014) — A scale-invariant similarity metric for keypoint detection using per-keypoint Gaussian falloff weighted by annotator variance, serving as the COCO keypoint evaluation standard and differentiable training loss.
📄 [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312) — Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, C. Lawrence Zitnick
💻 [cocodataset/cocoapi (OKS computation)](https://github.com/cocodataset/cocoapi)

---

**Intermediate Supervision / Stacked Hourglass Loss** (2016) — Applies heatmap MSE loss at the output of each stacked hourglass module, providing intermediate supervision that enables repeated bottom-up/top-down refinement of pose predictions.
📄 [Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937) — Alejandro Newell, Kaiyu Yang, Jia Deng
💻 [princeton-vl/pose-hg-train (Torch)](https://github.com/princeton-vl/pose-hg-train) · [bearpaw/pytorch-pose](https://github.com/bearpaw/pytorch-pose)

---

**Associative Embedding Loss** (2017) — A tag-based grouping loss for bottom-up multi-person pose estimation: a pull loss brings tags of same-person joints together, and a push loss separates tags of different persons.
📄 [Associative Embedding: End-to-End Learning for Joint Detection and Grouping](https://arxiv.org/abs/1611.05424) — Alejandro Newell, Zhiao Huang, Jia Deng
💻 [princeton-vl/pose-ae-train](https://github.com/princeton-vl/pose-ae-train)

---

**Integral Regression Loss (Soft-Argmax)** (2018) — Replaces non-differentiable argmax with a differentiable soft-argmax (integral) operation over heatmaps, enabling end-to-end joint coordinate regression with L1/L2 loss on continuous coordinates.
📄 [Integral Human Pose Regression](https://arxiv.org/abs/1711.08229) — Xiao Sun, Bin Xiao, Fangyin Wei, Shuang Liang, Yichen Wei
💻 [JimmySuen/integral-human-pose](https://github.com/JimmySuen/integral-human-pose) · [mks0601/Integral-Human-Pose-Regression-for-3D-Human-Pose-Estimation](https://github.com/mks0601/Integral-Human-Pose-Regression-for-3D-Human-Pose-Estimation)

---

**Wing Loss** (2018) — A piecewise loss function that amplifies the influence of small-to-medium range errors for facial landmark / keypoint localization, outperforming L1/L2/smooth-L1 for precise alignment.
📄 [Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks](https://arxiv.org/abs/1711.06753) — Zhen-Hua Feng, Josef Kittler, Muhammad Awais, Patrik Huber, Xiao-Jun Wu
💻 [TropComplique/wing-loss (PyTorch)](https://github.com/TropComplique/wing-loss)

---

**Adaptive Wing Loss** (2019) — Adapts its shape to different ground-truth heatmap pixel values (foreground vs. background), providing stronger gradients near Gaussian peaks for more accurate heatmap regression.
📄 [Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression](https://arxiv.org/abs/1904.07399) — Xinyao Wang, Liefeng Bo, Li Fuxin
💻 [protossw512/AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss)

---

**Bone Length Loss** (2020–2024) — Penalizes deviations of predicted bone lengths (Euclidean distance between connected joints) from anatomical priors or ground truth, enforcing skeletal structure consistency in 3D pose.
📄 [A Geometry Loss Combination for 3D Human Pose Estimation](https://openaccess.thecvf.com/content/WACV2024/papers/Matsune_A_Geometry_Loss_Combination_for_3D_Human_Pose_Estimation_WACV_2024_paper.pdf) — Shunsuke Matsune et al. (WACV 2024); also used in Bigalke et al. (MICCAI 2022)
💻 [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) · [Various implementations in 3D pose papers]

---

**Joint Angle Loss** (2020–2024) — Constrains predicted joint angles to anatomically plausible ranges, penalizing physically impossible limb configurations in 3D pose estimation.
📄 [Domain Adaptation through Anatomical Constraints for 3D Human Pose Estimation](https://proceedings.mlr.press/v172/bigalke22a/bigalke22a.pdf) — Alexander Bigalke et al. (MICCAI 2022)
💻 [Implementations in domain-adaptation pose repos]

---

**SimCC Loss (Simple Coordinate Classification)** (2022) — Reformulates keypoint localization as two 1D classification tasks (horizontal and vertical), using cross-entropy loss on sub-pixel coordinate bins instead of heatmap regression.
📄 [SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation](https://arxiv.org/abs/2107.03332) — Yanjie Li, Sen Yang, Peidong Liu, Shoukui Zhang, Yunxiao Wang, Zhicheng Wang, Wankou Yang, Shu-Tao Xia
💻 [open-mmlab/mmpose (SimCC codec)](https://github.com/open-mmlab/mmpose)

---

**Spatio-Temporal Pose Loss** (2022) — Combines MPJPE with temporal smoothness terms (velocity/acceleration penalties) for 3D motion reconstruction from video, enforcing both per-frame accuracy and motion coherence.
📄 [A New Spatio-Temporal Loss Function for 3D Motion Reconstruction](https://hal.science/hal-03966941/file/paper.pdf) — Mathis Tchenegnon et al.
💻 [Various video-based 3D pose repos]

---

## 📊 Summary Table

| # | Loss Function | Year | Domain | Key Innovation |
|---|---|---|---|---|
| 1 | Horn-Schunck Energy | 1981 | Optical Flow | Brightness constancy + smoothness |
| 2 | EPE Loss | 2012 | Optical Flow | Standard Euclidean flow metric |
| 3 | Charbonnier Penalty | 2013 | Optical Flow | Robust differentiable L1 approx |
| 4 | Heatmap MSE Loss | 2014 | Pose | Gaussian heatmap regression |
| 5 | MPJPE Loss | 2014 | Pose (3D) | Per-joint Euclidean distance |
| 6 | OKS-based Loss | 2014 | Pose | Scale-invariant keypoint similarity |
| 7 | Multi-Scale EPE / FlowNet | 2015 | Optical Flow | Multi-resolution supervised flow |
| 8 | Photometric Loss | 2016 | Optical Flow | Unsupervised brightness constancy |
| 9 | GDL (Gradient Difference) | 2016 | Video | Sharp edge preservation |
| 10 | Video Adversarial Loss | 2016 | Video | GAN for frame realism |
| 11 | Spatio-Temporal Adversarial | 2016 | Video | 3D conv discriminator |
| 12 | Intermediate Supervision | 2016 | Pose | Loss at each hourglass stage |
| 13 | Stacked Multi-Scale / FlowNet2 | 2017 | Optical Flow | Cascaded network loss scheduling |
| 14 | Spatial Pyramid / SPyNet | 2017 | Optical Flow | Per-level residual flow loss |
| 15 | Predictive Coding / PredNet | 2017 | Video | Hierarchical prediction error |
| 16 | Motion-Content Decomp. | 2017 | Video | Separate motion/content losses |
| 17 | Associative Embedding | 2017 | Pose | Pull/push tag grouping loss |
| 18 | Census Transform Loss | 2018 | Optical Flow | Illumination-invariant matching |
| 19 | Occlusion-Aware Flow Loss | 2018 | Optical Flow | Masked loss in non-occluded regions |
| 20 | PWC-Net Loss | 2018 | Optical Flow | Pyramid + warping + cost volume |
| 21 | Temporal Adversarial / MoCoGAN | 2018 | Video | Motion/content discriminators |
| 22 | Temporal Consistency Loss | 2018 | Video | Flow-warped frame consistency |
| 23 | Warping Loss / vid2vid | 2018 | Video | Flow-warped temporal smoothness |
| 24 | Integral Regression Loss | 2018 | Pose | Differentiable soft-argmax |
| 25 | Wing Loss | 2018 | Pose | Amplified small-error gradients |
| 26 | Temporal Self-Supervision | 2019 | Video | Self-supervised coherence |
| 27 | Adaptive Wing Loss | 2019 | Pose | Pixel-adaptive heatmap loss |
| 28 | Sequence Loss / RAFT | 2020 | Optical Flow | Exponentially weighted iterative |
| 29 | Bone Length Loss | 2020+ | Pose (3D) | Skeletal structure constraint |
| 30 | Joint Angle Loss | 2020+ | Pose (3D) | Anatomical plausibility |
| 31 | SimCC Loss | 2022 | Pose | 1D coordinate classification |
| 32 | Spatio-Temporal Pose Loss | 2022 | Pose (3D) | MPJPE + temporal smoothness |
| 33 | Mixture of Laplace / SEA-RAFT | 2024 | Optical Flow | Probabilistic flow distribution |
