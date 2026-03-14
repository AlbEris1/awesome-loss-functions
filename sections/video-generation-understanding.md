# Video Generation & Understanding — Loss Functions

> A chronological catalog of loss functions for video generation, video diffusion models, action recognition, and video-language learning.

---

## 🔵 Part I — Video Generation & Synthesis

**Video GAN Loss / VGAN** (2016) — Spatio-temporal adversarial loss with a 3D convolutional discriminator that separately models foreground motion and background scene for video generation.
📄 [Generating Videos with Scene Dynamics](https://arxiv.org/abs/1609.02612) — Carl Vondrick, Hamed Pirsiavash, Antonio Torralba
💻 [cvondrick/videogan](https://github.com/cvondrick/videogan)

---

**Temporal GAN (TGAN) Loss** (2017) — Decomposes video generation into a temporal generator that produces latent codes across time and an image generator that renders each frame, trained with adversarial loss and singular value clipping.
📄 [Temporal Generative Adversarial Nets with Singular Value Clipping](https://arxiv.org/abs/1611.06624) — Masaki Saito, Eiichi Matsumoto, Shunta Saito
💻 [pfnet-research/tgan](https://github.com/pfnet-research/tgan)

---

**MoCoGAN Loss** (2018) — Decomposes latent space into fixed content and stochastic motion codes, using separate image and video discriminators to enforce per-frame realism and temporal coherence.
📄 [MoCoGAN: Decomposing Motion and Content for Video Generation](https://arxiv.org/abs/1707.04993) — Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz
💻 [sergeytulyakov/mocogan](https://github.com/sergeytulyakov/mocogan)

---

**DVD-GAN Loss** (2019) — Dual Video Discriminator GAN with a spatial discriminator judging individual frames and a temporal discriminator evaluating motion across downsampled frame pairs, scaling to high-resolution video.
📄 [Adversarial Video Generation on Complex Datasets](https://arxiv.org/abs/1907.06571) — Aidan Clark, Jeff Donahue, Karen Simonyan
💻 [No official release; based on BigGAN architecture]

---

**VideoGPT Loss** (2021) — VQ-VAE with 3D convolutions and axial self-attention learns discrete video tokens, then a GPT-like autoregressive transformer is trained with cross-entropy loss on the token sequence.
📄 [VideoGPT: Video Generation using VQ-VAE and Transformers](https://arxiv.org/abs/2104.10157) — Wilson Yan, Yunzhi Zhang, Pieter Abbeel, Aravind Srinivas
💻 [wilson1yan/VideoGPT](https://github.com/wilson1yan/VideoGPT)

---

**Video Diffusion Loss** (2022) — Extends image diffusion to video with a 3D U-Net architecture, using a standard denoising score-matching objective with joint image-video training and reconstruction-guided conditional sampling.
📄 [Video Diffusion Models](https://arxiv.org/abs/2204.03458) — Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet
💻 [lucidrains/video-diffusion-pytorch](https://github.com/lucidrains/video-diffusion-pytorch)

---

**Make-A-Video Loss** (2022) — Spatiotemporal diffusion pipeline that first trains on image-text pairs, then extends to video via pseudo-temporal attention layers and frame interpolation, without requiring paired text-video data.
📄 [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://arxiv.org/abs/2209.14792) — Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, Yaniv Taigman
💻 [No official release; Meta Research]

---

**Imagen Video Loss** (2022) — Cascaded video diffusion system with a base model plus interleaved spatial and temporal super-resolution stages, using v-prediction parameterization and progressive distillation for efficient sampling.
📄 [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303) — Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, Tim Salimans
💻 [No official release; Google Research]

---

**CogVideo Loss** (2022) — 9B-parameter autoregressive transformer inheriting from CogView2, trained with cross-entropy on video tokens using a multi-frame-rate hierarchical strategy to improve text-video alignment.
📄 [CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](https://arxiv.org/abs/2205.15868) — Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, Jie Tang
💻 [THUDM/CogVideo](https://github.com/THUDM/CogVideo)

---

**Stable Video Diffusion Loss** (2023) — Latent video diffusion model using the EDM (Elucidating Diffusion Models) noise schedule framework, with a three-stage training pipeline: image pre-training, curated video pre-training, and high-quality video fine-tuning.
📄 [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127) — Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, Robin Rombach
💻 [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)

---

**VideoPoet Loss** (2024) — Multimodal autoregressive transformer trained with cross-entropy on tokens from MAGVIT-v2 (video), SoundStream (audio), and T5 (text) tokenizers, following an LLM-style pre-train then task-adapt protocol.
📄 [VideoPoet: A Large Language Model for Zero-Shot Video Generation](https://arxiv.org/abs/2312.14125) — Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Josh Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, Ming-Hsuan Yang, Irfan Essa, Huisheng Wang, David A. Ross, Bryan Seybold, Lu Jiang
💻 [No official release; Google Research]

---

## 🟢 Part II — Video Understanding & Action Recognition

**Two-Stream Loss** (2014) — Dual-stream architecture with a spatial network (RGB frames) and a temporal network (stacked optical flow), each trained with softmax cross-entropy and fused via late averaging or SVM stacking.
📄 [Two-Stream Convolutional Networks for Action Recognition in Videos](https://arxiv.org/abs/1406.2199) — Karen Simonyan, Andrew Zisserman
💻 [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2)

---

**C3D Loss** (2015) — 3D convolutional network with homogeneous 3×3×3 kernels trained with standard softmax cross-entropy, learning generic spatiotemporal features transferable across video tasks.
📄 [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767) — Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, Manohar Paluri
💻 [facebookresearch/C3D](https://github.com/facebookresearch/C3D)

---

**I3D Loss** (2017) — Inflated 3D ConvNets that bootstrap ImageNet-pretrained 2D filters into 3D by replicating weights along the temporal axis, trained with two-stream (RGB + flow) softmax cross-entropy on the Kinetics dataset.
📄 [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750) — João Carreira, Andrew Zisserman
💻 [deepmind/kinetics-i3d](https://github.com/deepmind/kinetics-i3d)

---

**SlowFast Loss** (2019) — Dual-pathway network with a Slow pathway (low frame rate, rich spatial features) and a Fast pathway (high frame rate, lightweight temporal features) connected by lateral connections, trained with a single cross-entropy loss on the fused representation.
📄 [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982) — Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He
💻 [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)

---

**TimeSformer Loss** (2021) — Pure transformer for video classification using divided space-time attention (separate temporal and spatial self-attention within each block), trained with standard cross-entropy classification loss.
📄 [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) — Gedas Bertasius, Heng Wang, Lorenzo Torresani
💻 [facebookresearch/TimeSformer](https://github.com/facebookresearch/TimeSformer)

---

**VideoMAE Loss** (2022) — Masked autoencoder for video with extremely high tube masking ratio (90–95%), reconstructing masked spatiotemporal patches in pixel space using MSE loss, enabling data-efficient self-supervised video pre-training.
📄 [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602) — Zhan Tong, Yibing Song, Jue Wang, Limin Wang
💻 [MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE)

---

**InternVideo Loss** (2022) — Unified video foundation model combining masked video modeling (generative) and video-language contrastive learning (discriminative) objectives, with a learnable coordination mechanism to fuse both representations.
📄 [InternVideo: General Video Foundation Models via Generative and Discriminative Learning](https://arxiv.org/abs/2212.03191) — Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, Yu Qiao
💻 [OpenGVLab/InternVideo](https://github.com/OpenGVLab/InternVideo)

---

## 🟣 Part III — Video-Language

**VideoCLIP Loss** (2021) — Contrastive learning between video and text using temporally overlapping (rather than exactly aligned) clips as positive pairs, enabling fine-grained zero-shot video-text understanding.
📄 [VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding](https://arxiv.org/abs/2109.14084) — Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, Christoph Feichtenhofer
💻 [facebookresearch/fairseq (VideoCLIP)](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT)

---

**Frozen in Time Loss** (2021) — Dual-encoder contrastive loss (video encoder + text encoder) with curriculum learning that starts from image-text pairs and gradually introduces video-text pairs, enabling joint image and video retrieval.
📄 [Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval](https://arxiv.org/abs/2104.00650) — Max Bain, Arsha Nagrani, Gül Varol, Andrew Zisserman
💻 [m-bain/frozen-in-time](https://github.com/m-bain/frozen-in-time)

---

## 🛠️ Unified Libraries

| Library | Repository | Description |
|---|---|---|
| MMAction2 | [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2) | OpenMMLab's comprehensive action recognition toolbox supporting Two-Stream, C3D, I3D, SlowFast, TimeSformer, VideoMAE, and more |
| PyTorchVideo | [facebookresearch/pytorchvideo](https://github.com/facebookresearch/pytorchvideo) | Meta's video understanding library with efficient implementations of SlowFast, X3D, and video transforms |
| video-diffusion-pytorch | [lucidrains/video-diffusion-pytorch](https://github.com/lucidrains/video-diffusion-pytorch) | Community implementation of video diffusion models in PyTorch, based on the 3D U-Net denoising framework |

---

## 📊 Quick Reference Table

| # | Loss Function | Year | Domain | Key Innovation |
|---|---|---|---|---|
| 1 | Two-Stream Loss | 2014 | Action Recognition | Dual spatial + temporal stream fusion |
| 2 | C3D Loss | 2015 | Video Understanding | 3D conv generic spatiotemporal features |
| 3 | Video GAN / VGAN | 2016 | Video Generation | Spatio-temporal 3D conv discriminator |
| 4 | TGAN Loss | 2017 | Video Generation | Temporal + image generator decomposition |
| 5 | I3D Loss | 2017 | Action Recognition | Inflated 2D→3D with Kinetics pre-training |
| 6 | MoCoGAN Loss | 2018 | Video Generation | Motion/content codes + dual discriminators |
| 7 | DVD-GAN Loss | 2019 | Video Generation | Spatial + temporal dual discriminator |
| 8 | SlowFast Loss | 2019 | Action Recognition | Dual-rate pathway with lateral connections |
| 9 | VideoGPT Loss | 2021 | Video Generation | VQ-VAE tokens + autoregressive cross-entropy |
| 10 | TimeSformer Loss | 2021 | Action Recognition | Divided space-time self-attention |
| 11 | VideoCLIP Loss | 2021 | Video-Language | Overlapping temporal contrastive pairs |
| 12 | Frozen in Time Loss | 2021 | Video-Language | Image→video curriculum contrastive learning |
| 13 | Video Diffusion Loss | 2022 | Video Generation | 3D U-Net denoising + joint image-video training |
| 14 | Make-A-Video Loss | 2022 | Video Generation | Image-text diffusion extended to video |
| 15 | Imagen Video Loss | 2022 | Video Generation | Cascaded diffusion with v-prediction |
| 16 | CogVideo Loss | 2022 | Video Generation | Multi-frame-rate hierarchical autoregressive |
| 17 | VideoMAE Loss | 2022 | Video Understanding | 90–95% masked spatiotemporal reconstruction |
| 18 | InternVideo Loss | 2022 | Video Understanding | Masked modeling + contrastive unification |
| 19 | Stable Video Diffusion Loss | 2023 | Video Generation | EDM framework + curated video pre-training |
| 20 | VideoPoet Loss | 2024 | Video Generation | Multimodal LLM with MAGVIT-v2 tokenizer |
