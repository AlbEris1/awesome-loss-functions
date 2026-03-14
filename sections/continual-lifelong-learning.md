# Continual Learning & Lifelong Learning — Loss Functions

> A chronological catalog of the most important loss functions and regularization methods for continual learning, lifelong learning, and catastrophic forgetting prevention.

---

## Methods

---

**Progressive Neural Networks** (2016) — Architecture-based approach using lateral connections to frozen columns, achieving zero forgetting by design.
📄 [Progressive Neural Networks](https://arxiv.org/abs/1606.04671) — Andrei A. Rusu, Neil C. Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, Raia Hadsell
💻 [Implementation](https://github.com/ContinualAI/avalanche) *(available in Avalanche)*

---

**Learning without Forgetting (LwF) Loss** (2016/2017) — Knowledge distillation loss that preserves old task outputs using only new task data, preventing catastrophic forgetting without storing exemplars.
📄 [Learning without Forgetting](https://arxiv.org/abs/1606.09282) — Zhizhong Li, Derek Hoiem
💻 [Implementation](https://github.com/lizhitwo/LearningWithoutForgetting)

---

**Elastic Weight Consolidation (EWC) Loss** (2017) — Quadratic penalty weighted by the Fisher information matrix that selectively slows down learning on weights important for previous tasks.
📄 [Overcoming Catastrophic Forgetting in Neural Networks](https://arxiv.org/abs/1612.00796) — James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, Raia Hadsell
💻 [Implementation](https://github.com/ContinualAI/avalanche) *(available in Avalanche & Mammoth)*

---

**Synaptic Intelligence (SI) Loss** (2017) — Online importance estimation that accumulates per-parameter contributions along the training trajectory, penalizing changes to important synapses.
📄 [Continual Learning Through Synaptic Intelligence](https://arxiv.org/abs/1703.04200) — Friedemann Zenke, Ben Poole, Surya Ganguli
💻 [Implementation](https://github.com/ganguli-lab/pathint)

---

**PackNet Loss** (2018) — Iterative pruning and re-training strategy that sequentially packs multiple tasks into a single network by freeing redundant parameters for new tasks.
📄 [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://arxiv.org/abs/1711.05769) — Arun Mallya, Svetlana Lazebnik
💻 [Implementation](https://github.com/arunmallya/packnet)

---

**Memory Aware Synapses (MAS) Loss** (2018) — Unsupervised importance weight estimation based on the sensitivity of the learned function output to parameter changes, penalizing modifications to critical weights.
📄 [Memory Aware Synapses: Learning What (not) to Forget](https://arxiv.org/abs/1711.09601) — Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, Tinne Tuytelaars
💻 [Implementation](https://github.com/ContinualAI/avalanche) *(available in Avalanche)*

---

**Averaged Gradient Episodic Memory (A-GEM) Loss** (2019) — Efficient projected gradient descent that constrains updates to not increase loss on episodic memory samples, using averaged gradients over the memory buffer.
📄 [Efficient Lifelong Learning with A-GEM](https://arxiv.org/abs/1812.00420) — Arslan Chaudhry, Marc'Aurelio Ranzato, Marcus Rohrbach, Mohamed Elhoseiny
💻 [Implementation](https://github.com/facebookresearch/agem)

---

**Experience Replay (ER) Loss** (2019) — Simple yet effective baseline combining reservoir sampling with balanced replay, jointly training on current task data and stored episodic memories.
📄 [On Tiny Episodic Memories in Continual Learning](https://arxiv.org/abs/1902.10486) — Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K. Dokania, Philip H. S. Torr, Marc'Aurelio Ranzato
💻 [Implementation](https://github.com/aimagelab/mammoth) *(available in Mammoth)*

---

**Bias Correction (BiC) Loss** (2019) — Linear bias correction layer appended after the classification head to counteract the strong bias toward new classes in class-incremental learning.
📄 [Large Scale Incremental Learning](https://arxiv.org/abs/1905.13260) — Yue Wu, Yinpeng Chen, Lijuan Wang, Yuancheng Ye, Zicheng Liu, Yandong Guo, Yun Fu
💻 [Implementation](https://github.com/wuyuebupt/LargeScaleIncrementalLearning)

---

**PODNet Loss** (2020) — Pooled Outputs Distillation that constrains intermediate representations across the network using spatial-based distillation with multi-proxy class vectors.
📄 [PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning](https://arxiv.org/abs/2004.13513) — Arthur Douillard, Matthieu Cord, Charles Ollion, Thomas Robert, Eduardo Valle
💻 [Implementation](https://github.com/arthurdouillard/incremental_learning.pytorch)

---

**DER / DER++ Loss** (2020/2021) — Dark Experience Replay that stores and replays soft logits alongside raw samples, matching the network's past predictions to promote consistency across the training trajectory.
📄 [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://arxiv.org/abs/2004.07211) — Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara
💻 [Implementation](https://github.com/aimagelab/mammoth)

---

**Co²L Loss** (2021) — Asymmetric supervised contrastive loss for continual learning that learns transferable representations and preserves them via self-supervised distillation across tasks.
📄 [Co²L: Contrastive Continual Learning](https://arxiv.org/abs/2106.14413) — Hyuntak Cha, Jaeho Lee, Jinwoo Shin
💻 [Implementation](https://github.com/chaht01/Co2L)

---

**FOSTER Loss** (2022) — Two-stage gradient-boosting-inspired paradigm that dynamically expands modules to fit residuals, then compresses via distillation to maintain a single backbone.
📄 [FOSTER: Feature Boosting and Compression for Class-Incremental Learning](https://arxiv.org/abs/2204.04662) — Fu-Yun Wang, Da-Wei Zhou, Han-Jia Ye, De-Chuan Zhan
💻 [Implementation](https://github.com/G-U-N/ECCV22-FOSTER)

---

**MEMO Loss** (2022) — Memory-efficient expandable model that extends specialized layers on top of shared generalized representations, balancing model expansion cost with exemplar storage.
📄 [A Model or 603 Exemplars: Towards Memory-Efficient Class-Incremental Learning](https://arxiv.org/abs/2205.13218) — Da-Wei Zhou, Qi-Wei Wang, Han-Jia Ye, De-Chuan Zhan
💻 [Implementation](https://github.com/wangkiw/ICLR23-MEMO)

---

**L2P Loss** (2022) — Prompt-pool-based continual learning that dynamically selects and optimizes small learnable prompt tokens attached to a frozen pre-trained vision transformer.
📄 [Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654) — Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, Tomas Pfister
💻 [Implementation](https://github.com/google-research/l2p)

---

**DualPrompt Loss** (2022) — Complementary prompting framework that attaches task-invariant (G-Prompt) and task-specific (E-Prompt) instructions to a frozen backbone for rehearsal-free continual learning.
📄 [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/abs/2204.04799) — Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoxi Sun, Han Zhang, Chen-Yu Lee, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, Tomas Pfister
💻 [Implementation](https://github.com/google-research/l2p)

---

**CODA-Prompt Loss** (2023) — End-to-end attention-based prompt composition that assembles prompt components with input-conditioned weights, improving plasticity over key-query prompt methods.
📄 [CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning](https://arxiv.org/abs/2211.13218) — James Seale Smith, Leonid Karlinsky, Vyshnavi Gutta, Paola Cascante-Bonilla, Donghyun Kim, Assaf Arbelle, Rameswar Panda, Rogerio Feris, Zsolt Kira
💻 [Implementation](https://github.com/GT-RIPL/CODA-Prompt)

---

**EASE Loss** (2024) — Expandable subspace ensemble that trains lightweight task-specific adapters spanning a high-dimensional feature space, with semantic-guided prototype complement for old classes.
📄 [Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning](https://arxiv.org/abs/2403.12030) — Da-Wei Zhou, Hai-Long Sun, Han-Jia Ye, De-Chuan Zhan
💻 [Implementation](https://github.com/sun-hailong/CVPR24-Ease)

---

## Quick Reference Table

| Method | Year | Category | Rehearsal-Free | Key Mechanism |
|--------|------|----------|:--------------:|---------------|
| Progressive Nets | 2016 | Architecture | ✅ | Frozen columns + lateral connections |
| LwF | 2016 | Distillation | ✅ | Knowledge distillation from old task outputs |
| EWC | 2017 | Regularization | ✅ | Fisher information matrix penalty |
| SI | 2017 | Regularization | ✅ | Online per-synapse importance accumulation |
| PackNet | 2018 | Architecture | ✅ | Iterative pruning and parameter packing |
| MAS | 2018 | Regularization | ✅ | Output sensitivity-based importance weights |
| A-GEM | 2019 | Replay | ❌ | Projected gradient with episodic memory |
| ER | 2019 | Replay | ❌ | Reservoir sampling + joint training |
| BiC | 2019 | Bias Correction | ❌ | Linear bias correction layer |
| PODNet | 2020 | Distillation | ❌ | Spatial pooled outputs distillation |
| DER/DER++ | 2020 | Replay | ❌ | Dark knowledge replay (stored logits) |
| Co²L | 2021 | Contrastive | ❌ | Asymmetric supervised contrastive loss |
| FOSTER | 2022 | Architecture | ❌ | Gradient boosting + feature compression |
| MEMO | 2022 | Architecture | ❌ | Shared backbone + expandable specialized layers |
| L2P | 2022 | Prompt-based | ✅ | Learnable prompt pool for frozen ViT |
| DualPrompt | 2022 | Prompt-based | ✅ | Complementary G-Prompt + E-Prompt |
| CODA-Prompt | 2023 | Prompt-based | ✅ | Attention-based prompt decomposition |
| EASE | 2024 | Architecture | ❌ | Task-specific adapter subspace ensemble |

---

## Unified Libraries & Frameworks

These libraries provide standardized implementations of many of the methods listed above:

- **[Avalanche](https://github.com/ContinualAI/avalanche)** — End-to-end continual learning library by ContinualAI. Includes EWC, SI, LwF, MAS, A-GEM, ER, and many more with unified training loops and benchmarks.
- **[Mammoth](https://github.com/aimagelab/mammoth)** — Modular framework for continual learning research. Implements DER/DER++, ER, EWC, SI, A-GEM, and other baselines with a clean, extensible codebase.
- **[PyCIL](https://github.com/G-U-N/PyCIL)** — Toolbox focused on class-incremental learning. Covers FOSTER, MEMO, PODNet, BiC, LwF, EASE, and other CIL-specific methods with consistent evaluation protocols.
