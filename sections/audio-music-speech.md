# Audio, Music & Speech Generation — Loss Functions

> A chronological catalog of loss functions for speech synthesis, neural vocoding, music generation, audio compression, and voice processing.

---

## Part I — Speech Synthesis & Vocoding

**WaveNet Autoregressive Loss** (2016) — Softmax cross-entropy over mu-law quantized 8-bit audio samples for autoregressive waveform generation, modeling the conditional distribution of each sample given all previous ones.
📄 [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) — Aäron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu
💻 [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)

---

**Tacotron 2 Mel-Spectrogram Loss** (2017) — L1 and L2 regression on predicted mel-spectrograms (before and after post-net) plus binary cross-entropy on the stop token for end-to-end text-to-speech.
📄 [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) — Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, Yonghui Wu
💻 [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2)

---

**WaveRNN Dual Softmax Loss** (2018) — Dual softmax cross-entropy over coarse and fine 8-bit components of 16-bit audio samples, enabling efficient single-RNN autoregressive synthesis.
📄 [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435) — Nal Kalchbrenner, Erich Elsen, Karen Simonyan, Seb Noury, Norman Casagrande, Edward Lockhart, Florian Stimberg, Aäron van den Oord, Sander Dieleman, Koray Kavukcuoglu
💻 [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)

---

**Multi-Resolution STFT Loss** (2019) — Sum of spectral convergence and log-magnitude STFT losses computed at multiple FFT sizes, window lengths, and hop sizes for parallel waveform generation without adversarial training.
📄 [Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://arxiv.org/abs/1910.11480) — Ryuichi Yamamoto, Eunwoo Song, Jae-Min Kim
💻 [csteinmetz1/auraloss](https://github.com/csteinmetz1/auraloss)

---

**Parallel WaveGAN Loss** (2020) — Combines multi-resolution STFT loss with a waveform-domain adversarial loss for fast, non-autoregressive neural vocoding that matches autoregressive quality.
📄 [Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://arxiv.org/abs/1910.11480) — Ryuichi Yamamoto, Eunwoo Song, Jae-Min Kim
💻 [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)

---

**HiFi-GAN Loss** (2020) — Multi-period discriminator (MPD) + multi-scale discriminator (MSD) adversarial losses combined with mel-spectrogram reconstruction loss and feature matching loss for high-fidelity, efficient neural vocoding.
📄 [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646) — Jungil Kong, Jaehyeon Kim, Jaekyoung Bae
💻 [jik876/hifi-gan](https://github.com/jik876/hifi-gan)

---

**Glow-TTS Loss** (2020) — Flow-based maximum likelihood estimation with monotonic alignment search (MAS) for parallel text-to-speech, combining log-likelihood of the flow and duration prediction loss.
📄 [Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search](https://arxiv.org/abs/2005.11129) — Jaehyeon Kim, Sungwon Kim, Jungil Kong, Sungroh Yoon
💻 [jaywalnut310/glow-tts](https://github.com/jaywalnut310/glow-tts)

---

**VITS Loss** (2021) — Variational inference with normalizing flows combining VAE reconstruction loss, KL divergence with flow-based prior, adversarial loss (HiFi-GAN discriminators), and monotonic alignment for fully end-to-end TTS.
📄 [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) — Jaehyeon Kim, Jungil Kong, Juhee Son
💻 [jaywalnut310/vits](https://github.com/jaywalnut310/vits)

---

## Part II — Audio & Music Generation

**SampleRNN Hierarchical Autoregressive Loss** (2017) — Cross-entropy loss applied at multiple temporal resolutions (frame-level and sample-level modules) for unconditional end-to-end neural audio generation.
📄 [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837) — Soroush Mehri, Kundan Kumar, Ishaan Gulrajani, Rithesh Kumar, Shubham Jain, Jose Sotelo, Aaron Courville, Yoshua Bengio
💻 [soroushmehr/sampleRNN_ICLR2017](https://github.com/soroushmehr/sampleRNN_ICLR2017)

---

**NSynth WaveNet Autoencoder Loss** (2017) — Reconstruction loss combining a WaveNet decoder with a temporal encoder, learning a latent space of musical notes for neural audio synthesis and interpolation.
📄 [Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders](https://arxiv.org/abs/1704.01279) — Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Mohammad Norouzi, Douglas Eck, Karen Simonyan
💻 [magenta/magenta](https://github.com/magenta/magenta)

---

**SoundStream RVQ Loss** (2021) — Residual vector quantization codec loss combining waveform reconstruction, VQ commitment loss, codebook loss, and multi-scale adversarial + feature matching losses for real-time neural audio compression.
📄 [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312) — Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, Marco Tagliasacchi
💻 [lucidrains/audiolm-pytorch](https://github.com/lucidrains/audiolm-pytorch)

---

**EnCodec Loss** (2022) — Multi-scale STFT adversarial loss (MS-STFT discriminator) combined with multi-scale mel-spectrogram reconstruction, VQ commitment loss, and perceptual loss for high-fidelity neural audio compression at multiple bitrates.
📄 [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Alexandre Défossez, Jade Copet, Gabriel Synnaeve, Yossi Adi
💻 [facebookresearch/encodec](https://github.com/facebookresearch/encodec)

---

**AudioLDM Loss** (2023) — Latent diffusion denoising loss conditioned on CLAP audio-text embeddings, training a U-Net to denoise mel-spectrogram latents for text-to-audio generation.
📄 [AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://arxiv.org/abs/2301.12503) — Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, Mark D. Plumbley
💻 [haoheliu/AudioLDM](https://github.com/haoheliu/AudioLDM)

---

**MusicGen Cross-Entropy Loss** (2023) — Cross-entropy on a delayed pattern of residual vector quantization tokens, enabling single-stage autoregressive music generation conditioned on text or melody.
📄 [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) — Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez
💻 [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)

---

**VampNet Masked Token Prediction Loss** (2023) — Cross-entropy on masked acoustic tokens at multiple RVQ levels, enabling parallel music generation and inpainting via iterative decoding with variable masking schedules.
📄 [VampNet: Music Generation via Masked Acoustic Token Modeling](https://arxiv.org/abs/2307.04686) — Hugo Flores García, Prem Seetharaman, Rithesh Kumar, Bryan Pardo
💻 [hugofloresgarcia/vampnet](https://github.com/hugofloresgarcia/vampnet)

---

**Stable Audio Latent Diffusion Loss** (2024) — Latent diffusion denoising loss with timing embeddings (start time, duration) conditioning on a variational autoencoder latent space for variable-length, long-form audio generation.
📄 [Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion](https://arxiv.org/abs/2402.04825) — Zach Evans, CJ Carr, Josiah Taylor, Scott H. Hawley, Jordi Pons
💻 [Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools)

---

## Part III — Voice Conversion & Separation

**AutoVC Reconstruction Loss** (2019) — Self-reconstruction loss with an information bottleneck on the content encoder, forcing disentanglement of content and speaker identity for zero-shot voice conversion.
📄 [AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://arxiv.org/abs/1905.05879) — Kaizhi Qian, Yang Zhang, Shiyu Chang, Xuesong Yang, Mark Hasegawa-Johnson
💻 [auspicious3000/autovc](https://github.com/auspicious3000/autovc)

---

**SI-SNR Loss (Scale-Invariant Signal-to-Noise Ratio)** (2019) — Scale-invariant signal-to-noise ratio that normalizes for arbitrary signal scaling, serving as the standard training objective for deep speech separation networks.
📄 [Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation](https://arxiv.org/abs/1809.07454) — Yi Luo, Nima Mesgarani
💻 [asteroid-team/asteroid](https://github.com/asteroid-team/asteroid)

---

**PIT Loss (Permutation Invariant Training)** (2017) — Computes the loss for all possible output-to-speaker permutations and selects the minimum, solving the label ambiguity problem in multi-speaker speech separation.
📄 [Permutation Invariant Training of Deep Models for Speaker-Independent Multi-talker Speech Separation](https://arxiv.org/abs/1607.00325) — Dong Yu, Morten Kolbæk, Zheng-Hua Tan, Jesper Jensen
💻 [asteroid-team/asteroid](https://github.com/asteroid-team/asteroid)

---

## Unified Libraries

| Library | Repository | Description |
|---|---|---|
| auraloss | [csteinmetz1/auraloss](https://github.com/csteinmetz1/auraloss) | Collection of audio-focused loss functions in PyTorch (multi-resolution STFT, mel-spectrogram, and perceptual losses) |
| ESPnet | [espnet/espnet](https://github.com/espnet/espnet) | End-to-end speech processing toolkit with built-in TTS and vocoder losses (Tacotron, VITS, HiFi-GAN, Parallel WaveGAN) |
| Coqui TTS | [coqui-ai/TTS](https://github.com/coqui-ai/TTS) | Deep learning toolkit for text-to-speech with implementations of VITS, Tacotron 2, Glow-TTS, and associated loss functions |
| audiocraft | [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) | Meta's audio generation library including EnCodec, MusicGen, and AudioGen with their training losses |

---

## Quick Reference Table

| # | Loss Function | Year | Paper / Method | Key Idea |
|---|---|---|---|---|
| 1 | WaveNet Autoregressive Loss | 2016 | van den Oord et al. | Softmax over mu-law quantized samples |
| 2 | Tacotron 2 Mel-Spectrogram Loss | 2017 | Shen et al. | L1/L2 mel regression + stop token BCE |
| 3 | WaveRNN Dual Softmax Loss | 2018 | Kalchbrenner et al. | Coarse + fine 8-bit dual softmax |
| 4 | Multi-Resolution STFT Loss | 2019 | Yamamoto et al. | Spectral convergence + log-mag at multiple FFT sizes |
| 5 | Parallel WaveGAN Loss | 2020 | Yamamoto et al. | Multi-res STFT + adversarial |
| 6 | HiFi-GAN Loss | 2020 | Kong et al. | MPD + MSD adversarial + mel + feature matching |
| 7 | Glow-TTS Loss | 2020 | Kim et al. | Flow-based MLE + monotonic alignment search |
| 8 | VITS Loss | 2021 | Kim et al. | VAE + KL + adversarial + flow alignment |
| 9 | SampleRNN Loss | 2017 | Mehri et al. | Hierarchical multi-resolution autoregressive |
| 10 | NSynth WaveNet Autoencoder Loss | 2017 | Engel et al. | Temporal autoencoder reconstruction |
| 11 | SoundStream RVQ Loss | 2021 | Zeghidour et al. | Reconstruction + commitment + codebook + adversarial |
| 12 | EnCodec Loss | 2022 | Défossez et al. | MS-STFT discriminator + VQ commitment + mel |
| 13 | AudioLDM Loss | 2023 | Liu et al. | Latent diffusion conditioned on CLAP embeddings |
| 14 | MusicGen Cross-Entropy Loss | 2023 | Copet et al. | Delayed-pattern RVQ token cross-entropy |
| 15 | VampNet Masked Token Loss | 2023 | Flores García et al. | Masked acoustic token prediction |
| 16 | Stable Audio Latent Diffusion Loss | 2024 | Evans et al. | Timing-conditioned latent diffusion |
| 17 | AutoVC Reconstruction Loss | 2019 | Qian et al. | Content bottleneck self-reconstruction |
| 18 | SI-SNR Loss | 2019 | Luo & Mesgarani | Scale-invariant signal-to-noise ratio |
| 19 | PIT Loss | 2017 | Yu et al. | Minimum over all permutation assignments |
