# Time Series Forecasting — Loss Functions

> A chronological catalog of loss functions for time series forecasting, temporal prediction, and sequential modeling.

---

## Part I — Classical & Probabilistic Forecasting Losses

---

**1. Mean Absolute Error (MAE) / L1 Loss** (Classical) — The average absolute difference between predicted and true values; robust to outliers and widely used as a baseline point forecasting loss.
📄 [Least Absolute Deviations (Wikipedia)](https://en.wikipedia.org/wiki/Least_absolute_deviations) — Classical statistical method
💻 [PyTorch `torch.nn.L1Loss`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/loss.py)

---

**2. Mean Squared Error (MSE) / L2 Loss** (Classical) — The average squared difference between predicted and true values; penalizes large errors disproportionately, making it sensitive to outliers.
📄 [Least Squares (Wikipedia)](https://en.wikipedia.org/wiki/Least_squares) — Classical statistical method (Gauss, Legendre)
💻 [PyTorch `torch.nn.MSELoss`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/loss.py)

---

**3. Huber Loss** (1964) — A piecewise loss that behaves as L2 for small errors and L1 for large errors, combining MSE's smoothness with MAE's robustness to outliers.
📄 [Robust Estimation of a Location Parameter](https://doi.org/10.1214/aoms/1177703732) — Peter J. Huber
💻 [PyTorch `torch.nn.HuberLoss`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/loss.py)

---

**4. Quantile Loss / Pinball Loss** (1978) — Asymmetric loss that penalizes over- and under-prediction differently based on a chosen quantile, enabling prediction interval estimation and probabilistic forecasting.
📄 [Regression Quantiles](https://doi.org/10.2307/1913643) — Roger Koenker, Gilbert Bassett Jr.
💻 [GluonTS QuantileLoss](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/modules/loss.py)

---

**5. MAPE (Mean Absolute Percentage Error)** (Classical) — Scale-independent percentage error measuring relative forecast accuracy; undefined when true values are zero and asymmetrically penalizes positive vs. negative errors.
📄 [Another Look at Measures of Forecast Accuracy](https://doi.org/10.1016/j.ijforecast.2006.03.001) — Rob J. Hyndman, Anne B. Koehler (2006, critical analysis)
💻 [Nixtla/neuralforecast (MAPE metric)](https://github.com/Nixtla/neuralforecast)

---

**6. sMAPE (Symmetric MAPE)** (1999) — Symmetric variant of MAPE that normalizes by the average of predicted and true values, addressing MAPE's asymmetry but still problematic near zero.
📄 [A Better Measure of Relative Prediction Accuracy for Model Selection and Model Estimation](https://doi.org/10.1016/S0169-2070(99)00007-2) — J. Scott Armstrong
💻 [Nixtla/neuralforecast (sMAPE metric)](https://github.com/Nixtla/neuralforecast)

---

**7. MASE (Mean Absolute Scaled Error)** (2006) — Scale-free error metric that normalizes MAE by the in-sample MAE of a naive (random walk) forecast, well-defined for zero values and suitable for comparing across series.
📄 [Another Look at Measures of Forecast Accuracy](https://doi.org/10.1016/j.ijforecast.2006.03.001) — Rob J. Hyndman, Anne B. Koehler
💻 [Nixtla/neuralforecast (MASE metric)](https://github.com/Nixtla/neuralforecast)

---

**8. Negative Log-Likelihood (Gaussian)** (Classical) — Probabilistic forecasting loss that jointly learns the predicted mean and variance of a Gaussian distribution, penalizing both inaccurate point predictions and miscalibrated uncertainty.
📄 [Pattern Recognition and Machine Learning, §1.2.4](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — Christopher M. Bishop (2006)
💻 [GluonTS GaussianOutput](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/distributions/gaussian.py)

---

**9. CRPS (Continuous Ranked Probability Score)** (2007) — A proper scoring rule for probabilistic forecasts that measures the integrated squared difference between the predicted CDF and the empirical CDF of the observation, generalizing MAE to distributions.
📄 [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437) — Tilmann Gneiting, Adrian E. Raftery
💻 [GluonTS EnergyScore / CRPS](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/ev/metrics.py)

---

**10. DILATE Loss** (2019) — Combines a shape-based loss (soft-DTW) with a Temporal Distortion Index (TDI) penalty, jointly optimizing for both shape accuracy and temporal alignment in time series prediction.
📄 [Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models](https://arxiv.org/abs/1909.09020) — Vincent Le Guen, Nicolas Thome
💻 [vincent-leguen/DILATE](https://github.com/vincent-leguen/DILATE)

---

## Part II — Deep Learning Forecasting Losses

---

**11. DeepAR Loss** (2020) — Autoregressive RNN trained with the negative log-likelihood of parametric distributions (Gaussian, negative binomial, beta, etc.), producing calibrated probabilistic forecasts via ancestral sampling.
📄 [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://doi.org/10.1016/j.ijforecast.2019.07.001) — David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski
💻 [awslabs/gluonts (DeepAR)](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/deepar/module.py)

---

**12. N-BEATS Loss** (2020) — Interpretable deep architecture using basis expansion with backward/forward residual stacking; trained with MAPE, sMAPE, or MASE losses depending on the evaluation metric, achieving pure DL state-of-the-art on M4.
📄 [N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting](https://arxiv.org/abs/1905.10437) — Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, Yoshua Bengio
💻 [ServiceNow/N-BEATS](https://github.com/ServiceNow/N-BEATS)

---

**13. Informer Loss** (2021) — MSE loss applied to long-sequence time series forecasting with ProbSparse self-attention and generative-style decoder, enabling direct multi-step prediction without autoregressive accumulation of error.
📄 [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) — Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang
💻 [zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020)

---

**14. Autoformer Loss** (2021) — MSE loss with a novel auto-correlation mechanism replacing standard self-attention, combined with progressive series decomposition (trend + seasonal) for long-term forecasting.
📄 [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008) — Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long
💻 [thuml/Autoformer](https://github.com/thuml/Autoformer)

---

**15. FEDformer Loss** (2022) — MSE loss with frequency-enhanced attention that operates in the Fourier/wavelet domain, capturing global temporal patterns with linear complexity for long-term forecasting.
📄 [FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting](https://arxiv.org/abs/2201.12740) — Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, Rong Jin
💻 [MAZiqing/FEDformer](https://github.com/MAZiqing/FEDformer)

---

**16. PatchTST Loss** (2023) — MSE loss with channel-independent patching that segments time series into subseries-level patches fed to a vanilla Transformer, reducing computation and capturing local semantic information for multivariate forecasting.
📄 [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730) — Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam
💻 [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)

---

**17. TimesNet Loss** (2023) — MSE loss with a 2D variation modeling approach that uses FFT-based period detection to reshape 1D time series into 2D tensors, capturing both intra-period and inter-period variations via 2D convolutions.
📄 [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://arxiv.org/abs/2210.02186) — Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long
💻 [thuml/Time-Series-Library (TimesNet)](https://github.com/thuml/Time-Series-Library)

---

**18. iTransformer Loss** (2024) — MSE loss with an inverted Transformer architecture that applies attention on the variate dimension (not time), treating each time series as a token to capture multivariate correlations more effectively.
📄 [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625) — Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long
💻 [thuml/iTransformer](https://github.com/thuml/iTransformer)

---

**19. TimesFM Loss** (2024) — Patched decoder-only transformer foundation model trained on a large corpus of real-world and synthetic time series, using quantile heads (quantile loss) for probabilistic forecasting with zero-shot generalization.
📄 [A Decoder-Only Foundation Model for Time-Series Forecasting](https://arxiv.org/abs/2310.10688) — Abhimanyu Das, Weihao Kong, Rajat Sen, Yichen Zhou
💻 [google-research/timesfm](https://github.com/google-research/timesfm)

---

## Part III — Specialized Time Series Losses

---

**20. DTW Loss (Dynamic Time Warping)** (1978) — Alignment-based distance that finds the optimal non-linear warping path between two time series, allowing temporal distortion; non-differentiable in its original form.
📄 [Dynamic Programming Algorithm Optimization for Spoken Word Recognition](https://doi.org/10.1109/TASSP.1978.1163055) — Hiroaki Sakoe, Seibi Chiba
💻 [tslearn DTW](https://github.com/tslearn-team/tslearn)

---

**21. Soft-DTW Loss** (2017) — A differentiable relaxation of DTW that replaces the hard minimum with a soft-minimum (log-sum-exp), enabling gradient-based optimization of DTW-like alignment losses for time series.
📄 [Soft-DTW: a Differentiable Loss Function for Time-Series](https://arxiv.org/abs/1703.01541) — Marco Cuturi, Mathieu Blondel
💻 [mblondel/soft-dtw](https://github.com/mblondel/soft-dtw)

---

**22. TDI (Temporal Distortion Index)** (2019) — Measures the temporal alignment quality between predicted and true time series by computing the area between the DTW warping path and the diagonal, quantifying how much temporal distortion exists.
📄 [Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models](https://arxiv.org/abs/1909.09020) — Vincent Le Guen, Nicolas Thome
💻 [vincent-leguen/DILATE (TDI component)](https://github.com/vincent-leguen/DILATE)

---

**23. Variational Inference Loss for State Space Models** (2018) — ELBO-based loss for deep state space models that combines a reconstruction term (negative log-likelihood) with a KL divergence regularizer, enabling probabilistic forecasting with learned latent dynamics.
📄 [Deep State Space Models for Time Series Forecasting](https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html) — Syama Sundar Rangapuram, Matthias W. Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, Tim Januschowski
💻 [awslabs/gluonts (DeepState)](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/deep_state/module.py)

---

## Unified Libraries

| Library | Description | Link |
|---------|-------------|------|
| **GluonTS** | AWS probabilistic time series modeling (DeepAR, DeepState, Transformer, etc.) | [awslabs/gluonts](https://github.com/awslabs/gluonts) |
| **NeuralForecast** | Nixtla's production-ready neural forecasting (N-BEATS, NHITS, PatchTST, etc.) | [Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast) |
| **pytorch-forecasting** | High-level PyTorch forecasting API (TFT, DeepAR, N-BEATS, etc.) | [jdb78/pytorch-forecasting](https://github.com/jdb78/pytorch-forecasting) |
| **TSlib (Time-Series-Library)** | Unified benchmark for time series (Informer, Autoformer, TimesNet, iTransformer, etc.) | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) |

---

## 📊 Summary Table

| # | Loss Function | Year | Category | Key Innovation |
|---|---|---|---|---|
| 1 | MAE / L1 Loss | Classical | Point | Absolute error, outlier-robust |
| 2 | MSE / L2 Loss | Classical | Point | Squared error, smooth gradients |
| 3 | Huber Loss | 1964 | Point | L1/L2 hybrid, robust |
| 4 | Quantile / Pinball Loss | 1978 | Probabilistic | Asymmetric quantile regression |
| 5 | MAPE | Classical | Point | Scale-independent percentage error |
| 6 | sMAPE | 1999 | Point | Symmetric percentage error |
| 7 | MASE | 2006 | Point | Scaled by naive forecast baseline |
| 8 | Gaussian NLL | Classical | Probabilistic | Learned mean + variance |
| 9 | CRPS | 2007 | Probabilistic | Proper scoring rule for CDFs |
| 10 | DILATE Loss | 2019 | Shape+Temporal | Soft-DTW + temporal distortion |
| 11 | DeepAR Loss | 2020 | Probabilistic | Autoregressive parametric NLL |
| 12 | N-BEATS Loss | 2020 | Point | Basis expansion + residual stacking |
| 13 | Informer Loss | 2021 | Point | ProbSparse attention + MSE |
| 14 | Autoformer Loss | 2021 | Point | Auto-correlation + decomposition |
| 15 | FEDformer Loss | 2022 | Point | Fourier/wavelet attention + MSE |
| 16 | PatchTST Loss | 2023 | Point | Channel-independent patching + MSE |
| 17 | TimesNet Loss | 2023 | Point | FFT-based 2D variation + MSE |
| 18 | iTransformer Loss | 2024 | Point | Inverted attention (variate dim) |
| 19 | TimesFM Loss | 2024 | Probabilistic | Foundation model + quantile heads |
| 20 | DTW Loss | 1978 | Alignment | Non-linear temporal warping |
| 21 | Soft-DTW Loss | 2017 | Alignment | Differentiable DTW relaxation |
| 22 | TDI | 2019 | Alignment | Warping path distortion area |
| 23 | VI Loss (Deep SSM) | 2018 | Probabilistic | ELBO for latent state dynamics |
