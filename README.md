# Time Series Library (TSLib)

<!-- report-home-20261030 -->
> **实验报告总入口（2026-11-01更新）**：[最近三轮诊断结论与证据链](docs/experiments/REPORTS_OVERVIEW_20261101.html) · [检索全部413项报告/方案](docs/experiments/REPORTS_HOME_20261030.html) · [空间整理与保留说明](docs/experiments/REPORTS_STORAGE_20261101.html)。最近结果以读出、驱动、边界三轮完整报告为准；下方旧进展按历史记录保留。
<!-- /report-home-20261030 -->

- 2026-10-09补测：[MA1贝叶斯关系四组test](docs/experiments/PDMBayesianRelation_test_20261009.md)：完整test已完成并独立核验；MSE最低quadratic 0.3198047679，MAE最低quadratic 0.3569156186；四组均沿用验证所选第2轮。

- 2026-10-09完成：[MA1贝叶斯关系四组结果](docs/experiments/PDMBayesianRelation_results_20261009.md)：四组7轮联合早停，均选第2轮；mean/bregman MSE约降0.093%/0.094%，MAE约升0.082%/0.191%；quadratic两项均退化；无明确整体收益。242源码及独立NumPy审计通过，无test。

- 2026-10-09按用户要求停止：[MA1条件转移Bregman三组实验](docs/experiments/PDMTransitionBregman_implementation_20261009.md)：完成1轮，保留检查点，无完整效果结论；转入贝叶斯关系实验。

- 2026-10-09核验完成：[整体逐真值窗口协方差损失结果](docs/experiments/PDMTargetNLL_results_20261009.md)：MA1纯高斯形式分数替换总MSE，验证MSE 0.39519635→0.60943552（+54.21%），MAE 0.41170477→0.52391361（+27.25%）；分数下降但幅值比0.7885→0.2441。26轮联合早停，独立离线审计通过，无本轮test。

- 2026-10-09启动：[整体真值窗口协方差损失](docs/experiments/PDMTargetNLL_implementation_20261009.md)：MA1纯总MSE对逐真值窗口Σ的高斯形式分数；9项CPU、1项CUDA及精确恢复/对照复现通过。完整train/val训练中，尚无效果结论。

- 2026-10-09实现并预检：[季节分支NLL](docs/experiments/PDMSeasonalBranchNLL_implementation_20261009.md)：趋势MSE＋季节MSE对趋势MSE＋季节NLL；10项新测试与7项旧回归通过，CUDA两轮捕获/恢复/冻结通过。仅32/16窗口功能预检，未启动正式训练。

- 2026-10-09完成：[MA1残差可预测性诊断](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/PDMResidualPredictability_results_20261009.md)：旧半段训练基线的训练区间内时间留出诊断；预选历史探针MSE 0.536040→0.439493（−18.01%），MAE +2.72%，仅HUFL/MUFL改善。非完整训练基线收益、非val/test结果；离线复核通过，无新模型forward。

- 2026-10-09核验完成：[MA1预测自协方差NLL结果](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/PDMPredictionNLL_results_20261008.md)：纯NLL点预测失败：验证MSE 0.39519635→1.74497449（+341.55%），MAE +137.01%；NLL 1.123311→−0.054145。七变量MSE/MAE均退化。17轮联合早停，独立离线核验通过，无本轮test。

- 2026-10-08启动：[MA1预测自协方差NLL配对实验](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/PDMPredictionNLL_protocol_20261008.md)：mse/nll同架构同初值，Σ=预测中心化样本自协方差Toeplitz+0.01I；纯NLL对纯MSE，完整train/val，尚无正式效果结论。

- 2026-10-08补测：[一次/三次残差MA完整测试](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/PDMResidualMA_test_results_20261008.md)：测试MSE 0.32060763→0.32160829（+0.3121%），MAE +0.0499%；三次分解未改善测试表现。两组均使用验证选定第2轮权重。

- 2026-10-08完成：[一次/三次残差MA完整配对结果](docs/experiments/PDMResidualMA_results_20261008.md)：7轮联合早停、均选第2轮；验证MSE 0.39519635→0.39268951（−0.6343%），MAE −0.7371%；HUFL/MUFL/LULL/OT的MSE退化，无本轮test。

- 2026-10-08新增：[1/2/3/4/6/8次残差分解比较](docs/experiments/PDMResidualRedecompositionSweep_results_20261008.md)：无统一最稳次数；96步尺度3次均值漂移最低，1次跨窗方差与ACF稳定性最好。仅分解，无模型。

- 2026-10-08新增：[季节残差二次MA纯分解实验](docs/experiments/PDMResidualRedecomposition_results_20261008.md)：局部均值漂移下降约47%，跨窗口方差CV上升约8%；无模型、训练或test。

> 实验记录已整理至2026-10-08：[全部实验及状态](docs/experiments/README.md) · [历史指标摘录](docs/experiments/Experiment_metrics_20261008.md) · [脚本索引](scripts/README.md) · [产物索引](tmp/README.md)。下方按日期保留的实验进展属于当时记录，当前状态以总索引为准。

- 2026-10-08：[STL与现有MA方差稳定性比较](docs/experiments/PDMSTLVarianceComparison_20261008.md)：历史窗口离线核验完成；STL季节＋剩余项未改善局部方差稳定性，纯季节项的窗口内与跨窗口结论不同，无新增训练或test。

- 2026-10-08：[PDM损失函数与概率头完整测试](docs/experiments/PDMLossFamily_test_results_20261008.md)：验证选轮不变，完整test及独立离线核验完成。
TSLib is an open-source library for deep learning researchers, especially for deep time series analysis.

> **中文文档**：[README_zh.md](./README_zh.md)

We provide a neat code base to evaluate advanced deep time series models or develop your model, which covers five mainstream tasks: **long- and short-term forecasting, imputation, anomaly detection, and classification.**

:triangular_flag_on_post:**News** (2026.04) Due to the limited bandwidth of the current maintainers, we will not be actively adding new features to this library. Since the library was originally released three years ago, many of its benchmarks may no longer be meaningful for evaluating the effectiveness or progress of current research. However, the baseline implementations remain correct. We therefore recommend seeking out newer benchmarks.

:triangular_flag_on_post:**News** (2025.12) Many thanks to the great work from [ailuntz](https://github.com/thuml/Time-Series-Library/pull/805), which provides an updated requirements and docker deployment, as well as a well-organized document. This is quite meaningful to this project and beginners.

:triangular_flag_on_post:**News** (2025.11) Considering the rapid development of Large Time Series Models (LTSMs), we have newly added a [[zero-shot forecasting]](https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_zero_shot_forecasting.py) feature in TSLib. You can try [this script](https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ETT_script/LTSM.sh) to evaluate LTSMs.

:triangular_flag_on_post:**News** (2025.10) Given the recent confusion among researchers regarding minor improvements on standard benchmarks, we propose the [[Accuracy Law]](https://arxiv.org/abs/2510.02729) to characterize the objectives of deep time series forecasting tasks, which can be used to identify saturated datasets.

:triangular_flag_on_post:**News** (2024.10) We have included [[TimeXer]](https://arxiv.org/abs/2402.19072), which defined a practical forecasting paradigm: Forecasting with Exogenous Variables. Considering both practicability and computation efficiency, we believe the new forecasting paradigm defined in TimeXer can be the "right" task for future research.

:triangular_flag_on_post:**News** (2024.10) Our lab has open-sourced [[OpenLTM]](https://github.com/thuml/OpenLTM), which provides a distinct pretrain-finetuning paradigm compared to TSLib. If you are interested in Large Time Series Models, you may find this repository helpful.

:triangular_flag_on_post:**News** (2024.07) We wrote a comprehensive survey of [[Deep Time Series Models]](https://arxiv.org/abs/2407.13278) with a rigorous benchmark based on TSLib. In this paper, we summarized the design principles of current time series models supported by insightful experiments, hoping to be helpful to future research.

:triangular_flag_on_post:**News** (2024.04) Many thanks for the great work from [frecklebars](https://github.com/thuml/Time-Series-Library/pull/378). The famous sequential model [Mamba](https://arxiv.org/abs/2312.00752) has been included in our library. See [this file](https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py), where you need to install `mamba_ssm` with pip at first.

:triangular_flag_on_post:**News** (2024.03) Given the inconsistent look-back length of various papers, we split the long-term forecasting in the leaderboard into two categories: Look-Back-96 and Look-Back-Searching. We recommend researchers read [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2), which includes both look-back length settings in experiments for scientific rigor.

:triangular_flag_on_post:**News** (2023.10) We add an implementation to [iTransformer](https://arxiv.org/abs/2310.06625), which is the state-of-the-art model for long-term forecasting. The official code and complete scripts of iTransformer can be found [here](https://github.com/thuml/iTransformer).

:triangular_flag_on_post:**News** (2023.09) We added a detailed [tutorial](https://github.com/thuml/Time-Series-Library/blob/main/tutorial/TimesNet_tutorial.ipynb) for [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq) and this library, which is quite friendly to beginners of deep time series analysis.

:triangular_flag_on_post:**News** (2023.02) We release the TSlib as a comprehensive benchmark and code base for time series models, which is extended from our previous GitHub repository [Autoformer](https://github.com/thuml/Autoformer).

## Leaderboard for Time Series Analysis

Till March 2024, the top three models for five different tasks are:

| Model<br>Ranking | Long-term<br>Forecasting<br>Look-Back-96              | Long-term<br/>Forecasting<br/>Look-Back-Searching     | Short-term<br>Forecasting                                    | Imputation                                                   | Classification                                               | Anomaly<br>Detection                               |
| ---------------- | ----------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------- |
| 🥇 1st            | [TimeXer](https://arxiv.org/abs/2402.19072)      | [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2) | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)       |
| 🥈 2nd            | [iTransformer](https://arxiv.org/abs/2310.06625) | [PatchTST](https://github.com/yuqinie98/PatchTST)     | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [FEDformer](https://github.com/MAZiqing/FEDformer) |
| 🥉 3rd            | [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2)          | [DLinear](https://arxiv.org/pdf/2205.13504.pdf)       | [FEDformer](https://github.com/MAZiqing/FEDformer)           | [Autoformer](https://github.com/thuml/Autoformer)            | [Informer](https://github.com/zhouhaoyi/Informer2020)        | [Autoformer](https://github.com/thuml/Autoformer)  |


**Note: We will keep updating this leaderboard.** If you have proposed advanced and awesome models, you can send us your paper/code link or raise a pull request. We will add them to this repo and update the leaderboard as soon as possible.

**Compared models of this leaderboard.** ☑ means that their codes have already been included in this repo.
  - [x] **TimeXer** - TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables [[NeurIPS 2024]](https://arxiv.org/abs/2402.19072) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeXer.py)
  - [x] **TimeMixer** - TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting [[ICLR 2024]](https://openreview.net/pdf?id=7oLshfEIC2) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py).
  - [x] **TSMixer** - TSMixer: An All-MLP Architecture for Time Series Forecasting [[arXiv 2023]](https://arxiv.org/pdf/2303.06053.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TSMixer.py)
  - [x] **iTransformer** - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
  - [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
  - [x] **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py).
  - [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py).
  - [x] **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]](https://arxiv.org/abs/2207.01186) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py).
  - [x] **ETSformer** - ETSformer: Exponential Smoothing Transformers for Time-series Forecasting [[arXiv 2022]](https://arxiv.org/abs/2202.01381) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py).
  - [x] **Non-stationary Transformer** - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/pdf?id=ucNDIDRNjjv) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py).
  - [x] **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py).
  - [x] **Pyraformer** - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [[ICLR 2022]](https://openreview.net/pdf?id=0EXmFzUn5I) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py).
  - [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py).
  - [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py).
  - [x] **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py).
  - [x] **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py).

See our latest paper [[TimesNet]](https://arxiv.org/abs/2210.02186) for the comprehensive benchmark. We will release a real-time updated online version soon.

**Newly added baselines.** We will add them to the leaderboard after a comprehensive evaluation.
  - [x] **MambaSL** - MambaSL: Exploring Single-Layer Mamba for Time Series Classification [[ICLR 2026]](https://openreview.net/forum?id=YDl4vqQqGP) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MambaSingleLayer.py)
  - [x] **TimeFilter** - TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting [[ICML 2025]](https://arxiv.org/abs/2501.13041) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeFilter.py)
  - [x] **KAN-AD** - KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks [[ICML 2025]](https://arxiv.org/abs/2411.00278) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/KANAD.py)
  - [x] **MultiPatchFormer** - A multiscale model for multivariate time series forecasting [[Scientific Reports 2025]](https://www.nature.com/articles/s41598-024-82417-4) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MultiPatchFormer.py)
  - [x] **WPMixer** - WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting [[AAAI 2025]](https://arxiv.org/abs/2412.17176) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/WPMixer.py)
  - [x] **MSGNet** - MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting [[AAAI 2024]](https://dl.acm.org/doi/10.1609/aaai.v38i10.28991) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MSGNet.py)
  - [x] **PAttn** - Are Language Models Actually Useful for Time Series Forecasting? [[NeurIPS 2024]](https://arxiv.org/pdf/2406.16964) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PAttn.py)
  - [x] **Mamba** - Mamba: Linear-Time Sequence Modeling with Selective State Spaces [[arXiv 2023]](https://arxiv.org/abs/2312.00752) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py)
  - [x] **SegRNN** - SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting [[arXiv 2023]](https://arxiv.org/abs/2308.11200.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/SegRNN.py).
  - [x] **Koopa** - Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors [[NeurIPS 2023]](https://arxiv.org/pdf/2305.18803.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Koopa.py).
  - [x] **FreTS** - Frequency-domain MLPs are More Effective Learners in Time Series Forecasting [[NeurIPS 2023]](https://arxiv.org/pdf/2311.06184.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FreTS.py).
  - [x] **MICN** - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=zt53IDUR1U)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MICN.py).
  - [x] **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py).
  - [x] **TiDE** - Long-term Forecasting with TiDE: Time-series Dense Encoder [[arXiv 2023]](https://arxiv.org/pdf/2304.08424.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TiDE.py).
  - [x] **SCINet** - SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction [[NeurIPS 2022]](https://openreview.net/pdf?id=AyajSjTAzmg)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/SCINet.py).
  - [x] **FiLM** - FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/forum?id=zTQdHSQUQWc)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FiLM.py).
  - [x] **TFT** - Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting [[arXiv 2019]](https://arxiv.org/abs/1912.09363)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TemporalFusionTransformer.py).

**Newly added Large Time Series Models.** This library also supports the zero-shot evaluation of the following LTSMs.

- [x] **Chronos2** - Chronos-2: From Univariate to Universal Forecasting [[arXiv 2025]](https://arxiv.org/abs/2510.15821) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Chronos2.py)
- [x] **TiRex** - TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning [[NeurIPS 2025]](https://arxiv.org/pdf/2505.23719) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TiRex.py)
- [x] **Sundial** - Sundial: A Family of Highly Capable Time Series Foundation Models [[ICML 2025]](https://arxiv.org/pdf/2502.00816) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Sundial.py)
- [x] **Time-MoE** - Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts [[ICLR 2025]](https://arxiv.org/pdf/2409.16040) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMoE.py)
- [x] **Toto** - Toto: Time Series Optimized Transformer for Observability [arXiv 2024](https://arxiv.org/pdf/2407.07874)
- [x] **Chronos** - Chronos: Learning the Language of Time Series [[TMLR 2024]](https://arxiv.org/pdf/2403.07815) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Chronos.py)
- [x] **Moirai** - Unified Training of Universal Time Series Forecasting Transformers [[ICML 2024]](https://arxiv.org/pdf/2402.02592)
- [x] **TimesFM** - A decoder-only foundation model for time-series forecasting [[ICML 2024]](https://arxiv.org/abs/2310.10688) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesFM.py)


 
## Getting Started

### Prepare Data
You can obtain the well-preprocessed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) or [[Hugging Face]](https://huggingface.co/datasets/thuml/Time-Series-Library). Then place the downloaded data in the folder `./dataset`.

### Installation
1. Clone this repository.
   ```bash
   git clone https://github.com/thuml/Time-Series-Library.git
   cd Time-Series-Library
   ```

2. Create a new Conda environment.
   ```bash
   conda create -n tslib python=3.11
   conda activate tslib
   ```

3. Install Core Dependencies
   > ⚠️ **CUDA Compatibility Notice**
   > The torch prebuilt package is **CUDA-version specific**. (See https://pytorch.org/get-started/previous-versions/)
   > Please make sure to install the package that matches your local CUDA version (e.g., `cu118` or `cu121`).
   > Recommended: torch==2.5.1

   ```bash
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

   pip install -r requirements.txt
   ```

4. Install Dependencies for Mamba Model (Required for Time-Series-Library/models/Mamba.py)
   > ⚠️ **Linux only**
   > ⚠️ **CUDA Compatibility Notice**
   > The prebuilt Mamba wheel is **CUDA-version specific**.
   > Please make sure to install the wheel that matches your local CUDA version
   > (e.g., `cu11` or `cu12`). Installing a mismatched version may result in
   > runtime errors or import failures.

   Example for **CUDA 12**:

   ```bash
   pip install https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
   ```

5. Install Dependencies for Moirai Model (Required for Time-Series-Library/models/Moirai.py)
   ```bash
   pip install uni2ts --no-deps
   ```

### Docker Deployment
```bash
# Build and start the Docker container in detached mode
docker compose -f 'Time-Series-Library/docker-compose.yml' up -d --build

# Download / place the dataset into a newly created folder ./dataset at the repository root
mkdir -p dataset  # create the dataset directory

# Copy the local dataset into the container at /workspace/dataset
docker cp ./dataset tslib:/workspace/dataset

# Enter the running container to continue training / evaluation
docker exec -it tslib bash

# Switch to the workspace directory inside the container
cd /workspace

# Run zero-shot forecasting with the pre-trained Moirai model
python -u run.py \
  --task_name zero_shot_forecast \   # task type: zero-shot forecasting
  --is_training 0 \                  # 0 = inference only (no training)
  --root_path ./dataset/ETT-small/ \ # root directory of the dataset
  --data_path ETTh1.csv \            # dataset file name
  --model_id ETTh1_512_96 \          # experiment/model identifier
  --model Moirai \                   # model name (TimesFM / Moirai)
  --data ETTh1 \                     # dataset name
  --features M \                     # multivariate forecasting
  --seq_len 512 \                    # input sequence length
  --pred_len 96 \                    # prediction horizon
  --enc_in 7 \                       # number of input variables
  --des 'Exp' \                      # experiment description
  --itr 1                             # number of runs
```


### Quick Test

Quick test for all 5 tasks (1 epoch each):

```bash
# Run quick tests for all 5 tasks
export CUDA_VISIBLE_DEVICES=0

# 1. Long-term forecasting
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id test_long --model DLinear --data ETTh1 --features M --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --train_epochs 1 --num_workers 2

# 2. Short-term forecasting (using ETT dataset with shorter prediction length)
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id test_short --model TimesNet --data ETTh1 --features M --seq_len 24 --label_len 12 --pred_len 24 --e_layers 2 --d_layers 1 --d_model 16 --d_ff 32 --enc_in 7 --dec_in 7 --c_out 7 --top_k 5 --train_epochs 1 --num_workers 2

# 3. Imputation
python -u run.py --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id test_imp --model TimesNet --data ETTh1 --features M --seq_len 96 --e_layers 2 --d_layers 1 --d_model 16 --d_ff 32 --enc_in 7 --dec_in 7 --c_out 7 --top_k 3 --train_epochs 1 --num_workers 2 --label_len 0 --pred_len 0 --mask_rate 0.125 --learning_rate 0.001

# 4. Anomaly detection
python -u run.py --task_name anomaly_detection --is_training 1 --root_path ./dataset/PSM --model_id test_ad --model TimesNet --data PSM --features M --seq_len 100 --pred_len 0 --d_model 64 --d_ff 64 --e_layers 2 --enc_in 25 --c_out 25 --anomaly_ratio 1.0 --top_k 3 --train_epochs 1 --batch_size 128 --num_workers 2

# 5. Classification
python -u run.py --task_name classification --is_training 1 --root_path ./dataset/Heartbeat/ --model_id Heartbeat --model TimesNet --data UEA --e_layers 2 --d_layers 1 --factor 3 --d_model 64 --d_ff 128 --top_k 3 --train_epochs 1 --batch_size 16 --learning_rate 0.001 --num_workers 0
```


### Logging and TensorBoard

Every `run.py` invocation automatically saves a UTF-8 text log under `./logs/`
and TensorBoard events under `./runs/`. Existing shell scripts need no changes.
Names include the task, model, model ID, sequence/prediction lengths, timestamp,
and process ID. Repetitions (`--itr`) use separate `iteration_N` event directories.

Text logs include arguments, runtime versions, existing Python console output,
training progress, and uncaught exceptions after argument parsing. Terminal
colors are removed from files. Native subprocess/file-descriptor output is not
captured. TensorBoard records epoch losses, learning rate, epoch duration, and
task-specific evaluation metrics; classification also records accuracy.
M4 aggregate metrics appear only once all six seasonal forecasts are available.

```bash
# Run an existing experiment, then start TensorBoard in another terminal.
bash scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1.sh
tensorboard --logdir ./runs --port 6006
```

Open http://localhost:6006. Optional `run.py` flags: `--log_dir ./logs`,
`--tensorboard_dir ./runs`, `--log_level INFO`, and `--no_tensorboard`.
The log level filters text records (captured stdout is INFO; stderr is WARNING);
it does not filter TensorBoard metrics or existing terminal prints.
TensorBoard is already listed in `requirements.txt`.

All models share `utils/logger.py` through their task's experiment pipeline.
Custom models/tasks can call the same API without creating their own writers:

```python
from utils.logger import get_logger, log_metrics

logger = get_logger(__name__)
logger.info("Model initialized")
log_metrics({"loss/train": loss.item()}, step=epoch + 1)
```

Call metrics from the main training process, not DataLoader workers. `run.py`
manages setup, repetition boundaries, flushing, and cleanup. Standalone callers
can use `experiment_logging(args)` and `start_experiment(args)`; text logging
does not require TensorBoard. Run logger regression tests with
`python -m unittest discover -s tests -p 'test_*.py'`.

### Train and Evaluate

We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

> ⚠️ Some scripts have `CUDA_VISIBLE_DEVICES` set by default. Please modify or remove this setting according to your actual GPU configuration, otherwise it may prevent GPU usage.

```bash
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# imputation
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh
# anomaly detection
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
# classification
bash ./scripts/classification/TimesNet.sh
```

### Develop Your Own Model
- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Create the corresponding scripts under the folder `./scripts`.

### Note: 

(1) About classification: Since we include all five tasks in a unified code base, the accuracy of each subtask may fluctuate but the average performance can be reproduced (even a bit better). We have provided the reproduced checkpoints [here](https://github.com/thuml/Time-Series-Library/issues/494).

(2) About anomaly detection: Some discussion about the adjustment strategy in anomaly detection can be found [here](https://github.com/thuml/Anomaly-Transformer/issues/14). The key point is that the adjustment strategy corresponds to an event-level metric.

### Inspect the project structure:

```
Time-Series-Library/
├── README.md                     # Official README with tasks, leaderboard, usage
├── requirements.txt              # pip dependency list for quick environment setup
├── LICENSE / CONTRIBUTING.md     # Upstream license and contribution guide
├── run.py                        # Unified entry that parses args and dispatches tasks
├── exp/                          # Task pipelines wrapping train/val/test
│   ├── exp_basic.py              # Experiment base class, registers models, builds flows
│   ├── exp_long_term_forecasting.py    # Long-term forecasting logic
│   ├── exp_short_term_forecasting.py   # Short-term forecasting logic
│   ├── exp_imputation.py               # Missing-value imputation
│   ├── exp_anomaly_detection.py        # Anomaly detection
│   ├── exp_classification.py           # Classification
│   └── exp_zero_shot_forecasting.py    # LTSM zero-shot evaluation
├── data_provider/                # Dataset loaders and splits
│   ├── data_factory.py           # Chooses the proper DataLoader per task
│   ├── data_loader.py            # Generic TS reader with sliding-window logic
│   ├── uea.py / m4.py            # Parsers for UEA, M4 and other formats
│   └── __init__.py               # Exposes factory interfaces upward
├── models/                       # All model implementations
│   ├── TimesNet.py, TimeMixer.py # Main forecasting models
│   ├── Chronos2.py, TiRex.py     # LTSM zero-shot models
│   └── __init__.py               # Enables name-based instantiation inside exp
├── layers/                       # Reusable attention / conv / embedding blocks
│   ├── Transformer_EncDec.py     # Transformer stacks
│   ├── AutoCorrelation.py        # Auto-correlation operator
│   ├── MultiWaveletCorrelation.py# Frequency-domain unit
│   └── Embed.py etc.             # Shared primitives
├── utils/                        # Utility toolbox
│   ├── metrics.py                # MSE / MAE / DTW and other metrics
│   ├── tools.py                  # General helpers such as EarlyStopping
│   ├── augmentation.py           # Augmentations for classification / detection
│   ├── print_args.py             # Unified argument printer
│   └── masking.py / losses.py    # Task-specific helpers
├── scripts/                      # Bash recipes for reproducible experiments
│   ├── long_term_forecast/       # Long-term forecasting per dataset/model
│   ├── short_term_forecast/      # M4 and other short-term scripts
│   ├── imputation/               # Imputation scripts
│   ├── anomaly_detection/        # SMD / SMAP / SWAT detection scripts
│   ├── classification/           # UEA classification scripts
│   └── exogenous_forecast/       # TimeXer exogenous forecasting flow
├── tutorial/                     # TimesNet tutorial notebook and figures
└── pic/                          # README figures (dataset overview, etc.)
```

### Understand the project architecture:

- **E2E flow**: configure experiments via `scripts/*.sh` → run `python run.py ...` → `run.py` parses arguments and selects the proper `Exp_*` via `task_name` → the experiment builds datasets through `data_provider`, instantiates networks from `models`, and drives train/val/test with utilities in `utils` → metrics and checkpoints are written to `./checkpoints`.
- **Experiment layer (`exp/`)**: `Exp_Basic` registers models and devices; subclasses implement `_get_data`, `train`, and `test` to encapsulate task-specific differences so the same model can be reused.
- **Model & layer layer (`models/` + `layers/`)**: model files define architectures, while reusable attention/conv/frequency components live in `layers/` to minimize duplication.
- **Data layer (`data_provider/`)**: `data_factory` returns the correct `Dataset/DataLoader`; `data_loader` handles windowing, masking, and sampling, with arguments controlling window length, missing ratio, anomaly ratio, etc.
- **Script layer (`scripts/`)**: bash scripts capture paper configurations (dataset, window, model, GPU) for reproducibility and serve as templates for custom runs.
- **Utility layer (`utils/`)**: `metrics` centralizes evaluation, `tools` bundles essentials like `EarlyStopping` and `adjust_learning_rate`, while `augmentation`/`masking` cover task-specific preprocessing.
- **Learning path**: recommended reading order is `scripts -> run.py -> exp/exp_basic.py -> corresponding Exp subclass -> data_provider -> models`, using `tutorial/TimesNet_tutorial.ipynb` as a guided walkthrough before diving deeper.

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}

@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  booktitle={arXiv preprint arXiv:2407.13278},
  year={2024},
}
```

## Contact
If you have any questions or suggestions, feel free to contact our maintenance team:

Current:
- Haixu Wu (Ph.D., wuhaixu98@gmail.com)
- Yuxuan Wang (Ph.D. student, wangyuxu22@mails.tsinghua.edu.cn)
- Yong Liu (Ph.D. student, liuyong21@mails.tsinghua.edu.cn)
- Ailuntz (Student from Open-source Community, ailuntz@icloud.com)

Previous:
- Huikun Weng (Undergraduate, wenghk22@mails.tsinghua.edu.cn)
- Tengge Hu (Master student, htg21@mails.tsinghua.edu.cn)
- Haoran Zhang (Master student, z-hr20@mails.tsinghua.edu.cn)
- Jiawei Guo (Undergraduate, guo-jw21@mails.tsinghua.edu.cn)

Or describe it in Issues.

## Acknowledgement

This library is constructed based on the following repos:

- Forecasting: https://github.com/thuml/Autoformer.

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer.

- Classification: https://github.com/thuml/Flowformer.

All the experiment datasets are public, and we obtain them from the following links:

- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer.

- Short-term Forecasting: https://github.com/ServiceNow/N-BEATS.

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer.

- Classification: https://www.timeseriesclassification.com/.

## All Thanks To Our Contributors

<a href="https://github.com/thuml/Time-Series-Library/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=thuml/Time-Series-Library" />
</a>

### TimeMixer++ PyPOTS forecasting

The retained local TimeMixerPPPyPOTS model uses average pooling, independent channel
encoding, coarsest-scale FFT periods, within-period seasonal / across-period trend
attention, and an equal-weight ensemble of all prediction heads.
Run the standard four-horizon ETTm1 script:

```bash
bash scripts/long_term_forecast/ETT_script/TimeMixerPPPyPOTS_ETTm1.sh
```

See [implementation notes](docs/TimeMixerPPPyPOTS.md) for the retained 96-step
baseline, checkpoint compatibility and provenance. This is the only retained
TimeMixer++ implementation. The earlier independent TimeMixerPP implementation,
shared-convolution/original-period ablations and their CLI options/scripts have
been removed. Historical experiment results and checkpoints remain available.

### SeasonTrendMixer forecasting

SeasonTrendMixer uses patch-local convolution and sinusoidal patch positions before
Transformer encoding on two fine seasonal scales. It combines these seasonal features
with six independent temporal linear preprocessors (two coarse seasonal and all four
trend scales), directional scale mixing, and eight summed forecast heads.

```bash
bash scripts/long_term_forecast/ETT_script/SeasonTrendMixer_ETTm1.sh
```

See [architecture, defaults and isolated smoke test](docs/SeasonTrendMixer.md).

### LagARFormer forecasting

LagARFormer v5 uses four resolutions (96/48/24/12 with the default input), each
with FFT-period TimeMixer++ row/column encoding, parallel lag message propagation
and a linear trend head. Each branch forecasts at its own sampling rate; outputs
are expanded to the original time grid and combined by learned linear fusion.
One shared normalization and original-resolution decomposition precede pooling.

~~~bash
RUN_TAG=lar_multi_v5 bash scripts/long_term_forecast/ETT_script/LagARFormer_ETTm1.sh
~~~

New checkpoints are required. The standard script uses three 2x average-pooling
steps, all local lags, two message layers and four independent forecast commands.
See [architecture and isolated CPU smoke test](docs/LagARFormer.md).

### PDM最粗MA因果验证（2026-10-02）

ETTm1 96→96验证集配对实验已恢复完成并核验：两组7轮早停、均选第2轮。最粗MA3恢复曲率但验证MSE较MA25增加0.0634%，LUFL粗分支负贡献未减轻。单种子、复用验证区间。随后按用户授权完成完整测试：MA25 MSE 0.32815561/MAE 0.36548027；MA3 MSE 0.32731138/MAE 0.36473707。见[完整结果与恢复审计](docs/experiments/OscillatorPDMCoarseMA_results_20261002.md)。

### PDM 趋势路径衰减配对实验（2026-10-02，已核验）

seed2021 两组7轮早停、均选第2轮。取消12个趋势上采样权重的衰减恢复了路径增益，但验证MSE由0.394567372增至0.394632710，LUFL粗分支负贡献增大；未通过筛查，按协议不运行后续两种子。本轮仅验证，无test访问。[完整结果与离线审计](docs/experiments/OscillatorPDMTrendDecay_results_20261002.md)。

- [PDM自动尺度MA三组结果（2026-10-03）](docs/experiments/OscillatorPDMScaledMA_results_20261003.md)：三组7轮早停、均选第2轮；自动窗口验证MSE较MA25降低0.02837%，但LUFL粗分支负贡献增大，仅单种子微小收益。


## PDM自动尺度MA＋自相关KL配对（2026-10-03）

已完成并离线核验：两组联合早停7轮，均选第2轮，各7525更新。自相关KL/d降低18.44%，但验证MSE 0.39420532→0.39859542（+1.11%）、MAE +1.41%，波形误差+2.31%；HUFL/MUFL贡献约90.35%的MSE净增量。固定系数1的本配置未改善预测，仅验证、无test。166项源码及冻结权重/预测重构核验通过。 [完整结果](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/OscillatorPDMAutocorr_results_20261003.md)。

[实验协议](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/OscillatorPDMAutocorr_protocol_20261003.md)；[预检审计](/home/abc/PythonProject/Time-Series-Library-main/tmp/pdm_autocorr_preflight_20261003/preflight_audit.json)。47项回归、GPU两轮配对/冻结、精确恢复及旧基线一致性已通过。

- [自相关KL密度与梯度诊断（2026-10-03）](docs/experiments/OscillatorPDMAutocorr_density_diagnosis_20261003.md)：固定512验证窗口对比9×9/97×97；加密未消除HUFL/MUFL幅值方向冲突，非重训结论。

- PDM相关/协方差KL三组配对（2026-10-03，已完成离线核验）：7轮联合早停，各7525更新，三组选第2/1/2轮。验证MSE为0.39420532/0.41245982/0.40159537；协方差较纯MSE增加1.87%，幅值误差下降8.95%但形状对齐误差增加13.06%，未改善总体预测。171项源码、冻结权重及分支重构通过，严格仅train/val。[完整结果](docs/experiments/OscillatorPDMKLNormalization_results_20261003.md)。

- [PDM季节分支递推头实现（2026-10-03）](docs/experiments/OscillatorPDMSeasonalHead_implementation_20261003.md)：新增模型 `OscillatorPDMSeasonalHead`，最后一层季节特征进递推头、趋势特征进线性头，保留原始趋势旁路；实现及合成检查已完成，尚无正式训练结果。

- PDM季节递推头配对（已核验）：第7轮联合早停，两组均选第2轮、各7525更新；验证MSE 0.39420532→0.39676270（+0.65%），LUFL改善但LULL/MUFL/HUFL等退化，未改善总体。严格train/val，无test。[完整结果](docs/experiments/OscillatorPDMSeasonalHead_results_20261003.md)。

- PDM预测入口分解三组（已核验）：7轮联合早停、均选2轮、各7525更新；baseline/full_trend/seasonal_trend验证MSE 0.39420532/0.39369256/0.39342754。季节入口较原版MSE降0.20%但MAE升0.80%，HUFL/MUFL幅值误差仍恶化；完整入口加趋势头同时小幅降低MSE/MAE。严格train/val。[完整结果](docs/experiments/OscillatorPDMReadoutSplit_results_20261004.md)。
- [PDM趋势平滑残差三组协议（2026-10-04）](docs/experiments/OscillatorPDMTrendSmoothResidual_protocol_20261004.md)：原模型/自由线性/平滑残差，后两组同参数、同单位初始化，平滑新增修正量。68项CPU及GPU编排/恢复预检通过，正式seed2021于17:19启动，已完成核验；平滑有效抑制趋势粗糙化，但总体验证MSE未改善。
- [PDM趋势融合前时间线性层（2026-10-04）](docs/experiments/OscillatorPDMTrendPreLinear_implementation_20261004.md)：新增 `OscillatorPDMTrendPreLinear`，两层四尺度共8个独立时间映射，单位初始化；62项PDM回归及GPU完整预检通过，正式配对已完成核验，总体MSE未改善，MAE小幅下降。
- [趋势融合前线性层配对协议](docs/experiments/OscillatorPDMTrendPreLinear_protocol_20261004.md)：seed2021，seasonal_trend/trend_prelinear两组，最多30轮、联合patience5，严格train/val；2026-10-04 12:45启动，已完成并核验。

- PDM趋势融合前线性层配对（已核验）：7轮联合早停、均选2轮、各7525更新；验证MSE 0.393427544→0.393637805（+0.0534%），MAE降0.2656%。LUFL改善约3.05%，HUFL/MUFL退化抵消；无约束时间映射大幅增加特征趋势二阶差分，未降低总体MSE。严格train/val。[完整结果](docs/experiments/OscillatorPDMTrendPreLinear_results_20261004.md)。

- PDM趋势平滑残差三组（已核验）：7轮联合早停、均选2轮、各7525更新。原/自由/平滑MSE 0.393427544/0.393637805/0.395379152；平滑较原MSE +0.4961%、MAE −0.1188%。趋势粗糙化大幅缓解，LUFL收益大部保留、HUFL/MUFL较自由组恢复，但HULL/MULL/LULL尤其水平误差退化，未降低总体MSE。[完整结果](docs/experiments/OscillatorPDMTrendSmoothResidual_results_20261004.md)。


## PDM旁路适应隔离四组实验（2026-10-04，已完成核验）

[协议](docs/experiments/OscillatorPDMBypassFreeze_protocol_20261004.md)；[完整结果](docs/experiments/OscillatorPDMBypassFreeze_results_20261004.md)。四组11轮联合早停，各11825次更新，所选轮2/2/6/6。原模型/平滑/原模型固定/平滑固定MSE为0.393427544/0.395379152/0.391593124/0.395517425。冻结没有救回平滑改动，MSE交互+0.001972692；原模型冻结虽总体MSE −0.4663%、MAE −0.1593%，LUFL MSE却+9.3679%，不算全面修复。单种子复用val，无test。 191项源码及四组冻结产物核验通过，固定旁路全程不变；额外模型前向0。

- [去掉原始趋势旁路（2026-10-05）](docs/experiments/OscillatorPDMNoRawBypass_implementation_20261005.md)：新增 `OscillatorPDMNoRawBypass`，保留季节递推与特征趋势预测，参数206470、8分量；仅结构回归通过，未正式训练，FFT路由尚未调整。

- [季节分支专用 FFT 编码（2026-10-05）](docs/experiments/OscillatorPDMSeasonalFFT_implementation_20261005.md)：`OscillatorPDMSeasonalFFT` 入口一次 MA 分解、季节路径一次 FFT 编码，之后两层仅做分路多尺度混合；无重复分解/编码或原始趋势旁路，163966参数、8分量，5项回归通过，正式配对已于14:06启动。

- [季节振荡器对直接预测头实验（2026-10-06，已核验）](docs/experiments/OscillatorPDMDecoderStudy_results_20261006.md)：11轮联合早停，最佳轮2/6；直接头MSE 0.409435对振荡器0.395196（+3.603%），MAE +3.233%，参数少23.514%。LUFL改善，其余六变量MSE恶化；输入归一化未改，完整解码器家族比较，单种子复用val。
- [跨尺度混合1层对2层结果（2026-10-06）](docs/experiments/OscillatorPDMMixingDepth_results_20261006.md)：11轮联合早停；1层/2层最佳轮2/6，MSE 0.395196/0.400156（−1.23943%）、MAE 0.411705/0.415219（−0.84625%），参数136150/168062；LUFL改善但LULL及部分变量退化；209项源码与保存产物离线核验通过，单种子复用val。
- [趋势条件动力学配对结果（2026-10-06）](docs/experiments/OscillatorPDMTrendCondition_results_20261006.md)：11轮联合早停，最佳轮2/6；MSE 0.402801→0.400156（−0.65667%），MAE −1.14743%；LULL改善、LUFL MSE +3.90623%，后期验证退化加重；204项源码及保存产物核验通过。
- [FFT切段后位置编码配对结果（2026-10-05，已核验）](docs/experiments/OscillatorPDMSegmentPE_results_20261005.md)：7轮联合早停，两组均选第2轮；MSE 0.405092→0.402801（−0.5655%）、MAE 0.420781→0.420038（−0.1765%）。收益主要来自LUFL，HUFL等五变量退化，幅值误差增大；198项源码与冻结产物核验通过。
- [FFT切段后二维位置编码（2026-10-05）](docs/experiments/OscillatorPDMSegmentPE_structure_20261005.md)：段序号与段内位置在季节FFT选周期/reshape之后加入，参数数保持163966；实现回归检查通过，配对结果见上。
- [移除历史位置编码实验结果（2026-10-05）](docs/experiments/OscillatorPDMNoHistoryPE_results_20261005.md)：9轮联合早停；MSE 0.405642→0.405092（−0.136%），MAE +0.247%。LUFL MSE改善但MUFL/LULL退化，不能认定已解决根因；191项源码与保存产物离线核验通过。
- [单次分解与季节FFT实验结果（2026-10-05）](docs/experiments/OscillatorPDMSingleFFT_results_20261005.md)：第9轮联合早停，原组/新组选择第2/4轮；新结构MSE 0.405642 vs 0.393428（+3.10%）、MAE +1.45%，LUFL MSE +15.79%。185项源码/冻结产物及8/12分量重构核验通过；非单因素、单种子复用val，无test。

- [历史锚定自协方差损失三组结果（2026-10-06，已核验）](docs/experiments/OscillatorPDMAutocovLoss_results_20261006.md)：三组7轮联合早停、均选第2轮；MSE/cov/cov_anchor验证MSE为0.395196/0.392670/0.395073。仅cov的MSE −0.6393%、MAE −0.0642%，但HUFL/MUFL退化；完整组MAE +0.2990%，未带来进一步收益。220项源码及冻结产物、辅助损失离线重算核验通过；归一化未改，单种子复用val，无test。

- [历史观测校正配对结果（2026-10-06，已核验）](docs/experiments/OscillatorPDMInnovation_results_20261006.md)：7轮联合早停，两组均选第2轮；MSE 0.395196→0.397175（+0.5007%），MAE +0.0876%，low4 +0.7103%。滤波参数学到变化但未改善总体预测，HUFL/MUFL/LULL退化、LUFL改善。226项源码及产物审计通过，旧对照精确复现；归一化未改，单种子复用val，无test。


- **可预测自协方差校正（2026-10-07）：实现与3场景×5组实验核验完成；有限样本MSE无一致收益，ETTm1训练段5/7变量未通过可靠性筛查，按协议未启动正式PDM训练。** [完整结果](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/PredictableAutocov_results_20261007.md)。输入归一化及旧226项源码未改。


- **PDM输出时间对损失（2026-10-08）：强制重启后已中断，保留5轮检查点，未自动恢复。** 纯MSE、原分辨率位置保留HF/FF、4步池化HF/FF；相同136150参数与scratch初值，无概率头、归一化不改，按完整val MSE选轮。[协议](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/PDMPositionPair_protocol_20261008.md)。
- **ETTm1有向条件高斯完整训练（2026-10-07）：完成并通过离线审计。** 固定均值下联合NLL 7.83007→2.65656，但90%边际覆盖97.66%→80.14%，有向头欠覆盖；共同MSE0.65455/MAE0.54695不变。均值选5轮、两头选22轮并于27轮联合早停。[完整结果](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/PDMDirectedGaussianFull_results_20261007.md)。
- **PDM有向条件高斯头（2026-10-07）：实现、8项测试及96→96七变量集成检查通过。** 合成相关场景NLL 0.92123→0.89982，独立场景略退化；均值固定，点MSE不变。未进行ETTm1训练或真实因果识别。[结果与调用方式](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/PDMDirectedGaussian_results_20261007.md)。

- **PDM非线性修正实验（2026-10-08）：完成并通过离线核验，固定并行4轮未改善。** 两组第7轮联合早停、均选第2轮；验证MSE 0.39519635→0.39796322（+0.70%），MAE 0.41170477→0.41660664（+1.19%）。新增14528参数；所选前16验证窗口顺序参考最大状态差2.3861，不能据此否定精确顺序递推。归一化不改，无test。[结果](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/PDMNonlinearState_results_20261008.md)。

- **PDM学习率对照（2026-10-08）：完成并核验。** 同一136150参数模型与scratch初值，纯MSE；11轮联合早停。1e-3 / 3e-4最佳验证MSE为0.395196 / 0.398782，MAE为0.411705 / 0.418821；降低学习率未改善（MSE +0.9072%、MAE +1.7284%）。归一化不改，无本轮test。[结果](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/PDMLearningRate_results_20261008.md) · [协议](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/PDMLearningRate_protocol_20261008.md)。

- **PDM非线性状态更新（2026-10-08）：实现与CPU核验通过，完整配对实验已完成并核验。** 顺序参考＋并行迭代，8项测试和96→96七变量两步检查；4轮是近似计算，需要监测残差。旧学习率实验保留。[实现说明](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/PDMNonlinearState_implementation_20261008.md)。

- [MA1纯MSE日程对照结果 20261010](docs/experiments/PDMSchedule_results_20261010.md)

- [MA1贝叶斯二次关系lambda对照 20261010](docs/experiments/PDMBayesianLambda_results_20261010.md)

- [MA1核自协方差结构损失对照 20261010](docs/experiments/PDMKernelAutocovariance_results_20261010.md)

- [MA1核自协方差跨界点对实验](docs/experiments/PDMKernelBoundaryIndependent_results_20261010.md)

- [MA1核结构损失权重5/10实验协议 20261010](docs/experiments/PDMKernelWeight_protocol_20261010.md)

- [MA1核自协方差权重5与10对照 20261010](docs/experiments/PDMKernelWeight_results_20261010.md)

- 2026-10-10：[MA1增量片段三组小样本诊断](tmp/pdm_increment_patch_trial_20261010/report.html)：同初值、1/16窗口各800步，含目标修正检查与固定验证探针；非完整效果实验，未访问test。

- [MA1增量片段三组完整实验 20261010](docs/experiments/PDMIncrementPatch_results_20261010.html)

- [MA1不重叠片段核协方差两组完整实验 20261011](docs/experiments/PDMPatchCovariance_results_20261011.html)

- [MA1联合片段读出两组完整实验 20261011](docs/experiments/PDMJointPatch_results_20261011.html)

- [MA1单状态四点读出两组完整实验 20261011](docs/experiments/PDMStatePatch_results_20261011.html)

- 单状态四点读出离线复核：核组比同结构MSE的test MSE降低2.32%，但仅略优于原逐点基线，片段边界误差增加；[六组对照与完整审计](docs/experiments/PDMStatePatch_results_20261011.html#offline-final-interpretation)。

- [MA1顺序拼接投影单状态四点读出两组完整实验 20261012](docs/experiments/PDMStatePatchConcat_results_20261012.html)

- [拼接投影离线复核：相对均值聚合，纯MSE test MSE +0.42%；核组边界误差 -3.88%，但test MSE +1.62%、MAE +0.68%，未获整体提升。](docs/experiments/PDMStatePatchConcat_results_20261012.html)

- [MA1直接24特征单状态四点读出两组完整实验 20261012](docs/experiments/PDMDirectPatch_results_20261012.html)

- [MA1 mean-state-four historical additive versus RoPE 20261012](docs/experiments/PDMStatePatchRoPE_results_20261012.html)

- [MA1 patch relation KL versus MSE 20261012](docs/experiments/PDMPatchRelationKL_results_20261012.html)

- [MA1 CKA and lagged patch covariance 20261012](docs/experiments/PDMPatchCKALag_results_20261012.html)

- [MA1 disconnected dynamics and recurrent trend 20261013](docs/experiments/PDMStatePatchDisconnected_results_20261013.html)

<!-- disconnected offline review 20261013 -->
两组均未超过旧均值版 MSE+D 基准0.319869。断连test MSE0.327048/MAE0.360798；双递推0.324521/0.358619。双递推比断连改善，但相对旧基准MSE仍恶化1.4543%，MAE恶化0.5495%。 离线核验：9轮/每组9675更新，选轮4/2；单seed、非等参数，与旧组停止预算不同。


<!-- dual encoder offline review 20261013 -->
- [双分支历史振荡编码→PDM→FMM：完整结果与离线核验](docs/experiments/OscillatorDualEncoderTimeMixer_results_20261013.html)
新双分支历史递推→PDM→FMM未超过旧均值版核基准：test MSE0.331973（+3.78%）、MAE0.364621（+2.23%）。9轮早停/9675更新，验证选第4轮；HUFL、MUFL退化，幅值误差+10.39%，边界增量误差−8.25%。单seed、非等参数、非等停止预算，多处结构变化不能唯一归因。


<!-- dual encoder gates offline review 20261013 -->
- [遗忘与写入门：完整配对结果及离线核验](docs/experiments/OscillatorDualEncoderGates_results_20261013.html)
遗忘＋写入门未获整体提升：test MSE0.331808，相对新无门对照0.331973仅下降0.05%；MAE0.365999，相对0.364621上升0.38%。两组9轮共同早停、各选第4轮、每组9675更新。门控仍差于旧mean核0.319869/0.356659（MSE+3.73%、MAE+2.62%）；LUFL/LULL相对新对照退化。两门有梯度且参数更新，写入门多数接近常开；单seed、新增参数和初始函数差异限制归因。


<!-- dual encoder residual launch 20261013 -->
- [双编码器输入残差：配对实验已完成](docs/experiments/OscillatorDualEncoderResidual_results_20261013.html) · [冻结协议](docs/experiments/OscillatorDualEncoderResidual_protocol_20261013.json)
control 与 residual 均199633参数；仅增加两支四尺度固定输入旁路，初始参数相同、初始函数不同。7测试与真实CUDA预检通过；完整训练及离线核验已完成，详细结果见下方登记。


<!-- dual encoder residual offline review 20261013 -->
- [固定输入残差：完整配对结果与离线核验](docs/experiments/OscillatorDualEncoderResidual_results_20261013.html)
固定输入残差带来有限改善：test MSE0.325849，相对新对照0.331973下降1.84%；MAE0.364091，相对0.364621仅下降0.15%。两组9轮共同早停，control选4、residual选3，每组9675更新。仍差于旧mean+D的0.319869/0.356659（MSE+1.87%、MAE+2.08%）；HUFL/MUFL改善，但LULL退化、LUFL的MAE上升，幅值误差+3.78%。等参数但初始函数不同，单seed复用val/test，不能唯一归因或声称跨种子稳定。


<!-- mean patch readout skip launch 20261013 -->
- [旧均值网络32维读出直通：配对实验已完成](docs/experiments/PDMStatePatchReadoutSkip_results_20261013.html) · [冻结协议](docs/experiments/PDMStatePatchReadoutSkip_protocol_20261013.json)
control 与 residual 均136546参数；仅在四个季节头的32维读出隐藏特征处加入已有patch均值特征，保留趋势调制、原24次A递推。初始参数相同、函数不同。7测试与真实CUDA预检通过；完整训练及离线核验已完成，详细结果见下方登记。


<!-- state patch readout skip offline review 20261013 -->
- [旧均值网络读出直通：完整配对结果与离线核验](docs/experiments/PDMStatePatchReadoutSkip_results_20261013.html)
固定32维读出直通未带来收益：test MSE从0.319869升至0.326426（+2.05%），MAE从0.356659升至0.360955（+1.20%）。两组11轮共同早停，control选6、residual选4，各11825更新。直通组val MSE略降0.097%，但test七变量中六个MSE/MAE均退化，LUFL和LULL尤甚；四个时间段及预测区间均退化。新control的全部11轮曲线、逐轮权重及选定val/test预测精确复现旧mean+D；单seed复用数据不构成独立种子证据。


<!-- mean patch output skip launch 20261014 -->
- [旧均值网络独立输出旁路：配对实验已完成](docs/experiments/PDMStatePatchOutputSkip_results_20261014.html) · [冻结协议](docs/experiments/PDMStatePatchOutputSkip_protocol_20261014.json)
control136546参数，residual137058参数；四个零初始化、无bias的独立32→4旁路在输出相加，保留mean4、趋势调制和原24次A递推。共有初值、初始预测及共有梯度一致，新增512参数。7测试与CUDA预检通过；完整训练及离线核验已完成。


<!-- state patch output skip offline review 20261014 -->
- [旧均值网络独立输出旁路：完整配对结果与离线核验](docs/experiments/PDMStatePatchOutputSkip_results_20261014.html)
独立零初始化输出旁路仍未超过旧mean+D：test MSE从0.319869升至0.321657（+0.56%），MAE从0.356659升至0.356705（+0.013%，基本持平但略差）。两组11轮共同早停，均选第6轮，各11825更新。四个旁路均已学习；边界增量误差下降2.24%，但低频和水平误差增加，LUFL/LULL及后段预测退化。新对照逐轮权重、曲线及选定预测精确复现旧mean+D。相比先前共享读出隐藏直通0.326426有所改善，仍不是超过0.319869的有效改进。


<!-- mean patch trend mlp launch 20261014 -->
- [旧均值网络趋势四点双MLP：配对实验已完成](docs/experiments/PDMStatePatchTrendMLP_results_20261014.html) · [冻结协议](docs/experiments/PDMStatePatchTrendMLP_protocol_20261014.json)
control136546参数，trend_mlp150478参数；四尺度趋势头时间MLP T→96→24、特征MLP32→32→4，各加GELU，24块拼成96点；季节分支保持原样。初始函数不同，7测试与CUDA预检通过，完整训练及离线核验已完成。


<!-- state patch trend mlp offline review 20261014 -->
- [旧均值网络趋势四点双MLP：完整配对结果与离线核验](docs/experiments/PDMStatePatchTrendMLP_results_20261014.html)
趋势四点向量＋双MLP未超过原0.319网络：test MSE从0.319869升至0.323503（+1.14%），MAE从0.356659升至0.358346（+0.47%）。val MSE/MAE分别下降0.62%/0.63%。两组11轮共同早停，均选第6轮，各11825更新。新趋势头32张量均学习；test七变量MSE全部上升，LUFL最明显，远期和低频误差增加。control逐轮权重、曲线和选定预测精确复现旧mean+D。分块、非线性、参数容量和初始函数同时变化，单seed不能唯一归因。


<!-- mean patch trend interp launch 20261014 -->
- [旧均值网络八控制点修正与低频损失：配对实验已完成](docs/experiments/PDMStatePatchTrendInterp_results_20261014.html) · [冻结协议](docs/experiments/PDMStatePatchTrendInterp_protocol_20261014.json)
control136546参数；仅修正与修正＋低频损失各178690参数。保留原趋势头，新增零末层320→32→8全局控制点修正并插值96点；第三组加固定1×MA25总预测误差。三组初始预测一致，9测试与CUDA预检通过，三组完整训练、各一次test及离线核验已完成；仅修正退化，加低频后总体MSE基本持平，变量存在取舍，详见结果。


<!-- mean patch separate mlp launch 20261014 -->
- [旧均值网络特征MLP拆开：配对实验已完成](docs/experiments/PDMStatePatchSeparateMLP_results_20261014.html) · [冻结协议](docs/experiments/PDMStatePatchSeparateMLP_protocol_20261014.json)
原0.319网络仅拆开PDM共享特征MLP，初始预测完全相同；control136546参数，separate_mlp140738参数。原趋势调制、季节mean4/A24、四线性趋势头及MSE+D保留。7项测试和CUDA预检通过；完整配对及离线核验已完成，本次test小幅改善，val MSE与LUFL退化，详见报告。


<!-- state patch trend interp offline review 20261014 -->
- [八控制点趋势修正＋低频损失：三组完整结果与离线图表](docs/experiments/PDMStatePatchTrendInterp_results_20261014.html)
八控制点修正单独加入后，test MSE 0.323270755，比对照增加1.0634%，MAE增加0.3403%。加MA25低频损失后，test MSE 0.319854681、MAE 0.355763933；相对仅修正组MSE下降1.0567%，但相对原对照MSE仅下降0.0046%（绝对差-0.000014741），基本持平，MAE下降0.2510%。HUFL/MUFL改善，LUFL MSE增加7.99%且MAE增加5.74%，LULL亦退化；总MA25趋势误差仍比对照高0.7365%。三组11轮共同早停，各11825更新，分别选6/6/4轮。单seed复用val/test，不能宣称稳定超过0.319。


<!-- state patch separate mlp offline review 20261014 -->
- [季节/趋势特征MLP拆开：完整配对结果与分解图](docs/experiments/PDMStatePatchSeparateMLP_results_20261014.html)
拆开季节/趋势共享特征MLP后，本次test MSE从0.319869422降至0.318078366（下降0.5599%），MAE从0.356659286降至0.354397267（下降0.6342%）。MA25趋势误差下降0.2759%，但幅值误差增加，低频幅度比与相关略降，未解决大尺度变化不足。HUFL/MUFL改善，LUFL MSE上升3.52%且MAE上升1.89%，末24步MSE退化；val MSE上升0.7558%。两组11轮共同停，各11825更新，分别选6/4轮。四张新增MLP参数均学习且与季节MLP明显分化。初始函数相同，但新增4192参数与优化路径变化，单seed复用val/test，不是独立显著性或稳定提升证据。


<!-- mean patch raw predecomp launch 20261014 -->
- [原始MA与FFT/滑动patch编码：三组实验进行中](docs/experiments/PDMStatePatchPreDecomp_results_20261014.html) · [冻结协议](docs/experiments/PDMStatePatchPreDecomp_protocol_20261014.json)
0.318结构为参考：三组均原始序列先MA分解，再FFT/滑动取patch后投影，取消Linear(1,32)。季节用TimeMixer++ row/column注意力；第三组趋势用滑动patch注意力和A24递推，新增季节调制趋势。9项测试和CUDA预检通过，效果待验证。


<!-- mean patch strict ma first comparisons launch 20261014 -->
- [严格原0.318 MA顺序对照：三组实验进行中](docs/experiments/PDMStatePatchMAFirstComparisons_results_20261014.html) · [冻结协议](docs/experiments/PDMStatePatchMAFirstComparisons_protocol_20261014.json)
第一组严格使用0.318独立MLP网络，仅MA移至原Linear(1,32)之前，140738参数；第二组完全解耦170818，第三组季节FFT row/column、趋势滑动attention+A24+季节调制235628保持之前设计。新scratch/no旧训练恢复；效果待验证。


<!-- mean patch ma separate linear comparisons launch 20261014 -->
- [原0.318 MA前移与仅拆线性层：三组实验进行中](docs/experiments/PDMStatePatchMALinearComparisons_results_20261014.html) · [冻结协议](docs/experiments/PDMStatePatchMALinearComparisons_protocol_20261014.json)
第一组原0.318仅MA/原Linear换序140738；第二组只将第一组共享Linear(1,32)拆为两套独立投影140866（+128），其他连接包括trend→season保持；第三组原季节FFT patch双轴/趋势滑动attention+A24+季节调制235628不变。全部fresh重新训练；效果待验证。


<!-- mean patch ma fully decoupled comparisons launch 20261014 -->
- [原0.318 MA前移与完全解耦：三组实验已完成](docs/experiments/PDMStatePatchMAFullyDecoupledComparisons_results_20261014.html) · [冻结协议](docs/experiments/PDMStatePatchMAFullyDecoupledComparisons_protocol_20261014.json)
第一组原0.318仅MA/原Linear换序140738；第二组独立Linear(1,32)且移除全部趋势→季节条件，实现两路可训练参数/路径完全分离136770；第三组季节FFT patch双轴/趋势sliding attention+A24/双向历史条件235628不变。fresh重新训练，效果待验证。


<!-- state patch ma fully decoupled offline review 20261014 -->
- [MA前移/完全解耦/季节调制趋势三组：完整配对结果与分解图](docs/experiments/PDMStatePatchMAFullyDecoupledComparisons_results_20261014.html)
三组11轮共同早停，各11825次更新，选轮4/6/2。第一组test MSE/MAE=0.316722980/0.353293177，优于另两组。完全解耦MSE/MAE=0.329341222/0.364037706，相对第一组上升3.984%/3.041%；第三组为0.326608442/0.361247246，上升3.121%/2.251%。两组test MA25误差分别上升4.992%/4.097%。第三组val MSE最好，但没有转化为test改善。第一组较历史0.318 test MSE下降0.426%，MAE下降0.312%；同seed复用val/test，不能称稳定或独立显著提升。


<!-- ma cause diagnosis 20261015 -->
- [MA三组原因定位及仅拆Linear新增对照](docs/experiments/PDMStatePatchCauseDiagnosis_20261015.html)
退化主要集中低频：完全解耦test新增MSE约88%在前四DCT模态，窗口水平项约69%；第三组新增MSE全部来自low4，高92模态略改善。完全解耦训练MSE11/11轮更差、val10/11轮更差；第三val9/11轮更好而选定test更差，不能简单归因于训练不够或只看选轮。全参数D/MSE采样比始终<1，attention/cross/readout有后续非零梯度和权重变化，未发现断连或全局D支配证据。具体结构原因仍待仅拆Linear新增对照。


<!-- ma separate linear only launch 20261015 -->
- [仅拆分解后的Linear：新增第四组已完成](docs/experiments/PDMStatePatchSeparateLinearAblation_results_20261015.html) · [冻结协议](docs/experiments/PDMStatePatchSeparateLinearAblation_protocol_20261015.json)
新增第四组：仅拆MA之后共享Linear(1,32)，保留全部trend→season调制和原编码/读出/PDM，140866（+128）。固定11轮/11825更新、原cosine30；相同fresh初值、完整val选轮，一次新test；旧三组不重训重测。效果待核验。


<!-- state patch separate linear ablation offline review 20261015 -->
- [仅拆MA后Linear原因定位：完整配对结果与分解图](docs/experiments/PDMStatePatchSeparateLinearAblation_results_20261015.html)
仅拆MA后Linear的新B完成11轮11825更新，val选第4轮，一次11425test。B testMSE/MAE=0.316620100/0.353503554；相对A MSE-0.0325%、MAE+0.0595%、MA25-0.1335%，总体近乎持平，不能称稳定改善。C相对B MSE上升4.018%、MAE上升2.980%、MA25上升5.132%；C在11/11同epoch trainMSE及10/11 valMSE更差。未支持仅解除投影共享解释原完全解耦的主要退化，更支持删除trend→season条件及减少4096参数这一复合改动有害。单seed同val/test、容量不同，不能唯一归因于调制或声称独立显著/跨seed稳定；第三attention/递推复合设计原因未被此组隔离。


<!-- trend temporal conv launch 20261015 -->
- [趋势时间残差卷积：已完成](docs/experiments/PDMStatePatchTrendTemporalConv_results_20261015.html) · [冻结协议](docs/experiments/PDMStatePatchTrendTemporalConv_protocol_20261015.json)
趋势单因素组：以已保存0.316723 MA前移基线为对照，仅在趋势升维后/PDM前加入四尺度独立因果时间残差卷积，145986参数（+5248）；原季节结构、共享embedding、预测头、trend→season调制及MSE+D保留。固定11轮11825更新、cosine30，fresh初始函数/RNG相同，ownval选轮，仅新组一次test；旧组不重训重测。效果待核验。


<!-- season period token launch 20261015 -->
- [季节整周期token：已完成](docs/experiments/PDMStatePatchSeasonPeriodToken_results_20261015.html) · [冻结协议](docs/experiments/PDMStatePatchSeasonPeriodToken_protocol_20261015.json)
季节整周期token：以保存MA前移基线为对照，rawMA残差FFT选周期P，完整周期向量pad到nativeT后Linear(T,32)，cyclePE/column注意力，再逐位置解码还原nativeT32；取消季节外部标量Linear与row注意力，原trend路径/调制/PDM/预测头不变。309694参数，固定11轮11825更新、cosine30，freshscratch/ownval选轮，仅新组一次test。初始函数不同，容量与编码几何同时变；效果待核验。


<!-- time position mmd launch 20261015 -->
- [时间位置核MMD：已完成](docs/experiments/PDMStatePatchTimePositionMMD_results_20261015.html) · [冻结协议](docs/experiments/PDMStatePatchTimePositionMMD_protocol_20261015.json)
时间位置核MMD：保持已保存MA前移基线140738网络和236freshstate，仅将辅助协方差D替换成时间位置乘积核biased MMD²。四点patch/24patch；数值sigma1，时间sigma2个patch，权重1，FP64到FP32、targetdetach。固定11轮11825更新、cosine30、ownfullval选轮，仅新组一次11425test。原D仅同forward诊断，不参与更新；不叠加TCN或周期token。


<!-- season period token offline final 20261015 -->
- [季节整周期token：完整离线结果](docs/experiments/PDMStatePatchSeasonPeriodToken_results_20261015.html)
整周期token组309694参数完成11轮11825更新，val选第6轮，一次11425test，离线核验通过。testMSE/MAE=0.322894281/0.357822101，较保存A分别+1.948%/+1.282%；MA25误差0.217764682（+1.787%）。本次未改善总体预测；整周期压缩/解码、row删除、rawFFT权重及多168956参数同时改变，不能仅归因于token定义。单seed复用val/test，无独立显著或跨seed稳定声明。


<!-- trend temporal conv offline final 20261015 -->
- [趋势时间残差卷积：完整离线结果](docs/experiments/PDMStatePatchTrendTemporalConv_results_20261015.html)
趋势卷积组145986参数完成11轮11825更新，val选第4轮，一次11425test，离线核验通过。testMSE/MAE=0.317322433/0.353282356，较保存A分别+0.189%/-0.003%；MA25误差0.214468090（+0.246%）。整体接近基线，未见明确改善。新增5248容量与局部偏置同时变；新层种子文档与冻结入口偏差已单列记录，未改源或重训。单seed复用val/test无独立显著/跨seed稳定声明。


<!-- time position mmd offline final 20261015 -->
- [时间位置核MMD：完整离线结果](docs/experiments/PDMStatePatchTimePositionMMD_results_20261015.html)
时间位置核MMD组保持MAFirst140738/236state结构，完成11轮11825更新，val选第4轮，一次11425test，离线核验通过。test MSE/MAE=0.314177886/0.349772139，较保存A分别-0.804%/-0.997%；low4误差-0.540%，MA25误差0.212865150（-0.503%）。窗口水平误差+0.489%，总体改善不代表所有低频指标改善；LULL MSE+0.378%，末24步MSE+0.499%。此次同时更换协方差统计量并加入时间位置，不能唯一归因于顺序核；单seed复用val/test，无独立显著或跨seed稳定声明。


<!-- trend temporal conv no decay launch 20261015 -->
- [趋势卷积仅取消新增权重衰减：已完成](docs/experiments/PDMStatePatchTrendTemporalConvNoDecay_results_20261015.html) · [冻结协议](docs/experiments/PDMStatePatchTrendTemporalConvNoDecay_protocol_20261015.json)
趋势卷积仅取消新增权重衰减：原TCN145986结构和全部252fresh初值精确保留，只将16张新增卷积权重weight_decay从1e-4改为0；其余参数策略、原MSE+covarianceD、Adam.001、batch/stepseed和11轮11825更新/cosine30保持。own完整val选轮后仅新组一次test；不叠加时间MMD，不重训重测旧组。持续检查逐层数据梯度、权重范数和残差RMS，效果待核验。


<!-- trend temporal conv no decay offline final 20261015 -->
- [趋势卷积取消新增衰减：完整离线结果](docs/experiments/PDMStatePatchTrendTemporalConvNoDecay_results_20261015.html)
卷积无衰减组保持原TCN145986/全部252fresh初值，只将16卷积权重decay1e-4改0，完成11轮11825更新、val选第4轮、一次11425test，离线核验通过。test MSE/MAE=0.314912138/0.352637400；相对旧TCN分别-0.760%/-0.183%，相对A分别-0.572%/-0.186%。MA25误差0.212300116（较TCN-1.011%，较A-0.767%）。全程后续采样及选轮val/test残差未归零，卷积数值塌缩避免。相对A，幅值误差+1.083%，LULL MSE+0.428%，第49–72步MSE+0.763%。单seed复用val/test，无独立显著或跨seed稳定声明。


<!-- trend conv internal residual mmd studies 20261015 -->
三组独立实验均145986参数、252相同fresh初值、第1/2组卷积衰减1e-4、第3组衰减0：1三层内部残差+原MSE/D；2内部残差+时间位置MMD；3原无衰减串联卷积+时间位置MMD。每组独立11轮11825更新/cosine30/ownfullval选轮/一次11425test。按顺序运行以限制主机内存，不共享模型或选轮，正式失败不恢复重跑，旧实验不重训重测。

- [trend_temporal_conv_inner_residual](docs/experiments/PDMStatePatch_trend_temporal_conv_inner_residual_results_20261015.html)
- [trend_temporal_conv_inner_residual_mmd](docs/experiments/PDMStatePatch_trend_temporal_conv_inner_residual_mmd_results_20261015.html)
- [trend_temporal_conv_no_decay_mmd](docs/experiments/PDMStatePatch_trend_temporal_conv_no_decay_mmd_results_20261015.html)


<!-- trend conv residual mmd parallel amendment 20261015 -->
三组实验按用户后续要求于北京时间2026-10-15下午清理旧VSCode扩展/Pylance后改为资源门控并行，第1组原进程保留，不重启训练；冻结模型、优化器、损失和11轮预算均不变。执行修订证据：`tmp/pdm_state_patch_trend_conv_residual_mmd_parallel_amendment_20261015.json`。


<!-- trend conv inner residual mmd formal failed 20261015 -->
第2组内部残差+时间位置MMD正式失败：退出码-9(SIGKILL)，完整完成1轮，最后日志第2轮step850；保留epoch1 last.pt，test0。未找到匹配PID749980的内核/系统OOM记录，原因未确定；当前内存压力不能单独证明OOM。未自动恢复/重训/重测，其余两组继续。


<!-- DistDF Fredformer matched 20261015 -->
- [DistDF＋Fredformer：匹配预算完整离线结果](docs/experiments/DistDFFredformerMatched_results_20261015.html)
官方Fredformer＋DistDF在本次匹配训练预算下完成11轮11825更新，val选第8轮，一次11425test。test MSE/MAE=0.331420028/0.363351772，MA25=0.221313509；较保存A分别+4.640%/+2.847%，MA25 +3.446%。此次未超过保存A、无TCN MMD或无衰减TCN。这不是官方原训练超参数复现；不同网络和目标同时变化，单seed不能唯一归因DistDF或作显著性结论。

<!-- trend conv inner residual offline final 20261015 -->
- [三层内部残差＋原协方差D（有衰减）：完整离线结果](docs/experiments/PDMStatePatch_trend_temporal_conv_inner_residual_results_20261015.html)
三层内部残差＋原协方差D（卷积有衰减）完成11轮11825更新，val选第2轮，仅一次11425test。test MSE/MAE=0.324952539/0.359145348，MA25=0.220568081；较主要对照TCN MSE/MAE=+2.405%/+1.660%，较A=+2.598%/+1.656%。最后轮四尺度残差均持续非零：True。模块存活与预测改善分别判断；单seed不作显著性结论。


<!-- trend conv no decay mmd offline final 20261015 -->
- [串联卷积无衰减＋时间位置MMD：完整离线结果](docs/experiments/PDMStatePatch_trend_temporal_conv_no_decay_mmd_results_20261015.html)
串联卷积无衰减＋时间位置MMD完成11轮11825更新，val选第6轮，仅一次11425test。test MSE/MAE=0.319503766/0.355139639，MA25=0.217044771；较主要对照NoDecay MSE/MAE=+1.458%/+0.710%，较A=+0.878%/+0.523%。最后轮四尺度残差均持续非零：True。模块存活与预测改善分别判断；单seed不作显著性结论。


<!-- group2 manual recovery 20261015 -->
- [第2组内部残差＋MMD：用户授权接续](docs/experiments/PDMStatePatch_trend_temporal_conv_inner_residual_mmd_recovery1_results_20261015.html)
第2组内部残差＋时间位置MMD（卷积衰减1e-4）按用户新指示，从原第1轮/1075更新完整模型及Adam状态接续，保持原超参数，单组继续第2至11轮。第2轮原未保存部分需重做，总实际尝试算力高于原不间断11轮；不声称完全相同计算预算。原SIGKILL失败目录与记录保留，旧冻结源未改，新接续入口/执行修订另存。CUDA3轮对照预检912tensor347scalar5array及所有梯度记录精确，无test。当前第2轮已启动，尚无测试结论。


<!-- trend conv inner residual mmd recovery1 offline final 20261015 -->
- [第2组内部残差＋时间位置MMD（有衰减，授权接续）：完整离线结果](docs/experiments/PDMStatePatch_trend_temporal_conv_inner_residual_mmd_recovery1_results_20261015.html)
第2组：三层内部残差＋时间位置MMD（有衰减，授权接续）保留完整11轮11825有效更新，val选第2轮，仅一次11425test；失败后从epoch1接续，重做未保存epoch2部分，累计实际算力高于不中断11轮。test MSE/MAE=0.323840719/0.357878600，MA25=0.220644742；较主要对照Group1 MSE/MAE=-0.342%/-0.353%，较A=+2.247%/+1.298%。较原MAFirst＋MMD MSE/MAE=+3.076%/+2.318%。最后轮四尺度残差均持续非零：True。模块存活与预测改善分别判断；单seed不作显著性结论。


<!-- DistDF TST matched 20261015 -->
- [DistDF＋Fredformer：TST调度对照完整离线结果](docs/experiments/DistDFFredformerTSTMatched_results_20261015.html)
Fredformer＋DistDF的官方TST调度对照完成11轮11825更新，val选第11轮，一次11425test。test MSE/MAE=0.321247713/0.357296802，MA25=0.217451973；较同模型cosine30分别-3.069%/-1.666%，MA25 -1.745%。本组仅替换官方TST调度政策，包含学习率和Adam beta1循环。这不是官方原训练超参数复现；与上一组模型和目标相同；不能把整个调度政策的效果单独归因学习率或beta1，单seed不作显著性结论。


<!-- DistDF official recipe 20261015 -->
- [DistDF＋Fredformer：作者公开脚本配方完整离线结果](docs/experiments/DistDFFredformerOfficialRecipe_results_20261015.html)
作者公开脚本配方完成60轮/16080更新（最多100轮，patience10），官方shuffle/drop_last128验证选第50轮，一次11425test batch1。test MSE/MAE=0.315720075/0.356492335，MA25=0.213430727；较上一组TST11 MSE/MAE -1.721%/-0.225%；较作者四舍五入MSE0.316差-0.000280（-0.089%）。seed/batch/衰减/学习率与周期/早停/验证策略同时对齐作者脚本，不单独归因某一项；单seed复用test，不作显著性结论。


<!-- MAFirst MMD official recipe 20261015 -->
- [原MAFirst＋MMD：对齐DistDF训练配方完整离线结果](docs/experiments/PDMStatePatchMAFirstMMDOfficialRecipe_results_20261015.html)
原MAFirst＋时间位置MMD，对齐DistDF公开脚本训练配方，完成41轮/10988更新（最多100轮、patience10），选第31轮，一次11425test。test MSE/MAE=0.321013338/0.358368651，MA25=0.216928029；较旧同架构MMD MSE/MAE +2.176%/+2.458%。模型140738参数及MSE＋MMD不变；seed/batch/衰减/调度周期/早停/验证/RNG同时变，不单独归因。单seed复用test，不作显著性或跨seed稳定结论。


<!-- MAFirst MMD official recipe repeat2 20261015 -->
- [原MAFirst＋MMD：同设置重复2完整离线结果](docs/experiments/PDMStatePatchMAFirstMMDOfficialRecipeRepeat2_results_20261015.html)
原MAFirst＋时间位置MMD，DistDF配方原样重复2，完成41轮/10988更新（最多100轮、patience10），选第31轮，一次11425test。test MSE/MAE=0.321013338/0.358368651，MA25=0.216928029；较上一轮同设置MMD MSE/MAE +0.000%/+0.000%。模型140738参数、MSE＋MMD与seed2023及全部训练超参数保持上一轮设置。仅同seed重复，不作独立seed稳定性或显著性结论。


<!-- MAFirst MMD original recipe seed2022 20261016 -->
- [原MAFirst＋MMD：原配方独立seed2022完整离线结果](docs/experiments/PDMStatePatchTimePositionMMDSeed2022_results_20261016.html)
原MAFirst＋MMD原配方独立seed2022确认完成11轮11825更新，val选第1轮，一次11425test。test MSE/MAE=0.321970725/0.357786145，MA25=0.218403212；较原seed2021 .314177886/.349772139，MSE/MAE +2.480%/+2.291%，MA25 +2.602%。模型、MSE＋MMD、batch32/lr.001/原decaygroups/cosine30/fixed11/ownfullval选择不变，仅seed2021改2022。只有两seed，不声称排除偶然或统计显著。


<!-- MAFirst MMD seed2021 max30patience5 20261016 -->
- [原MAFirst＋MMD：原配方seed2021 max30/patience5完整离线结果](docs/experiments/PDMStatePatchTimePositionMMDSeed2021EarlyStop5_results_20261016.html)
原MAFirst＋MMD seed2021，max30/patience5，完成9轮/9675更新，完整val选第4轮，一次11425test。test MSE/MAE=0.314177886/0.349772139，MA25=0.212865150。保留原batch32/lr.001/Adam衰减分组/cosine30/逐step种子政策。


<!-- MAFirst MMD seed2023 max30patience5 20261016 -->
- [原MAFirst＋MMD：原配方seed2023 max30/patience5完整离线结果](docs/experiments/PDMStatePatchTimePositionMMDSeed2023EarlyStop5_results_20261016.html)
原MAFirst＋MMD seed2023，max30/patience5，完成9轮/9675更新，完整val选第4轮，一次11425test。test MSE/MAE=0.318177583/0.355058654，MA25=0.216304513。保留原batch32/lr.001/Adam衰减分组/cosine30/逐step种子政策。


<!-- MAFirst MMD seed2024 max30patience5 20261016 -->
- [原MAFirst＋MMD：原配方seed2024 max30/patience5完整离线结果](docs/experiments/PDMStatePatchTimePositionMMDSeed2024EarlyStop5_results_20261016.html)
原MAFirst＋MMD seed2024，max30/patience5，完成7轮/7525更新，完整val选第2轮，一次11425test。test MSE/MAE=0.324662904/0.361754184，MA25=0.221594704。保留原batch32/lr.001/Adam衰减分组/cosine30/逐step种子政策。


<!-- MAFirst MMD multiseed max30patience5 20261016 -->
- [原MAFirst＋MMD三seed早停综合结果](docs/experiments/PDMStatePatchTimePositionMMDMultiSeedEarlyStop5_results_20261016.html)
原MAFirst＋MMD max30/patience5三seed2021/2023/2024全部完成；test MSE均值±样本标准差=0.319006124±0.005291386，范围[0.314177886,0.324662904]；MAE=0.355528326±0.006004814。仅3seed、复用val/test，不作显著性或排除偶然保证。


<!-- DistDF official seed2021 20261016 -->
- [DistDF＋Fredformer：官方配方seed2021完整离线结果](docs/experiments/DistDFFredformerOfficialSeed2021_results_20261016.html)
作者公开脚本配方完成45轮/12060更新（最多100轮，patience10），官方shuffle/drop_last128验证选第35轮，一次11425test batch1。test MSE/MAE=0.325386327/0.360594907，MA25=0.218958378；较同配方seed2023 MSE/MAE +3.062%/+1.151%；较作者四舍五入MSE0.316差+0.009386（+2.970%）。仅seed变化；同配方对照已有2023，不重跑旧训练或test；待2021/2023/2024完整汇总后评价种子波动，不作显著性保证。


<!-- DistDF official seed2024 20261016 -->
- [DistDF＋Fredformer：官方配方seed2024完整离线结果](docs/experiments/DistDFFredformerOfficialSeed2024_results_20261016.html)
作者公开脚本配方完成46轮/12328更新（最多100轮，patience10），官方shuffle/drop_last128验证选第36轮，一次11425test batch1。test MSE/MAE=0.323261183/0.359596490，MA25=0.217876873；较同配方seed2023 MSE/MAE +2.389%/+0.871%；较作者四舍五入MSE0.316差+0.007261（+2.298%）。仅seed变化；同配方对照已有2023，不重跑旧训练或test；待2021/2023/2024完整汇总后评价种子波动，不作显著性保证。


<!-- DistDF official multiseed 20261016 -->
- [DistDF官方配方三seed综合结果](docs/experiments/DistDFFredformerOfficialMultiSeed_results_20261016.html)
DistDF官方配方seed2021/2023/2024：test MSE=0.321455862±0.005079714（样本标准差ddof1），范围[0.315720075,0.325386327]；MAE=0.358894578±0.002139459。2023复用既有已核验结果；新增2021/2024各一次训练和test。仅三seed、复用val/test，不作显著性或稳健保证。

<!-- MAFirst MMD horizon192 seed2021 20261016 -->
- [ETTm1 96→192 seed2021完整离线结果](docs/experiments/PDMStatePatchMMDH192Seed2021_results_20261016.html)
ETTm1 seed2021，输入96，预测192；10轮选5，验证MSE 0.511363341；test MSE 0.370313339 / MAE 0.383543388 / MA25趋势MSE 0.257788194。完整保存数组离线核验通过；单seed窗口扩展，不作稳定性或显著性结论。

<!-- MAFirst MMD horizon336 seed2021 20261016 -->
- [ETTm1 96→336 seed2021完整离线结果](docs/experiments/PDMStatePatchMMDH336Seed2021_results_20261016.html)
ETTm1 seed2021，输入96，预测336；8轮选3，验证MSE 0.652465426；test MSE 0.409279717 / MAE 0.407243441 / MA25趋势MSE 0.287044726。完整保存数组离线核验通过；单seed窗口扩展，不作稳定性或显著性结论。

<!-- MAFirst MMD horizon720 seed2021 20261016 -->
- [ETTm1 96→720 seed2021完整离线结果](docs/experiments/PDMStatePatchMMDH720Seed2021_results_20261016.html)
ETTm1 seed2021，输入96，预测720；12轮选7，验证MSE 0.973568040；test MSE 0.457395035 / MAE 0.445180093 / MA25趋势MSE 0.325453272。完整保存数组离线核验通过；单seed窗口扩展，不作稳定性或显著性结论。

<!-- MAFirst MMD ETTm1 multihorizon2021 20261016 -->
- [ETTm1 96/192/336/720综合结果](docs/experiments/PDMStatePatchMMDMultiHorizonSeed2021_results_20261016.html)
ETTm1 seed2021，输入96，预测窗口96/192/336/720：H96: MSE 0.314177886, MAE 0.349772139（9轮选4）；H192: MSE 0.370313339, MAE 0.383543388（10轮选5）；H336: MSE 0.409279717, MAE 0.407243441（8轮选3）；H720: MSE 0.457395035, MAE 0.445180093（12轮选7）。96沿用，另三窗口独立新训练/一次test并离线核验。单seed扩窗，不作跨seed稳定性或显著性结论。

<!-- MAFirst MMD time mass horizon96 seed2021 20261016 -->
- [ETTm1 96→96 seed2021完整离线结果](docs/experiments/PDMStatePatchMMDTimeMassH96Seed2021_results_20261016.html)
ETTm1 seed2021，输入96，预测96；9轮选4，验证MSE 0.396092131；test MSE 0.314177886 / MAE 0.349772139 / MA25趋势MSE 0.212865150。完整保存数组离线核验通过；单seed核质量补偿消融，不作稳定性或显著性结论。

<!-- MAFirst MMD time mass horizon192 seed2021 20261016 -->
- [ETTm1 96→192 seed2021完整离线结果](docs/experiments/PDMStatePatchMMDTimeMassH192Seed2021_results_20261016.html)
ETTm1 seed2021，输入96，预测192；10轮选5，验证MSE 0.508094064；test MSE 0.367767538 / MAE 0.380118412 / MA25趋势MSE 0.255547063。完整保存数组离线核验通过；单seed核质量补偿消融，不作稳定性或显著性结论。

<!-- MMD time mass 96 192 seed2021 20261016 -->
- [MMD时间核质量补偿96/192](docs/experiments/PDMStatePatchMMDTimeMass_results_20261016.html)
ETTm1 seed2021时间核质量补偿：H96 lambda=1.000000, MSE 0.314177886→0.314177886 (+0.000%), MAE 0.349772139, MA25 0.212865150；H192 lambda=1.932723, MSE 0.370313339→0.367767538 (-0.687%), MAE 0.380118412, MA25 0.255547063。H96保存预测/真值/历史/分量等10数组、9轮权重及54梯度精确复现旧96；独立浮点归约可能产生末位指标差异，不算改善。

<!-- future mean information probe 20261017 -->
- [ETTm1未来片段均值：归一化信息线性对照](docs/experiments/PDMStatePatchFutureMeanInformationProbe_results_20261017.html)
本次线性探针中，历史均值/标准差含有可利用的未来水平变化信息。各自训练内选择正则的主A/B对照，验证片段均值MSE下降3.125%，最后48步下降5.896%，但LUFL和LULL整体略变差。固定训练内共同选择的正则后，加入两项使均值MSE下降4.982%，最后48步下降6.099%；只加入均值下降3.920%，只加入标准差下降2.172%。这些幅度不可相加。完整深度模型的片段均值MSE0.332296仍优于B的0.361088（主对照）及0.357546（共同正则）；因此不能据此认定归一化是全部根因或直接替换主干。

<!-- frozen mean head diagnostic 20261017 -->
- [ETTm1 H192冻结主干：历史水平与尺度旁路诊断](docs/experiments/PDMStatePatchFrozenMeanHead_results_20261017.html)
冻结H192补偿版主干后，新增平滑偏移头B（归一化历史+历史均值/标准差）使验证逐点MSE从0.508094064降到0.498378330，下降1.912%；后96步下降2.612%，最后48步下降2.835%。A只读归一化历史反升0.128%，固定偏置反升0.299%；只读均值/标准差的C下降1.540%。共同正则对照B仍比A低2.013%。B在验证起点四个时间段均改善，7变量中6个改善，但LUFL变差。支持下一步保留历史水平/尺度条件旁路，不能据此认定全部根因或替代端到端/新测试验证。

<!-- MAFirst level head E2E horizon96 seed2021 20261017 -->
- [ETTm1 96→96 seed2021完整离线结果](docs/experiments/PDMStatePatchLevelHeadE2EH96Seed2021_results_20261017.html)
ETTm1 seed2021，输入96，预测96；9轮选4，验证MSE 0.394526006；test MSE 0.311538166 / MAE 0.349548011 / MA25趋势MSE 0.210382977。完整保存数组离线核验通过；单seed端到端水平头消融，不作稳定性或显著性结论。

<!-- MAFirst level head E2E horizon192 seed2021 20261017 -->
- [ETTm1 96→192 seed2021完整离线结果](docs/experiments/PDMStatePatchLevelHeadE2EH192Seed2021_results_20261017.html)
ETTm1 seed2021，输入96，预测192；10轮选5，验证MSE 0.496088244；test MSE 0.370844424 / MAE 0.385983659 / MA25趋势MSE 0.258680122。完整保存数组离线核验通过；单seed端到端水平头消融，不作稳定性或显著性结论。

<!-- MAFirst level head E2E horizon336 seed2021 20261017 -->
- [ETTm1 96→336 seed2021完整离线结果](docs/experiments/PDMStatePatchLevelHeadE2EH336Seed2021_results_20261017.html)
ETTm1 seed2021，输入96，预测336；8轮选3，验证MSE 0.622392855；test MSE 0.404983727 / MAE 0.407419745 / MA25趋势MSE 0.283115555。完整保存数组离线核验通过；单seed端到端水平头消融，不作稳定性或显著性结论。

<!-- MAFirst level head E2E horizon720 seed2021 20261017 -->
- [ETTm1 96→720 seed2021完整离线结果](docs/experiments/PDMStatePatchLevelHeadE2EH720Seed2021_results_20261017.html)
ETTm1 seed2021，输入96，预测720；11轮选6，验证MSE 0.869684114；test MSE 0.456115848 / MAE 0.443947320 / MA25趋势MSE 0.323980814。完整保存数组离线核验通过；单seed端到端水平头消融，不作稳定性或显著性结论。

<!-- MAFirst level head E2E group 20261017 -->
- [ETTm1 96/192/336/720端到端水平头结果](docs/experiments/PDMStatePatchLevelHeadE2E_results_20261017.html)

<!-- MAFirst level head root cause diagnosis 20261017 -->
- [MAFirst水平头：具体失效机制与改进顺序](docs/experiments/PDMStatePatchLevelHeadRootCauses_results_20261017.html)
保存数组离线诊断：OT统计头在低水平工况错误上调；720步约81%误差为振幅/形状，负载三变量约77%。提出日历、长历史、收缩门控与分阶段训练的独立对照，未启动新训练。

<!-- MAFirst recurrent gradient diagnosis 20261017 -->
- [MAFirst递推梯度诊断：192/720原模型与水平头](docs/experiments/PDMStatePatchRecurrentGradientDiagnosis_results_20261017.html)
固定训练样本前向/VJP：720最粗递推支路有严重早期记忆衰减及读出收缩，其他路径仍传递远端梯度；无新训练、验证或测试评估。

<!-- MAFirst memory factorial 20261017 -->
- [MAFirst慢记忆×读出正则2×2对照：192/720](docs/experiments/PDMStatePatchMemoryFactorial_results_20261017.html)
原主干四组，两个训练内时间折及标准划分完整对照；所有组统一报告，仅新full组各一次test。状态以QUEUE为准。

- [MAFirst H720：PS方差与相关性损失消融](docs/experiments/PDMStatePatchPSLoss_results_20261019.html)

- 2026-10-20：[MAFirst H720 递推横向残差](docs/experiments/PDMStatePatchTemporalResidual_results_20261020.html)：仅一组新训练，复用原control，无多折或权重搜索。

- 2026-10-20：[MAFirst H720 跨8状态残差](docs/experiments/PDMStatePatchLagResidual_results_20261020.html)：仅新增一组，排在相邻残差之后串行运行，不增加折或搜索。

- 2026-10-20：[MAFirst H720 残差系数均1](docs/experiments/PDMStatePatchUnitResidual_results_20261020.html)：按用户要求替代两旧0.5组，独立预检后才正式训练。

- 2026-10-20：[MAFirst H720 单位残差数值修复](docs/experiments/PDMStatePatchUnitBounded_results_20261020.html)：相邻/跨8两组，系数均1，每步状态RMS约束与梯度范数上限；串行fresh训练。

- 2026-10-20：[MAFirst H720 连续递推两组对照](docs/experiments/PDMStatePatchContinuousCorrection_results_20261020.html)：递推输出去均值、历史条件门控参考状态；各自fresh、串行预检与正式训练。

- 2026-10-20：[MAFirst H720 趋势头读出对照](docs/experiments/PDMStatePatchTrendReadout_results_20261020.html)：原control保留递推均值，仅增加趋势读出32→64→1 GELU残差；单组fresh预检和正式训练。

- 2026-10-20：[MAFirst H720 趋势头读出对照](docs/experiments/PDMStatePatchTrendPatchTransformer_results_20261020.html)：原control保留递推均值，仅增加历史滑动patch4/stride2→1层Transformer→MLP720联合修正；单组fresh预检和正式训练。

- 2026-10-20：[MAFirst H720 混合前Transformer对照](docs/experiments/PDMStatePatchTrendPreMix_results_20261020.html)：保留现有混合后组，新增一组Transformer在PDM前、MLP在PDM后，严格串行。
