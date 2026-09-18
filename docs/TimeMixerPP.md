# TimeMixer++ in TSLib

本实现依据用户提供的 TimeMixer++ 论文（ICLR 2025，arXiv:2410.16032v5）
第 3 节与附录 B 独立编写。论文：[arXiv](https://arxiv.org/abs/2410.16032)；
作者仓库：[kwuking/TimeMixer](https://github.com/kwuking/TimeMixer)。

2026-09-18 核对作者仓库时，main 分支的 models 目录仅有原版 TimeMixer.py。
**这里不是官方代码移植，也不声称已复现论文表格中的精度。**

## 文件与框架接口

- `models/TimeMixerPP.py`：可训练下采样、通道注意力、输入嵌入、任务输出头。
- `layers/TimeMixerPP.py`：MRTI、TID、MCM、MRM 和残差 MixerBlock。
- `tests/test_timemixerpp.py`：结构、任务接口、掩码、梯度和重载测试。
- 自动模型发现机制直接支持 `--model TimeMixerPP`，无需修改模型注册表。
- 所有任务沿用公共 `utils/logger.py`，文本日志和 TensorBoard 分别写入
  `logs/`、`runs/`。模型初始化信息也通过公共 logger 输出。

支持长期预测、短期预测、补全、异常检测和分类。预测返回
`[B,pred_len,c_out]`，重建返回 `[B,seq_len,c_out]`，分类返回
`[B,num_class]`。编码器不读取 decoder 数值或未来时间标记。
沿用相应任务的优化器、损失和早停逻辑；没有改变原版 TimeMixer。

## 论文结构对应

| 论文 | 实现 |
|---|---|
| 式 (1) 多尺度输入 | 注册在 ModuleList 中的 stride=2 Conv1d，长度 floor(T/2^m) |
| 式 (2) 输入投影 | 最粗尺度沿变量轴做 Q/K/V 注意力，再嵌入所有尺度 |
| 式 (5)-(6) MRTI | 最粗尺度 FFT 排除 DC 后选 top-K，补零并形成时间图像 |
| 式 (7) TID | Conv2d 生成 Q/K/V，周期内季节性注意力与跨周期趋势注意力 |
| 式 (8)-(10) MCM | 季节性从细到粗卷积混合，趋势从粗到细转置卷积混合 |
| 式 (11) MRM | 使用每个样本的 FFT 幅度 softmax 融合各分辨率 |
| 式 (4) 残差块 | 各尺度分别执行 residual + LayerNorm |
| 式 (3) 输出 | 每个尺度独立预测头，均值集成 |

## 论文未完全明确的实现细节

1. **周期**：按式 (5) 采用 `p=ceil(T/f)`，T 为原始长度，所有尺度共享周期。
   频率索引根据 batch/嵌入维平均幅度选择，融合权重保留各样本幅度并可求梯度。
   因此预测可能随同批样本变化，基准比较应固定 batch 配置。
   K 超过可用非零频率数时自动截断，常数输入不会选择 DC 或发生除零。
2. **双轴注意力**：图像布局为 `[B,D,period,ceil(T_m/period)]`，每列是连续周期。
   按图 7 与“周期内/跨周期”的定义实现；正文中 Q/K/V 形状标注与该定义有矛盾。
   两个轴分别使用 3×3 Q/K/V 卷积和多头注意力；同一块内参数在所有尺度、
   所有周期之间共享。右侧补零位置不参与有效 key/query。
3. **尺度混合**：每条边为两层卷积和 GELU，净 stride=2，第二层 stride=1。
   若两层均 stride=2 会错误缩放四倍。仅沿周期数量轴缩放；
   宽度舍入差用右侧裁剪/补零处理，周期内轴不做插值。
4. **具体层配置**：输入卷积 kernel=3、padding=1，奇数长度裁剪到 floor(T/2)；
   输入嵌入用共享 Linear，并可叠加框架已有的历史时间特征。
   反向混合第一层用 kernel=(3,4) 的 ConvTranspose2d。输出取各尺度均值。
   这些核宽、隐层配置与均值集成是对论文未给定细节的明确选择。
5. **数据适配**：非分类任务默认逐样本归一化/逆变换，`--use_norm 0` 可关闭。
   补全只统计已观测值，全缺失通道保持有限值；分类在输入、尺度、注意力
   和输出头处屏蔽填充位置。MS 单目标使用数据加载器最后一列的统计量。
6. **范围**：实现从头监督训练。论文预训练、linear warm-up、跨数据集 few/zero-shot
   实验及预训练权重未复现。直接选择 `zero_shot_forecast` 会明确报错，
   避免把随机初始化网络当作预训练模型。附带脚本使用本地 Adam + cosine 配置。

## 参数与启动

| 参数 | 含义 |
|---|---|
| `--e_layers 2` | MixerBlock 数 L |
| `--down_sampling_layers 3` | 下采样次数 M，共 M+1 个尺度，最粗长度须 >=2 |
| `--down_sampling_window 2 --down_sampling_method conv` | 可训练二倍下采样 |
| `--top_k 3` | FFT 分辨率数量 K |
| `--d_model 32 --d_ff 64 --n_heads 4` | 嵌入维、尺度卷积隐层维、双轴注意力头数 |
| `--channel_independence 0` | 明确声明通道混合；该模型始终执行论文通道注意力 |

`d_model` 必须能被 `n_heads` 整除；通常 `c_out=enc_in`。
多变量预测单目标可用 `--features MS --c_out 1`，目标为最后一个通道。
`--down_sampling_layers 0` 支持单尺度消融。短序列应减少尺度数。

```bash
conda activate TSLib

# ETTm1：预测长度 96 / 192 / 336 / 720
bash scripts/long_term_forecast/ETT_script/TimeMixerPP_ETTm1.sh

# M4 六个子集（需先准备完整 M4 数据）
bash scripts/short_term_forecast/TimeMixerPP_M4.sh

tensorboard --logdir ./runs --port 6006
```

脚本与仓库其他模型采用相同格式：顶部定义公共参数，各个预测长度或 M4 子集
分别使用一段 `python -u run.py` 命令，每个参数独占一行。请在项目根目录运行。
修改顶部的 `batch_size`、`learning_rate`、`train_epochs`、`num_workers`
即可调整实验；重复次数在命令块的 `--itr` 中修改。只运行某一组实验时，
单独执行对应命令块；单轮验证可将 `train_epochs` 改为 `1`。
ETTm1 起始配置为 T=96、M=3、K=3、L=2、D=32，batch=16、lr=0.001、
cosine；这是可运行配置，未完成论文级超参数搜索。

其他任务沿用该任务原有数据参数，设置模型名、卷积下采样和通道混合。
补全与异常检测可设 `pred_len=0`，分类实验类会自动配置通道、长度和类别数。

## 验证

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

测试覆盖周期提取、图像顺序、双轴注意力含义、双向跨尺度信息流、
五类任务输出和有限梯度、缺失/填充不泄漏、卷积参数更新、
checkpoint 重载一致性、单尺度和 CUDA 混合精度（无 CUDA 时跳过）。
小样本训练检查工程链路，不代表论文基准精度。

已在 TSLib 环境（PyTorch 2.13.0+cu130，RTX 5060）完成 ETTm1 四个预测长度
96/192/336/720 的小样本 GPU 单轮训练、验证与测试，核对了 checkpoint、
日志及 TensorBoard 指标。每次仅取训练 32 个、验证/测试各 16 个窗口；
未运行完整数据基准，也未完成 M4/分类等其他数据集的精度评测。
