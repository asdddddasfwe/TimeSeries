# SeasonTrendMixer

独立的季节/趋势混合预测网络，入口为`models/SeasonTrendMixer.py`，
可复用组件位于`layers/SeasonTrendMixer.py`。框架自动发现`--model SeasonTrendMixer`。
这是自定义实验结构，效果需要完整训练验证。

## 数据流

输入`[B,T,C]` → 一次无仿射 RevIN → 一次原始尺度移动平均分解
（默认窗口25）→ 季节与趋势分别进行三次二倍平均池化。
默认四尺度为96、48、24、12，不重复分解或分支归一化。

| 分支 | 尺度 | 混合前处理 | 预测头 |
|---|---|---|---|
| 季节细尺度 | 96、48 | FFT分段 → patch内卷积 → token＋位置编码 → Transformer → 解码 | 各自 Linear(T_i,pred_len) |
| 季节粗尺度 | 24、12 | 各自 Linear(T_i,T_i) | 各自 Linear(T_i,pred_len) |
| 趋势全部尺度 | 96、48、24、12 | 各自 Linear(T_i,T_i) | 各自 Linear(T_i,pred_len) |

**六个前置时间线性层**独立、单位矩阵初始化、零偏置，不附加激活。
**八个预测头**独立，权重初始化为`1/T_i`、偏置为零；参数仅跨变量共享。

季节按细→粗、趋势按粗→细进行跨尺度混合，使用 TimeMixer 形式的
`Linear→GELU→Linear`映射与加法残差。因此趋势前置层与预测头是线性的，
跨尺度混合含非线性。先将同尺度季节和趋势预测相加，再**求和四尺度**，
最后统一逆 RevIN 一次。四尺度不会除以4。

## 周期patch与Transformer

- 每个样本、每个变量在各自细尺度季节序列上选非零rFFT频率，默认top-k=3，
  超过可用频率数时截断。周期宽度为`P=floor(T_i/频率索引)`。
- 每个周期是一段连续、不重叠的patch。序列末尾补零至P的整数倍，
  得到`[B*C, ceil(T_i/P), P]`，最后一个不完整patch也保留。
- 每个patch单独进入`Conv1d(1,D,kernel=3,padding=1) → GELU → Conv1d(D,1,kernel=1)`，
  输出与原patch残差相加。三点卷积提取局部波形，1×1卷积将D个特征通道投影回单通道。
  卷积输入为`[B*C*patch数, 1, P]`，因此不会跨patch、变量或样本串联数据。
  卷积边界使用零填充；最后一个不完整patch保留末尾补零，不额外使用mask。
- 为共享不同周期的编码器，**局部卷积之后**每段patch右侧再补零至本尺度最大宽度T_i，
  经同一个`Linear(T_i,D)`编码为token。保留原始点序，无重采样、插值或平均池化。
  所有参数在初始化时注册，不在forward中创建新的线性层。
- token加入按**patch顺序**索引的固定正弦位置编码，然后进入标准
  Pre-LN Transformer：多头注意力、GELU FFN、残差和dropout。
  注意力仅在周期token之间计算，`e_layers`现在表示Transformer层数。
- `Linear(D,T_i)`解码每个token，截取前P点，再按patch顺序拼接并裁回T_i。
  各周期结果还原到相同时间网格后，按FFT幅值softmax权重融合。
- 重复周期先合并权重，同周期的样本/变量分组计算。每尺度每次前向只选一次周期。
  两个尺度参数独立；同尺度的局部卷积、编码、Transformer和解码在所有周期、变量之间共享。
- **不使用填充mask**。P=T_i时只有一个token，此时没有跨patch交互，
  但投影和FFN仍正常执行。不修改选频规则来强制增加token。
- 通道独立、不读取未来输入或时间标记；评估不依赖batch同伴。
  半精度FFT仍以float32计算。

例如T_i=96、P=24：96点划成4个24点patch，每段先做局部卷积与残差，再补零到96维，
经Linear(96,D)得到4个token，分别加入第0、1、2、3个patch的位置编码；
Transformer之后以Linear(D,96)解码，
每个token取前24点，拼接回96点。P只决定分段长度，**不由`--patch_len`控制**。

## 配置与运行

固定四尺度、二倍平均池化、通道独立、移动平均分解及输入归一化。
`seq_len`须被8整除且至少16；`moving_avg`须为正奇数；
`d_model`须被`n_heads`整除。不符合配置直接报错。

支持长期/短期预测和M/S/MS。S需要`enc_in=c_out=1`；
M需要`c_out=enc_in`。MS支持全通道输出，或设置`c_out=1`在融合及逆归一化
后选最后一列。短期任务通过标准模型接口；数据集和损失仍由现有训练器负责。

标准ETTm1脚本包含96、192、336、720四段独立命令：

```bash
RUN_TAG=stm_patch_conv_v1 bash scripts/long_term_forecast/ETT_script/SeasonTrendMixer_ETTm1.sh
```

默认输入96，D=32，FFN=64，heads=4，patch Transformer层数=2，dropout=0.2，
batch=64，Adam，lr=0.001，weight_decay=1e-4，cosine，patience=10，
最多30轮，freq=h。`run.py`实际固定随机种子2021；`--seed`是数据增强参数。
`freq=h`只是框架时间特征配置，不会把ETTm1的15分钟观测重采样成小时，
且本模型不消费时间标记。

RUN_TAG进入模型/实验ID以隔离日志、TensorBoard、检查点和结果；
未指定时使用时间戳和进程号。脚本ID前缀为STMpatchconv，区别于旧patch版的STMpatch和行列版的STM。
不同实验应使用不同RUN_TAG。
初始化日志记录encoding=period_patch_conv、patch内残差卷积、逐patch位置编码、六个前置层、FFT周期公式、
无填充屏蔽和尺度求和。

## 测试与隔离冒烟

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/smoke_season_trend_mixer.py --device cuda
```

CPU可使用`--device cpu`。冒烟工具读取真实`dataset/ETT-small/ETTm1.csv`，
保留标准数据划分及训练集拟合的标准化器，仅取训练前32个窗口、
验证和测试各前16个窗口，batch=8、workers=0、训练一轮、实际种子2021。
复用原训练器、日志、TensorBoard和早停检查点，严格恢复并比较预测。

产物全部隔离在`tmp/seasontrendmixer_patch_conv_smoke_<时间>_<PID>/`，
包括`summary.json`、日志、events、checkpoint、预测及指标；
不会写入已有实验目录。子集指标仅用于确认训练流程可运行，**不能用于比较模型效果**。
专项测试覆盖分解/下采样、周期patch、局部卷积与边界隔离、逐patch位置编码、
填充裁剪、token轴与尺度方向、卷积及六个前置层与八个头更新、
有限梯度、一次逆归一化、batch/通道独立、未来输入隔离、MS和严格检查点往返。

## 检查点与本次验收

当前版本在季节细两尺度的周期patch编码中增加局部卷积，位置编码继续按patch序号加入。
旧纯线性patch检查点缺少局部卷积参数，旧行列版检查点的参数结构也不同，
均不能直接严格加载；本版本需重新训练。运行中的进程不会自动更新为新网络。
六个前置线性层、八个预测头、尺度混合、归一化与训练超参数保持原定义。

2026-09-21：23项模型专项测试和全部49项仓库回归测试通过。
为不占用已有GPU训练，本次在CPU完成真实ETTm1的32/16/16窗口、
batch=8、workers=0、seed=2021单轮冒烟。
四个局部卷积权重矩阵、六个前置层与八个预测头均更新，TensorBoard标量有限，
新检查点严格恢复的预测完全一致。
产物见[局部卷积patch版冒烟摘要](../tmp/seasontrendmixer_patch_conv_smoke_20260921_143629_1222512/summary.json)。
历史结果及检查点均保留；修改前源码备份位于
`tmp/seasontrendmixer_patch_conv_change_20260921_143431/`。
本次未启动完整训练，子集损失不作为预测效果结论。
