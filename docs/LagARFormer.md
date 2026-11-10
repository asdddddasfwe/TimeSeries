# LagARFormer v5：四分辨率滞后消息预测

> 2026-09-30 清理说明：本文涉及的已移除诊断脚本已保存至 [归档](/home/abc/PythonProject/Time-Series-Library-main/experiments/cleanup_20260930_diagnostic_scripts/retired_diagnostic_scripts.zip)；[文件清单](/home/abc/PythonProject/Time-Series-Library-main/experiments/cleanup_20260930_diagnostic_scripts/manifest.json) 记录原始路径及 SHA256。历史命令需先按原路径恢复相应脚本；既有结果保持不变。

当前模型保留 FFT 周期行列注意力、并行滞后消息预测以及趋势线性处理，
扩展为四个独立分辨率的预测分支，在原始时间网格上融合。

## 数据流与四个尺度

    输入 [B,T,C] -> 一次 RevIN -> 原始长度移动平均分解（默认窗口 25）
                       季节 / 趋势分别平均池化三次，步长均为 2
           stride=1       stride=2       stride=4       stride=8
           T             T/2            T/4            T/8
      每尺度季节：FFT 周期行列编码 -> 未来特征 -> 滞后消息传播 -> Linear(D,1)
      每尺度趋势：Linear(T_i,H_i)
                       同尺度季节与趋势预测相加
                       粗尺度恢复原始步长，裁至 H
                       Linear(4,1) 可学习融合
                       一次逆 RevIN -> [B,H,C]

输入长度须能被 8 整除且至少为 16，最粗尺度至少有两个观测。
只在最细尺度归一化和分解一次；分别池化季节与趋势，所有分支共享归一化坐标系。
各尺度编码器、时间投影、消息层和预测头参数独立；同尺度内参数跨变量共享，
不进行变量间注意力。M/S/MS 与长期/短期预测接口保留。

## 时间单位与粗尺度输出对齐

历史长度 T_i=T/stride_i，未来长度 H_i=ceil(H/stride_i)。
以 T=96 为例：

| 步长 | 历史长度 | H=96 的本地预测长度 | H=13 的本地预测长度 | 默认候选 lag |
|---|---:|---:|---:|---|
| 1 | 96 | 96 | 13 | 1..95 |
| 2 | 48 | 48 | 7 | 1..47 |
| 4 | 24 | 24 | 4 | 1..23 |
| 8 | 12 | 12 | 2 | 1..11 |

每个分支的历史和未来都使用相同采样间隔：
粗尺度 lag=tau 对应原始时间的 stride_i*tau 个采样点。
FFT 与自相关在该尺度的数据上重新计算，周期 P=floor(T_i/频率索引)。

本地未来第 j 个值通过 repeat_interleave(stride_i) 映射到原始未来区间
[j*stride_i,(j+1)*stride_i)，超过 H 的末尾裁掉。
粗尺度表达区间内的平滑水平，最细尺度仍能表达逐点变化。
这种对齐不使用真实未来标签，也不会把粗尺度 lag 误当成原始步长。

## 每个季节分支

对本地季节历史独立做 rFFT，排除 DC，选 top_k 个主频。
Linear(1,D) 加本地正弦位置编码后的时域特征，按 FFT 周期补零并 reshape：

    [B*C,T_i,D] -> [B*C,D,ceil(T_i/P),P]

复用 TimeMixer++ 的 RowAttention 和 ColAttention：周期内部与跨周期同相位两个分支并行。
每组是 x + Dropout((Row(x)-x)+(Col(x)-x)) 后 LayerNorm，默认 e_layers=2。
沿用上游单头注意力、Q/K/V 1x1 卷积参数和零初始化 gamma；
实际计算将六组 1x1 投影合并为一次线性运算，注意力使用 scale=1 的 SDPA 保持原始未缩放公式。
d_ff 表示 Q/K 宽度，n_heads 不参与这些模块。补零不屏蔽，重复周期合并幅值权重。
还原并裁回 T_i 后，按 FFT 幅值 softmax 融合，保留完整时间特征。

Linear(T_i,H_i) 并行生成全部本地未来向量，并添加本地未来位置编码。
默认两层同步消息传播；全部候选 lag 都参与，自相关只是关系特征，不硬筛选五个。
每个 lag 使用七项历史统计：lag/T_i、全窗/前半窗/后半窗 ACF、前后差和配对比例。
长 lag 或很短尺度上的半窗统计若不可估计，则统计与可靠性为零。

每层固定读取上一层未来快照及历史编码：

    source(h,tau) = history[T_i+h-tau]    当 h-tau < 0
                  previous_future[h-tau] 当 h-tau >= 0

目标、来源、lag 向量分别投影，相加后 GELU 生成消息；按 lag 汇总并除以 sqrt(L)，
经 LayerNorm 和残差 MLP 更新未来特征，最终 LayerNorm+Linear(D,1) 输出季节预测。
层内全部未来位置并行更新，层间顺序执行；没有 AR 联立求解或 DCT 驱动项。

## 趋势与融合

四个趋势头都是 Linear(T_i,H_i)，初始化权重为 1/T_i、偏置为零，无额外非线性处理。
同尺度季节/趋势预测相加，恢复原始分辨率后堆叠为 [B,H,C,4]。
无偏置 Linear(4,1) 初始权重均为 0.25，即四尺度平均；
权重可训练且不经 softmax，不要求训练后为正或和为 1。
融合后只逆归一化一次；MS/c_out=1 在逆变换后取最后一列。
各尺度在预测层融合。

## 配置和运行

~~~bash
conda activate TSLib
RUN_TAG=lar_multi_v5 bash scripts/long_term_forecast/ETT_script/LagARFormer_ETTm1.sh
~~~

标准脚本为 H=96/192/336/720 四条独立命令，T=96、D=32、d_ff=64、
top_k=5、e_layers=2、lag_layers=2、dropout=0.2、batch=32。
Adam lr=0.001、weight_decay=1e-4、cosine、patience=10、最多 30 轮、实际种子 2021。

必须使用 down_sampling_layers=3、down_sampling_window=2、down_sampling_method=avg。
lag_max=0 表示每尺度的全部 1..T_i-1；正数表示本地 lag 上限，各尺度截断至 T_i-1，
因此同一 lag_max 在不同尺度对应不同的实际时间范围。
lag_chunk_size=8 只控制计算分块，不删减 lag；训练使用激活重计算减少消息内存。
top_k 仅控制每尺度 FFT 主频数，自动截断到可用非零频率数。

freq=h 保留现有脚本设置，不会重采样 ETTm1 文件；原始 ETTm1 四个采样点仍为一小时。
检查点前缀 LARmulti_v5_；v4 和更早的检查点不兼容，需重新训练。
旧 AR 参数已删除。v4 源文件与说明备份在
tmp/lagarformer_multires_change_20260921_210150/，既有实验产物与其它网络保留。

## 验证与限制

~~~bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/smoke_lag_ar_former.py
~~~

真实 ETTm1 CPU 冒烟：标准划分前 32/16/16 个训练/验证/测试窗口，
batch=8、workers=0、一轮，隔离目录 tmp/lagarformer_multires_smoke_<时间>_<pid>/。
核对全部分支更新、四尺度本地预测长度、完整本地 lag、日志、TensorBoard、
严格检查点恢复；子集损失不用于评价模型优劣。

单元测试覆盖池化对齐、四尺度分解重构、粗尺度时间单位、非整除预测长度、
尺度融合、参数独立性，以及 FFT、同步消息、梯度、未来隔离、M/S/MS 和 batch 独立性。

池化和粗尺度展开引入平滑；这是多分辨率分支的建模偏置。最细尺度保留逐点路径。
模型参数量增加；分块重计算节约内存但增加开销。多尺度不保证减轻过拟合或改善精度，
需要完整训练对照。当前仍由网络学习各 lag 的作用，没有额外的可学习 lag 选择门。

## 本次验证记录（2026-09-21）

- 全仓库 79 项测试通过，其中本模型专项 30 项。
- 真实 ETTm1 CPU 冒烟：实际 seed=2021，32/16/16 窗口，batch=8、workers=0、一轮。
- 四个季节编码器、四个趋势头和尺度融合层共 281/281 参数张量发生更新。
- 本地长度/预测长度为 [96,48,24,12]，候选 lag 数为 [95,47,23,11]。
- 预测形状 [16,96,7]，日志/TensorBoard 标量有限，严格恢复检查点后预测逐值一致。
- [完整冒烟报告](../tmp/lagarformer_multires_smoke_20260921_210548_1313494/summary.json)。
- 四段脚本通过 Bash 语法检查；未启动完整训练。

## 等价加速（2026-09-22）

本次保持四尺度、全部 lag、FFT top-k、行列注意力和消息层数量不变：

- ACF 使用补零 FFT 计算线性自相关，保留原来的中心化、能量归一化与不可估计 lag 补零；不是循环相关。
- 每尺度一次性拷贝整数周期索引到 CPU 分组，避免每个周期调用 GPU nonzero 引发同步；幅值权重与梯度留在原设备。
- 每层六组 Q/K/V 投影合并，并在同次前向的周期组之间复用参数拼接；行列分支仍并行读取相同输入。
- SDPA 显式 scale=1、attention dropout=0，保留原始残差 dropout。Q/K 较宽时对 V 补零，计算后裁回原宽度。
- 消息分块的重计算关闭 RNG 状态保存；分块内部没有随机算子，外部 dropout 保留。

v5 参数名称、形状及版本标识保持一致，旧 v5 检查点可严格加载。
浮点运算顺序变化会造成微小误差，不承诺完整训练轨迹或最终指标逐值相同。
运行中的 Python 进程不会自动切换代码；下一次新启动的训练或评估生效，无需修改脚本参数。

### 性能对照

RTX 5060、PyTorch 2.13.0+cu130、FP32、seed=2021，T=96、H=192、C=7、
batch=32、lag_chunk_size=8。合成输入，包含前向、反向、有限值检查和 Adam 更新，
不含数据加载；两实现各预热 3 步，交替测试 10 步，报告中位数：

| 实现 | 单步耗时 |
|---|---:|
| 优化前 | 0.750 s |
| 优化后 | 0.436 s |

约 1.72 倍加速、耗时减少 42%。与既有训练共享 GPU，耗时有波动，不能直接当作独占 GPU 的完整训练吞吐。
同权重评估（注意力 gamma=0.2）最大绝对差为 2.38e-7。
[原始 A/B 报告](../tmp/lagarformer_speed_20260922_092239/benchmark_cuda_b32_h192_c8.json)。

~~~bash
python scripts/benchmark_lag_ar_former.py \
  --reference-dir tmp/lagarformer_speed_20260922_092239 \
  --device cuda --batch-size 32 --pred-len 192 --warmup 3 --steps 10
~~~

reference-dir 是本机优化前源码备份；其它机器需自行保存同结构的旧版本源码。
日志 speed 的单位为 s/iter，即单次迭代秒数，虽然每 100 步才打印一次。

### 验证

全仓库 81 项测试通过，其中 LagARFormer 32 项；新增 FFT ACF 和融合行列注意力
对原始公式的双精度输出、输入梯度和全部参数梯度对照。
真实 ETTm1 CPU 冒烟使用 32/16/16 个窗口、batch=8、一轮、seed=2021：
281/281 参数张量更新，日志和 TensorBoard 正常，严格检查点恢复逐值一致。
[本次冒烟报告](../tmp/lagarformer_multires_smoke_20260922_093108_1460913/summary.json)。
冒烟子集指标不用于评价预测效果；未启动完整训练，也未停止既有训练。

## ACF 通路训练改进（2026-09-22）

标准 ETTm1 脚本现在默认设置 lag_relation_weight_decay=0：
四尺度 lag_embedding 和各消息层 lag_projection 共 24 个参数张量不做权重衰减，
其余 257 个参数张量继续使用 weight_decay=1e-4，包括 source_projection。
优化器仍为 Adam，网络结构、预测公式、参数名称和 v5 检查点格式不变。
从正常初始化开始训练，避免直接在已经退化的关系通路上微调。

新增参数仅用于长期预测训练：
- --lag_relation_weight_decay：默认 -1 继承整体 weight_decay；0 关闭关系通路衰减；正数指定该通路的衰减。
- --lag_monitor：可选监控，标准脚本开启；不参与损失、早停或参数更新。
- --lag_monitor_samples：默认 16，固定均匀抽取验证窗口。

每轮首批反向传播后记录 lag/scale0..3/relation_parameter_l2、
relation_gradient_l2、source_parameter_l2、source_gradient_l2；
AMP 梯度范数会除去 GradScaler 缩放。训练开始和每轮结束记录
lag/probe/full_mse、zero_acf_mse、prediction_change_rms、mse_change。
ACF 置零只移除四列相关统计，保留 lag 位置和配对比例。
监控使用 eval/FP32，不遍历随机 DataLoader；结束后恢复训练模式和 RevIN 临时统计，
不修改梯度、检查点或训练随机序列。probe MSE 是小子集指标，不能替代完整验证结果。

~~~bash
# 新配置：四段独立命令，依次训练 96/192/336/720。
RUN_TAG=acf_train_v1 bash scripts/long_term_forecast/ETT_script/LagARFormer_ETTm1.sh

# 同结构旧衰减基线：只改变关系通路的衰减设置。
LAG_RELATION_WEIGHT_DECAY=-1 RUN_TAG=acf_baseline bash scripts/long_term_forecast/ETT_script/LagARFormer_ETTm1.sh
~~~

当前脚本产物前缀包含 LARmulti_v5_adam_acfwd0 或 LARmulti_v5_adam_acfwd-1，以及 RUN_TAG。
手动调用 run.py 时，添加 --lag_relation_weight_decay 0 --lag_monitor 即可启用；
未指定新选项时保留原训练行为。其他模型不启用这些 LagARFormer 监控选项。

### 配对短测结果

ETTm1 96→96，RTX 5060，FP32，seed=2021、batch=32，
均匀选取训练 2048 个窗口、验证 256 个窗口，四轮共 256 个优化步骤/模型。
两组初始权重、批次和每步 dropout 随机状态相同；没有使用测试集。
学习率按原 30 轮 cosine 曲线的前四轮执行。敏感性探针另取上述验证子集中的 32 个窗口。

| 第四轮指标 | 原衰减 | 关系通路衰减为 0 |
|---|---:|---:|
| 验证子集 MSE | 0.416765 | 0.416758 |
| 最细尺度关系参数 L2 范数 | 1.699 | 6.858 |
| ACF 置零后的预测 RMS 变化 | 3.09e-6 | 3.30e-4 |

短测显示早期参数收缩减轻，模型对 ACF 更敏感；验证精度基本持平。
这不是完整训练或多种子精度结论，也不证明粗尺度来源通路已恢复。
本次只落实诊断计划的第一项，未同时加入新消息门控、辅助损失或历史检索。

[配对短测报告](../tmp/lag_relation_pilot_20260922_100219/summary.json)；
[复现脚本](../scripts/pilot_lag_relation_decay.py)，默认仅运行四轮子集对照。
86 项仓库测试通过，覆盖参数组完整性、衰减行为、多组 cosine、v5 严格加载、
监控的随机数/梯度/训练模式/RevIN 无副作用及 MS。
[真实数据冒烟报告](../tmp/lagarformer_multires_smoke_20260922_100044_1467383/summary.json)：
281/281 参数张量更新，新增 TensorBoard 标签有限，严格恢复预测逐值一致。
未启动标准四任务完整训练。

## 结构实验候选（2026-09-22）

新增独立候选 [LagARFormerGated](LagARFormerGated.md)，仅替换最细尺度的关系编码与消息聚合。
原 v5 模型和上述脚本保留。实际对照、训练预算及结论见
[研究记录](experiments/LagARFormer_research_20260922.md)，不同预算的指标不能直接比较。

## 优化器显式对照

长期预测训练器增加 --optimizer adam/adamw，默认仍为adam。
标准 LagARFormer 脚本可用 OPTIMIZER 环境变量切换，产物前缀包含优化器名称：

~~~bash
OPTIMIZER=adamw RUN_TAG=optimizer_trial bash scripts/long_term_forecast/ETT_script/LagARFormer_ETTm1.sh
~~~

两种优化器都保留关系参数分组和cosine调度。相同weight_decay数值在Adam耦合L2
与AdamW解耦衰减中不代表相同正则化强度；选择依据见研究报告。
另有可选[直接历史检索候选](LagARFormerMemory.md)，其当前验证收益较小，未替换默认模型。

## 季节历史特征混合实验

独立模型 [LagARFormerMix](LagARFormerMix.md) 在 FFT 行列编码后、F0 和 lag 解码前加入96→48→24→12的细到粗残差混合。原模型保持不变；FFT/ACF仍从未混合的季节数值计算。

## 线性趋势融合版本

[趋势融合说明](LagARFormerTrendMix.md)提供 LagARFormerTrendMix（仅趋势粗到细）及 LagARFormerDualMix（季节细到粗＋趋势粗到细）。两者使用独立脚本和检查点，已有版本保持原样。

## 混合退化与ACF依赖诊断

[StableMix实验说明](LagARFormerStableMix.md)记录混合参数保护、ACF训练期随机关闭和冻结检查点的强度校准。候选方案仍处于验证阶段，不替换原最佳检查点。

## Source-only messages and exact output recurrence (experimental)

[Source-only single-resolution messages](LagARFormerSource.md) and [same-output linear recurrence](LagARFormerRecurrence.md) are independent research candidates. The [two-seed structural study](experiments/LagARFormer_structure_20260922.md) did not show either outperforming the original four-scale model; existing defaults remain unchanged.

## Parallel oscillatory state decoding

[OscillatorMixer](OscillatorMixer.md) preserves the four-resolution framework and replaces seasonal lag messages with a history-conditioned, same-layer state recurrence computed by an affine tree scan. It is an independent model; existing defaults and checkpoints are unchanged. Implementation, CPU smoke and CUDA/AMP verification are recorded in its guide; prediction quality has not yet been benchmarked.
