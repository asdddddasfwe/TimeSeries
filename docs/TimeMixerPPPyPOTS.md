# TimeMixer++：保留最佳已测配置

当前只保留 `TimeMixerPPPyPOTS` 这一套 TimeMixer++ 实现。
依据 ETTm1 96→96 的已有消融记录，保留 avg/coarsest 配置；
这是当前已测试配置中测试表现最好的结构，不代表所有数据集或预测长度均最优。

## 固定网络路径

- 平均池化下采样，CI=1逐通道共享编码；最粗尺度保留通道注意力。
- 最粗尺度FFT排除DC，周期为 `floor(T_coarse/f)`；top-k受可用频率数量限制。
- 周期内部注意力进入细→粗季节混合，跨周期注意力进入粗→细趋势混合。
- 不屏蔽填充位置；所有尺度预测头等权平均，再用最细尺度RevIN逆变换一次。
- 保留半精度FFT转float32、补零dtype和常量输入等稳定性修复。
- 周期选择仍对batch/嵌入维频谱求均值，因此评估必须保持相同batch大小与顺序。

模型仅支持长期/短期预测，不使用日历标记或decoder未来值。
MS单目标在全通道预测与逆RevIN之后取最后一列。
指标默认在数据集标准化尺度上；框架 `--inverse` 才恢复数据原始单位。
`freq=h` 不改变ETTm1的15分钟采样间隔。

## 运行与默认配置

```bash
conda activate TSLib
RUN_TAG=my_run bash scripts/long_term_forecast/ETT_script/TimeMixerPPPyPOTS_ETTm1.sh
```

脚本仍为四段独立命令，预测长度96、192、336、720。
输入96，D=32，FFN=64，2层，heads=4，top-k=3，num_kernels=3；
3次二倍平均池化，batch=64，Adam lr=0.001，weight_decay=1e-4，
cosine，dropout=0.2，最多10轮，patience=3，freq=h。
实际随机种子为2021；`--seed` 是数据增强参数。
RUN_TAG隔离产物，未指定时默认时间戳。

要求输入长度被 `down_sampling_window**down_sampling_layers` 整除，
最粗长度至少2，heads不超过最粗长度，d_model为正偶数。
`down_sampling_method` 仅接受avg；CI仅接受1。

## 保留依据

以下均使用验证集选择各次训练的最佳检查点，再报告测试指标：

| 配置 | 最佳验证MSE | 测试MSE | 测试MAE |
|---|---:|---:|---:|
| **保留：avg/coarsest，10轮上限、patience=3** | 0.3933821 | **0.3165866** | **0.3557742** |
| avg/coarsest，30轮上限、patience=10 | 0.3936028 | 0.3166398 | 0.3557787 |
| conv/coarsest，30轮上限、patience=10 | 0.3997545 | 0.3208680 | 0.3582434 |
| conv/original，30轮上限、patience=10 | 0.3882262 | 0.3236443 | 0.3611622 |

保留配置最佳第2轮、第5轮早停。历史同结构ensemble测试MSE为0.316573，
与复跑基本一致。original+conv验证指标更低，但测试指标没有改善。
这些是单种子历史比较，清理过程没有重新调参或挑选训练中的最低测试轮次。

详细证据保存在[基线报告](experiments/TimeMixerPPPyPOTS_baseline_96_20260920.md)、
[延长早停报告](experiments/TimeMixerPPPyPOTS_downsampling_patience10_96_20260920.md)和
[original+conv报告](experiments/TimeMixerPPPyPOTS_original_conv_patience10_96_20260920.md)。

## 清理范围与检查点

已移除旧独立 `models/TimeMixerPP.py`、对应层/测试/文档、ETTm1和M4脚本、
正则化与pipeline对照脚本，以及PyPOTS共享卷积/original周期代码和两个长训练消融脚本。
CLI删除 `--timemixerpp_embedding`、`--timemixerpp_head_dropout`、
`--pypots_period_basis`；旧消融选项会明确报错，不会静默解释成最佳配置。

最佳avg/coarsest检查点的参数名、形状和初始化顺序保持不变，可严格加载。
其他架构/周期的历史检查点不保证语义兼容，即使形状相同也不应混用。
历史日志、TensorBoard、结果、检查点和报告保留；旧报告中的命令是历史记录，
不再作为当前运行入口。原版TimeMixer、SeasonTrendMixer和公共训练流程未变。

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## 来源

移植自PyPOTS commit `53b3eac34be9491ac3f28e65ee1993436e9318af`。
`layers/pypots_timemixerpp/`保留BSD-3-Clause LICENSE和UPSTREAM.json。
多尺度均值输出与注意力路由是本地修改，不宣称与上游默认路径或论文完全等价。

## 2026-09-21清理验证

44项剩余仓库回归测试通过；两个保留模型的四预测长度脚本均通过Bash语法和真实CLI解析检查。
清理前后最佳路径的初始化权重、随机状态、训练输出、全部梯度逐元素一致；
历史最佳avg检查点严格加载通过，其预测也逐元素一致。本次未重新训练。
核验与可恢复源码备份位于 `tmp/timemixerpp_cleanup_20260921_102310/`。
