# 实验脚本索引

- 新增残差MA次数实验：[编排](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_residual_ma_experiment.py)、[训练](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_residual_ma.py)、[协议](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/PDMResidualMA_protocol_20261008.md)。仅比较入口一次/三次分解，旧脚本保留。

整理日期：2026-10-08。顶层Python脚本共78个；所有训练入口和共享实现保留，名称较旧不表示可以删除。

当前基线为一层OscillatorPDMTrendCondition。最近的学习率和非线性配对实验均已完成；时间对损失实验中断。具体运行状态以[总索引](/home/abc/PythonProject/Time-Series-Library-main/docs/experiments/README.md)为准。

## 当前和近期实验入口

| 实验 | 编排 | 训练或验证 |
| --- | --- | --- |
| learning_rate | [run_timemix_pdm_learning_rate_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_learning_rate_experiment.py) | [train_timemix_pdm_learning_rate.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_learning_rate.py) |
| nonlinear_state | [run_timemix_pdm_nonlinear_state_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_nonlinear_state_experiment.py) | [train_timemix_pdm_nonlinear_state.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_nonlinear_state.py) |
| position_pair | [run_timemix_pdm_position_pair_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_position_pair_experiment.py) | [train_timemix_pdm_position_pair.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_position_pair.py) |
| autocov_loss | [run_timemix_pdm_autocov_loss_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_autocov_loss_experiment.py) | [train_timemix_pdm_autocov_loss.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_autocov_loss.py) |
| innovation | [run_timemix_pdm_innovation_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_innovation_experiment.py) | [train_timemix_pdm_innovation.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_innovation.py) |
| decoder_study | [run_timemix_pdm_decoder_study_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_decoder_study_experiment.py) | [train_timemix_pdm_decoder_study.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_decoder_study.py) |

## 全部顶层Python脚本及静态引用

“被导入”仅列出顶层scripts间静态导入，不代表没有列出的脚本无人使用；测试、动态导入、协议和冻结清单也可能依赖它。

| 脚本 | 顶层scripts导入它的数量 |
| --- | --- |
| [diagnose_timemix_linear_reference.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/diagnose_timemix_linear_reference.py) | 14 |
| [evaluate_frozen_timemix.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/evaluate_frozen_timemix.py) | 30 |
| [pdm_bypass_freeze_study.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/pdm_bypass_freeze_study.py) | 1 |
| [pdm_kl_normalization_study.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/pdm_kl_normalization_study.py) | 1 |
| [pdm_trend_decay_study.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/pdm_trend_decay_study.py) | 20 |
| [run_directed_gaussian_study.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_directed_gaussian_study.py) | 0 |
| [run_predictable_autocov_study.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_predictable_autocov_study.py) | 0 |
| [run_timemix_autocorr_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_autocorr_experiment.py) | 0 |
| [run_timemix_blend_objective_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_blend_objective_experiment.py) | 0 |
| [run_timemix_combined_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_combined_experiment.py) | 2 |
| [run_timemix_feature_pdm_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_feature_pdm_experiment.py) | 0 |
| [run_timemix_gaussian_kl_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_gaussian_kl_experiment.py) | 0 |
| [run_timemix_joint_gaussian_kl_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_joint_gaussian_kl_experiment.py) | 0 |
| [run_timemix_joint_gaussian_kl_unweighted_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_joint_gaussian_kl_unweighted_experiment.py) | 0 |
| [run_timemix_low_lr_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_low_lr_experiment.py) | 0 |
| [run_timemix_pdm_autocorr_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_autocorr_experiment.py) | 0 |
| [run_timemix_pdm_autocov_loss_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_autocov_loss_experiment.py) | 0 |
| [run_timemix_pdm_bypass_freeze_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_bypass_freeze_experiment.py) | 0 |
| [run_timemix_pdm_coarse_ma_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_coarse_ma_experiment.py) | 20 |
| [run_timemix_pdm_decoder_study_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_decoder_study_experiment.py) | 0 |
| [run_timemix_pdm_innovation_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_innovation_experiment.py) | 0 |
| [run_timemix_pdm_kl_normalization_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_kl_normalization_experiment.py) | 0 |
| [run_timemix_pdm_learning_rate_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_learning_rate_experiment.py) | 0 |
| [run_timemix_pdm_mixing_depth_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_mixing_depth_experiment.py) | 0 |
| [run_timemix_pdm_no_history_pe_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_no_history_pe_experiment.py) | 0 |
| [run_timemix_pdm_nonlinear_state_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_nonlinear_state_experiment.py) | 0 |
| [run_timemix_pdm_position_pair_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_position_pair_experiment.py) | 0 |
| [run_timemix_pdm_readout_split_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_readout_split_experiment.py) | 0 |
| [run_timemix_pdm_scale_allocation_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_scale_allocation_experiment.py) | 0 |
| [run_timemix_pdm_scaled_ma_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_scaled_ma_experiment.py) | 0 |
| [run_timemix_pdm_seasonal_head_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_seasonal_head_experiment.py) | 0 |
| [run_timemix_pdm_segment_pe_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_segment_pe_experiment.py) | 0 |
| [run_timemix_pdm_single_fft_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_single_fft_experiment.py) | 0 |
| [run_timemix_pdm_trend_bypass_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_trend_bypass_experiment.py) | 0 |
| [run_timemix_pdm_trend_condition_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_trend_condition_experiment.py) | 0 |
| [run_timemix_pdm_trend_decay_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_trend_decay_experiment.py) | 0 |
| [run_timemix_pdm_trend_prelinear_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_trend_prelinear_experiment.py) | 0 |
| [run_timemix_pdm_trend_smooth_residual_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_pdm_trend_smooth_residual_experiment.py) | 0 |
| [run_timemix_recurrent_rank16_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_recurrent_rank16_experiment.py) | 1 |
| [run_timemix_season_norm_experiment.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/run_timemix_season_norm_experiment.py) | 0 |
| [smoke_season_trend_mixer.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/smoke_season_trend_mixer.py) | 35 |
| [train_pdm_directed_gaussian_full.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_pdm_directed_gaussian_full.py) | 0 |
| [train_timemix_autocorr.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_autocorr.py) | 1 |
| [train_timemix_blend_objective.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_blend_objective.py) | 1 |
| [train_timemix_combined.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_combined.py) | 1 |
| [train_timemix_feature_pdm.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_feature_pdm.py) | 2 |
| [train_timemix_gaussian_kl.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_gaussian_kl.py) | 1 |
| [train_timemix_joint_gaussian_kl.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_joint_gaussian_kl.py) | 1 |
| [train_timemix_joint_gaussian_kl_unweighted.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_joint_gaussian_kl_unweighted.py) | 1 |
| [train_timemix_local_drive.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_local_drive.py) | 0 |
| [train_timemix_low_lr.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_low_lr.py) | 1 |
| [train_timemix_pdm_autocorr.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_autocorr.py) | 1 |
| [train_timemix_pdm_autocov_loss.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_autocov_loss.py) | 1 |
| [train_timemix_pdm_bypass_freeze.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_bypass_freeze.py) | 1 |
| [train_timemix_pdm_coarse_ma.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_coarse_ma.py) | 1 |
| [train_timemix_pdm_decoder_study.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_decoder_study.py) | 1 |
| [train_timemix_pdm_innovation.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_innovation.py) | 1 |
| [train_timemix_pdm_kl_normalization.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_kl_normalization.py) | 1 |
| [train_timemix_pdm_learning_rate.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_learning_rate.py) | 2 |
| [train_timemix_pdm_mixing_depth.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_mixing_depth.py) | 7 |
| [train_timemix_pdm_no_history_pe.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_no_history_pe.py) | 2 |
| [train_timemix_pdm_nonlinear_state.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_nonlinear_state.py) | 1 |
| [train_timemix_pdm_position_pair.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_position_pair.py) | 1 |
| [train_timemix_pdm_readout_split.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_readout_split.py) | 5 |
| [train_timemix_pdm_scale_allocation.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_scale_allocation.py) | 1 |
| [train_timemix_pdm_scaled_ma.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_scaled_ma.py) | 1 |
| [train_timemix_pdm_seasonal_head.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_seasonal_head.py) | 1 |
| [train_timemix_pdm_segment_pe.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_segment_pe.py) | 2 |
| [train_timemix_pdm_single_fft.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_single_fft.py) | 2 |
| [train_timemix_pdm_trend_bypass.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_trend_bypass.py) | 12 |
| [train_timemix_pdm_trend_condition.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_trend_condition.py) | 2 |
| [train_timemix_pdm_trend_decay.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_trend_decay.py) | 1 |
| [train_timemix_pdm_trend_prelinear.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_trend_prelinear.py) | 3 |
| [train_timemix_pdm_trend_smooth_residual.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_pdm_trend_smooth_residual.py) | 2 |
| [train_timemix_recurrent_rank16.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_recurrent_rank16.py) | 1 |
| [train_timemix_regularization.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_regularization.py) | 0 |
| [train_timemix_season_norm.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/train_timemix_season_norm.py) | 1 |
| [validate_pdm_nonlinear_state.py](/home/abc/PythonProject/Time-Series-Library-main/scripts/validate_pdm_nonlinear_state.py) | 0 |

上游按任务和数据集组织的子目录保留原样。早期已归档入口见experiments/cleanup_20260929*和cleanup_20260930*的归档与清单；本次不重新删除或恢复它们。

本次仅删除tmp内10个已结束实验的一次性启动器；[可恢复归档和清单](/home/abc/PythonProject/Time-Series-Library-main/experiments/organization_20261008/README.md)。
