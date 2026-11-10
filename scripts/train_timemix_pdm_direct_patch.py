"""MA1 One recurrent state to four output values paired MSE/covariance training."""
import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data_provider.data_loader import Dataset_ETT_minute
from models.OscillatorPDMScaledMA import Model as ScaledModel, scaled_odd_kernel
from models.OscillatorPDMReadoutSplit import Model as ReadoutSplit
from scripts.train_timemix_pdm_mixing_depth import initialize_pair as old_initialize_pair
from models.OscillatorPDMSegmentPE import Model as SegmentPE
from models.OscillatorPDMDirectPatch import Model as TrendCondition
from utils.patch_covariance_loss import parts as covariance_parts
from utils.timemix_decay_study import parameter_groups as decay_groups, sample_response_std
from utils.retained_timemix_helpers import summarize_tensor
from utils.pdm_residual_ma_diagnostics import setup_hooks, trace
from scripts.smoke_season_trend_mixer import framework_args
from utils.retained_timemix_helpers import metrics
from utils.logger import experiment_logging, start_experiment, log_metrics
from scripts.pdm_trend_decay_study import gradient_audit
from utils.pdm_residual_ma_diagnostics import SelectedValidationCapture



BANDWIDTH = 1.0
PATCH_LENGTH = 4
SHARED_WEIGHT = 1.0
WEIGHTS = {'mse': 0., 'patch_covariance': SHARED_WEIGHT}
ARMS = tuple(WEIGHTS)
DEPTHS = {name: 1 for name in ARMS}
MODEL_NAMES = {name: 'OscillatorPDMDirectPatch' for name in ARMS}
PASSES = {name: 1 for name in ARMS}
TREND_BYPASS = {name: False for name in ARMS}
VARIANTS = {name: TrendCondition for name in ARMS}

def arm_config(cfg, arm):
    return SimpleNamespace(**dict(vars(cfg), model=MODEL_NAMES[arm], e_layers=1, learning_rate=LEARNING_RATES[arm], residual_decomposition_passes=PASSES[arm]))
LEARNING_RATES = {name: .001 for name in ARMS}

def kernel_audit(cfg):
    lengths = [cfg.seq_len // cfg.down_sampling_window**i for i in range(cfg.down_sampling_layers+1)]
    return {arm: {length: scaled_odd_kernel(cfg.moving_avg, cfg.seq_len, length)
                  for length in lengths} for arm in ARMS}


def learning_rate(arm, completed_epochs, total_epochs):
    return LEARNING_RATES[arm] * .5 * (1 + math.cos(math.pi * completed_epochs / total_epochs))

def selection_score(validation):
    return validation['neural_mse']


def parameter_groups(model, arm):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    return decay_groups(model, 'adam_norm_bias')


def build_optimizer(groups, arm):
    return torch.optim.Adam(groups, lr=LEARNING_RATES[arm])


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset(cfg, flag, limit):
    if flag not in ('train', 'val'):
        raise ValueError('This study never opens the test split')
    data = Dataset_ETT_minute(cfg, cfg.root_path, flag=flag, size=[96, 0, 96],
                             features='M', data_path='ETTm1.csv', timeenc=1, freq='h')
    if limit:
        ids = np.unique(np.linspace(0, len(data)-1, min(limit, len(data)), dtype=int)).tolist()
        return Subset(data, ids)
    return data


def loss_configuration(opts):
    return dict(objective='pdm_direct_patch_v1',weights=WEIGHTS,patch_length=PATCH_LENGTH,
        stride=PATCH_LENGTH,lag=0,bandwidth=BANDWIDTH,centering='J=I-11/N',denominator='N^2',
        relation='HS squared distance of centered RKHS covariance operators on 24 disjoint length4 raw-value patches; standard RBF sum of coordinate squares; self+self-2cross',
        target_detached=True,history_unused=True,auxiliary_parameters=0,no_added_window_standardization=True,
        calibration='None. Fixed lambda1 and sigma1; no tuning using train/val/test.')


def objective(prediction,target,history,arm,config):
    if arm not in ARMS and arm!='diagnostic':raise ValueError('Unknown arm')
    weight=config['weights'].get(arm,0.)
    value,diagnostics=covariance_parts(prediction,target,config['bandwidth'],config['patch_length'])
    mse=(prediction-target.detach()).square().mean()
    loss=mse if weight==0 else mse+weight*value
    diagnostics.update(lambda_weight=weight,weighted_relation=float(weight*value.detach()),
        weighted_relation_to_mse=float(weight*value.detach()/mse.detach().clamp_min(1e-30)))
    return dict(loss=loss,mse=mse,relation=value.detach(),relation_for_gradient=value,relation_diagnostics=diagnostics)


def gradient_comparison(model, parts, weight):
    """Same-forward separate gradients; never changes .grad or model RNG."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    gm = torch.autograd.grad(parts['mse'], [p for _, p in named], retain_graph=True, allow_unused=True)
    gd = torch.autograd.grad(parts['relation_for_gradient'], [p for _, p in named], retain_graph=True, allow_unused=True)
    groups = {'all': list(range(len(named))), 'fine_trend_head': [i for i, (n, _) in enumerate(named) if n.startswith('feature_trend_heads.0.')]}
    groups['state_patch_heads']=[i for i,(n,_) in enumerate(named) if '.readout.output_projection.' in n]
    groups['direct_temporal_projection']=[i for i,(n,_) in enumerate(named) if '.temporal_projection.' in n]
    result = {}
    for group, ids in groups.items():
        mm = dd = dot = 0.
        for i in ids:
            a, b = gm[i], gd[i]
            if a is not None:
                if not torch.isfinite(a).all(): raise RuntimeError('Nonfinite MSE component gradient')
                mm += float(a.detach().double().square().sum())
            if b is not None:
                if not torch.isfinite(b).all(): raise RuntimeError('Nonfinite patch-covariance component gradient')
                dd += float(b.detach().double().square().sum())
            if a is not None and b is not None: dot += float((a.detach().double()*b.detach().double()).sum())
        result[group] = dict(mse_norm=math.sqrt(mm), auxiliary_norm=math.sqrt(dd), weighted_auxiliary_norm=weight*math.sqrt(dd),
            weighted_ratio=weight*math.sqrt(dd)/max(math.sqrt(mm), 1e-30),
            cosine=dot/max(math.sqrt(mm*dd), 1e-30), dot=dot, parameter_tensors=len(ids))
    return result


def training_losses(neural, target, arm, config, history):
    parts = objective(neural,target,history,arm,config)
    parts['auxiliary_relation'] = neural.new_zeros(())
    return parts


RELATION_KEYS=('patch_covariance','prediction_covariance_norm2','target_covariance_norm2','cross_covariance_inner_product')


def evaluate_validation(model,loader,device,loss_config):
    model.eval();squared=absolute=count=windows=0.;sums={k:0. for k in RELATION_KEYS}
    with torch.no_grad():
        for x,y,_,_ in loader:
            x,target=x.float().to(device),y[:,-96:].float().to(device)
            prediction=model(x)
            if prediction.shape!=target.shape or not torch.isfinite(prediction).all():raise RuntimeError('Invalid validation prediction')
            e=prediction.double()-target.double();squared+=float(e.square().sum());absolute+=float(e.abs().sum());count+=target.numel();windows+=len(x)
            _,diag=covariance_parts(prediction,target,loss_config['bandwidth'],loss_config['patch_length'])
            for key in sums:sums[key]+=diag[key]*len(x)
    values={k:v/windows for k,v in sums.items()}
    return dict(neural_mse=squared/count,neural_mae=absolute/count,relation=values['patch_covariance'],**values)


def evaluate(model,loader,device,retain=False,loss_config=None):
    model.eval();predictions=[];targets=[];histories=[];sums={k:0. for k in RELATION_KEYS}
    with torch.no_grad():
        for x,y,_,_ in loader:
            x,target=x.float().to(device),y[:,-96:].float().to(device)
            prediction=model(x)
            if not torch.isfinite(prediction).all():raise RuntimeError('Invalid selected prediction')
            predictions.append(prediction.cpu().numpy());targets.append(target.cpu().numpy());histories.append(x.cpu().numpy())
            _,diag=covariance_parts(prediction,target,loss_config['bandwidth'],loss_config['patch_length'])
            for key in sums:sums[key]+=diag[key]*len(x)
    p,y,x=np.concatenate(predictions),np.concatenate(targets),np.concatenate(histories)
    result=metrics(p.astype('float64'),y.astype('float64'));result.update({k:v/len(p) for k,v in sums.items()})
    result['low4_mse']=result['mse']*result['low_frequency_error_share']['4']
    result['per_channel_mse']=np.mean((p.astype('float64')-y)**2,axis=(0,1)).tolist()
    return result,p,y,x


def initialize_pair(cfg):
    pointwise=old_initialize_pair(cfg)['depth1']
    with torch.random.fork_rng(devices=[]):control=TrendCondition(arm_config(cfg,ARMS[0]))
    control.load_pointwise_initial_state(pointwise.state_dict())
    result={ARMS[0]:control}
    for name in ARMS[1:]:
        with torch.random.fork_rng(devices=[]):
            model=TrendCondition(arm_config(cfg,name))
        model.load_state_dict(control.state_dict(),strict=True)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in control.state_dict().items())
        result[name]=model
    return result


def joint_stale_count(stale, improved):
    """Stop all arms only after patience consecutive epochs improving no arm."""
    return 0 if improved else stale + 1


def publish_report(out, report):
    atomic_json(out/'report.json', report)
    lines = ['# MA1直接24特征片段读出两组完整对照', '', '状态：'+report['status'], '',
             '| 种子 | 分组 | 所选轮次 | 验证均方误差 |', '| --- | --- | --- | --- |']
    names = {name:name for name in ARMS}
    for row in report['runs']:
        lines.append(f"| {row['seed']} | {names[row['variant']]} | {row['best_epoch']} | {row['validation_mse']:.8f} |")
    lines += ['', '两组直接24特征MA1同初值，学习率均.001；纯MSE与MSE加不重叠原值片段核协方差；固定lambda1，不做校准；按完整验证MSE选轮。']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.5,
                        help='CUDA allocator fraction for new runs, in (0, 1]; resume uses saved settings')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2021])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--train-limit', type=int, default=0, help='0 means all windows; nonzero is smoke only')
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--skip-test', action='store_true', default=True, help='Always validation-only; retained for smoke compatibility')
    parser.add_argument('--stop-after-epochs', type=int, default=0, help='Controlled checkpoint/resume verification only')
    
    parser.add_argument('--log-every', type=int, default=50)
    opts = parser.parse_args()
    if not math.isfinite(opts.gpu_memory_fraction) or not 0 < opts.gpu_memory_fraction <= 1:
        parser.error('--gpu-memory-fraction must be finite and in (0, 1]')
    invocation_stop = opts.stop_after_epochs
    if opts.resume:
        out = opts.resume.resolve()
        protocol = json.loads((out/'protocol.json').read_text())
        if protocol.get('loss_config', {}).get('objective') != 'pdm_direct_patch_v1':
            raise ValueError('Cannot resume a different architecture experiment')
        opts = SimpleNamespace(**protocol['options'])
        cfg = SimpleNamespace(**protocol['model_args'])
        for name, digest in protocol['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Source changed since launch; refusing ambiguous resume: ' + name)
    else:
        if min(opts.epochs, opts.patience) < 1 or min(opts.train_limit, opts.val_limit) < 0:
            raise ValueError('Invalid budget')
        if not opts.skip_test and (opts.train_limit or opts.val_limit):
            raise ValueError('Limited smoke runs must use --skip-test')
        if opts.log_every < 1:
            raise ValueError('Invalid logging interval')
        out = (opts.output or ROOT/'tmp'/('lag_study_timemix_pdm_direct_patch_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
        out.mkdir(parents=True, exist_ok=False)
        cfg = framework_args(['--task_name','long_term_forecast','--is_training','1',
            '--model','OscillatorPDMDirectPatch','--model_id',out.name,
            '--data','ETTm1','--root_path',str(ROOT/'dataset/ETT-small'),'--data_path','ETTm1.csv',
            '--features','M','--freq','h','--seq_len','96','--label_len','0','--pred_len','96',
            '--enc_in','7','--dec_in','7','--c_out','7','--moving_avg','25',
            '--d_model','32','--d_ff','64','--e_layers','1','--top_k','5','--dropout','.2',
            '--down_sampling_layers','3','--down_sampling_window','2','--down_sampling_method','avg',
            '--channel_independence','1','--use_norm','1','--ssm_modes','8','--ssm_dynamics_weight_decay','0',
            '--batch_size','32','--num_workers','0','--optimizer','adam','--learning_rate','.001',
            '--weight_decay','.0001','--lradj','cosine','--train_epochs',str(opts.epochs),
            '--patience',str(opts.patience),'--log_dir',str(out/'logs'),'--tensorboard_dir',str(out/'runs')])
        cfg.seasonal_normalize = False
        cfg.seed = opts.seeds[0]
        cfg.use_gpu = opts.device == 'cuda'
        # Expose only the byte-identical train/validation prefix to Dataset_ETT_minute.
        import itertools
        prefix = out/'train_val_data'
        prefix.mkdir()
        with (ROOT/'dataset/ETT-small/ETTm1.csv').open('rb') as src, (prefix/'ETTm1.csv').open('xb') as dst:
            lines = list(itertools.islice(src, 46081))
            assert len(lines) == 46081
            dst.writelines(lines)
        cfg.root_path = str(prefix)
        opts.train_val_csv_sha256 = hashlib.sha256((prefix/'ETTm1.csv').read_bytes()).hexdigest()
        files = [Path(__file__).relative_to(ROOT), Path('run.py')]
        for directory in ['models','layers','data_provider','utils']:
            files.extend(sorted(Path(directory).rglob('*.py')))
        files.extend([Path('tests/test_pdm_coarse_ma.py'), Path('scripts/train_timemix_pdm_trend_bypass.py'), Path('tests/test_pdm_trend_bypass.py'), Path('scripts/train_timemix_feature_pdm.py'), Path('tests/test_feature_pdm_study.py'), Path('docs/experiments/OscillatorPDMCoarseMA_protocol_20261002.md'), Path('scripts/smoke_season_trend_mixer.py'), Path('scripts/run_timemix_pdm_coarse_ma_experiment.py'), Path('scripts/run_timemix_pdm_trend_bypass_experiment.py'), Path('scripts/train_timemix_joint_gaussian_kl_unweighted.py'), Path('scripts/run_timemix_joint_gaussian_kl_unweighted_experiment.py'), Path('utils/season_norm_study_metrics.py'), Path('scripts/diagnose_timemix_linear_reference.py'), Path('scripts/evaluate_frozen_timemix.py')])
        files.extend([Path('models/OscillatorPDMScaleAllocation.py'),Path('scripts/train_timemix_pdm_scale_allocation.py'),Path('scripts/run_timemix_pdm_scale_allocation_experiment.py'),Path('tests/test_pdm_scale_allocation.py'),Path('docs/experiments/OscillatorPDMScaleAllocation_protocol_20261001.md')])
        files.extend([Path('scripts/train_timemix_pdm_coarse_ma.py'), Path('scripts/pdm_trend_decay_study.py'), Path('scripts/run_timemix_pdm_trend_decay_experiment.py'), Path('tests/test_pdm_trend_decay.py'), Path('docs/experiments/OscillatorPDMTrendDecay_protocol_20261002.md')])
        files.extend([Path('scripts/run_timemix_pdm_scaled_ma_experiment.py'), Path('tests/test_pdm_scaled_ma.py'), Path('tests/test_pdm_scaled_ma_study.py'), Path('docs/experiments/OscillatorPDMScaledMA_protocol_20261003.md')])
        files.extend([Path('scripts/run_timemix_pdm_seasonal_head_experiment.py'), Path('tests/test_pdm_seasonal_head.py'), Path('tests/test_pdm_seasonal_head_study.py'), Path('docs/experiments/OscillatorPDMSeasonalHead_protocol_20261003.md')])
        files.extend([Path('scripts/run_timemix_pdm_readout_split_experiment.py'), Path('tests/test_pdm_readout_split.py'), Path('docs/experiments/OscillatorPDMReadoutSplit_protocol_20261004.md')])
        files.extend([Path('scripts/train_timemix_pdm_readout_split.py'), Path('scripts/run_timemix_pdm_single_fft_experiment.py'), Path('tests/test_pdm_seasonal_fft.py'), Path('tests/test_pdm_single_fft_study.py'), Path('docs/experiments/OscillatorPDMSingleFFT_protocol_20261005.md')])
        files.extend([Path('scripts/train_timemix_pdm_single_fft.py'), Path('scripts/run_timemix_pdm_segment_pe_experiment.py'), Path('tests/test_pdm_segment_pe.py'), Path('docs/experiments/OscillatorPDMNoHistoryPE_protocol_20261005.md')])
        files.extend([Path('scripts/train_timemix_pdm_no_history_pe.py'), Path('scripts/run_timemix_pdm_no_history_pe_experiment.py'), Path('tests/test_pdm_no_history_pe.py'), Path('docs/experiments/OscillatorPDMNoHistoryPE_protocol_20261005.md'), Path('tests/test_pdm_segment_pe_study.py'), Path('docs/experiments/OscillatorPDMSegmentPE_protocol_20261005.md')])
        files.extend([Path('scripts/train_timemix_pdm_segment_pe.py'), Path('scripts/run_timemix_pdm_trend_condition_experiment.py'), Path('tests/test_pdm_trend_condition.py'), Path('docs/experiments/OscillatorPDMTrendCondition_protocol_20261006.md')])
        files.extend([Path('scripts/train_timemix_pdm_trend_condition.py'), Path('scripts/run_timemix_pdm_mixing_depth_experiment.py'), Path('tests/test_pdm_mixing_depth.py'), Path('docs/experiments/OscillatorPDMMixingDepth_protocol_20261006.md')])
        files.extend([Path('scripts/train_timemix_pdm_mixing_depth.py'), Path('scripts/run_timemix_pdm_decoder_study_experiment.py'), Path('tests/test_pdm_decoder_study.py'), Path('docs/experiments/OscillatorPDMDecoderStudy_protocol_20261006.md')])
        files.extend([Path('scripts/run_timemix_pdm_target_nll_experiment.py'), Path('tests/test_pdm_target_nll.py'), Path('tests/test_label_covariance_nll.py'), Path('docs/experiments/PDMTargetNLL_protocol_20261009.md')])
        files.extend([Path('scripts/run_timemix_pdm_bayesian_relation_experiment.py'),Path('tests/test_pdm_bayesian_relation.py'),Path('docs/experiments/PDMBayesianRelation_protocol_20261009.md')])
        files.extend([Path('scripts/train_timemix_pdm_bayesian_relation.py'), Path('scripts/run_timemix_pdm_kernel_autocovariance_experiment.py'), Path('scripts/audit_pdm_kernel_autocovariance.py'), Path('tests/test_pdm_kernel_autocovariance.py'), Path('docs/experiments/PDMKernelAutocovariance_protocol_20261010.md')])
        files.extend(map(Path, ['scripts/run_timemix_pdm_kernel_weight_experiment.py', 'scripts/train_timemix_pdm_kernel_autocovariance.py', 'tests/test_pdm_kernel_weight.py', 'docs/experiments/PDMKernelWeight_protocol_20261010.md']))
        files.extend(map(Path, ['scripts/run_timemix_pdm_direct_patch_experiment.py','scripts/audit_pdm_patch_covariance.py','tests/test_pdm_direct_patch.py','docs/experiments/PDMDirectPatch_protocol_20261012.json']))
        files.extend(map(Path, json.loads((ROOT/'tmp/lag_study_timemix_pdm_increment_patch_20261010_seed2021_earlystop5/training/protocol.json').read_text())['source_sha256']))
        files.extend(map(Path, json.loads((ROOT/'tmp/lag_study_timemix_pdm_raw_patch_20261011_seed2021_earlystop5/training/protocol.json').read_text())['source_sha256']))
        files.extend(map(Path, json.loads((ROOT/'tmp/lag_study_timemix_pdm_patch_covariance_20261011_seed2021_earlystop5/training/protocol.json').read_text())['source_sha256']))
        files.extend(map(Path, json.loads((ROOT/'tmp/lag_study_timemix_pdm_joint_patch_20261011_seed2021_earlystop5/training/protocol.json').read_text())['source_sha256']))
        files.extend(map(Path, json.loads((ROOT/'tmp/lag_study_timemix_pdm_state_patch_20261011_seed2021_earlystop5/training/protocol.json').read_text())['source_sha256']))
        files.append(Path('tests/test_pdm_direct_patch_study.py'))
        files.extend(map(Path, json.loads((ROOT/'tmp/lag_study_timemix_pdm_state_patch_concat_20261012_seed2021_earlystop5/training/protocol.json').read_text())['source_sha256']))
        files = list(dict.fromkeys(files))
        hashes = {}
        for relative in files:
            destination = out/'source'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, destination)
            hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
        options = {k:(str(v) if isinstance(v, Path) else v) for k,v in vars(opts).items()}
        protocol = dict(options=options, model_args=vars(cfg), source_sha256=hashes,
            preprocessing=dict(raw_trend_bypass_by_arm=TREND_BYPASS, seasonal_normalize=False,
                scale_variants=list(ARMS), kernels_by_arm=kernel_audit(cfg), pdm='Direct history-to-24 patch features at each seasonal scale; no 96-position future feature tensor or runtime aggregation. Final temporal projection rows and fixed PE are folded from old scratch four-row means at initialization only. Original A24 and state16->32->4, trend96 and history unchanged. 129058 parameters; common tensors exact; initial mean-model function within tolerance, not bitwise. Same representable linear driver family as fixed mean, different parameterization and optimizer trajectory; not extra retained position information.', mixing_depths=DEPTHS,
                pdm_trend_bypass='Neither arm; eight components each'),
            initialization='新两组参数buffer及初始函数同scratch初值；4输出层复制旧scratch初始行，新投影为[I/4]*4；与旧均值聚合共有参数精确相同，初始函数仅数值容差内相同；训练后路由可能分化。',
            factor='MSE versus fixed-weight disjoint patch-covariance lag-zero RKHS covariance distance; same new single-state patch MA1 and normalization', loss_config=loss_configuration(opts),
            selection='各自按完整纯网络验证MSE严格下降选最佳轮，第0轮不参与。',
            early_stopping=dict(kind='joint', patience=opts.patience, min_delta=0.0,
                max_epochs=opts.epochs, rule='任一组验证MSE严格改善则共同stale清零；否则加1；达到patience两组共同停；各自选最佳。'),
            pairing='相同批次和每步随机种子；FP32，Adam初始学习率均.001，相同cosine形状与wd=.0001；FFT路由分化后不保证全部dropout掩码相同。两组无预训练。', learning_rates=LEARNING_RATES, residual_decomposition_passes=PASSES,
            test='训练只访问train/val；编排待两组验证选轮冻结后自动各运行一次完整test。',
            caveat='验证区间已用于历史研究，不是新外部数据集。',
            torch_version=torch.__version__, created=datetime.now().isoformat())
        atomic_json(out/'protocol.json', protocol)
    assert hashlib.sha256((Path(cfg.root_path)/'ETTm1.csv').read_bytes()).hexdigest() == opts.train_val_csv_sha256
    loss_config = loss_configuration(opts)
    torch.set_num_threads(2)
    if opts.device == 'cuda':
        # Legacy protocols used a fixed 30% cap; never silently rewrite their settings.
        torch.cuda.set_per_process_memory_fraction(getattr(opts, 'gpu_memory_fraction', .30))
    lock = (out/'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    report = json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else dict(status='training', runs=[])
    if report['status'] == 'completed':
        print('ALREADY COMPLETED', out)
        lock.close()
        return
    train = dataset(cfg, 'train', opts.train_limit)
    val = dataset(cfg, 'val', opts.val_limit)
    val_loader = DataLoader(val, batch_size=32, shuffle=False, num_workers=0)
    report.pop('error', None)
    report.update(status='training', train_windows=len(train), validation_windows=len(val), protocol=str(out/'protocol.json'))
    publish_report(out, report)
    try:
        with experiment_logging(cfg) as logger:
            start_experiment(cfg)
            logger.info('SCRATCH study=%s all_train=%d all_val=%d seeds=%s no pretrained weights; no test during training', out, len(train), len(val), opts.seeds)
            for seed in opts.seeds:
                if len([v for v in report['runs'] if v['seed']==seed]) == len(VARIANTS):
                    continue
                folder = out/str(seed)
                folder.mkdir(exist_ok=True)
                seed_all(seed)
                models = initialize_pair(cfg)
                if not (folder/'initialization_audit.json').exists():
                    atomic_json(folder/'initialization_audit.json', dict(
                    common_parameters_and_buffers_equal=True, kernels_by_arm=kernel_audit(cfg),
                    all_state_keys_equal=True, initial_predictions_expected_equal=True, residual_decomposition_passes=PASSES,
                    parameter_counts={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},
                    initialization="MA1两组完整参数buffer和初始预测相同；仅训练目标不同"))
                if not (folder/'initial.pth').exists():
                    atomic_torch(folder/'initial.pth', {name: model.state_dict() for name, model in models.items()})
                models = {name:model.to(opts.device) for name,model in models.items()}
                groups = {name:parameter_groups(model,name) for name,model in models.items()}
                optimizers = {name:build_optimizer(group,name) for name,group in groups.items()}
                audit = {}
                decay_by_id = {}
                for name,model in models.items():
                    decay_by_id[name] = {id(p):g['weight_decay'] for g in groups[name] for p in g['params']}
                    audit[name] = {key:decay_by_id[name][id(p)] for key,p in model.named_parameters()}
                assert all(audit[ARMS[0]][k] == v for k,v in audit[ARMS[1]].items() if k in audit[ARMS[0]]), 'Common optimizer policy mismatch'
                assert all(audit[ARMS[0]] == audit[name] for name in ARMS), 'Optimizer policies differ'
                for name, model in models.items():
                    ids = [id(v) for group in groups[name] for v in group['params']]
                    assert len(ids) == len(set(ids)) and set(ids) == {id(v) for v in model.parameters()}
                atomic_json(folder/'optimizer_groups.json',audit)
                logger.info('STATE PATCH READOUT STUDY: identical initial tensors/functions; MA1; MSE versus MSE+patch_covariance; LR .001; loss_config=%s; optimizer audit=%s', loss_config, folder/'optimizer_groups.json')
                probe_ids=np.unique(np.linspace(0,len(train)-1,min(16,len(train)),dtype=int))
                probe_x=torch.tensor(np.stack([train[int(i)][0] for i in probe_ids]),dtype=torch.float32,device=opts.device)
                branch_traces = json.loads((folder/'branch_traces.json').read_text()) if (folder/'branch_traces.json').exists() else []
                best_states = {}
                completed, stale, best, best_epoch, curves = 0, 0, {k:float('inf') for k in models}, {}, []
                if (folder/'last.pt').exists():
                    last = torch.load(folder/'last.pt', map_location=opts.device, weights_only=True)
                    if last['loss_config'] != loss_config:
                        raise RuntimeError('Resume loss configuration differs')
                    if last['completed_updates'] != last['epoch'] * math.ceil(len(train)/32):
                        raise RuntimeError('Resume update count differs')
                    for name,model in models.items():
                        model.load_state_dict(last['models'][name], strict=True)
                        optimizers[name].load_state_dict(last['optimizers'][name])
                    completed,stale,best,best_epoch,curves = (last[k] for k in ['epoch','stale','best','best_epoch','curves'])
                    best_states = last['best_models']
                    branch_traces = [r for r in branch_traces if r['epoch'] <= completed]
                    for name, saved in best_states.items():
                        atomic_torch(folder/(name+'_best.pth'), saved)
                    logger.info('RESUME seed=%d completed_epoch=%d joint_stale=%d',seed,completed,stale)
                seed_all(seed)
                with torch.no_grad():
                    if completed==0:
                        x=train[0][0]
                        x=torch.as_tensor(x).float()[None].to(opts.device)
                        for model in models.values(): model.eval()
                        audit = json.loads((folder/'initialization_audit.json').read_text())
                        for mode in ('eval', 'train'):
                            values, states = [], []
                            for model in models.values():
                                model.train(mode == 'train')
                                seed_all(seed)
                                values.append(model(probe_x))
                                states.append(torch.cuda.get_rng_state() if opts.device=='cuda' else torch.get_rng_state())
                            audit[mode+'_max_prediction_difference'] = {name:float((values[0]-v).abs().max()) for name,v in zip(ARMS,values)}
                            assert all(v.shape == values[0].shape and torch.isfinite(v).all() for v in values), 'Invalid initial predictions'
                            audit[mode+'_all_rng_equal'] = all(torch.equal(states[0], state) for state in states)
                            assert all(torch.equal(values[0], value) for value in values)
                            assert audit[mode+'_all_rng_equal']
                        atomic_json(folder/'initialization_audit.json', audit)
                for epoch in range(completed+1, opts.epochs+1):
                    if stale >= opts.patience:
                        break
                    epoch_seed=seed*1000+epoch
                    order=torch.randperm(len(train),generator=torch.Generator().manual_seed(epoch_seed)).tolist()
                    loader=DataLoader(Subset(train,order),batch_size=32,shuffle=False,num_workers=0,drop_last=False)
                    losses={k:0. for k in models}
                    neural_losses={k:0. for k in models}
                    relation_losses={k:0. for k in models}
                    auxiliary_relation_losses={k:0. for k in models}
                    for model in models.values():model.train()
                    start=time.monotonic()
                    for step,(x,y,_,_) in enumerate(loader):
                        x,y=x.float().to(opts.device),y[:,-96:].float().to(opts.device)
                        for name,model in models.items():
                            seed_all(seed*1000000+epoch*10000+step)
                            optimizer=optimizers[name]
                            optimizer.zero_grad(set_to_none=True)
                            record_branch = step == 0 or (step+1) % 200 == 0
                            captured,handles = setup_hooks(model) if record_branch else ({},[])
                            try:
                                loss_parts=training_losses(model(x),y,name,loss_config,x)
                                loss,neural_loss=loss_parts['loss'],loss_parts['mse']
                            finally:
                                for handle in handles: handle.remove()
                            if not all(torch.isfinite(loss_parts[key]).all() for key in ('loss','mse','relation','auxiliary_relation')):
                                raise RuntimeError('Nonfinite MSE/patch-covariance distance/total loss')
                            comparison = gradient_comparison(model, loss_parts, WEIGHTS[name]) if record_branch else None
                            loss.backward()
                            grads=[p.grad for p in model.parameters() if p.requires_grad]
                            if any(g is None for g in grads):raise RuntimeError('Missing gradient')
                            if not torch.stack([torch.isfinite(g).all() for g in grads]).all():raise RuntimeError('Nonfinite gradient')
                            if record_branch:
                                branch_row=trace(model,captured,step+1,loss,decay_by_id[name])
                                branch_row['gradient_comparison'] = comparison
                                branch_row['relation_diagnostics'] = loss_parts['relation_diagnostics']
                                branch_row['state_head_norms']={str(i):dict(weight=float(e.decoder.readout.output_projection.weight.detach().norm()),bias=float(e.decoder.readout.output_projection.bias.detach().norm()),gradient=float(e.decoder.readout.output_projection.weight.grad.detach().norm())) for i,e in enumerate(model.seasonals)}
                                branch_row['direct_projection_norms']={str(i):{n:dict(norm=float(p.detach().norm()),gradient=float(p.grad.detach().norm())) for n,p in e.temporal_projection.named_parameters()} for i,e in enumerate(model.seasonals)}
                                branch_row.update(epoch=epoch,variant=name,trend_gradient_audit=gradient_audit(model,decay_by_id[name]))
                                branch_row['trend_condition_parameters'] = {n: dict(norm=float(p.detach().double().norm()), gradient_norm=float(p.grad.double().norm()), weight_decay=decay_by_id[name][id(p)]) for n,p in model.named_parameters() if n.endswith('.trend_condition.weight')}
                                branch_traces.append(branch_row)
                                atomic_json(folder/'branch_traces.json',branch_traces)
                                log_metrics({'coarse_output_std':branch_row['scales'][3]['activations']['output']['std']},
                                            step=(epoch-1)*len(loader)+step+1,prefix=f'branches/{name}/seed{seed}')
                            optimizer.step()
                            captured.clear()
                            losses[name]+=float(loss.detach())*len(x)
                            neural_losses[name]+=float(neural_loss.detach())*len(x)
                            relation_losses[name]+=float(loss_parts['relation'])*len(x)
                            auxiliary_relation_losses[name]+=float(loss_parts['auxiliary_relation'])*len(x)
                            if step % opts.log_every == 0 or step+1 == len(loader):
                                logger.info('BATCH seed=%d arm=%s epoch=%d/%d step=%d/%d mse=%.8f relation=%.8f auxiliary_relation=%.8f total=%.8f lr=%.8f', seed,name,epoch,opts.epochs,step+1,len(loader),float(neural_loss.detach()),float(loss_parts['relation']),float(loss_parts['auxiliary_relation']),float(loss.detach()),optimizer.param_groups[0]['lr'])
                        if step%opts.log_every==0 or step+1==len(loader):
                            atomic_json(out/'status.json',dict(status='training',seed=seed,epoch=epoch,step=step+1,steps=len(loader),updated=datetime.now().isoformat()))
                            logger.info('PROGRESS seed=%d epoch=%d/%d paired_step=%d/%d',seed,epoch,opts.epochs,step+1,len(loader))
                    improved=False
                    atomic_json(out/'status.json',dict(status='validation',seed=seed,epoch=epoch,updated=datetime.now().isoformat()))
                    for name,model in models.items():
                        validation=evaluate_validation(model,val_loader,opts.device,loss_config)
                        score=selection_score(validation)
                        if not math.isfinite(score):
                            raise RuntimeError('Nonfinite validation MSE')
                        row=dict(seed=seed,variant=name,epoch=epoch,train_loss=losses[name]/len(train),
                            train_mse=neural_losses[name]/len(train),train_relation=relation_losses[name]/len(train),train_auxiliary_relation=auxiliary_relation_losses[name]/len(train),
                            validation_neural_mse=validation['neural_mse'],validation_mse=score,
                            validation_mae=validation['neural_mae'],validation_relation=validation[name] if name!='mse' else validation['patch_covariance'],validation_covariance=validation['patch_covariance'],validation_prediction_norm2=validation['prediction_covariance_norm2'],learning_rate=optimizers[name].param_groups[0]['lr'])
                        captured,handles=setup_hooks(model)
                        try:
                            with torch.no_grad(): model(probe_x)
                            branch_probe={f'scale{i}':{**summarize_tensor(captured[f'output_{i}']),
                                'sample_response_std':sample_response_std(captured[f'output_{i}'],len(probe_x),cfg.enc_in)} for i in range(4)}
                        finally:
                            for handle in handles: handle.remove()
                        atomic_json(folder/f'{name}_epoch{epoch:02d}_branches.json',branch_probe)
                        epoch_folder=folder/'epochs';epoch_folder.mkdir(exist_ok=True)
                        atomic_torch(epoch_folder/f'{name}_epoch{epoch:02d}.pth',model.state_dict())
                        row['scale_weights']=model.scale_weights().detach().cpu().tolist()
                        row['coarse_probe_std']=branch_probe['scale3']['std']
                        row['coarse_sample_response_std']=branch_probe['scale3']['sample_response_std']
                        curves.append(row)
                        log_metrics({k:v for k,v in row.items() if k not in ('seed','variant','scale_weights')},step=epoch,prefix=f'{name}/seed{seed}')
                        if score<best[name]:
                            best[name],best_epoch[name]=score,epoch
                            improved=True
                            best_states[name] = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                            atomic_torch(folder/(name+'_best.pth'),best_states[name])
                        captured.clear()
                        for group in optimizers[name].param_groups:
                            group['lr']=learning_rate(name,epoch,opts.epochs)
                        logger.info('EPOCH seed=%d arm=%s epoch=%d train_mse=%.8f train_relation=%.8f total=%.8f val_mse=%.8f val_mae=%.8f best_epoch=%d',seed,name,epoch,row['train_mse'],row['train_relation'],row['train_loss'],score,row['validation_mae'],best_epoch[name])
                    stale=joint_stale_count(stale, improved)
                    atomic_torch(folder/'last.pt',dict(models={k:m.state_dict() for k,m in models.items()},
                        optimizers={k:o.state_dict() for k,o in optimizers.items()},loss_config=loss_config,completed_updates=epoch*len(loader),best_models=best_states,epoch=epoch,stale=stale,best=best,best_epoch=best_epoch,curves=curves))
                    atomic_json(folder/'progress.json',dict(epoch=epoch,stale=stale,best=best,best_epoch=best_epoch,curves=curves,paired_epoch_seconds=time.monotonic()-start))
                    if invocation_stop and epoch>=invocation_stop:
                        atomic_json(out/'status.json',dict(status='checkpointed_verification_stop',seed=seed,epoch=epoch))
                        print('CHECKPOINTED VERIFICATION STOP',out)
                        return
                for name in models:
                    report['runs']=[v for v in report['runs'] if not(v['seed']==seed and v['variant']==name)]
                    report['runs'].append(dict(seed=seed,variant=name,best_epoch=best_epoch[name],validation_mse=best[name],
                        trained_epochs=curves[-1]['epoch'],stop_reason='joint_patience' if stale>=opts.patience else 'max_epochs',joint_stale=stale,checkpoint=str(folder/(name+'_best.pth'))))
                publish_report(out,report)
                del model, groups
                del models,optimizers
                import gc
                gc.collect()
                if opts.device=='cuda':torch.cuda.empty_cache()
            # Full validation diagnostics for selected weights, without test access.
            report['status']='selected_validation'
            publish_report(out,report)
            for row in report['runs']:
                if 'selected_validation' in row: continue
                model=VARIANTS[row['variant']](arm_config(cfg,row['variant'])).to(opts.device).eval()
                model.load_state_dict(torch.load(row['checkpoint'],map_location=opts.device,weights_only=True),strict=True)
                atomic_json(out/'status.json',dict(status='selected_validation',seed=row['seed'],variant=row['variant'],updated=datetime.now().isoformat()))
                row['model_config'] = dict(vars(arm_config(cfg,row['variant'])), seasonal_normalize=False, feature_decomposition=True,
                                           raw_trend_bypass=TREND_BYPASS[row['variant']], readout_split_variant='seasonal')
                row['model_config']['model'] = MODEL_NAMES[row['variant']]
                with SelectedValidationCapture(model, out/str(row['seed']), row['variant'], len(val)) as capture:
                    row['selected_validation'],prediction,truth,history=evaluate(model,val_loader,opts.device,retain=True,loss_config=loss_config)
                row['mechanism_diagnostics']=capture.report(prediction,truth)
                atomic_json(out/str(row['seed'])/(row['variant']+'_mechanism_diagnostics.json'),row['mechanism_diagnostics'])
                if abs(row['selected_validation']['mse']-row['validation_mse']) > 1e-7:
                    raise RuntimeError('Selected neural MSE differs from recorded selection')
                row['validation_quarters'] = []
                for indices in np.array_split(np.arange(len(truth)), 4):
                    metric = metrics(prediction[indices].astype('float64'), truth[indices].astype('float64'))
                    metric['low4_mse'] = metric['mse'] * metric['low_frequency_error_share']['4']
                    row['validation_quarters'].append(metric)
                np.savez_compressed(out/str(row['seed'])/(row['variant']+'_selected_validation.npz'), prediction=prediction, truth=truth, history=history)
                publish_report(out,report)
                del capture, model
            report['status']='completed'
            publish_report(out,report)
            atomic_json(out/'status.json',dict(status='completed',updated=datetime.now().isoformat()))
            print('STUDY COMPLETE',out,flush=True)
    except BaseException as exc:
        if isinstance(exc,KeyboardInterrupt):status='interrupted'
        else:status='failed'
        report.update(status=status,error=repr(exc))
        publish_report(out,report)
        atomic_json(out/'status.json',dict(status=status,error=repr(exc),updated=datetime.now().isoformat()))
        raise
    finally:
        lock.close()


if __name__=='__main__':
    main()
