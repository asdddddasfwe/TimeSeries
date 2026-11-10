"""完整递推秩16学习率实验：纯网络训练/选轮，分别冻结纯网络和融合。"""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from layers.LinearForecastBlend import LinearForecastBlend
from scripts.train_timemix_low_lr import atomic_json, atomic_torch, LEARNING_RATES
from scripts.run_timemix_recurrent_rank16_experiment import sha
from scripts.evaluate_frozen_timemix import verify_artifact
from scripts.diagnose_timemix_linear_reference import score

NAMES = {'control': '学习率0.001', 'low_lr': '学习率0.0003'}
MODELS = {'neural': 'OscillatorLowRankTimeMix', 'blend': 'OscillatorLowRankLinearBlend'}
PREVIOUS = {
    'neural': ROOT/'tmp/lag_study_timemix_recurrent_rank16_neural_test_20260927/artifact',
    'blend': ROOT/'tmp/lag_study_timemix_recurrent_rank16_20260927_seed2021/combined_artifact',
}


def phase(out, status, **extra):
    atomic_json(out/'status.json', dict(status=status, updated=datetime.now(timezone.utc).isoformat(), **extra))


def run(command):
    subprocess.run([sys.executable, '-u', *command], cwd=ROOT, check=True)


def make_artifact(folder, kind, cfg, protocol, row, weights):
    config = dict(vars(cfg), model=MODELS[kind], is_training=0,
                  learning_rate=LEARNING_RATES[row['variant']])
    if folder.exists():
        raise FileExistsError('已有产物不自动覆盖：'+str(folder))
    for name, digest in protocol['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('源码身份变化：'+name)
    neural = importlib.import_module('models.OscillatorLowRankTimeMix').Model(cfg)
    source = torch.load(row['checkpoint'], map_location='cpu', weights_only=True)
    neural.load_state_dict(source, strict=True)
    folder.mkdir(parents=True)
    if kind == 'neural':
        shutil.copyfile(row['checkpoint'], folder/'checkpoint.pth')
    else:
        atomic_torch(folder/'checkpoint.pth', LinearForecastBlend(neural, weights).state_dict())
    atomic_json(folder/'model_config.json', config)
    identity = dict(checkpoint_sha256=sha(folder/'checkpoint.pth'),
                    model_config_sha256=sha(folder/'model_config.json'))
    smoke = bool(protocol['options']['train_limit'] or protocol['options']['val_limit'])
    manifest = dict(kind=MODELS[kind], selection_status='smoke_validation_only' if smoke else 'frozen_for_test',
        **identity, adapter_source_sha256=protocol['source_sha256'], seed=row['seed'],
        best_epoch=row['best_epoch'], source_checkpoint_sha256=sha(Path(row['checkpoint'])),
        selection='仅按纯神经完整验证MSE选轮，融合只记录；不按测试选择',
        output='纯神经网络' if kind == 'neural' else '同一选定网络与训练拟合线性各0.5')
    atomic_json(folder/'manifest.json', manifest)
    atomic_json(folder/'freeze.json', dict(**identity, validation_evidence=row,
        reason='两组训练及纯网络验证选轮完成；冻结两种输出身份，不按融合或测试重新选轮',
        created=datetime.now(timezone.utc).isoformat()))
    restored = importlib.import_module('models.'+MODELS[kind]).Model(SimpleNamespace(**config))
    restored.load_state_dict(torch.load(folder/'checkpoint.pth', map_location='cpu', weights_only=True), strict=True)
    verify_artifact(folder, 'val' if smoke else 'test')
    return str(folder)


def reuse_test(artifact, previous):
    """兼容纯网络/融合两种历史凭证；不满足身份时不复用。"""
    artifact, previous = Path(artifact), Path(previous)
    if not (previous/'test_result.json').exists():
        return None
    current, config = verify_artifact(artifact, 'test')
    old = json.loads((previous/'manifest.json').read_text())
    if current['checkpoint_sha256'] != sha(previous/'checkpoint.pth'):
        return None
    old_config = json.loads((previous/'model_config.json').read_text())
    ignored = ('log_dir', 'tensorboard_dir')
    if {k:v for k,v in config.items() if k not in ignored} != {k:v for k,v in old_config.items() if k not in ignored}:
        return None
    if not (old.get('adapter_source_sha256') or old.get('neural_training_source_sha256')):
        return None
    try:
        verify_artifact(previous, 'test')
    except ValueError:
        return None
    receipt = json.loads((previous/'test_result.json').read_text())
    result = json.loads(Path(receipt['report']).read_text())
    check_test_result(result, receipt, old)
    audit = dict(reused=True, previous_artifact=str(previous), report=receipt['report'],
        checkpoint_sha256=current['checkpoint_sha256'], model_config_sha256=current['model_config_sha256'],
        previous_model_config_sha256=old['model_config_sha256'], full_config_equal_except=list(ignored),
        previous_source_hashes_verified=True)
    atomic_json(artifact/'test_reuse_evidence.json', audit)
    atomic_json(artifact/'test_result.json', audit)
    return audit


def check_test_result(result, receipt, manifest):
    if not (result['status']=='completed' and result['split']=='test' and not result['smoke']
            and result['windows']==result['official_windows']==11425
            and result['checkpoint_sha256']==receipt['checkpoint_sha256']==manifest['checkpoint_sha256']
            and result['model_config_sha256']==manifest['model_config_sha256']
            and result['source_hashes_verified']):
        raise RuntimeError('测试凭证核验失败')


def complete_test(artifact, output, previous=None, device='cuda'):
    artifact = Path(artifact)
    manifest, _ = verify_artifact(artifact, 'test')
    receipt = artifact/'test_result.json'
    if not receipt.exists() and previous is not None:
        reuse_test(artifact, previous)
    if not receipt.exists():
        if output.exists():
            raise RuntimeError('已有测试目录而无完成凭证，不自动重复：'+str(output))
        run(['scripts/evaluate_frozen_timemix.py', '--artifact', str(artifact), '--split', 'test',
             '--device', device, '--output', str(output)])
    evidence = json.loads(receipt.read_text())
    result = json.loads(Path(evidence['report']).read_text())
    if evidence.get('reused'):
        checked = reuse_test(artifact, Path(evidence['previous_artifact']))
        if checked is None:
            raise RuntimeError('已有复用凭证身份不再匹配')
    else:
        check_test_result(result, evidence, manifest)
    return dict(metrics=result['metrics'], report=evidence['report'], reused=evidence.get('reused', False))


def write_results(out, report):
    path = out/'results_zh.md' if report['smoke'] else ROOT/'docs/experiments/OscillatorLowLR_results_20260927.md'
    lines = ['# 完整递推秩16：学习率配对实验结果', '',
        '两组均为完整递推加秩16模型，从完全相同初值训练；唯一配置干预为基础学习率0.001与0.0003，余弦日程形状相同。纯神经MSE训练并按纯神经验证MSE选轮，融合验证只记录。', '',
        '| 分组 | 完成轮数 | 所选轮次 | 神经验证MSE | 神经验证MAE | 融合验证MSE | 纯网络测试MSE | 纯网络测试MAE | 融合测试MSE | 融合测试MAE |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in report['runs']:
        def value(kind, key):
            return f"{row[kind]['metrics'][key]:.8f}" if kind in row else '未测试'
        lines.append('| '+' | '.join([NAMES[row['variant']], str(row['trained_epochs']), str(row['best_epoch']),
            f"{row['selected_validation']['mse']:.8f}", f"{row['selected_validation']['mae']:.8f}",
            f"{row['selected_blend_validation']['mse']:.8f}", value('neural_test','mse'), value('neural_test','mae'),
            value('blend_test','mse'),value('blend_test','mae')])+' |')
    if all('neural_test' in row for row in report['runs']) and len(report['runs'])==2:
        c,n = report['runs']
        lines += ['', '## 本轮配对差值', '']
        for kind,label in [('neural_test','纯网络测试'),('blend_test','融合测试')]:
            for key,metric in [('mse','均方误差'),('mae','平均绝对误差')]:
                a,b=c[kind]['metrics'][key],n[kind]['metrics'][key]
                lines.append(f'- {label}{metric}：低学习率减对照 {b-a:+.8f}，相对变化 {(b/a-1)*100:+.3f}%。')
    for row in report['runs']:
        lines += ['', '## '+NAMES[row['variant']], '', '| 输出 | 低四阶误差 | 幅值比 | 窗口水平误差 |', '| --- | --- | --- | --- |']
        for key,label in [('selected_validation','神经验证'),('selected_blend_validation','融合验证'),('neural_test','纯网络测试'),('blend_test','融合测试')]:
            if key not in row: continue
            metric=row[key]['metrics'] if key.endswith('_test') else row[key]
            lines.append('| '+label+' | '+' | '.join(f'{metric[k]:.8f}' for k in ['low4_mse','centered_amplitude_ratio','window_offset_mse'])+' |')
    lines += ['', '## 证据边界', '',
        '随机种子仅2021；两组最多30轮、联合patience10，获得相同轮次和更新次数。低学习率改变给定预算内的优化轨迹及衰减更新幅度，不能单独证明过拟合的唯一根因。不得按测试延长训练或选择轮次。',
        '线性组件只用训练集拟合，岭惩罚0.001、共享惩罚0.1，两组固定各半融合。纯网络与融合成绩分别报告。验证和测试区间此前已被研究使用，不是新的独立留出集。',
        '逐轮神经/融合验证曲线、分支探针、选定预测、冻结身份与测试凭证保存在实验目录。既有测试只有身份严格一致才复用，存在未完成测试目录时拒绝自动重测。', '', str(out), '']
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def main():
    os.chdir(ROOT)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--skip-test', action='store_true')
    opts=parser.parse_args()
    if (opts.train_limit or opts.val_limit) and not opts.skip_test:
        raise ValueError('有限窗口冒烟不能访问test')
    out=opts.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    lock=(out/'pipeline.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    atomic_json(out/'pipeline_options.json', {k:str(v) if isinstance(v,Path) else v for k,v in vars(opts).items()})
    report=dict(status='running', runs=[], smoke=bool(opts.train_limit or opts.val_limit), test_accessed=False)
    atomic_json(out/'report.json',report); torch.set_num_threads(2)
    try:
        training=out/'training'; phase(out,'paired_training')
        run(['scripts/train_timemix_low_lr.py','--output',str(training),'--device',opts.device,
             '--epochs',str(opts.epochs),'--patience',str(opts.patience),'--seeds','2021',
             '--train-limit',str(opts.train_limit),'--val-limit',str(opts.val_limit)])
        tr=json.loads((training/'report.json').read_text()); tp=json.loads((training/'protocol.json').read_text())
        if tr['status']!='completed' or len(tr['runs'])!=2: raise RuntimeError('训练未完整完成')
        if not report['smoke'] and (tr['train_windows'],tr['validation_windows'])!=(34369,11425):
            raise RuntimeError('训练/验证窗口不完整')
        cfg=SimpleNamespace(**tp['model_args']); weights=np.load(training/'linear/coefficients.npy')
        if sha(training/'linear/coefficients.npy')!=tr['linear_sha256']: raise RuntimeError('线性身份变化')
        for source in tr['runs']:
            row=dict(source); row['artifacts']={}
            cache=np.load(training/'2021'/(row['variant']+'_selected_blend_validation.npz'))
            row['blend_validation_quarters']=[score(cache['prediction'][ix].astype('float64'),cache['truth'][ix].astype('float64')) for ix in np.array_split(np.arange(len(cache['truth'])),4)]
            for kind in MODELS:
                phase(out,'freezing',variant=row['variant'],kind=kind)
                row['artifacts'][kind]=make_artifact(out/(row['variant']+'_'+kind+'_artifact'),kind,cfg,tp,row,weights)
            report['runs'].append(row); atomic_json(out/'report.json',report)
        # 两组、两种输出均冻结后才访问test。
        if not opts.skip_test:
            for row in report['runs']:
                for kind in MODELS:
                    phase(out,'final_test',variant=row['variant'],kind=kind)
                    previous=PREVIOUS[kind] if row['variant']=='control' else None
                    row[kind+'_test']=complete_test(row['artifacts'][kind],out/(row['variant']+'_'+kind+'_test'),previous,opts.device)
                    report['test_accessed']=True; atomic_json(out/'report.json',report)
        report['status']='completed'; report['result_document']=str(write_results(out,report))
        atomic_json(out/'report.json',report); phase(out,'completed')
        if not report['smoke']:
            state_path=ROOT/'tmp/timemix_0308_goal_state.json'; state=json.loads(state_path.read_text())
            state['low_lr_results']=dict(status='completed',report=str(out/'report.json'),document=report['result_document'],test_accessed=report['test_accessed'])
            atomic_json(state_path,state)
        print('LOW LR EXPERIMENT COMPLETE',out,flush=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc)); atomic_json(out/'report.json',report)
        phase(out,'failed',error=repr(exc)); raise


if __name__=='__main__':
    main()
