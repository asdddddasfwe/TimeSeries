"""Pure-network MSE versus joint history-future Gaussian autocovariance KL paired experiment."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
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
from models.OscillatorLowRankTimeMix import Model
from scripts.train_timemix_joint_gaussian_kl import atomic_json, ARMS
from scripts.evaluate_frozen_timemix import verify_artifact
from scripts.diagnose_timemix_linear_reference import score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase(out, status, **extra):
    atomic_json(out/'status.json', dict(status=status, updated=datetime.now(timezone.utc).isoformat(), **extra))
    print('PHASE', status, extra, flush=True)


def run(command):
    subprocess.run([sys.executable, '-u', *command], cwd=ROOT, check=True)


def make_artifact(folder, cfg, protocol, row):
    config = dict(vars(cfg), model='OscillatorLowRankTimeMix', is_training=0)
    for name, digest in protocol['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Source changed since training: '+name)
    restored = Model(SimpleNamespace(**config))
    restored.load_state_dict(torch.load(row['checkpoint'], map_location='cpu', weights_only=True), strict=True)
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(row['checkpoint'], folder/'checkpoint.pth')
    atomic_json(folder/'model_config.json', config)
    identity = dict(checkpoint_sha256=sha(folder/'checkpoint.pth'), model_config_sha256=sha(folder/'model_config.json'))
    smoke = bool(protocol['options']['train_limit'] or protocol['options']['val_limit'])
    manifest = dict(kind=config['model'], selection_status='smoke_validation_only' if smoke else 'frozen_for_test',
                    **identity, neural_training_source_sha256=protocol['source_sha256'], seed=row['seed'],
                    best_epoch=row['best_epoch'], variant=row['variant'], loss_config=protocol['loss_config'],
                    selection='Pure neural complete validation MSE; KL never selects checkpoints',
                    output='pure_neural_network')
    atomic_json(folder/'manifest.json', manifest)
    atomic_json(folder/'freeze.json', dict(**identity, validation_evidence=row,
                reason='Both paired arms completed validation selection; no test-dependent selection',
                created=datetime.now(timezone.utc).isoformat()))
    verify_artifact(folder, 'val' if smoke else 'test')
    return str(folder)


def check_test_result(result, receipt, manifest):
    if not (result['status']=='completed' and result['split']=='test' and not result['smoke']
            and result['windows']==result['official_windows']==11425
            and result['checkpoint_sha256']==receipt['checkpoint_sha256']==manifest['checkpoint_sha256']
            and result['model_config_sha256']==manifest['model_config_sha256']
            and result['source_hashes_verified']):
        raise RuntimeError('Invalid full-test receipt')


def complete_test(artifact, output, device):
    artifact = Path(artifact)
    manifest, _ = verify_artifact(artifact, 'test')
    receipt_path = artifact/'test_result.json'
    if not receipt_path.exists():
        if output.exists():
            raise RuntimeError('Incomplete existing test output; refusing automatic repeat')
        run(['scripts/evaluate_frozen_timemix.py', '--artifact', str(artifact), '--split', 'test',
             '--device', device, '--output', str(output)])
    receipt = json.loads(receipt_path.read_text())
    result = json.loads(Path(receipt['report']).read_text())
    check_test_result(result, receipt, manifest)
    return dict(metrics=result['metrics'], report=receipt['report'])


def audit_predictions(path, expected):
    with np.load(path) as cache:
        pred, truth = cache['prediction'].astype('float64'), cache['truth'].astype('float64')
    if pred.shape != truth.shape or pred.shape[1:] != (96, 7) or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise RuntimeError('Invalid stored predictions')
    checked = score(pred, truth)
    for key in ['mse', 'mae', 'low4_mse', 'centered_amplitude_ratio', 'window_offset_mse']:
        if abs(checked[key]-expected[key]) > 1e-10:
            raise RuntimeError('Offline metric differs: '+key)
    return dict(path=str(path), sha256=sha(path), windows=len(pred), verified=True,
                per_channel_mae=np.abs(pred-truth).mean(axis=(0,1)).tolist(),
                quarters=[score(pred[ix],truth[ix]) for ix in np.array_split(np.arange(len(pred)),4)])


def write_results(out, report):
    path = out/'results_zh.md' if report['smoke'] else ROOT/'docs/experiments/OscillatorJointGaussianKL_results_20260928.md'
    lines = ['# 历史未来联合高斯KL：纯网络配对实验结果', '',
        '两组完整递推秩16网络、相同初值、学习率0.001；对照纯MSE，候选MSE加历史未来联合高斯KL；均按纯网络验证MSE选轮。', '',
        '| 分组 | 完成轮次 | 所选轮次 | 验证MSE | 验证MAE | 测试MSE | 测试MAE |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in report['runs']:
        val=row['selected_validation']; test=row.get('test',{}).get('metrics',{})
        lines.append('| '+' | '.join([row['variant'], str(row['trained_epochs']), str(row['best_epoch']),
                     f"{val['mse']:.8f}",f"{val['mae']:.8f}",
                     f"{test['mse']:.8f}" if test else '未测试',f"{test['mae']:.8f}" if test else '未测试'])+' |')
    if all('test' in row for row in report['runs']):
        control,candidate=({row['variant']:row for row in report['runs']}[name]['test']['metrics'] for name in ARMS)
        for key in ['mse','mae']:
            delta=candidate[key]-control[key]
            lines.append(f"\n候选测试{key}相对本轮对照：{delta:+.8f}（{100*delta/control[key]:+.3f}%）。")
    lines += ['', '单种子2021；KL下降不是总体预测改善的证据。只评估纯网络；历史测试已用于研究，不是新的独立留出集。',
              '保存预测的离线复算、七变量、四时段、low4、幅值及窗口水平误差见report.json。训练/验证曲线及分支响应见training/2021。']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return path


def main():
    os.chdir(ROOT)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--train-limit',type=int,default=0)
    parser.add_argument('--val-limit',type=int,default=0)
    parser.add_argument('--kl-weight',type=float,default=.001)
    parser.add_argument('--kl-warmup-epochs',type=float,default=2.)
    parser.add_argument('--variance-floor',type=float,default=.01)
    parser.add_argument('--kl-jitter',type=float,default=.01)
    parser.add_argument('--log-every',type=int,default=50)
    parser.add_argument('--skip-test',action='store_true')
    opts=parser.parse_args()
    if (opts.train_limit or opts.val_limit) and not opts.skip_test:
        raise ValueError('Limited smoke runs cannot open test data')
    out=opts.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    atomic_json(out/'pipeline_options.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(opts).items()})
    report=dict(status='running',runs=[],smoke=bool(opts.train_limit or opts.val_limit),test_accessed=False)
    atomic_json(out/'report.json',report);torch.set_num_threads(2)
    try:
        training=out/'training';phase(out,'paired_training')
        command=['scripts/train_timemix_joint_gaussian_kl.py','--output',str(training),'--device',opts.device,'--seeds','2021']
        for key in ['epochs','patience','train_limit','val_limit','kl_weight','kl_warmup_epochs','variance_floor','kl_jitter','log_every']:
            command += ['--'+key.replace('_','-'),str(getattr(opts,key))]
        run(command)
        tr=json.loads((training/'report.json').read_text());tp=json.loads((training/'protocol.json').read_text())
        if tr['status']!='completed' or len(tr['runs'])!=2:raise RuntimeError('Paired training incomplete')
        if not report['smoke'] and (tr['train_windows'],tr['validation_windows'])!=(34369,11425):
            raise RuntimeError('Incomplete training or validation coverage')
        cfg=SimpleNamespace(**tp['model_args'])
        for source in tr['runs']:
            row=dict(source)
            row['validation_audit']=audit_predictions(training/'2021'/(row['variant']+'_selected_validation.npz'),row['selected_validation'])
            phase(out,'freezing',variant=row['variant'])
            row['artifact']=make_artifact(out/(row['variant']+'_artifact'),cfg,tp,row)
            report['runs'].append(row);atomic_json(out/'report.json',report)
        # BOTH arms are frozen before either test is accessed.
        if not opts.skip_test:
            for row in report['runs']:
                phase(out,'final_test',variant=row['variant']);report['test_accessed']=True
                atomic_json(out/'report.json',report)
                row['test']=complete_test(row['artifact'],out/(row['variant']+'_test'),opts.device)
                row['test_audit']=audit_predictions(Path(row['test']['report']).parent/'predictions.npz',row['test']['metrics'])
                atomic_json(out/'report.json',report)
        report['status']='completed';report['result_document']=str(write_results(out,report))
        atomic_json(out/'report.json',report);phase(out,'completed')
        if not report['smoke']:
            path=ROOT/'tmp/timemix_0308_goal_state.json';state=json.loads(path.read_text())
            previous=state.get('joint_gaussian_kl_training_results',{})
            previous.update(status='completed_offline_verified',report=str(out/'report.json'),document=report['result_document'],test_accessed=report['test_accessed'])
            state['joint_gaussian_kl_training_results']=previous;atomic_json(path,state)
        print('JOINT GAUSSIAN KL EXPERIMENT COMPLETE',out,flush=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));atomic_json(out/'report.json',report)
        phase(out,'failed',error=repr(exc));raise


if __name__=='__main__':
    main()
