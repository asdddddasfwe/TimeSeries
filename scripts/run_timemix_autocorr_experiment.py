"""Season-normalized MSE, covariance KL and correlation KL from-scratch comparison."""
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
import time
from types import SimpleNamespace

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from models.OscillatorSeasonNormTimeMix import Model
from scripts.train_timemix_autocorr import atomic_json, ARMS, NORMALIZED
from utils.season_norm_study_metrics import shape_metrics
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
    config = dict(row['model_config'], model='OscillatorSeasonNormTimeMix', is_training=0)
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
                reason='All three arms completed validation selection; no test-dependent selection',
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
    shape = shape_metrics(pred, truth)
    for key in ['mse', 'mae', 'low4_mse', 'centered_amplitude_ratio', 'window_offset_mse']:
        if abs(checked[key]-expected[key]) > 1e-10:
            raise RuntimeError('Offline metric differs: '+key)
    return dict(path=str(path), sha256=sha(path), windows=len(pred), verified=True, shape_metrics=shape,
                per_channel_mae=np.abs(pred-truth).mean(axis=(0,1)).tolist(),
                quarters=[score(pred[ix],truth[ix]) for ix in np.array_split(np.arange(len(pred)),4)])


def pairwise_contrasts(rows, split):
    metrics = {v['variant']:v['selected_validation'] if split == 'val' else v['test']['metrics'] for v in rows}
    result = {}
    for name, base in [('covariance_kl','control'), ('correlation_kl','control'), ('correlation_kl','covariance_kl')]:
        result[name+'_minus_'+base] = {}
        for key in ('mse','mae','low4_mse','window_offset_mse','centered_amplitude_ratio'):
            delta = metrics[name][key] - metrics[base][key]
            result[name+'_minus_'+base][key] = dict(absolute=delta, percent=100*delta/metrics[base][key] if metrics[base][key] else None)
    return result


def wait_for_previous(out, previous):
    """Queue without starting a training/evaluation process until prior pipeline exits."""
    previous = previous.resolve()
    old_pid = int((previous/'pid.txt').read_text())
    proc = Path('/proc')/str(old_pid)
    expected = b'run_timemix_season_norm_experiment.py'
    cmd = (proc/'cmdline').read_bytes() if proc.exists() else b''
    if cmd and (expected not in cmd or str(previous).encode() not in cmd):
        raise RuntimeError('Previous pipeline PID identity mismatch')
    starttime = (proc/'stat').read_text().split()[21] if cmd else None
    paths = set()
    for folder in ('models','layers','data_provider','utils','scripts'):
        paths.update((ROOT/folder).rglob('*.py'))
    paths.update([ROOT/'run.py', ROOT/'docs/experiments/OscillatorAutocorr_protocol_20260929.md', ROOT/'tests/test_joint_gaussian_autocorr_loss.py'])
    pinned = {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    atomic_json(out/'queued_source_sha256.json', pinned)
    atomic_json(out/'queue.json',dict(wait_for=str(previous),pid=old_pid,process_starttime=starttime,
                queued_at=datetime.now(timezone.utc).isoformat(),source_sha256_file=str(out/'queued_source_sha256.json')))
    while True:
        prior = json.loads((previous/'report.json').read_text())
        if prior['status'] in ('failed','interrupted'):
            raise RuntimeError('Previous experiment failed; queue will not auto-resume it: '+str(prior.get('error')))
        alive = False
        if proc.exists() and starttime:
            try:
                alive = ((proc/'stat').read_text().split()[21] == starttime and (proc/'cmdline').read_bytes() == cmd)
            except FileNotFoundError:
                pass
        if prior['status'] == 'completed' and not alive:
            if len(prior.get('runs', [])) != 4 or not all(v.get('validation_audit',{}).get('verified') and v.get('test_audit',{}).get('verified') for v in prior['runs']):
                raise RuntimeError('Previous experiment missing completed validation/test audits')
            break
        if not alive:
            raise RuntimeError('Previous pipeline disappeared before completion; no automatic recovery')
        phase(out,'queued',wait_for=str(previous),previous_status=prior['status'])
        time.sleep(30)
    for name, digest in pinned.items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Queued source changed before launch: '+name)
    atomic_json(out/'queue_released.json', dict(released_at=datetime.now(timezone.utc).isoformat(),source_hashes_verified=True))


def write_results(out, report):
    path = out/'results_zh.md' if report['smoke'] else ROOT/'docs/experiments/OscillatorAutocorr_results_20260929.md'
    lines = ['# 自相关矩阵KL三组从头训练结果', '',
        '三组均保留季节归一化，相同初始参数、预测和固定预算。纯MSE、MSE+协方差KL/d、MSE+自相关矩阵KL/d；无lambda无预热。仅纯验证MSE选轮。', '',
        '|分组|完成轮次|所选轮次|验证MSE|验证MAE|测试MSE|测试MAE|', '|---|---|---|---|---|---|---|']
    for row in report['runs']:
        val=row['selected_validation']; test=row.get('test',{}).get('metrics',{})
        lines.append('| '+' | '.join([row['variant'],str(row['trained_epochs']),str(row['best_epoch']),
            f"{val['mse']:.8f}", f"{val['mae']:.8f}",f"{test['mse']:.8f}" if test else '未测试', f"{test['mae']:.8f}" if test else '未测试'])+' |')
    report['contrasts']={'val':pairwise_contrasts(report['runs'],'val')}
    if all('test' in row for row in report['runs']):
        report['contrasts']['test']=pairwise_contrasts(report['runs'],'test')
    lines += ['', '## 绝对和百分比差值', '', '```json',json.dumps(report['contrasts'],ensure_ascii=False,indent=2),'```']
    for row in report['runs']:
        lines += ['', '## '+row['variant'], '', '```json',json.dumps({k:row[k] for k in ['selected_validation','validation_quarters','validation_audit','test','test_audit'] if k in row},ensure_ascii=False,indent=2),'```']
    progress=json.loads((out/'training/2021/progress.json').read_text())
    lines += ['', '## 逐轮训练与验证', '', '|轮|分组|训练MSE|协方差KL/d|自相关KL/d|总损失|验证MSE|验证MAE|', '|---|---|---|---|---|---|---|---|']
    for v in progress['curves']:
        lines.append(f"|{v['epoch']}|{v['variant']}|{v['train_mse']:.8f}|{v['train_cov_kl']:.8f}|{v['train_corr_kl']:.8f}|{v['train_loss']:.8f}|{v['validation_mse']:.8f}|{v['validation_mae']:.8f}|")
    lines += ['', '每组同时记录两种KL；control的auxiliary_kl为0，其余组等于对应KL/d，不能再次除9。', '',
        '分支响应见training/2021/branch_traces.json及各轮branches.json；内部readout位于季节尺度还原前。', '',
        '自相关矩阵采用各自对角归一化；低方差时收缩到单位阵，再加jitter。高于方差阈值时完整拼接序列的共同缩放不改变相关矩阵，但固定历史时仅缩放未来仍会改变联合相关性。', '',
        '该实验同时改变幅值敏感性、矩阵条件数和梯度大小；相同系数不意味着相同梯度强度。相关KL下降不是逐点预测改善，不能单独证明幅值约束是唯一根因。相位诊断是平滑曲线对齐描述，不是逐峰因果定位。', '',
        '单种子2021，不宣称跨种子稳定；历史验证测试已用于研究，不是新的独立留出集。无融合预测或指标，测试只在三组冻结后各一次。']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return path


def main():
    os.chdir(ROOT)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--patience',type=int,default=31)
    parser.add_argument('--train-limit',type=int,default=0)
    parser.add_argument('--val-limit',type=int,default=0)
    parser.add_argument('--variance-floor',type=float,default=.01)
    parser.add_argument('--kl-jitter',type=float,default=.01)
    parser.add_argument('--log-every',type=int,default=50)
    parser.add_argument('--skip-test',action='store_true')
    parser.add_argument('--wait-for',type=Path,help='Wait for the authorized prior four-arm pipeline to complete')
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
        if opts.wait_for:
            report['status']='queued';report['wait_for']=str(opts.wait_for.resolve());atomic_json(out/'report.json',report)
            wait_for_previous(out,opts.wait_for)
            report['status']='running';atomic_json(out/'report.json',report)
        training=out/'training';phase(out,'paired_training')
        command=['scripts/train_timemix_autocorr.py','--output',str(training),'--device',opts.device,'--seeds','2021']
        for key in ['epochs','patience','train_limit','val_limit','variance_floor','kl_jitter','log_every']:
            command += ['--'+key.replace('_','-'),str(getattr(opts,key))]
        run(command)
        tr=json.loads((training/'report.json').read_text());tp=json.loads((training/'protocol.json').read_text())
        if tr['status']!='completed' or len(tr['runs'])!=3:raise RuntimeError('Paired training incomplete')
        if not report['smoke'] and (tr['train_windows'],tr['validation_windows'])!=(34369,11425):
            raise RuntimeError('Incomplete training or validation coverage')
        cfg=SimpleNamespace(**tp['model_args'])
        for source in tr['runs']:
            row=dict(source)
            row['validation_audit']=audit_predictions(training/'2021'/(row['variant']+'_selected_validation.npz'),row['selected_validation'])
            phase(out,'freezing',variant=row['variant'])
            row['artifact']=make_artifact(out/(row['variant']+'_artifact'),cfg,tp,row)
            report['runs'].append(row);atomic_json(out/'report.json',report)
        # All THREE arms are frozen before any test is accessed.
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
            previous=state.get('autocorr_training_results',{})
            previous.update(status='completed_offline_verified',report=str(out/'report.json'),document=report['result_document'],test_accessed=report['test_accessed'])
            state['autocorr_training_results']=previous;atomic_json(path,state)
        print('AUTOCORRELATION EXPERIMENT COMPLETE',out,flush=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));atomic_json(out/'report.json',report)
        phase(out,'failed',error=repr(exc));raise


if __name__=='__main__':
    main()
