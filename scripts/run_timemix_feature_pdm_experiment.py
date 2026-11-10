"""Original decomposition versus layerwise feature PDM: pure-MSE scratch experiment."""
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
from models.OscillatorFeaturePDMTimeMix import Model
from scripts.train_timemix_feature_pdm import atomic_json, ARMS, FEATURE_PDM
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
    config = dict(row['model_config'], model='OscillatorFeaturePDMTimeMix', is_training=0)
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
                reason='Both architecture arms completed validation selection; no test-dependent selection',
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


def write_results(out, report):
    path = out/'results_zh.md' if report['smoke'] else ROOT/'docs/experiments/OscillatorFeaturePDM_results_20260929.md'
    lines = ['# 原分解与每层特征PDM：纯MSE从头训练', '',
        '两组均无季节残差二次归一化，按纯完整验证MSE选已完成轮次。新模型保留FFT和完整递推，逐层进行TimeMixer风格的特征分解与季节/趋势定向混合。', '',
        '|分组|参数数|完成轮次|所选轮次|验证MSE|验证MAE|测试MSE|测试MAE|','|---|---|---|---|---|---|---|---|']
    audit=json.loads((out/'training/2021/initialization_audit.json').read_text())
    for row in report['runs']:
        v=row['selected_validation'];t=row.get('test',{}).get('metrics',{})
        lines.append('| '+' | '.join([row['variant'],str(audit['parameter_counts'][row['variant']]),str(row['trained_epochs']),str(row['best_epoch']),f"{v['mse']:.8f}",f"{v['mae']:.8f}",f"{t['mse']:.8f}" if t else '未测试',f"{t['mae']:.8f}" if t else '未测试'])+' |')
    by={v['variant']:v for v in report['runs']};report['contrasts']={}
    for split in ['val']+(['test'] if all('test' in v for v in report['runs']) else []):
        a=by['control']['selected_validation'] if split=='val' else by['control']['test']['metrics']
        b=by['feature_pdm']['selected_validation'] if split=='val' else by['feature_pdm']['test']['metrics']
        report['contrasts'][split]={k:dict(absolute=b[k]-a[k],percent=100*(b[k]-a[k])/a[k] if a[k] else None) for k in ['mse','mae','low4_mse','window_offset_mse','centered_amplitude_ratio']}
    lines += ['', '## 新结构减原结构', '', '```json',json.dumps(report['contrasts'],ensure_ascii=False,indent=2),'```']
    for row in report['runs']:
        lines += ['', '## '+row['variant'], '', '```json',json.dumps({k:row[k] for k in ['selected_validation','validation_quarters','validation_audit','test','test_audit'] if k in row},ensure_ascii=False,indent=2),'```']
    progress=json.loads((out/'training/2021/progress.json').read_text())
    lines += ['', '## 逐轮曲线', '', '|轮|组|训练MSE|KL诊断|auxiliary KL|总损失|验证MSE|验证MAE|', '|---|---|---|---|---|---|---|---|']
    for v in progress['curves']:
        lines.append(f"|{v['epoch']}|{v['variant']}|{v['train_mse']:.8f}|{v['train_kl']:.8f}|{v['train_auxiliary_kl']:.8f}|{v['train_loss']:.8f}|{v['validation_mse']:.8f}|{v['validation_mae']:.8f}|")
    lines += ['', '两组auxiliary_kl始终0是预期，只优化MSE；联合协方差KL仅诊断。', '',
        '该干预改变分解位置、逐层混合、各尺度归一化、FFT路由输入及原始趋势直连支路，同时改变参数数和表示。共同参数相同，但特有模块不同，初始预测不要求一致，不能将效果唯一归因于某一个环节。', '',
        '不是完整TimeMixer：保留原FFT编码和振子完整递推、各尺度线性嵌入与位置编码，窗口Normalize无affine。原网络已有的LayerNorm保留，仅取消新增季节残差归一化。', '',
        '分支响应见training/2021/branch_traces.json。新结构递推读出包含PDM合并特征，不能与原季节读出作同一物理分量解释。相位仅描述平滑对齐，不等于逐峰因果定位。', '',
        '单种子2021，固定30轮，无融合。历史验证/测试已用于研究，不是新的独立留出集；不宣称跨种子稳定。']
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
        command=['scripts/train_timemix_feature_pdm.py','--output',str(training),'--device',opts.device,'--seeds','2021']
        for key in ['epochs','patience','train_limit','val_limit','variance_floor','kl_jitter','log_every']:
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
        # Both arms are frozen before any test is accessed.
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
            previous=state.get('feature_pdm_training_results',{})
            previous.update(status='completed_offline_verified',report=str(out/'report.json'),document=report['result_document'],test_accessed=report['test_accessed'])
            state['feature_pdm_training_results']=previous;atomic_json(path,state)
        print('FEATURE PDM EXPERIMENT COMPLETE',out,flush=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));atomic_json(out/'report.json',report)
        phase(out,'failed',error=repr(exc));raise


if __name__=='__main__':
    main()
