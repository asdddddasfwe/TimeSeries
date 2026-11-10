"""从头配对训练、固定线性拟合、冻结融合与一次完整测试的组合实验。"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from layers.LinearForecastBlend import LinearForecastBlend
from scripts.train_timemix_recurrent_rank16 import dataset, evaluate, atomic_json, atomic_torch
from scripts.diagnose_timemix_linear_reference import sufficient_statistics, score, predict
from utils.hierarchical_linear_helpers import solve_hierarchical
from torch.utils.data import DataLoader

MODELS = {"control": "OscillatorTimeMix", "combined": "OscillatorLowRankTimeMix"}
ADAPTERS = {"control": "OscillatorLinearBlend", "combined": "OscillatorLowRankLinearBlend"}
NAMES = {"control": "原方案重新训练", "combined": "完整递推与秩16组合"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase(out, value, **extra):
    atomic_json(out / "status.json", dict(status=value, updated=datetime.now(timezone.utc).isoformat(), **extra))


def run(command):
    subprocess.run([sys.executable, "-u", *command], cwd=ROOT, check=True)


def train_linear(cfg, limit, out):
    directory = out / "linear"
    directory.mkdir(exist_ok=True)
    receipt = directory / "fit_report.json"
    if receipt.exists():
        recorded = json.loads(receipt.read_text())
        if recorded['status'] != 'completed' or sha(directory/'coefficients.npy') != recorded['sha256']:
            raise ValueError('线性组件凭证不一致')
        return np.load(directory / "coefficients.npy")
    data = dataset(cfg, "train", limit)
    gram, rhs = np.zeros((7, 97, 97)), np.zeros((7, 97, 96))
    for start in range(0, len(data), 512):
        batch = [data[i] for i in range(start, min(start + 512, len(data)))]
        x = np.stack([b[0] for b in batch]).astype(np.float64)
        y = np.stack([b[1][-96:] for b in batch]).astype(np.float64)
        g, h = sufficient_statistics(x, y)
        gram += g
        rhs += h
        phase(out, "linear_fitting", windows=min(start + 512, len(data)), total=len(data))
    gram /= len(data)
    rhs /= len(data)
    weights = solve_hierarchical(gram, rhs, .001, .1)
    np.savez_compressed(directory / "training_statistics.npz", gram=gram, rhs=rhs)
    np.save(directory / "coefficients.npy", weights)
    atomic_json(receipt, dict(status="completed", train_windows=len(data), ridge=.001, sharing=.1,
        validation_used_for_this_fit=False, hyperparameters_fixed_from_historical_study=True,
        test_accessed=False, sha256=sha(directory / "coefficients.npy")))
    return weights


def make_artifact(folder, model, cfg, training_protocol, row):
    if folder.exists():
        raise FileExistsError('已有未完成产物，不自动覆盖或重测：' + str(folder))
    folder.mkdir(parents=True)
    atomic_torch(folder / "checkpoint.pth", model.state_dict())
    config = dict(vars(cfg))
    config.update(model=ADAPTERS[row['variant']], is_training=0)
    atomic_json(folder / "model_config.json", config)
    identity = dict(checkpoint_sha256=sha(folder/'checkpoint.pth'), model_config_sha256=sha(folder/'model_config.json'))
    sources = training_protocol['source_sha256']
    for name, digest in sources.items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('启动后源码发生改变：'+name)
    manifest = dict(kind=config['model'], selection_status=('smoke_validation_only' if training_protocol['options']['train_limit'] or training_protocol['options']['val_limit'] else 'frozen_for_test'), **identity,
        adapter_source_sha256=sources, seed=row['seed'], best_epoch=row['best_epoch'],
        neural_selection='神经预测完整验证均方误差；不按融合或测试重新选择轮次',
        combination='神经/线性各0.5；线性岭惩罚.001，共享惩罚.1，全部固定')
    atomic_json(folder/'manifest.json', manifest)
    atomic_json(folder/'freeze.json', dict(**identity, reason='完成两组训练及验证选择；固定等权融合，不再改变轮次、结构或超参数',
        validation_evidence=row, created=datetime.now(timezone.utc).isoformat()))
    restored = importlib.import_module('models.'+config['model']).Model(SimpleNamespace(**config))
    restored.load_state_dict(torch.load(folder/'checkpoint.pth', map_location='cpu', weights_only=True), strict=True)
    return restored


def write_results(out, report):
    if report['smoke']:
        path = out / 'results_zh.md'
    else:
        path = ROOT / 'docs/experiments/OscillatorRecurrentRank16_results_20260927.md'
    lines = ['# 完整递推测试收益组合：九十六步实验结果', '',
        '两组均从头训练；对照为原最佳融合的神经结构，两组均保留四尺度完整未来递推，候选仅增加最细秩16时间投影。两组已有趋势输入及归一化、偏置不衰减；各自与同一个重新拟合的线性组件固定等权融合。', '',
        '神经组件按验证均方误差选轮；没有按融合验证成绩或测试成绩重选轮次。线性岭惩罚为0.001，共享惩罚0.1。', '',
        '| 分组 | 完成轮数 | 所选轮次 | 神经验证均方误差 | 融合验证均方误差 | 融合验证平均绝对误差 | 测试均方误差 | 测试平均绝对误差 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in report['runs']:
        test = row.get('test', {})
        val = row['blend_validation']
        lines.append('| '+ ' | '.join([NAMES[row['variant']], str(row['trained_epochs']), str(row['best_epoch']),
            f"{row['selected_validation']['mse']:.8f}", f"{val['mse']:.8f}", f"{val['mae']:.8f}",
            f"{test['mse']:.8f}" if test else '未评估', f"{test['mae']:.8f}" if test else '未评估']) + ' |')
    lines += ['', '## 分变量、低频及幅值', '']
    for split, label in [('blend_validation', '融合验证'), ('test', '融合测试')]:
        lines += [f'### {label}', '', '| 分组 | 低四阶频率误差 | 幅值比 | 窗口水平误差 |', '| --- | --- | --- | --- |']
        for row in report['runs']:
            m = row.get(split)
            if m:
                lines.append('| '+NAMES[row['variant']]+' | '+' | '.join(f'{m[k]:.8f}' for k in ['low4_mse','centered_amplitude_ratio','window_offset_mse'])+' |')
        lines += ['', '| 变量 | 原方案 | 组合方案 |', '| --- | --- | --- |']
        for i, c in enumerate(['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']):
            values = [row.get(split, {}).get('per_channel_mse', [None]*7)[i] for row in report['runs']]
            lines.append('| `'+c+'` | '+' | '.join('未记录' if v is None else f'{v:.8f}' for v in values)+' |')
    lines += ['', '## 四个连续验证时段', '', '| 时段 | 原方案 | 组合方案 |', '| --- | --- | --- |']
    for i in range(4):
        lines.append('| 第'+str(i+1)+'时段 | '+' | '.join(f"{row['blend_validation_quarters'][i]['mse']:.8f}" for row in report['runs'])+' |')
    if len(report['runs']) == 2:
        c, n = report['runs']
        lines += ['', '## 改善幅度', '']
        for split, label in [('blend_validation', '完整验证'), ('test','完整测试')]:
            if split in c and split in n:
                for key, metric_name in [('mse','均方误差'), ('mae','平均绝对误差')]:
                    delta = n[split][key]-c[split][key]
                    reduction = -100*delta/c[split][key]
                    lines.append(f'- {label}{metric_name}：组合减对照为 {delta:+.8f}；相对降低 {reduction:+.3f}%。')
        if 'test' in n:
            historical = .31718265209843827
            lines.append(f"- 相对历史最佳融合测试均方误差 {historical:.8f}：组合差值为 {n['test']['mse']-historical:+.8f}。历史结果只作背景，主要依据本轮配对对照。")
    lines += ['', '## 证据范围', '',
        '- 随机种子仅2021，不宣称跨种子稳定；改动组合的收益不能简单相加，也不单独定位某个模块的因果贡献。',
        '- 验证与测试区间此前已被研究使用；本轮固定方案的测试不是全新独立留出集。',
        '- 候选共享参数初始化相同，四尺度递推公式相同，但低秩投影不同，初始预测不同。因子衰减不等于稠密矩阵衰减。',
        '- 移除最细尺度取消驱动累积方案；不加入自相关损失、直接解码器、原序列编码、特征对齐或尚无测试收益的融合损失训练；输入长度保持96。',
        '- 秩16的历史测试收益来自局部驱动背景；能否迁移到完整递推背景由本轮检验，不能提前宣称已证明有效。',
        '- 本实验是两阶段组合：神经模型从头训练，线性模型独立闭式拟合，最后固定等权融合；不是端到端联合优化融合损失。', '',
        '产物目录：`'+str(out.relative_to(ROOT))+'`。协议、每轮曲线、分支梯度、所选验证预测、冻结产物和测试凭证均保留。', '']
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path



def wait_for_dependency(out, directory):
    """排队等待既有编排结束；不停止、恢复或重启依赖实验。"""
    if directory is None:
        return
    directory = Path(directory).resolve()
    while True:
        report_path=directory/'report.json'
        if not report_path.exists():
            raise RuntimeError('依赖实验报告缺失：'+str(report_path))
        previous=json.loads(report_path.read_text())
        if previous.get('status') in ('failed','interrupted'):
            raise RuntimeError('依赖实验异常，排队实验不自动继续：'+str(previous.get('error')))
        active=False
        for name in ('pid.txt','training/pid.txt','control_test/pid.txt','blend_loss_test/pid.txt'):
            file=directory/name
            if not file.exists():continue
            proc=Path('/proc')/file.read_text().strip()
            try: command=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
            except (FileNotFoundError,ProcessLookupError):continue
            if str(directory) in command:active=True
        if previous.get('status')=='completed' and not active:
            atomic_json(out/'dependency_completion.json',dict(directory=str(directory),
                completed=datetime.now(timezone.utc).isoformat(),report_sha256=sha(report_path)))
            return
        if not active:
            raise RuntimeError('依赖实验未完成且进程已退出，不自动继续')
        phase(out,'waiting_dependency',dependency=str(directory),dependency_status=previous.get('status'))
        time.sleep(30)


def try_reuse_control_test(artifact, previous):
    """只复用同一权重、相同推理配置和未改变源码下的完整测试。"""
    previous=Path(previous)
    receipt=previous/'test_result.json'
    if not receipt.exists():return None
    evidence=json.loads(receipt.read_text())
    current_hash=sha(artifact/'checkpoint.pth')
    if current_hash!=sha(previous/'checkpoint.pth') or current_hash!=evidence['checkpoint_sha256']:
        return None
    a=json.loads((artifact/'model_config.json').read_text())
    b=json.loads((previous/'model_config.json').read_text())
    # 只有日志位置允许变化，其他全部配置逐项匹配。
    ignored=('log_dir','tensorboard_dir')
    if {k:v for k,v in a.items() if k not in ignored}!={k:v for k,v in b.items() if k not in ignored}:
        return None
    manifest=json.loads((previous/'manifest.json').read_text())
    if manifest['checkpoint_sha256']!=current_hash or manifest['model_config_sha256']!=sha(previous/'model_config.json'):
        raise RuntimeError('旧对照凭证身份不一致')
    for name,digest in manifest['adapter_source_sha256'].items():
        if not (ROOT/name).exists() or sha(ROOT/name)!=digest:
            return None
    result=json.loads(Path(evidence['report']).read_text())
    if not (result['status']=='completed' and result['split']=='test' and result['windows']==11425
            and result['checkpoint_sha256']==current_hash
            and result['model_config_sha256']==manifest['model_config_sha256']
            and result['source_hashes_verified']):
        raise RuntimeError('旧对照测试报告不完整')
    audit=dict(reused=True,previous_artifact=str(previous),report=evidence['report'],
        checkpoint_sha256=current_hash,full_config_equal_except=list(ignored),
        previous_source_hashes_verified=True,reason='权重字节哈希相同；全部配置除日志目录外一致；旧源码身份复核通过')
    atomic_json(artifact/'test_reuse_evidence.json',audit)
    atomic_json(artifact/'test_result.json',dict(checkpoint_sha256=current_hash,report=evidence['report'],reused=True))
    return audit


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--skip-test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wait-for', type=Path)
    opts = parser.parse_args()
    if (opts.train_limit or opts.val_limit) and not opts.skip_test:
        raise ValueError('冒烟不允许测试')
    out = opts.output.resolve()
    out.mkdir(parents=True, exist_ok=opts.resume)
    lock = (out/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    if opts.resume:
        saved = json.loads((out/'pipeline_options.json').read_text())
        for key in ('device','epochs','patience','train_limit','val_limit','skip_test'):
            if getattr(opts,key) != saved[key]:
                raise ValueError('恢复设置不一致：'+key)
    else:
        atomic_json(out/'pipeline_options.json', {k:str(v) if isinstance(v,Path) else v for k,v in vars(opts).items()})
    report = json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else dict(status='running', runs=[], smoke=bool(opts.train_limit or opts.val_limit), test_accessed=False)
    if report['status'] == 'completed':
        print('ALREADY COMPLETED',out); return
    atomic_json(out/'report.json', report)
    torch.set_num_threads(2)
    try:
        queue_sources=out/'queue_source_sha256.json'
        if not queue_sources.exists():
            files=[Path(__file__).relative_to(ROOT),Path('scripts/train_timemix_recurrent_rank16.py')]
            for directory in ('models','layers','data_provider','utils'):
                files.extend(Path(directory).rglob('*.py'))
            atomic_json(queue_sources,{str(f):sha(ROOT/f) for f in files})
        wait_for_dependency(out,opts.wait_for)
        for name,digest in json.loads(queue_sources.read_text()).items():
            if sha(ROOT/name)!=digest:raise RuntimeError('排队期间源码发生改变：'+name)
        training = out/'training'
        phase(out, 'paired_training')
        command = ['scripts/train_timemix_recurrent_rank16.py', '--device', opts.device]
        if training.exists():
            if not opts.resume: raise FileExistsError(training)
            if json.loads((training/'report.json').read_text())['status'] != 'completed':
                run(command+['--resume',str(training)])
        else:
            run(command+['--output',str(training),'--seeds','2021','--epochs',str(opts.epochs),'--patience',str(opts.patience),
                '--train-limit',str(opts.train_limit),'--val-limit',str(opts.val_limit)])
        tr = json.loads((training/'report.json').read_text())
        assert tr['status'] == 'completed' and len(tr['runs']) == 2
        if not report['smoke']:
            assert tr['train_windows'] == 34369 and tr['validation_windows'] == 11425
        tp = json.loads((training/'protocol.json').read_text())
        cfg = SimpleNamespace(**tp['model_args'])
        weights = train_linear(cfg, opts.train_limit, out)
        data = dataset(cfg, 'val', opts.val_limit)
        loader = DataLoader(data,batch_size=32,shuffle=False,num_workers=0)
        for row in tr['runs']:
            existing = next((v for v in report['runs'] if v['variant']==row['variant']), None)
            if existing: continue
            phase(out, 'blend_selected_validation', variant=row['variant'])
            neural = importlib.import_module('models.'+MODELS[row['variant']]).Model(cfg)
            neural.load_state_dict(torch.load(row['checkpoint'],map_location='cpu',weights_only=True),strict=True)
            model = LinearForecastBlend(neural, weights, neural_weight=.5).to(opts.device).eval()
            metric, pred, truth = evaluate(model, loader, opts.device, retain=True)
            cached = np.load(training/'2021'/(row['variant']+'_selected_validation.npz'))
            np.testing.assert_array_equal(truth, cached['truth'])
            xs = torch.tensor(np.stack([data[i][0] for i in range(min(2,len(data)))]),dtype=torch.float32,device=opts.device)
            with torch.no_grad():
                np.testing.assert_allclose(model.linear_forecast(xs).cpu().numpy(),predict(xs.cpu().double().numpy(),weights),rtol=3e-5,atol=3e-6)
                torch.testing.assert_close(model(xs),.5*(model.neural(xs)+model.linear_forecast(xs)),atol=0,rtol=0)
            entry = dict(row, blend_validation=metric,
                blend_validation_quarters=[score(pred[q].astype('float64'),truth[q].astype('float64')) for q in np.array_split(np.arange(len(truth)),4)])
            np.savez_compressed(out/(row['variant']+'_blend_validation.npz'),prediction=pred,truth=truth)
            artifact = out/(row['variant']+'_artifact')
            restored = make_artifact(artifact, model.cpu(), cfg, tp, entry)
            with torch.no_grad():
                torch.testing.assert_close(restored(xs.cpu()),model(xs.cpu()),atol=2e-6,rtol=1e-5)
            entry['artifact'] = str(artifact)
            report['runs'].append(entry)
            atomic_json(out/'report.json',report)
            del model, neural, restored
            if opts.device=='cuda': torch.cuda.empty_cache()
        # 所有训练和两个产物冻结均完成后，才进入测试阶段。
        if not opts.skip_test:
            for row in report['runs']:
                phase(out,'final_test',variant=row['variant'])
                artifact = Path(row['artifact'])
                receipt = artifact/'test_result.json'
                if row['variant']=='control' and not receipt.exists():
                    audit=try_reuse_control_test(artifact,ROOT/'tmp/lag_study_timemix_combined_20260927_seed2021/control_artifact')
                    if audit: row['test_reuse']=audit
                if not receipt.exists():
                    target = out/(row['variant']+'_test')
                    if target.exists(): raise RuntimeError('测试目录存在而无完成凭证，不自动重复测试：'+str(target))
                    run(['scripts/evaluate_frozen_timemix.py','--artifact',str(artifact),'--split','test','--device',opts.device,'--output',str(target)])
                evidence = json.loads(receipt.read_text())
                assert evidence['checkpoint_sha256'] == sha(artifact/'checkpoint.pth')
                result = json.loads(Path(evidence['report']).read_text())
                assert result['status']=='completed' and result['windows']==11425
                row['test'] = result['metrics']
                row['test_report'] = evidence['report']
                report['test_accessed'] = True
                atomic_json(out/'report.json', report)
        report['status'] = 'completed'
        report['result_document'] = str(write_results(out, report))
        atomic_json(out/'report.json', report)
        phase(out,'completed')
        if not report['smoke']:
            state_path = ROOT/'tmp/timemix_0308_goal_state.json'
            state = json.loads(state_path.read_text())
            state['recurrent_rank16_results'] = dict(status='completed',report=str(out/'report.json'),document=report['result_document'],test_accessed=report['test_accessed'])
            atomic_json(state_path,state)
        print('RECURRENT RANK16 EXPERIMENT COMPLETE',out,flush=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        atomic_json(out/'report.json',report)
        phase(out,'failed',error=repr(exc))
        raise


if __name__=='__main__':
    main()
