"""完整配对训练：原融合神经组件与局部驱动加秩16组合；训练阶段不访问测试集。"""
import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data_provider.data_loader import Dataset_ETT_minute
from models.OscillatorTimeMix import Model as Baseline
from models.OscillatorLowRankLocalTimeMix import Model as Candidate
from utils.timemix_decay_study import parameter_groups as decay_groups, sample_response_std
from utils.retained_timemix_helpers import setup_hooks, trace, summarize_tensor
from scripts.smoke_season_trend_mixer import framework_args
from utils.retained_timemix_helpers import metrics
from utils.logger import experiment_logging, start_experiment, log_metrics


ARMS = ('control', 'combined')
VARIANTS = dict(control=Baseline, combined=Candidate)


def parameter_groups(model, arm):
    return decay_groups(model, 'adam_norm_bias')


def build_optimizer(groups, arm):
    return torch.optim.Adam(groups, lr=.001)


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


def evaluate(model, loader, device, retain=False):
    model.eval()
    squared, count, predictions, targets = 0., 0, [], []
    with torch.no_grad():
        for x, y, _, _ in loader:
            y = y[:, -96:].float()
            p = model(x.float().to(device)).cpu()
            if not torch.isfinite(p).all():
                raise RuntimeError('Nonfinite evaluation prediction')
            squared += float((p.double()-y.double()).square().sum())
            count += y.numel()
            if retain:
                predictions.append(p.numpy())
                targets.append(y.numpy())
    if not retain:
        return squared / count
    p, y = np.concatenate(predictions), np.concatenate(targets)
    result = metrics(p.astype('float64'), y.astype('float64'))
    result['low4_mse'] = result['mse'] * result['low_frequency_error_share']['4']
    result['per_channel_mse'] = np.mean((p.astype('float64')-y)**2, axis=(0, 1)).tolist()
    return result, p, y


def publish_report(out, report):
    atomic_json(out/'report.json', report)
    lines = ['# 有效改动组合：完整配对训练', '', '状态：'+report['status'], '',
             '| 种子 | 分组 | 所选轮次 | 验证均方误差 |', '| --- | --- | --- | --- |']
    names = dict(control='原融合神经组件', combined='局部驱动与秩16组合')
    for row in report['runs']:
        lines.append(f"| {row['seed']} | {names[row['variant']]} | {row['best_epoch']} | {row['validation_mse']:.8f} |")
    lines += ['', '两组均从头训练，按神经预测验证均方误差选择已完成轮次；线性融合由外层流程完成。']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2021])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--train-limit', type=int, default=0, help='0 means all windows; nonzero is smoke only')
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--skip-test', action='store_true', default=True, help='Always validation-only; retained for smoke compatibility')
    parser.add_argument('--stop-after-epochs', type=int, default=0, help='Controlled checkpoint/resume verification only')
    opts = parser.parse_args()
    invocation_stop = opts.stop_after_epochs
    if opts.resume:
        out = opts.resume.resolve()
        protocol = json.loads((out/'protocol.json').read_text())
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
        out = (opts.output or ROOT/'tmp'/('lag_study_timemix_combined_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
        out.mkdir(parents=True, exist_ok=False)
        cfg = framework_args(['--task_name','long_term_forecast','--is_training','1',
            '--model','OscillatorTimeMix','--model_id',out.name,
            '--data','ETTm1','--root_path',str(ROOT/'dataset/ETT-small'),'--data_path','ETTm1.csv',
            '--features','M','--freq','h','--seq_len','96','--label_len','0','--pred_len','96',
            '--enc_in','7','--dec_in','7','--c_out','7','--moving_avg','25',
            '--d_model','32','--d_ff','64','--e_layers','2','--top_k','5','--dropout','.2',
            '--down_sampling_layers','3','--down_sampling_window','2','--down_sampling_method','avg',
            '--channel_independence','1','--use_norm','1','--ssm_modes','8','--ssm_dynamics_weight_decay','0',
            '--batch_size','32','--num_workers','0','--optimizer','adam','--learning_rate','.001',
            '--weight_decay','.0001','--lradj','cosine','--train_epochs',str(opts.epochs),
            '--patience',str(opts.patience),'--log_dir',str(out/'logs'),'--tensorboard_dir',str(out/'runs')])
        cfg.seed = opts.seeds[0]
        cfg.use_gpu = opts.device == 'cuda'
        files = [Path(__file__).relative_to(ROOT), Path('run.py')]
        for directory in ['models','layers','data_provider','utils']:
            files.extend(sorted(Path(directory).rglob('*.py')))
        files.extend([Path('scripts/smoke_season_trend_mixer.py'), Path('utils/retained_timemix_helpers.py'), Path('scripts/run_timemix_combined_experiment.py'), Path('utils/hierarchical_linear_helpers.py'), Path('scripts/diagnose_timemix_linear_reference.py'), Path('scripts/evaluate_frozen_timemix.py')])
        hashes = {}
        for relative in files:
            destination = out/'source'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, destination)
            hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
        options = {k:(str(v) if isinstance(v, Path) else v) for k,v in vars(opts).items()}
        protocol = dict(options=options, model_args=vars(cfg), source_sha256=hashes,
            initialization='从头随机初始化；共享参数和缓冲区逐项相同；候选最细投影由同一随机矩阵截断SVD初始化。初始预测不同。',
            factor='组合最细局部驱动与秩16投影。两组均保留趋势输入、相同Adam归一化/偏置和动力学豁免，训练后固定等权融合相同的训练集拟合线性组件。',
            selection='Full official validation; best MSE among completed epochs. Random epoch0 is not a selection candidate.',
            early_stopping='Joint paired patience: stop seed after neither arm has improved validation for 10 consecutive epochs (or configured patience). Both arms get equal epochs and optimizer updates.',
            pairing='相同批次与逐步dropout种子；FP32，Adam .001，普通衰减.0001，cosine。control=OscillatorTimeMix，combined=OscillatorLowRankLocalTimeMix。无预训练初始化。',
            test='本训练入口不访问test；外层流程在两组验证选择完成并冻结后各测试最终融合一次。',
            caveat='Validation was used in prior model research; this is not a new external dataset.',
            torch_version=torch.__version__, created=datetime.now().isoformat())
        atomic_json(out/'protocol.json', protocol)
    torch.set_num_threads(2)
    if opts.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.30)
    lock = (out/'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    report = json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else dict(status='training', runs=[])
    if report['status'] == 'completed':
        print('ALREADY COMPLETED', out)
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
                base = Baseline(cfg)
                models = {name: VARIANTS[name](cfg) for name in ARMS}
                models['control'].load_state_dict(base.state_dict(), strict=True)
                models['combined'].initialize_from_control(base)
                atomic_json(folder/'initialization_audit.json', dict(
                    shared_parameters_equal=True, rank=16,
                    parameter_counts={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},
                    initialization="共享初值逐项相同；最细投影来自同一随机矩阵的截断奇异值分解；初始预测不同"))
                if not (folder/'initial.pth').exists():
                    atomic_torch(folder/'initial.pth', base.state_dict())
                del base
                models = {name:model.to(opts.device) for name,model in models.items()}
                groups = {name:parameter_groups(model,name) for name,model in models.items()}
                optimizers = {name:build_optimizer(group,name) for name,group in groups.items()}
                audit = {}
                decay_by_id = {}
                for name,model in models.items():
                    decay_by_id[name] = {id(p):g['weight_decay'] for g in groups[name] for p in g['params']}
                    audit[name] = {key:decay_by_id[name][id(p)] for key,p in model.named_parameters()}
                atomic_json(folder/'optimizer_groups.json',audit)
                logger.info('COMBINED STUDY: local drive plus rank16, identical optimizer rules. Full audit: %s', folder/'optimizer_groups.json')
                probe_ids=np.unique(np.linspace(0,len(train)-1,min(16,len(train)),dtype=int))
                probe_x=torch.tensor(np.stack([train[int(i)][0] for i in probe_ids]),dtype=torch.float32,device=opts.device)
                branch_traces = json.loads((folder/'branch_traces.json').read_text()) if (folder/'branch_traces.json').exists() else []
                best_states = {}
                completed, stale, best, best_epoch, curves = 0, 0, {k:float('inf') for k in models}, {}, []
                if (folder/'last.pt').exists():
                    last = torch.load(folder/'last.pt', map_location=opts.device, weights_only=True)
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
                        reference = models[ARMS[0]](x)
                        for model in models.values():
                            if not torch.isfinite(model(x)).all(): raise RuntimeError('Nonfinite initialization')
                for epoch in range(completed+1, opts.epochs+1):
                    if stale >= opts.patience:
                        break
                    epoch_seed=seed*1000+epoch
                    order=torch.randperm(len(train),generator=torch.Generator().manual_seed(epoch_seed)).tolist()
                    loader=DataLoader(Subset(train,order),batch_size=32,shuffle=False,num_workers=0,drop_last=False)
                    losses={k:0. for k in models}
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
                                loss=(model(x)-y).square().mean()
                            finally:
                                for handle in handles: handle.remove()
                            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                            loss.backward()
                            grads=[p.grad for p in model.parameters() if p.requires_grad]
                            if any(g is None for g in grads):raise RuntimeError('Missing gradient')
                            if not torch.stack([torch.isfinite(g).all() for g in grads]).all():raise RuntimeError('Nonfinite gradient')
                            if record_branch:
                                branch_row=trace(model,captured,step+1,loss,decay_by_id[name])
                                branch_row.update(epoch=epoch,variant=name)
                                branch_traces.append(branch_row)
                                atomic_json(folder/'branch_traces.json',branch_traces)
                                log_metrics({'coarse_output_std':branch_row['scales'][3]['activations']['output']['std'],
                                             'coarse_drive_norm':branch_row['scales'][3]['parameters']['decoder.drive_norm.weight']['norm']},
                                            step=(epoch-1)*len(loader)+step+1,prefix=f'branches/{name}/seed{seed}')
                            optimizer.step()
                            losses[name]+=float(loss.detach())*len(x)
                        if step%50==0:
                            atomic_json(out/'status.json',dict(status='training',seed=seed,epoch=epoch,step=step+1,steps=len(loader),updated=datetime.now().isoformat()))
                            logger.info('PROGRESS seed=%d epoch=%d/%d paired_step=%d/%d',seed,epoch,opts.epochs,step+1,len(loader))
                    improved=False
                    atomic_json(out/'status.json',dict(status='validation',seed=seed,epoch=epoch,updated=datetime.now().isoformat()))
                    for name,model in models.items():
                        score=evaluate(model,val_loader,opts.device)
                        row=dict(seed=seed,variant=name,epoch=epoch,train_mse=losses[name]/len(train),validation_mse=score)
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
                        row['coarse_probe_std']=branch_probe['scale3']['std']
                        row['coarse_sample_response_std']=branch_probe['scale3']['sample_response_std']
                        curves.append(row)
                        log_metrics({k:v for k,v in row.items() if k not in ('seed','variant')},step=epoch,prefix=f'{name}/seed{seed}')
                        if score<best[name]:
                            best[name],best_epoch[name]=score,epoch
                            improved=True
                            best_states[name] = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                            atomic_torch(folder/(name+'_best.pth'),best_states[name])
                        for group in optimizers[name].param_groups:
                            group['lr']=.001*.5*(1+math.cos(math.pi*epoch/opts.epochs))
                        logger.info('EPOCH seed=%d %s epoch=%d train=%.7f val=%.7f best_epoch=%d',seed,name,epoch,row['train_mse'],score,best_epoch[name])
                    stale=0 if improved else stale+1
                    atomic_torch(folder/'last.pt',dict(models={k:m.state_dict() for k,m in models.items()},
                        optimizers={k:o.state_dict() for k,o in optimizers.items()},best_models=best_states,epoch=epoch,stale=stale,best=best,best_epoch=best_epoch,curves=curves))
                    atomic_json(folder/'progress.json',dict(epoch=epoch,stale=stale,best=best,best_epoch=best_epoch,curves=curves,paired_epoch_seconds=time.monotonic()-start))
                    if invocation_stop and epoch>=invocation_stop:
                        atomic_json(out/'status.json',dict(status='checkpointed_verification_stop',seed=seed,epoch=epoch))
                        print('CHECKPOINTED VERIFICATION STOP',out)
                        return
                for name in models:
                    report['runs']=[v for v in report['runs'] if not(v['seed']==seed and v['variant']==name)]
                    report['runs'].append(dict(seed=seed,variant=name,best_epoch=best_epoch[name],validation_mse=best[name],
                        trained_epochs=curves[-1]['epoch'],checkpoint=str(folder/(name+'_best.pth'))))
                publish_report(out,report)
                del models,optimizers
                if opts.device=='cuda':torch.cuda.empty_cache()
            # Full validation diagnostics for selected weights, without test access.
            report['status']='selected_validation'
            publish_report(out,report)
            for row in report['runs']:
                if 'selected_validation' in row: continue
                model=VARIANTS[row['variant']](cfg).to(opts.device).eval()
                model.load_state_dict(torch.load(row['checkpoint'],map_location=opts.device,weights_only=True),strict=True)
                atomic_json(out/'status.json',dict(status='selected_validation',seed=row['seed'],variant=row['variant'],updated=datetime.now().isoformat()))
                row['selected_validation'],prediction,truth=evaluate(model,val_loader,opts.device,retain=True)
                row['validation_quarters'] = []
                for indices in np.array_split(np.arange(len(truth)), 4):
                    metric = metrics(prediction[indices].astype('float64'), truth[indices].astype('float64'))
                    metric['low4_mse'] = metric['mse'] * metric['low_frequency_error_share']['4']
                    row['validation_quarters'].append(metric)
                np.savez_compressed(out/str(row['seed'])/(row['variant']+'_selected_validation.npz'), prediction=prediction, truth=truth)
                publish_report(out,report)
                del model
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


if __name__=='__main__':
    main()
