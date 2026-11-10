"""Paired old mean-patch control / fixed hidden readout skip; freeze both before one test each."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import copy
import fcntl
import html
import json
import math
from pathlib import Path
import shutil
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from models.OscillatorPDMStatePatch import Model as Control
from scripts.train_timemix_pdm_state_patch import initialize_pair as old_initialize_pair
from torch.utils.data import DataLoader, Subset
from utils.pdm_state_patch_skip_diagnostics import FeatureStatistics, gradient_probe
from models.OscillatorPDMStatePatchReadoutSkip import Model as Residual
from scripts import run_timemix_dual_encoder_experiment as shared
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts

ARMS = ('control', 'residual')
COUNTS = {'control': 136546, 'residual': 136546}
MODEL_NAMES = {'control': 'OscillatorPDMStatePatch', 'residual': 'OscillatorPDMStatePatchReadoutSkip'}
STATE_KEY = 'pdm_state_patch_readout_skip_results'
PROTOCOL = ROOT/'docs/experiments/PDMStatePatchReadoutSkip_protocol_20261013.json'
REPORT = ROOT/'docs/experiments/PDMStatePatchReadoutSkip_results_20261013.html'
# Reuse implementation utilities, with this study's own snapshot protocol and
# registration namespace. None of the completed study's files are edited.
shared.PROTOCOL, shared.STATE_KEY = PROTOCOL, STATE_KEY


def arm_config(cfg, arm):
    return SimpleNamespace(**{**vars(cfg), 'model': MODEL_NAMES[arm]})


def initialize_pair(cfg):
    control = old_initialize_pair(cfg)['patch_covariance']
    with torch.random.fork_rng(devices=[]):
        residual = Residual(arm_config(cfg, 'residual'))
    common = control.state_dict()
    state = residual.state_dict()
    state.update({name: value.clone() for name, value in common.items()})
    residual.load_state_dict(state, strict=True)
    assert all(torch.equal(value, residual.state_dict()[name]) for name, value in common.items())
    return dict(control=control, residual=residual)


def initialization_audit(models):
    base, residual = [models[arm] for arm in ARMS]
    common = base.state_dict()
    assert set(common) == set(residual.state_dict())
    assert all(torch.equal(value, residual.state_dict()[name]) for name, value in common.items())
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(base.parameters(), residual.parameters()))
    result = dict(parameter_counts=COUNTS, common_tensors_exact=True, additional_tensors=[],
                  initial_outputs_expected_equal=False, residual_coefficient=1., initialization='scratch')
    probe = torch.randn(2, 96, 7, generator=torch.Generator().manual_seed(101)).to(next(base.parameters()).device)
    for training in (False, True):
        outputs, rngs = [], []
        for model in models.values():
            model.train(training)
            with torch.random.fork_rng(devices=[probe.device.index or 0] if probe.is_cuda else []):
                shared.seed_all(314)
                with torch.no_grad():
                    outputs.append(model(probe))
                rngs.append(torch.cuda.get_rng_state() if probe.is_cuda else torch.get_rng_state())
        result['train' if training else 'eval'] = dict(
            max_prediction_difference=float((outputs[0]-outputs[1]).abs().max()),
            rng_equal=torch.equal(*rngs))
        assert result['train' if training else 'eval']['max_prediction_difference'] > 0
        assert result['train' if training else 'eval']['rng_equal']
    return result


OLD = ROOT/'tmp/lag_study_timemix_pdm_state_patch_20261011_seed2021_earlystop5'


def config(root_path):
    values = dict(shared.read(OLD/'training/protocol.json')['model_args'])
    values.update(root_path=str(root_path), model=MODEL_NAMES['control'],
                  log_dir=str(Path(root_path).parent/'logs'), tensorboard=False)
    return SimpleNamespace(**values)


def loader(data, epoch=None):
    if epoch is not None:
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(2021*1000+epoch)).tolist()
        data = Subset(data, order)
    return DataLoader(data, batch_size=32, shuffle=False, num_workers=0, drop_last=False)


shared.config = config
shared.loader = loader


def evaluate(model, data, device, folder, label, capture=False):
    model.eval()
    statistics = FeatureStatistics(model) if capture else None
    sums = dict(mse=0., mae=0., patch_covariance=0., prediction_covariance_norm2=0.,
                target_covariance_norm2=0., cross_covariance_inner_product=0.)
    squared = absolute = count = 0
    arrays = {}
    if capture:
        destination = folder/label
        destination.mkdir(exist_ok=False)
        for key, tail in [('prediction', (96, 7)), ('truth', (96, 7)), ('history', (96, 7)),
                          ('components', (96, 7, 8)), ('history_mean', (1, 7))]:
            arrays[key] = np.lib.format.open_memmap(destination/(key+'.npy'), mode='w+',
                dtype='float32', shape=(len(data), *tail))
    offset = 0
    try:
        with torch.no_grad():
            for batch_index, (x, target, _, _) in enumerate(loader(data)):
                x, target = x.float().to(device), target[:, -96:].float().to(device)
                components = model.predict_components(x)
                norm = model.normalize
                assert not norm.affine and not norm.non_norm and not norm.subtract_last
                prediction = norm(components[..., :4].sum(-1)+components[..., 4:].sum(-1), 'denorm')
                if not torch.isfinite(prediction).all():
                    raise RuntimeError('Nonfinite prediction')
                _, diagnostic = parts(prediction, target)
                if diagnostic['minimum_unclipped_distance'] < -1e-10:
                    raise RuntimeError('Negative kernel distance')
                n = len(x)
                error = prediction.double()-target.double()
                squared += float(error.square().sum()); absolute += float(error.abs().sum()); count += target.numel()
                for key in sums.keys()-{'mse', 'mae'}:
                    sums[key] += diagnostic[key]*n
                if capture:
                    values = dict(prediction=prediction, truth=target, history=x,
                        components=components*norm.stdev.unsqueeze(-1), history_mean=norm.mean)
                    for key, value in values.items():
                        arrays[key][offset:offset+n] = value.cpu().numpy()
                offset += n
                if batch_index % 50 == 0:
                    shared.phase(folder, label, windows=offset, total_windows=len(data))
    finally:
        if statistics:
            statistics.close()
    for array in arrays.values():
        array.flush()
    result = {key: value/offset for key, value in sums.items()}
    result.update(mse=squared/count, mae=absolute/count)
    if statistics:
        shared.atomic_json(folder/(label+'_features.json'), statistics.summary())
    return result


def verify_policy(last, train_windows, maximum):
    best = {a: math.inf for a in ARMS}; chosen = {a: 0 for a in ARMS}; stale = 0
    for epoch in range(1, last['epoch']+1):
        assert stale < 5
        improved = False
        rows = [row for row in last['curves'] if row['epoch'] == epoch]
        assert {row['arm'] for row in rows} == set(ARMS)
        for row in rows:
            assert row['learning_rate'] == shared.learning_rate(epoch)
            assert abs(row['train_loss']-row['train_mse']-row['train_d']) < 2e-6
            arm = row['arm']
            if row['val']['mse'] < best[arm]:
                best[arm], chosen[arm], improved = row['val']['mse'], epoch, True
        stale = 0 if improved else stale+1
    assert best == last['best'] and chosen == last['best_epoch'] and stale == last['stale']
    assert stale == 5 or last['epoch'] == maximum
    updates = last['epoch']*math.ceil(train_windows/32)
    assert last['updates_per_arm'] == updates
    for arm in ARMS:
        assert all(int(v['step']) == updates for v in last['optimizers'][arm]['state'].values())
    return dict(epochs=last['epoch'], selected_epochs=chosen, joint_stale=stale, updates_per_arm=updates)


def references():
    result = {}
    for rel in ('report.json', 'completion_audit.json', 'offline_comparison.json',
                'training/protocol.json', 'training/2021/initial.pth', 'training/2021/optimizer_groups.json',
                'patch_covariance_artifact/checkpoint.pth', 'patch_covariance_artifact/manifest.json',
                'test/patch_covariance/report.json'):
        path = OLD/rel
        result[str(path.relative_to(ROOT))] = shared.sha(path)
    assert shared.read(OLD/'status.json')['status'] == 'completed_verified'
    return result


def publish(out, report):
    page = '<!doctype html><meta charset="utf-8"><title>Mean-patch readout skip</title>'
    page += '<h1>旧均值聚合网络：原结构 / 32维读出直通配对实验</h1><p>两组均136546参数，共有scratch参数buffer精确相同；'
    page += '新组GELU(W_in h)+u，再经原32→4输出层；u是含未来PE的4点均值特征。初始函数不同，保留趋势调制。单seed复用val/test，不能唯一归因。</p>'
    page += '<pre>'+html.escape(json.dumps(report, ensure_ascii=False, indent=2))+'</pre>'
    with (ROOT/'tmp/pdm_state_patch_rope_registration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        temporary = REPORT.with_suffix('.html.tmp'); temporary.write_text(page); temporary.replace(REPORT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.4)
    parser.add_argument('--stop-after-epochs', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.epochs <= 30 or min(args.train_limit, args.val_limit) < 0 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error('Invalid experiment budget')
    smoke = bool(args.train_limit or args.val_limit)
    if smoke and not (args.train_limit >= 4 and args.val_limit >= 4):
        parser.error('Preflight requires both limits >=4')
    if args.resume and not smoke:
        parser.error('No automatic formal resume')
    out = args.output.resolve()
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    lock = (out/'pipeline.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    shared.atomic_json(out/'process.json', shared.process_record())
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    torch.set_num_threads(2)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    shared.seed_all(2021)
    try:
        if not args.resume:
            shared.phase(out, 'preparing')
            prefix = out/'train_val_data'; prefix.mkdir()
            with (ROOT/'dataset/ETT-small/ETTm1.csv').open('rb') as src, (prefix/'ETTm1.csv').open('wb') as dst:
                for _ in range(46081):
                    line = src.readline(); assert line; dst.write(line)
            cfg = shared.config(prefix)
            shared.atomic_json(out/'data_manifest.json', dict(train_val_csv_sha256=shared.sha(prefix/'ETTm1.csv'),
                full_csv_sha256=shared.sha(ROOT/'dataset/ETT-small/ETTm1.csv'), scaler_fit_rows=34560, train_val_rows=46080))
            shared.atomic_json(out/'reference_evidence.json', references())
            shared.atomic_json(out/'protocol.json', dict(model_config=vars(cfg), options={**vars(args), 'output': str(out)},
                formal_protocol=shared.read(PROTOCOL), smoke=smoke, started=shared.now()))
            manifest = shared.snapshot_sources(out)
            shutil.copyfile(ROOT/'scripts/monitor_timemix_pdm_state_patch_readout_skip.py', out/'monitor_check.py')
            for arm in ARMS:
                (out/arm/'epochs').mkdir(parents=True)
            if not smoke:
                shared.register(out, 'running', **shared.process_record(), source_count=len(manifest),
                    protocol=str(PROTOCOL), parameter_counts=COUNTS, arms=list(ARMS), launch_log=str(out)+'_launch.log')
        else:
            saved = shared.read(out/'protocol.json')
            assert saved['smoke'] and shared.read(out/'status.json')['status'] == 'controlled_stop'
            for key in ('epochs', 'train_limit', 'val_limit', 'device'):
                assert getattr(args, key) == saved['options'][key]
            cfg = SimpleNamespace(**saved['model_config'])
        shared.verify_sources(out)
        assert shared.sha(Path(cfg.root_path)/'ETTm1.csv') == shared.read(out/'data_manifest.json')['train_val_csv_sha256']
        train, val = shared.dataset(cfg, 'train', args.train_limit), shared.dataset(cfg, 'val', args.val_limit)
        models = {a: m.to(args.device) for a, m in initialize_pair(cfg).items()}
        assert {a: sum(p.numel() for p in m.parameters()) for a, m in models.items()} == COUNTS
        optimizers = {a: torch.optim.Adam(norm_bias_no_decay_groups(m, 1e-4), lr=.001) for a, m in models.items()}
        if not args.resume:
            initial_audit = initialization_audit(models)
            shared.atomic_json(out/'initialization_audit.json', initial_audit)
            initial = {a: m.state_dict() for a, m in models.items()}
            shared.atomic_torch(out/'initial.pt', initial)
            old_initial = torch.load(OLD/'training/2021/initial.pth', map_location='cpu', weights_only=True)['patch_covariance']
            assert all(torch.equal(v.cpu(), old_initial[k]) for k, v in initial['control'].items())
            optimizer_audit = {}
            for arm in ARMS:
                names = {id(p): n for n, p in models[arm].named_parameters()}
                optimizer_audit[arm] = [dict(weight_decay=g['weight_decay'], group_name=g['group_name'],
                    names=[names[id(p)] for p in g['params']]) for g in optimizers[arm].param_groups]
            shared.atomic_json(out/'optimizer_groups.json', optimizer_audit)
            old_decay = shared.read(OLD/'training/2021/optimizer_groups.json')['patch_covariance']
            for arm in ARMS:
                current_decay = {n: g['weight_decay'] for g in optimizer_audit[arm] for n in g['names']}
                assert current_decay == old_decay

            curves, best, chosen, stale, updates, start = [], {a: math.inf for a in ARMS}, {a: 0 for a in ARMS}, 0, 0, 1
        else:
            last = torch.load(out/'last.pt', map_location=args.device, weights_only=True)
            for arm in ARMS:
                models[arm].load_state_dict(last['models'][arm], strict=True)
                optimizers[arm].load_state_dict(last['optimizers'][arm])
            curves, best, chosen = last['curves'], last['best'], last['best_epoch']
            stale, updates, start = last['stale'], last['updates_per_arm'], last['epoch']+1
        for epoch in range(start, args.epochs+1):
            if stale >= 5:
                break
            for model in models.values():
                model.train()
            for opt in optimizers.values():
                for group in opt.param_groups:
                    group['lr'] = shared.learning_rate(epoch)
            totals = {a: np.zeros(3) for a in ARMS}; windows = 0
            for step, (x, target, _, _) in enumerate(shared.loader(train, epoch), 1):
                x, target = x.float().to(args.device), target[:, -96:].float().to(args.device)
                sampled = step in (1, 200, 400, 600, 800, 1000)
                losses = {}
                for arm in ARMS:
                    shared.seed_all(2021*1000000+epoch*10000+step-1)
                    model, opt = models[arm], optimizers[arm]
                    statistics = FeatureStatistics(model) if sampled else None
                    opt.zero_grad(set_to_none=True)
                    prediction = model(x)
                    mse = (prediction-target.detach()).square().mean()
                    distance, diagnostic = parts(prediction, target)
                    loss = mse+distance
                    assert torch.isfinite(loss)
                    assert diagnostic['minimum_unclipped_distance'] >= -1e-10
                    if sampled:
                        row = dict(epoch=epoch, step=step, arm=arm, scopes=gradient_probe(model, mse, distance),
                                   mse=float(mse.detach()), distance=float(distance.detach()), diagnostics=diagnostic)
                        row['features'] = statistics.summary()
                        statistics.close()
                        with (out/'gradients.jsonl').open('a') as file:
                            file.write(json.dumps(row)+'\n')
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                            raise RuntimeError('Missing/nonfinite gradient: '+arm+'/'+name)
                    opt.step()
                    totals[arm] += np.array([float(loss.detach()), float(mse.detach()), float(distance.detach())])*len(x)
                    losses[arm] = float(loss.detach())
                windows += len(x); updates += 1
                if step == 1 or step%50 == 0:
                    shared.phase(out, 'training', epoch=epoch, step=step, steps=math.ceil(len(train)/32),
                                 losses=losses, completed_epochs=epoch-1)
            improved = False
            for arm in ARMS:
                shared.phase(out, 'validation', arm=arm, epoch=epoch)
                validation = evaluate(models[arm], val, args.device, out/arm, 'validation')
                if validation['mse'] < best[arm]:
                    best[arm], chosen[arm], improved = validation['mse'], epoch, True
                    shared.atomic_torch(out/arm/'best.pt', models[arm].state_dict())
                curves.append(dict(epoch=epoch, arm=arm, train_loss=float(totals[arm][0]/windows),
                    train_mse=float(totals[arm][1]/windows), train_d=float(totals[arm][2]/windows),
                    val=validation, learning_rate=shared.learning_rate(epoch)))
                shared.atomic_torch(out/arm/'epochs'/f'epoch{epoch:02d}.pt', models[arm].state_dict())
            stale = 0 if improved else stale+1
            last = dict(epoch=epoch, models={a: m.state_dict() for a, m in models.items()},
                optimizers={a: o.state_dict() for a, o in optimizers.items()}, curves=curves,
                best=best, best_epoch=chosen, stale=stale, updates_per_arm=updates)
            shared.atomic_torch(out/'last.pt', last)
            shared.atomic_json(out/'progress.json', {k:v for k,v in last.items() if k not in ('models', 'optimizers')})
            if args.stop_after_epochs and epoch >= args.stop_after_epochs:
                shared.phase(out, 'controlled_stop', completed_epochs=epoch)
                return
        shared.phase(out, 'freezing', completed_epochs=last['epoch'])
        policy = verify_policy(last, len(train), args.epochs)
        shared.verify_sources(out)
        for rel, digest in shared.read(out/'reference_evidence.json').items():
            assert shared.sha(ROOT/rel) == digest, rel
        digests = {}
        for arm in ARMS:
            artifact = out/arm/'artifact'; artifact.mkdir()
            shutil.copyfile(out/arm/'best.pt', artifact/'checkpoint.pt')
            digests[arm] = shared.sha(artifact/'checkpoint.pt')
            selected = torch.load(artifact/'checkpoint.pt', map_location=args.device, weights_only=True)
            epoch_state = torch.load(out/arm/'epochs'/f'epoch{chosen[arm]:02d}.pt', map_location=args.device, weights_only=True)
            assert all(torch.equal(v, epoch_state[k]) for k, v in selected.items())
            models[arm].load_state_dict(selected, strict=True)
            shared.atomic_json(artifact/'model_config.json', vars(arm_config(cfg, arm)))
        shared.atomic_json(out/'freeze.json', dict(checkpoint_sha256=digests, policy=policy,
            selection='Both arms complete; own validation MSE only, no epoch0; test not opened', created=shared.now(), smoke=smoke))
        report = dict(status='preflight_verified' if smoke else 'validation_frozen', parameter_counts=COUNTS,
                      initialization=shared.read(out/'initialization_audit.json'), policy=policy, arms={})
        for arm in ARMS:
            shared.phase(out, 'capture_val', arm=arm)
            measured = evaluate(models[arm], val, args.device, out/arm, 'val', capture=True)
            assert abs(measured['mse']-best[arm]) < 1e-7
            shared.phase(out, 'validation_audit', arm=arm)
            report['arms'][arm] = dict(validation=shared.audit_capture(out/arm, 'val', measured))
        if not smoke:
            assert shared.sha(ROOT/'dataset/ETT-small/ETTm1.csv') == shared.read(out/'data_manifest.json')['full_csv_sha256']
            with (out/'test_started.json').open('x') as file:
                json.dump(dict(created=shared.now(), checkpoint_sha256=digests, reason='Both selected val artifacts frozen/audited'), file)
            test = shared.dataset(shared.config(ROOT/'dataset/ETT-small'), 'test')
            for arm in ARMS:
                shared.phase(out, 'test', arm=arm)
                with (out/arm/'test_started.json').open('x') as file:
                    json.dump(dict(checkpoint_sha256=digests[arm], created=shared.now()), file)
                measured = evaluate(models[arm], test, args.device, out/arm, 'test', capture=True)
                assert shared.sha(out/arm/'artifact/checkpoint.pt') == digests[arm]
                shared.atomic_json(out/arm/'test_receipt.json', dict(status='completed', windows=11425,
                    checkpoint_sha256=digests[arm], selected_epoch=chosen[arm], evaluations=1, created=shared.now()))
                shared.phase(out, 'test_audit', arm=arm)
                report['arms'][arm]['test'] = shared.audit_capture(out/arm, 'test', measured)
            report['status'] = 'completed_verified'
        shared.verify_sources(out)
        shared.atomic_json(out/'report.json', report)
        shared.atomic_json(out/'completion_audit.json', dict(status='verified', smoke=smoke, policy=policy,
            checkpoint_sha256=digests, test_windows_per_arm=0 if smoke else 11425,
            source_count=len(shared.read(out/'sources_manifest.json')), created=shared.now()))
        if not smoke:
            publish(out, report)
            shared.register(out, 'completed_verified', report=str(REPORT), policy=policy,
                test_metrics={a: {k: report['arms'][a]['test']['metrics'][k] for k in ('mse', 'mae')} for a in ARMS})
        shared.phase(out, report['status'], **policy)
    except BaseException as error:
        shared.atomic_json(out/'failure.json', dict(created=shared.now(), error=repr(error), traceback=traceback.format_exc()))
        shared.phase(out, 'failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
