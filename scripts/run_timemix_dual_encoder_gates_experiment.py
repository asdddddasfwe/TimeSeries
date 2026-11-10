"""Paired ungated / forget+write history gates; freeze both before one test each."""
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

from models.OscillatorDualEncoderTimeMixer import Model as Ungated
from models.OscillatorDualEncoderGatedTimeMixer import Model as Gated
from scripts import run_timemix_dual_encoder_experiment as shared
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts

ARMS = ('ungated', 'gated')
COUNTS = {'ungated': 199633, 'gated': 203857}
MODEL_NAMES = {'ungated': 'OscillatorDualEncoderTimeMixer', 'gated': 'OscillatorDualEncoderGatedTimeMixer'}
STATE_KEY = 'dual_encoder_gates_results'
PROTOCOL = ROOT/'docs/experiments/OscillatorDualEncoderGates_protocol_20261013.json'
REPORT = ROOT/'docs/experiments/OscillatorDualEncoderGates_results_20261013.html'
# Reuse implementation utilities, with this study's own snapshot protocol and
# registration namespace. None of the completed study's files are edited.
shared.PROTOCOL, shared.STATE_KEY = PROTOCOL, STATE_KEY


def arm_config(cfg, arm):
    return SimpleNamespace(**{**vars(cfg), 'model': MODEL_NAMES[arm], 'history_gate_initial': .99})


def initialize_pair(cfg):
    control = Ungated(arm_config(cfg, 'ungated'))
    with torch.random.fork_rng(devices=[]):
        gated = Gated(arm_config(cfg, 'gated'))
    common = control.state_dict()
    state = gated.state_dict()
    state.update({name: value.clone() for name, value in common.items()})
    gated.load_state_dict(state, strict=True)
    assert all(torch.equal(value, gated.state_dict()[name]) for name, value in common.items())
    return dict(ungated=control, gated=gated)


def initialization_audit(models):
    base, gate = [models[arm] for arm in ARMS]
    common = base.state_dict()
    additional = sorted(set(gate.state_dict())-set(common))
    assert len(additional) == 32 and all('.forget_gate.' in k or '.write_gate.' in k for k in additional)
    result = dict(parameter_counts=COUNTS, common_tensors_exact=True, additional_gate_tensors=additional,
                  initial_outputs_expected_equal=False, gate_initial=.99)
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


def gradient_probe(model, mse, distance):
    named = list(model.named_parameters())
    gm = torch.autograd.grad(mse, [p for _, p in named], retain_graph=True, allow_unused=True)
    gd = torch.autograd.grad(distance, [p for _, p in named], retain_graph=True, allow_unused=True)
    scopes = {'all': lambda n: True, 'season_recurrence': lambda n: n.startswith('season_oscillators.'),
              'trend_recurrence': lambda n: n.startswith('trend_oscillators.'),
              'trend_attention': lambda n: n.startswith('trend_encoders.'),
              'pdm': lambda n: n.startswith('pdm.'), 'fmm': lambda n: n.startswith('predict_layers.'),
              'forget_gate': lambda n: '.forget_gate.' in n, 'write_gate': lambda n: '.write_gate.' in n}
    result = {}
    for label, accept in scopes.items():
        mm = dd = dot = 0.
        count = 0
        for (name, _), a, b in zip(named, gm, gd):
            if not accept(name):
                continue
            count += 1
            if a is not None:
                assert torch.isfinite(a).all()
                mm += float(a.detach().double().square().sum())
            if b is not None:
                assert torch.isfinite(b).all()
                dd += float(b.detach().double().square().sum())
            if a is not None and b is not None:
                dot += float((a.detach().double()*b.detach().double()).sum())
        result[label] = dict(parameter_tensors=count, mse_norm=math.sqrt(mm), d_norm=math.sqrt(dd),
            weighted_ratio=math.sqrt(dd)/max(math.sqrt(mm), 1e-30), cosine=dot/max(math.sqrt(mm*dd), 1e-30))
    return result


def gate_tensors(model):
    return {f'{branch}/{i}': module.last_gate_tensors for branch, modules in
            [('season', model.season_oscillators), ('trend', model.trend_oscillators)]
            for i, module in enumerate(modules)}


def accumulate_gates(totals, label, values):
    for name, tensor in values.items():
        v = tensor.double()
        assert torch.isfinite(v).all() and (v >= 0).all() and (v <= 1+1e-6).all()
        stats = torch.stack((v.sum(), v.square().sum(), v.min(), v.max(),
                             (v < .01).sum().double(), (v > .99).sum().double())).cpu().tolist()
        row = totals.setdefault(label+'/'+name, dict(count=0, sum=0., square_sum=0., minimum=1., maximum=0., low=0, high=0))
        row['count'] += v.numel()
        row['sum'] += stats[0]; row['square_sum'] += stats[1]
        row['minimum'] = min(row['minimum'], stats[2]); row['maximum'] = max(row['maximum'], stats[3])
        row['low'] += int(stats[4]); row['high'] += int(stats[5])


def summarize_gates(totals):
    return {label: dict(count=r['count'], mean=r['sum']/r['count'],
        std=math.sqrt(max(0., r['square_sum']/r['count']-(r['sum']/r['count'])**2)),
        minimum=r['minimum'], maximum=r['maximum'], fraction_below_01=r['low']/r['count'],
        fraction_above_99=r['high']/r['count']) for label, r in totals.items()}


def evaluate(model, data, device, folder, label, capture=False):
    if not isinstance(model, Gated) or not capture:
        return shared.evaluate(model, data, device, folder, label, capture)
    totals, handles = {}, []
    model.enable_gate_statistics(True)
    for branch, modules in [('season', model.season_oscillators), ('trend', model.trend_oscillators)]:
        for i, module in enumerate(modules):
            def hook(module, args, output, key=f'{branch}/{i}'):
                accumulate_gates(totals, key, module.last_gate_tensors)
            handles.append(module.register_forward_hook(hook))
    try:
        result = shared.evaluate(model, data, device, folder, label, capture)
    finally:
        for handle in handles:
            handle.remove()
        model.enable_gate_statistics(False)
    shared.atomic_json(folder/(label+'_gates.json'), summarize_gates(totals))
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
    result = shared.reference_evidence()
    old = ROOT/'tmp/lag_study_timemix_dual_encoder_20261013_seed2021_earlystop5'
    assert shared.read(old/'completion_audit.json')['status'] == 'verified'
    for rel in ('initial.pt', 'report.json', 'offline_final_review.json', 'completion_audit.json',
                'test_receipt.json', 'artifact/checkpoint.pt', 'val/report.json', 'test/report.json'):
        path = old/rel; result[str(path.relative_to(ROOT))] = shared.sha(path)
    return result


def publish(out, report):
    page = '<!doctype html><meta charset="utf-8"><title>Forget and Write Gates</title>'
    page += '<h1>原历史递推 / 遗忘与写入门配对实验</h1><p>公共scratch参数精确相同；两门初始0.99，初始函数不同。'
    page += '199633/203857参数，单seed、复用val/test、不等参数，不能唯一归因或声称显著性。</p>'
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
            shutil.copyfile(ROOT/'scripts/monitor_timemix_dual_encoder_gates.py', out/'monitor_check.py')
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
            old_initial = torch.load(ROOT/'tmp/lag_study_timemix_dual_encoder_20261013_seed2021_earlystop5/initial.pt',
                                     map_location='cpu', weights_only=True)
            assert all(torch.equal(v.cpu(), old_initial[k]) for k, v in initial['ungated'].items())
            optimizer_audit = {}
            for arm in ARMS:
                names = {id(p): n for n, p in models[arm].named_parameters()}
                optimizer_audit[arm] = [dict(weight_decay=g['weight_decay'], group_name=g['group_name'],
                    names=[names[id(p)] for p in g['params']]) for g in optimizers[arm].param_groups]
            shared.atomic_json(out/'optimizer_groups.json', optimizer_audit)
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
                    shared.seed_all(2021+epoch*10000+step)
                    model, opt = models[arm], optimizers[arm]
                    if arm == 'gated':
                        model.enable_gate_statistics(sampled)
                    opt.zero_grad(set_to_none=True)
                    prediction = model(x)
                    mse = (prediction-target.detach()).square().mean()
                    distance, diagnostic = parts(prediction, target)
                    loss = mse+distance
                    assert torch.isfinite(loss)
                    if sampled:
                        row = dict(epoch=epoch, step=step, arm=arm, scopes=gradient_probe(model, mse, distance))
                        if arm == 'gated':
                            values = {}
                            for label, tensors in gate_tensors(model).items():
                                accumulate_gates(values, label, tensors)
                            row['gates'] = summarize_gates(values)
                        with (out/'gradients.jsonl').open('a') as file:
                            file.write(json.dumps(row)+'\n')
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                            raise RuntimeError('Missing/nonfinite gradient: '+arm+'/'+name)
                    opt.step()
                    totals[arm] += np.array([float(loss.detach()), float(mse.detach()), float(distance.detach())])*len(x)
                    losses[arm] = float(loss.detach())
                    if arm == 'gated':
                        model.enable_gate_statistics(False)
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
