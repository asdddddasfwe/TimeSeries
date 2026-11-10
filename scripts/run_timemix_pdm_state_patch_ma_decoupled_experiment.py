"""Three raw-decomposition arms from the independent-MLP 0.318 architecture."""
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

from scripts.train_timemix_pdm_state_patch import initialize_pair as old_initialize_pair
from torch.utils.data import DataLoader, Subset
from utils.pdm_state_patch_ma_decoupled_diagnostics import FeatureStatistics, gradient_probe
from models.OscillatorPDMStatePatchSeparateMLP import Model as SeparateMLP
from models.OscillatorPDMStatePatchMAFirst import Model as PositionOnly
from models.OscillatorPDMStatePatchMAFirstFullyDecoupled import Model as FullyDecoupled
from models.OscillatorPDMStatePatchPreDecompSliding import Model as Sliding
from scripts import run_timemix_dual_encoder_experiment as shared
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts

ARMS = ('position_only', 'fully_decoupled', 'season_to_trend')
CLASSES = dict(position_only=PositionOnly, fully_decoupled=FullyDecoupled, season_to_trend=Sliding)
COUNTS = {}  # Filled from the frozen protocol; optimizer names audit every tensor.
MODEL_NAMES = {a: cls.__module__.split('.')[-1] for a, cls in CLASSES.items()}
STATE_KEY = 'pdm_state_patch_predecomp_results'
PROTOCOL = ROOT/'docs/experiments/PDMStatePatchMAFullyDecoupledComparisons_protocol_20261014.json'
REPORT = ROOT/'docs/experiments/PDMStatePatchMAFullyDecoupledComparisons_results_20261014.html'
# Reuse implementation utilities, with this study's own snapshot protocol and
# registration namespace. None of the completed study's files are edited.
shared.PROTOCOL, shared.STATE_KEY = PROTOCOL, STATE_KEY
if PROTOCOL.exists():
    COUNTS.update(json.loads(PROTOCOL.read_text())['parameters'])


def arm_config(cfg, arm):
    return SimpleNamespace(**{**vars(cfg), 'model': MODEL_NAMES[arm]})


def scratch_reference(cfg):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2021)
        original = old_initialize_pair(cfg)['patch_covariance']
        with torch.random.fork_rng(devices=[]):
            source = SeparateMLP(cfg)
        state = source.state_dict()
        state.update({n: v.clone() for n, v in original.state_dict().items()})
        source.load_state_dict(state, strict=True)
        source.clone_trend_mlps()
        return source


def initialize_pair(cfg):
    source = scratch_reference(cfg)
    result = {}
    for arm in ARMS:
        with torch.random.fork_rng(devices=[]):
            # Deterministic new attention parameters, isolated from old scratch.
            torch.manual_seed(20211014)
            model = CLASSES[arm](arm_config(cfg, arm))
        state = model.state_dict()
        state.update({n: v.clone() for n, v in source.state_dict().items()
                      if n in state and state[n].shape == v.shape})
        model.load_state_dict(state, strict=True)
        if hasattr(model,'clone_embeddings'):
            model.clone_embeddings(source)
        if hasattr(model, 'clone_new_heads'):
            model.clone_new_heads()
        result[arm] = model
    return result


def initialization_audit(models):
    cfg = config(ROOT/'dataset/ETT-small')
    source = scratch_reference(cfg)
    old = torch.load(OLD/'training/2021/initial.pth', map_location='cpu', weights_only=True)['patch_covariance']
    assert all(torch.equal(source.state_dict()[n], v) for n, v in old.items())
    result = dict(parameter_counts={a: sum(p.numel() for p in m.parameters()) for a,m in models.items()},
                  old_scratch_common_232_exact=True, old_trained_weights_loaded=False, first_arm_only_ma_linear_order_swapped=True,
                  initial_outputs_expected_equal=False, arms={})
    for arm, model in models.items():
        state = model.state_dict()
        common = [n for n,v in source.state_dict().items() if n in state and v.shape == state[n].shape]
        assert all(torch.equal(source.state_dict()[n], state[n].cpu()) for n in common)
        assert all(source.state_dict()[n].data_ptr() != state[n].data_ptr() for n in common)
        result['arms'][arm] = dict(common_states_exact=len(common),
            states=len(state), parameter_tensors=len(list(model.parameters())),
            added_states=sorted(set(state)-set(source.state_dict())),
            removed_states=sorted(set(source.state_dict())-set(state)))
    old_mlp_initial = torch.load(MLP_OLD/'initial.pt',map_location='cpu',weights_only=True)['separate_mlp']
    assert set(models['position_only'].state_dict()) == set(old_mlp_initial)
    assert all(torch.equal(v.cpu(),old_mlp_initial[n]) for n,v in models['position_only'].state_dict().items())
    assert result['parameter_counts']['position_only'] == 140738
    result['first_arm_all_236_states_exact_old0318_initial'] = True
    split=models['fully_decoupled']
    for encoder,embedding,old_encoder in zip(split.seasonals,split.trend_embeddings,source.seasonals):
        assert encoder.value_embedding.bias is None
        assert torch.equal(embedding.weight.cpu(),old_encoder.value_embedding.weight)
        assert torch.equal(embedding.bias.cpu(),old_encoder.value_embedding.bias)
        assert embedding.weight.data_ptr()!=encoder.value_embedding.weight.data_ptr()
    result['second_arm_independent_embedding_and_removed_condition']=True
    assert not any('.trend_condition.' in n or '.season_condition.' in n for n,_ in split.named_parameters())
    assert result['parameter_counts']['fully_decoupled']==136770
    assert set(models['position_only'].state_dict())-set(split.state_dict())==({f'seasonals.{i}.value_embedding.bias' for i in range(4)} | {f'seasonals.{i}.decoder.trend_condition.weight' for i in range(4)})
    assert set(split.state_dict())-set(models['position_only'].state_dict())=={f'trend_embeddings.{i}.{key}' for i in range(4) for key in ('weight','bias')}
    probe = torch.randn(2,96,7,generator=torch.Generator().manual_seed(101))
    device = next(models['position_only'].parameters()).device
    probe = probe.to(device)
    for training in (False, True):
        outputs = {}
        for arm, model in models.items():
            model.train(training)
            with torch.random.fork_rng(devices=[device.index or 0] if probe.is_cuda else []):
                shared.seed_all(314)
                with torch.no_grad(): outputs[arm] = model(probe)
                assert torch.isfinite(outputs[arm]).all()
        assert torch.equal(outputs['position_only'],outputs['fully_decoupled'])
        result['train' if training else 'eval'] = dict(
            max_difference_vs_position_only={a: float((p-outputs['position_only']).abs().max()) for a,p in outputs.items()},
            equality_required=False, reason='Arms1/2 are equal at fresh zero-cross-condition initialization; arm2 independently embeds both branches and removes the cross edge. Arm3 retains its different patch/attention/recurrent design.')
    return result


OLD = ROOT/'tmp/lag_study_timemix_pdm_state_patch_20261011_seed2021_earlystop5'
MLP_OLD = ROOT/'tmp/lag_study_timemix_pdm_state_patch_separate_mlp_20261014_seed2021_earlystop5'


def config(root_path):
    values = dict(shared.read(OLD/'training/protocol.json')['model_args'])
    values.update(root_path=str(root_path), model=MODEL_NAMES['position_only'],
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
    for rel in ('initial.pt','optimizer_groups.json','report.json','completion_audit.json','sources_manifest.json',
                'separate_mlp/artifact/checkpoint.pt','separate_mlp/artifact/model_config.json','separate_mlp/test_receipt.json'):
        path = MLP_OLD/rel
        result[str(path.relative_to(ROOT))] = shared.sha(path)
    assert shared.read(MLP_OLD/'status.json')['status'] == 'completed_verified'
    assert shared.read(OLD/'status.json')['status'] == 'completed_verified'
    return result


def publish(out, report):
    page = '<!doctype html><meta charset="utf-8"><title>Raw MA before embedding</title>'
    page += '<h1>原0.318 MA前移＋完全解耦＋滑动趋势递推</h1><p>第一组仅MA/Linear换序140738。第二组在第一组基础上独立季节/趋势Linear(1,32)，并移除四尺度全部trend→season调制，136770（新增128投影参数/移除4096条件参数）。两路原独立PDM MLP、尺度混合、预测头保留，原FFT/rowcolumn/PE不变，最终相加，结构无跨路可训练连接但total loss共用。第三组保留原季节FFT patch双轴/趋势sliding attention+A24/双向历史条件235628。fresh训练、MSE+D、single seed，无唯一因果或统计独立声明。</p>'
    page += '<pre>'+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre>'
    with (ROOT/'tmp/pdm_state_patch_rope_registration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        temporary=REPORT.with_suffix('.html.tmp');temporary.write_text(page);temporary.replace(REPORT)


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
            shutil.copyfile(ROOT/'scripts/monitor_timemix_pdm_state_patch_ma_decoupled.py', out/'monitor_check.py')
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
            optimizer_audit = {}
            old_decay = shared.read(OLD/'training/2021/optimizer_groups.json')['patch_covariance']
            for arm in ARMS:
                names = {id(p): n for n, p in models[arm].named_parameters()}
                optimizer_audit[arm] = [dict(weight_decay=g['weight_decay'], group_name=g['group_name'],
                    names=[names[id(p)] for p in g['params']]) for g in optimizers[arm].param_groups]
                current_decay = {n:g['weight_decay'] for g in optimizer_audit[arm] for n in g['names']}
                assert set(current_decay) == set(names.values())
                assert all(current_decay[n] == old_decay[n] for n in set(current_decay)&set(old_decay))
                for n in current_decay:
                    if '.trend_out_cross_layer.' in n:
                        assert current_decay[n] == current_decay[n.replace('.trend_out_cross_layer.','.out_cross_layer.')]
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
            selection='All three arms complete; own validation MSE only, no epoch0; test not opened', created=shared.now(), smoke=smoke))
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
                json.dump(dict(created=shared.now(), checkpoint_sha256=digests, reason='All three selected val artifacts frozen/audited'), file)
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
