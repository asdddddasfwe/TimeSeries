"""Single scratch dual-encoder experiment: train -> freeze -> one full test.

Limited runs are preflight only and never open test. Resume is deliberately
restricted to controlled preflight stops; formal failures require user review.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import argparse
import fcntl
import hashlib
import html
import json
import math
import random
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_provider.data_loader import Dataset_ETT_minute
from models.OscillatorDualEncoderTimeMixer import Model
from scripts.audit_pdm_patch_covariance import numpy_covariance_parts
from scripts.diagnose_timemix_linear_reference import score
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts

STATE_KEY = 'dual_encoder_timemixer_results'
PROTOCOL = ROOT / 'docs/experiments/OscillatorDualEncoderTimeMixer_protocol_20261013.json'
REPORT_PATH = ROOT / 'docs/experiments/OscillatorDualEncoderTimeMixer_results_20261013.html'
CHANNELS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def phase(out, status, **extra):
    row = dict(status=status, updated=now(), **extra)
    atomic_json(out / 'status.json', row)
    print(json.dumps(row), flush=True)


def process_record():
    return dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                command=[v.decode() for v in Path('/proc/self/cmdline').read_bytes().split(b'\0') if v])


def register(out, status, **extra):
    with (ROOT / 'tmp/pdm_state_patch_rope_registration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT / 'tmp/timemix_0308_goal_state.json'
        state = read(path)
        section = state.setdefault(STATE_KEY, {})
        previous = section.get('current_run', {})
        event = dict(event=status, status=status, updated=now(), output=str(out), **extra)
        section['current_run'] = {**(previous if previous.get('output') == str(out) else {}), **event}
        section['status'] = status
        section.setdefault('run_history', []).append(event)
        atomic_json(path, state)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config(root_path):
    return SimpleNamespace(task_name='long_term_forecast', model='OscillatorDualEncoderTimeMixer',
        seq_len=96, pred_len=96, label_len=0, enc_in=7, c_out=7, features='M', moving_avg=25,
        d_model=32, d_ff=64, e_layers=1, dropout=.2, top_k=5, ssm_modes=8,
        trend_patch_length=4, trend_patch_stride=2, trend_patch_heads=4,
        down_sampling_layers=3, down_sampling_window=2, down_sampling_method='avg',
        channel_independence=1, use_norm=1, decomp_method='moving_avg',
        augmentation_ratio=0, root_path=str(root_path), data_path='ETTm1.csv')


def dataset(cfg, split, limit=0):
    data = Dataset_ETT_minute(cfg, cfg.root_path, flag=split, size=[96, 0, 96],
                              features='M', data_path='ETTm1.csv', timeenc=1, freq='h')
    expected = 34369 if split == 'train' else 11425
    assert len(data) == expected, (split, len(data))
    if limit:
        ids = np.unique(np.linspace(0, len(data)-1, min(limit, len(data)), dtype=int)).tolist()
        return Subset(data, ids)
    return data


def loader(data, epoch=None):
    generator = torch.Generator().manual_seed(2021 if epoch is None else 2021+epoch)
    return DataLoader(data, batch_size=32, shuffle=epoch is not None, generator=generator,
                      num_workers=0, drop_last=False)


def learning_rate(epoch):
    # The planned cosine horizon remains 30 even for a short controlled preflight.
    return .001 * .5 * (1 + math.cos(math.pi * (epoch-1) / 30))


def snapshot_sources(out):
    files = [path for directory in ('models', 'layers', 'utils', 'data_provider', 'exp', 'scripts', 'tests')
             for path in (ROOT/directory).rglob('*.py') if '__pycache__' not in path.parts]
    files += [ROOT/'run.py', PROTOCOL]
    manifest = {}
    for path in sorted(set(files)):
        rel = path.relative_to(ROOT)
        destination = out/'source'/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        manifest[str(rel)] = sha(path)
    atomic_json(out/'sources_manifest.json', manifest)
    return manifest


def verify_sources(out):
    for rel, digest in read(out/'sources_manifest.json').items():
        assert sha(ROOT/rel) == sha(out/'source'/rel) == digest, rel


def reference_evidence():
    paths = [ROOT/'tmp/lag_study_timemix_pdm_state_patch_20261011_seed2021_earlystop5/report.json',
             ROOT/'tmp/lag_study_timemix_pdm_state_patch_20261011_seed2021_earlystop5/completion_audit.json',
             ROOT/'tmp/lag_study_timemix_pdm_state_patch_fusion_20261013_seed2021_earlystop5/user_stop_record.json']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def gradient_probe(model, mse, distance):
    named = list(model.named_parameters())
    gm = torch.autograd.grad(mse, [p for _, p in named], retain_graph=True, allow_unused=True)
    gd = torch.autograd.grad(distance, [p for _, p in named], retain_graph=True, allow_unused=True)
    scopes = {'all': '', 'season_recurrence': 'season_oscillators.',
              'trend_recurrence': 'trend_oscillators.', 'season_attention': 'season_encoders.',
              'trend_attention': 'trend_encoders.', 'pdm': 'pdm.', 'fmm': 'predict_layers.'}
    result = {}
    for label, prefix in scopes.items():
        mm = dd = dot = 0.
        for (name, _), a, b in zip(named, gm, gd):
            if not name.startswith(prefix):
                continue
            if a is not None:
                assert torch.isfinite(a).all()
                mm += float(a.detach().double().square().sum())
            if b is not None:
                assert torch.isfinite(b).all()
                dd += float(b.detach().double().square().sum())
            if a is not None and b is not None:
                dot += float((a.detach().double()*b.detach().double()).sum())
        result[label] = dict(mse_norm=math.sqrt(mm), d_norm=math.sqrt(dd),
                             weighted_ratio=math.sqrt(dd)/max(math.sqrt(mm), 1e-30),
                             cosine=dot/max(math.sqrt(mm*dd), 1e-30))
    return result


def evaluate(model, data, device, out, label, capture=False):
    model.eval()
    sums = dict(mse=0., mae=0., patch_covariance=0., prediction_covariance_norm2=0.,
                target_covariance_norm2=0., cross_covariance_inner_product=0.)
    arrays = {}
    if capture:
        destination = out/label
        destination.mkdir(exist_ok=False)
        for key, tail in [('prediction', (96, 7)), ('truth', (96, 7)), ('history', (96, 7)),
                          ('components', (96, 7, 4)), ('history_mean', (1, 7))]:
            arrays[key] = np.lib.format.open_memmap(destination/(key+'.npy'), mode='w+',
                dtype='float32', shape=(len(data), *tail))
    offset = 0
    with torch.no_grad():
        for batch_index, (x, target, _, _) in enumerate(loader(data)):
            x, target = x.float().to(device), target[:, -96:].float().to(device)
            components = model.predict_components(x)
            norm = model.normalizers[0]
            prediction = norm(components.sum(-1), 'denorm')
            if not torch.isfinite(prediction).all():
                raise RuntimeError('Nonfinite prediction')
            _, diagnostic = parts(prediction, target)
            n = len(x)
            error = prediction.double()-target.double()
            sums['mse'] += float(error.square().mean())*n
            sums['mae'] += float(error.abs().mean())*n
            for key in sums.keys()-{'mse', 'mae'}:
                sums[key] += diagnostic[key]*n
            if capture:
                values = dict(prediction=prediction, truth=target, history=x,
                              components=components*norm.stdev.unsqueeze(-1), history_mean=norm.mean)
                for key, value in values.items():
                    arrays[key][offset:offset+n] = value.cpu().numpy()
            offset += n
            if batch_index % 50 == 0:
                phase(out, label, windows=offset, total_windows=len(data))
    for array in arrays.values():
        array.flush()
    return {key: value/offset for key, value in sums.items()}


def offline_summary(p, y):
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    result = score(p, y)
    amplitude = float(((p.std(1)-y.std(1))**2).mean())
    result['amplitude_error_mse'] = amplitude
    result['shape_alignment_mse'] = result['mse']-result['window_offset_mse']-amplitude
    dp, dy = np.diff(p, axis=1), np.diff(y, axis=1)
    boundary = (np.arange(95)+1) % 4 == 0
    result['increments'] = {}
    for name, mask in [('boundary', boundary), ('interior', ~boundary)]:
        result['increments'][name] = dict(count=int(mask.sum()),
            error_mse=float(((dp[:, mask]-dy[:, mask])**2).mean()),
            prediction_energy=float((dp[:, mask]**2).mean()), truth_energy=float((dy[:, mask]**2).mean()))
    return result


def audit_capture(out, split, measured):
    folder = out/split
    p, y = [np.load(folder/(name+'.npy'), mmap_mode='r') for name in ('prediction', 'truth')]
    components = np.load(folder/'components.npy', mmap_mode='r')
    mean = np.load(folder/'history_mean.npy', mmap_mode='r')
    maximum = 0.
    for i in range(0, len(p), 128):
        c, m = components[i:i+128].astype(float), mean[i:i+128].astype(float)
        assert np.isfinite(c).all() and np.isfinite(m).all()
        maximum = max(maximum, float(abs(c.sum(-1)+m-p[i:i+128]).max()))
    assert maximum < 2e-5
    relation = numpy_covariance_parts(p, y)
    for key, value in relation.items():
        assert abs(value-measured[key]) < 1e-6, key
    metric = offline_summary(p, y)
    for key in ('mse', 'mae'):
        assert abs(metric[key]-measured[key]) < 1e-7
    result = dict(status='verified', windows=len(p), metrics=metric, covariance=relation,
        reconstruction_max_abs=maximum,
        per_channel={name: offline_summary(p[:, :, j:j+1], y[:, :, j:j+1]) for j, name in enumerate(CHANNELS)},
        time_quarters=[offline_summary(p[ids], y[ids]) for ids in np.array_split(np.arange(len(p)), 4)],
        horizon_quarters=[dict(mse=float(((p[:, j:j+24].astype(float)-y[:, j:j+24])**2).mean()),
                               mae=float(abs(p[:, j:j+24].astype(float)-y[:, j:j+24]).mean())) for j in range(0, 96, 24)],
        files={file.name: sha(file) for file in folder.glob('*.npy')})
    atomic_json(folder/'report.json', result)
    return result


def verify_policy(out, last, train_windows):
    best, selected, stale = math.inf, 0, 0
    for row in last['curves']:
        assert stale < 5
        assert row['learning_rate'] == learning_rate(row['epoch'])
        assert abs(row['train_loss']-row['train_mse']-row['train_d']) < 2e-6
        if row['val']['mse'] < best:
            best, selected, stale = row['val']['mse'], row['epoch'], 0
        else:
            stale += 1
    assert best == last['best'] and selected == last['best_epoch'] and stale == last['stale']
    assert last['updates'] == last['epoch']*math.ceil(train_windows/32)
    for value in last['optimizer']['state'].values():
        assert int(value['step']) == last['updates']
    return dict(epochs=last['epoch'], selected_epoch=selected, stale=stale, updates=last['updates'])


def publish(out, report):
    with (ROOT/'tmp/pdm_state_patch_rope_registration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        content = '<!doctype html><meta charset="utf-8"><title>Dual Encoder PDM FMM</title>'
        content += '<h1>季节FFT / 趋势patch → 双分支振荡历史编码 → PDM → FMM</h1>'
        content += '<p>单组单seed；199633参数。旧mean核参考test MSE 0.3198694221 / MAE 0.3566592859。'
        content += '非等参数、非独立验证，结构多处改变；不能唯一归因或声称跨seed显著性。</p>'
        content += '<pre>'+html.escape(json.dumps(report, ensure_ascii=False, indent=2))+'</pre>'
        temporary = REPORT_PATH.with_suffix('.html.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(REPORT_PATH)


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
    if not 1 <= args.epochs <= 30 or min(args.train_limit, args.val_limit) < 0:
        parser.error('Invalid training budget')
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error('Invalid GPU fraction')
    smoke = bool(args.train_limit or args.val_limit)
    if smoke and not (args.train_limit >= 4 and args.val_limit >= 4):
        parser.error('Preflight requires both limits >=4')
    if args.resume and not smoke:
        parser.error('Formal resume is disabled; report failures without automatically retrying')
    out = args.output.resolve()
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    lock = (out/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(out/'process.json', process_record())
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    torch.set_num_threads(2)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    seed_all(2021)
    try:
        if not args.resume:
            phase(out, 'preparing')
            prefix = out/'train_val_data'
            prefix.mkdir()
            full = ROOT/'dataset/ETT-small/ETTm1.csv'
            with full.open('rb') as source, (prefix/'ETTm1.csv').open('wb') as dest:
                for _ in range(46081):
                    line = source.readline()
                    assert line
                    dest.write(line)
            cfg = config(prefix)
            atomic_json(out/'data_manifest.json', dict(train_val_csv_sha256=sha(prefix/'ETTm1.csv'),
                full_csv_sha256=sha(full), scaler_fit_rows=34560, train_val_rows=46080))
            atomic_json(out/'reference_evidence.json', reference_evidence())
            atomic_json(out/'protocol.json', dict(model_config=vars(cfg), options={**vars(args), 'output': str(out)},
                formal_protocol=read(PROTOCOL), smoke=smoke, started=now()))
            manifest = snapshot_sources(out)
            shutil.copyfile(ROOT/'scripts/monitor_timemix_dual_encoder.py', out/'monitor_check.py')
            if not smoke:
                register(out, 'running', **process_record(), source_count=len(manifest),
                         protocol=str(PROTOCOL), launch_log=str(out)+'_launch.log')
        else:
            saved = read(out/'protocol.json')
            assert saved['smoke'] and read(out/'status.json')['status'] == 'controlled_stop'
            for key in ('epochs', 'train_limit', 'val_limit', 'device'):
                assert getattr(args, key) == saved['options'][key]
            cfg = SimpleNamespace(**saved['model_config'])
        verify_sources(out)
        assert sha(Path(cfg.root_path)/'ETTm1.csv') == read(out/'data_manifest.json')['train_val_csv_sha256']
        train, val = dataset(cfg, 'train', args.train_limit), dataset(cfg, 'val', args.val_limit)
        model = Model(cfg).to(args.device)
        assert sum(p.numel() for p in model.parameters()) == 199633
        optimizer = torch.optim.Adam(norm_bias_no_decay_groups(model, 1e-4), lr=.001)
        names = {id(p): name for name, p in model.named_parameters()}
        if not args.resume:
            atomic_torch(out/'initial.pt', model.state_dict())
            atomic_json(out/'optimizer_groups.json', [dict(weight_decay=g['weight_decay'],
                group_name=g['group_name'], names=[names[id(p)] for p in g['params']]) for g in optimizer.param_groups])
            curves, best, best_epoch, stale, updates, start = [], math.inf, 0, 0, 0, 1
            (out/'epochs').mkdir()
        else:
            last = torch.load(out/'last.pt', map_location=args.device, weights_only=True)
            model.load_state_dict(last['model'], strict=True)
            optimizer.load_state_dict(last['optimizer'])
            curves, best, best_epoch = last['curves'], last['best'], last['best_epoch']
            stale, updates, start = last['stale'], last['updates'], last['epoch']+1
        for epoch in range(start, args.epochs+1):
            if stale >= 5:
                break
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(epoch)
            total = np.zeros(3)
            windows = 0
            model.train()
            for step, (x, target, _, _) in enumerate(loader(train, epoch), 1):
                seed_all(2021 + epoch*10000 + step)
                x, target = x.float().to(args.device), target[:, -96:].float().to(args.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                mse = (prediction-target.detach()).square().mean()
                distance, diagnostic = parts(prediction, target)
                loss = mse+distance
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                if step in (1, 200, 400, 600, 800, 1000):
                    probe = gradient_probe(model, mse, distance)
                    with (out/'gradients.jsonl').open('a') as file:
                        file.write(json.dumps(dict(epoch=epoch, step=step, scopes=probe))+'\n')
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise RuntimeError('Missing/nonfinite gradient: '+name)
                optimizer.step()
                updates += 1
                total += np.array([float(loss.detach()), float(mse.detach()), float(distance.detach())])*len(x)
                windows += len(x)
                if step == 1 or step % 50 == 0:
                    phase(out, 'training', epoch=epoch, step=step, steps=math.ceil(len(train)/32),
                          loss=float(loss.detach()), completed_epochs=epoch-1)
            validation = evaluate(model, val, args.device, out, 'validation')
            improved = validation['mse'] < best
            if improved:
                best, best_epoch, stale = validation['mse'], epoch, 0
                atomic_torch(out/'best.pt', model.state_dict())
            else:
                stale += 1
            row = dict(epoch=epoch, train_loss=float(total[0]/windows), train_mse=float(total[1]/windows),
                       train_d=float(total[2]/windows), val=validation, learning_rate=learning_rate(epoch))
            curves.append(row)
            last = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        curves=curves, best=best, best_epoch=best_epoch, stale=stale, updates=updates)
            atomic_torch(out/'epochs'/f'epoch{epoch:02d}.pt', model.state_dict())
            atomic_torch(out/'last.pt', last)
            atomic_json(out/'progress.json', dict(epoch=epoch, curves=curves, best=best,
                        best_epoch=best_epoch, stale=stale, updates=updates))
            if args.stop_after_epochs and epoch >= args.stop_after_epochs:
                phase(out, 'controlled_stop', completed_epochs=epoch)
                return
        phase(out, 'freezing', completed_epochs=last['epoch'])
        policy = verify_policy(out, last, len(train))
        verify_sources(out)
        for rel, digest in read(out/'reference_evidence.json').items():
            assert sha(ROOT/rel) == digest, rel
        artifact = out/'artifact'
        artifact.mkdir()
        shutil.copyfile(out/'best.pt', artifact/'checkpoint.pt')
        selected = torch.load(artifact/'checkpoint.pt', map_location=args.device, weights_only=True)
        epoch_state = torch.load(out/'epochs'/f'epoch{best_epoch:02d}.pt', map_location=args.device, weights_only=True)
        assert all(torch.equal(selected[k], epoch_state[k]) for k in selected)
        model.load_state_dict(selected, strict=True)
        digest = sha(artifact/'checkpoint.pt')
        atomic_json(artifact/'model_config.json', vars(cfg))
        atomic_json(out/'freeze.json', dict(checkpoint_sha256=digest, policy=policy,
            selection='Own full validation MSE; no epoch0; test not opened', created=now(), smoke=smoke))
        measured = evaluate(model, val, args.device, out, 'val', capture=True)
        assert abs(measured['mse']-best) < 1e-7
        phase(out, 'validation_audit')
        validation_audit = audit_capture(out, 'val', measured)
        report = dict(status='preflight_verified' if smoke else 'validation_frozen',
                      parameter_count=199633, policy=policy, validation=validation_audit)
        if not smoke:
            assert sha(ROOT/'dataset/ETT-small/ETTm1.csv') == read(out/'data_manifest.json')['full_csv_sha256']
            # Exclusive creation prevents accidental repeat testing on rerun.
            with (out/'test_started.json').open('x') as file:
                json.dump(dict(created=now(), checkpoint_sha256=digest), file)
            test_cfg = config(ROOT/'dataset/ETT-small')
            test = dataset(test_cfg, 'test')
            assert len(test) == 11425
            measured = evaluate(model, test, args.device, out, 'test', capture=True)
            assert sha(artifact/'checkpoint.pt') == digest
            atomic_json(out/'test_receipt.json', dict(status='completed', windows=11425,
                checkpoint_sha256=digest, selected_epoch=best_epoch, created=now(), evaluations=1))
            phase(out, 'test_audit')
            report['test'] = audit_capture(out, 'test', measured)
            report['status'] = 'completed_verified'
        verify_sources(out)
        atomic_json(out/'report.json', report)
        atomic_json(out/'completion_audit.json', dict(status='verified', created=now(), smoke=smoke,
            source_count=len(read(out/'sources_manifest.json')), checkpoint_sha256=digest,
            policy=policy, test_windows=0 if smoke else 11425))
        if not smoke:
            publish(out, report)
            register(out, 'completed_verified', report=str(REPORT_PATH), policy=policy,
                     test_mse=report['test']['metrics']['mse'], test_mae=report['test']['metrics']['mae'])
        phase(out, report['status'], **policy)
    except BaseException as error:
        atomic_json(out/'failure.json', dict(created=now(), error=repr(error), traceback=traceback.format_exc()))
        phase(out, 'failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
