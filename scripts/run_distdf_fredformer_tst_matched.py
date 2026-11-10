"""Official Fredformer+DistDF TST/OneCycle scheduler at the local fixed11 budget."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from data_provider.data_loader import Dataset_ETT_minute
from models.DistDFFredformer import Model
from utils.distdf_joint_wasserstein import distdf_loss

KEY = 'distdf_fredformer_tst_matched_results'
PROTOCOL = ROOT/'docs/experiments/DistDFFredformerTSTMatched_protocol_20261015.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def process_record():
    return dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                command=[v.decode() for v in Path('/proc/self/cmdline').read_bytes().split(b'\0') if v])


def register(out, status, **extra):
    with (ROOT/'tmp/pdm_state_patch_rope_registration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read(ROOT/'tmp/timemix_0308_goal_state.json')
        section = state.setdefault(KEY, {})
        prior = section.get('current_run', {})
        event = dict(status=status, event=status, output=str(out), updated=now(), **extra)
        section['current_run'] = {**(prior if prior.get('output') == str(out) else {}), **event}
        section['status'] = status
        section.setdefault('run_history', []).append(event)
        atomic(ROOT/'tmp/timemix_0308_goal_state.json', state)


def phase(out, status, **extra):
    record = dict(status=status, updated=now(), **extra)
    atomic(out/'status.json', record)
    print(json.dumps(record), flush=True)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def config(root):
    # Official ETTm1/96 structure and parser defaults, without changing dimensions.
    return SimpleNamespace(task_name='long_term_forecast', model='DistDFFredformer',
        root_path=str(root), data_path='ETTm1.csv', augmentation_ratio=0,
        seq_len=96, label_len=0, pred_len=96, enc_in=7, dec_in=7, c_out=7, features='M',
        e_layers=3, n_heads=16, d_model=24, d_ff=256, dropout=.2, fc_dropout=.2,
        head_dropout=0., patch_len=4, stride=4, padding_patch='end', individual=0,
        revin=1, affine=0, subtract_last=0, use_nys=0, ablation=0,
        cf_dim=128, cf_depth=2, cf_heads=8, cf_mlp=96, cf_head_dim=32,
        cf_drop=.2, mlp_hidden=64, mlp_drop=.3)


def optimizer_groups(model):
    protected = {id(p) for m in model.modules() for n, p in m.named_parameters(recurse=False)
                 if isinstance(m, nn.LayerNorm) or n == 'bias' or n.endswith('_bias')}
    return [dict(params=[p for p in model.parameters() if (id(p) in protected) == exempt],
                 weight_decay=0. if exempt else 1e-4,
                 group_name='norm_bias_no_decay' if exempt else 'weights')
            for exempt in (False, True)]


def make_scheduler(optimizer, epochs, steps):
    # Same official TST constructor and defaults; only the user-matched peak/horizon differ.
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, steps_per_epoch=steps, epochs=epochs,
        max_lr=[g['lr'] for g in optimizer.param_groups], pct_start=.4)


def dataset(cfg, split, limit=0):
    data = Dataset_ETT_minute(cfg, cfg.root_path, flag=split, size=[96, 0, 96],
                             features='M', data_path='ETTm1.csv', timeenc=1, freq='h')
    assert len(data) == (34369 if split == 'train' else 11425)
    if limit:
        data = Subset(data, np.unique(np.linspace(0, len(data)-1, min(limit, len(data)), dtype=int)).tolist())
    return data


def loader(data, epoch=None):
    if epoch is not None:
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(2021*1000+epoch)).tolist()
        data = Subset(data, order)
    return DataLoader(data, batch_size=32, num_workers=0, shuffle=False, drop_last=False)


def snapshot(out):
    paths = [p for directory in ('models', 'layers', 'utils', 'data_provider', 'exp', 'scripts', 'tests', 'third_party/DistDF')
             for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and (p.suffix == '.py' or directory == 'third_party/DistDF')]
    paths += [PROTOCOL, ROOT/'models/distdf_official/ADAPTATION.json']
    manifest = {}
    for p in sorted(set(paths)):
        rel = p.relative_to(ROOT)
        target = out/'source'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        manifest[str(rel)] = sha(p)
    atomic(out/'sources_manifest.json', manifest)


def verify_sources(out):
    for rel, digest in read(out/'sources_manifest.json').items():
        assert sha(ROOT/rel) == sha(out/'source'/rel) == digest, rel


def numpy_distdf(p, y, h):
    x, z = [np.concatenate((h, v), 1).astype(np.float64).transpose(2, 0, 1) for v in (p, y)]
    xm, zm = x.mean(1, keepdims=True), z.mean(1, keepdims=True)
    x, z = x-xm, z-zm
    eye = np.eye(x.shape[-1])[None]
    a = x.transpose(0, 2, 1) @ x / x.shape[1]+.005*eye
    b = z.transpose(0, 2, 1) @ z / z.shape[1]+.005*eye
    cov = np.trace(a, axis1=1, axis2=2)+np.trace(b, axis1=1, axis2=2)
    cov -= 2*np.sqrt(np.maximum((a*b.transpose(0, 2, 1)).sum((1, 2)), 1e-8))
    return float(np.sqrt(np.maximum(((xm-zm)**2).sum((1, 2))+.002*cov, 1e-8)).mean())


def evaluate(model, data, device, out, label, capture=False):
    model.eval()
    totals = np.zeros(3, dtype=np.float64)
    offset = 0
    arrays = {}
    if capture:
        destination = out/label
        destination.mkdir(exist_ok=False)
        arrays = {n: np.lib.format.open_memmap(destination/(n+'.npy'), mode='w+', dtype='float32',
                  shape=(len(data), 96, 7)) for n in ('prediction', 'truth', 'history')}
    with torch.no_grad():
        for index, (x, target, _, _) in enumerate(loader(data)):
            x, y = x.float().to(device), target[:, -96:].float().to(device)
            prediction = model(x)
            assert prediction.shape == y.shape and torch.isfinite(prediction).all()
            distance = distdf_loss(prediction, y, x)
            assert torch.isfinite(distance) and distance >= 0
            err = prediction.double()-y.double()
            totals += np.array([float(err.square().mean()), float(err.abs().mean()), float(distance)])*len(x)
            if capture:
                for name, value in [('prediction', prediction), ('truth', y), ('history', x)]:
                    arrays[name][offset:offset+len(x)] = value.cpu().numpy()
            offset += len(x)
            if index % 50 == 0:
                phase(out, label, windows=offset, total_windows=len(data))
    for array in arrays.values():
        array.flush()
    result = dict(zip(('mse', 'mae', 'distdf'), (totals/offset).tolist()))
    result['distdf_evaluation_batch_size'] = 32
    if capture:
        independent = np.zeros(3)
        for start in range(0, offset, 32):
            p, y, h = [np.asarray(arrays[n][start:start+32]) for n in ('prediction', 'truth', 'history')]
            assert all(np.isfinite(v).all() for v in (p, y, h))
            error = p.astype(np.float64)-y
            independent += np.array([np.mean(error**2), np.mean(np.abs(error)), numpy_distdf(p, y, h)])*len(p)
        independent /= offset
        assert max(abs(independent[:2]-np.array([result['mse'], result['mae']]))) < 1e-9
        assert abs(independent[2]-result['distdf']) < 2e-5*max(1., abs(independent[2]))
        atomic(destination/'report.json', dict(status='verified', windows=offset, metrics=result,
            independent_numpy=dict(zip(('mse', 'mae', 'distdf'), independent.tolist())),
            arrays_sha256={n: sha(destination/(n+'.npy')) for n in arrays}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--epochs', type=int, default=11)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--stop-after-epochs', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    smoke = bool(args.train_limit or args.val_limit)
    if smoke:
        assert min(args.train_limit, args.val_limit) >= 4 and 1 <= args.epochs <= 11
    else:
        assert args.epochs == 11 and not args.resume and not args.stop_after_epochs
    out = args.output.resolve()
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    lock = (out/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic(out/'process.json', process_record())
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.4)
    seed(2021)
    try:
        if not args.resume:
            phase(out, 'preparing')
            prefix = out/'train_val_data'
            prefix.mkdir()
            with (ROOT/'dataset/ETT-small/ETTm1.csv').open('rb') as src, (prefix/'ETTm1.csv').open('wb') as dst:
                for _ in range(46081):
                    line = src.readline()
                    assert line
                    dst.write(line)
            cfg = config(prefix)
            atomic(out/'protocol.json', dict(model_config=vars(cfg), options={**vars(args), 'output': str(out)},
                smoke=smoke, formal_protocol=read(PROTOCOL), started=now()))
            atomic(out/'data_manifest.json', dict(prefix_sha256=sha(prefix/'ETTm1.csv'),
                full_sha256=sha(ROOT/'dataset/ETT-small/ETTm1.csv'), scaler_fit_rows=34560, prefix_rows=46080))
            snapshot(out)
            shutil.copyfile(ROOT/'scripts/monitor_distdf_fredformer_tst_matched.py', out/'monitor_check.py')
            (out/'epochs').mkdir()
            if not smoke:
                register(out, 'running', **process_record(), protocol=str(PROTOCOL), launch_log=str(out)+'_launch.log')
        else:
            previous = read(out/'protocol.json')
            assert previous['smoke'] and read(out/'status.json')['status'] == 'controlled_stop'
            assert all(vars(args)[n] == previous['options'][n] for n in ('train_limit', 'val_limit', 'epochs', 'device'))
            cfg = SimpleNamespace(**previous['model_config'])
        verify_sources(out)
        assert sha(Path(cfg.root_path)/'ETTm1.csv') == read(out/'data_manifest.json')['prefix_sha256']
        train, val = dataset(cfg, 'train', args.train_limit), dataset(cfg, 'val', args.val_limit)
        model = Model(cfg).to(args.device)
        optimizer = torch.optim.Adam(optimizer_groups(model), lr=.001)
        scheduler = make_scheduler(optimizer, args.epochs, math.ceil(len(train)/32))
        names = {id(p): n for n, p in model.named_parameters()}
        if not args.resume:
            save(out/'initial.pt', model.state_dict())
            old_initial = torch.load(ROOT/'tmp/distdf_fredformer_matched_20261015_seed2021_fixed11/initial.pt', map_location=args.device, weights_only=True)
            assert len(old_initial) == len(model.state_dict()) == 72
            assert all(torch.equal(v, old_initial[n]) and v.data_ptr() != old_initial[n].data_ptr() for n,v in model.state_dict().items())
            del old_initial
            atomic(out/'optimizer_groups.json', [dict(group_name=g['group_name'], weight_decay=g['weight_decay'],
                names=[names[id(p)] for p in g['params']]) for g in optimizer.param_groups])
            atomic(out/'initialization_audit.json', dict(seed=2021, old_trained_weights_loaded=False,
                parameter_count=sum(p.numel() for p in model.parameters()),
                parameter_tensors=len(list(model.parameters())), states=len(model.state_dict()),
                backbone='official Fredformer ETTm1/96; structurally different from MAFirst'))
            curves, best, chosen, updates, start = [], math.inf, 0, 0, 1
        else:
            last = torch.load(out/'last.pt', map_location=args.device, weights_only=True)
            model.load_state_dict(last['model'])
            optimizer.load_state_dict(last['optimizer'])
            scheduler.load_state_dict(last['scheduler'])
            curves, best, chosen, updates, start = last['curves'], last['best'], last['best_epoch'], last['updates'], last['epoch']+1
        for epoch in range(start, args.epochs+1):
            lr = optimizer.param_groups[0]['lr']
            beta1_start = optimizer.param_groups[0]['betas'][0]
            model.train()
            totals, windows = np.zeros(3), 0
            for step, (x, target, _, _) in enumerate(loader(train, epoch), 1):
                seed(2021*1000000+epoch*10000+step-1)
                x, y = x.float().to(args.device), target[:, -96:].float().to(args.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                mse = (prediction-y.detach()).square().mean()
                distance = distdf_loss(prediction, y, x)
                loss = .5*mse+.5*distance
                assert torch.isfinite(loss) and torch.isfinite(prediction).all()
                if step in (1, 200, 400, 600, 800, 1000):
                    gm, gd = [torch.autograd.grad(v, prediction, retain_graph=True)[0].detach().double().flatten()
                              for v in (mse, distance)]
                    nm, nd = float(gm.norm()), float(gd.norm())
                    row = dict(epoch=epoch, step=step, mse=float(mse.detach()), distdf=float(distance.detach()),
                               mse_norm=nm, distdf_norm=nd, weighted_ratio=nd/max(nm, 1e-30),
                               cosine=float(gm@gd)/max(nm*nd, 1e-30))
                    with (out/'gradients.jsonl').open('a') as file:
                        file.write(json.dumps(row, allow_nan=False)+'\n')
                loss.backward()
                inactive = []
                for name, p in model.named_parameters():
                    if p.grad is None:
                        inactive.append(name)
                    elif not torch.isfinite(p.grad).all():
                        raise RuntimeError('Nonfinite gradient: '+name)
                # The official backbone defines unused ablation/output layers.
                if epoch == 1 and step == 1:
                    atomic(out/'inactive_parameters.json', inactive)
                else:
                    assert inactive == read(out/'inactive_parameters.json')
                used_lr = [g['lr'] for g in optimizer.param_groups]
                used_beta1 = [g['betas'][0] for g in optimizer.param_groups]
                optimizer.step()
                scheduler.step()
                with (out/'scheduler_trace.jsonl').open('a') as file:
                    file.write(json.dumps(dict(epoch=epoch, step=step, update=updates+1,
                        learning_rates=used_lr, beta1=used_beta1,
                        next_learning_rates=[g['lr'] for g in optimizer.param_groups],
                        next_beta1=[g['betas'][0] for g in optimizer.param_groups],
                        scheduler_last_epoch=scheduler.last_epoch), allow_nan=False)+'\n')
                assert all(torch.isfinite(p).all() for p in model.parameters())
                totals += np.array([float(loss.detach()), float(mse.detach()), float(distance.detach())])*len(x)
                windows += len(x)
                updates += 1
                if step == 1 or step % 50 == 0:
                    phase(out, 'training', epoch=epoch, step=step, completed_epochs=epoch-1, loss=float(loss.detach()))
            phase(out, 'validation', epoch=epoch)
            measured = evaluate(model, val, args.device, out, 'validation')
            if measured['mse'] < best:
                best, chosen = measured['mse'], epoch
                save(out/'best.pt', model.state_dict())
            row = dict(epoch=epoch, train_loss=float(totals[0]/windows), train_mse=float(totals[1]/windows),
                       train_distdf=float(totals[2]/windows), val=measured, learning_rate=lr, beta1_start=beta1_start,
                       next_learning_rate=optimizer.param_groups[0]['lr'], next_beta1=optimizer.param_groups[0]['betas'][0],
                       scheduler_last_epoch=scheduler.last_epoch)
            assert abs(row['train_loss']-.5*row['train_mse']-.5*row['train_distdf']) < 2e-6
            curves.append(row)
            save(out/'epochs'/f'epoch{epoch:02d}.pt', model.state_dict())
            last = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        curves=curves, best=best, best_epoch=chosen, updates=updates, scheduler=scheduler.state_dict())
            save(out/'last.pt', last)
            atomic(out/'progress.json', {k: v for k, v in last.items() if k not in ('model', 'optimizer', 'scheduler')})
            if args.stop_after_epochs and epoch >= args.stop_after_epochs:
                phase(out, 'controlled_stop', completed_epochs=epoch)
                return
        assert updates == args.epochs*math.ceil(len(train)/32)
        assert scheduler.last_epoch == updates
        assert len((out/'scheduler_trace.jsonl').read_text().splitlines()) == updates
        assert chosen == min(curves, key=lambda r: r['val']['mse'])['epoch']
        verify_sources(out)
        phase(out, 'freezing', completed_epochs=args.epochs)
        artifact = out/'artifact'
        artifact.mkdir()
        shutil.copyfile(out/'best.pt', artifact/'checkpoint.pt')
        digest = sha(artifact/'checkpoint.pt')
        atomic(artifact/'model_config.json', vars(cfg))
        selected = torch.load(artifact/'checkpoint.pt', map_location=args.device, weights_only=True)
        epoch_state = torch.load(out/'epochs'/f'epoch{chosen:02d}.pt', map_location=args.device, weights_only=True)
        assert all(torch.equal(v, epoch_state[n]) for n, v in selected.items())
        model.load_state_dict(selected)
        atomic(out/'freeze.json', dict(checkpoint_sha256=digest, selected_epoch=chosen, updates=updates,
                                     created=now(), test_opened=False))
        measured = evaluate(model, val, args.device, out, 'val', capture=True)
        assert abs(measured['mse']-best) < 1e-7
        report = dict(status='preflight_verified' if smoke else 'completed_verified', validation=measured,
                      selected_epoch=chosen, epochs=args.epochs, updates=updates)
        if not smoke:
            verify_sources(out)
            assert sha(ROOT/'dataset/ETT-small/ETTm1.csv') == read(out/'data_manifest.json')['full_sha256']
            with (out/'test_started.json').open('x') as file:
                json.dump(dict(checkpoint_sha256=digest, created=now()), file)
            test = dataset(config(ROOT/'dataset/ETT-small'), 'test')
            report['test'] = evaluate(model, test, args.device, out, 'test', capture=True)
            assert sha(artifact/'checkpoint.pt') == digest
            atomic(out/'test_receipt.json', dict(status='completed', windows=11425, evaluations=1,
                selected_epoch=chosen, checkpoint_sha256=digest, created=now()))
        verify_sources(out)
        atomic(out/'report.json', report)
        atomic(out/'completion_audit.json', dict(status='verified', smoke=smoke, test_evaluations=0 if smoke else 1,
                                               checkpoint_sha256=digest, epochs=args.epochs, updates=updates))
        if not smoke:
            register(out, 'completed_verified', metrics=report['test'], selected_epoch=chosen)
        phase(out, report['status'], completed_epochs=args.epochs)
    except BaseException as error:
        atomic(out/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), created=now()))
        phase(out, 'failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
