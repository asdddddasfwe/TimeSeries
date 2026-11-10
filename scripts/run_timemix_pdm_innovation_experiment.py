"""Two architecture arms; single seed, validation-only freeze and offline audit."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
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
import torch
from scripts.train_timemix_pdm_innovation import atomic_json, ARMS
from scripts.run_timemix_pdm_coarse_ma_experiment import audit_predictions
from scripts.evaluate_frozen_timemix import verify_artifact


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase(out, status, **extra):
    atomic_json(out/'status.json', dict(status=status, updated=datetime.now(timezone.utc).isoformat(), **extra))
    print('PHASE', status, extra, flush=True)


def make_artifact(folder, cfg, protocol, row):
    config = dict(row['model_config'], is_training=0)
    for name, digest in protocol['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Source changed since training: '+name)
    restored = importlib.import_module('models.' + config['model']).Model(SimpleNamespace(**config))
    restored.load_state_dict(torch.load(row['checkpoint'], map_location='cpu', weights_only=True), strict=True)
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(row['checkpoint'], folder/'checkpoint.pth')
    atomic_json(folder/'model_config.json', config)
    identity = dict(checkpoint_sha256=sha(folder/'checkpoint.pth'), model_config_sha256=sha(folder/'model_config.json'))
    smoke = bool(protocol['options']['train_limit'] or protocol['options']['val_limit'])
    manifest = dict(kind=config['model'], selection_status='smoke_validation_only' if smoke else 'frozen_validation_diagnostic',
                    **identity, neural_training_source_sha256=protocol['source_sha256'], seed=row['seed'],
                    best_epoch=row['best_epoch'], variant=row['variant'], loss_config=protocol['loss_config'],
                    selection='Pure neural complete validation MSE; KL never selects checkpoints',
                    output='pure_neural_network')
    atomic_json(folder/'manifest.json', manifest)
    atomic_json(folder/'freeze.json', dict(**identity, validation_evidence=row,
                reason='Both architecture arms completed validation selection; no test-dependent selection',
                created=datetime.now(timezone.utc).isoformat()))
    verify_artifact(folder, 'val')
    return str(folder)


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--gpu-memory-fraction', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=0)
    opts = parser.parse_args()
    if not 0 < opts.gpu_memory_fraction <= 1:
        parser.error('Invalid GPU fraction')
    out = opts.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    lock = (out/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    atomic_json(out/'pipeline_options.json', {k:str(v) if isinstance(v,Path) else v for k,v in vars(opts).items()})
    report = dict(status='running', runs=[], smoke=bool(opts.train_limit or opts.val_limit), test_accessed=False)
    atomic_json(out/'report.json', report)
    torch.set_num_threads(2)
    try:
        phase(out, 'paired_training')
        training = out/'training'
        command = [sys.executable, '-u', 'scripts/train_timemix_pdm_innovation.py', '--output', str(training), '--device', opts.device, '--seeds', '2021']
        for key in ['epochs','patience','train_limit','val_limit','gpu_memory_fraction']:
            command += ['--'+key.replace('_','-'), str(getattr(opts,key))]
        with (out/'training_launch.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            proc = Path('/proc')/str(child.pid)
            identity = dict(pid=child.pid, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start_ticks=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19], command=command)
            atomic_json(out/'process.json', identity)
            if child.wait() != 0:
                raise RuntimeError('Training process failed; see training_launch.log')
        tr = json.loads((training/'report.json').read_text())
        tp = json.loads((training/'protocol.json').read_text())
        if tr['status'] != 'completed' or {v['variant'] for v in tr['runs']} != set(ARMS):
            raise RuntimeError('Incomplete paired training')
        if not report['smoke'] and (tr['train_windows'],tr['validation_windows']) != (34369,11425):
            raise RuntimeError('Incomplete dataset coverage')
        last = torch.load(training/'2021/last.pt', map_location='cpu', weights_only=True)
        if len({v['trained_epochs'] for v in tr['runs']}) != 1:
            raise RuntimeError('Unequal completed epochs')
        updates = last['epoch'] * ((tr['train_windows']+31)//32)
        if last['completed_updates'] != updates:
            raise RuntimeError('Unexpected update count')
        for name in ARMS:
            for state in last['optimizers'][name]['state'].values():
                if int(state['step']) != updates:
                    raise RuntimeError('Unequal optimizer update count')
        cfg = SimpleNamespace(**tp['model_args'])
        for source in tr['runs']:
            row = dict(source)
            row['completed_updates'] = updates
            row['validation_audit'] = audit_predictions(training/'2021'/(row['variant']+'_selected_validation.npz'), row['selected_validation'])
            phase(out, 'freezing', variant=row['variant'])
            row['artifact'] = make_artifact(out/(row['variant']+'_artifact'), cfg, tp, row)
            report['runs'].append(row)
            atomic_json(out/'report.json', report)
        by = {v['variant']: v for v in report['runs']}
        report['contrasts'] = {'innovation_minus_oscillator': {
            key: by['innovation']['selected_validation'][key]-by['oscillator']['selected_validation'][key]
            for key in ['mse','mae','low4_mse','centered_amplitude_ratio','window_offset_mse']}}
        report['comparison_caveat'] = 'One-layer common architecture; history additive scan versus conditional feature-observation filter; 32 extra Q/R parameters; same shared initialization; initial functions differ; future decoder retained; MSE only; not calibrated output uncertainty; single seed reused validation'
        report.update(status='completed', source_hash_count=len(tp['source_sha256']), test_accessed=False)
        atomic_json(out/'report.json', report)
        phase(out, 'completed')
        print('INNOVATION STUDY COMPLETE', out, flush=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        atomic_json(out/'report.json', report)
        phase(out, 'failed', error=repr(exc))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
