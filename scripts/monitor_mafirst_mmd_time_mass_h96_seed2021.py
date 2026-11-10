"""Stdlib-only monitor; copied into each output as monitor_check.py."""
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    out = Path(__file__).resolve().parent
    record = read(out/'process.json')
    status, progress = read(out/'status.json'), read(out/'progress.json')
    previous = read(out/'monitor_state.json')
    alerts, live, identity = [], False, {}
    root = out.parent.parent
    current = read(root/'tmp/timemix_0308_goal_state.json').get('pdm_state_patch_mmd_time_mass_h96_seed2021_results', {}).get('current_run', {})
    if current and current.get('output') != str(out):
        alerts.append('current_run points to another output; monitor must follow current_run')
    if (out/'pid.txt').exists() and (out/'pid.txt').read_text().strip() != str(record.get('pid')):
        alerts.append('pid.txt does not match process.json')
    if current.get('output') == str(out):
        for key in ('pid', 'boot_id', 'start_ticks'):
            if key in current and str(current[key]) != str(record.get(key)):
                alerts.append('current_run/process.json identity mismatch: '+key)
    process = Path('/proc')/str(record.get('pid', 0))
    if process.exists():
        try:
            fields = (process/'stat').read_text().rsplit(')', 1)[1].split()
            command = [v.decode() for v in (process/'cmdline').read_bytes().split(b'\0') if v]
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            matches = (fields[19] == str(record.get('start_ticks')) and
                       command == record.get('command') and boot == record.get('boot_id'))
            live = matches and fields[0] != 'Z'
            identity = dict(matches=matches, state=fields[0], cpu_ticks=int(fields[11])+int(fields[12]),
                            io=(process/'io').read_text() if matches else '')
            if not matches and fields[0] != 'Z':
                alerts.append('process identity mismatch; old PID must not be trusted')
        except FileNotFoundError:
            pass
    completed = status.get('status') in ('completed_verified', 'preflight_verified', 'stopped_by_user', 'controlled_stop')
    if not live and not completed:
        alerts.append('process exited before completion')
    if (out/'failure.json').exists():
        alerts.append('pipeline failure: '+str(read(out/'failure.json').get('error')))

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            alerts.append('nonfinite recorded metric')
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in ('distance', 'patch_covariance', 'minimum_unclipped_distance', 'train_d','time_position_mmd','train_mmd','minimum_unclipped_mmd') and isinstance(item, (int, float)) and item < -1e-10:
                    alerts.append('negative kernel distance below tolerance: '+key)
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
    finite(progress)
    watched = [out/name for name in ('status.json', 'progress.json', 'last.pt', 'gradients.jsonl',
               'freeze.json', 'test_started.json', 'test_receipt.json', 'report.json', 'completion_audit.json')]
    watched += list(out.glob('*/epochs/*.pt'))
    watched += list(out.glob('*/*/*.npy'))
    watched += list(out.glob('*/test_receipt.json'))
    watched += list(out.glob('*/*/report.json'))
    watched += list(out.glob('*/val_features.json'))
    watched += list(out.glob('*/test_features.json'))
    watched += list(out.glob('*/*_encoder_features.json'))
    watched += list(out.glob('*/status.json'))
    watched += [Path(str(out)+'_launch.log')]
    for path in watched:
        if path.exists() and path.suffix == '.json':
            try:
                finite(read(path))
            except (OSError, json.JSONDecodeError):
                alerts.append('unreadable JSON evidence: '+str(path))
    gradients = out/'gradients.jsonl'
    if gradients.exists():
        lines = gradients.read_text().splitlines(keepends=True)
        for line in lines:
            if line.endswith('\n'):
                try:
                    finite(json.loads(line))
                except json.JSONDecodeError:
                    alerts.append('malformed complete gradient record')
    log = Path(str(out)+'_launch.log')
    if log.exists():
        content = log.read_text(errors='replace')
        if re.search(r'Traceback \(most recent call last\)|CUDA error:|CUDA out of memory|OutOfMemoryError|AssertionError|RuntimeError:', content):
            alerts.append('Python/CUDA exception in launch log')
    evidence = {str(p.relative_to(out)) if p.is_relative_to(out) else str(p):
                [p.stat().st_size, p.stat().st_mtime_ns] for p in watched if p.exists()}
    signature = hashlib.sha256(json.dumps([evidence, identity], sort_keys=True).encode()).hexdigest()
    last_change = previous.get('last_change', time.time()) if previous.get('signature') == signature else time.time()
    if live and time.time()-last_change >= 2700:
        alerts.append('all progress, file, CPU and IO evidence unchanged for >=45 minutes')
    alerts = sorted(set(alerts))
    new_alerts = [v for v in alerts if v not in previous.get('reported_alerts', [])]
    report = dict(checked=datetime.now(timezone.utc).isoformat(), live=live, identity=identity,
                  phase=status.get('status'), completed_epochs=progress.get('epoch', 0),
                  epoch=status.get('epoch'), step=status.get('step'), best_epoch=progress.get('best_epoch'),
                  best_val_mse=progress.get('best'), last_checkpoint=str(out/'last.pt'),
                  last_checkpoint_exists=(out/'last.pt').exists(), signature=signature,
                  last_change=last_change, alerts=alerts, new_alerts=new_alerts,
                  reported_alerts=sorted(set(previous.get('reported_alerts', [])+alerts)))
    temporary = out/'monitor_state.json.tmp'
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(out/'monitor_state.json')
    with (out/'monitor_checks.jsonl').open('a') as file:
        file.write(json.dumps(report)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('identity', 'signature', 'reported_alerts')}))


if __name__ == '__main__':
    main()
