"""Isolated five-dataset experiment; all metadata writes use the shared lock."""
from pathlib import Path
from contextlib import contextmanager
import os,json,hashlib,fcntl,datetime,subprocess,shutil
R=Path(__file__).resolve().parents[1]
HORIZON=int(os.environ['MAFIRST_FUTURE_HIDDEN_ONLY_HORIZON_20261107'])
assert HORIZON in (96,192,336,720)
DATASET=os.environ['MAFIRST_FUTURE_HIDDEN_ONLY_DATASET_20261107']
SPECS = {
    'ETTh1': ('ETT-small/ETTh1.csv', 7, 8640, 11520, 14400, 'h'),
    'ETTh2': ('ETT-small/ETTh2.csv', 7, 8640, 11520, 14400, 'h'),
    'Weather': ('weather/weather.csv', 21, 36886, 42156, 52695, '10min'),
    'Electricity': ('electricity/electricity.csv', 321, 18412, 21043, 26303, 'h'),
    'Traffic': ('traffic/traffic.csv', 862, 12280, 14035, 17543, 'h'),
}
CSV,CHANNELS,TRAIN_END,VAL_END,TEST_END,FREQ = SPECS[DATASET]
LARGE = CHANNELS > 32
TAG='spectral_timevary_future_hidden_only_fixed_cosine'
TRAIN_WINDOWS=TRAIN_END-96-HORIZON+1
VAL_WINDOWS=VAL_END-TRAIN_END-HORIZON+1
EVAL_WINDOWS=TEST_END-VAL_END-HORIZON+1
TEST_SUPPORT=dict(input_start=VAL_END-96,target_start=VAL_END,end_exclusive=TEST_END,train_scaler_rows=TRAIN_END)
UPDATES_PER_EPOCH=TRAIN_WINDOWS//128
G=R/'tmp/mafirst_future_hidden_only_five_datasets_full100_20261109'/DATASET.lower()/f'h{HORIZON}'/'group'
OUT={'mafirst_season':G.parent/'output'}
PY='/home/abc/anaconda3/envs/TSLib/bin/python'
ARMS=['mafirst_season']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256();buf=bytearray(65536);v=memoryview(buf)
 with Path(p).open('rb',buffering=0) as f:
  while n:=f.readinto(buf):h.update(v[:n])
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
@contextmanager
def locked():
 with (R/'tmp/pdm_state_patch_rope_registration.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def write_allowed(p):
 p=Path(p).resolve()
 assert p.is_relative_to(G.resolve()) or any(p.is_relative_to(x.resolve()) for x in OUT.values()),str(p)
 return p
def atomic_unlocked(p,v):
 p=write_allowed(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def atomic(p,v):
 with locked():atomic_unlocked(p,v)
def save(p,v):
 import torch
 with locked():
  p=write_allowed(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');torch.save(v,t);t.replace(p)
def append(p,v):
 with locked():
  p=write_allowed(p)
  with p.open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
def identity(pid=None):
 pid=os.getpid() if pid is None else pid;p=Path('/proc')/str(pid)
 return dict(pid=pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19],command=[s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s])
def alive(rec):
 try:return identity(rec['pid'])=={k:rec[k] for k in ['pid','boot_id','start_ticks','command']} and (Path('/proc')/str(rec['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
 except (OSError,KeyError):return False
def guard():
 m=read(G/'registration.json')['hashes']
 for rel,h in m.items():assert sha(R/rel)==h,rel
 for rel,h in read(G/'frozen_sources.json').items():assert sha(R/rel)==h,rel
 for rel,h in read(G/'protected_records.json').items():assert sha(R/rel)==h,rel
 return dict(status='verified',sources=len(read(G/'frozen_sources.json')),protected_records=len(read(G/'protected_records.json')),checked=now(),process=identity())
ROOT=R
B=G
CASES=[(DATASET,0,'season')]

def stem(d=None,f=0,p='season'):
 assert (DATASET if d is None else d,f,p)==(DATASET,0,'season')
 return 'mafirst_future_hidden_only_five_datasets_full100_20261109'

def group(d=None,f=0,p='season'):
 stem(d,f,p);return G

def output(d=None,f=0,p='season'):
 stem(d,f,p);return OUT['mafirst_season']


import time
def run(name, command, root=B):
    e = root / 'execution' / name
    with locked():
        e.mkdir(parents=True, exist_ok=False)
        (e / 'dispatch_once.json').write_text(json.dumps(dict(command=command, process=identity(), created=now())))
    with (e / 'log.txt').open('xb') as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(100):
            try:
                rec = identity(proc.pid)
                if rec['command'] == command:
                    break
            except OSError:
                pass
            time.sleep(0.02)
        assert rec['command'] == command
        atomic(e / 'process.json', rec)
        atomic(root / 'status.json', dict(status=name, child=rec, process=identity(), updated=now()))
        code = proc.wait()
    atomic(e / 'log.exit.json', dict(exit_code=code, real_wait=True, created=now()))
    assert code == 0, (name, code)
