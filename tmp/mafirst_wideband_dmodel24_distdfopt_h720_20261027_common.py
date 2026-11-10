"""Only the new original-model RQ-MMD namespace may be written."""
from pathlib import Path
from contextlib import contextmanager
import os,json,hashlib,fcntl,datetime,subprocess,shutil
R=Path(__file__).resolve().parents[1]
G=R/'tmp/mafirst_wideband_dmodel24_distdfopt_h720_group_20261027'
O=R/'tmp/mafirst_wideband_dmodel24_distdfopt_h720_20261027_full_seed2021'
BASE=R/'tmp/mafirst_mmd_ettm1_20261016_h720_seed2021_max30_patience5'
PLAN=G/'plan.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256();buf=bytearray(65536);view=memoryview(buf)
 with Path(p).open('rb',buffering=0) as f:
  while n:=f.readinto(buf):h.update(view[:n])
 return h.hexdigest()

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
@contextmanager
def locked():
 with (R/'tmp/pdm_state_patch_rope_registration.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def atomic_unlocked(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def atomic(p,v):
 with locked():atomic_unlocked(p,v)
def save(p,v):
 import torch
 with locked():
  p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');torch.save(v,t);t.replace(p)
def identity(pid=None):
 pid=os.getpid() if pid is None else pid;p=Path('/proc')/str(pid)
 return dict(pid=pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19],command=[s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s])
def alive(rec):
 try:return identity(rec['pid'])=={k:rec[k] for k in ('pid','boot_id','start_ticks','command')} and (Path('/proc')/str(rec['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
 except (OSError,KeyError):return False
def guard(full=False):
 import time
 start=time.monotonic();sources=read(G/'frozen_sources.json');proof=read(G/'baseline_immutable_proof.json');seen={};checked=0;raw=0;last=start
 def verify(rel,h):
  nonlocal checked,raw,last
  p=R/rel;s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
  if key not in seen:
   seen[key]=sha(p);raw+=s.st_size;t=p.stat();assert (t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns)==key,rel
  assert seen[key]==h,rel;checked+=1
  if full and time.monotonic()-last>20:
   atomic(G/'integrity_progress.json',dict(status='checking',checked_paths=checked,unique_bytes=raw,updated=now(),process=identity()));last=time.monotonic()
 for rel,h in sources.items():verify(rel,h)
 assert sha(R/'tmp/timemix_0308_goal_state.json')==proof['old_global_state_sha256']
 if full:
  for rel,h in proof['files'].items():verify(rel,h)
  for rel in proof['authorized_retirement']:assert not (R/rel).exists(),rel
  for name in ('primary_peer_baseline_freeze.json','queued_peer_baselines_freeze.json'):
   for rel,h in read(G/name)['files'].items():verify(rel,h)
 return dict(status='verified',checked=now(),frozen_sources=len(sources),old_files=len(proof['files']) if full else 'deferred',old_global_state_unchanged=True,read_buffer_bytes=65536,unique_bytes_hashed=raw,seconds=time.monotonic()-start)

def resources():
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]);required=4500
 if (G/'memory/verification.json').exists():required=max(required,int(read(G/'memory/verification.json')['peak_reserved_mib'])+512)
 disk=shutil.disk_usage(R).free/2**30
 return dict(gpu_free_mib=free,gpu_required_mib=required,disk_free_gib=disk,ready=free>=required and disk>=30)

PEERG=R/'tmp/mafirst_original_h720_wideband_group_20261026'
PEER=R/'tmp/mafirst_original_h720_wideband_20261026_full_seed2021'

PEERG=R/'tmp/mafirst_wideband_distdfopt_h720_group_20261026'
PEER=R/'tmp/mafirst_wideband_distdfopt_h720_20261026_full_seed2021'
Q=R/'tmp/mafirst_bandwidth_factorial_requeue_20261027'
