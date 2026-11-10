"""Only the new original-model RQ-MMD namespace may be written."""
from pathlib import Path
from contextlib import contextmanager
import os,json,hashlib,fcntl,datetime,subprocess,shutil
R=Path(__file__).resolve().parents[1]
G=R/'tmp/mafirst_original_h720_rqmmd_group_20261025'
O=R/'tmp/mafirst_original_h720_rqmmd_20261025_full_seed2021'
BASE=R/'tmp/mafirst_mmd_ettm1_20261016_h720_seed2021_max30_patience5'
PLAN=G/'plan.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
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
 sources=read(G/'frozen_sources.json')
 for p,h in sources.items():assert sha(R/p)==h,p
 proof=read(G/'baseline_immutable_proof.json');assert sha(R/'tmp/timemix_0308_goal_state.json')==proof['old_global_state_sha256']
 if full:
  for p,h in proof['files'].items():assert sha(R/p)==h,p
 return dict(status='verified',checked=now(),frozen_sources=len(sources),old_files=len(proof['files']) if full else 'deferred',old_global_state_unchanged=True)
def resources():
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]);required=4500
 if (G/'memory/verification.json').exists():required=max(required,int(read(G/'memory/verification.json')['peak_reserved_mib'])+512)
 disk=shutil.disk_usage(R).free/2**30
 return dict(gpu_free_mib=free,gpu_required_mib=required,disk_free_gib=disk,ready=free>=required and disk>=30)
