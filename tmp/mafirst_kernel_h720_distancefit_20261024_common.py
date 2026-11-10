"""Only this new group's files may be changed; old global state is immutable."""
from pathlib import Path
import contextlib,datetime,fcntl,hashlib,json,os,subprocess,shutil
R=Path(__file__).resolve().parents[1]
G=R/"tmp/mafirst_kernel_h720_distancefit_group_20261024"
PLAN=R/"tmp/mafirst_kernel_h720_distancefit_plan_20261024.json"
Q=R/"tmp/mafirst_kernel_h720_distancefit_queue_20261024.json"
LOCK=R/"tmp/pdm_state_patch_rope_registration.lock"
read=lambda p:json.loads(Path(p).read_text())
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
@contextlib.contextmanager
def locked():
    with LOCK.open("a+") as f:
        fcntl.flock(f,fcntl.LOCK_EX);yield
def atomic_unlocked(p,value):
    p=Path(p);t=p.with_name(p.name+"."+str(os.getpid())+".tmp")
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n");t.replace(p)
def atomic(p,value):
    with locked():atomic_unlocked(p,value)
def identity(pid=None):
    pid=pid or os.getpid();p=Path('/proc')/str(pid)
    return dict(pid=pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19],
        command=[s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s])
def alive(record):
    if not record:return False
    try:
        v=identity(record['pid'])
        return all(v[k]==record[k] for k in ('pid','boot_id','start_ticks','command'))
    except (OSError,KeyError):return False
def guard(files=False):
    p=read(G/'baseline_immutable_proof.json')
    assert sha(R/'tmp/timemix_0308_goal_state.json')==p['old_global_state_sha256']
    for rel,h in read(G/'frozen_sources.json').items():assert sha(R/rel)==h,rel
    if files:
        for rel,h in p['files'].items():assert sha(R/rel)==h,rel
    return dict(status='verified',checked=now(),old_files=len(p['files']) if files else 'deferred',
        frozen_sources=len(read(G/'frozen_sources.json')),old_global_state_unchanged=True)
def stage(status,**details):
    with locked():
        guard(False);q=read(Q);q.update(status=status,updated=now(),**details);atomic_unlocked(Q,q)
def resource_status():
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    gpu=int(r.stdout.strip().splitlines()[0]) if r.returncode==0 else 0
    mem=next(int(l.split()[1])/1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    disk=shutil.disk_usage(R).free/2**30
    required=4500
    measured=G/'development_memory_preflight_verifications.json'
    if measured.exists():required=max(required,int(max(v['peak_reserved_mib'] for v in read(measured).values()))+512)
    return dict(ready=gpu>=required and disk>=30,gpu_free_mib=gpu,gpu_required_mib=required,disk_free_gib=disk,
        ram_available_mib=mem,ram_gate_enabled=False)
