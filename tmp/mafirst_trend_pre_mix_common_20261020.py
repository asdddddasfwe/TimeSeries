"""New scope I/O, source integrity and locked registrations; stdlib only."""
from pathlib import Path
import datetime,hashlib,json,os,fcntl
R=Path(__file__).resolve().parents[1]
STATE=R/'tmp/timemix_0308_goal_state.json'
LOCK=R/'tmp/pdm_state_patch_rope_registration.lock'
PY='/home/abc/anaconda3/envs/TSLib/bin/python'
PLAN=R/'tmp/mafirst_trend_pre_mix_plan_20261020.json'
G=R/'tmp/mafirst_trend_pre_mix_group_20261020_h720_seed2021'
KEY='pdm_state_patch_trend_pre_mix_h720_results'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def atomic(p,x):
    p=Path(p);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def identity(pid=None):
    pid=os.getpid() if pid is None else pid;d=Path('/proc')/str(pid)
    return dict(pid=pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        start_ticks=(d/'stat').read_text().rsplit(')',1)[1].split()[19],
        command=[x.decode() for x in (d/'cmdline').read_bytes().split(b'\0') if x])
def live(rec):
    try:
        return identity(rec['pid'])=={k:rec[k] for k in ('pid','boot_id','start_ticks','command')} and (Path('/proc')/str(rec['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (KeyError,FileNotFoundError,ProcessLookupError):return False
def guard(files=False):
    state=read(STATE);proof=read(G/'baseline_immutable_proof.json')
    assert all(state[k]==v for k,v in proof['state_sections'].items()),'protected historical state changed'
    for f,h in read(G/'frozen_sources.json').items():assert sha(R/f)==h,f
    for f,h in proof['metadata'].items():assert sha(R/f)==h,f
    if files:
        for f,h in proof['files'].items():assert sha(R/f)==h,f
    final=G/'predecessor_final_proof.json'
    if final.exists():
        d=read(final)
        assert all(state[k]==v for k,v in d['state_sections'].items()),'closed predecessor state changed'
        for f,h in d['metadata'].items():assert sha(R/f)==h,f
        if files:
            for f,h in d['files'].items():assert sha(R/f)==h,f

def register(status,job=None,process=None,**fields):
    plan=read(PLAN)
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);s=read(STATE);proof=read(G/'baseline_immutable_proof.json')
        assert all(s[k]==v for k,v in proof['state_sections'].items())
        entry=dict(status=status,updated=now(),output=str(G),queue=plan['queue'],document=plan['group_report'],**fields)
        s[KEY].update(status=status,current_run=entry)
        if job:
            event=dict(entry,output=str(R/job['output']),job_id=job['id'])
            if process:event.update(process)
            s[job['state_key']].update(status=status,current_run=event)
        # While predecessor is active keep its current panel untouched.
        if (G/'predecessor_final_proof.json').exists():
            reg=read(G/'monitor_registration.json')
            s['current']=dict(entry,experiment=KEY,conclusion_notified=False,monitor_status=reg['status'],automation_id=reg['automation_id'])
        atomic(STATE,s)
