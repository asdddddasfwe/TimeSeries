"""Queue-only SIGSTOP/SIGCONT of the old supervisor; scientific child is untouched."""
from pathlib import Path
import json,os,signal,time,fcntl,hashlib,datetime,sys
R=Path(__file__).resolve().parents[1]
B=R/'tmp/mafirst_single_unbounded_ettm1_ettm2_full100_20261107'
NEXT=R/'tmp/mafirst_future_hidden_only_ettm1_ettm2_full100_20261107'
def read(p):return json.loads(p.read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def identity(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=s[19],command=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]),s[0]
def exact(rec):
 actual,state=identity(rec['pid']);assert actual=={k:rec[k] for k in actual};return actual,state
def atomic(p,d):
 assert not p.exists();t=p.with_suffix('.json.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(t,p);assert read(p)==d

def pause():
 dest=B/'priority_future_hidden_supervisor_pause_receipt.json';assert not dest.exists()
 with (R/'tmp/pdm_state_patch_rope_registration.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  launch=read(B/'launch.json');root=read(B/'status.json');g=read(B/'ettm1/h96/group/status.json');o=read(B/'ettm1/h96/output/status.json');parent=launch['coordinator']
  assert root['dataset']=='ETTm1' and root['horizon']==96 and g['status']=='formal_train' and o['status']=='formal_running'
  assert parent['pid']==292730 and parent['start_ticks']=='100901060' and parent['command'][-1]=='supervise';exact(parent);exact(g['process']);exact(g['child'])
  assert (Path('/proc')/str(g['process']['pid'])/'stat').read_text().rsplit(')',1)[1].split()[1]==str(parent['pid'])
  for d in ('ettm1','ettm2'):
   for h in (96,192,336,720):
    if (d,h)!=('ettm1',96):assert not (B/d/f'h{h}/group/arm_once.json').exists()
  os.kill(parent['pid'],signal.SIGSTOP)
  for _ in range(100):
   _,state=exact(parent)
   if state in ('T','t'):break
   time.sleep(.02)
  assert state in ('T','t')
  record=dict(status='supervisor_intentionally_stopped_for_priority_only',created=now(),user_request='Future-hidden-only eight fresh100 groups before unstarted single-unbounded groups',supervisor=parent,actual_supervisor_state=state,current_arm=g['process'],current_train=g['child'],current_train_status_at_pause=o,current_scientific_arm_and_train_not_signalled=True,priority_root=NEXT.relative_to(R).as_posix(),resume_only_after_priority_all8_science_completed=True,source_science_bytes_unchanged=True,not_scientific_failure=True,old_current_arm_real_parent_wait_pending_until_supervisor_resume=True)
  atomic(dest,record)
  fcntl.flock(lock,fcntl.LOCK_UN)
 print(json.dumps(record),flush=True)

def resume():
 receipt=read(B/'priority_future_hidden_supervisor_pause_receipt.json');p=receipt['supervisor'];actual,state=exact(p);assert state in ('T','t')
 assert read(NEXT/'status.json')['status']=='completed' and read(NEXT/'results.json')['status']=='verified';e=read(NEXT/'successful_stage_completion.json');assert e['successful_stages']==80 and e['successful_arms']==8
 assert not (B/'priority_future_hidden_supervisor_resume_receipt.json').exists();os.kill(p['pid'],signal.SIGCONT)
 for _ in range(500):
  path=B/'ettm1/h96/arm_exit.json'
  if path.exists():break
  time.sleep(.02)
 e=read(path);assert e['real_wait'] and e['exit_code']==0
 actual,state=exact(p);assert state not in ('T','t','Z')
 record=dict(status='supervisor_resumed_after_priority_all8_science',created=now(),supervisor=actual,state=state,prior_ETTm1_h96_real_parent_wait=e,priority_completion=read(NEXT/'status.json'),original_live_scientific_state_not_reloaded_or_reset=True)
 atomic(B/'priority_future_hidden_supervisor_resume_receipt.json',record);print(json.dumps(record),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
