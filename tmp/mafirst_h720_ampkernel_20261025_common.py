"""Only this amplitude-kernel group may be written. Prior experiments immutable."""
from pathlib import Path
import json
from tmp.mafirst_h720_pointonly_20261025_common import locked,atomic,atomic_unlocked,sha,now,identity,alive,guard as prior_guard,resources as prior_resources
R=Path(__file__).resolve().parents[1]
G=R/'tmp/mafirst_kernel_h720_ampkernel_group_20261025'
PLAN=G/'plan.json'
def read(p):return json.loads(Path(p).read_text())
def guard(full=False):
 if not (G/'frozen_sources.json').exists():return prior_guard(False)
 for f,h in read(G/'frozen_sources.json').items():assert sha(R/f)==h,f
 p=read(G/'baseline_immutable_proof.json');assert sha(R/'tmp/timemix_0308_goal_state.json')==p['old_global_state_sha256']
 if full:
  for f,h in p['files'].items():assert sha(R/f)==h,f
 return dict(status='verified',checked=now(),old_files=len(p['files']) if full else 'deferred',frozen_sources=len(read(G/'frozen_sources.json')),old_global_state_unchanged=True)
def resources():
 d=prior_resources();p=G/'development_cuda/verification.json';required=max(4500,int(read(p)['peak_reserved_mib'])+512) if p.exists() else 4500
 d.update(gpu_required_mib=required,ready=d['gpu_free_mib']>=required and d['disk_free_gib']>=30);return d
