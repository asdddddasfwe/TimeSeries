"""Local fixed4-window diagnosis. Only new group and new report may be written."""
from pathlib import Path
import json,shutil
from tmp.mafirst_kernel_h720_distancefit_20261024_common import locked,atomic,atomic_unlocked,sha,now,identity,alive,resource_status as old_resources,guard as prior_guard
R=Path(__file__).resolve().parents[1]
G=R/'tmp/mafirst_kernel_h720_outputgram_overfit_group_20261025'
PLAN=G/'plan.json'
def read(p):return json.loads(Path(p).read_text())
def guard(full=False):
 if not (G/'frozen_sources.json').exists():return prior_guard(False)
 sources=read(G/'frozen_sources.json')
 for f,h in sources.items():assert sha(R/f)==h,f
 proof=read(G/'baseline_immutable_proof.json');assert sha(R/'tmp/timemix_0308_goal_state.json')==proof['old_global_state_sha256']
 if full:
  for f,h in proof['files'].items():assert sha(R/f)==h,f
 return dict(status='verified',checked=now(),frozen_sources=len(sources),old_files=len(proof['files']) if full else 'deferred',old_global_state_unchanged=True)
def resources():
 d=old_resources();mem=G/'development_cuda/verification.json';required=max(4500,int(read(mem)['peak_reserved_mib'])+512) if mem.exists() else 4500
 d.update(gpu_required_mib=required,ready=d['gpu_free_mib']>=required and d['disk_free_gib']>=30);return d
