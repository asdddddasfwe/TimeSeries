"""Fresh100 queue: future-hidden-only fixed-cosine periods, ETTm1 then ETTm2."""
import argparse,importlib,json,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ.setdefault('MAFIRST_FUTURE_HIDDEN_ONLY_HORIZON_20261107','96')
os.environ.setdefault('MAFIRST_FUTURE_HIDDEN_ONLY_DATASET_20261107','ETTm1')
NEW='mafirst_future_hidden_only_ettm1_ettm2_full100_20261107'
BATCH=ROOT/'tmp'/NEW
HS=(96,192,336,720);DATASETS=('ETTm1','ETTm2');ARM='mafirst_season'
MODEL='OscillatorPDMStatePatchSpectralTimevaryFutureHiddenHorizons20261107'
COUNTS={96:179546,192:207194,336:248666,720:359258}
STAGES=('CPU_route','CUDA_route','actual128_memory','preflight_continuous','preflight_stop1','preflight_resume','preflight_saved_exact','formal_train','official_test','saved_review')
from tmp.mafirst_future_hidden_only_ettm1_ettm2_full100_20261107_common import alive,identity,locked,now,read,sha

def source(h):
 return ROOT/(f'tmp/mafirst_spectral_timevary_distdf_rng_ettm1_all_horizons_seed2021_20261105/h{h}' if h!=720 else 'tmp/mafirst_best_combinations_distdf_rng_ettm1_h720_seed2021_20261104/spectral_timevary')

def write(p,v):
 assert p.resolve().is_relative_to(BATCH.resolve()),p
 with locked():
  p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');t.replace(p)
 assert read(p)==v

def common(d,h):
 os.environ['MAFIRST_FUTURE_HIDDEN_ONLY_DATASET_20261107']=d;os.environ['MAFIRST_FUTURE_HIDDEN_ONLY_HORIZON_20261107']=str(h)
 c=importlib.import_module('tmp.'+NEW+'_common')
 return c if (c.DATASET,c.HORIZON)==(d,h) else importlib.reload(c)

def prepare():
 from tmp.mafirst_future_hidden_only_ettm1_ettm2_full100_20261107_prepare import prepare_batch
 prepare_batch()

def verify(d,h,device):
 import torch,numpy as np,random,hashlib
 from scripts.run_distdf_fredformer_official_recipe import seed
 from models.OscillatorPDMStatePatchSpectralTimevaryHorizons20261105 import Model as Original
 from models.OscillatorPDMStatePatchSpectralTimevaryFutureHiddenHorizons20261107 import Model
 import models.OscillatorPDMStatePatchSpectralTimevaryFutureHiddenV320261105 as future_model
 c=common(d,h);c.guard();f=importlib.import_module('tmp.'+NEW+'_factory');torch.set_num_threads(2)
 cfg=f.configuration(ARM);seed(2021);control=Original(cfg);crng=torch.get_rng_state().clone();prng=random.getstate();nrng=np.random.get_state()
 m,_=f.fresh(ARM);n2=np.random.get_state();assert torch.equal(crng,torch.get_rng_state()) and prng==random.getstate();assert nrng[0]==n2[0] and np.array_equal(nrng[1],n2[1]) and nrng[2:]==n2[2:]
 ref=torch.load(c.G/'reference_initial/mafirst_season.pt',map_location='cpu',weights_only=True);control.load_state_dict(ref,strict=True);assert all(torch.equal(v,m.state_dict()[k]) for k,v in ref.items());assert not hasattr(m,'learnable_period_offsets');assert m.predict_components.__func__ is future_model.InheritedCombination.predict_components
 expected=torch.cos(torch.pi*(torch.arange(h//4,dtype=torch.float32)[:,None]+.5)*torch.arange(1,5,dtype=torch.float32)[None,:]/(h//4));assert torch.equal(m.position_frequency_basis,expected)
 if device=='cuda':torch.cuda.set_per_process_memory_fraction(.9)
 m.to(device).eval();control.to(device).eval();x=torch.randn(2,96,7,device=device);captures=[];original_delta=future_model.future_rotation_delta
 def capture(dec,blocks,angle,upper,basis,head):
  delta=original_delta(dec,blocks,angle,upper,basis,head);assert blocks.shape[:3]==(14,h//4,4) and delta.shape==(14,h//4,dec.modes);assert torch.isfinite(delta).all() and (delta.abs()<=.25*torch.minimum(angle,upper-angle)+1e-7).all();captures.append((dec,blocks.detach().clone(),angle.detach().clone(),upper,basis,head));return delta
 future_model.future_rotation_delta=capture
 try:
  with torch.no_grad():y=m(x);z=control(x);torch.testing.assert_close(y,z,atol=1e-6,rtol=1e-5)
  assert y.shape==(2,h,7) and torch.isfinite(y).all() and len(captures)==4;assert all(q['future_hidden_conditioning'] for q in m.last_gate_context_diagnostics)
  dec,blocks,angle,upper,basis,head=captures[0]
  with torch.no_grad():
   head.weight.normal_(0,.02);a=original_delta(dec,blocks,angle,upper,basis,head);b=original_delta(dec,blocks+torch.randn_like(blocks)*.1,angle,upper,basis,head);dependence=float((a-b).abs().max());assert dependence>1e-9
   altered=blocks.clone();altered[:,0]+=.25;local=original_delta(dec,altered,angle,upper,basis,head);assert float((a[:,0]-local[:,0]).abs().max())>1e-9 and torch.equal(a[:,1:],local[:,1:]);change=float((a[:,1:]-a[:,:-1]).abs().max());assert change>1e-9;head.weight.zero_()
  m.zero_grad(set_to_none=True);m.train();m(x).square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());norms=[float(a.weight.grad.norm()) for a in m.position_frequency_heads];assert all(a>0 for a in norms)
 finally:future_model.future_rotation_delta=original_delta
 tr,va=f.dataset(ARM,'train'),f.dataset(ARM,'val');assert len(tr)==c.TRAIN_WINDOWS and len(va)==c.EVAL_WINDOWS
 import pandas as pd
 from sklearn.preprocessing import StandardScaler
 data=pd.read_csv(c.G/'train_val_data'/(d+'.csv'));expected=StandardScaler().fit(data.iloc[:34560,1:].values);assert len(data)==46080 and np.array_equal(tr.scaler.mean_,expected.mean_) and np.array_equal(tr.scaler.scale_,expected.scale_)
 record=dict(status='verified',dataset=d,horizon=h,device=device,shape=list(y.shape),parameters=sum(p.numel() for p in m.parameters()),common_own_horizon_untrained_states_exact=True,construction_CPU_Python_NumPy_RNG_matches_original=True,construction_CPU_RNG_sha256=hashlib.sha256(crng.numpy().tobytes()).hexdigest(),neutral_prediction_matches_original=dict(atol=1e-6,rtol=1e-5,max_abs=float((y-z).abs().max())),historical_future_hidden_forward_function_identical=True,actual_four_scale_frequency_head_condition='future_block_mean_and_last',activated_future_block_dependence_max_abs=dependence,changing_one_future_block_only_changes_that_delta_block_exact=True,activated_temporal_change_max_abs=change,frequency_head_gradient_norms=norms,all_parameter_gradients_finite=True,no_learnable_period_offsets=True,fixed_normalized_four_cosine_basis_exact=True,per_step_margin_verified=True,official_train_scaler_fit_first34560_verified=True,train_windows=len(tr),full_val_windows=len(va),test_evaluations=0,preparation_weights_discarded=True,created=now())
 c.atomic(c.G/f'{device}_verification.json',record);print(json.dumps(record),flush=True)

def precheck():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 for d in DATASETS:
  for h in HS:
   c=common(d,h);c.guard();c.run('CPU_route',[c.PY,'-B','-u',str(Path(__file__).resolve()),'verify','--dataset',d,'--horizon',str(h),'--device','cpu'])
 write(BATCH/'prepared.json',dict(status='verified',cases=8,CPU_stages_true_wait0=True,created=now()))
 os.environ.pop('CUDA_VISIBLE_DEVICES',None)

def wait_resources(c):
 while True:
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]);jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  peaks=[read(p)['peak_reserved_mib'] for p in BATCH.glob('*/*/group/memory/*/verification.json')];required=max(2500,math.ceil(max(peaks,default=0))+512);disk=shutil.disk_usage(ROOT).free/2**30
  if not jobs and free>=required and disk>=30:return
  c.atomic(c.G/'status.json',dict(status='waiting_resources',gpu_free_mib=free,required_mib=required,compute_pids=jobs,disk_gib=disk,updated=now()));time.sleep(30)

def arm(d,h):
 c=common(d,h);c.guard()
 with locked():
  with (c.G/'arm_once.json').open('x') as f:json.dump(identity(),f)
 e=read(c.G/'execution/CPU_route/log.exit.json');assert e['real_wait'] and e['exit_code']==0
 runner=[c.PY,'-B','-u',str(ROOT/'tmp'/('run_'+NEW+'.py')),'--arm',ARM];wait_resources(c)
 c.run('CUDA_route',[c.PY,'-B','-u',str(Path(__file__).resolve()),'verify','--dataset',d,'--horizon',str(h),'--device','cuda'])
 c.run('actual128_memory',runner+['--phase','memory']);wait_resources(c);pre=c.G/'preflight'/ARM
 for name,directory,extra in [('continuous','continuous',[]),('stop1','resumed',['--stop-after','1']),('resume','resumed',['--resume'])]:c.run('preflight_'+name,runner+['--phase','preflight','--output',str(pre/directory)]+extra)
 c.run('preflight_saved_exact',[c.PY,'-B','-u',str(ROOT/'tmp'/('verify_'+NEW+'_preflight.py'))]);wait_resources(c)
 c.run('formal_train',runner+['--phase','full'])
 c.run('official_test',[c.PY,'-B','-u',str(ROOT/'tmp'/('test_'+NEW+'.py')),'--dataset',d])
 c.run('saved_review',[c.PY,'-B','-u',str(ROOT/'tmp'/('review_'+NEW+'.py')),'--dataset',d])
 result=read(c.OUT[ARM]/'official_test_result.json');assert result['dataset']==d and result['horizon']==h and result['epochs']==100 and result['windows']==c.EVAL_WINDOWS and result['official_test_evaluations']==1
 audit=read(c.OUT[ARM]/'full_decay_completion_audit.json');assert audit['source_epochs']==0 and audit['total_epochs']==100 and audit['total_updates']==100*c.UPDATES_PER_EPOCH
 prior=None
 if d=='ETTm1' and h==720:
  old=ROOT/'tmp/mafirst_spectral_timevary_future_hidden_priority_v3_ettm1_h720_seed2021_20261105/spectral_timevary_future_hidden/output/official_test_result.json'
  prior=read(old);assert prior['truth_sha256']==result['truth_sha256']
 c.guard();c.atomic(c.G/'completion_verification.json',dict(status='verified',test_metrics=result,historical_future_hidden_only=prior,historical_budget_epochs=None if prior is None else prior['epochs'],new_budget_epochs=100,independent_fresh_training=True,full_decay_audit=audit,created=now()));c.atomic(c.G/'status.json',dict(status='completed',process=identity(),updated=now()))

def wait_current_old_arm():
 old=ROOT/'tmp/mafirst_single_unbounded_ettm1_ettm2_full100_20261107';pause=read(old/'priority_future_hidden_supervisor_pause_receipt.json');parent=pause['supervisor'];arm_rec=pause['current_arm']
 while alive(arm_rec):
  assert alive(parent) and (Path('/proc')/str(parent['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0] in ('T','t')
  assert not (old/'failure.json').exists()
  write(BATCH/'status.json',dict(status='waiting_current_single_unbounded_ETTm1_H96_only',old_supervisor_intentionally_stopped=True,current_old_arm=arm_rec,updated=now()));time.sleep(30)
 exits={n:read(old/'ettm1/h96/group/execution'/n/'log.exit.json') for n in STAGES};assert all(x['real_wait'] and x['exit_code']==0 for x in exits.values());assert read(old/'ettm1/h96/group/completion_verification.json')['status']=='verified'
 write(BATCH/'priority_predecessor_completion.json',dict(status='verified_current_group_science_completed',all10_scientific_stages_true_wait0=exits,current_old_arm=arm_rec,old_arm_parent_real_wait_pending_until_resume=True,created=now()))

def supervise():
 with locked():
  with (BATCH/'supervisor_once.json').open('x') as f:json.dump(identity(),f)
 wait_current_old_arm();precheck();os.environ.pop('CUDA_VISIBLE_DEVICES',None)
 effective={};results={}
 for d in DATASETS:
  for h in HS:
   c=common(d,h);cmd=[c.PY,'-B','-u',str(Path(__file__).resolve()),'arm','--dataset',d,'--horizon',str(h)]
   write(BATCH/'status.json',dict(status='running',dataset=d,horizon=h,process=identity(),updated=now()))
   with (c.G.parent/'arm.log').open('xb') as log:p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);code=p.wait()
   write(c.G.parent/'arm_exit.json',dict(exit_code=code,real_wait=True,created=now()));assert code==0,(d,h,code)
   exits={s:read(c.G/'execution'/s/'log.exit.json') for s in STAGES};assert all(x['real_wait'] and x['exit_code']==0 for x in exits.values());effective[f'{d}_h{h}']=exits;results[f'{d}_h{h}']=read(c.G/'completion_verification.json')
   write(BATCH/'progress_results.json',dict(status='in_progress',completed=results,created=now()))
 write(BATCH/'successful_stage_completion.json',dict(status='verified',successful_stages=80,successful_arms=8,groups=effective,created=now()));write(BATCH/'results.json',dict(status='verified',results=results,created=now()));write(BATCH/'status.json',dict(status='completed',updated=now()))
 control=ROOT/'tmp/mafirst_priority_future_hidden_queue_control_20261107.py'
 with (BATCH/'queue_resume.log').open('xb') as log:
  p=subprocess.Popen([c.PY,'-B','-u',str(control),'resume'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
 write(BATCH/'queue_resume_exit.json',dict(exit_code=code,real_wait=True,created=now()));assert code==0

def launch():
 assert read(BATCH/'queued_prepared.json')['status']=='registered_pending_CPU_verification'
 for d in DATASETS:
  for h in HS:common(d,h).guard()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')!='','Supervisor requires CUDA visibility'
 with locked():
  with (BATCH/'dispatch_once.json').open('x') as f:json.dump(dict(process=identity(),created=now()),f)
 cmd=[c.PY,'-B','-u',str(Path(__file__).resolve()),'supervise']
 with (BATCH/'supervisor.log').open('xb') as log:p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
 for _ in range(100):
  try:
   rec=identity(p.pid)
   if rec['command']==cmd:break
  except OSError:pass
  time.sleep(.02)
 assert rec['command']==cmd;write(BATCH/'launch.json',dict(coordinator=rec,source_sha256=sha(Path(__file__)),created=now()));print(json.dumps(read(BATCH/'launch.json')),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','precheck','verify','launch','supervise','arm']);p.add_argument('--dataset',choices=DATASETS);p.add_argument('--horizon',type=int,choices=HS);p.add_argument('--device',choices=['cpu','cuda']);a=p.parse_args()
 try:
  if a.mode=='verify':verify(a.dataset,a.horizon,a.device)
  elif a.mode=='arm':arm(a.dataset,a.horizon)
  else:globals()[a.mode]()
 except BaseException as e:
  if BATCH.exists():write(BATCH/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),phase=a.mode,dataset=a.dataset,horizon=a.horizon,process=identity(),automatic_retry=False,created=now()))
  raise
