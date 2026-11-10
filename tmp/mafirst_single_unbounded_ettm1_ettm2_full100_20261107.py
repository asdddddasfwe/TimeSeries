"""Fresh100 queue: historical-hidden single unbounded periods, ETTm1 then ETTm2."""
import argparse,importlib,json,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ.setdefault('MAFIRST_SINGLE_UNBOUNDED_HORIZON_20261107','96')
os.environ.setdefault('MAFIRST_SINGLE_UNBOUNDED_DATASET_20261107','ETTm1')
NEW='mafirst_single_unbounded_ettm1_ettm2_full100_20261107'
BATCH=ROOT/'tmp'/NEW
HS=(96,192,336,720);DATASETS=('ETTm1','ETTm2');ARM='mafirst_season'
MODEL='OscillatorPDMStatePatchSingleUnboundedHorizons20261107'
COUNTS={96:179548,192:207196,336:248668,720:359260}
STAGES=('CPU_route','CUDA_route','actual128_memory','preflight_continuous','preflight_stop1','preflight_resume','preflight_saved_exact','formal_train','official_test','saved_review')
from tmp.mafirst_single_unbounded_ettm1_ettm2_full100_20261107_common import alive,identity,locked,now,read,sha

def source(h):
 return ROOT/(f'tmp/mafirst_spectral_timevary_distdf_rng_ettm1_all_horizons_seed2021_20261105/h{h}' if h!=720 else 'tmp/mafirst_best_combinations_distdf_rng_ettm1_h720_seed2021_20261104/spectral_timevary')

def write(p,v):
 assert p.resolve().is_relative_to(BATCH.resolve()),p
 with locked():
  p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');t.replace(p)
 assert read(p)==v

def common(d,h):
 os.environ['MAFIRST_SINGLE_UNBOUNDED_DATASET_20261107']=d;os.environ['MAFIRST_SINGLE_UNBOUNDED_HORIZON_20261107']=str(h)
 c=importlib.import_module('tmp.'+NEW+'_common')
 return c if (c.DATASET,c.HORIZON)==(d,h) else importlib.reload(c)

def prepare():
 from tmp.mafirst_single_unbounded_ettm1_ettm2_full100_20261107_prepare import prepare_batch
 prepare_batch()

def verify(d,h,device):
 import torch,numpy as np,random,hashlib
 from scripts.run_distdf_fredformer_official_recipe import seed
 from models.OscillatorPDMStatePatchSpectralTimevaryHorizons20261105 import Model as Original
 from models.OscillatorPDMStatePatchSingleUnboundedHorizons20261107 import Model
 import models.OscillatorPDMStatePatchUnboundedPeriodAngle20261105 as old
 from models.OscillatorUnboundedPeriodAngle20261105 import angular_delta,learned_periods
 c=common(d,h);c.guard();f=importlib.import_module('tmp.'+NEW+'_factory');torch.set_num_threads(2)
 cfg=f.configuration(ARM);seed(2021);control=Original(cfg);crng=torch.get_rng_state().clone();prng=random.getstate();nrng=np.random.get_state()
 m,_=f.fresh(ARM);n2=np.random.get_state();assert torch.equal(crng,torch.get_rng_state()) and prng==random.getstate();assert nrng[0]==n2[0] and np.array_equal(nrng[1],n2[1]) and nrng[2:]==n2[2:]
 ref=torch.load(c.G/'reference_initial/mafirst_season.pt',map_location='cpu',weights_only=True);control.load_state_dict({k:v for k,v in ref.items() if k!='learnable_period_offsets'},strict=True)
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in ref.items());assert torch.equal(m.learnable_period_offsets,torch.zeros(2));assert Model.predict_components is old.InheritedCombination.predict_components
 if device=='cuda':torch.cuda.set_per_process_memory_fraction(.9)
 m.to(device).eval();control.to(device).eval();x=torch.randn(2,96,7,device=device);captures=[];original_transition=old.position_transitions
 def capture(dec,history,transition,frequency,head,basis,offsets,*args,**kwargs):
  delta,margin=angular_delta(dec,history,frequency,head,len(basis),period_offsets=offsets)
  assert history.ndim==3 and history.shape[0]==14 and history.shape[1] in (96,48,24,12)
  assert delta.shape==(14,h//4,dec.modes) and torch.isfinite(delta).all();assert (delta.abs()<=margin[:,None]+1e-7).all()
  captures.append((dec,history.detach().clone(),frequency.detach().clone(),head,len(basis),delta.detach().clone()))
  return original_transition(dec,history,transition,frequency,head,basis,offsets,*args,**kwargs)
 old.position_transitions=capture
 try:
  with torch.no_grad():y=m(x);z=control(x);torch.testing.assert_close(y,z,atol=1e-6,rtol=1e-5)
  assert y.shape==(2,h,7) and torch.isfinite(y).all() and len(captures)==4
  dec,history,frequency,head,steps,_=captures[0]
  with torch.no_grad():
   head.weight.normal_(0,.02);a,_=angular_delta(dec,history,frequency,head,steps,period_offsets=m.learnable_period_offsets);b,_=angular_delta(dec,history+torch.randn_like(history)*.1,frequency,head,steps,period_offsets=m.learnable_period_offsets)
   dependence=float((a-b).abs().max());assert dependence>1e-9;change=float((a[:,1:]-a[:,:-1]).abs().max());assert change>1e-9
   captures.clear();m(x);delta_before=[q[-1] for q in captures];positions=[e.future_positions.detach().clone() for e in m.seasonals]
   for e in m.seasonals:e.future_positions.add_(.125)
   captures.clear();m(x);assert all(torch.equal(q[-1],a) for q,a in zip(captures,delta_before))
   for e,p in zip(m.seasonals,positions):e.future_positions.copy_(p)
  crossed=learned_periods(m.learnable_period_offsets.new_tensor([-2.,2.])).tolist();assert crossed[0]<48 and crossed[1]>768
  a,_=angular_delta(dec,history,frequency,head,180,period_offsets=m.learnable_period_offsets);b,_=angular_delta(dec,history,frequency,head,48,period_offsets=m.learnable_period_offsets);torch.testing.assert_close(a[:,:48],b,atol=1e-7,rtol=1e-6)
  b.square().mean().backward();pg=m.learnable_period_offsets.grad.detach().clone();assert torch.isfinite(pg).all() and (pg.abs()>1e-10).all()
  with torch.no_grad():head.weight.zero_()
  m.zero_grad(set_to_none=True);m.train();m(x).square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());norms=[float(a.weight.grad.norm()) for a in m.position_frequency_heads];assert all(a>0 for a in norms)
 finally:old.position_transitions=original_transition
 tr,va=f.dataset(ARM,'train'),f.dataset(ARM,'val');assert len(tr)==c.TRAIN_WINDOWS and len(va)==c.EVAL_WINDOWS
 import pandas as pd
 from sklearn.preprocessing import StandardScaler
 data=pd.read_csv(c.G/'train_val_data'/(d+'.csv'));expected=StandardScaler().fit(data.iloc[:34560,1:].values)
 assert len(data)==46080 and np.array_equal(tr.scaler.mean_,expected.mean_) and np.array_equal(tr.scaler.scale_,expected.scale_)
 record=dict(status='verified',dataset=d,horizon=h,device=device,shape=list(y.shape),parameters=sum(p.numel() for p in m.parameters()),common_own_horizon_untrained_states_exact=True,construction_CPU_Python_NumPy_RNG_matches_original=True,construction_CPU_RNG_sha256=hashlib.sha256(crng.numpy().tobytes()).hexdigest(),neutral_prediction_matches_original=dict(atol=1e-6,rtol=1e-5,max_abs=float((y-z).abs().max())),single_unbounded_original_forward_function_identical=True,actual_four_scale_frequency_head_condition='history_mean_and_last',future_position_perturbation_does_not_change_angular_delta_exact=True,activated_history_dependence_max_abs=dependence,activated_temporal_change_max_abs=change,frequency_head_gradient_norms=norms,all_parameter_gradients_finite=True,period_gradients=pg.cpu().tolist(),unbounded_periods_at_eta_minus2_plus2=crossed,physical_prefix48_vs180_matches=True,per_step_margin_verified=True,official_train_scaler_fit_first34560_verified=True,train_windows=len(tr),full_val_windows=len(va),test_evaluations=0,preparation_weights_discarded=True,created=now())
 c.atomic(c.G/f'{device}_verification.json',record);print(json.dumps(record),flush=True)

def precheck():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only preparation must hide CUDA'
 for d in DATASETS:
  for h in HS:
   c=common(d,h);c.guard();c.run('CPU_route',[c.PY,'-B','-u',str(Path(__file__).resolve()),'verify','--dataset',d,'--horizon',str(h),'--device','cpu'])

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
 if d=='ETTm1':
  old=ROOT/'tmp'/('mafirst_unbounded_period_angle_distdf_ettm1_h720_seed2021_20261105/spectral_timevary_unbounded_period_angle' if h==720 else f'mafirst_full_decay_distdf_ettm1_h{h}_seed2021_20261106/unbounded')/'output/official_test_result.json'
  if old.exists():
   prior=read(old);assert prior['truth_sha256']==result['truth_sha256']
 c.guard();c.atomic(c.G/'completion_verification.json',dict(status='verified',test_metrics=result,historical_single_unbounded=prior,historical_budget_epochs=None if prior is None else prior['epochs'],new_budget_epochs=100,independent_fresh_training=True,full_decay_audit=audit,created=now()));c.atomic(c.G/'status.json',dict(status='completed',process=identity(),updated=now()))

def supervise():
 with locked():
  with (BATCH/'supervisor_once.json').open('x') as f:json.dump(identity(),f)
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

def launch():
 assert read(BATCH/'prepared.json')['status']=='verified'
 for d in DATASETS:
  for h in HS:
   c=common(d,h);c.guard();e=read(c.G/'execution/CPU_route/log.exit.json');assert e['real_wait'] and e['exit_code']==0
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
