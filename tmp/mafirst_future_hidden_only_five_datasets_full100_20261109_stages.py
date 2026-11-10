"""Fresh100 queue: future-hidden-only fixed-cosine periods, ETTh and custom datasets."""
import argparse,importlib,json,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ.setdefault('MAFIRST_FUTURE_HIDDEN_ONLY_HORIZON_20261107','96')
os.environ.setdefault('MAFIRST_FUTURE_HIDDEN_ONLY_DATASET_20261107','ETTh1')
NEW='mafirst_future_hidden_only_five_datasets_full100_20261109'
BATCH=ROOT/'tmp'/NEW
HS=(96,192,336,720);DATASETS=('ETTh1','ETTh2','Weather','Electricity','Traffic');ARM='mafirst_season'
MODEL='OscillatorPDMStatePatchSpectralTimevaryFutureHiddenHorizons20261107'
STAGES=('CPU_route','CUDA_route','actual128_memory','preflight_continuous','preflight_stop1','preflight_resume','preflight_saved_exact','formal_train','official_test','saved_review')
from tmp.mafirst_future_hidden_only_five_datasets_full100_20261109_common import alive,identity,locked,now,read,sha

def write(p,v):
 assert p.resolve().is_relative_to(BATCH.resolve()),p
 with locked():
  p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');t.replace(p)
 assert read(p)==v

def common(d,h):
 os.environ['MAFIRST_FUTURE_HIDDEN_ONLY_DATASET_20261107']=d;os.environ['MAFIRST_FUTURE_HIDDEN_ONLY_HORIZON_20261107']=str(h)
 c=importlib.import_module('tmp.'+NEW+'_common')
 return c if (c.DATASET,c.HORIZON)==(d,h) else importlib.reload(c)

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
 m.to(device).eval();control.to(device).eval();x=torch.randn(2,96,c.CHANNELS,device=device);captures=[];original_delta=future_model.future_rotation_delta
 def capture(dec,blocks,angle,upper,basis,head):
  delta=original_delta(dec,blocks,angle,upper,basis,head);assert blocks.shape[:3]==(2*c.CHANNELS,h//4,4) and delta.shape==(2*c.CHANNELS,h//4,dec.modes);assert torch.isfinite(delta).all() and (delta.abs()<=.25*torch.minimum(angle,upper-angle)+1e-7).all();captures.append((dec,blocks.detach().clone(),angle.detach().clone(),upper,basis,head));return delta
 future_model.future_rotation_delta=capture
 try:
  with torch.no_grad():y=m(x);z=control(x);torch.testing.assert_close(y,z,atol=1e-6,rtol=1e-5)
  assert y.shape==(2,h,c.CHANNELS) and torch.isfinite(y).all() and len(captures)==4;assert all(q['future_hidden_conditioning'] for q in m.last_gate_context_diagnostics)
  dec,blocks,angle,upper,basis,head=captures[0]
  with torch.no_grad():
   head.weight.normal_(0,.02);a=original_delta(dec,blocks,angle,upper,basis,head);b=original_delta(dec,blocks+torch.randn_like(blocks)*.1,angle,upper,basis,head);dependence=float((a-b).abs().max());assert dependence>1e-9
   altered=blocks.clone();altered[:,0]+=.25;local=original_delta(dec,altered,angle,upper,basis,head);assert float((a[:,0]-local[:,0]).abs().max())>1e-9 and torch.equal(a[:,1:],local[:,1:]);change=float((a[:,1:]-a[:,:-1]).abs().max());assert change>1e-9;head.weight.zero_()
  m.zero_grad(set_to_none=True);m.train();m(x).square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());norms=[float(a.weight.grad.norm()) for a in m.position_frequency_heads];assert all(a>0 for a in norms)
 finally:future_model.future_rotation_delta=original_delta
 tr,va=f.dataset(ARM,'train'),f.dataset(ARM,'val');assert len(tr)==c.TRAIN_WINDOWS and len(va)==c.VAL_WINDOWS
 import pandas as pd
 from sklearn.preprocessing import StandardScaler
 cfg=f.configuration(ARM);data=pd.read_csv(Path(cfg.root_path)/cfg.data_path);expected=StandardScaler().fit(data.iloc[:c.TRAIN_END,1:].values);assert len(data)==c.VAL_END and np.array_equal(tr.scaler.mean_,expected.mean_) and np.array_equal(tr.scaler.scale_,expected.scale_)
 record=dict(status='verified',dataset=d,horizon=h,device=device,shape=list(y.shape),parameters=sum(p.numel() for p in m.parameters()),common_own_horizon_untrained_states_exact=True,construction_CPU_Python_NumPy_RNG_matches_original=True,construction_CPU_RNG_sha256=hashlib.sha256(crng.numpy().tobytes()).hexdigest(),neutral_prediction_matches_original=dict(atol=1e-6,rtol=1e-5,max_abs=float((y-z).abs().max())),historical_future_hidden_forward_function_identical=True,actual_four_scale_frequency_head_condition='future_block_mean_and_last',activated_future_block_dependence_max_abs=dependence,changing_one_future_block_only_changes_that_delta_block_exact=True,activated_temporal_change_max_abs=change,frequency_head_gradient_norms=norms,all_parameter_gradients_finite=True,no_learnable_period_offsets=True,fixed_normalized_four_cosine_basis_exact=True,per_step_margin_verified=True,official_train_prefix_scaler_verified=True,train_windows=len(tr),full_val_windows=len(va),test_evaluations=0,preparation_weights_discarded=True,created=now())
 c.atomic(c.G/f'{device}_verification.json',record);print(json.dumps(record),flush=True)

def arm(d,h):
 c=common(d,h);c.guard()
 with locked():
  with (c.G/'arm_once.json').open('x') as f:json.dump(identity(),f)
 c.run('CPU_route',[c.PY,'-B','-u',str(Path(__file__).with_name(NEW+'.py').resolve()),'verify','--dataset',d,'--horizon',str(h),'--device','cpu'])
 runner=[c.PY,'-B','-u',str(ROOT/'tmp'/('run_'+NEW+'.py')),'--arm',ARM];wait_resources(c)
 c.run('CUDA_route',[c.PY,'-B','-u',str(Path(__file__).with_name(NEW+'.py').resolve()),'verify','--dataset',d,'--horizon',str(h),'--device','cuda'])
 c.run('actual128_memory',runner+['--phase','memory']);wait_resources(c);pre=c.G/'preflight'/ARM
 for name,directory,extra in [('continuous','continuous',[]),('stop1','resumed',['--stop-after','1']),('resume','resumed',['--resume'])]:c.run('preflight_'+name,runner+['--phase','preflight','--output',str(pre/directory)]+extra)
 c.run('preflight_saved_exact',[c.PY,'-B','-u',str(ROOT/'tmp'/('verify_'+NEW+'_preflight.py'))]);wait_resources(c)
 c.run('formal_train',runner+['--phase','full'])
 c.run('official_test',[c.PY,'-B','-u',str(ROOT/'tmp'/('test_'+NEW+'.py')),'--dataset',d])
 c.run('saved_review',[c.PY,'-B','-u',str(ROOT/'tmp'/('review_'+NEW+'.py')),'--dataset',d])
 result=read(c.OUT[ARM]/'official_test_result.json');assert result['dataset']==d and result['horizon']==h and result['epochs']==100 and result['windows']==c.EVAL_WINDOWS and result['official_test_evaluations']==1
 audit=read(c.OUT[ARM]/'full_decay_completion_audit.json');assert audit['source_epochs']==0 and audit['total_epochs']==100 and audit['total_updates']==100*c.UPDATES_PER_EPOCH
 c.guard()
 c.atomic(c.G/'completion_verification.json',dict(status='verified',test_metrics=result,full_decay_audit=audit,created=now()))
 c.atomic(c.G/'status.json',dict(status='completed',process=identity(),updated=now()))

