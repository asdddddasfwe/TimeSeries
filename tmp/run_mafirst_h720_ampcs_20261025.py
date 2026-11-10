"""One strict D output-kernel substitution; four original unit losses, one fresh full run."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import sys,argparse,time,math,random,json,traceback
import numpy as np,pandas as pd,torch
from sklearn.preprocessing import StandardScaler
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from tmp.mafirst_h720_ampcs_20261025_common import *
from tmp.mafirst_h720_ampcs_20261025_factory import fresh,optimizer_groups
from tmp.run_mafirst_kernel_h720_distancefit_20261024 import Windows,batches,seed,groups_audit,gradient_capture
from tmp.evaluate_mafirst_h720_ampcs_20261025 import evaluate as diagnostic_evaluate
from utils.amplitude_kernel_local_cs_20261025 import training_terms,gradient_summary,WEIGHTS
def loss_terms(p,y,m):return training_terms(p,y,m.predicted_gram)[0]
CH=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'];torch.set_num_threads(2)
def save(p,v):
 with locked():
  t=p.with_suffix(p.suffix+'.tmp');torch.save(v,t);t.replace(p)
def point_metrics(p,y):
 e=(p.detach().double()-y.detach().double());per=e.square().mean(1)
 return dict(mse=float(e.square().mean()),mae=float(e.abs().mean()),max_abs_error=float(e.abs().max()),max_channel_mse=float(per.max()),per_channel_mse=per.cpu().tolist())
def data_for():
 plan=read(PLAN);assert sha(R/plan['train_val_prefix'])==plan['train_val_prefix_sha256']
 raw=pd.read_csv(R/plan['train_val_prefix'])[CH].to_numpy();assert len(raw)==46080
 scaler=StandardScaler().fit(raw[:34560]);a=scaler.transform(raw).astype('float32')
 return Windows(a,720,34560,46080,'train'),Windows(a,720,34560,46080,'val'),scaler,a
def wait_resources():
 while True:
  d=resources();atomic(G/'resources.json',dict(updated=now(),**d))
  if d['ready']:return
  atomic(G/'status.json',dict(status='waiting_resources',resources=d,process=identity()));time.sleep(60)
def configure():
 seed(2021);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False)
 assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.75)
def hook_calls(m):
 calls={'norm':0,'denorm':0}
 def hook(mod,args,out):calls[args[1]]+=1
 h=m.normalize.register_forward_hook(hook)
 def forward(x):
  before=dict(calls);p=m(x);assert calls['norm']-before['norm']==calls['denorm']-before['denorm']==1;return p
 return forward,h
def gradients(m):
 grads=[p.grad for p in m.parameters()];assert len(grads)==227 and all(v is not None and torch.isfinite(v).all() for v in grads)
 return dict(parameters_with_finite_gradients=227,raw_norm=sum(float(v.double().square().sum()) for v in grads)**.5,raw_not_Adam_update=True)
def memory():
 assert not (G/'frozen_sources.json').exists();guard();wait_resources();configure();o=G/'development_cuda';o.mkdir(exist_ok=False)
 m,c=fresh(o/'logs');m.cuda().train();opt=torch.optim.Adam(optimizer_groups(m),lr=.001);train,_,_,_=data_for();fw,h=hook_calls(m);torch.cuda.reset_peak_memory_stats();start=time.monotonic();rows=[]
 for step,(x,y) in enumerate(batches(train,1),1):
  seed(2021010000+step-1);x,y=x.cuda(),y.cuda();opt.zero_grad(set_to_none=True);terms=loss_terms(fw(x),y,m);loss=sum(WEIGHTS[n]*v for n,v in terms.items());
  if step==1:atomic(o/'four_term_raw_gradients.json',gradient_summary(terms,m))
  loss.backward();gg=gradients(m);opt.step();assert all(torch.isfinite(p).all() for p in m.parameters());rows.append(dict(step=step,batch=len(x),loss=float(loss.detach()),**gg))
  if step==32:break
 torch.cuda.synchronize();h.remove();v=dict(status='verified',steps=32,batch_size=32,checkpoint_saved=False,weights_discarded=True,test_evaluations=0,validation_evaluations=0,applied_loss=WEIGHTS,all227_gradients_finite=True,normalization_once=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,duration_seconds=time.monotonic()-start,rows=rows);atomic(o/'verification.json',v);print(json.dumps({k:v[k] for k in ['status','steps','peak_allocated_mib','peak_reserved_mib','duration_seconds']}),flush=True)
def point_evaluate(m,ds):
 m.eval();sq=ab=n=0
 with torch.no_grad():
  for x,y in batches(ds):
   p=m(x.cuda());e=p.double()-y.cuda().double();assert torch.isfinite(e).all();sq+=float(e.square().sum());ab+=float(e.abs().sum());n+=e.numel()
 return dict(mse=sq/n,mae=ab/n,windows=len(ds),elements=n)
def full():
 guard();assert read(G/'startup_integrity.json')['status']=='verified';assert read(G/'development_cuda/verification.json')['status']=='verified';wait_resources();configure();plan=read(PLAN);job=plan['jobs'][0];o=R/job['output']
 with locked():
  with (G/'formal_once.json').open('x') as f:json.dump(identity(),f)
 o.mkdir(exist_ok=False);atomic(o/'process.json',identity());atomic(o/'protocol.json',plan);atomic(o/'backend_execution.json',dict(cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),matches_D_declared_source_defaults=True,historical_D_process_flags_not_recorded=True));m,c=fresh(o/'logs');m.cuda();groups=optimizer_groups(m);opt=torch.optim.Adam(groups,lr=.001);train,val,scaler,data=data_for();fw,h=hook_calls(m)
 save(o/'initial.pt',m.state_dict());atomic(o/'initialization.json',m.initialization_audit);atomic(o/'model_config.json',vars(c));atomic(o/'optimizer_groups.json',groups_audit(m,groups));atomic(o/'data_manifest.json',dict(train_end=34560,val_end=46080,test_end=57600,train_windows=len(train),val_windows=len(val),scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),train_val_prefix_sha256=sha(G/'train_val_prefix.csv')))
 atomic(o/'loss_definition_receipt.json',dict(status='verified',terms=list(WEIGHTS),weights=WEIGHTS,exact_D_four_loss_names_and_external_weights=True,only_final_output_kernel_changed=True,head_gram_mse_weight=1.,head_local_cs_weight=1.,output_local_cs_weight=1.,point_mse_weight=1.,output_gram_mse_weight=0.,extra_diagonal_mse_weight=0.,time_mmd_weight=0.,distance_huber_weight=0.,ordinary_MMD_added=False,four_labelled_output_kernel_channels=True,diagonal_excluded_from_localCS=True,kernel_sigma=1.,softplus_scale=1.,window_size=12,stride=6,windows_per_channel=841,stabilizer_C=.0009,independent_head_and_its_RBF_targets_unchanged=True,target_detached=True,architecture_unchanged=True,underlying_kernel_identifiability_not_loss_identifiability=True));atomic(o/'normalization_flow_receipt.json',dict(status='verified',norm_once_before_pooling=True,denorm_once_after_sum=True,one_affine_pair=True,no_future_statistics=True))
 curves=[];best=float('inf');chosen=stale=updates=0;start=time.monotonic()
 for epoch in range(1,31):
  m.train();lr=.001*.5*(1+math.cos(math.pi*(epoch-1)/30))
  for group in opt.param_groups:group['lr']=lr
  total=seen=0.;term_totals=np.zeros(4);loader=batches(train,epoch);order_sha=__import__('hashlib').sha256(np.asarray(loader.dataset.indices,dtype=np.int64).tobytes()).hexdigest()
  for step,(x,y) in enumerate(loader,1):
   seed(2021000000+epoch*10000+step-1);x,y=x.cuda(),y.cuda();opt.zero_grad(set_to_none=True);p=fw(x);terms=loss_terms(p,y,m);loss=sum(WEIGHTS[n]*v for n,v in terms.items());assert torch.isfinite(loss)
   if step in (1,200,400,600,800,1000):
    with (o/'loss_gradients.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,step=step,scopes=gradient_summary(terms,m)))+'\n')
   loss.backward();gg=gradients(m);opt.step();updates+=1;total+=float(loss.detach())*len(x);term_totals+=np.array([float(v.detach()) for v in terms.values()])*len(x);seen+=len(x)
   if step in (1,200,400,600,800,1000):
    with (o/'gradients.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,step=step,terms=list(WEIGHTS),weights=WEIGHTS,**gg))+'\n')
   if step==1 or step%50==0:atomic(G/'status.json',dict(status='formal_running',process=identity(),epoch=epoch,step=step,steps=len(loader),completed_epochs=epoch-1,best_val_mse=None if not math.isfinite(best) else best,best_epoch=chosen,elapsed_seconds=time.monotonic()-start))
   if step%50==0:assert all(torch.isfinite(p).all() for p in m.parameters()) and all(torch.isfinite(v).all() for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor))
  atomic(G/'status.json',dict(status='validation',process=identity(),epoch=epoch));measured=point_evaluate(m,val);improved=measured['mse']<best
  if improved:best=measured['mse'];chosen=epoch;save(o/'best.pt',m.state_dict())
  stale=0 if improved else stale+1;delta=m.global_seasonal_delta.detach().double();off=delta-torch.diag_embed(delta.diagonal());row=dict(epoch=epoch,lr=lr,train_loss=total/seen,train_mse=float(term_totals[0]/seen),train_head_gram_mse=float(term_totals[1]/seen),train_head_local_cs=float(term_totals[2]/seen),train_output_local_cs=float(term_totals[3]/seen),val=measured,order_sha256=order_sha,improved=improved,selected_epoch=chosen,stale=stale,applied_weights=WEIGHTS,affine_parameters={n:p.detach().cpu().tolist() for n,p in m.named_parameters() if n.endswith(('affine_weight','affine_bias'))},global_output_head=dict(delta_frobenius=float(delta.norm()),offdiagonal_frobenius=float(off.norm()),max_abs=float(delta.abs().max())),elapsed_seconds=time.monotonic()-start);curves.append(row)
  (o/'epochs').mkdir(exist_ok=True);save(o/'epochs'/f'epoch{epoch:02d}.pt',m.state_dict());save(o/'last.pt',dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,chosen=chosen,stale=stale,updates=updates,curves=curves,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=dict(algorithm=np.random.get_state()[0],keys=np.random.get_state()[1].tolist(),position=int(np.random.get_state()[2]),has_gauss=int(np.random.get_state()[3]),cached_gaussian=float(np.random.get_state()[4]))));atomic(o/'curves.json',curves);print(json.dumps(dict(epoch=epoch,train_loss=total/seen,train_mse=float(term_totals[0]/seen),train_head_gram_mse=float(term_totals[1]/seen),train_head_local_cs=float(term_totals[2]/seen),train_output_local_cs=float(term_totals[3]/seen),val_mse=measured['mse'],best=best,chosen=chosen,stale=stale,elapsed_seconds=time.monotonic()-start)),flush=True)
  if stale>=5:break
 h.remove();assert stale==5 or epoch==30;m.load_state_dict(torch.load(o/'best.pt',map_location='cuda',weights_only=True));m.eval();selected_sha=sha(o/'best.pt');atomic(o/'freeze.json',dict(selected_epoch=chosen,checkpoint_sha256=selected_sha,epochs=epoch,updates=updates,stale=stale,maximum=30,patience=5,strict_improvement=True))
 atomic(G/'status.json',dict(status='saving_selected_validation',process=identity()));valmetrics=diagnostic_evaluate(m,val,'cuda',o/'selected_val');assert abs(valmetrics['mse']-best)<1e-10
 snaps={}
 for begin in (0,8000,16000):
  x=torch.from_numpy(np.stack([data[i:i+96] for i in range(begin,begin+8)])).cuda();y=torch.from_numpy(np.stack([data[i+96:i+816] for i in range(begin,begin+8)])).cuda();snaps[str(begin)]=gradient_capture(m,x,y,o,f'gradient_snapshot_{begin}')
 atomic(o/'selected_gradient_snapshots.json',snaps)
 with (o/'test_armed.json').open('x') as f:json.dump(dict(checkpoint_sha256=selected_sha,created=now()),f)
 assert sha(R/plan['full_csv'])==plan['full_csv_sha256'];raw=pd.read_csv(R/plan['full_csv'])[CH].to_numpy();scaled=scaler.transform(raw).astype('float32');test=Windows(scaled,720,46080,57600,'val')
 with (o/'test_started.json').open('x') as f:json.dump(dict(checkpoint_sha256=selected_sha,created=now()),f)
 atomic(G/'status.json',dict(status='unique_test_running',process=identity()));testmetrics=diagnostic_evaluate(m,test,'cuda',o/'test');assert sha(o/'best.pt')==selected_sha;atomic(o/'test_receipt.json',dict(evaluations=1,windows=len(test),checkpoint_sha256=selected_sha,metrics=testmetrics));atomic(o/'completion.json',dict(status='completed_verified',job=job,validation=valmetrics,test=testmetrics,test_evaluations=1,epochs=epoch,selected_epoch=chosen,updates=updates,checkpoint_sha256=selected_sha,smoke=False,duration_seconds=time.monotonic()-start,D_four_unit_weights=WEIGHTS,only_output_kernel_changed=True,head_supervision_unchanged=True));atomic(G/'status.json',dict(status='formal_completed_pending_saved_review',process=None,selected_epoch=chosen,epochs=epoch,test=testmetrics));print(json.dumps(dict(status='completed',epochs=epoch,selected_epoch=chosen,test_mse=testmetrics['mse'],test_mae=testmetrics['mae'])),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['memory','full'],required=True);args=ap.parse_args()
 try:globals()[args.phase]()
 except BaseException as e:atomic(G/('development_failure.json' if args.phase=='memory' else 'failure.json'),dict(status='failed_stopped',phase=args.phase,error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
