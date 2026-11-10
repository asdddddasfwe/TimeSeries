"""One fresh full-sequence model, fixed4train windows, outputGramMSE only."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import sys,json,time,random,argparse,traceback
import numpy as np,pandas as pd,torch
from sklearn.preprocessing import StandardScaler
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from tmp.mafirst_h720_outputgram_overfit_20261025_common import *
from tmp.mafirst_kernel_h720_distancefit_20261024_factory import fresh as reference_fresh
from models.MAFirstKernelTransferAblation import value_gram
CH=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
torch.set_num_threads(2)
def data():
 plan=read(PLAN);assert sha(G/'train_prefix.csv')==plan['train_prefix_sha256']
 raw=pd.read_csv(G/'train_prefix.csv')[CH].to_numpy();assert raw.shape==(34560,7)
 scaler=StandardScaler().fit(raw);z=scaler.transform(raw).astype('float32');origins=plan['origins']
 assert len(origins)==4 and all(0<=i and i+816<=34560 for i in origins) and all(origins[i]+816<=origins[i+1] for i in range(3))
 x=torch.from_numpy(np.stack([z[i:i+96] for i in origins]));y=torch.from_numpy(np.stack([z[i+96:i+816] for i in origins]));return x,y,scaler

def grams(sequence):
 if sequence.ndim!=3 or sequence.shape[1]!=720 or sequence.shape[2]!=7:raise ValueError('B,720,7 required')
 b,h,c=sequence.shape;return value_gram(sequence.transpose(1,2).reshape(b*c,180,4))
def objective(prediction,target_gram):
 g=grams(prediction);n=g.shape[-1];mask=~torch.eye(n,dtype=torch.bool,device=g.device)
 loss=(g.double()-target_gram.detach().double()).square()[...,mask].mean().to(prediction.dtype)
 return loss,g

def metrics(p,y,g,gold):
 mask=~torch.eye(180,device=p.device,dtype=torch.bool);a=g.double()[...,mask];b=gold.double()[...,mask];err=(a-b).square().mean(-1);ac=a-a.mean(-1,keepdim=True);bc=b-b.mean(-1,keepdim=True);av=ac.square().mean(-1);bv=bc.square().mean(-1);den=(av*bv).sqrt();rho=torch.where(den>1e-12,(ac*bc).mean(-1)/den.clamp_min(1e-12),torch.zeros_like(den));e=p.double()-y.double()
 return dict(gram_mse=float(err.mean()),max_matrix_mse=float(err.max()),per_matrix_mse=err.reshape(-1,7).cpu().tolist(),mean_correlation=float(rho.mean()),per_matrix_correlation=rho.reshape(-1,7).cpu().tolist(),prediction_gram_std=av.sqrt().reshape(-1,7).cpu().tolist(),target_gram_std=bv.sqrt().reshape(-1,7).cpu().tolist(),point_mse_diagnostic=float(e.square().mean()),point_mae_diagnostic=float(e.abs().mean()),target_flat_count=int((bv<=1e-8).sum()),matrix_count=err.numel())

def fresh(folder):
 m,c=reference_fresh(720,'h720_distance_fit',folder/'logs');ref=torch.load(G/'reference_initial/h720.pt',map_location='cpu',weights_only=True)
 assert len(ref)==259 and all(torch.equal(v,ref[n]) and v.data_ptr()!=ref[n].data_ptr() for n,v in m.state_dict().items())
 assert all(p.requires_grad for p in m.parameters()) and len(list(m.parameters()))==227 and sum(p.numel() for p in m.parameters())==906416
 m.initialization_audit.update(diagnostic_group=str(G.relative_to(R)),factory_reference_job='h720_distance_fit',new_run_weights='exact common untrained259, not old trained or preflight weights')
 m.eval();assert not any(mod.training for mod in m.modules());return m,c

def optimizer(m):return torch.optim.Adam(m.parameters(),lr=.001,weight_decay=0.)
def save(p,v):
 with locked():
  t=p.with_suffix(p.suffix+'.tmp');torch.save(v,t);t.replace(p)
def setting():
 random.seed(2021);np.random.seed(2021);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
def wait_resources():
 while True:
  d=resources();atomic(G/'resources.json',dict(updated=now(),**d))
  if d['ready']:return
  print(json.dumps(dict(waiting_resources=d)),flush=True);time.sleep(60)

def preflight():
 assert not (G/'frozen_sources.json').exists() and not (G/'formal_once.json').exists();guard(False);wait_resources();folder=G/'development_cuda';folder.mkdir(exist_ok=False)
 setting();torch.cuda.set_per_process_memory_fraction(.75);m,c=fresh(folder);m.cuda();opt=optimizer(m);x,y,_=data();x,y=x.cuda(),y.cuda();gold=grams(y).detach();torch.cuda.reset_peak_memory_stats();rows=[]
 for step in range(1,4):
  opt.zero_grad(set_to_none=True);p=m(x);loss,g=objective(p,gold);loss.backward();gr=[v.grad for v in m.parameters() if v.grad is not None];assert gr and all(torch.isfinite(v).all() for v in gr);opt.step();torch.cuda.synchronize();rows.append(dict(step=step,gram_mse=float(loss.detach()),grad_norm=sum(float(v.double().square().sum()) for v in gr)**.5,parameters_with_grad=len(gr)))
 v=dict(status='verified',steps=3,batch_size=4,device='CUDA',fresh_weights_discarded=True,checkpoint_saved=False,validation_or_test_used=False,only_applied_loss='output_gram_mse',model_eval_mode_autograd_on=True,dropout_off=True,weight_decay=0.,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,all_observed_gradients_finite=True,rows=rows)
 atomic(folder/'verification.json',v);print(json.dumps(v),flush=True)

def thresholds_met(chosen,plan):
 return chosen['gram_mse']<=plan['success_mean_MSE'] and chosen['max_matrix_mse']<=plan['success_max_matrix_MSE']

def formal():
 assert (G/'frozen_sources.json').exists();guard(False);assert read(G/'startup_integrity.json')['status']=='verified';assert read(G/'development_cuda/verification.json')['status']=='verified';wait_resources()
 with locked():
  with (G/'formal_once.json').open('x') as f:json.dump(dict(created=now(),process=identity()),f)
  atomic_unlocked(G/'status.json',dict(status='running',process=identity(),started=now(),step=0))
 setting();torch.cuda.set_per_process_memory_fraction(.75);plan=read(PLAN);O=G/'run';O.mkdir(exist_ok=False);m,c=fresh(O);m.cuda();opt=optimizer(m);x,y,scaler=data();x,y=x.cuda(),y.cuda();gold=grams(y).detach();assert not gold.requires_grad
 save(O/'initial.pt',m.state_dict());atomic(O/'model_config.json',vars(c));atomic(O/'initialization.json',m.initialization_audit)
 atomic(O/'data_manifest.json',dict(origins=plan['origins'],train_end=34560,train_prefix_sha256=sha(G/'train_prefix.csv'),scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),only_training_windows=True,validation_windows=0,test_windows=0))
 atomic(O/'loss_definition_receipt.json',dict(status='verified',weights=plan['loss_weights'],objective='only outputGram offdiagonal MSE',all227_parameters_trainable=True,independent_head_only_via_existing_feedback=True,target_detached=True,no_future_target_inputs=True,dropout_off_eval_with_autograd=True,weight_decay=0.,selection='strict min evaluated fixed4window GramMSE',no_point_MSE_selection=True,normalization_unchanged=True))
 calls={'norm':0,'denorm':0}
 def hook(mod,args,out):calls[args[1]]+=1
 handle=m.normalize.register_forward_hook(hook)
 def forward():
  before=dict(calls);p=m(x);assert calls['norm']-before['norm']==calls['denorm']-before['denorm']==1;return p
 history=[];best=float('inf');best_step=0;start=time.monotonic();lastgrad={}
 def check(step):
  nonlocal best,best_step
  with torch.no_grad():p=forward();loss,g=objective(p,gold);z=metrics(p,y,g,gold)
  z.update(step=step,elapsed_seconds=time.monotonic()-start,gradient_diagnostics=lastgrad)
  if z['gram_mse']<best:
   best=z['gram_mse'];best_step=step;save(O/'best.pt',m.state_dict());atomic(O/'best_receipt.json',dict(step=step,gram_mse=best,max_matrix_mse=z['max_matrix_mse'],checkpoint_sha256=sha(O/'best.pt')))
  history.append(z);atomic(O/'curves.json',history);atomic(G/'status.json',dict(status='running',updated=now(),process=identity(),step=step,max_steps=plan['max_steps'],best_step=best_step,best_gram_mse=best,latest=z))
  print(json.dumps({k:z[k] for k in ['step','gram_mse','max_matrix_mse','mean_correlation','point_mse_diagnostic','elapsed_seconds']}),flush=True)
  return z
 z=check(0)
 with torch.no_grad():np.save(O/'initial_prediction.npy',forward().cpu().numpy())
 success=False
 for step in range(1,plan['max_steps']+1):
  opt.zero_grad(set_to_none=True);p=forward();loss,g=objective(p,gold);assert torch.isfinite(loss);loss.backward();gr=[(n,v.grad) for n,v in m.named_parameters() if v.grad is not None];assert gr and all(torch.isfinite(v).all() for n,v in gr)
  if step%25==0 or step==1:lastgrad=dict(norm=sum(float(v.double().square().sum()) for n,v in gr)**.5,parameters_with_grad=len(gr),none_gradient_names=[n for n,v in m.named_parameters() if v.grad is None],raw_loss_gradient_not_Adam_update=True)
  opt.step();assert all(torch.isfinite(v).all() for v in m.parameters())
  if step%plan['check_every']==0 or step==plan['max_steps']:
   z=check(step);success=thresholds_met(read(O/'best_receipt.json'),plan)
   if success:break
 save(O/'final_optimizer.pt',opt.state_dict());save(O/'final.pt',m.state_dict());m.load_state_dict(torch.load(O/'best.pt',map_location='cuda',weights_only=True));m.eval()
 with torch.no_grad():
  comps=m.predict_components(x);p=m.normalize(comps[...,:4].sum(-1)+comps[...,4:].sum(-1),'denorm');_,g=objective(p,gold);final=metrics(p,y,g,gold);scale,offset=m.inverse_coordinates()
  for n,v in dict(prediction=p,truth=y,history=x,components=comps*scale.unsqueeze(-1),inverse_scale=scale,inverse_offset=offset,history_mean=m.normalize.mean).items():np.save(O/(n+'.npy'),v.cpu().numpy())
  np.savez_compressed(O/'gram_matrices.npz',prediction=g.cpu().numpy(),target=gold.cpu().numpy())
 assert abs(final['gram_mse']-best)<1e-10;handle.remove();guard(False)
 v=dict(status='completed',experiment='fixed4trainwindow outputGram-only overfit',steps=step,max_steps=plan['max_steps'],best_step=best_step,success_thresholds_met=success,initial=history[0],best=final,duration_seconds=time.monotonic()-start,validation_evaluations=0,test_evaluations=0,full_data_training_runs=0,trained_model_reused=False,initial_reference_sha256=sha(G/'reference_initial/h720.pt'),best_checkpoint_sha256=sha(O/'best.pt'),CPU_threads=2,device='CUDA',model_eval_mode_with_autograd=True,applied_loss='only output_gram_mse',no_formal_resume_or_retry=True)
 atomic(O/'completion.json',v);atomic(G/'status.json',dict(status='training_completed_pending_saved_review',updated=now(),process=None,best_step=best_step,best_gram_mse=best,success_thresholds_met=success));print(json.dumps(dict(status=v['status'],steps=step,best_step=best_step,best_gram_mse=best,success_thresholds_met=success)),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');ap.add_argument('--formal',action='store_true');args=ap.parse_args();assert args.preflight != args.formal
 try:preflight() if args.preflight else formal()
 except BaseException as e:
  atomic(G/('development_failure.json' if args.preflight else 'failure.json'),dict(status='failed_stopped',error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
