"""One fresh original-model experiment; only small preflight permits resume."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import sys,argparse,time,random,math,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from torch.utils.data import Subset
from tmp.distdf_h192_h336_20261026_common import *
from tmp import distdf_h192_h336_20261026_factory as original
import importlib
ARM='h192';H=192
def adapter():return original
def fresh(p):return adapter().fresh(ARM,p)
def optimizer_groups(m):return adapter().optimizer_groups(m)
def groups_audit(m,g):return adapter().groups_audit(ARM,m,g)
def auxiliary(p,y,x):
 if ARM=="multitime":return mix_parts(p,y)
 from utils.distdf_joint_wasserstein import distdf_loss
 v=distdf_loss(p,y,x);return v,dict(distdf_distance=float(v.detach()),var_weight=.002,reg=.005,joint_history_future=True,batch_dependent=True)
def combined(mse,aux):return mse+aux if ARM=="multitime" else read(PLAN)['arms'][ARM]['point_mse_weight']*mse+read(PLAN)['arms'][ARM]['DistDF_weight']*aux
from utils.time_position_multiband_patch_mmd_20261026 import parts as mix_parts
from utils.time_position_patch_mmd import parts as old_parts
torch.set_num_threads(2)
def seed(v):original.seed(v)
def configure(device):
 seed(2021);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False)
 if device=='cuda':assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.6)
def append(p,row):
 with locked():
  with p.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
def gradients(m,mse,mmd):
 ps=list(m.parameters());vs=[]
 for term in (mse,mmd):
  gs=torch.autograd.grad(term,ps,retain_graph=True,allow_unused=True)
  assert all(g is None or torch.isfinite(g).all() for g in gs)
  vs.append(torch.cat([(torch.zeros_like(p) if g is None else g).detach().double().flatten() for p,g in zip(ps,gs)]))
 a,b=vs;an,bn=float(a.norm()),float(b.norm())
 return dict(point_raw_norm=an,mmd_raw_norm=bn,mmd_to_point_norm=bn/max(an,1e-30),cosine=float(a.dot(b)/max(an*bn,1e-30)),raw_not_Adam_update=True)
def finite(m):
 expected=[] if ARM=='multitime' else read(G/'reference_initial'/(ARM+'_inactive.json'))
 inactive=[n for n,p in m.named_parameters() if p.grad is None]
 assert inactive==expected,inactive
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
 return True
def data(cfg,small):
 tr=original.dataset(cfg,'train',64 if small else 0);va=original.dataset(cfg,'val',32 if small else 0)
 return tr,va
def evaluate(m,ds,device,folder=None,relations=False):
 m.eval();sq=ab=n=offset=0;arrays={};sums=np.zeros(8);native_sum=0.
 if folder:
  folder.mkdir(exist_ok=False)
  for k,tail in ([('prediction',(H,7)),('truth',(H,7)),('history',(96,7)),('components',(H,7,8)),('history_mean',(1,7)),('inverse_offset',(1,7)),('inverse_scale',(1,7))] if ARM=='multitime' else [('prediction',(H,7)),('truth',(H,7)),('history',(96,7))]):
   arrays[k]=np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype='float32',shape=(len(ds),*tail))
 with torch.no_grad():
  for x,y,_,_ in original.loader(ds):
   x,y=x.float().to(device),y[:,-H:].float().to(device)
   if folder:
    if ARM=='multitime':
     c=m.predict_components(x);p=m.normalize(c[...,:4].sum(-1)+c[...,4:].sum(-1),'denorm')
     values=dict(prediction=p,truth=y,history=x,components=c*m.normalize.stdev.unsqueeze(-1),history_mean=m.normalize.mean,inverse_offset=m.normalize.mean,inverse_scale=m.normalize.stdev)
    else:
     p=m(x);values=dict(prediction=p,truth=y,history=x)
    for k,v in values.items():arrays[k][offset:offset+len(x)]=v.cpu().numpy()
   else:p=m(x)
   e=p.double()-y.double();assert torch.isfinite(e).all();sq+=float(e.square().sum());ab+=float(e.abs().sum());n+=e.numel()
   if relations:
    _,a=mix_parts(p,y);_,b=old_parts(p,y)
    if ARM!="multitime":native_sum+=float(auxiliary(p,y,x)[0])*len(x)
    sums+=np.array([a[k] for k in ('time_position_mmd','mmd_prediction_self','mmd_target_self','mmd_cross')]+[b[k] for k in ('time_position_mmd','mmd_prediction_self','mmd_target_self','mmd_cross')])*len(x)
   offset+=len(x)
 for a in arrays.values():a.flush()
 v=dict(mse=sq/n,mae=ab/n,windows=len(ds),elements=n)
 if relations:v.update(multitime_mmd=dict(zip(('mmd2','pp','yy','py'),map(float,sums[:4]/len(ds)))),original_mmd=dict(zip(('mmd2','pp','yy','py'),map(float,sums[4:]/len(ds)))))
 if relations and ARM!="multitime":v.update(distdf_distance=native_sum/len(ds),distdf_evaluation_batch_size=32)
 return v
def rng():
 a=np.random.get_state()
 return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],python=random.getstate(),numpy=dict(algorithm=a[0],keys=a[1].tolist(),position=a[2],has_gauss=a[3],cached_gaussian=a[4]))
def restore(r):
 torch.set_rng_state(r['cpu'].cpu());random.setstate(r['python']);a=r['numpy'];np.random.set_state((a['algorithm'],np.array(a['keys'],dtype=np.uint32),a['position'],a['has_gauss'],a['cached_gaussian']))
 if torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in r['cuda']])
def wait_resources():
 while True:
  d=resources(ARM);atomic(G/'resources.json',dict(updated=now(),**d))
  if d['ready']:return
  atomic(G/'status.json',dict(status='waiting_resources',process=identity(),resources=d));time.sleep(60)
def memory(device):
 assert not (G/'frozen_sources.json').exists();wait_resources();configure(device);out=G/'memory'/ARM;out.mkdir(parents=True,exist_ok=False)
 m,cfg=fresh(out/'logs');m.to(device).train();opt=torch.optim.Adam(optimizer_groups(m),lr=.001);groups_audit(m,opt.param_groups);tr,_=data(cfg,False);rows=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 for step,(x,y,_,_) in enumerate(original.loader(tr,1),1):
  seed(2021010000+step-1);x,y=x.float().to(device),y[:,-H:].float().to(device);opt.zero_grad(set_to_none=True);p=m(x);mse=(p-y.detach()).square().mean();mmd,md=auxiliary(p,y,x);loss=combined(mse,mmd)
  if step==1:atomic(out/'raw_gradients.json',gradients(m,mse,mmd))
  loss.backward();finite(m);opt.step();rows.append(dict(step=step,loss=float(loss.detach()),**md))
  assert all(torch.isfinite(p).all() for p in m.parameters())
  if step==32:break
 torch.cuda.synchronize();v=dict(status='verified',steps=32,batch_size=32,all_active_gradients_finite=True,arm=ARM,parameter_tensors=len(list(m.parameters())),inactive_tensors=0 if ARM=="multitime" else len(read(G/'reference_initial'/(ARM+'_inactive.json'))),test_evaluations=0,checkpoint_saved=False,weights_discarded=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,duration_seconds=time.monotonic()-start,rows=rows)
 atomic(out/'verification.json',v);print(json.dumps({k:v[k] for k in ('status','peak_allocated_mib','peak_reserved_mib','duration_seconds')}),flush=True)
def train(args):
 small=args.phase=='preflight';device=args.device;out=Path(args.output) if small else O
 assert small or (not args.resume and args.stop_after==0 and args.epochs==30)
 if not small:
  guard();assert read(G/'startup_integrity.json')['status']=='verified' and read(G/'preflight'/ARM/'verification.json')['status']=='verified';wait_resources()
  with locked():
   with (G/(ARM+'_formal_once.json')).open('x') as f:json.dump(identity(),f)
 configure(device)
 if not args.resume:out.mkdir(exist_ok=False)
 assert small or not (out/'test_started.json').exists()
 atomic(out/'process.json',identity());m,cfg=fresh(out/'logs');m.to(device);opt=torch.optim.Adam(optimizer_groups(m),lr=.001);audit=groups_audit(m,opt.param_groups);tr,va=data(cfg,small)
 if not args.resume:
  save(out/'initial.pt',m.state_dict());atomic(out/'model_config.json',vars(cfg));atomic(out/'optimizer_groups.json',audit)
  meta=dict(parameters=sum(p.numel() for p in m.parameters()),parameter_tensors=len(list(m.parameters())),states=len(m.state_dict()),buffers=len(list(m.buffers())))
  atomic(out/'initialization.json',dict(status='verified',arm=ARM,**meta,all_reference_states_exact=True,independent_storage=True,trained_weights_loaded=False,reference_sha256=sha(G/'reference_initial'/(ARM+'.pt'))))
  atomic(out/'backend_execution.json',dict(cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,torch_deterministic=torch.are_deterministic_algorithms_enabled(),historical_process_flags_not_recorded=True))
  definition=dict(status='verified',arm=ARM,architecture_unchanged=True,optimizer_rules='original exact' if ARM=='multitime' else 'matched LayerNorm and bias decay exemption')
  if ARM=='multitime':definition.update(point_mse_weight=1.,multitime_MMD2_weight=1.,sigma_value=1.,time_sigmas=[2.,24.,90.],time_weights=[1/3]*3,time_formula='sum w_s*T_s*mean(T_2)/mean(T_s)',time_mass_matches_original_sigma2=True,unit_diagonal_not_preserved_or_required=True,PP_YY_PY_same_kernel=True,diagonals_included=True,N_squared_reduction=True,FP64_internal=True,target_detached=True,ordinary_valueonly_MMD=False,CS_or_Gram_MSE_added=False,all204_optimizer_rules_exact=True,original_four_normalizations_preserved=True)
  else:definition.update(official_horizon=H,d_model=24,cf_dim=128,point_mse_weight=read(PLAN)['arms'][ARM]['point_mse_weight'],DistDF_weight=read(PLAN)['arms'][ARM]['DistDF_weight'],var_weight=.002,mean_weight=1.,reg=.005,eps=1e-8,joint_history_future=True,batch_dependent=True,history_and_target_detached=True,not_exact_Bures=True,not_MMD=True,official_model_unchanged=True,matched_training_not_official_recipe=True)
  atomic(out/'loss_definition_receipt.json',definition)
  best=float('inf');chosen=stale=updates=0;curves=[];start_epoch=1
 else:
  assert small and read(out/'status.json')['status']=='controlled_stop'
  last=torch.load(out/'last.pt',map_location=device,weights_only=True);m.load_state_dict(last['model']);opt.load_state_dict(last['optimizer']);best,chosen,stale,updates,curves,start_epoch=last['best'],last['chosen'],last['stale'],last['updates'],last['curves'],last['epoch']+1;restore(last['rng'])
 start=time.monotonic()
 for epoch in range(start_epoch,args.epochs+1):
  m.train();lr=.001*.5*(1+math.cos(math.pi*(epoch-1)/30))
  for g in opt.param_groups:g['lr']=lr
  sums=np.zeros(3);seen=0;loader=original.loader(tr,epoch);order_sha=__import__('hashlib').sha256(np.asarray(loader.dataset.indices,dtype=np.int64).tobytes()).hexdigest()
  for step,(x,y,_,_) in enumerate(loader,1):
   seed(2021000000+epoch*10000+step-1);x,y=x.float().to(device),y[:,-H:].float().to(device);opt.zero_grad(set_to_none=True);p=m(x);mse=(p-y.detach()).square().mean();mmd,md=auxiliary(p,y,x);loss=combined(mse,mmd);assert torch.isfinite(loss)
   if step in (1,200,400,600,800,1000):append(out/'gradients.jsonl',dict(epoch=epoch,step=step,point_mse=float(mse.detach()),auxiliary_loss=float(mmd.detach()),arm=ARM,terms=md,**gradients(m,mse,mmd)))
   loss.backward();finite(m);opt.step();updates+=1;sums+=np.array([float(loss.detach()),float(mse.detach()),float(mmd.detach())])*len(x);seen+=len(x)
   if step==1 or step%50==0:
    assert all(torch.isfinite(p).all() for p in m.parameters())
    atomic((out if small else G)/'status.json',dict(status='preflight_running' if small else 'formal_running',arm=ARM,process=identity(),epoch=epoch,step=step,steps=len(loader),completed_epochs=epoch-1,best_epoch=chosen,best_val_mse=None if not math.isfinite(best) else best,elapsed_seconds=time.monotonic()-start))
  measured=evaluate(m,va,device);improved=measured['mse']<best
  if improved:best=measured['mse'];chosen=epoch;save(out/'best.pt',m.state_dict())
  stale=0 if improved else stale+1;row=dict(epoch=epoch,lr=lr,train_loss=float(sums[0]/seen),train_mse=float(sums[1]/seen),train_auxiliary=float(sums[2]/seen),arm=ARM,val=measured,order_sha256=order_sha,improved=improved,selected_epoch=chosen,stale=stale,training_windows=seen);curves.append(row)
  save(out/'epochs'/f'epoch{epoch:02d}.pt',m.state_dict());save(out/'last.pt',dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,chosen=chosen,stale=stale,updates=updates,curves=curves,rng=rng()));atomic(out/'curves.json',curves);print(json.dumps(row),flush=True)
  if args.stop_after and epoch==args.stop_after:atomic(out/'status.json',dict(status='controlled_stop',epoch=epoch));return
  if stale>=5:break
 if not small:assert stale==5 or epoch==30
 m.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True));selected_sha=sha(out/'best.pt');atomic(out/'freeze.json',dict(selected_epoch=chosen,epochs=epoch,updates=updates,stale=stale,checkpoint_sha256=selected_sha,epoch0_selected=False,selection='strict own fullval point MSE',patience=5))
 valmetrics=evaluate(m,va,device,out/'selected_val',True);assert abs(valmetrics['mse']-best)<1e-10
 if small:
  atomic(out/'completion.json',dict(status='preflight_verified',validation=valmetrics,test_evaluations=0));atomic(out/'status.json',dict(status='preflight_verified'));return
 plan=read(PLAN);assert sha(R/plan['full_csv'])==plan['full_csv_sha256']
 with locked():
  with (out/'test_started.json').open('x') as f:json.dump(dict(created=now(),checkpoint_sha256=selected_sha),f)
 test=original.dataset(original.config(ARM,R/'dataset/ETT-small'),'test')
 atomic(G/'status.json',dict(status='unique_test_running',arm=ARM,process=identity()))
 measured=evaluate(m,test,device,out/'test',True);assert sha(out/'best.pt')==selected_sha
 atomic(out/'test_receipt.json',dict(evaluations=1,windows=len(test),checkpoint_sha256=selected_sha,metrics=measured))
 atomic(out/'completion.json',dict(status='completed_verified',validation=valmetrics,test=measured,epochs=epoch,selected_epoch=chosen,updates=updates,test_evaluations=1,checkpoint_sha256=selected_sha,duration_seconds=time.monotonic()-start))
 atomic(G/'status.json',dict(status='formal_completed_pending_saved_review',arm=ARM,process=None));print(json.dumps(dict(status='completed',test=measured)),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['h192','h336'],required=True);ap.add_argument('--phase',choices=['memory','preflight','full'],required=True);ap.add_argument('--device',default='cuda',choices=['cuda','cpu']);ap.add_argument('--output');ap.add_argument('--epochs',default=30,type=int);ap.add_argument('--stop-after',default=0,type=int);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 global ARM,O,H
 ARM=args.arm;O=OUTPUTS[ARM];H=HORIZONS[ARM]
 try:
  if args.phase=='memory':memory(args.device)
  else:train(args)
 except BaseException as e:
  atomic(G/('failure.json' if args.phase=='full' else 'development_failure.json'),dict(status='failed_stopped',arm=ARM,phase=args.phase,error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
