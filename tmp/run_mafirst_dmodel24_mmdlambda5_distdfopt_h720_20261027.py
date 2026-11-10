"""One fresh original-model experiment; only small preflight permits resume."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import sys,argparse,time,random,math,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from torch.utils.data import Subset
from tmp.mafirst_dmodel24_mmdlambda5_distdfopt_h720_20261027_common import *
from tmp.mafirst_dmodel24_mmdlambda5_distdfopt_h720_20261027_factory import fresh,groups_audit,original
from tmp.mafirst_wideband_kernel_20261026 import parts as wide_parts
from tmp.mafirst_dmodel24_lambda5_objective_20261027 import combine,MMD_WEIGHT
from utils.time_position_patch_mmd import parts as old_parts
from tmp.mafirst_dmodel24_lambda5_contract_20261027 import make_optimizer,make_scheduler,training_loader
torch.set_num_threads(2)
def seed(v):original.shared.seed_all(v)
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
 return dict(point_raw_norm=an,mmd_raw_norm=bn,mmd_to_point_norm=bn/max(an,1e-30),cosine=float(a.dot(b)/max(an*bn,1e-30)),raw_not_Adam_update=True,mmd_weight=MMD_WEIGHT,mmd_weighted_norm=MMD_WEIGHT*bn,weighted_mmd_to_point_norm=MMD_WEIGHT*bn/max(an,1e-30))
def finite(m):
 ps=list(m.parameters());assert len(ps)==204 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps)
 return True
def data(cfg,small):
 tr=original.dataset(cfg,'train',512 if small else 0);va=original.dataset(cfg,'val',128 if small else 0)
 return tr,va
def evaluate(m,ds,device,folder=None,relations=False):
 m.eval();sq=ab=n=offset=0;arrays={};sums=np.zeros(8)
 if folder:
  folder.mkdir(exist_ok=False)
  for k,tail in [('prediction',(720,7)),('truth',(720,7)),('history',(96,7)),('components',(720,7,8)),('history_mean',(1,7)),('inverse_offset',(1,7)),('inverse_scale',(1,7))]:
   arrays[k]=np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype='float32',shape=(len(ds),*tail))
 with torch.no_grad():
  for x,y,_,_ in original.loader(ds):
   x,y=x.float().to(device),y[:,-720:].float().to(device)
   if folder:
    c=m.predict_components(x);p=m.normalize(c[...,:4].sum(-1)+c[...,4:].sum(-1),'denorm')
    values=dict(prediction=p,truth=y,history=x,components=c*m.normalize.stdev.unsqueeze(-1),history_mean=m.normalize.mean,inverse_offset=m.normalize.mean,inverse_scale=m.normalize.stdev)
    for k,v in values.items():arrays[k][offset:offset+len(x)]=v.cpu().numpy()
   else:p=m(x)
   e=p.double()-y.double();assert torch.isfinite(e).all();sq+=float(e.square().sum());ab+=float(e.abs().sum());n+=e.numel()
   if relations:
    _,a=wide_parts(p,y);_,b=old_parts(p,y)
    sums+=np.array([a[k] for k in ('time_position_mmd','mmd_prediction_self','mmd_target_self','mmd_cross')]+[b[k] for k in ('time_position_mmd','mmd_prediction_self','mmd_target_self','mmd_cross')])*len(x)
   offset+=len(x)
 for a in arrays.values():a.flush()
 v=dict(mse=sq/n,mae=ab/n,windows=len(ds),elements=n)
 if relations:v.update(wide_mmd=dict(zip(('mmd2','pp','yy','py'),map(float,sums[:4]/len(ds)))),original_mmd=dict(zip(('mmd2','pp','yy','py'),map(float,sums[4:]/len(ds)))))
 return v
def rng():
 a=np.random.get_state()
 return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],python=random.getstate(),numpy=dict(algorithm=a[0],keys=a[1].tolist(),position=a[2],has_gauss=a[3],cached_gaussian=a[4]))
def restore(r):
 torch.set_rng_state(r['cpu'].cpu());random.setstate(r['python']);a=r['numpy'];np.random.set_state((a['algorithm'],np.array(a['keys'],dtype=np.uint32),a['position'],a['has_gauss'],a['cached_gaussian']))
 if torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in r['cuda']])
def wait_resources():
 while True:
  d=resources();atomic(G/'resources.json',dict(updated=now(),**d))
  if d['ready']:return
  atomic(G/'status.json',dict(status='waiting_resources',process=identity(),resources=d));time.sleep(60)
def memory(device):
 wait_resources();configure(device);out=G/'memory';out.mkdir(exist_ok=False)
 m,cfg=fresh(out/'logs');m.to(device).train();opt=make_optimizer(m);groups_audit(m,opt.param_groups);tr,_=data(cfg,False);sched=make_scheduler(opt,263);rows=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 for step,(xx,yy,_,_) in enumerate(training_loader(tr,1),1):
  opt.zero_grad(set_to_none=True);values=[]
  for micro in range(4):
   seed(2021010000+(step-1)*4+micro);x=xx[micro*32:(micro+1)*32].float().to(device);y=yy[micro*32:(micro+1)*32,-720:].float().to(device);p=m(x)
   mse=(p-y.detach()).square().mean();mmd,md=wide_parts(p,y);loss=combine(mse,mmd)
   if step==1 and micro==0:atomic(out/'raw_gradients.json',dict(**gradients(m,mse,mmd),raw_batch_size=32,effective_batch_size=128))
   (loss/4).backward();finite(m);values.append(float(loss.detach()))
  opt.step();sched.step(verbose=False);rows.append(dict(step=step,loss=float(np.mean(values)),lr=opt.param_groups[0]['lr']))
  assert all(torch.isfinite(p).all() for p in m.parameters())
  if step==8:break
 torch.cuda.synchronize();v=dict(status='verified',steps=8,effective_batch_size=128,micro_batch_size=32,accumulations=4,all204_gradients_finite=True,test_evaluations=0,checkpoint_saved=False,weights_discarded=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,duration_seconds=time.monotonic()-start,rows=rows)
 atomic(out/'verification.json',v);print(json.dumps(v),flush=True)

def train(args):
 small=args.phase=='preflight';device=args.device;out=Path(args.output) if small else O
 assert small or (not args.resume and args.stop_after==0 and args.epochs==100)
 if not small:
  guard();assert read(G/'startup_integrity.json')['status']=='verified' and read(G/'preflight/verification.json')['status']=='verified';wait_resources()
  with locked():
   with (G/'formal_once.json').open('x') as f:json.dump(identity(),f)
 configure(device)
 if not args.resume:out.mkdir(exist_ok=False)
 assert small or not (out/'test_started.json').exists()
 atomic(out/'process.json',identity());m,cfg=fresh(out/'logs');m.to(device);opt=make_optimizer(m);audit=groups_audit(m,opt.param_groups);tr,va=data(cfg,small);sched=make_scheduler(opt,4 if small else 263)
 if not args.resume:
  save(out/'initial.pt',m.state_dict());atomic(out/'model_config.json',vars(cfg));atomic(out/'optimizer_groups.json',audit)
  atomic(out/'initialization.json',dict(status='verified',parameters=read(G/'expected_model_card.json')['parameters'],parameter_tensors=204,states=236,buffers=32,all236_dmodel24_scratch_exact=True,independent_storage=True,trained_weights_loaded=False,reference_sha256=sha(G/'reference_initial.pt')))
  atomic(out/'backend_execution.json',dict(cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,torch_deterministic=torch.are_deterministic_algorithms_enabled(),historical_process_flags_not_recorded=True))
  atomic(out/'loss_definition_receipt.json',dict(status='verified',point_mse_weight=1.,wide_timeMMD2_weight=MMD_WEIGHT,sigma_value=2.,sigma_time_patches=24.,patch_dimension=4,kernel_family='RBF',mass_matching=False,no_amplitude_weighting=True,PP_YY_PY_same_kernel=True,diagonals_included=True,N_squared_reduction=True,target_detached=True,ordinary_valueonly_MMD=False,CS_or_Gram_MSE_added=False,architecture_unchanged_from_D24_lambda1=True,only_external_MMD_coefficient_changed=True,official_Adam_all204_parameters=True,original_four_normalizations_preserved=True,affine_parameters=0))
  atomic(out/'optimization_receipt.json',dict(status='verified',source='DistDF official ETTm1/H720',optimizer='Adam',weight_decay=0.,max_lr=.0001,initial_lr=opt.param_groups[0]['lr'],effective_batch_size=128,micro_batch_size=32,accumulations=4,train_drop_last=True,full_validation_and_test=True,schedule='OneCycleLR',epochs=100,pct_start=.4,patience=10,ties_accepted=True,only_external_MMD_coefficient_changed=True,architecture_kernel_optimizer_unchanged=True))
  best=float('inf');chosen=stale=updates=0;curves=[];start_epoch=1
 else:
  assert small and read(out/'status.json')['status']=='controlled_stop'
  last=torch.load(out/'last.pt',map_location=device,weights_only=True);m.load_state_dict(last['model']);opt.load_state_dict(last['optimizer']);sched.scheduler.load_state_dict(last['scheduler']);sched.last_lr=opt.param_groups[0]['lr'];best,chosen,stale,updates,curves,start_epoch=last['best'],last['chosen'],last['stale'],last['updates'],last['curves'],last['epoch']+1;restore(last['rng'])
 start=time.monotonic()
 for epoch in range(start_epoch,args.epochs+1):
  m.train();lr=float(opt.param_groups[0]['lr']);sums=np.zeros(3);seen=0;loader=training_loader(tr,epoch);order_sha=__import__('hashlib').sha256(np.asarray(loader.dataset.indices,dtype=np.int64).tobytes()).hexdigest()
  for step,(xx,yy,_,_) in enumerate(loader,1):
   opt.zero_grad(set_to_none=True);batch_terms=np.zeros(3)
   for micro in range(4):
    seed(2021000000+epoch*10000+(step-1)*4+micro)
    x=xx[micro*32:(micro+1)*32].float().to(device);y=yy[micro*32:(micro+1)*32,-720:].float().to(device)
    p=m(x);mse=(p-y.detach()).square().mean();mmd,md=wide_parts(p,y);loss=combine(mse,mmd);assert torch.isfinite(loss)
    if micro==0 and step in (1,50,100,150,200,250):append(out/'gradients.jsonl',dict(epoch=epoch,step=step,point_mse=float(mse.detach()),wide_mmd=float(mmd.detach()),terms=md,raw_batch_size=32,effective_batch_size=128,**gradients(m,mse,mmd)))
    (loss/4).backward();finite(m);batch_terms+=np.array([float(loss.detach()),float(mse.detach()),float(mmd.detach())])/4
   opt.step();sched.step(verbose=False);updates+=1;sums+=batch_terms*128;seen+=128
   if step==1 or step%25==0:
    assert all(torch.isfinite(p).all() for p in m.parameters())
    atomic((out if small else G)/'status.json',dict(status='preflight_running' if small else 'formal_running',process=identity(),epoch=epoch,step=step,steps=len(loader),completed_epochs=epoch-1,best_epoch=chosen,best_val_mse=None if not math.isfinite(best) else best,elapsed_seconds=time.monotonic()-start,lr=opt.param_groups[0]['lr']))
  measured=evaluate(m,va,device);improved=measured['mse']<=best
  if improved:best=measured['mse'];chosen=epoch;save(out/'best.pt',m.state_dict())
  stale=0 if improved else stale+1;row=dict(epoch=epoch,lr=lr,lr_end=float(opt.param_groups[0]['lr']),scheduler_last_epoch=sched.scheduler.last_epoch,train_loss=float(sums[0]/seen),train_mse=float(sums[1]/seen),train_mmd=float(sums[2]/seen),val=measured,order_sha256=order_sha,improved=improved,selected_epoch=chosen,stale=stale,training_windows=seen);curves.append(row)
  save(out/'epochs'/f'epoch{epoch:02d}.pt',m.state_dict());save(out/'last.pt',dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.scheduler.state_dict(),epoch=epoch,best=best,chosen=chosen,stale=stale,updates=updates,curves=curves,rng=rng()));atomic(out/'curves.json',curves);print(json.dumps(row),flush=True)
  if args.stop_after and epoch==args.stop_after:atomic(out/'status.json',dict(status='controlled_stop',epoch=epoch));return
  if stale>=10:break
 if not small:assert stale==10 or epoch==100
 m.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True));selected_sha=sha(out/'best.pt');atomic(out/'freeze.json',dict(selected_epoch=chosen,epochs=epoch,updates=updates,stale=stale,checkpoint_sha256=selected_sha,epoch0_selected=False,selection='own fullval point MSE; ties accepted like official',patience=10))
 valmetrics=evaluate(m,va,device,out/'selected_val',True);assert abs(valmetrics['mse']-best)<1e-10
 if small:
  atomic(out/'completion.json',dict(status='preflight_verified',validation=valmetrics,test_evaluations=0));atomic(out/'status.json',dict(status='preflight_verified'));return
 plan=read(PLAN);assert sha(R/plan['full_csv'])==plan['full_csv_sha256']
 with locked():
  with (out/'test_started.json').open('x') as f:json.dump(dict(created=now(),checkpoint_sha256=selected_sha),f)
 test=original.dataset(original.config(R/'dataset/ETT-small'),'test')
 atomic(G/'status.json',dict(status='unique_test_running',process=identity()))
 measured=evaluate(m,test,device,out/'test',True);assert sha(out/'best.pt')==selected_sha
 atomic(out/'test_receipt.json',dict(evaluations=1,windows=len(test),checkpoint_sha256=selected_sha,metrics=measured))
 atomic(out/'completion.json',dict(status='completed_verified',validation=valmetrics,test=measured,epochs=epoch,selected_epoch=chosen,updates=updates,test_evaluations=1,checkpoint_sha256=selected_sha,duration_seconds=time.monotonic()-start))
 atomic(G/'status.json',dict(status='formal_completed_pending_saved_review',process=None));print(json.dumps(dict(status='completed',test=measured)),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['memory','preflight','full'],required=True);ap.add_argument('--device',default='cuda',choices=['cuda','cpu']);ap.add_argument('--output');ap.add_argument('--epochs',default=100,type=int);ap.add_argument('--stop-after',default=0,type=int);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 try:
  if args.phase=='memory':memory(args.device)
  else:train(args)
 except BaseException as e:
  atomic(G/('failure.json' if args.phase=='full' else 'development_failure.json'),dict(status='failed_stopped',phase=args.phase,error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
