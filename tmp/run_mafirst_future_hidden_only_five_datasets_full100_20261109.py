"""Fresh100 MAFirst future-hidden-conditioned fixed-cosine training; test remains sealed."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import sys,argparse,time,math,random,traceback,shutil
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch,numpy as np
from tmp.mafirst_future_hidden_only_five_datasets_full100_20261109_common import *
from tmp.mafirst_future_hidden_only_five_datasets_full100_20261109_factory import *
torch.set_num_threads(2)
def configure(device):
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
 if device=='cuda':assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.9)
def rng():
 n=np.random.get_state()
 return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],python=random.getstate(),numpy=dict(algorithm=n[0],keys=n[1].tolist(),position=n[2],has_gauss=n[3],cached=n[4]))
def restore(r):
 torch.set_rng_state(r['cpu'].cpu());random.setstate(r['python']);n=r['numpy'];np.random.set_state((n['algorithm'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached']))
 if torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])
def finite(m,opt=None):
 assert all(torch.isfinite(p).all() for p in m.parameters())
 if opt:
  assert all(torch.isfinite(v).all() for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor))
def backward_step(arm,m,opt,sched,xx,yy,device,epoch,step,probe=None):
 assert arm=='mafirst_season' and len(xx)==128
 opt.zero_grad(set_to_none=True)
 x=xx.float().to(device);y=yy[:,-HORIZON:].float().to(device)
 assert x.shape[0]==y.shape[0]==128
 p=training_prediction(m,x);assert p.shape==(128,HORIZON,CHANNELS)
 total,mse,reg=loss(arm,p,y,x);assert torch.isfinite(total) and torch.isfinite(p).all()
 if probe is not None:
  a,b=[torch.autograd.grad(z,p,retain_graph=True)[0].detach().double().flatten() for z in [mse,reg]]
  na,nb=float(a.norm()),float(b.norm())
  append(probe,dict(epoch=epoch,step=step,mse=float(mse.detach()),regularizer=float(reg.detach()),point_output_grad_norm=na,regularizer_output_grad_norm=nb,output_gradient_cosine=float(a@b)/max(na*nb,1e-30),probe_batch_size=128,effective_batch_size=128,actual_full_batch=128,accumulations=1,raw_output_gradient_not_Adam_contribution=True))
 total.backward()
 inactive=[n for n,p in m.named_parameters() if p.grad is None]
 assert inactive==read(G/'expected_model_cards.json')[arm]['inactive_parameters'],inactive
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
 usedlr=float(opt.param_groups[0]['lr']);beta=float(opt.param_groups[0]['betas'][0]);opt.step();sched.step()
 return np.array([float(total.detach()),float(mse.detach()),float(reg.detach())]),usedlr,beta

def selection_validation(arm,m,ds,device,out,epoch):
 from tmp.mafirst_distdf_selection_aligned_20261028 import selection_validation as select
 return select(arm,m,ds,device,out,epoch,HORIZON,atomic)

def evaluate(arm,m,ds,device,folder=None):
 m.eval();sq=ab=offset=elements=0;arrays={}
 if folder:
  with locked():folder.mkdir(exist_ok=False)
  shapes={'prediction':(HORIZON,CHANNELS),'truth':(HORIZON,CHANNELS),'history':(96,CHANNELS)}
  if arm=='mafirst_season':shapes.update(components=(HORIZON,CHANNELS,8),history_mean=(1,CHANNELS),inverse_offset=(1,CHANNELS),inverse_scale=(1,CHANNELS))
  for n,tail in shapes.items():arrays[n]=np.lib.format.open_memmap(folder/(n+'.npy'),mode='w+',dtype='float32',shape=(len(ds),*tail))
 with torch.no_grad():
  for x,yy,_,_ in loader(ds):
   x=x.float().to(device);y=yy[:,-HORIZON:].float().to(device)
   if folder and arm=='mafirst_season':
    c=m.predict_components(x);p=m.normalize(c[...,:4].sum(-1)+c[...,4:].sum(-1),'denorm')
    values=dict(prediction=p,truth=y,history=x,components=c*m.normalize.stdev.unsqueeze(-1),history_mean=m.normalize.mean,inverse_offset=m.normalize.mean,inverse_scale=m.normalize.stdev)
   else:p=m(x);values=dict(prediction=p,truth=y,history=x)
   err=p.double()-y.double();assert torch.isfinite(err).all()
   sq+=float(err.square().sum());ab+=float(err.abs().sum());elements+=err.numel()
   for n,a in arrays.items():a[offset:offset+len(x)]=values[n].cpu().numpy()
   offset+=len(x)
 for a in arrays.values():a.flush()
 assert offset==len(ds)
 return dict(mse=sq/elements,mae=ab/elements,windows=offset,elements=elements)
def memory(arm,device):
 configure(device);root=G/'memory'/arm
 with locked():root.mkdir(parents=True,exist_ok=False)
 m,cfg=fresh(arm);m.to(device).train();opt=optimizer(m);sc=scheduler(opt,UPDATES_PER_EPOCH);tr=dataset(arm,'train')
 # CUDA forward/gradient/RNG equality of full-batch exact Gram recomputation against original.
 from tmp.mafirst_wideband_kernel_20261026 import parts as original_parts
 from utils.time_position_patch_mmd_memory import parts as memory_parts
 with torch.random.fork_rng(devices=[0]):
  a=torch.randn(2,cfg.pred_len,7,device=device,requires_grad=True);b=torch.randn_like(a)
  rs=torch.cuda.get_rng_state().clone();old=original_parts(a,b)[0];ga=torch.autograd.grad(old,a)[0]
  new=tiled_parts(a,b).to(a.dtype) if LARGE else memory_parts(a,b)[0];gb=torch.autograd.grad(new,a)[0]
  assert torch.equal(old,new) and torch.equal(ga,gb) and torch.equal(rs,torch.cuda.get_rng_state())
  atomic(root/'GPU_loss_equivalence.json',dict(status='verified',actual_CUDA_loss_and_gradient_bitwise_exact=True,RNG_unchanged=True,full_128_training_batch_not_split=True,test_evaluations=0,created=now()))
  del a,b,old,new,ga,gb
  if LARGE:
   # Compare across a channel-tile boundary with the actual old loss.
   a=torch.randn(2,96,65,device=device,requires_grad=True);b=torch.randn_like(a)
   left=memory_parts(a,b)[0];ga=torch.autograd.grad(left,a)[0]
   right=tiled_parts(a,b).to(a.dtype);gb=torch.autograd.grad(right,a)[0]
   torch.testing.assert_close(left,right,rtol=0,atol=0)
   torch.testing.assert_close(ga,gb,rtol=0,atol=0)
   del a,b,left,right,ga,gb
   # Same stochastic forward, all parameter gradients, and CUDA RNG.
   x=torch.randn(2,96,CHANNELS,device=device)
   rs=torch.cuda.get_rng_state()
   expected=m(x);expected.square().mean().backward()
   after=torch.cuda.get_rng_state();grads=[q.grad.detach().clone() for q in m.parameters()]
   m.zero_grad(set_to_none=True);torch.cuda.set_rng_state(rs)
   actual=training_prediction(m,x);actual.square().mean().backward()
   assert torch.equal(expected,actual) and torch.equal(after,torch.cuda.get_rng_state())
   assert all(torch.equal(q.grad,g) for q,g in zip(m.parameters(),grads))
   m.zero_grad(set_to_none=True)
   atomic(root/'GPU_large_memory_equivalence.json',dict(status='verified',output_gradients_and_rng_bitwise_exact=True,mmd_channel_tile_boundary_bitwise_exact=True))
   del x,expected,actual,grads
 torch.cuda.empty_cache()
 start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 for step,(_,x,y,_,_) in enumerate(loader(tr,1),1):
  terms,_,_=backward_step(arm,m,opt,sc,x,y,device,1,step);finite(m,opt)
  if step==8:break
 m.eval()
 with torch.inference_mode():
  eval_x=next(iter(loader(dataset(arm,'val'),batch=32)))[0].float().to(device)
  assert len(eval_x)==32 and torch.isfinite(m(eval_x)).all()
 torch.cuda.synchronize();atomic(root/'verification.json',dict(status='verified',steps=8,effective_batch=128,actual_loss_batch=128,accumulations=1,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,weights_discarded=True,test_evaluations=0,seconds=time.monotonic()-start,process=identity()))
 print(json.dumps(read(root/'verification.json')),flush=True)
def train(args):
 arm=args.arm;small=args.phase=='preflight';out=Path(args.output).resolve() if small else OUT[arm]
 assert small or not args.resume and not args.stop_after
 if not small:
  assert read(G/'preflight'/arm/'verification.json')['status']=='verified'
  guard()
  with locked():
   with (G/f'{arm}_formal_once.json').open('x') as f:json.dump(identity(),f)
 configure(args.device)
 if not args.resume:
  with locked():out.mkdir(parents=True,exist_ok=False)
 atomic(out/'process.json',identity())
 m,cfg=fresh(arm);m.to(args.device);opt=optimizer(m)
 tr=dataset(arm,'train',512 if small else 0);va=dataset(arm,'val',128 if small else 0)
 steps=len(tr)//128;sc=scheduler(opt,steps)
 if args.resume:
  assert small and read(out/'status.json')['status']=='controlled_stop'
  last=torch.load(out/'last.pt',map_location=args.device,weights_only=True)
  m.load_state_dict(last['model']);opt.load_state_dict(last['optimizer']);sc.load_state_dict(last['scheduler'])
  curves=last['curves'];best=last['best'];chosen=last['chosen'];stale=last['stale'];updates=last['updates'];start_epoch=last['epoch']+1;restore(last['rng'])
 else:
  curves=[];best=math.inf;chosen=stale=updates=0;start_epoch=1
  atomic(out/'model_config.json',vars(cfg))
  atomic(out/'initialization.json',dict(status='verified',reference_sha256=sha(G/'reference_initial'/f'{arm}.pt'),all_fresh_states_exact=True,trained_weights_loaded=False,**read(G/'expected_model_cards.json')[arm]))
  atomic(out/'optimization_receipt.json',dict(optimizer='Adam',max_lr=.0001,initial_lr=opt.param_groups[0]['lr'],weight_decay=0.,schedule='OneCycle100',pct_start=.4,effective_batch_size=128,actual_loss_batch_size=128,accumulations=1,train_drop_last=True,training_workers=10,training_windows=UPDATES_PER_EPOCH*128,discarded_windows=TRAIN_WINDOWS%128,full_validation_selection=False,selection_validation_shuffle=True,selection_validation_drop_last=True,selection_validation_batch_size=128,selection_validation_workers=10,patience=10,ties_accepted=True,loss_weights=[1.,1.] if arm=='mafirst_season' else [.8,.2],only_best_and_last_checkpoints=True,seed_policy="2021 once before fresh model; continuous RNG",sampler_generator="None; global continuous",full_batch_forward_calls_per_update=1))
 if not small:
  receipt=read(out/'optimization_receipt.json');receipt.update(model_activation_checkpointing=LARGE,mmd_channel_tiling=LARGE,patience=None,early_stopping=False,force_total_epochs=100,fresh_training=True,future_hidden_conditioned_dynamic_frequency=True,frequency_head_condition="future_block_mean_and_last",learnable_periods=False,period_parameterization=None,initial_periods=None);atomic(out/'optimization_receipt.json',receipt)
 maxepochs=3 if small else 100;start=time.monotonic()
 for epoch in range(start_epoch,maxepochs+1):
  m.train();totals=np.zeros(3);seen=0;initial_lr=opt.param_groups[0]['lr'];dl=loader(tr,epoch)
  order=[]
  for step,(ids,xx,yy,_,_) in enumerate(dl,1):
   order.extend(ids.tolist())
   terms,lr,beta=backward_step(arm,m,opt,sc,xx,yy,args.device,epoch,step,out/'gradients.jsonl' if step in [1,50,100,150,200,250] else None)
   if step in [1,50,100,150,200,250]:append(out/'gate_trace.jsonl',dict(epoch=epoch,step=step,policy=m.gate_context_policy,kernels=[25,13,7,3],mean4_drive=True,endpoint_readout=True,scales=[{k:(float(v.detach()) if isinstance(v,torch.Tensor) else v) for k,v in z.items()} for z in m.last_gate_context_diagnostics]))
   totals+=terms*128;seen+=128;updates+=1
   if step==1 or step%25==0:
    finite(m);atomic(out/'status.json',dict(status='preflight_running' if small else 'formal_running',process=identity(),arm=arm,epoch=epoch,step=step,steps=steps,completed_epochs=epoch-1,best_epoch=chosen,lr=opt.param_groups[0]['lr'],elapsed_seconds=time.monotonic()-start))
  order_sha=hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest()
  finite(m,opt);measured=selection_validation(arm,m,va,args.device,out,epoch)
  improved=measured['mse']<=best
  if improved:best=measured['mse'];chosen=epoch;stale=0;save(out/'best.pt',m.state_dict())
  else:stale+=1
  row=dict(epoch=epoch,train_loss=float(totals[0]/seen),train_mse=float(totals[1]/seen),train_regularizer=float(totals[2]/seen),val=measured,lr_start=float(initial_lr),lr_end=float(opt.param_groups[0]['lr']),scheduler_last_epoch=sc.last_epoch,updates=updates,training_windows=seen,order_sha256=order_sha,stale=stale,chosen=chosen,improved=improved,model_and_Adam_all_finite=True)
  w=[1.,1.] if arm=='mafirst_season' else [.8,.2]
  assert abs(row['train_loss']-w[0]*row['train_mse']-w[1]*row['train_regularizer'])<2e-6
  curves.append(row);save(out/'last.pt',dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sc.state_dict(),curves=curves,best=best,chosen=chosen,stale=stale,updates=updates,epoch=epoch,rng=rng()))
  atomic(out/'curves.json',curves);append(out/'epoch_receipts.jsonl',dict(epoch=epoch,selected_epoch=chosen,best_checkpoint_sha256=sha(out/'best.pt'),last_checkpoint_sha256=sha(out/'last.pt'),finite=True))
  print(json.dumps(row),flush=True)
  if small and args.stop_after==epoch:atomic(out/'status.json',dict(status='controlled_stop',epoch=epoch));return
  if small and stale>=10:break
 assert updates==len(curves)*steps and sc.last_epoch==updates
 if not small:
  prefix=[]
  assert curves[:len(prefix)]==prefix and epoch==len(curves)==100 and updates==sc.last_epoch==100*UPDATES_PER_EPOCH
  assert abs(curves[39]['lr_end']-1e-4)<1e-10 and curves[-1]['lr_end']<1e-8
  assert curves[-1]['lr_end']<curves[39]['lr_end']
  atomic(out/'full_decay_completion_audit.json',dict(status='verified',source_epochs=len(prefix),total_epochs=100,added_epochs=100-len(prefix),total_updates=100*UPDATES_PER_EPOCH,source_prefix_unchanged=True,early_stopping=False,peak_epoch=40,peak_lr=curves[39]['lr_end'],last_lr=curves[-1]['lr_end'],scheduler_last_epoch=sc.last_epoch,validation_only_selection=True,test_evaluations=0,created=now()))
 selected=next(x for x in curves if x['epoch']==chosen)
 assert selected['val']['mse']==best and chosen>0 and (small or epoch==100)
 checkpoint=sha(out/'best.pt');m.load_state_dict(torch.load(out/'best.pt',map_location=args.device,weights_only=True))
 atomic(out/'freeze.json',dict(selected_epoch=chosen,epochs=epoch,updates=updates,checkpoint_sha256=checkpoint,selection_validation_mse=best,selection="DistDF shuffled/drop128 FP32 batch-MSE mean",epoch0_selectable=False,test_opened=False))
 val=evaluate(arm,m,va,args.device,out/'selected_val' if small else None) # Audit only; never reselect after best freeze.
 result=dict(status='preflight_verified' if small else 'completed_verified',arm=arm,dataset=DATASET,validation=val,selection_validation_mse=best,selection_validation_windows=len(va)//128*128,epochs=epoch,selected_epoch=chosen,updates=updates,test_evaluations=0,checkpoint_sha256=checkpoint,duration_seconds=time.monotonic()-start)
 if not small:result['status']='training_completed_best_frozen'
 atomic(out/'completion.json',result);atomic(out/'status.json',dict(status=result['status'],process=identity()))
 print(json.dumps(result),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--phase',choices=['memory','preflight','full'],required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--output');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int,default=0);args=p.parse_args()
 try:
  if args.phase=='memory':memory(args.arm,args.device)
  else:train(args)
 except BaseException as exc:
  atomic(G/('failure.json' if args.phase=='full' else 'development_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),phase=args.phase,arm=args.arm,process=identity(),automatic_retry=False,created=now()));raise
if __name__=='__main__':main()
