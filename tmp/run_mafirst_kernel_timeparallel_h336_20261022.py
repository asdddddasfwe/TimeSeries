"""One fresh time-consistent run; only the small preflight may resume."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import argparse, datetime, fcntl, hashlib, importlib, json, math, random, shutil, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset,DataLoader,Subset
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from tmp.mafirst_kernel_timeparallel_h336_20261022_factory import fresh as fresh_model, optimizer_groups
from tmp.mafirst_kernel_timeparallel_h336_20261022_common import G,guard,locked,atomic_unlocked
from utils.time_position_patch_mmd import parts as mmd_parts
from utils.patch_covariance_loss import parts as old_parts
torch.set_num_threads(2)
PLAN=R/'tmp/mafirst_kernel_timeparallel_plan_20261022.json'
CH=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(p,v):
    with locked():
        atomic_unlocked(p,v)

def save_torch(p,v):
    t=p.with_suffix(p.suffix+'.tmp');torch.save(v,t);t.replace(p)
def seed(v):
    random.seed(v);np.random.seed(v);torch.manual_seed(v)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(v)
def identity():
    return dict(pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],command=[s.decode() for s in Path('/proc/self/cmdline').read_bytes().split(b'\0') if s])
def phase(out,status,**kw):atomic(out/'status.json',dict(status=status,updated=now(),**kw))

def fresh(H,arm,logdir):
    assert H==336
    return fresh_model(arm,logdir)

class Windows(Dataset):
    def __init__(self,data,H,train_end,end,split):
        self.data,self.H=data,H
        self.start=0 if split=='train' else train_end-96
        self.length=(train_end if split=='train' else end-self.start)-96-H+1
    def __len__(self):return self.length
    def __getitem__(self,i):
        s=self.start+i
        return self.data[s:s+96],self.data[s+96:s+96+self.H]
def data_for(job,limit_train=0,limit_val=0):
    plan=read(PLAN);assert sha(R/plan['train_val_prefix'])==plan['train_val_prefix_sha256']
    raw=pd.read_csv(R/plan['train_val_prefix'],nrows=job['val_end'])[CH].to_numpy()
    scaler=StandardScaler().fit(raw[:job['train_end']]);data=scaler.transform(raw[:job['val_end']]).astype('float32')
    ds=[Windows(data,job['horizon'],job['train_end'],job['val_end'],s) for s in ('train','val')]
    for i,limit in enumerate((limit_train,limit_val)):
        if limit:ds[i]=Subset(ds[i],np.unique(np.linspace(0,len(ds[i])-1,min(limit,len(ds[i])),dtype=int)).tolist())
    return (*ds,scaler,data)
def batches(ds,epoch=None):
    if epoch is not None:
        order=torch.randperm(len(ds),generator=torch.Generator().manual_seed(2021000+epoch)).tolist();ds=Subset(ds,order)
    return DataLoader(ds,batch_size=32,shuffle=False,num_workers=0,drop_last=False)
def groups_audit(m,groups):
    names={id(p):n for n,p in m.named_parameters()}
    return [dict(group_name=g.get('group_name'),weight_decay=g['weight_decay'],names=[names[id(p)] for p in g['params']]) for g in groups]
def snapshot(out):
    manifest=read(G/"frozen_sources.json")
    for rel,digest in manifest.items():
        assert sha(R/rel)==digest,rel
        dest=out/"source"/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(R/rel,dest);assert sha(dest)==digest
    atomic(out/"sources_manifest.json",manifest)


def evaluate(m,ds,device,folder=None):
    m.eval();square=absolute=count=0;offset=0;arrays={};kernel_sum=kernel_constant=0.;kernel_bins=np.zeros(4)
    if folder:
        folder.mkdir(exist_ok=False)
        for key,shape in [('prediction',(m.pred_len,7)),('truth',(m.pred_len,7)),('history',(96,7)),('components',(m.pred_len,7,8)),('history_mean',(1,7)),('inverse_offset',(1,7)),('inverse_scale',(1,7))]:
            arrays[key]=np.lib.format.open_memmap(folder/(key+'.npy'),mode='w+',dtype='float32',shape=(len(ds),*shape))
    with torch.no_grad():
        for x,y in batches(ds):
            x,y=x.to(device),y.to(device)
            if folder:
                components=m.predict_components(x);p=m.normalize(components[...,:4].sum(-1)+components[...,4:].sum(-1),'denorm')
            else:p=m(x)
            if m.study_arm=='kernel':
                from models.MAFirstKernelValueOnly336 import kernel_supervision
                kl,gold=kernel_supervision(m.predicted_gram,y)
                kernel_sum+=float(kl)*len(x)
                K=gold.shape[-1];mask=~torch.eye(K,device=device,dtype=torch.bool)
                kernel_constant+=float((1-gold).square()[...,mask].mean())*len(x)
                distances=(torch.arange(K,device=device)[:,None]-torch.arange(K,device=device)[None,:]).abs()
                for q,(lo,hi) in enumerate([(1,4),(5,16),(17,42),(43,83)]):
                    maskq=(distances>=lo)&(distances<=hi)
                    kernel_bins[q]+=float((m.predicted_gram-gold).square()[...,maskq].mean())*len(x)
                if folder:
                    for selected in (0,len(ds)//2,len(ds)-1):
                        if offset<=selected<offset+len(x):
                            j=selected-offset
                            np.savez_compressed(folder/f'kernel_{selected}.npz',predicted=m.predicted_gram.reshape(len(x),7,K,K)[j].cpu().numpy(),target=gold.reshape(len(x),7,K,K)[j].cpu().numpy())
            e=p.double()-y.double();assert torch.isfinite(e).all()
            square+=float(e.square().sum());absolute+=float(e.abs().sum());count+=e.numel()
            if folder:
                scale,inv_offset=m.inverse_coordinates()
                values=dict(prediction=p,truth=y,history=x,components=components*scale.unsqueeze(-1),history_mean=m.normalize.mean,inverse_offset=inv_offset,inverse_scale=scale)
                for key,v in values.items():arrays[key][offset:offset+len(x)]=v.cpu().numpy()
            offset+=len(x)
    for a in arrays.values():a.flush()
    return dict(mse=square/count,mae=absolute/count,windows=len(ds),elements=count,**(dict(kernel_mse=kernel_sum/len(ds),constant_ones_kernel_mse=kernel_constant/len(ds),kernel_distance_bins_mse=(kernel_bins/len(ds)).tolist()) if m.study_arm=='kernel' else {}))

def gradient_capture(m,x,y,folder,label):
    """d far-loss / d additive future drive is the recurrent total state adjoint."""
    from models import MAFirstKernelValueOnly336 as module
    captured=[]
    attr='parallel_future_states' if m.study_parallel else 'recurrent_future_states'
    original=getattr(module,attr)
    def capture(a,u,h):
        s=original(a,u,h);captured.append((a,u,h,s));return s
    setattr(module,attr,capture);m.eval()
    try:p=m(x)
    finally:setattr(module,attr,original)
    loss=(p[:,-m.pred_len//4:]-y[:,-m.pred_len//4:]).square().mean()
    gs=torch.autograd.grad(loss,[v for r in captured for v in (r[1],r[2],r[3])])
    assert len(captured)==4
    arrays={};result={}
    for i,(a,u,h,s) in enumerate(captured):
        gu,gh,local=gs[3*i:3*i+3];K=u.shape[1]
        total=gu.detach().double().square().mean((0,2,3)).sqrt().cpu().numpy()
        loc=local.detach().double().square().mean((0,2,3)).sqrt().cpu().numpy()
        assert np.isfinite(total).all() and np.isfinite(loc).all()
        if m.study_parallel:assert np.array_equal(total[:3*K//4],np.zeros(3*K//4))
        arrays[f's{i}_adjoint_rms']=total;arrays[f's{i}_local_gradient_rms']=loc
        arrays[f's{i}_initial_gradient']=gh.detach().cpu().numpy()
        result[str(i)]=dict(initial_gradient_rms=float(gh.double().square().mean().sqrt()),
            early_drive_zero_expected=m.study_parallel,
            definition='far-loss gradient wrt additive future drive; local output gradient saved separately')
    np.savez_compressed(folder/(label+'.npz'),**arrays)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--output',required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--train-limit',type=int,default=0);ap.add_argument('--val-limit',type=int,default=0);ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--stop-after',type=int,default=0);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    plan=read(PLAN);job=next(j for j in plan['jobs'] if j['id']==args.job);out=R/args.output
    guard(False)
    assert job["arm"] in ("recurrent","parallel")
    expected=G/"preflights"/job["id"]
    if args.train_limit or args.val_limit:
        assert args.train_limit==64 and args.val_limit==32 and args.epochs==3
        assert out in (expected/"preflight_continuous",expected/"preflight_resumed")
    else:
        assert out==R/job["output"]
    smoke=bool(args.train_limit or args.val_limit);assert smoke or (args.epochs==30 and not args.resume and not args.stop_after)
    if args.device=='cuda':
        assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.6 if job['horizon']==720 else .4)
    if not args.resume:
        out.mkdir(parents=True,exist_ok=False);snapshot(out);atomic(out/'process.json',identity());atomic(out/'protocol.json',dict(job=job,args=vars(args),test_evaluations=0))
    else:assert smoke and (out/'last.pt').exists()
    m,cfg=fresh(job['horizon'],job['arm'],out/'logs');m.to(args.device)
    groups=optimizer_groups(m);opt=torch.optim.Adam(groups,lr=.001)
    train,val,scaler,data=data_for(job,args.train_limit,args.val_limit)
    if not args.resume:
        save_torch(out/'initial.pt',m.state_dict());atomic(out/'initialization.json',m.initialization_audit);atomic(out/'optimizer_groups.json',groups_audit(m,groups));atomic(out/'model_config.json',vars(cfg))
        atomic(out/'data_manifest.json',dict(prefix_sha256=sha(R/plan['train_val_prefix']),train_end=job['train_end'],val_end=job['val_end'],train_windows=len(train),val_windows=len(val),scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist()))
        best=float('inf');chosen=stale=updates=0;curves=[];start=1
    else:
        last=torch.load(out/'last.pt',map_location=args.device,weights_only=True);m.load_state_dict(last['model']);opt.load_state_dict(last['optimizer']);best=last['best'];chosen=last['chosen'];stale=last['stale'];updates=last['updates'];curves=last['curves'];start=last['epoch']+1
    if args.resume:
        torch.set_rng_state(last["cpu_rng"].cpu())
        if args.device=="cuda":torch.cuda.set_rng_state_all([v.cpu() for v in last["cuda_rng"]])
        random.setstate(last["python_rng"])
        ns=last["numpy_rng"]
        np.random.set_state((ns["algorithm"],np.asarray(ns["keys"],dtype=np.uint32),ns["position"],ns["has_gauss"],ns["cached_gaussian"]))
    for epoch in range(start,args.epochs+1):
        m.train();lr=.001*.5*(1+math.cos(math.pi*(epoch-1)/30))
        for g in opt.param_groups:g['lr']=lr
        total=np.zeros(5);seen=0;loader=batches(train,epoch)
        order_sha=hashlib.sha256(np.asarray(loader.dataset.indices,dtype=np.int64).tobytes()).hexdigest()
        for step,(x,y) in enumerate(loader,1):
            seed(2021000000+epoch*10000+step-1);x,y=x.to(args.device),y.to(args.device)
            opt.zero_grad(set_to_none=True);p=m(x);mse=(p-y).square().mean();mm,meta=mmd_parts(p,y);old,oldmeta=old_parts(p,y);aux=p.new_zeros(())
            if m.study_arm=='kernel':
                from models.MAFirstKernelValueOnly336 import kernel_supervision
                aux,_=kernel_supervision(m.predicted_gram,y)
            loss=mse+mm+aux
            assert torch.isfinite(loss) and meta['minimum_unclipped_mmd']>=-1e-10 and oldmeta['minimum_unclipped_distance']>=-1e-10
            if step in (1,200,400,600,800,1000):
                named=list(m.named_parameters());ga=torch.autograd.grad(mse,[p for _,p in named],retain_graph=True);gb=torch.autograd.grad(mm,[p for _,p in named],retain_graph=True);gc=torch.autograd.grad(aux,[p for _,p in named],retain_graph=True,allow_unused=True)
                scopes={}
                for name,pred in [('all',lambda n:True),('coarse_readout',lambda n:n.startswith('seasonals.3.decoder.readout.')),('history_encoder',lambda n:'.blocks.' in n or n.startswith('pdm_blocks.')),('state_core',lambda n:'.decoder.' in n and '.readout.' not in n)]:
                    aa=bb=cc=dot=ac=bc=0.
                    for (n,param),a,b,c in zip(named,ga,gb,gc):
                        if pred(n):
                            aa+=float(a.double().square().sum());bb+=float(b.double().square().sum());dot+=float((a.double()*b.double()).sum())
                            if c is not None:
                                cc+=float(c.double().square().sum());ac+=float((a.double()*c.double()).sum());bc+=float((b.double()*c.double()).sum())
                    scopes[name]=dict(mse_norm=aa**.5,mmd_norm=bb**.5,gram_aux_norm=cc**.5,cosine=dot/max((aa*bb)**.5,1e-30),mse_gram_cosine=ac/max((aa*cc)**.5,1e-30),mmd_gram_cosine=bc/max((bb*cc)**.5,1e-30),mse_mmd_cosine_defined=aa*bb>0,mse_gram_cosine_defined=aa*cc>0,mmd_gram_cosine_defined=bb*cc>0)
                with (out/'gradients.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,step=step,scopes=scopes))+'\n')
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            opt.step()
            if smoke or step%50==0:
                assert all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if isinstance(v,torch.Tensor))
            total+=np.array([float(loss.detach()),float(mse.detach()),float(mm.detach()),float(old.detach()),float(aux.detach())])*len(x);seen+=len(x);updates+=1
            if step==1 or step%50==0:phase(out,'training',epoch=epoch,step=step,steps=len(loader),completed_epochs=epoch-1)
        phase(out,'validation',epoch=epoch,completed_epochs=epoch-1)
        measured=evaluate(m,val,args.device);improved=measured['mse']<best
        if improved:best=measured['mse'];chosen=epoch;save_torch(out/'best.pt',m.state_dict())
        stale=0 if improved else stale+1
        row=dict(epoch=epoch,lr=lr,train_loss=float(total[0]/seen),train_mse=float(total[1]/seen),train_mmd=float(total[2]/seen),old_distance_diagnostic=float(total[3]/seen),val=measured,order_sha256=order_sha,training_windows=seen,all_gradients_finite=True,parameter_tensors=len(list(m.parameters())),train_kernel=float(total[4]/seen))
        row['affine_parameters']={n:p.detach().cpu().tolist() for n,p in m.named_parameters() if n.endswith(('affine_weight','affine_bias'))}
        assert all(torch.isfinite(p).all() for p in m.parameters())
        curves.append(row);(out/'epochs').mkdir(exist_ok=True);save_torch(out/'epochs'/f'epoch{epoch:02d}.pt',m.state_dict())
        last=dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,chosen=chosen,stale=stale,updates=updates,curves=curves,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else [],
            python_rng=random.getstate(),numpy_rng=dict(algorithm=np.random.get_state()[0],
            keys=np.random.get_state()[1].tolist(),position=int(np.random.get_state()[2]),
            has_gauss=int(np.random.get_state()[3]),cached_gaussian=float(np.random.get_state()[4])))
        save_torch(out/'last.pt',last);atomic(out/'curves.json',curves)
        print(json.dumps(dict(job=job['id'],epoch=epoch,val=measured['mse'],best=best,chosen=chosen,stale=stale)),flush=True)
        if args.stop_after and epoch>=args.stop_after:phase(out,'controlled_stop',epoch=epoch);return
        if stale>=5:break
    m.load_state_dict(torch.load(out/'best.pt',map_location=args.device,weights_only=True));m.eval()
    selected_sha=sha(out/'best.pt');atomic(out/'freeze.json',dict(selected_epoch=chosen,checkpoint_sha256=selected_sha,epochs=epoch,updates=updates,stale=stale,maximum=30,patience=5,strict_improvement=True))
    valmetrics=evaluate(m,val,args.device,out/'selected_val')
    assert abs(valmetrics['mse']-best)<1e-10
    snaps={}
    for begin in (0,8000,16000):
        x=torch.from_numpy(np.stack([data[i:i+96] for i in range(begin,begin+8)])).to(args.device)
        y=torch.from_numpy(np.stack([data[i+96:i+96+job['horizon']] for i in range(begin,begin+8)])).to(args.device)
        snaps[str(begin)]=gradient_capture(m,x,y,out,f'gradient_snapshot_{begin}')
    atomic(out/'selected_gradient_snapshots.json',snaps)
    atomic(out/'architecture_receipt.json',dict(status='verified',arm=m.study_arm,affine=m.affine_enabled,
        initialization=m.initialization_audit,study_parallel=m.study_parallel,time_kernel_removed=False,model_positions_preserved=True,future_states=84,patch_size=4,all236_scratch_exact=True,
        kernel_targets_are_not_forecast_inputs=True,affine_parameters_decay=0))
    testmetrics=None
    if job['stage']=='full' and not smoke:
        # Exclusive, once-only evaluation for this NEW full-data arm/run.
        with (out/'test_armed.json').open('x') as f:json.dump(dict(checkpoint_sha256=selected_sha,created=now()),f)
        assert sha(R/plan['full_csv'])==plan['full_csv_sha256']
        raw=pd.read_csv(R/plan['full_csv'])[CH].to_numpy();scaled=scaler.transform(raw).astype('float32')
        test=Windows(scaled,job['horizon'],46080,57600,'val')
        with (out/'test_started.json').open('x') as f:json.dump(dict(checkpoint_sha256=selected_sha,created=now()),f)
        testmetrics=evaluate(m,test,args.device,out/'test')
        atomic(out/'test_receipt.json',dict(evaluations=1,windows=len(test),checkpoint_sha256=selected_sha,metrics=testmetrics))
    assert sha(out/'best.pt')==selected_sha
    atomic(out/'completion.json',dict(status='completed_verified',job=job,validation=valmetrics,test=testmetrics,test_evaluations=int(testmetrics is not None),epochs=epoch,selected_epoch=chosen,updates=updates,checkpoint_sha256=selected_sha,smoke=smoke))
    phase(out,'completed_verified',epoch=epoch,selected_epoch=chosen)
if __name__=='__main__':main()
