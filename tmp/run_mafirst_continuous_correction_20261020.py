"""Fresh single-arm delayed-state residual runner; base rho is not full delayed-system transport. Resume is exclusively for the new small preflight."""
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
from models.OscillatorPDMStatePatchMAFirstContinuousCorrection import Model
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
optimizer_groups=lambda m:norm_bias_no_decay_groups(m,1e-4)
from utils.time_position_patch_mmd import parts as mmd_parts
from utils.patch_covariance_loss import parts as old_parts
from models import OscillatorPDMStatePatchMAFirstContinuousCorrection as patch
torch.set_num_threads(2)
PLAN=R/'tmp/mafirst_continuous_correction_plan_20261020.json'
CH=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(p,v):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def save_torch(p,v):
    t=p.with_suffix(p.suffix+'.tmp');torch.save(v,t);t.replace(p)
def seed(v):
    random.seed(v);np.random.seed(v);torch.manual_seed(v)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(v)
def identity():
    return dict(pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],command=[s.decode() for s in Path('/proc/self/cmdline').read_bytes().split(b'\0') if s])
def phase(out,status,**kw):atomic(out/'status.json',dict(status=status,updated=now(),**kw))

def fresh(H,arm,logdir):
    init=importlib.import_module('tmp.run_mafirst_level_head_e2e_h192_seed2021_20261017')
    cfg=init.config(R/'dataset/ETT-small');cfg.pred_len=H;cfg.log_dir=str(logdir);cfg.tensorboard=False
    assert arm in ('centered_recurrence','gated_anchor')
    cfg.correction_arm=arm
    reference=init.control_reference(cfg)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20211014);m=Model(cfg)
    state=reference.state_dict()
    missing,unexpected=m.load_state_dict(state,strict=False)
    assert not unexpected
    expected=[k for k in m.state_dict() if '.correction_anchor.' in k or '.correction_gate.' in k]
    assert set(missing)==set(expected)
    original=torch.load(R/f'tmp/mafirst_mmd_ettm1_20261016_h{H}_seed2021_max30_patience5/initial.pt',map_location='cpu',weights_only=True)['time_position_mmd']
    assert len(state)==236 and all(torch.equal(v,original[n]) and torch.equal(v,m.state_dict()[n]) and v.data_ptr()!=m.state_dict()[n].data_ptr() for n,v in state.items())
    assert sum(p.numel() for p in m.parameters())==(320450 if arm=='centered_recurrence' else 323618)
    return m,cfg

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
    raw=pd.read_csv(R/plan['train_val_prefix'])[CH].to_numpy()
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
    plan=read(PLAN);base=R/plan['source_reference'];manifest=read(base/'sources_manifest.json')
    for rel,digest in manifest.items():
        assert sha(R/rel)==digest and sha(base/'source'/rel)==digest,rel
    for rel in plan['extra_sources']:manifest[rel]=sha(R/rel)
    for rel,digest in manifest.items():
        p=out/'source'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/rel,p);assert sha(p)==digest
    atomic(out/'sources_manifest.json',manifest)

def evaluate(m,ds,device,folder=None):
    m.eval();square=absolute=count=0;offset=0;arrays={}
    if folder:
        folder.mkdir(exist_ok=False)
        for key,shape in [('prediction',(m.pred_len,7)),('truth',(m.pred_len,7)),('history',(96,7)),('components',(m.pred_len,7,8)),('history_mean',(1,7))]:
            arrays[key]=np.lib.format.open_memmap(folder/(key+'.npy'),mode='w+',dtype='float32',shape=(len(ds),*shape))
    with torch.no_grad():
        for x,y in batches(ds):
            x,y=x.to(device),y.to(device)
            if folder:
                components=m.predict_components(x);p=m.normalize(components[...,:4].sum(-1)+components[...,4:].sum(-1),'denorm')
            else:p=m(x)
            e=(p-y).double();assert torch.isfinite(e).all()
            square+=float(e.square().sum());absolute+=float(e.abs().sum());count+=e.numel()
            if folder:
                values=dict(prediction=p,truth=y,history=x,components=components*m.normalize.stdev.unsqueeze(-1),history_mean=m.normalize.mean)
                for key,v in values.items():arrays[key][offset:offset+len(x)]=v.cpu().numpy()
            offset+=len(x)
    for a in arrays.values():a.flush()
    return dict(mse=square/count,mae=absolute/count,windows=len(ds),elements=count)

def gradient_capture(m,x,y,folder,label):
    m.eval();captured=[];original=patch.correction_scan
    def scan(a,u,h,*args,**kwargs):
        s=original(a,u,h,*args,**kwargs);captured.append((a,u,h,s));return s
    patch.correction_scan=scan
    try:p=m(x)
    finally:patch.correction_scan=original
    H=m.pred_len;loss=(p[:,-H//4:]-y[:,-H//4:]).square().mean()
    targets=[t for c in captured for t in (c[1],c[2],c[3])]
    g=torch.autograd.grad(loss,targets);result={};arrays={}
    for s,(a,u,h,states) in enumerate(captured):
        gu,gh,local=g[3*s:3*s+3];K=u.shape[1];start=3*K//4
        curve=gu.detach().double().square().mean((0,2,3)).sqrt().cpu().numpy()
        rho=a.detach().double()[:,0].norm(dim=-1).cpu().numpy()
        result[str(s)]=dict(early_to_far_start_ratio=float(curve[0]/max(curve[start],1e-30)),initial_gradient_rms=float(gh.double().square().mean().sqrt()),base_one_step_radius_quantiles=np.quantile(rho,[.1,.5,.9]).tolist(),effective_one_step_radius_quantiles=np.quantile(rho[:,None]*(1-m.last_gates[s].cpu().numpy()[...,0]) if m.last_gates else rho,[.1,.5,.9]).tolist(),readout_local_gradient_rms=float(local.double().square().mean().sqrt()),maximum_state_rms=float(states.detach().double().square().mean((-2,-1)).sqrt().max()))
        arrays[f's{s}_adjoint_rms']=curve;arrays[f's{s}_rho']=rho
    np.savez_compressed(folder/(label+'.npz'),**arrays)
    return result

def nonfinite(out,phase_name,epoch,step,**details):
    atomic(out/'numerical_failure.json',dict(status='failed_nonfinite',phase=phase_name,epoch=epoch,step=step,**details))
    raise FloatingPointError(f'{phase_name}: nonfinite; stop without retry; see numerical_failure.json')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--output',required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--train-limit',type=int,default=0);ap.add_argument('--val-limit',type=int,default=0);ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--stop-after',type=int,default=0);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    plan=read(PLAN);job=next(j for j in plan['jobs'] if j['id']==args.job);out=R/args.output
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
        save_torch(out/'initial.pt',m.state_dict());atomic(out/'optimizer_groups.json',groups_audit(m,groups));atomic(out/'model_config.json',vars(cfg))
        atomic(out/'data_manifest.json',dict(prefix_sha256=sha(R/plan['train_val_prefix']),train_end=job['train_end'],val_end=job['val_end'],train_windows=len(train),val_windows=len(val),scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist()))
        best=float('inf');chosen=stale=updates=0;curves=[];start=1
    else:
        last=torch.load(out/'last.pt',map_location=args.device,weights_only=True);m.load_state_dict(last['model']);opt.load_state_dict(last['optimizer']);best=last['best'];chosen=last['chosen'];stale=last['stale'];updates=last['updates'];curves=last['curves'];start=last['epoch']+1
    for epoch in range(start,args.epochs+1):
        m.train();lr=.001*.5*(1+math.cos(math.pi*(epoch-1)/30))
        for g in opt.param_groups:g['lr']=lr
        total=np.zeros(4);seen=0;loader=batches(train,epoch)
        order_sha=hashlib.sha256(np.asarray(loader.dataset.indices,dtype=np.int64).tobytes()).hexdigest()
        for step,(x,y) in enumerate(loader,1):
            seed(2021000000+epoch*10000+step-1);x,y=x.to(args.device),y.to(args.device)
            opt.zero_grad(set_to_none=True);p=m(x)
            mask=~torch.isfinite(p)
            if mask.any():
                steps_bad=mask.any(dim=(0,2)).nonzero().flatten()
                nonfinite(out,'forward',epoch,step,nonfinite_elements=int(mask.sum()),elements=p.numel(),first_nonfinite_forecast_step=int(steps_bad[0])+1)
            mse=(p-y).square().mean();mm,meta=mmd_parts(p,y);old,oldmeta=old_parts(p,y);loss=mse+mm
            if not (torch.isfinite(loss) and meta['minimum_unclipped_mmd']>=-1e-10 and oldmeta['minimum_unclipped_distance']>=-1e-10):
                nonfinite(out,'loss_or_kernel_diagnostic',epoch,step,loss_repr=repr(float(loss)),mse_repr=repr(float(mse)),mmd_repr=repr(float(mm)),old_distance_repr=repr(float(old)))
            if step in (1,200,400,600,800,1000):
                named=list(m.named_parameters());ga=torch.autograd.grad(mse,[p for _,p in named],retain_graph=True);gb=torch.autograd.grad(mm,[p for _,p in named],retain_graph=True)
                scopes={}
                for name,pred in [('all',lambda n:True),('coarse_readout',lambda n:n.startswith('seasonals.3.decoder.readout.')),('history_encoder',lambda n:'.blocks.' in n or n.startswith('pdm_blocks.')),('dynamics',lambda n:'.decoder.' in n and ('log_decay' in n or '.condition.' in n or '.trend_condition.' in n))]:
                    aa=bb=dot=0.
                    for (n,_),a,b in zip(named,ga,gb):
                        if pred(n):aa+=float(a.double().square().sum());bb+=float(b.double().square().sum());dot+=float((a.double()*b.double()).sum())
                    scopes[name]=dict(mse_norm=aa**.5,mmd_norm=bb**.5,cosine=dot/max((aa*bb)**.5,1e-30))
                with (out/'gradients.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,step=step,scopes=scopes))+'\n')
            loss.backward()
            bad_grad=[n for n,p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
            if bad_grad:nonfinite(out,'gradient',epoch,step,parameter_names=bad_grad)
            opt.step()
            bad_state=[n+':'+key for n,p in m.named_parameters() for key,v in opt.state[p].items() if isinstance(v,torch.Tensor) and not torch.isfinite(v).all()]
            if bad_state:nonfinite(out,'Adam_state',epoch,step,state_names=bad_state)
            bad_weight=[n for n,p in m.named_parameters() if not torch.isfinite(p).all()]
            if bad_weight:nonfinite(out,'parameter',epoch,step,parameter_names=bad_weight)
            total+=np.array([float(loss.detach()),float(mse.detach()),float(mm.detach()),float(old.detach())])*len(x);seen+=len(x);updates+=1
            if step==1 or step%50==0:phase(out,'training',epoch=epoch,step=step,steps=len(loader),completed_epochs=epoch-1)
        phase(out,'validation',epoch=epoch,completed_epochs=epoch-1)
        measured=evaluate(m,val,args.device);improved=measured['mse']<best
        if improved:best=measured['mse'];chosen=epoch;save_torch(out/'best.pt',m.state_dict())
        stale=0 if improved else stale+1
        row=dict(epoch=epoch,lr=lr,train_loss=float(total[0]/seen),train_mse=float(total[1]/seen),train_mmd=float(total[2]/seen),old_distance_diagnostic=float(total[3]/seen),val=measured,order_sha256=order_sha,training_windows=seen,all_parameter_gradients_finite=True,parameter_tensors=len(list(m.parameters())))
        curves.append(row);(out/'epochs').mkdir(exist_ok=True);save_torch(out/'epochs'/f'epoch{epoch:02d}.pt',m.state_dict())
        last=dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,chosen=chosen,stale=stale,updates=updates,curves=curves,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else [])
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
