"""MAFirst+MMD single-factor peak learning-rate .001 vs frozen .0001 control."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from data_provider.data_loader import Dataset_ETT_minute
from models.OscillatorPDMStatePatchMAFirst import Model
from models.OscillatorPDMStatePatchSeparateMLP import Model as SeparateMLP
from scripts.train_timemix_pdm_state_patch import initialize_pair as state_patch_scratch
from utils.time_position_patch_mmd import parts as mmd_parts, numpy_parts as numpy_mmd_parts
from utils.patch_covariance_loss import parts as covariance_parts
from utils.pdm_state_patch_ma_linear_diagnostics import FeatureStatistics, gradient_probe

KEY = 'pdm_state_patch_mafirst_mmd_peak_lr001_results'
PROTOCOL = ROOT/'docs/experiments/PDMStatePatchMAFirstMMDPeakLR001_protocol_20261015.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def process_record():
    return dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                command=[v.decode() for v in Path('/proc/self/cmdline').read_bytes().split(b'\0') if v])


def register(out, status, **extra):
    with (ROOT/'tmp/pdm_state_patch_rope_registration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read(ROOT/'tmp/timemix_0308_goal_state.json')
        section = state.setdefault(KEY, {})
        prior = section.get('current_run', {})
        event = dict(status=status, event=status, output=str(out), updated=now(), **extra)
        section['current_run'] = {**(prior if prior.get('output') == str(out) else {}), **event}
        section['status'] = status
        section.setdefault('run_history', []).append(event)
        atomic(ROOT/'tmp/timemix_0308_goal_state.json', state)


def phase(out, status, **extra):
    record = dict(status=status, updated=now(), **extra)
    atomic(out/'status.json', record)
    print(json.dumps(record), flush=True)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


BASELINE = ROOT/'tmp/lag_study_timemix_pdm_state_patch_time_position_mmd_20261015_seed2021_fixed11'


def config(root):
    values = dict(read(BASELINE/'protocol.json')['model_config'])
    # Keep every architecture option. label48 is ignored by this history-only model.
    values.update(root_path=str(root),label_len=48,batch_size=128,num_workers=10,
                  learning_rate=.001,weight_decay=0.,train_epochs=100,patience=10,
                  lradj='TST',seed=2023,log_dir=str(Path(root).parent/'logs'),tensorboard=False)
    return SimpleNamespace(**values)


def initialize_model(cfg):
    # Same old scratch-conversion/MLP-cloning recipe, fed by the caller's seed.
    # No checkpoint is read and no private seed2021/20211014 is imposed.
    original = state_patch_scratch(cfg)['patch_covariance']
    with torch.random.fork_rng(devices=[]):
        source = SeparateMLP(cfg)
    state = source.state_dict()
    state.update({n:v.clone() for n,v in original.state_dict().items()})
    source.load_state_dict(state,strict=True)
    source.clone_trend_mlps()
    with torch.random.fork_rng(devices=[]):
        model = Model(cfg)
    model.load_state_dict(source.state_dict(),strict=True)
    assert len(model.state_dict())==236 and len(list(model.parameters()))==204
    assert sum(p.numel() for p in model.parameters())==140738
    return model


def output_probe(p,mse,mmd,old):
    gs=[torch.autograd.grad(v,p,retain_graph=True)[0].detach().double() for v in (mse,mmd,old)]
    t=torch.arange(96,device=p.device,dtype=torch.float64);k=torch.arange(4,device=p.device,dtype=torch.float64)
    basis=torch.cos(torch.pi/96*(t[:,None]+.5)*k[None,:]);basis[:,0]*=(1/96)**.5;basis[:,1:] *= (2/96)**.5
    def project(g):return torch.einsum('bkc,tk->btc',torch.einsum('btc,tk->bkc',g,basis),basis)
    result={}
    for label,values in [('all',gs),('low4',[project(g) for g in gs]),('remaining92',[g-project(g) for g in gs]),('level',[g.mean(1,keepdim=True).expand_as(g) for g in gs])]:
        a,b,c=[v.flatten() for v in values];na,nb,nc=[float(v.norm()) for v in (a,b,c)]
        result[label]=dict(mse_norm=na,mmd_norm=nb,old_d_norm=nc,mmd_mse_ratio=nb/max(na,1e-30),mmd_mse_cosine=float(a@b)/max(na*nb,1e-30),old_d_mse_ratio=nc/max(na,1e-30),old_d_mse_cosine=float(a@c)/max(na*nc,1e-30))
    return result


def optimizer_groups(model):
    return [dict(params=list(model.parameters()), weight_decay=0., group_name='official_all_parameters')]



def make_scheduler(optimizer, epochs, steps):
    # Same TST policy, peak .001 only; preserve100epoch horizon and beta1 schedule.
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, steps_per_epoch=steps, epochs=epochs,
        max_lr=[g['lr'] for g in optimizer.param_groups], pct_start=.4)


def dataset(cfg, split, limit=0):
    data = Dataset_ETT_minute(cfg, cfg.root_path, flag=split, size=[96,48,96],
                             features='M', data_path='ETTm1.csv', timeenc=1, freq='h')
    assert len(data)==(34369 if split=='train' else 11425)
    if limit:
        data=Subset(data,np.unique(np.linspace(0,len(data)-1,min(limit,len(data)),dtype=int)).tolist())
    return data



class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,index):return (index,*self.base[index])


def loader(data, purpose='train'):
    # Exact official train/val sampler policy and default worker count. No private generator.
    selection=purpose in ('train','selection_val')
    return DataLoader(IndexedDataset(data),batch_size=128 if selection else 1,
                      num_workers=10,shuffle=selection,drop_last=selection)



def snapshot(out):
    paths = [p for directory in ('models', 'layers', 'utils', 'data_provider', 'exp', 'scripts', 'tests', 'third_party/DistDF')
             for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and (p.suffix == '.py' or directory == 'third_party/DistDF')]
    paths += [PROTOCOL, ROOT/'models/distdf_official/ADAPTATION.json']
    manifest = {}
    for p in sorted(set(paths)):
        rel = p.relative_to(ROOT)
        target = out/'source'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        manifest[str(rel)] = sha(p)
    atomic(out/'sources_manifest.json', manifest)


def verify_sources(out):
    for rel, digest in read(out/'sources_manifest.json').items():
        assert sha(ROOT/rel) == sha(out/'source'/rel) == digest, rel


def selection_validation(model,data,device,out,epoch):
    model.eval();losses=[];indices=[]
    with torch.no_grad():
        for ids,x,target,_,_ in loader(data,'selection_val'):
            x,y=x.float().to(device),target[:,-96:].float().to(device)
            pred=model(x);assert torch.isfinite(pred).all()
            losses.append(nn.functional.mse_loss(pred.detach(),y.detach()))
            indices.extend(ids.tolist())
    value=torch.mean(torch.stack(losses)).item()
    assert len(indices)==len(data)//128*128 and len(set(indices))==len(indices)
    atomic(out/'selection_validation'/f'epoch{epoch:03d}.json',dict(epoch=epoch,mse=value,windows=len(indices),batch_size=128,shuffle=True,drop_last=True,indices=indices))
    model.train();return value


def numpy_covariance_parts(prediction,target):
    total=np.zeros(4,dtype=float)
    for start in range(0,len(prediction),64):
        p=np.asarray(prediction[start:start+64],dtype=float).transpose(0,2,1).reshape(-1,7,24,4)
        y=np.asarray(target[start:start+64],dtype=float).transpose(0,2,1).reshape(p.shape)
        def term(a,b):
            g=np.exp(-.5*((a[..., :,None,:]-b[...,None,:,:])**2).sum(-1))
            g=g-g.mean(-1,keepdims=True)-g.mean(-2,keepdims=True)+g.mean((-1,-2),keepdims=True)
            return (g*g).mean((-1,-2))
        pp,yy,py=term(p,p),term(y,y),term(p,y)
        raw=pp+yy-2*py;assert np.min(raw)>=-1e-10
        total+=np.array([np.maximum(raw,0).mean(),pp.mean(),yy.mean(),py.mean()])*len(p)
    return dict(zip(('patch_covariance','prediction_covariance_norm2','target_covariance_norm2','cross_covariance_inner_product'),map(float,total/len(prediction))))


def evaluate(model,data,device,out,label,capture=False):
    assert capture
    model.eval();destination=out/label;destination.mkdir(exist_ok=False)
    tails={'prediction':(96,7),'truth':(96,7),'history':(96,7),'components':(96,7,8),'history_mean':(1,7)}
    arrays={n:np.lib.format.open_memmap(destination/(n+'.npy'),mode='w+',dtype='float32',shape=(len(data),*tail)) for n,tail in tails.items()}
    offset=0;statistics=FeatureStatistics(model)
    totals={};squared=absolute=0.;count=0
    try:
        with torch.no_grad():
            for ids,x,target,_,_ in loader(data,'capture'):
                x,y=x.float().to(device),target[:,-96:].float().to(device)
                components=model.predict_components(x);norm=model.normalize
                assert not norm.affine and not norm.non_norm and not norm.subtract_last
                p=norm(components[...,:4].sum(-1)+components[...,4:].sum(-1),'denorm')
                assert ids.tolist()==[offset] and torch.isfinite(p).all() and torch.isfinite(components).all()
                _,md=mmd_parts(p,y);_,od=covariance_parts(p,y)
                for k,v in {**md,**od}.items():
                    if isinstance(v,(float,int)):totals[k]=totals.get(k,0.)+v
                values=dict(prediction=p,truth=y,history=x,components=components*norm.stdev.unsqueeze(-1),history_mean=norm.mean)
                for name,v in values.items():
                    assert torch.isfinite(v).all(),name
                    arrays[name][offset:offset+1]=v.cpu().numpy()
                error=p.double()-y.double();squared+=float(error.square().sum());absolute+=float(error.abs().sum());count+=y.numel();offset+=1
                if offset==1 or offset%1000==0:phase(out,label,windows=offset,total_windows=len(data))
    finally:statistics.close()
    for a in arrays.values():a.flush()
    assert offset==len(data)
    measured=dict(mse=squared/count,mae=absolute/count,**{k:v/offset for k,v in totals.items()})
    independent=numpy_mmd_parts(arrays['prediction'],arrays['truth'])
    assert all(abs(measured[k]-v)<1e-6 for k,v in independent.items())
    # Per-sequence kernels are batch-independent. Old NumPy covariance audit also uses saved arrays only.
    oldparts=numpy_covariance_parts(arrays['prediction'],arrays['truth'])
    for k,v in oldparts.items():
        if k in measured:assert abs(measured[k]-v)<1e-6,(k,measured[k],v)
    reconstruction=arrays['components'][...,:4].sum(-1,dtype=np.float32)+arrays['components'][...,4:].sum(-1,dtype=np.float32)+arrays['history_mean']
    assert np.max(np.abs(reconstruction-arrays['prediction']))<2e-5
    atomic(out/(label+'_features.json'),statistics.summary())
    atomic(destination/'report.json',dict(status='verified',windows=offset,metrics=measured,model_evaluation_batch_size=1,all_windows_retained=True,not_used_for_reselection=label=='val',mmd_numpy=independent,old_covariance_numpy=oldparts,component_reconstruction_maxabs=float(np.max(np.abs(reconstruction-arrays['prediction']))),arrays_sha256={n:sha(destination/(n+'.npy')) for n in arrays}))
    return measured


def rng_snapshot():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python']);n=s['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]));torch.set_rng_state(s['torch_cpu'].cpu())
    if s['torch_cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in s['torch_cuda']])


def early_update(value,best,chosen,stale,epoch):
    # Official delta0: equality counts as improvement and replaces checkpoint.
    return (value,epoch,0,True) if value<=best else (best,chosen,stale+1,False)


def main():
    parser=argparse.ArgumentParser(description='Original MAFirst+MMD, matched public recipe training hyperparameters; unchanged architecture and objective.')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--epochs',type=int,default=100);parser.add_argument('--train-limit',type=int,default=0);parser.add_argument('--val-limit',type=int,default=0)
    parser.add_argument('--stop-after-epochs',type=int,default=0);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();smoke=bool(args.train_limit or args.val_limit)
    if smoke:assert min(args.train_limit,args.val_limit)>=256 and 1<=args.epochs<=100
    else:assert args.epochs==100 and not args.resume and not args.stop_after_epochs
    out=args.output.resolve()
    if not args.resume:out.mkdir(parents=True,exist_ok=False)
    lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(out/'process.json',process_record());(out/'pid.txt').write_text(str(os.getpid())+'\n')
    torch.set_num_threads(1)
    if args.device=='cuda':torch.cuda.set_per_process_memory_fraction(.8)
    seed(2023)
    try:
        if not args.resume:
            phase(out,'preparing');prefix=out/'train_val_data';prefix.mkdir()
            with (ROOT/'dataset/ETT-small/ETTm1.csv').open('rb') as source,(prefix/'ETTm1.csv').open('wb') as dest:
                for _ in range(46081):line=source.readline();assert line;dest.write(line)
            cfg=config(prefix)
            atomic(out/'protocol.json',dict(model_config=vars(cfg),options={**vars(args),'output':str(out)},smoke=smoke,formal_protocol=read(PROTOCOL),started=now()))
            atomic(out/'data_manifest.json',dict(prefix_sha256=sha(prefix/'ETTm1.csv'),full_sha256=sha(ROOT/'dataset/ETT-small/ETTm1.csv'),scaler_fit_rows=34560,prefix_rows=46080))
            snapshot(out);shutil.copyfile(ROOT/'scripts/monitor_mafirst_mmd_peak_lr001.py',out/'monitor_check.py')
            (out/'epochs').mkdir();(out/'selection_validation').mkdir()
            atomic(out/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,device=args.device,gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,threads=1,num_workers=10,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,gpu_memory_fraction=.8))
            if not smoke:register(out,'running',**process_record(),protocol=str(PROTOCOL),launch_log=str(out)+'_launch.log')
        else:
            previous=read(out/'protocol.json');assert previous['smoke'] and read(out/'status.json')['status']=='controlled_stop'
            assert all(vars(args)[n]==previous['options'][n] for n in ('epochs','train_limit','val_limit','device'));cfg=SimpleNamespace(**previous['model_config'])
        verify_sources(out);assert sha(Path(cfg.root_path)/'ETTm1.csv')==read(out/'data_manifest.json')['prefix_sha256']
        model=initialize_model(cfg).to(args.device)
        train,val=dataset(cfg,'train',args.train_limit),dataset(cfg,'val',args.val_limit)
        optimizer=torch.optim.Adam(optimizer_groups(model),lr=.001);steps=len(train)//128;scheduler=make_scheduler(optimizer,args.epochs,steps)
        if not args.resume:
            save(out/'initial.pt',model.state_dict());names={id(p):n for n,p in model.named_parameters()}
            atomic(out/'optimizer_groups.json',[dict(group_name=g['group_name'],weight_decay=g['weight_decay'],names=[names[id(p)] for p in g['params']]) for g in optimizer.param_groups])
            atomic(out/'initialization_audit.json',dict(seed=2023,old_trained_weights_loaded=False,parameter_count=sum(p.numel() for p in model.parameters()),parameter_tensors=len(list(model.parameters())),states=len(model.state_dict()),backbone='unchanged OscillatorPDMStatePatchMAFirst; no added trend TCN',initialization='Original scratch-conversion/MLP-clone recipe parameterized by seed2023; no trained/preflight weights loaded',seeding='once before original model construction; no epoch/step reseeding'))
            curves,best,chosen,stale,updates,start=[],math.inf,0,0,0,1
        else:
            last=torch.load(out/'last.pt',map_location=args.device,weights_only=True);model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer']);scheduler.load_state_dict(last['scheduler'])
            curves,best,chosen,stale,updates,start=last['curves'],last['best'],last['best_epoch'],last['stale'],last['updates'],last['epoch']+1;restore_rng(last['rng'])
        for epoch in range(start,args.epochs+1):
            model.train();totals=np.zeros(3);windows=0;order=[];lr=optimizer.param_groups[0]['lr'];beta1=optimizer.param_groups[0]['betas'][0]
            for step,(ids,x,target,_,_) in enumerate(loader(train,'train'),1):
                order.extend(ids.tolist());x,y=x.float().to(args.device),target[:,-96:].float().to(args.device);optimizer.zero_grad(set_to_none=True)
                sampled=step in (1,100,200,steps);statistics=FeatureStatistics(model) if sampled else None
                p=model(x);mse=nn.functional.mse_loss(p,y);distance,md=mmd_parts(p,y);old,od=covariance_parts(p,y);loss=mse+distance
                assert torch.isfinite(p).all() and torch.isfinite(loss) and torch.isfinite(distance)
                if step in (1,100,200,steps):
                    gm,gd=[torch.autograd.grad(v,p,retain_graph=True)[0].detach().double().flatten() for v in (mse,distance)];nm,nd=float(gm.norm()),float(gd.norm())
                    row=dict(epoch=epoch,step=step,mse=float(mse.detach()),mmd=float(distance.detach()),old_covariance=float(old.detach()),mse_norm=nm,mmd_norm=nd,weighted_ratio=nd/max(nm,1e-30),cosine=float(gm@gd)/max(nm*nd,1e-30),mmd_diagnostics=md,old_covariance_diagnostics=od,scopes=gradient_probe(model,mse,distance),old_covariance_scopes=gradient_probe(model,mse,old),output_gradients=output_probe(p,mse,distance,old),features=statistics.summary())
                    statistics.close()
                    with (out/'gradients.jsonl').open('a') as file:file.write(json.dumps(row,allow_nan=False)+'\n')
                loss.backward();inactive=[]
                for name,v in model.named_parameters():
                    if v.grad is None:inactive.append(name)
                    else:assert torch.isfinite(v.grad).all(),name
                assert not inactive, inactive
                if epoch==1 and step==1:atomic(out/'inactive_parameters.json',inactive)
                else:assert inactive==read(out/'inactive_parameters.json')
                ulr=optimizer.param_groups[0]['lr'];ub=optimizer.param_groups[0]['betas'][0];optimizer.step();scheduler.step();updates+=1
                with (out/'scheduler_trace.jsonl').open('a') as file:file.write(json.dumps(dict(epoch=epoch,step=step,update=updates,learning_rates=[ulr],beta1=[ub],next_learning_rates=[optimizer.param_groups[0]['lr']],next_beta1=[optimizer.param_groups[0]['betas'][0]],scheduler_last_epoch=scheduler.last_epoch),allow_nan=False)+'\n')
                assert all(torch.isfinite(v).all() for v in model.parameters());totals+=np.array([float(loss.detach()),float(mse.detach()),float(distance.detach())]);windows+=len(x)
                if step==1 or step%25==0:phase(out,'training',epoch=epoch,step=step,steps_per_epoch=steps,completed_epochs=epoch-1,loss=float(loss.detach()))
            assert len(order)==steps*128 and len(set(order))==len(order)
            phase(out,'validation',epoch=epoch,completed_epochs=epoch-1)
            measured=selection_validation(model,val,args.device,out,epoch);best,chosen,stale,improved=early_update(measured,best,chosen,stale,epoch)
            if improved:save(out/'best.pt',model.state_dict())
            row=dict(epoch=epoch,train_loss=float(totals[0]/steps),train_mse=float(totals[1]/steps),train_mmd=float(totals[2]/steps),val=dict(mse=measured,windows=len(val)//128*128,shuffle=True,drop_last=True),learning_rate=lr,beta1_start=beta1,next_learning_rate=optimizer.param_groups[0]['lr'],next_beta1=optimizer.param_groups[0]['betas'][0],scheduler_last_epoch=scheduler.last_epoch,stale=stale,training_windows=windows,training_indices_sha256=hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest())
            assert abs(row['train_loss']-row['train_mse']-row['train_mmd'])<2e-6;curves.append(row)
            save(out/'epochs'/f'epoch{epoch:03d}.pt',model.state_dict())
            atomic(out/'epochs'/f'epoch{epoch:03d}_weights.json',{n:dict(count=v.numel(),l2=float(v.detach().double().norm()),maxabs=float(v.detach().abs().max())) for n,v in model.named_parameters()})
            last=dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_snapshot(),curves=curves,best=best,best_epoch=chosen,stale=stale,updates=updates)
            save(out/'last.pt',last);atomic(out/'progress.json',{k:v for k,v in last.items() if k not in ('model','optimizer','scheduler','rng')})
            if args.stop_after_epochs and epoch>=args.stop_after_epochs:phase(out,'controlled_stop',completed_epochs=epoch);return
            if stale>=10:break
        assert updates==len(curves)*steps and scheduler.last_epoch==updates and (len(curves)==args.epochs or stale>=10)
        verify_sources(out);phase(out,'freezing',completed_epochs=len(curves));artifact=out/'artifact';artifact.mkdir();shutil.copyfile(out/'best.pt',artifact/'checkpoint.pt');digest=sha(artifact/'checkpoint.pt');atomic(artifact/'model_config.json',vars(cfg))
        selected=torch.load(artifact/'checkpoint.pt',map_location=args.device,weights_only=True);ep=torch.load(out/'epochs'/f'epoch{chosen:03d}.pt',map_location=args.device,weights_only=True);assert all(torch.equal(v,ep[n]) for n,v in selected.items());model.load_state_dict(selected)
        atomic(out/'freeze.json',dict(checkpoint_sha256=digest,selected_epoch=chosen,updates=updates,selection_validation_mse=best,selection='official shuffled/drop_last128 val; ties replace',created=now(),test_opened=False))
        fullval=evaluate(model,val,args.device,out,'val',True)
        report=dict(status='preflight_verified' if smoke else 'completed_verified',validation=fullval,selection_validation_mse=best,selected_epoch=chosen,epochs=len(curves),maximum_epochs=args.epochs,updates=updates,early_stopped=stale>=10,stale=stale,scheduler_horizon_epochs=args.epochs,steps_per_epoch=steps)
        if not smoke:
            verify_sources(out);assert sha(ROOT/'dataset/ETT-small/ETTm1.csv')==read(out/'data_manifest.json')['full_sha256']
            with (out/'test_started.json').open('x') as f:json.dump(dict(checkpoint_sha256=digest,created=now()),f)
            test=dataset(config(ROOT/'dataset/ETT-small'),'test');report['test']=evaluate(model,test,args.device,out,'test',True)
            assert sha(artifact/'checkpoint.pt')==digest;atomic(out/'test_receipt.json',dict(status='completed',windows=11425,evaluations=1,selected_epoch=chosen,checkpoint_sha256=digest,created=now(),evaluation_batch_size=1))
        verify_sources(out);atomic(out/'report.json',report);atomic(out/'completion_audit.json',dict(status='verified',smoke=smoke,test_evaluations=0 if smoke else 1,checkpoint_sha256=digest,epochs=len(curves),maximum_epochs=args.epochs,updates=updates,selection_validation_mse=best))
        if not smoke:register(out,'completed_verified',metrics=report['test'],selected_epoch=chosen)
        phase(out,report['status'],completed_epochs=len(curves))
    except BaseException as error:
        atomic(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),created=now()));phase(out,'failed',error=repr(error));raise


if __name__=='__main__':main()
