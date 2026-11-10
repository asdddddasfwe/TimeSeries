"""Official Fredformer+DistDF ETTm1/96 script recipe: seed2024 batch128 TST100 patience10."""
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
from models.DistDFFredformer import Model
from utils.distdf_joint_wasserstein import distdf_loss

KEY = 'distdf_fredformer_official_seed2024_results'
PROTOCOL = ROOT/'docs/experiments/DistDFFredformerOfficialSeed2024_protocol_20261016.json'


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


def config(root):
    # Official ETTm1/96 structure and parser defaults, without changing dimensions.
    return SimpleNamespace(task_name='long_term_forecast', model='DistDFFredformer',
        root_path=str(root), data_path='ETTm1.csv', augmentation_ratio=0,
        seq_len=96, label_len=48, pred_len=96, enc_in=7, dec_in=7, c_out=7, features='M',
        e_layers=3, n_heads=16, d_model=24, d_ff=256, dropout=.2, fc_dropout=.2,
        head_dropout=0., patch_len=4, stride=4, padding_patch='end', individual=0,
        revin=1, affine=0, subtract_last=0, use_nys=0, ablation=0,
        cf_dim=128, cf_depth=2, cf_heads=8, cf_mlp=96, cf_head_dim=32,
        cf_drop=.2, mlp_hidden=64, mlp_drop=.3)


def optimizer_groups(model):
    return [dict(params=list(model.parameters()), weight_decay=0., group_name='official_all_parameters')]



def make_scheduler(optimizer, epochs, steps):
    # Exact official TST constructor/defaults with .0001 peak and maximum100 epochs.
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


def numpy_distdf(p, y, h):
    x, z = [np.concatenate((h, v), 1).astype(np.float64).transpose(2, 0, 1) for v in (p, y)]
    xm, zm = x.mean(1, keepdims=True), z.mean(1, keepdims=True)
    x, z = x-xm, z-zm
    eye = np.eye(x.shape[-1])[None]
    a = x.transpose(0, 2, 1) @ x / x.shape[1]+.005*eye
    b = z.transpose(0, 2, 1) @ z / z.shape[1]+.005*eye
    cov = np.trace(a, axis1=1, axis2=2)+np.trace(b, axis1=1, axis2=2)
    cov -= 2*np.sqrt(np.maximum((a*b.transpose(0, 2, 1)).sum((1, 2)), 1e-8))
    return float(np.sqrt(np.maximum(((xm-zm)**2).sum((1, 2))+.002*cov, 1e-8)).mean())


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


def evaluate(model,data,device,out,label,capture=False):
    # Official test batch1. Full validation is post-selection audit only.
    assert capture
    model.eval();destination=out/label;destination.mkdir(exist_ok=False)
    arrays={n:np.lib.format.open_memmap(destination/(n+'.npy'),mode='w+',dtype='float32',shape=(len(data),96,7)) for n in ('prediction','truth','history')}
    offset=0
    with torch.no_grad():
        for ids,x,target,_,_ in loader(data,'capture'):
            x,y=x.float().to(device),target[:,-96:].float().to(device);p=model(x)
            assert torch.isfinite(p).all() and ids.tolist()==[offset]
            for name,v in [('prediction',p),('truth',y),('history',x)]:arrays[name][offset:offset+len(x)]=v.cpu().numpy()
            offset+=len(x)
            if offset==1 or offset%1000==0:phase(out,label,windows=offset,total_windows=len(data))
    for a in arrays.values():a.flush()
    assert offset==len(data)
    mse=mae=0.
    for start in range(0,offset,128):
        p,y,h=[np.asarray(arrays[n][start:start+128]) for n in ('prediction','truth','history')];assert all(np.isfinite(v).all() for v in (p,y,h))
        err=p.astype(float)-y;mse+=float((err**2).sum());mae+=float(abs(err).sum())
    metrics=dict(mse=mse/(offset*96*7),mae=mae/(offset*96*7))
    atomic(destination/'report.json',dict(status='verified',windows=offset,metrics=metrics,model_evaluation_batch_size=1,all_windows_retained=True,not_used_for_reselection=label=='val',arrays_sha256={n:sha(destination/(n+'.npy')) for n in arrays}))
    return metrics




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
    parser=argparse.ArgumentParser(description='Fredformer+DistDF official ETTm1/96 script recipe, isolated dependencies and evidence logging.')
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
    seed(2024)
    try:
        if not args.resume:
            phase(out,'preparing');prefix=out/'train_val_data';prefix.mkdir()
            with (ROOT/'dataset/ETT-small/ETTm1.csv').open('rb') as source,(prefix/'ETTm1.csv').open('wb') as dest:
                for _ in range(46081):line=source.readline();assert line;dest.write(line)
            cfg=config(prefix)
            atomic(out/'protocol.json',dict(model_config=vars(cfg),options={**vars(args),'output':str(out)},smoke=smoke,formal_protocol=read(PROTOCOL),started=now()))
            atomic(out/'data_manifest.json',dict(prefix_sha256=sha(prefix/'ETTm1.csv'),full_sha256=sha(ROOT/'dataset/ETT-small/ETTm1.csv'),scaler_fit_rows=34560,prefix_rows=46080))
            snapshot(out);shutil.copyfile(ROOT/'scripts/monitor_distdf_fredformer_official_seed2024.py',out/'monitor_check.py')
            (out/'epochs').mkdir();(out/'selection_validation').mkdir()
            atomic(out/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,device=args.device,gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,threads=1,num_workers=10,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,gpu_memory_fraction=.8))
            if not smoke:register(out,'running',**process_record(),protocol=str(PROTOCOL),launch_log=str(out)+'_launch.log')
        else:
            previous=read(out/'protocol.json');assert previous['smoke'] and read(out/'status.json')['status']=='controlled_stop'
            assert all(vars(args)[n]==previous['options'][n] for n in ('epochs','train_limit','val_limit','device'));cfg=SimpleNamespace(**previous['model_config'])
        verify_sources(out);assert sha(Path(cfg.root_path)/'ETTm1.csv')==read(out/'data_manifest.json')['prefix_sha256']
        model=Model(cfg).to(args.device)
        train,val=dataset(cfg,'train',args.train_limit),dataset(cfg,'val',args.val_limit)
        optimizer=torch.optim.Adam(optimizer_groups(model),lr=.0001);steps=len(train)//128;scheduler=make_scheduler(optimizer,args.epochs,steps)
        if not args.resume:
            save(out/'initial.pt',model.state_dict());names={id(p):n for n,p in model.named_parameters()}
            atomic(out/'optimizer_groups.json',[dict(group_name=g['group_name'],weight_decay=g['weight_decay'],names=[names[id(p)] for p in g['params']]) for g in optimizer.param_groups])
            atomic(out/'initialization_audit.json',dict(seed=2024,old_trained_weights_loaded=False,parameter_count=sum(p.numel() for p in model.parameters()),parameter_tensors=len(list(model.parameters())),states=len(model.state_dict()),backbone='official Fredformer ETTm1/96',seeding='once before original model construction; no epoch/step reseeding'))
            curves,best,chosen,stale,updates,start=[],math.inf,0,0,0,1
        else:
            last=torch.load(out/'last.pt',map_location=args.device,weights_only=True);model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer']);scheduler.load_state_dict(last['scheduler'])
            curves,best,chosen,stale,updates,start=last['curves'],last['best'],last['best_epoch'],last['stale'],last['updates'],last['epoch']+1;restore_rng(last['rng'])
        for epoch in range(start,args.epochs+1):
            model.train();totals=np.zeros(3);windows=0;order=[];lr=optimizer.param_groups[0]['lr'];beta1=optimizer.param_groups[0]['betas'][0]
            for step,(ids,x,target,_,_) in enumerate(loader(train,'train'),1):
                order.extend(ids.tolist());x,y=x.float().to(args.device),target[:,-96:].float().to(args.device);optimizer.zero_grad(set_to_none=True)
                p=model(x);mse=nn.functional.mse_loss(p,y);distance=distdf_loss(p,y,x);loss=.5*mse+.5*distance
                assert torch.isfinite(p).all() and torch.isfinite(loss) and torch.isfinite(distance)
                if step in (1,100,200,steps):
                    gm,gd=[torch.autograd.grad(v,p,retain_graph=True)[0].detach().double().flatten() for v in (mse,distance)];nm,nd=float(gm.norm()),float(gd.norm())
                    row=dict(epoch=epoch,step=step,mse=float(mse.detach()),distdf=float(distance.detach()),mse_norm=nm,distdf_norm=nd,weighted_ratio=nd/max(nm,1e-30),cosine=float(gm@gd)/max(nm*nd,1e-30))
                    with (out/'gradients.jsonl').open('a') as file:file.write(json.dumps(row,allow_nan=False)+'\n')
                loss.backward();inactive=[]
                for name,v in model.named_parameters():
                    if v.grad is None:inactive.append(name)
                    else:assert torch.isfinite(v.grad).all(),name
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
            row=dict(epoch=epoch,train_loss=float(totals[0]/steps),train_mse=float(totals[1]/steps),train_distdf=float(totals[2]/steps),val=dict(mse=measured,windows=len(val)//128*128,shuffle=True,drop_last=True),learning_rate=lr,beta1_start=beta1,next_learning_rate=optimizer.param_groups[0]['lr'],next_beta1=optimizer.param_groups[0]['betas'][0],scheduler_last_epoch=scheduler.last_epoch,stale=stale,training_windows=windows,training_indices_sha256=hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest())
            assert abs(row['train_loss']-.5*row['train_mse']-.5*row['train_distdf'])<2e-6;curves.append(row)
            save(out/'epochs'/f'epoch{epoch:03d}.pt',model.state_dict())
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
