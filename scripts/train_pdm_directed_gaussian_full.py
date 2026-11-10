"""Full ETTm1 fixed-mean conditional Gaussian comparison; no test split.

Stage 1 fits a fresh point backbone on early training rows. Stage 2 freezes it
and fits both probability heads on disjoint later training windows. Cached
features are exact eval forwards, not a learned approximation. Resume is explicit.
"""
import argparse,copy,fcntl,hashlib,json,math,os,random,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,Subset
from sklearn.preprocessing import StandardScaler
from models.OscillatorPDMTrendCondition import Model as PointModel
from layers.DirectedGaussianHead import DirectedGaussianHead
from utils.directed_gaussian import gaussian_nll,predictive_covariance,sample_paths
from utils.timemix_decay_study import parameter_groups as decay_groups

ARMS=('diagonal','directed');LAGS=(1,2,4,8,12,24)
BASE_PROTOCOL=ROOT/'tmp/lag_study_timemix_pdm_innovation_20261006_seed2021_earlystop5/training/protocol.json'

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def atomic_torch(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def status(out,phase,**kwargs):atomic_json(out/'status.json',dict(status=phase,updated_utc=now(),**kwargs))
def lr_at(epoch,total):return .001*.5*(1+math.cos(math.pi*epoch/total))

class Windows(Dataset):
    def __init__(self,data,start,end,limit=0):
        self.data=data;self.ids=np.arange(start,end-191,dtype=np.int64)
        if limit:self.ids=self.ids[np.unique(np.linspace(0,len(self.ids)-1,min(limit,len(self.ids)),dtype=int))]
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        i=self.ids[index];return self.data[i:i+96].astype('float32'),self.data[i+96:i+192].astype('float32')

def datasets(opts):
    # Read train+validation prefix only; never read test rows.
    frame=pd.read_csv(ROOT/'dataset/ETT-small/ETTm1.csv',nrows=46080)
    assert len(frame)==46080 and frame.shape[1]==8
    raw=frame.iloc[:,1:].to_numpy(dtype='float64');scaler=StandardScaler().fit(raw[:34560]);data=scaler.transform(raw)
    sets=dict(mean=Windows(data,0,17280,opts.train_limit),probability=Windows(data,17280,34560,opts.train_limit),val=Windows(data,34464,46080,opts.val_limit))
    details=dict(channels=frame.columns[1:].tolist(),raw_rows_read=46080,training_rows=34560,mean_raw_bounds=[0,17280],probability_raw_bounds=[17280,34560],validation_history_start=34464,validation_target_bounds=[34560,46080],window_counts={k:len(v) for k,v in sets.items()},scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),raw_prefix_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),test_accessed=False)
    return sets,details

def loader(dataset,seed=None):
    if seed is not None:dataset=Subset(dataset,torch.randperm(len(dataset),generator=torch.Generator().manual_seed(seed)).tolist())
    return DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0,drop_last=False)

def check_grads(model):
    grads=[p.grad for p in model.parameters() if p.requires_grad]
    if any(g is None for g in grads) or not all(bool(torch.isfinite(g).all()) for g in grads):raise RuntimeError('Missing/nonfinite parameter gradient')

@torch.no_grad()
def evaluate_mean(model,dataset,device,out):
    model.eval();sq=ab=0.;count=0
    for step,(x,y) in enumerate(loader(dataset)):
        p=model(x.to(device));err=(p-y.to(device)).double()
        if not torch.isfinite(err).all():raise RuntimeError('Nonfinite mean validation')
        sq+=float(err.square().sum());ab+=float(err.abs().sum());count+=err.numel()
        if step%50==0:status(out,'mean_validation',step=step+1)
    return dict(mse=sq/count,mae=ab/count)

def mean_stage(out,state,cfg,sets,opts):
    seed_all(opts.seed);model=PointModel(cfg).to(opts.device)
    optimizer=torch.optim.Adam(decay_groups(model,'adam_norm_bias'),lr=.001)
    if state.get('mean_model') is not None:model.load_state_dict(state['mean_model']);optimizer.load_state_dict(state['mean_optimizer'])
    else:
        atomic_torch(out/'mean_initial.pt',cpu_state(model));state.update(mean_epoch=0,mean_best_score=1e30,mean_best_epoch=0,mean_stale=0,mean_curves=[])
    for epoch in range(state['mean_epoch']+1,opts.mean_epochs+1):
        if state['mean_stale']>=opts.patience:break
        for group in optimizer.param_groups:group['lr']=lr_at(epoch-1,opts.mean_epochs)
        model.train();total=0.;start=time.monotonic();batches=loader(sets['mean'],opts.seed*1000+epoch)
        for step,(x,y) in enumerate(batches):
            seed_all(opts.seed*1000000+epoch*10000+step);optimizer.zero_grad(set_to_none=True)
            p=model(x.to(opts.device));loss=(p-y.to(opts.device)).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite mean MSE')
            loss.backward();check_grads(model);optimizer.step();total+=float(loss.detach())*len(x)
            if step%50==0 or step+1==len(batches):status(out,'mean_training',epoch=epoch,step=step+1,steps=len(batches),mse=float(loss.detach()));print('MEAN',epoch,step+1,len(batches),float(loss.detach()),flush=True)
        validation=evaluate_mean(model,sets['val'],opts.device,out);improved=validation['mse']<state['mean_best_score']
        if improved:
            state.update(mean_best_score=validation['mse'],mean_best_epoch=epoch);atomic_torch(out/'mean_best.pt',cpu_state(model))
        state['mean_stale']=0 if improved else state['mean_stale']+1
        state['mean_curves'].append(dict(epoch=epoch,train_mse=total/len(sets['mean']),validation=validation,updates=epoch*len(batches),learning_rate=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-start))
        state.update(mean_epoch=epoch,mean_model=cpu_state(model),mean_optimizer=optimizer.state_dict());atomic_torch(out/'last.pt',state)
        atomic_json(out/'mean_progress.json',{k:state[k] for k in ['mean_epoch','mean_best_score','mean_best_epoch','mean_stale','mean_curves']})
        if opts.stop_mean and epoch>=opts.stop_mean:status(out,'verification_stop',stage='mean',epoch=epoch);return False
    state['stage']='cache';atomic_torch(out/'last.pt',state);return True

def context_forward(model,x):
    seen=[]
    def capture(module,args,output):seen.append(torch.cat([v for stream in output for z in stream for v in (z.mean(1),z[:,-1])],-1))
    handle=model.pdm_blocks[-1].register_forward_hook(capture)
    try:mean=model(x)
    finally:handle.remove()
    if len(seen)!=1:raise RuntimeError('Unexpected encoding count')
    return seen[0].reshape(len(x),7,-1),mean,model.normalize.stdev.square().expand(-1,96,-1)

@torch.no_grad()
def cache_stage(out,state,cfg,sets,opts):
    model=PointModel(cfg).to(opts.device).eval();model.load_state_dict(torch.load(out/'mean_best.pt',map_location=opts.device,weights_only=True))
    for split in ['probability','val']:
        folder=out/'cache'/split;folder.mkdir(parents=True,exist_ok=True);manifest=folder/'manifest.json'
        if manifest.exists():
            meta=json.loads(manifest.read_text());assert meta['mean_checkpoint_sha256']==sha(out/'mean_best.pt')
            assert all(sha(folder/name)==digest for name,digest in meta['sha256'].items());continue
        arrays={};cursor=0
        for step,(x,y) in enumerate(loader(sets[split])):
            context,mean,scale=context_forward(model,x.to(opts.device))
            values=dict(context=context.cpu().numpy(),mean=mean.cpu().numpy(),target=y.numpy(),scale=scale.cpu().numpy())
            for name,value in values.items():
                if not np.isfinite(value).all():raise RuntimeError('Nonfinite cache '+name)
                if name not in arrays:arrays[name]=np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',dtype='float32',shape=(len(sets[split]),*value.shape[1:]))
                arrays[name][cursor:cursor+len(x)]=value
            cursor+=len(x)
            if step%25==0:status(out,'cache',split=split,windows=cursor,total=len(sets[split]));print('CACHE',split,cursor,flush=True)
        assert cursor==len(sets[split])
        for value in arrays.values():value.flush()
        shapes={name:list(v.shape) for name,v in arrays.items()};del arrays
        np.save(folder/'window_start_indices.npy',sets[split].ids)
        atomic_json(manifest,dict(count=cursor,shapes=shapes,mean_checkpoint_sha256=sha(out/'mean_best.pt'),sha256={p.name:sha(p) for p in folder.glob('*.npy')}))
    state['stage']='probability';atomic_torch(out/'last.pt',state)

class Cached(Dataset):
    def __init__(self,folder):self.arrays={name:np.load(folder/(name+'.npy'),mmap_mode='r') for name in ['context','mean','target','scale']}
    def __len__(self):return len(self.arrays['mean'])
    def __getitem__(self,i):return tuple(np.array(v[i],copy=True) for v in self.arrays.values())

def make_heads(width,seed,device):
    heads={}
    for arm in ARMS:
        seed_all(seed);heads[arm]=DirectedGaussianHead(width,96,() if arm=='diagonal' else LAGS).to(device)
    for key,value in heads['diagonal'].state_dict().items():
        if key not in ('lag_mask',):torch.testing.assert_close(value,heads['directed'].state_dict()[key],rtol=0,atol=0)
    return heads

def head_groups(head):
    return [dict(params=[p for n,p in head.named_parameters() if not n.endswith('bias')],weight_decay=1e-4),dict(params=[p for n,p in head.named_parameters() if n.endswith('bias')],weight_decay=0.)]

@torch.no_grad()
def evaluate_heads(heads,dataset,device,out):
    for head in heads.values():head.eval()
    nll={k:0. for k in heads};sq=ab=0.;count=0
    for step,(context,mean,target,scale) in enumerate(loader(dataset)):
        context,mean,target,scale=[v.to(device) for v in (context,mean,target,scale)]
        for name,head in heads.items():
            b,d=head(context);loss=gaussian_nll(mean,target,b,d*scale,head.lags)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite probability validation NLL')
            nll[name]+=float(loss)*target.numel()
        err=(mean-target).double();sq+=float(err.square().sum());ab+=float(err.abs().sum());count+=target.numel()
        if step%50==0:status(out,'probability_validation',step=step+1)
    return {k:dict(nll=v/count,mse=sq/count,mae=ab/count) for k,v in nll.items()}

def probability_stage(out,state,opts):
    train=Cached(out/'cache/probability');val=Cached(out/'cache/val');heads=make_heads(train.arrays['context'].shape[-1],opts.seed,opts.device)
    optimizers={k:torch.optim.Adam(head_groups(h),lr=.001) for k,h in heads.items()}
    if 'heads' in state:
        for k in heads:heads[k].load_state_dict(state['heads'][k]);optimizers[k].load_state_dict(state['head_optimizers'][k])
    else:
        initial={k:cpu_state(h) for k,h in heads.items()};atomic_torch(out/'probability_initial.pt',initial)
        atomic_json(out/'initialization_audit.json',dict(common_head_parameters_exact=True,initial_coefficients_zero=True,initial_normalized_variance=1.,trainable_parameters={k:sum(p.numel() for p in h.parameters()) for k,h in heads.items()},frozen_mean_sha256=sha(out/'mean_best.pt')))
        state.update(probability_epoch=0,probability_stale=0,best={k:1e30 for k in heads},best_epoch={k:0 for k in heads},probability_curves=[])
    for epoch in range(state['probability_epoch']+1,opts.epochs+1):
        if state['probability_stale']>=opts.patience:break
        for name,head in heads.items():
            head.train()
            for group in optimizers[name].param_groups:group['lr']=lr_at(epoch-1,opts.epochs)
        total={k:0. for k in heads};batches=loader(train,opts.seed*1000+epoch);start=time.monotonic()
        for step,(context,mean,target,scale) in enumerate(batches):
            context,mean,target,scale=[v.to(opts.device) for v in (context,mean,target,scale)]
            for name,head in heads.items():
                optimizer=optimizers[name];optimizer.zero_grad(set_to_none=True);b,d=head(context);d=d*scale
                loss=gaussian_nll(mean,target,b,d,head.lags)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite probability train NLL')
                loss.backward();check_grads(head)
                if step==0 or (step+1)%200==0:
                    row=dict(epoch=epoch,step=step+1,arm=name,nll=float(loss.detach()),variance_min=float(d.detach().min()),variance_max=float(d.detach().max()),coefficient_row_l1_max=float(b.detach().abs().sum(-1).max()) if name=='directed' else 0.,parameter_gradients={n:float(p.grad.norm()) for n,p in head.named_parameters()})
                    with (out/'gradient_traces.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                optimizer.step();total[name]+=float(loss.detach())*len(context)
            if step%50==0 or step+1==len(batches):status(out,'probability_training',epoch=epoch,step=step+1,steps=len(batches));print('PROBABILITY',epoch,step+1,len(batches),flush=True)
        validation=evaluate_heads(heads,val,opts.device,out);improved=False
        for name,head in heads.items():
            if validation[name]['nll']<state['best'][name]:
                state['best'][name]=validation[name]['nll'];state['best_epoch'][name]=epoch;atomic_torch(out/(name+'_best.pt'),cpu_state(head));improved=True
            state['probability_curves'].append(dict(epoch=epoch,arm=name,train_nll=total[name]/len(train),validation=validation[name],updates=epoch*len(batches),learning_rate=optimizers[name].param_groups[0]['lr']))
        state['probability_stale']=0 if improved else state['probability_stale']+1
        state.update(probability_epoch=epoch,heads={k:cpu_state(h) for k,h in heads.items()},head_optimizers={k:o.state_dict() for k,o in optimizers.items()});atomic_torch(out/'last.pt',state)
        atomic_json(out/'progress.json',{**{k:state[k] for k in ['probability_epoch','probability_stale','best','best_epoch','probability_curves']},'epoch_seconds':time.monotonic()-start})
        print('EPOCH',epoch,validation,'best',state['best_epoch'],'stale',state['probability_stale'],flush=True)
        if opts.stop_probability and epoch>=opts.stop_probability:status(out,'verification_stop',stage='probability',epoch=epoch);return False
    state['stage']='capture';atomic_torch(out/'last.pt',state);return True

@torch.no_grad()
def capture(out,state,opts):
    val=Cached(out/'cache/val');heads=make_heads(val.arrays['context'].shape[-1],opts.seed,opts.device);runs={}
    for name,head in heads.items():
        head.load_state_dict(torch.load(out/(name+'_best.pt'),map_location=opts.device,weights_only=True));head.eval();folder=out/'selected'/name;folder.mkdir(parents=True,exist_ok=True);arrays={};cursor=0
        for step,(context,mean,target,scale) in enumerate(loader(val)):
            context,mean,target,scale=[v.to(opts.device) for v in (context,mean,target,scale)];b,d=head(context);d=d*scale
            terms=gaussian_nll(mean,target,b,d,head.lags,reduction='none');cov=predictive_covariance(mean,b,d,head.lags);marginal=cov.diagonal(dim1=-2,dim2=-1).transpose(1,2)
            values=dict(innovation_variance=d.cpu().numpy(),marginal_variance=marginal.cpu().numpy(),nll_terms=terms.cpu().numpy())
            if name=='directed':values['coefficients']=b.cpu().numpy()
            for key,v in values.items():
                if not np.isfinite(v).all():raise RuntimeError('Nonfinite selected '+key)
                if key not in arrays:arrays[key]=np.lib.format.open_memmap(folder/(key+'.npy'),mode='w+',dtype='float32',shape=(len(val),*v.shape[1:]))
                arrays[key][cursor:cursor+len(context)]=v
            if cursor==0:
                paths=sample_paths(mean[:16],b[:16],d[:16],head.lags,samples=64,generator=torch.Generator(device=opts.device).manual_seed(opts.seed+100))
                np.savez_compressed(folder/'fixed_first16_probe.npz',mean=mean[:16].cpu().numpy(),target=target[:16].cpu().numpy(),coefficients=b[:16].cpu().numpy(),variance=d[:16].cpu().numpy(),covariance=cov[:16].cpu().numpy(),samples=paths.cpu().numpy())
            cursor+=len(context)
            if step%25==0:status(out,'selected_validation_capture',arm=name,windows=cursor,total=len(val))
        for v in arrays.values():v.flush()
        mu=np.array(val.arrays['mean'],dtype='float64');truth=np.array(val.arrays['target'],dtype='float64');err=truth-mu;var=np.array(arrays['marginal_variance'],dtype='float64');nll=np.array(arrays['nll_terms'],dtype='float64');std=np.sqrt(var)
        coverage={};width={}
        for level,z in [('80',1.2815515655446004),('90',1.6448536269514722),('95',1.959963984540054)]:coverage[level]=float(np.mean(np.abs(err)<=z*std));width[level]=float(np.mean(2*z*std))
        rows=[]
        for c in range(7):rows.append(dict(mse=float(np.mean(err[:,:,c]**2)),mae=float(np.mean(np.abs(err[:,:,c]))),nll=float(np.mean(nll[:,:,c])),coverage90=float(np.mean(np.abs(err[:,:,c])<=1.6448536269514722*std[:,:,c])),width90=float(np.mean(2*1.6448536269514722*std[:,:,c]))))
        nll_value=float(nll.mean());assert abs(nll_value-state['best'][name])<2e-5
        quarters={}
        for kind,axis in [('time',0),('horizon',1)]:
            quarters[kind]=[dict(nll=float(np.take(nll,ids,axis=axis).mean()),mse=float(np.mean(np.take(err,ids,axis=axis)**2)),coverage90=float(np.mean(np.abs(np.take(err,ids,axis=axis))<=1.6448536269514722*np.take(std,ids,axis=axis)))) for ids in np.array_split(np.arange(err.shape[axis]),4)]
        runs[name]=dict(selected_epoch=state['best_epoch'][name],selected_updates=state['best_epoch'][name]*math.ceil(len(Cached(out/'cache/probability'))/32),nll=nll_value,mse=float(np.mean(err**2)),mae=float(np.mean(np.abs(err))),coverage=coverage,interval_width=width,per_channel=rows,quarters=quarters,checkpoint_sha256=sha(out/(name+'_best.pt')),artifact_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
        del arrays,mu,truth,err,var,nll,std
    assert runs['diagonal']['mse']==runs['directed']['mse'] and runs['diagonal']['mae']==runs['directed']['mae']
    manifest=dict(mean_checkpoint_sha256=sha(out/'mean_best.pt'),selected=runs,selection_metric='validation_joint_nll',selection_status='frozen_validation_diagnostic',test_accessed=False)
    atomic_json(out/'freeze_manifest.json',manifest)
    result=dict(status='completed_pending_audit',mode='fixed_mean_disjoint_training',mean_epochs=state['mean_epoch'],mean_selected_epoch=state['mean_best_epoch'],probability_epochs=state['probability_epoch'],stop_reason='joint_patience' if state['probability_stale']>=opts.patience else 'max_epochs',runs=runs,point_mean_shared=True,normalization_modified=False,test_accessed=False,finished_utc=now())
    atomic_json(out/'report.json',result);state['stage']='completed';atomic_torch(out/'last.pt',state);status(out,'completed_pending_audit');print('TRAINING AND CAPTURE COMPLETE',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--resume',type=Path);parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');parser.add_argument('--seed',type=int,default=2021);parser.add_argument('--epochs',type=int,default=30);parser.add_argument('--mean-epochs',type=int,default=30);parser.add_argument('--patience',type=int,default=5);parser.add_argument('--train-limit',type=int,default=0);parser.add_argument('--val-limit',type=int,default=0);parser.add_argument('--stop-mean',type=int,default=0);parser.add_argument('--stop-probability',type=int,default=0);opts=parser.parse_args()
    stop_mean,stop_probability=opts.stop_mean,opts.stop_probability;out=(opts.resume or opts.output).resolve()
    torch.set_num_threads(2)
    if opts.resume:
        protocol=json.loads((out/'protocol.json').read_text());opts=SimpleNamespace(**protocol['options']);opts.stop_mean=stop_mean;opts.stop_probability=stop_probability
        for p,digest in protocol['source_sha256'].items():assert sha(ROOT/p)==digest,('Source changed',p)
        state=torch.load(out/'last.pt',map_location=opts.device,weights_only=False)
    else:
        if min(opts.epochs,opts.mean_epochs,opts.patience)<1:raise ValueError('Invalid budget')
        out.mkdir(parents=True,exist_ok=False);cfg=json.loads(BASE_PROTOCOL.read_text())['model_args'];cfg.update(log_dir=str(out/'logs'),tensorboard=False,tensorboard_dir=str(out/'tensorboard'))
        sources={}
        for directory in ['models','layers','utils']:
            for p in (ROOT/directory).rglob('*.py'):sources[str(p.relative_to(ROOT))]=sha(p)
        for p in [Path(__file__),ROOT/'docs/experiments/PDMDirectedGaussianFull_protocol_20261007.md']:
            sources[str(p.relative_to(ROOT))]=sha(p)
        for path in sources:
            dest=out/'source'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/path).read_bytes())
        options={k:str(v) if isinstance(v,Path) else v for k,v in vars(opts).items()};protocol=dict(objective='fixed_mean_directed_gaussian_v1',options=options,model_args=cfg,source_sha256=sources,started_utc=now());atomic_json(out/'protocol.json',protocol);state={'stage':'mean'}
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if opts.device=='cuda':torch.cuda.set_per_process_memory_fraction(1.)
    atomic_json(out/'process.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),command=sys.argv,started_utc=now()))
    try:
        cfg=SimpleNamespace(**protocol['model_args']);sets,details=datasets(opts)
        if (out/'data_protocol.json').exists():assert json.loads((out/'data_protocol.json').read_text())==details
        else:atomic_json(out/'data_protocol.json',details)
        if state['stage']=='mean' and not mean_stage(out,state,cfg,sets,opts):return
        if state['stage']=='cache':cache_stage(out,state,cfg,sets,opts)
        if state['stage']=='probability' and not probability_stage(out,state,opts):return
        if state['stage']=='capture':capture(out,state,opts)
    except BaseException as exc:
        status(out,'failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:lock.close()
if __name__=='__main__':main()
