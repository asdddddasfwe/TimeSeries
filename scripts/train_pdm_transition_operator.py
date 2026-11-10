"""Train/freeze a history-conditioned transition operator using train-only data."""
import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from scripts.train_timemix_pdm_target_nll import dataset, atomic_json, atomic_torch, seed_all
from utils.conditional_transition_bregman import TransitionOperator
from types import SimpleNamespace


def evaluate(models, loader, device):
    result={}
    for name,model in models.items():
        model.eval()
        sums=torch.zeros(len(model.destinations),dtype=torch.double,device=device)
        count=0
        with torch.no_grad():
            for x,y,_,_ in loader:
                x,y=x.float().to(device),y.float().to(device)
                a,b=model(x)
                if name=='history_only': a=torch.zeros_like(a)
                error=model.residual(y,(a,b))
                sums+=error.double().square().sum((0,1))
                count+=len(x)*y.shape[-1]
        per=sums/count
        by_lag={}
        for k in model.options['lags']:
            mask=(model.destinations-model.parents[:,0])==k
            by_lag[str(k)]=float(per[mask].mean())
        result[name]=dict(mse=float(per.mean()),per_lag=by_lag)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--limit',type=int,default=0,help='Preflight only')
    opts=parser.parse_args()
    out=opts.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    if opts.device=='cuda':torch.cuda.set_per_process_memory_fraction(.5)
    seed_all(2021)
    prefix=out/'train_only_data';prefix.mkdir()
    with (ROOT/'dataset/ETT-small/ETTm1.csv').open('rb') as src,(prefix/'ETTm1.csv').open('xb') as dst:
        lines=list(itertools.islice(src,34561));assert len(lines)==34561;dst.writelines(lines)
    cfg=SimpleNamespace(root_path=str(prefix),augmentation_ratio=0)
    data=dataset(cfg,'train',0)
    assert len(data)==34369
    # Fit windows end before row 27552. The internal holdout begins at row
    # 27552; there is no overlapping input or future observation across splits.
    train_ids=list(range(0,27552-192+1,4))
    holdout_ids=list(range(27552,len(data)))
    if opts.limit:
        train_ids=train_ids[:opts.limit]
        holdout_ids=holdout_ids[:opts.limit]
    model=TransitionOperator()
    models={'transition':model.to(opts.device),
            'history_only':copy.deepcopy(model).to(opts.device)}
    optimizers={k:torch.optim.Adam(v.parameters(),lr=.001,weight_decay=1e-4) for k,v in models.items()}
    val_loader=DataLoader(Subset(data,holdout_ids),batch_size=128,shuffle=False,num_workers=0)
    protocol=dict(options=vars(opts)|{'output':str(out)},seed=2021,
        train_indices=train_ids,holdout_indices=holdout_ids,
        fit_last_raw_row=max(train_ids)+191,holdout_first_raw_row=min(holdout_ids),
        train_csv_lines=34561,train_csv_sha256=hashlib.sha256((prefix/'ETTm1.csv').read_bytes()).hexdigest(),
        scaler_fit_rows=34560,external_val_accessed=False,test_accessed=False,
        q=4,lags=[1,4,12,24],coefficient_l1_bound=2.,coefficient_l2_penalty=.001,
        train_stride=4,selection='minimum train-internal purged holdout transition MSE',
        smoke=bool(opts.limit),source_sha256={str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
            for p in [Path('scripts/train_pdm_transition_operator.py'),Path('utils/conditional_transition_bregman.py')]})
    atomic_json(out/'protocol.json',protocol)
    best={k:float('inf') for k in models};best_epoch={};curves=[];stale=0
    for epoch in range(1,opts.epochs+1):
        order=torch.randperm(len(train_ids),generator=torch.Generator().manual_seed(2021000+epoch)).tolist()
        loader=DataLoader(Subset(data,[train_ids[i] for i in order]),batch_size=128,num_workers=0)
        sums={k:0. for k in models}
        for m in models.values():m.train()
        for step,(x,y,_,_) in enumerate(loader):
            x,y=x.float().to(opts.device),y.float().to(opts.device)
            for name,m in models.items():
                optimizers[name].zero_grad(set_to_none=True)
                a,b=m(x)
                if name=='history_only':a=a*0
                error=m.residual(y,(a,b))
                loss=error.square().mean()+.001*a.square().mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite operator loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters()):
                    raise RuntimeError('Nonfinite operator gradient')
                optimizers[name].step()
                sums[name]+=float(error.detach().square().mean())*len(x)
            atomic_json(out/'status.json',dict(status='training_operator',epoch=epoch,step=step+1,steps=len(loader)))
        scores=evaluate(models,val_loader,opts.device);improved=False
        for name,m in models.items():
            curves.append(dict(epoch=epoch,variant=name,train_mse=sums[name]/len(train_ids),**scores[name]))
            if scores[name]['mse']<best[name]:
                best[name]=scores[name]['mse'];best_epoch[name]=epoch;improved=True
                atomic_torch(out/(name+'_best.pth'),dict(options=m.options,state_dict={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}))
        stale=0 if improved else stale+1
        atomic_json(out/'progress.json',dict(epoch=epoch,best=best,best_epoch=best_epoch,curves=curves,stale=stale))
        print('OPERATOR EPOCH',epoch,scores,flush=True)
        if stale>=3:break
    for name,m in models.items():m.load_state_dict(torch.load(out/(name+'_best.pth'),weights_only=True)['state_dict'])
    scores=evaluate(models,val_loader,opts.device)
    improvement=1-scores['transition']['mse']/scores['history_only']['mse']
    artifact=out/'transition_best.pth'
    report=dict(status='completed',scores=scores,relative_improvement=improvement,
        gate_passed=bool(improvement>0),formal_gate=not bool(opts.limit),best_epoch=best_epoch,
        operator_checkpoint=str(artifact),operator_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        parameter_count=sum(p.numel() for p in models['transition'].parameters()),
        relations=len(model.destinations),test_accessed=False,external_val_accessed=False)
    atomic_json(out/'report.json',report);atomic_json(out/'status.json',dict(status='completed'))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
