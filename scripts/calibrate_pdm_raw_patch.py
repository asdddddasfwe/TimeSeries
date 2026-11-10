"""Train-only one-shot calibration at the common scratch initialization."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys, json, hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from scripts.train_timemix_pdm_increment_patch import dataset, seed_all, TrendCondition, gradient_comparison
from utils.raw_patch_loss import parts


def main():
    os.chdir(ROOT)
    out=ROOT/'tmp/pdm_raw_patch_preflight_20261011'
    out.mkdir(exist_ok=True)
    path=out/'calibration.json'
    if path.exists():
        raise RuntimeError('Calibration already frozen; do not overwrite or retune')
    ref=ROOT/'tmp/lag_study_timemix_pdm_increment_patch_20261010_seed2021_earlystop5/training'
    cfg=SimpleNamespace(**json.loads((ref/'protocol.json').read_text())['model_args'])
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.5)
    seed_all(2021)
    model=TrendCondition(cfg).cuda()
    initial=torch.load(ref/'2021/initial.pth',map_location='cpu',weights_only=True)['mse']
    model.load_state_dict(initial,strict=True)
    model.train()
    train=dataset(cfg,'train',0)
    assert len(train)==34369 and sum(p.numel() for p in model.parameters())==136150
    ids=np.linspace(0,len(train)-1,256,dtype=int).tolist()
    rows=[]
    for i,(x,y,_,_) in enumerate(DataLoader(Subset(train,ids),batch_size=32,shuffle=False,num_workers=0)):
        seed_all(2021+100000+i)
        x=x.float().cuda();y=y[:,-96:].float().cuda()
        p=model(x);q,k,patch=parts(p,y,1.,4)
        mse=(p-y.detach()).square().mean()
        g=gradient_comparison(model,dict(mse=mse,relation_for_gradient=k),1.)
        rows.append(dict(batch=i,seed=2021+100000+i,mse=float(mse.detach()),quadratic=float(q.detach()),kernel=float(k.detach()),saturated_q_gt10=float((patch>10).double().mean()),gradients=g))
    ratio=float(np.median([r['gradients']['all']['weighted_ratio'] for r in rows]))
    weight=.1/ratio
    assert np.isfinite(weight) and weight>0
    assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    record=dict(bandwidth=1.,patch_length=4,weight=weight,target_initial_gradient_ratio=.1,unweighted_ratio_median=ratio,
        rule='Fixed sigma=1 in original training-fitted StandardScaler units; lambda=0.1/median(norm(grad kernel)/norm(grad MSE)) over eight batches at common scratch initial state, train mode dropout0.2; no optimizer steps, no val/test.',
        train_windows=len(train),indices=ids,seed=2021,rows=rows,test_accessed=False,validation_accessed=False,
        identities={str(p.relative_to(ROOT)):sha(p) for p in [ref/'2021/initial.pth',ref/'train_val_data/ETTm1.csv',Path(__file__).resolve(),ROOT/'utils/raw_patch_loss.py']})
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['rows','indices','identities']},indent=2))


if __name__=='__main__':main()
