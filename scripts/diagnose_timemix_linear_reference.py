"""Exact regularized linear reference on identical 96-point histories; no test."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from utils.retained_timemix_helpers import dataset, atomic_json
from utils.retained_timemix_helpers import metrics

LAMBDAS=(1e-5,1e-4,1e-3,1e-2,1e-1,1.)


def design(x):
    # RevIN -> Linear -> inverse RevIN, rewritten to minimize actual output MSE.
    # p = mean(x) + [x-mean(x), std(x)] @ coefficient.
    # Equivalent normalized W,b; avoids incorrectly weighting normalized targets.
    mean=x.mean(axis=1,keepdims=True)
    std=np.sqrt(x.var(axis=1,keepdims=True)+1e-5)
    z=np.concatenate((x-mean,std),axis=1).transpose(0,2,1)
    return z,mean


def sufficient_statistics(x,y):
    z,mean=design(x)
    target=(y-mean).transpose(0,2,1)
    gram=np.einsum('nci,ncj->cij',z,z,optimize=True)
    rhs=np.einsum('nci,nch->cih',z,target,optimize=True)
    return gram,rhs


def predict(x,weights):
    z,mean=design(x)
    return mean+np.einsum('nci,cih->nhc',z,weights,optimize=True)


def score(p,y):
    r=metrics(p,y);r['low4_mse']=r['mse']*r['low_frequency_error_share']['4']
    r['per_channel_mse']=np.mean((p-y)**2,axis=(0,1)).tolist()
    return r


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    opts=parser.parse_args();out=opts.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    cfg=SimpleNamespace(**json.loads(opts.protocol.read_text())['model_args'])
    train=dataset(cfg,'train',0);val=dataset(cfg,'val',0)
    report=dict(status='fitting',test_accessed=False,train_windows=len(train),validation_windows=len(val),
        input_length=96,horizon=96,lambda_grid=LAMBDAS,
        objective='Mean output-space squared error plus lambda times squared temporal weights; bias unpenalized.',
        selection='Full official validation MSE; select ridge separately for shared and per-channel architectures. No test.',
        warning='Diagnostic linear baselines, not causal evidence that arbitrary nonlinear models lack information.',runs=[])
    atomic_json(out/'report.json',report)
    gram=np.zeros((7,97,97));rhs=np.zeros((7,97,96))
    for start in range(0,len(train),512):
        batch=[train[i] for i in range(start,min(start+512,len(train)))]
        x=np.stack([b[0] for b in batch]).astype('float64');y=np.stack([b[1][-96:] for b in batch]).astype('float64')
        g,r=sufficient_statistics(x,y);gram+=g;rhs+=r
        atomic_json(out/'status.json',dict(status='fitting',windows=min(start+512,len(train)),updated=datetime.now().isoformat()))
    gram/=len(train);rhs/=len(train)
    penalty=np.eye(97);penalty[-1,-1]=0.
    candidates={}
    for shared in (True,False):
        g=gram.mean(0,keepdims=True) if shared else gram
        r=rhs.mean(0,keepdims=True) if shared else rhs
        for lam in LAMBDAS:
            w=np.linalg.solve(g+lam*penalty[None],r)
            if shared:w=np.repeat(w,7,axis=0)
            candidates[('shared' if shared else 'individual')+'_'+str(lam)]=w
    # Accumulate validation predictions in batches; observations never enter fit.
    predictions={key:[] for key in candidates};targets=[]
    for start in range(0,len(val),512):
        batch=[val[i] for i in range(start,min(start+512,len(val)))]
        x=np.stack([b[0] for b in batch]).astype('float64');targets.append(np.stack([b[1][-96:] for b in batch]).astype('float64'))
        for key,w in candidates.items():predictions[key].append(predict(x,w))
        atomic_json(out/'status.json',dict(status='validation',windows=min(start+512,len(val)),updated=datetime.now().isoformat()))
    truth=np.concatenate(targets)
    for key,chunks in predictions.items():
        p=np.concatenate(chunks);r=score(p,truth);assert np.isfinite(r['mse'])
        report['runs'].append(dict(variant=key,validation=r))
        print(key,round(r['mse'],8),flush=True)
    report['selected']={kind:min((r for r in report['runs'] if r['variant'].startswith(kind+'_')),key=lambda r:r['validation']['mse'])['variant'] for kind in ['shared','individual']}
    np.savez_compressed(out/'coefficients.npz',**candidates)
    report['status']='completed';atomic_json(out/'report.json',report)
    atomic_json(out/'status.json',dict(status='completed',updated=datetime.now().isoformat()))
    print('COMPLETE',out,report['selected'],flush=True)


if __name__=='__main__':main()
