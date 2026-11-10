"""Evaluate an immutable 96->96 candidate; test requires prior recorded freeze."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from data_provider.data_loader import Dataset_ETT_minute
from scripts.diagnose_timemix_linear_reference import score
from utils.logger import experiment_logging,start_experiment,log_metrics


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def verify_artifact(artifact,split,limit=0):
    artifact=Path(artifact)
    if split not in ('val','test') or limit<0:raise ValueError('Invalid split or limit')
    manifest=json.loads((artifact/'manifest.json').read_text())
    if split=='test':
        if limit:raise ValueError('Final test must use every official test window')
        if manifest['selection_status']!='frozen_for_test':
            raise ValueError('Record final validation-based selection before opening test data')
        freeze=json.loads((artifact/'freeze.json').read_text())
        if not freeze.get('reason') or not freeze.get('validation_evidence'):
            raise ValueError('Freeze must record selection reason and validation evidence')
        for key in ('checkpoint_sha256','model_config_sha256'):
            if freeze[key]!=manifest[key]:raise ValueError('Freeze identity mismatch: '+key)
    for name,key in [('checkpoint.pth','checkpoint_sha256'),('model_config.json','model_config_sha256')]:
        if sha(artifact/name)!=manifest[key]:raise ValueError('Artifact changed: '+name)
    runtime_path=artifact/'runtime_manifest.json'
    if runtime_path.exists():
        runtime=json.loads(runtime_path.read_text())
        if runtime['original_manifest_sha256']!=sha(artifact/'manifest.json'):
            raise ValueError('Historical manifest changed')
        for key in ('checkpoint_sha256','model_config_sha256'):
            if runtime[key]!=manifest[key]:raise ValueError('Runtime lineage mismatch: '+key)
        sources=runtime['source_sha256']
        if not sources:raise ValueError('Empty runtime source manifest')
        for name,digest in sources.items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT.resolve()) or sha(path)!=digest:
                raise ValueError('Runtime source differs from candidate: '+name)
    else:
        for field in ('adapter_source_sha256','neural_training_source_sha256'):
            for name,digest in manifest.get(field,{}).items():
                if sha(ROOT/name)!=digest:raise ValueError('Source differs from candidate: '+name)
    cfg=json.loads((artifact/'model_config.json').read_text())
    if (cfg['seq_len'],cfg['pred_len'],cfg['features'],cfg['data'])!=(96,96,'M','ETTm1'):
        raise ValueError('This goal evaluation requires ETTm1 M 96->96')
    if cfg.get('inverse',False):raise ValueError('Goal metrics use dataset-standardized units')
    if cfg['model']!=manifest['kind']:raise ValueError('Model identity mismatch')
    return manifest,cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--limit',type=int,default=0,help='Validation smoke only; never final test')
    parser.add_argument('--verify-only',action='store_true')
    opts=parser.parse_args();artifact=opts.artifact.resolve()
    manifest,config=verify_artifact(artifact,opts.split,opts.limit)
    if opts.verify_only:print('VERIFIED',artifact,opts.split);return
    if opts.output is None:raise ValueError('--output is required for evaluation')
    lock=(artifact/(opts.split+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=artifact/'test_result.json'
    if opts.split=='test' and receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['checkpoint_sha256']==manifest['checkpoint_sha256']:
            print('TEST ALREADY RECORDED',saved['report']);return
        raise ValueError('A different checkpoint already has a test receipt')
    out=opts.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'pid.txt').write_text(str(os.getpid())+'\n');torch.set_num_threads(2)
    cfg=SimpleNamespace(**config);cfg.is_training=0;cfg.use_gpu=opts.device=='cuda'
    cfg.log_dir=str(out/'logs');cfg.tensorboard_dir=str(out/'runs');cfg.model_id=out.name
    report=dict(status='running',split=opts.split,smoke=bool(opts.limit),artifact=str(artifact),
                checkpoint_sha256=manifest['checkpoint_sha256'],model_config_sha256=manifest['model_config_sha256'],
                device=opts.device,source_hashes_verified=True,started=datetime.now().isoformat())
    atomic_json(out/'report.json',report)
    try:
        with experiment_logging(cfg):
            start_experiment(cfg)
            model=importlib.import_module('models.'+cfg.model).Model(cfg).to(opts.device)
            model.load_state_dict(torch.load(artifact/'checkpoint.pth',map_location=opts.device,weights_only=True),strict=True)
            model.eval()
            data=Dataset_ETT_minute(cfg,cfg.root_path,flag=opts.split,size=[96,0,96],features='M',data_path=cfg.data_path,timeenc=1,freq=cfg.freq)
            official_windows=len(data)
            if opts.limit:data=Subset(data,np.unique(np.linspace(0,len(data)-1,min(opts.limit,len(data)),dtype=int)).tolist())
            predictions=[];targets=[]
            with torch.no_grad():
                for step,(x,y,_,_) in enumerate(DataLoader(data,batch_size=32,shuffle=False,num_workers=0)):
                    pred=model(x.float().to(opts.device)).cpu()
                    if pred.shape!=y[:,-96:].shape or not torch.isfinite(pred).all():raise RuntimeError('Invalid forecast')
                    predictions.append(pred.numpy());targets.append(y[:,-96:].numpy())
                    if step%25==0:atomic_json(out/'status.json',dict(status='evaluating',split=opts.split,step=step+1,updated=datetime.now().isoformat()))
            prediction=np.concatenate(predictions);truth=np.concatenate(targets)
            if opts.split=='test' and len(prediction)!=official_windows:raise RuntimeError('Incomplete test coverage')
            metric=score(prediction.astype('float64'),truth.astype('float64'))
            np.savez_compressed(out/'predictions.npz',prediction=prediction,truth=truth)
            log_metrics({k:metric[k] for k in ('mse','mae','low4_mse','centered_amplitude_ratio')},step=0,prefix=opts.split)
            report.update(status='completed',windows=len(data),official_windows=official_windows,metrics=metric,finished=datetime.now().isoformat())
            if sha(artifact/'checkpoint.pth')!=manifest['checkpoint_sha256']:raise RuntimeError('Checkpoint changed during evaluation')
            atomic_json(out/'report.json',report);atomic_json(out/'status.json',dict(status='completed',updated=datetime.now().isoformat()))
            if opts.split=='test':atomic_json(receipt,dict(checkpoint_sha256=manifest['checkpoint_sha256'],report=str(out/'report.json')))
            print('COMPLETE',out,'MSE',metric['mse'],flush=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));atomic_json(out/'report.json',report)
        atomic_json(out/'status.json',dict(status='failed',error=repr(exc),updated=datetime.now().isoformat()));raise


if __name__=='__main__':main()
