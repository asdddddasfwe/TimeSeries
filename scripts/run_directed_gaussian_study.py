"""Small reproducible history-conditioned Gaussian learning experiment."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from layers.DirectedGaussianHead import DirectedGaussianHead
from utils.directed_gaussian import gaussian_nll, predictive_covariance, sample_paths, transition_matrix

SEED=2021
LAGS=(1,2,4,8,12,24)
HORIZON=96
SOURCES=['utils/directed_gaussian.py','layers/DirectedGaussianHead.py','models/OscillatorPDMDirectedGaussian.py','tests/test_pdm_directed_gaussian.py','scripts/run_directed_gaussian_study.py','docs/experiments/PDMDirectedGaussian_protocol_20261007.md']

def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def generate(scenario):
    generator=torch.Generator().manual_seed(SEED)
    context=torch.randn(2560,1,4,generator=generator)
    h=torch.arange(HORIZON).float()
    mean=(.3*context[:,0,0,None]*(2*torch.pi*h/24).sin()+.2*context[:,0,1,None]*(2*torch.pi*h/48).cos())[:,:,None]
    variance=(.25+.15*context[:,0,2].sigmoid())[:,None,None]*(1+.2*h[None,:,None]/95)
    coeff=torch.zeros(2560,HORIZON,1,len(LAGS))
    if scenario=='directed':
        coeff[:,1:,0,0]=.30*context[:,0,0,None].tanh()
        coeff[:,4:,0,2]=.12*context[:,0,1,None].tanh()
    # Independent serial data generator, not sample_paths from the fitted API.
    noise=torch.randn(mean.shape,generator=generator)*variance.sqrt();residual=[]
    for h in range(HORIZON):
        value=noise[:,h]
        for i,k in enumerate(LAGS):
            if h>=k:value=value+coeff[:,h,:,i]*residual[h-k]
        residual.append(value)
    truth=mean+torch.stack(residual,1)
    return context,mean,truth,coeff,variance

@torch.no_grad()
def evaluate(model,context,mean,truth):
    model.eval();total=0.
    for i in range(0,len(context),128):
        b,d=model(context[i:i+128]);total+=gaussian_nll(mean[i:i+128],truth[i:i+128],b,d,model.lags).item()*len(b)
    return total/len(context)

@torch.no_grad()
def summarize(mean,truth,b,d,lags,oracle_b,oracle_d):
    cov=predictive_covariance(mean,b,d,lags)
    oracle=predictive_covariance(mean,oracle_b,oracle_d,LAGS)
    marginal=cov.diagonal(dim1=-2,dim2=-1).transpose(1,2)
    radius=1.6448536269514722*marginal.sqrt()
    dense=transition_matrix(b,lags);true_dense=transition_matrix(oracle_b,LAGS)
    errors=(truth-mean)
    metrics=dict(nll=float(gaussian_nll(mean,truth,b,d,lags)),point_mse=float(errors.square().mean()),point_mae=float(errors.abs().mean()),coverage90=float((errors.abs()<=radius).float().mean()),interval_width90=float((2*radius).mean()),covariance_mse=float((cov-oracle).square().mean()),coefficient_matrix_mse=float((dense-true_dense).square().mean()),innovation_variance_mse=float((d-oracle_d).square().mean()),max_row_l1=float(b.abs().sum(-1).max()) if len(lags) else 0.)
    lag_summary={str(k):dict(estimated_mean=float(dense[:,:,k:,:-k].diagonal(dim1=-2,dim2=-1).mean()),true_mean=float(true_dense[:,:,k:,:-k].diagonal(dim1=-2,dim2=-1).mean()),estimated_rms=float(dense[:,:,k:,:-k].diagonal(dim1=-2,dim2=-1).square().mean().sqrt()),true_rms=float(true_dense[:,:,k:,:-k].diagonal(dim1=-2,dim2=-1).square().mean().sqrt())) for k in LAGS}
    return metrics,lag_summary,cov

def run(scenario,out):
    folder=out/scenario;folder.mkdir()
    context,mean,truth,oracle_b,oracle_d=generate(scenario)
    np.savez_compressed(folder/'data.npz',context=context.numpy(),mean=mean.numpy(),truth=truth.numpy(),oracle_coefficients=oracle_b.numpy(),oracle_variance=oracle_d.numpy(),train_end=2048)
    models={};opts={};initial={};best={};best_epoch={};best_nll={};curves=[];gradient=[]
    for name,lags in [('diagonal',()),('directed',LAGS)]:
        torch.manual_seed(SEED);model=DirectedGaussianHead(4,HORIZON,lags=lags)
        models[name]=model;opts[name]=torch.optim.Adam(model.parameters(),lr=.003)
        initial[name]={k:v.clone() for k,v in model.state_dict().items()};best_nll[name]=float('inf')
    torch.save(initial,folder/'initial.pt')
    for epoch in range(1,26):
        order=torch.randperm(2048,generator=torch.Generator().manual_seed(SEED+epoch))
        for name,model in models.items():
            model.train();total=0.
            for step,index in enumerate(order.split(64)):
                opts[name].zero_grad(set_to_none=True);b,d=model(context[index]);loss=gaussian_nll(mean[index],truth[index],b,d,model.lags)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward()
                for p in model.parameters():
                    if p.grad is None or not torch.isfinite(p.grad).all():raise RuntimeError('Missing/nonfinite gradient')
                if step==0:gradient.append(dict(epoch=epoch,arm=name,variance_grad=float(model.variance.weight.grad.norm()),edge_grad=float(model.edges.weight.grad.norm()) if model.edges is not None else None))
                opts[name].step();total+=loss.item()*len(index)
            nll=evaluate(model,context[2048:],mean[2048:],truth[2048:]);curves.append(dict(epoch=epoch,arm=name,train_nll=total/2048,validation_nll=nll,updates=epoch*32))
            if nll<best_nll[name]:
                best_nll[name]=nll;best_epoch[name]=epoch;best[name]={k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch%5==0:print(scenario,epoch,{name:round(best_nll[name],6) for name in models},flush=True)
    torch.save(dict(initial=initial,best=best,final={name:model.state_dict() for name,model in models.items()},optimizers={name:opt.state_dict() for name,opt in opts.items()},best_epoch=best_epoch),folder/'checkpoints.pt')
    result=dict(runs={},curves=curves,gradient_samples=gradient)
    for name,model in models.items():
        model.load_state_dict(best[name]);model.eval()
        with torch.no_grad():
            b,d=model(context[2048:]);metrics,lag_summary,cov=summarize(mean[2048:],truth[2048:],b,d,model.lags,oracle_b[2048:],oracle_d[2048:])
            paths=sample_paths(mean[2048:2064],b[:16],d[:16],model.lags,samples=256,generator=torch.Generator().manual_seed(SEED+100))
        np.savez_compressed(folder/(name+'_selected.npz'),mean=mean[2048:].numpy(),truth=truth[2048:].numpy(),coefficients=b.numpy(),innovation_variance=d.numpy(),covariance=cov.numpy(),lags=np.array(model.lags,dtype=int),sample_paths=paths.numpy())
        result['runs'][name]=dict(selected_epoch=best_epoch[name],selected_updates=best_epoch[name]*32,final_updates=800,parameter_count=sum(p.numel() for p in model.parameters()),metrics=metrics,lag_summary=lag_summary)
    result['oracle_metrics']=summarize(mean[2048:],truth[2048:],oracle_b[2048:],oracle_d[2048:],LAGS,oracle_b[2048:],oracle_d[2048:])[0]
    write(folder/'report.json',result)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2);out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    old=json.loads((ROOT/'tmp/lag_study_timemix_pdm_innovation_20261006_seed2021_earlystop5/training/protocol.json').read_text())['source_sha256']
    previous=json.loads((ROOT/'tmp/predictable_autocov_study_20261007_seed2021/run1/protocol.json').read_text())['source_sha256']
    frozen={**old,**previous}
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in frozen.items())
    hashes={}
    for p in SOURCES:
        content=(ROOT/p).read_bytes();hashes[p]=hashlib.sha256(content).hexdigest();copy=out/'source'/p;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(content)
    write(out/'protocol.json',dict(seed=SEED,device='cpu',threads=2,horizon=HORIZON,lags=LAGS,epochs=25,batch_size=64,adam_lr=.003,training_windows=2048,validation_windows=512,source_sha256=hashes,started_utc=datetime.now(timezone.utc).isoformat()))
    write(out/'status.json',dict(status='running'))
    results={}
    for scenario in ['independent','directed']:results[scenario]=run(scenario,out)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in frozen.items())
    result=dict(status='completed',scenarios=results,prior_226_sources_unchanged=True,prior_predictable_covariance_sources_unchanged=True,model_normalization_modified=False,ettm1_training_started=False,official_validation_accessed=False,test_accessed=False,finished_utc=datetime.now(timezone.utc).isoformat())
    write(out/'report.json',result);write(out/'status.json',dict(status='completed'))
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
