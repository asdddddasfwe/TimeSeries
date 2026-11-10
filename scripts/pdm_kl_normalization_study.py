"""Train-only fixed calibration and loss-gradient diagnostics; no optimizer updates."""
import math
import numpy as np
import torch
from utils.joint_gaussian_autocorr_loss import joint_gaussian_autocorr_kl_loss
from utils.joint_gaussian_autocov_loss import joint_gaussian_autocov_kl_loss, DEFAULT_OFFSETS
FUNCTIONS={'correlation_kl':joint_gaussian_autocorr_kl_loss,'covariance_kl':joint_gaussian_autocov_kl_loss}

def auxiliary(history, prediction, target, kind, config):
    return FUNCTIONS[kind](history,prediction,target,offsets=config['offsets'],variance_floor=config['variance_floor'],jitter=config['jitter'])

def pair_stats(first,second):
    a=sum(float(g.detach().double().square().sum()) for g in first)
    b=sum(float(g.detach().double().square().sum()) for g in second)
    ab=sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(first,second))
    if not all(math.isfinite(x) for x in [a,b,ab]):raise RuntimeError('Nonfinite diagnostic gradient')
    return dict(mse_squared_norm=a,kl_squared_norm=b,inner_product=ab,
                cosine=ab/math.sqrt(a*b) if a*b else None,norm_ratio=math.sqrt(b/a) if a else None)

def calibrated_weights(rows,ratio):
    a=sum(row['mse_squared_norm'] for row in rows)
    if not math.isfinite(a) or a<=0:raise RuntimeError('Degenerate calibration MSE gradient')
    result={'mse':0.0}
    for kind in FUNCTIONS:
        b=sum(row[kind]['kl_squared_norm'] for row in rows)
        if not math.isfinite(b) or b<=0:raise RuntimeError('Degenerate calibration KL gradient')
        result[kind]=ratio*math.sqrt(a/b)
    return result

def calibrate(model,train,config,seed,device,ratio=.1):
    if not 0<ratio<=1:raise ValueError('Invalid calibration ratio')
    ids=np.unique(np.linspace(0,len(train)-1,min(128,len(train)),dtype=int))
    rows=[];mode=model.training;before={k:v.detach().clone() for k,v in model.state_dict().items()}
    devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.train()
            for i in range(0,len(ids),32):
                ix=ids[i:i+32];torch.manual_seed(seed+910000+i)
                x=torch.as_tensor(np.stack([train[int(j)][0] for j in ix]),dtype=torch.float32,device=device)
                y=torch.as_tensor(np.stack([train[int(j)][1][-96:] for j in ix]),dtype=torch.float32,device=device)
                p=model(x);mse=(p-y.detach()).square().mean();params=list(model.parameters())
                gm=torch.autograd.grad(mse,params,retain_graph=True)
                row=dict(ids=ix.tolist(),mse_squared_norm=sum(float(g.double().square().sum()) for g in gm))
                for kind in FUNCTIONS:
                    value=auxiliary(x,p,y,kind,config);gk=torch.autograd.grad(value,params,retain_graph=True)
                    row[kind]=pair_stats(gm,gk)
                    del gk,value
                rows.append(row)
                del gm,p,mse,params
    finally:model.train(mode)
    if not all(torch.equal(v,before[k]) for k,v in model.state_dict().items()):raise RuntimeError('Calibration changed model state')
    if any(p.grad is not None for p in model.parameters()):raise RuntimeError('Calibration accumulated gradients')
    return dict(seed=seed,training_windows=len(train),sample_ids=ids.tolist(),target_gradient_ratio=ratio,
                method='ratio * sqrt(sum batch MSE parameter-gradient squared norms / sum batch KL squared norms)',
                mode='train',rng_restored=True,model_unchanged=True,weights=calibrated_weights(rows,ratio),rows=rows)

def gradient_diagnostic(model,history,prediction,target,config):
    params=list(model.parameters());mse=(prediction-target.detach()).square().mean()
    gm=torch.autograd.grad(mse,params,retain_graph=True)
    om=2*(prediction.detach()-target.detach())/prediction.numel()
    centered=prediction.detach()-prediction.detach().mean(1,keepdim=True)
    result={}
    for kind in FUNCTIONS:
        value=auxiliary(history,prediction,target,kind,config)
        gk=torch.autograd.grad(value,params,retain_graph=True)
        op=torch.autograd.grad(value,prediction,retain_graph=True)[0].detach()
        stats=pair_stats(gm,gk);weight=config['kl_weight_by_arm'][kind]
        stats.update(weight=weight,weighted_norm_ratio=weight*stats['norm_ratio'] if stats['norm_ratio'] is not None else None)
        def output(a,b,d):
            aa=a.double().square().sum((0,1));bb=b.double().square().sum((0,1));dot=(a.double()*b.double()).sum((0,1))
            return dict(cosine=(dot/(aa*bb).sqrt().clamp_min(1e-30)).cpu().tolist(),
                mse_amplitude_derivative=(a*d).sum((0,1)).cpu().tolist(),
                kl_amplitude_derivative=(b*d).sum((0,1)).cpu().tolist())
        stats['output_per_channel']=output(om,op,centered)
        stats['output_horizon_quarters']=[output(om[:,i:i+24],op[:,i:i+24],centered[:,i:i+24]) for i in range(0,96,24)]
        if not torch.isfinite(op).all():raise RuntimeError('Nonfinite output diagnostic gradient')
        result[kind]=stats
        del value,gk,op
    return result
