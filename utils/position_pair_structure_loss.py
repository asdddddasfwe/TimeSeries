"""Position-preserving past/future and future/future product loss.

Statistics of deterministic paths, not a conditional covariance estimator.
Only the loss pools/centers; no model normalization or forward is changed.
"""
import torch
import torch.nn.functional as F
FUTURE_LAGS=(0,4,8,12,24,48)
BOUNDARY_LAGS=(4,8,12,24,48,96)

def position_pair_structure_loss(history,prediction,target,channel_variance,block=1,
                                 future_lags=FUTURE_LAGS,boundary_lags=BOUNDARY_LAGS):
    if prediction.ndim!=3 or target.shape!=prediction.shape or history.ndim!=3:
        raise ValueError('Expected [B,T,C] paths')
    if history.shape[0]!=prediction.shape[0] or history.shape[2]!=prediction.shape[2]:raise ValueError('Mismatched channels/batch')
    if history.device!=prediction.device or target.device!=prediction.device:raise ValueError('Device mismatch')
    if not prediction.is_floating_point():raise ValueError('Floating predictions required')
    length,horizon=history.shape[1],prediction.shape[1]
    if type(block)!=int or block<1 or length%block or horizon%block:raise ValueError('Block must divide both lengths')
    for lags,minimum,maximum in [(future_lags,0,horizon-1),(boundary_lags,1,min(length,horizon))]:
        if not lags or len(set(lags))!=len(lags) or any(type(k)!=int or k<minimum or k>maximum or k%block for k in lags):raise ValueError('Invalid physical lags')
    dtype=torch.float32 if prediction.dtype in (torch.float16,torch.bfloat16) else prediction.dtype
    p=prediction.to(dtype);x=history.detach().to(dtype);y=target.detach().to(dtype)
    variance=torch.as_tensor(channel_variance,device=p.device,dtype=dtype).detach()
    if variance.ndim>1 or variance.numel() not in (1,p.shape[2]) or not bool(torch.isfinite(variance).all() and (variance>0).all()):raise ValueError('Positive finite train variance required')
    scale=variance.clamp_min(1e-12)
    def pool(z):return z.reshape(z.shape[0],z.shape[1]//block,block,z.shape[2]).mean(2)
    x,p,y=pool(x),pool(p),pool(y)
    x=x-x.mean(1,keepdim=True);p=p-p.mean(1,keepdim=True);y=y-y.mean(1,keepdim=True)
    h,l=p.shape[1],x.shape[1]
    ff=[];hf=[]
    for physical in future_lags:
        k=physical//block
        error=(p[:,:h-k]*p[:,k:]-y[:,:h-k]*y[:,k:])/scale
        ff.append(F.smooth_l1_loss(error,torch.zeros_like(error),beta=1.,reduction='none'))
    for physical in boundary_lags:
        k=physical//block
        error=x[:,l-k:]*(p[:,:k]-y[:,:k])/scale
        hf.append(F.smooth_l1_loss(error,torch.zeros_like(error),beta=1.,reduction='none'))
    # Average individual pair penalties, not a penalty on the averaged product.
    return dict(covariance=torch.cat(ff,dim=1).mean(),anchor=torch.cat(hf,dim=1).mean())
