"""Four labelled amplitude-weighted PSD kernels on final four-point patches.
No channel averaging before squared loss. Diagonals retain absolute coordinates.
"""
import math
import torch

def patch_values(sequence):
    if sequence.ndim != 3 or sequence.shape[1] % 4 or sequence.shape[1] < 8:
        raise ValueError('Expected nonempty B,H,C with H divisible by4 and >=8')
    if not sequence.is_floating_point() or not torch.isfinite(sequence).all():
        raise ValueError('Finite floating point sequence required')
    b,h,c=sequence.shape
    return sequence.double().transpose(1,2).reshape(b*c,h//4,4)

def positive_map(q,scale=1.):
    if not math.isfinite(scale) or scale<=0:raise ValueError('Positive fixed scale required')
    z=q/scale
    return torch.logaddexp(z,torch.zeros_like(z))

def base_and_amplitude(sequence,sigma=1.,scale=1.):
    if not math.isfinite(sigma) or sigma<=0:raise ValueError('Positive fixed sigma required')
    q=patch_values(sequence);h=positive_map(q,scale)
    base=torch.exp(-.5*((q[:,:,None,:]-q[:,None,:,:])/sigma).square().sum(-1))
    return base,h

def kernel_matrices(sequence,sigma=1.,scale=1.):
    base,h=base_and_amplitude(sequence,sigma,scale)
    # [B*C,4,N,N], all4 phase identities retained, no diagonal replacement.
    return torch.stack([base*h[:,:,r,None]*h[:,None,:,r] for r in range(4)],dim=1)

def parts(prediction,target,sigma=1.,scale=1.,metadata=False):
    if prediction.shape!=target.shape or prediction.device!=target.device:raise ValueError('Matching sequences required')
    kp,hp=base_and_amplitude(prediction,sigma,scale)
    with torch.no_grad():ky,hy=base_and_amplitude(target.detach(),sigma,scale)
    n=kp.shape[-1];mask=~torch.eye(n,device=kp.device,dtype=torch.bool);off=[];diagonal=[];rho=[];ps=[];ts=[]
    for r in range(4):
        p=kp*hp[:,:,r,None]*hp[:,None,:,r];y=ky*hy[:,:,r,None]*hy[:,None,:,r]
        x,t=p[:,mask],y[:,mask];off.append((x-t).square().mean(-1));diagonal.append((hp[:,:,r].square()-hy[:,:,r].square()).square().mean(-1))
        if metadata:
            with torch.no_grad():
                xc=x-x.mean(-1,keepdim=True);tc=t-t.mean(-1,keepdim=True);px=xc.square().mean(-1).sqrt();tx=tc.square().mean(-1).sqrt();rho.append((xc*tc).mean(-1)/(px*tx).clamp_min(1e-30));ps.append(px);ts.append(tx)
    off=torch.stack(off,-1);diagonal=torch.stack(diagonal,-1)
    v=dict(offdiag_mse=off.mean().to(prediction.dtype),diag_mse=diagonal.mean().to(prediction.dtype))
    if metadata:
        b,_,c=prediction.shape
        v['meta']=dict(per_variable_phase_offdiag_MSE=off.detach().reshape(b,c,4).mean(0).cpu().tolist(),per_variable_phase_diag_MSE=diagonal.detach().reshape(b,c,4).mean(0).cpu().tolist(),offdiag_correlation=torch.stack(rho,-1).reshape(b,c,4).mean(0).cpu().tolist(),prediction_offdiag_std=torch.stack(ps,-1).reshape(b,c,4).mean(0).cpu().tolist(),target_offdiag_std=torch.stack(ts,-1).reshape(b,c,4).mean(0).cpu().tolist(),positive_map_zero_fraction=float((hp==0).double().mean()),positive_map_below_1e_12_fraction=float((hp<1e-12).double().mean()),no_unit_diagonal=True)
    return v

def training_terms(prediction,target):
    k=parts(prediction,target)
    return dict(point_mse=(prediction-target.detach()).square().mean(),amp_kernel_offdiag_mse=k['offdiag_mse'],amp_kernel_diag_mse=k['diag_mse'])

WEIGHTS={'point_mse':1.,'amp_kernel_offdiag_mse':.1,'amp_kernel_diag_mse':.1}
def objective(prediction,target):
    terms=training_terms(prediction,target)
    return sum(WEIGHTS[n]*v for n,v in terms.items())


def gradient_summary(terms,model):
    parameters=list(model.parameters());vectors={};out={}
    for name,term in terms.items():
        grads=torch.autograd.grad(term,parameters,retain_graph=True,allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in grads)
        v=torch.cat([(torch.zeros_like(p) if g is None else g).detach().double().reshape(-1) for p,g in zip(parameters,grads)])
        vectors[name]=v;norm=float(v.norm());out[name]=dict(raw_norm=norm,weighted_norm=WEIGHTS[name]*norm,parameters_with_gradient=sum(g is not None for g in grads))
    names=list(terms);angles={}
    for i,name in enumerate(names):
        for other in names[i+1:]:
            a,b=vectors[name],vectors[other];den=a.norm()*b.norm();angles[name+'__'+other]=float(a.dot(b)/den) if float(den)>0 else None
    return dict(norms=out,cosines=angles,weights=WEIGHTS,raw_not_Adam_update=True)
