"""D's four unit-weight losses; only final-output RBF construction changes.
Independent-head RBF supervision remains original. No diagonal-MSE term.
"""
import torch
from models.MAFirstKernelTransferAblation import kernel_supervision
from utils.local_offdiagonal_gram_cs import parts as local_cs
from utils.amplitude_weighted_patch_kernel_20261025 import base_and_amplitude
TERM_NAMES=('point_mse','head_gram_mse','head_local_cs','output_local_cs')
WEIGHTS={name:1. for name in TERM_NAMES}

def output_parts(prediction,target,kernel_builder=None):
    if prediction.shape!=target.shape:raise ValueError('Matching B,H,C sequences required')
    if kernel_builder is None:
        bp,hp=base_and_amplitude(prediction)
        with torch.no_grad():bt,ht=base_and_amplitude(target.detach())
        pairs=((bp*hp[:,:,r,None]*hp[:,None,:,r],bt*ht[:,:,r,None]*ht[:,None,:,r]) for r in range(4))
    else:
        p=kernel_builder(prediction);t=kernel_builder(target.detach()).detach()
        if p.shape!=t.shape or p.ndim!=4 or p.shape[1]!=4:raise ValueError('Four labelled channels required')
        pairs=zip(p.unbind(1),t.unbind(1))
    losses=[];metadata=[]
    for p,t in pairs:
        loss,meta=local_cs(p,t);losses.append(loss);metadata.append(meta)
    loss=torch.stack(losses).mean().to(prediction.dtype)
    return dict(local_cs=loss,phase_local_cs=torch.stack(losses).detach().cpu().tolist(),phase_meta=metadata,diagonal_excluded=True,extra_diagonal_MSE=False)

def training_terms(prediction,target,head,kernel_builder=None):
    target=target.detach();head_mse,gold=kernel_supervision(head,target)
    head_cs,head_meta=local_cs(head,gold);out=output_parts(prediction,target,kernel_builder)
    terms=dict(point_mse=(prediction-target).square().mean(),head_gram_mse=head_mse,head_local_cs=head_cs,output_local_cs=out['local_cs'])
    assert tuple(terms)==TERM_NAMES
    return terms,dict(output=out,head_meta=head_meta)

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
