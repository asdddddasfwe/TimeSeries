"""Losses on the actual final [B,H,C] forecast, four-point RBF Gram, no time W.
No additional ordinary MMD. Original independent-head losses remain untouched.
"""
import torch
from models.MAFirstKernelTransferAblation import value_gram,kernel_supervision
from utils.local_offdiagonal_gram_cs import parts as local_cs
from utils.time_position_patch_mmd import parts as time_mmd
TERM_NAMES=('point_mse','relation','head_gram_mse','head_local_cs','output_local_cs')

def output_parts(prediction,target):
    if prediction.ndim!=3 or prediction.shape!=target.shape or prediction.shape[1]%4 or prediction.shape[1]<12:
        raise ValueError('Matching B,H,C with H divisible by4 and at least12 required')
    if prediction.device!=target.device or not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError('Matching floating inputs on same device required')
    b,h,c=prediction.shape
    patches=prediction.transpose(1,2).reshape(b*c,h//4,4)
    gold_patches=target.detach().transpose(1,2).reshape(b*c,h//4,4)
    predicted=value_gram(patches);gold=value_gram(gold_patches)
    mask=~torch.eye(h//4,dtype=torch.bool,device=prediction.device)
    mse=(predicted.double()-gold.double()).square()[...,mask].mean().to(prediction.dtype)
    structure,meta=local_cs(predicted,gold)
    return dict(predicted=predicted,target=gold,mse=mse,local_cs=structure,meta=meta)

def training_terms(prediction,target,head,relation_mode):
    if relation_mode not in ('time_mmd','output_gram_mse'):raise ValueError('Unknown relation mode')
    head_mse,gold=kernel_supervision(head,target)
    head_cs,head_meta=local_cs(head,gold)
    out=output_parts(prediction,target)
    assert torch.equal(gold,out['target']) and not gold.requires_grad
    # In B the original timeMMD is a detached diagnostic, not an applied term.
    time_input=prediction if relation_mode=='time_mmd' else prediction.detach()
    tm,tm_meta=time_mmd(time_input,target)
    terms=dict(point_mse=(prediction-target).square().mean(),relation=tm if relation_mode=='time_mmd' else out['mse'],head_gram_mse=head_mse,head_local_cs=head_cs,output_local_cs=out['local_cs'])
    assert tuple(terms)==TERM_NAMES
    return terms,dict(output=out,head_meta=head_meta,time_mmd=tm,time_meta=tm_meta)

def gradient_summary(terms,named):
    """All five raw parameter gradients and ten pair angles; not Adam updates."""
    grads={key:torch.autograd.grad(value,[p for _,p in named],retain_graph=True,allow_unused=True) for key,value in terms.items()}
    result={}
    scopes=[('all',lambda n:True),('coarse_readout',lambda n:n.startswith('seasonals.3.decoder.readout.')),('history_encoder',lambda n:'.blocks.' in n or n.startswith('pdm_blocks.')),('state_core',lambda n:'.decoder.' in n and '.readout.' not in n)]
    for scope,include in scopes:
        norms={};pairs={}
        for key,row in grads.items():
            vals=[g.double() for (n,p),g in zip(named,row) if include(n) and g is not None]
            assert all(torch.isfinite(g).all() for g in vals)
            norms[key]=sum(float(g.square().sum()) for g in vals)**.5
        keys=list(terms)
        for i,a in enumerate(keys):
            for b in keys[i+1:]:
                dot=sum(float((ga.double()*gb.double()).sum()) for (n,p),ga,gb in zip(named,grads[a],grads[b]) if include(n) and ga is not None and gb is not None)
                den=norms[a]*norms[b];pairs[a+'__'+b]=dict(cosine=dot/max(den,1e-30),defined=den>0)
        result[scope]=dict(norms=norms,pairs=pairs)
    return result
