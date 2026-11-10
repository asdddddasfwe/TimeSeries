"""Two fixed-scale Huber losses on squared patch distances. No CS or MMD in sum."""
import torch
from torch.nn import functional as F
TERM_NAMES=('point_mse','head_distance_huber','output_distance_huber')
HEAD_MIN=1e-12

def squared_patch_distances(sequence,detach=False):
    if sequence.ndim!=3 or sequence.shape[1]%4:raise ValueError('B,H,C and H divisible by4 required')
    a=sequence.detach() if detach else sequence
    b,h,c=a.shape;q=a.double().transpose(1,2).reshape(b*c,h//4,4)
    return (q.unsqueeze(-2)-q.unsqueeze(-3)).square().sum(-1)

def head_distances(head):
    if not head.is_floating_point() or not torch.isfinite(head).all() or head.min()<0 or head.max()>1:raise ValueError('finite similarities in[0,1] required')
    return -2.*head.double().clamp_min(HEAD_MIN).log()

def distance_parts(prediction,target,head,scales):
    if prediction.shape!=target.shape:raise ValueError('Matching prediction/target shapes required')
    b,h,c=prediction.shape;n=h//4
    if head.shape!=(b*c,n,n):raise ValueError('Head shape must be B*C,N,N')
    scale=torch.as_tensor(scales,device=prediction.device,dtype=torch.double)
    if scale.shape!=(c,) or not torch.isfinite(scale).all() or (scale<=0).any():raise ValueError('positive fixed scale per channel required')
    scale=scale.detach().repeat(b).reshape(-1,1,1)
    gold=squared_patch_distances(target,detach=True);hd=head_distances(head);od=squared_patch_distances(prediction)
    mask=~torch.eye(n,device=prediction.device,dtype=torch.bool)
    eh=(hd-gold)/scale;eo=(od-gold)/scale
    hl=F.huber_loss(eh[...,mask],torch.zeros_like(eh[...,mask]),delta=1.,reduction='mean').to(prediction.dtype)
    ol=F.huber_loss(eo[...,mask],torch.zeros_like(eo[...,mask]),delta=1.,reduction='mean').to(prediction.dtype)
    return dict(head_huber=hl,output_huber=ol,head_distance=hd,output_distance=od,target_distance=gold,
        head_distance_mse=(hd-gold).square()[...,mask].mean(),output_distance_mse=(od-gold).square()[...,mask].mean(),
        head_clamp_fraction=(head[...,mask]<=HEAD_MIN).double().mean(),scales=scale)
