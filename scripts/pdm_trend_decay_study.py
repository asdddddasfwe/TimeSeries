"""Parameter isolation and bounded diagnostics for PDM trend-decay study."""
import math
import numpy as np
import torch
from utils.timemix_decay_study import parameter_groups as original_groups
from utils.pdm_coarse_ma_diagnostics import SelectedValidationCapture

ARMS = ('baseline_decay', 'trend_no_decay')

def targeted(name):
    return (name.startswith('pdm_blocks.') and
            '.mixing_multi_scale_trend.up_sampling_layers.' in name and
            name.endswith('.weight'))

def parameter_groups(model, arm):
    if arm not in ARMS:
        raise ValueError('Unknown trend-decay arm')
    groups = original_groups(model, 'adam_norm_bias')
    target_ids = {id(p) for n,p in model.named_parameters() if targeted(n)}
    if len(target_ids) != 12:
        raise RuntimeError('Expected twelve trend upsampling weight tensors')
    result = []
    for group in groups:
        for selected in (False, True):
            params = [p for p in group['params'] if (id(p) in target_ids) == selected]
            if params:
                result.append({**group, 'params': params, 'weight_decay':
                    0. if arm == 'trend_no_decay' and selected else group['weight_decay']})
    ids = [id(p) for g in result for p in g['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters()}:
        raise RuntimeError('Optimizer parameter coverage mismatch')
    return result

def gradient_audit(model, decay_by_id):
    result = {}
    for name,p in model.named_parameters():
        if not targeted(name): continue
        if p.grad is None: raise RuntimeError('Missing trend gradient')
        w,g=p.detach().double(),p.grad.detach().double()
        actual_decay = float(decay_by_id[id(p)])
        wn,gn=float(w.norm()),float(g.norm())
        dot=float((w*g).sum())
        result[name] = dict(weight_norm=wn, task_gradient_norm=gn,
            task_gradient_dot_weight=dot,
            task_gradient_weight_cosine=dot/(wn*gn) if wn*gn else None,
            actual_weight_decay=actual_decay, actual_decay_norm=actual_decay*wn,
            reference_decay_norm=1e-4*wn,
            reference_decay_to_task_gradient=1e-4*wn/gn if gn else None)
    return result

class TrendDecayCapture(SelectedValidationCapture):
    """Only detached scalar accumulators survive a batch; no RNG or weight changes."""
    def __enter__(self):
        super().__enter__()
        self.sensitivity={}
        for i,block in enumerate(self.model.pdm_blocks):
            self.handles.append(block.mixing_multi_scale_trend.register_forward_pre_hook(
                lambda module,args,i=i:self.measure_sensitivity(i,module,args[0])))
        self.trend_weight_norms={n:float(p.detach().norm()) for n,p in self.model.named_parameters() if targeted(n)}
        return self

    def measure_sensitivity(self, layer, mixer, values):
        # Fixed zero-mean temporal direction at length12, RMS=1e-3; no random draws.
        coarse=values[-1]
        direction=torch.linspace(-1,1,coarse.shape[-1],device=coarse.device,dtype=coarse.dtype)
        direction=direction/direction.square().mean().sqrt()*1e-3
        changed=list(values);changed[-1]=coarse+direction
        with torch.no_grad():
            base=mixer.forward(values)
            perturbed=mixer.forward(changed)
            for scale,(a,b) in enumerate(zip(base,perturbed)):
                d=(b-a).detach().double()
                stat=self.sensitivity.setdefault(f'layer{layer}_scale{scale}',dict(square=0.,elements=0))
                stat['square']+=float(d.square().sum());stat['elements']+=d.numel()

    def report(self,prediction,truth):
        report=super().report(prediction,truth)
        report['trend_weight_norms']=self.trend_weight_norms
        report['trend_path_directional_gain']={k:(v['square']/v['elements'])**.5/1e-3 for k,v in self.sensitivity.items()}
        report['sensitivity_definition']='Finite difference gain for fixed RMS .001 zero-mean ramp added to coarsest trend features; no fitting; scales0/1/2 are96/48/24. Scale3 is input identity.'
        p,y=prediction.astype('float64'),truth.astype('float64')
        err=p-y;centered=err-err.mean(1,keepdims=True)
        z=np.asarray(self.components[...,3],dtype='float64')
        report['per_channel_waveform_mse']=np.mean(centered**2,axis=(0,1)).tolist()
        report['coarse_residual_inner_product']=np.mean(z*(z-err),axis=(0,1)).tolist()
        return report
