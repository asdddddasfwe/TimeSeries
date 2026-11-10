"""Same-forward original/correction trend energies and gradients."""
import math
import torch


class FeatureStatistics:
    def __init__(self,model):
        self.totals,self.pending,self.handles={},{},[]
        for i,head in enumerate(model.feature_trend_heads):
            key=str(i)
            if hasattr(head,'coefficient_net'):
                def base_hook(module,args,output,key=key):
                    self.pending.setdefault(key,{})['base']=output.detach().squeeze(-1)
                def coefficient_hook(module,args,output,key=key):
                    self.pending.setdefault(key,{})['coefficients']=output.detach()
                self.handles += [head.readout.register_forward_hook(base_hook),head.coefficient_net.register_forward_hook(coefficient_hook)]
            def head_hook(module,args,output,key=key):
                with torch.no_grad():
                    x,y=args[0].detach().double(),output.detach().double()
                    active=hasattr(module,'coefficient_net')
                    cache=self.pending.pop(key,{})
                    base=cache['base'].double() if active else y
                    coefficients=cache['coefficients'].double() if active else y.new_zeros(y.shape[0],8)
                    correction=coefficients@module.future_basis.detach().double().T if active else torch.zeros_like(y)
                    assert all(torch.isfinite(v).all() for v in (x,y,base,coefficients,correction))
                    d=y.diff(dim=-1);boundary=torch.arange(d.shape[-1],device=d.device)%4==3
                    row=self.totals.setdefault(key,dict(input_count=0,output_count=0,coefficient_count=0,input_square=0.,prediction_square=0.,prediction_sum=0.,base_square=0.,correction_square=0.,coefficient_square=0.,base_correction_dot=0.,boundary_square=0.,boundary_count=0,interior_square=0.,interior_count=0,reconstruction_max_abs=0.,correction_active=active,coefficient_output_weight_norm=0.))
                    row['input_count']+=x.numel();row['output_count']+=y.numel();row['coefficient_count']+=coefficients.numel()
                    for label,v in [('input_square',x.square().sum()),('prediction_square',y.square().sum()),('prediction_sum',y.sum()),('base_square',base.square().sum()),('correction_square',correction.square().sum()),('coefficient_square',coefficients.square().sum()),('base_correction_dot',(base*correction).sum())]:row[label]+=float(v)
                    for label,mask in [('boundary',boundary),('interior',~boundary)]:
                        row[label+'_square']+=float(d[...,mask].square().sum());row[label+'_count']+=d[...,mask].numel()
                    row['reconstruction_max_abs']=max(row['reconstruction_max_abs'],float((base+correction-y).abs().max()))
                    row['coefficient_output_weight_norm']=float(module.coefficient_net[2].weight.detach().double().norm()) if active else 0.
            self.handles.append(head.register_forward_hook(head_hook))

    def close(self):
        for h in self.handles:h.remove()
        self.pending.clear()

    def summary(self):
        result={}
        for k,v in self.totals.items():
            n=v['output_count'];base=math.sqrt(v['base_square']/n);corr=math.sqrt(v['correction_square']/n)
            result[k]=dict(input_count=v['input_count'],output_count=n,coefficient_count=v['coefficient_count'],
                input_rms=math.sqrt(v['input_square']/v['input_count']),prediction_rms=math.sqrt(v['prediction_square']/n),prediction_mean=v['prediction_sum']/n,
                base_prediction_rms=base,correction_rms=corr,coefficient_rms=math.sqrt(v['coefficient_square']/v['coefficient_count']),
                correction_to_base_rms=corr/max(base,1e-30),base_correction_cosine=v['base_correction_dot']/max(math.sqrt(v['base_square']*v['correction_square']),1e-30),
                boundary_increment_rms=math.sqrt(v['boundary_square']/v['boundary_count']),interior_increment_rms=math.sqrt(v['interior_square']/v['interior_count']),
                correction_active=v['correction_active'],reconstruction_max_abs=v['reconstruction_max_abs'],coefficient_output_weight_norm=v['coefficient_output_weight_norm'])
        return result


def gradient_probe(model, mse, distance, lowfreq=None):
    named = list(model.named_parameters())
    gm = torch.autograd.grad(mse, [p for _, p in named], retain_graph=True, allow_unused=True)
    gd = torch.autograd.grad(distance, [p for _, p in named], retain_graph=True, allow_unused=True)
    gl = torch.autograd.grad(lowfreq,[p for _,p in named],retain_graph=True,allow_unused=True) if lowfreq is not None else [None]*len(named)
    scopes = {'all': lambda n: True, 'trend_heads': lambda n: n.startswith('feature_trend_heads.'), 'trend_temporal': lambda n: n.startswith('feature_trend_heads.') and '.temporal.' in n, 'trend_readout': lambda n: n.startswith('feature_trend_heads.') and '.readout.' in n, 'trend_correction': lambda n: '.coefficient_net.' in n, 'correction_hidden': lambda n: '.coefficient_net.0.' in n, 'correction_output': lambda n: '.coefficient_net.2.' in n, 'readout_hidden': lambda n: '.readout.input_projection.' in n,
        'state_patch_heads': lambda n: '.readout.output_projection.' in n,
        'trend_condition': lambda n: '.trend_condition.' in n,
        'fine_trend_head': lambda n: n.startswith('feature_trend_heads.0.'),
        'future_projection': lambda n: '.temporal_projection.' in n,
        'history_attention': lambda n: '.blocks.' in n,
        'pdm': lambda n: n.startswith('pdm_blocks.')}
    for scale in range(4):
        scopes['trend_scale'+str(scale)] = lambda n, s=scale: n.startswith(f'feature_trend_heads.{s}.')
    for scale in range(4):
        scopes['correction_scale'+str(scale)] = lambda n, s=scale: n.startswith(f'feature_trend_heads.{s}.coefficient_net.')
    result = {}
    for label, include in scopes.items():
        mm = dd = dot = ll = ld = lm = 0.; count = 0
        for (name, _), a, b, l in zip(named, gm, gd, gl):
            if not include(name):
                continue
            count += 1
            if a is not None:
                assert torch.isfinite(a).all()
                mm += float(a.detach().double().square().sum())
            if b is not None:
                assert torch.isfinite(b).all()
                dd += float(b.detach().double().square().sum())
            if a is not None and b is not None:
                dot += float((a.detach().double()*b.detach().double()).sum())
            if l is not None:
                assert torch.isfinite(l).all()
                ll+=float(l.detach().double().square().sum())
                if a is not None:lm+=float((l.detach().double()*a.detach().double()).sum())
                if b is not None:ld+=float((l.detach().double()*b.detach().double()).sum())
        result[label] = dict(parameter_tensors=count, mse_norm=math.sqrt(mm), d_norm=math.sqrt(dd),
            weighted_ratio=math.sqrt(dd)/max(math.sqrt(mm), 1e-30), cosine=dot/max(math.sqrt(mm*dd), 1e-30), lowfreq_norm=math.sqrt(ll), lowfreq_mse_ratio=math.sqrt(ll)/max(math.sqrt(mm),1e-30), lowfreq_mse_cosine=lm/max(math.sqrt(ll*mm),1e-30),lowfreq_d_cosine=ld/max(math.sqrt(ll*dd),1e-30))
    return result
