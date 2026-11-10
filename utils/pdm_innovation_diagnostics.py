"""Same-forward filter evidence alongside unchanged PDM diagnostics."""
import torch
from utils.pdm_decoder_study_diagnostics import SelectedValidationCapture as BaseCapture
from utils.pdm_decoder_study_diagnostics import setup_hooks as base_hooks, trace as base_trace


def evidence(observation,transition,states,gains,covariances,q):
    with torch.no_grad():
        previous=torch.cat((torch.zeros_like(states[:,:1]),states[:,:-1]),1)
        ar,ai=transition[...,0][:,None],transition[...,1][:,None]
        predicted=torch.stack((ar*previous[...,0]-ai*previous[...,1],ai*previous[...,0]+ar*previous[...,1]),-1)
        innovation=observation-predicted
        correction=states-predicted
        return dict(series=len(states),gain_mean=float(gains.double().mean()),gain_min=float(gains.min()),gain_max=float(gains.max()),
                    final_covariance_mean=float(covariances[:,-1].double().mean()),innovation_rms=float(innovation.double().square().mean().sqrt()),
                    correction_rms=float(correction.double().square().mean().sqrt()),process_variance=q.detach().double().cpu().tolist())


class ResetObserver:
    def __init__(self,decoder):self.decoder=decoder
    def remove(self):self.decoder.filter_observer=None


def setup_hooks(model):
    captured,handles=base_hooks(model)
    for i,e in enumerate(model.seasonals):
        if hasattr(e.decoder,'filter_observer'):
            e.decoder.filter_observer=lambda *args,i=i:captured.update({f'filter_{i}':evidence(*args)})
            handles.append(ResetObserver(e.decoder))
    return captured,handles


def trace(model,captured,step,loss,wd_by_id):
    filtered={k:v for k,v in captured.items() if not k.startswith('filter_')}
    row=base_trace(model,filtered,step,loss,wd_by_id)
    row['history_filter']={k:v for k,v in captured.items() if k.startswith('filter_')}
    row['filter_parameters']={n:dict(norm=float(p.detach().double().norm()),gradient_norm=float(p.grad.double().norm()),weight_decay=wd_by_id[id(p)]) for n,p in model.named_parameters() if n.endswith('.log_process_variance')}
    return row


class SelectedValidationCapture(BaseCapture):
    def __enter__(self):
        self.innovation=hasattr(self.model.seasonals[0].decoder,'filter_observer')
        self.filter_records={}
        super().__enter__()
        for i,e in enumerate(self.model.seasonals):
            if self.innovation:
                e.decoder.filter_observer=lambda *args,i=i:self.filter_records.setdefault(str(i),[]).append(evidence(*args))
                self.handles.append(ResetObserver(e.decoder))
        return self

    def report(self,prediction,truth):
        result=super().report(prediction,truth)
        result['history_state_estimator']='isotropic_feature_observation_filter' if self.innovation else 'additive_feature_drive_scan'
        result['filter_observation']='existing history drive projection; learned latent coordinates; H=I; R=I; P0=I'
        result['filter_noise_training']='MSE only; Q/R ratios are not calibrated noise estimates'
        result['history_filter']={}
        for scale,rows in self.filter_records.items():
            total=sum(v['series'] for v in rows)
            row=dict(series=total,gain_min=min(v['gain_min'] for v in rows),gain_max=max(v['gain_max'] for v in rows),process_variance=rows[0]['process_variance'])
            for k in ['gain_mean','final_covariance_mean']:
                row[k]=sum(v[k]*v['series'] for v in rows)/total
            for k in ['innovation_rms','correction_rms']:
                row[k]=(sum(v[k]**2*v['series'] for v in rows)/total)**.5
            result['history_filter'][scale]=row
        return result
