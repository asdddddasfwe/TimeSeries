"""Capture fusion re-decomposition during the existing validation forward."""
import math
from utils.pdm_disconnected_diagnostics import SelectedValidationCapture as BaseCapture
from utils.pdm_disconnected_diagnostics import setup_hooks, trace


class SelectedValidationCapture(BaseCapture):
    def __enter__(self):
        self.fused = hasattr(self.model.pdm_blocks[0], 'decomposition')
        self.fusion_stats = {}
        super().__enter__()
        if self.fused:
            for block in self.model.pdm_blocks:
                self.handles.append(block.decomposition.register_forward_hook(self.capture_fusion))
        return self

    def capture_fusion(self, module, args, output):
        z=args[0].detach().double();s,t=[v.detach().double() for v in output]
        row=self.fusion_stats.setdefault(str(z.shape[1]),dict(calls=0,elements=0,reconstruction_max_abs=0.,merged_square=0.,season_square=0.,trend_square=0.))
        row['calls']+=1;row['elements']+=z.numel()
        row['reconstruction_max_abs']=max(row['reconstruction_max_abs'],float((z-s-t).abs().max()))
        for key,v in [('merged',z),('season',s),('trend',t)]:row[key+'_square']+=float(v.square().sum())

    def report(self,prediction,truth):
        result=super().report(prediction,truth)
        result['pdm_fusion']=self.fused
        result['fusion_definition']='F=S+T+MLP(MS_s(S)+MS_t(T)); (F-MA(F),MA(F))' if self.fused else None
        result['fusion_decomposition']={k:dict(calls=v['calls'],reconstruction_max_abs=v['reconstruction_max_abs'],**{n+'_rms':math.sqrt(v[n+'_square']/v['elements']) for n in ['merged','season','trend']}) for k,v in self.fusion_stats.items()}
        result['explicit_trend_logit_conditioning']=False
        result['indirect_trend_to_season_via_fusion']=self.fused
        return result
