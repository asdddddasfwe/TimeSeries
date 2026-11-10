"""Capture raw maps and actual smoothed corrections from existing forwards."""
import math
import torch
from utils.pdm_trend_prelinear_diagnostics import (
    SelectedValidationCapture as PreLinearCapture, prelinear_parameter_audit)


class SelectedValidationCapture(PreLinearCapture):
    def __init__(self, *args):
        super().__init__(*args)
        self.filtered = {}

    def __enter__(self):
        super().__enter__()
        for i, block in enumerate(self.model.pdm_blocks):
            mixer = block.mixing_multi_scale_trend
            for length, smoother in zip(getattr(mixer, 'lengths', []), getattr(mixer, 'correction_filters', [])):
                key = f'layer{i}_length{length}'
                self.handles.append(smoother.register_forward_hook(
                    lambda module,args,value,key=key:self.capture_filter(key,args,value)))
        return self

    def capture_filter(self, key, args, output):
        trend, mapped = [x.detach().double() for x in args]
        output = output.detach().double()
        values = dict(input=trend, output=output, raw_correction=mapped-trend,
                      filtered_correction=output-trend)
        row = self.filtered.setdefault(key, dict(elements=0, diff_elements=0,
            **{name+suffix:0. for name in values for suffix in ['_square','_diff_square']}))
        row['elements'] += trend.numel()
        row['diff_elements'] += trend[...,2:].numel()
        for name,value in values.items():
            if not torch.isfinite(value).all():
                raise RuntimeError('Nonfinite smoothed trend diagnostics')
            row[name+'_square'] += float(value.square().sum())
            diff = value[...,2:] - 2*value[...,1:-1] + value[...,:-2]
            row[name+'_diff_square'] += float(diff.square().sum())

    def report(self, prediction, truth):
        result = super().report(prediction, truth)
        result['prelinear_transforms_definition'] = 'Raw WT+b before optional correction smoothing'
        result['smooth_residual_transforms'] = {k:{
            **{name+'_rms':math.sqrt(v[name+'_square']/v['elements']) for name in ['input','output','raw_correction','filtered_correction']},
            **{name+'_second_difference_rms':math.sqrt(v[name+'_diff_square']/v['diff_elements']) for name in ['input','output','raw_correction','filtered_correction']}
        } for k,v in self.filtered.items()}
        return result
