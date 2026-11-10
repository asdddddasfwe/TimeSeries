"""Existing PDM point mean plus a history-only conditional Gaussian head.

forward retains the parent's exact point-prediction API. Train the distribution
using forward_distribution + utils.directed_gaussian.gaussian_nll. Generic MSE
run.py training alone does NOT train this new head. No cross-variable edges.
"""
import torch
from layers.DirectedGaussianHead import DirectedGaussianHead
from models.OscillatorPDMTrendCondition import Model as PointModel


class Model(PointModel):
    def __init__(self, configs):
        super().__init__(configs)
        lags = getattr(configs, 'gaussian_lags', (1, 2, 4, 8, 12, 24))
        if isinstance(lags, str):
            lags = tuple(int(v) for v in lags.split(',') if v.strip())
        if self.normalize.affine:
            raise ValueError('This adapter supports the existing non-affine normalization only')
        width = self.seasonals[0].value_embedding.out_features * 4 * len(self.seasonals)
        # Preserve the parent initialization RNG state, including future dropout.
        with torch.random.fork_rng(devices=[]):
            self.probability_head = DirectedGaussianHead(
                width, self.pred_len, tuple(lags),
                hidden=int(getattr(configs, 'gaussian_hidden', 32)),
                variance_floor=float(getattr(configs, 'gaussian_variance_floor', 1e-4)),
                initial_variance=float(getattr(configs, 'gaussian_initial_variance', 1.)),
                row_bound=float(getattr(configs, 'gaussian_row_bound', .95)))

    def forward_distribution(self, x_enc, *args, detach_backbone=False, **kwargs):
        """One parent forward, capturing final existing seasonal/trend streams.

        Return mean/d in the parent's output units (dataset standardized units
        in existing experiments). Reuse the existing denorm std squared for d;
        do not alter or recompute any model normalization statistics.
        detach_backbone blocks likelihood gradients into the point backbone.
        Call eval() on the backbone when fitting a frozen-mean experiment.
        """
        captured = []
        def collect(module, inputs, output):
            seasons, trends = output
            features = [v for stream in (seasons, trends) for z in stream
                        for v in (z.mean(1), z[:, -1])]
            captured.append(torch.cat(features, -1))
        handle = self.pdm_blocks[-1].register_forward_hook(collect)
        try:
            mean = super().forward(x_enc, *args, **kwargs)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected exactly one final mixing forward')
        context = captured[0].reshape(x_enc.shape[0], self.enc_in, -1)
        if detach_backbone:
            mean, context = mean.detach(), context.detach()
        coefficients, variance = self.probability_head(context)
        if not self.normalize.non_norm:
            variance = variance * self.normalize.stdev.to(variance).square()
        if self.features == 'MS' and self.c_out == 1:
            coefficients, variance = coefficients[:, :, -1:], variance[:, :, -1:]
        return dict(mean=mean, coefficients=coefficients, variance=variance,
                    lags=self.probability_head.lags)
