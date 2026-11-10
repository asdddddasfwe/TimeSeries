"""Isolate entrance MA9 boundary extension; keep the validated mean drive model."""
import torch
from torch.nn import functional as F
from models.OscillatorPDMStatePatchDriveDiagnostic import Model as MeanDriveModel
from models.OscillatorPDMStatePatchMAKernel9 import KernelPolicyDecomposition


class BoundaryDecomposition(KernelPolicyDecomposition):
    """Same MA9 weights and length; reflection implemented with deterministic views."""
    def __init__(self, mode, lengths=(96, 48, 24, 12), channels=7):
        super().__init__('point9', lengths)
        if mode not in ('replicate', 'reflect'):
            raise ValueError('Require explicit replicate or reflect boundary')
        self.boundary_mode = mode
        self.channels = channels
        self.last_boundary_diagnostics = {}

    def forward(self, observed):
        length = observed.shape[1]
        if observed.ndim != 3 or str(length) not in self.decompositions:
            raise ValueError('Expected configured [B*C,L,1] observed history')
        if self.boundary_mode == 'replicate':
            seasonal, trend = super().forward(observed)
        else:
            # Exclude endpoints when reflecting; no CUDA reflection-pad backward atomics.
            padded = torch.cat((observed[:, 1:5].flip(1), observed,
                                observed[:, -5:-1].flip(1)), dim=1)
            trend = F.avg_pool1d(padded.transpose(1, 2), 9, 1).transpose(1, 2)
            seasonal = observed - trend
        with torch.no_grad():
            _, legacy = super().forward(observed.detach())
            delta = (trend.detach() - legacy).reshape(-1, self.channels, length).double()
            edges = torch.cat((delta[..., :4], delta[..., -4:]), -1)
            self.last_boundary_diagnostics[str(length)] = dict(
                mode=self.boundary_mode, length=length, kernel=9, affected_positions=8,
                edge_trend_change_mse_by_variable=edges.square().mean((0, 2)).cpu().tolist(),
                interior_trend_change_max=float(delta[..., 4:-4].abs().max()),
                reconstruction_max=float((seasonal.detach()+trend.detach()-observed.detach()).abs().max()),
                diagnostics_detached_no_RNG=True)
        return seasonal, trend


class Model(MeanDriveModel):
    """Only replace parameter-free entrance decomposition. No extra head or bypass."""
    def __init__(self, configs):
        if getattr(configs, 'state_drive_policy', None) != 'mean':
            raise ValueError('Boundary diagnosis requires unchanged mean drive')
        super().__init__(configs)
        self.boundary_padding = getattr(configs, 'boundary_padding', None)
        self.initial_decomposition = BoundaryDecomposition(self.boundary_padding, self.lengths, self.enc_in)