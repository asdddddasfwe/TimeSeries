"""Fixed-kernel policy comparison at the observed MAFirst entrance."""
import torch
from torch import nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp
from models.OscillatorPDMScaledMA import scaled_odd_kernel
from models.OscillatorPDMStatePatchMAFirstSeasonSliding import Model as Baseline


class KernelPolicyDecomposition(nn.Module):
    """Endpoint-replicated MA; policy changes no parameters, buffers or RNG."""

    def __init__(self, policy, lengths=(96, 48, 24, 12)):
        super().__init__()
        if policy not in ("physical9", "point9"):
            raise ValueError("Expected physical9 or point9")
        self.policy = policy
        self.kernels = {
            length: scaled_odd_kernel(9, 96, length) if policy == "physical9" else 9
            for length in lengths
        }
        self.decompositions = nn.ModuleDict({
            str(length): series_decomp(kernel)
            for length, kernel in self.kernels.items()
        })

    def forward(self, observed):
        if observed.ndim != 3 or str(observed.shape[1]) not in self.decompositions:
            raise ValueError("Expected configured [batch*channels, time, 1] history")
        return self.decompositions[str(observed.shape[1])](observed)


class Model(Baseline):
    """Keep all encoders, heads, mixing and normalization; change entrance MA."""

    def __init__(self, configs):
        policy = getattr(configs, "ma_kernel_policy", None)
        if configs.seq_len != 96 or policy not in ("physical9", "point9"):
            raise ValueError("Kernel9 comparison requires input96 and explicit policy")
        super().__init__(configs)
        self.initial_decomposition = KernelPolicyDecomposition(policy, self.lengths)
        self.ma_kernels = dict(self.initial_decomposition.kernels)

    def encode_multiscale(self, raw_scales):
        # This is the original entrance calculation with detached diagnostics.
        normalized = [self.normalize(raw_scales[0], "norm")]
        normalized += [
            norm(x, "norm") for norm, x in zip(self.coarse_normalizers, raw_scales[1:])
        ]
        batch, _, channels = normalized[0].shape
        seasons, trends, energies = [], [], []
        with torch.no_grad():
            coarse_trends, coarse_seasons = [], []
        for encoder, x in zip(self.seasonals, normalized):
            observed = x.transpose(1, 2).reshape(batch * channels, encoder.length, 1)
            seasonal, trend = self.initial_decomposition(observed)
            seasons.append(F.linear(seasonal, encoder.value_embedding.weight, None))
            trends.append(encoder.value_embedding(trend))
            with torch.no_grad():
                a, t, s = [
                    z.detach().reshape(batch, channels, encoder.length).double()
                    for z in (observed, trend, seasonal)
                ]
                energies.append(torch.stack([
                    a.square().mean((0, 2)), t.square().mean((0, 2)),
                    s.square().mean((0, 2)), (2 * t * s).mean((0, 2)),
                ], -1))
                # Average nonoverlapping bins to the actual coarsest grid.
                # This adds no observed history or interpolation-based samples.
                n = encoder.length // self.lengths[-1]
                coarse_trends.append(t.reshape(batch, channels, self.lengths[-1], n).mean(-1))
                coarse_seasons.append(s.reshape(batch, channels, self.lengths[-1], n).mean(-1))
        with torch.no_grad():
            correlations, valid_fractions = [], []
            for blocks in (coarse_trends, coarse_seasons):
                row, valid_row = [], []
                for i in range(4):
                    for j in range(i + 1, 4):
                        a = blocks[i] - blocks[i].mean(-1, keepdim=True)
                        b = blocks[j] - blocks[j].mean(-1, keepdim=True)
                        denom = (a.square().sum(-1) * b.square().sum(-1)).sqrt()
                        valid = denom > 1e-12
                        val = (a * b).sum(-1) / denom.clamp_min(1e-12)
                        row.append((val * valid).sum(0) / valid.sum(0).clamp_min(1))
                        valid_row.append(valid.double().mean(0))
                correlations.append(torch.stack(row))
                valid_fractions.append(torch.stack(valid_row))
            self.last_decomposition_diagnostics = {
                "energy_by_scale_variable": torch.stack(energies),
                "coarse_grid_correlation": torch.stack(correlations),
                "correlation_valid_fraction": torch.stack(valid_fractions),
            }
        seasons = self.encode_seasonal_features(seasons)
        for pdm in self.pdm_blocks:
            seasons, trends = pdm(seasons, trends)
        return normalized, (seasons, trends)
