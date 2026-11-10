"""Raw MA -> FFT seasonal dual-axis attention / sliding trend attention -> A24 heads.

No Linear(1,32) precedes patching. Each observed four-point scalar patch is
projected directly to width32. Attention uses observed-window patch tokens,
never FFT routes or future truth. History trend conditions seasonal dynamics;
history season also conditions trend dynamics at the corresponding logits.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from layers.LagARFormer import sinusoidal_positions
from models.OscillatorDualEncoderTimeMixer import SlidingPatchTrendEncoder
from models.OscillatorPDMStatePatch import state_patch_forecast
from models.OscillatorPDMStatePatchDual import TrendPatchHead
from models.OscillatorPDMStatePatchPreDecomp import Model as PreDecomp, fft_patch_features
from models.OscillatorPDMTrendCondition import TrendConditionedDecoder


class ScalarSlidingPatchEncoder(SlidingPatchTrendEncoder):
    """[N,T,1] -> Linear(4,32) patch tokens -> attention -> [N,T,32]."""
    def __init__(self, width, d_ff, dropout):
        super().__init__(width, d_ff, dropout, heads=4, patch_length=4, stride=2)
        self.patch_embedding = nn.Linear(self.patch_length, width)

    def patchify(self, observed):
        if observed.ndim != 3 or observed.shape[-1] != 1 or observed.shape[1] < 1:
            raise ValueError('Expected observed scalar history [N,T,1]')
        length = observed.shape[1]
        padded_length = max(length, self.patch_length)
        padded_length += (-(padded_length-self.patch_length)) % self.stride
        padded = F.pad(observed.transpose(1, 2), (0, padded_length-length), mode='replicate')
        return padded.unfold(-1, self.patch_length, self.stride).permute(0, 2, 3, 1), padded_length

    def forward(self, observed):
        patches, padded_length = self.patchify(observed)
        n, count, size, _ = patches.shape
        tokens = self.patch_embedding(patches.reshape(n, count, size))
        encoded = self.attention(tokens+sinusoidal_positions(count,self.width).to(tokens))
        decoded = self.patch_readout(encoded).reshape(n, count, size, self.width)
        restored = self.overlap_average(decoded, padded_length, observed.shape[1])
        # The Transformer has its token residuals. No separate scalar embedding
        # or extra input skip is introduced outside the patch projection.
        return self.output_norm(restored)


class SeasonConditionedDecoder(TrendConditionedDecoder):
    """History seasonal mean/last shifts trend decay/frequency logits."""
    def __init__(self, decoder):
        super().__init__(decoder)
        self.season_condition = self.trend_condition
        self.season_context_norm = self.trend_context_norm
        del self.trend_condition
        del self.trend_context_norm

    def dynamics(self, history, season_history=None):
        if season_history is None or season_history.shape != history.shape:
            raise ValueError('Expected same-scale trend/season histories [N,T,D]')
        context = self.context_norm(torch.cat((history.mean(1), history[:, -1]), -1))
        seasonal = self.season_context_norm(torch.cat((season_history.mean(1), season_history[:, -1]), -1))
        dtype = torch.float64 if context.dtype == torch.float64 else torch.float32
        shifts = self.condition(context).to(dtype) + self.season_condition(seasonal).to(dtype)
        decay, frequency = shifts.chunk(2, -1)
        radius = torch.exp(-F.softplus(self.log_decay.to(dtype) + decay))
        angle = math.pi * torch.sigmoid(self.frequency_logits.to(dtype) + frequency)
        transition = torch.stack((radius*angle.cos(), radius*angle.sin()), -1)
        initial = self.encode_history_states(history, transition)[:, -1]
        if self.stride != 1:
            radius, angle = radius.pow(1./self.stride), angle/self.stride
            transition = torch.stack((radius*angle.cos(), radius*angle.sin()), -1)
        return transition, initial, radius, angle


class Model(PreDecomp):
    def __init__(self, configs):
        super().__init__(configs)
        width = configs.d_model
        self.season_patch_codecs = self.shared_patch_codecs
        del self.shared_patch_codecs
        for codec in self.season_patch_codecs:
            codec.patch_projection.register_parameter('bias',None)
            codec.patch_readout.register_parameter('bias',None)
        self.trend_patch_encoders = nn.ModuleList([
            ScalarSlidingPatchEncoder(width,configs.d_ff,configs.dropout) for _ in self.seasonals])
        self.feature_trend_heads = nn.ModuleList([TrendPatchHead(e) for e in self.seasonals])
        for head in self.feature_trend_heads:
            # Remove the copied trend->season edge, then add the reversed edge.
            from models.OscillatorPDMStatePatchDisconnected import DisconnectedDecoder
            head.decoder = SeasonConditionedDecoder(DisconnectedDecoder(head.decoder))
        self.trend_history_gain_offset = nn.Parameter(torch.zeros(self.enc_in))
        self.trend_drive_gain_offset = nn.Parameter(torch.zeros(self.enc_in))

    def clone_new_heads(self):
        """After common fresh scratch load, copy head values independently."""
        for encoder, head in zip(self.seasonals, self.feature_trend_heads):
            head.temporal_projection.load_state_dict(encoder.temporal_projection.state_dict())
            state = head.decoder.state_dict()
            state.update({n:v.clone() for n,v in encoder.decoder.state_dict().items()
                          if n in state and state[n].shape == v.shape})
            head.decoder.load_state_dict(state)
            nn.init.zeros_(head.decoder.readout.output_projection.weight)
            nn.init.zeros_(head.decoder.readout.output_projection.bias)
            nn.init.zeros_(head.decoder.season_condition.weight)

    def encode_multiscale(self, raw_scales):
        normalized = [self.normalize(raw_scales[0], 'norm')]
        normalized += [norm(x, 'norm') for norm,x in zip(self.coarse_normalizers, raw_scales[1:])]
        batch, _, channels = normalized[0].shape
        seasons, trends = [], []
        for encoder,x,se,te in zip(self.seasonals,normalized,self.season_patch_codecs,self.trend_patch_encoders):
            observed = x.transpose(1, 2).reshape(batch*channels,encoder.length,1)
            seasonal,trend = self.initial_decomposition(observed)
            seasons.append(fft_patch_features(encoder,seasonal,se))
            trends.append(te(trend))
        for pdm in self.pdm_blocks:
            seasons,trends = pdm(seasons,trends)
        return normalized,(seasons,trends)

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len,self.enc_in):
            raise ValueError('Expected observed [B,seq_len,enc_in]')
        scales,(seasons,trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha,beta = self.branch_gains()
        seasonal = [state_patch_forecast(e,x,alpha,beta,s,t)
                    for e,x,s,t in zip(self.seasonals,scales,seasons,trends)]
        alpha_t,beta_t = 1+self.trend_history_gain_offset,1+self.trend_drive_gain_offset
        trend = [state_patch_forecast(e,x,alpha_t,beta_t,t,s)
                 for e,x,t,s in zip(self.feature_trend_heads,scales,trends,seasons)]
        return torch.stack(seasonal+trend,-1)

    def optimizer_parameter_groups(self, weight_decay):
        protected = set()
        for encoder in [*self.seasonals,*self.feature_trend_heads]:
            decoder = encoder.decoder
            protected.update((id(decoder.log_decay),id(decoder.frequency_logits)))
            for name in ('condition','trend_condition','season_condition'):
                if hasattr(decoder,name):
                    protected.update(id(p) for p in getattr(decoder,name).parameters())
        return [dict(params=[p for p in self.parameters() if id(p) not in protected],
                     weight_decay=weight_decay,group_name='regular'),
                dict(params=[p for p in self.parameters() if id(p) in protected],
                     weight_decay=self.dynamics_weight_decay,group_name='ssm_dynamics')]
