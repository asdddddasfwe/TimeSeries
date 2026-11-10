"""Preserve all PDM blocks; isolate decomposition at the decoder entrance."""
import torch
from torch import nn
from models.OscillatorPDMScaledMA import Model as ScaledMA, ScaleAwareDecomposition
from models.OscillatorPDMSeasonalHead import LinearFeatureTrendHead
from layers.OscillatorCalibratedMixer import calibrated_seasonal_forecast
from layers.OscillatorTimeMix import complex_oscillator_scan
from utils.logger import get_logger

class Model(ScaledMA):
    def __init__(self, configs, variant=None):
        self.readout_split_variant = variant or getattr(configs, 'readout_split_variant', 'seasonal')
        if self.readout_split_variant not in ('full', 'seasonal'):
            raise ValueError('Readout split variant must be full or seasonal')
        super().__init__(configs)
        self.readout_decomposition = ScaleAwareDecomposition(configs.moving_avg, configs.seq_len, self.lengths)
        self.feature_trend_heads = nn.ModuleList([
            LinearFeatureTrendHead(length, self.pred_len, configs.d_model) for length in self.lengths])
        # Start the added output path at zero; both matched arms share these
        # exact weights. Only the readout weight is zero, avoiding a dead path.
        for head in self.feature_trend_heads:
            nn.init.zeros_(head.readout.weight)
        self.register_buffer('readout_split_version', torch.tensor(1, dtype=torch.long))
        get_logger(__name__).info(
            'OscillatorPDMReadoutSplit overrides inherited routing descriptions: '
            'all FFT/PDM blocks unchanged; final H split S=H-MA(H), T=MA(H); '
            'recurrence receives %s; four zero-output-initialized linear T heads; '
            'original raw MA25 bypass retained; 12 components; shared history context',
            'H' if self.readout_split_variant == 'full' else 'S')

    def decoder_histories(self, histories):
        pairs = [self.readout_decomposition(h) for h in histories]
        seasons, trends = map(list, zip(*pairs))
        return (histories if self.readout_split_variant == 'full' else seasons), trends

    def predict_components(self, x_enc):
        """[B,H,C,12]: recurrent4, feature_trend4, raw_trend4; normalized."""
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected observed history [B,seq_len,enc_in]')
        scales, histories = self.encode_multiscale(self.downsample(x_enc))
        recurrent, trends = self.decoder_histories(histories)
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        outputs = [calibrated_seasonal_forecast(e, x, alpha, beta, encoded_history=h,
            scan_function=complex_oscillator_scan) for e,x,h in zip(self.seasonals,scales,recurrent)]
        feature = [head(t).reshape(batch,channels,self.pred_len).transpose(1,2)
            for head,t in zip(self.feature_trend_heads,trends)]
        _, raw = self.raw_trend_decomposition(scales[0])
        raw_scales = self.raw_trend_mix([x.transpose(1,2) for x in self.downsample(raw)])
        bypass = [head(x.transpose(1,2)).transpose(1,2) for head,x in zip(self.raw_trend_heads,raw_scales)]
        return torch.stack(outputs+feature+bypass,dim=-1)

    def forecast(self, x_enc, zero_acf=False, intervention='normal'):
        if zero_acf or intervention != 'normal':
            raise ValueError('Readout split supports normal forecasting only')
        parts = self.predict_components(x_enc)
        # Preserve the baseline's two grouped sums, then add the new trend path.
        p = parts[...,:4].sum(-1) + parts[...,8:].sum(-1)
        p = self.normalize(p + parts[...,4:8].sum(-1), 'denorm')
        return p[...,-1:] if self.features == 'MS' and self.c_out == 1 else p
