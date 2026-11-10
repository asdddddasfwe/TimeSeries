"""MA-first 0.318 model, ONLY untying the shared pointwise embedding.

The seasonal Linear(1,D) has no bias, as in the MA-first residual formula.
The trend embedding independently clones the existing weight and bias.
Keep trend->season dynamics, independent PDM MLPs and every original head.
This is independent projection parameters, not fully disconnected branches.
"""
import copy
from torch import nn
from models.OscillatorPDMStatePatchMAFirst import Model as MAFirst


class Model(MAFirst):
    def __init__(self,configs):
        super().__init__(configs)
        self.trend_embeddings=nn.ModuleList([copy.deepcopy(e.value_embedding) for e in self.seasonals])
        for encoder in self.seasonals:
            encoder.value_embedding.register_parameter('bias',None)

    def clone_embeddings(self,source):
        """After fresh common scratch load, copy independent trend embeddings."""
        for embedding,encoder in zip(self.trend_embeddings,source.seasonals):
            embedding.load_state_dict(encoder.value_embedding.state_dict())

    def encode_multiscale(self,raw_scales):
        normalized=[self.normalize(raw_scales[0],'norm')]
        normalized += [norm(x,'norm') for norm,x in zip(self.coarse_normalizers,raw_scales[1:])]
        batch,_,channels=normalized[0].shape
        seasons,trends=[],[]
        for encoder,trend_embedding,x in zip(self.seasonals,self.trend_embeddings,normalized):
            observed=x.transpose(1,2).reshape(batch*channels,encoder.length,1)
            seasonal,trend=self.initial_decomposition(observed)
            seasons.append(encoder.value_embedding(seasonal))
            trends.append(trend_embedding(trend))
        seasons=self.encode_seasonal_features(seasons)
        for pdm in self.pdm_blocks:
            seasons,trends=pdm(seasons,trends)
        return normalized,(seasons,trends)
