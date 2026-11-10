"""The 0.318 independent-MLP model, ONLY exchanging MA and Linear(1,D).

For E(x)=Wx+b, MA(E(x))=W MA(x)+b and E(x)-MA(E(x))=W(x-MA(x)).
Thus the existing embedding weight remains shared across the two branches;
its existing bias is applied to trend only. No new parameter, patch encoder,
attention, conditioning, loss, or prediction head is introduced here.
"""
from torch.nn import functional as F
from models.OscillatorPDMStatePatchSeparateMLP import Model as SeparateMLP


class Model(SeparateMLP):
    def encode_multiscale(self,raw_scales):
        normalized=[self.normalize(raw_scales[0],'norm')]
        normalized += [norm(x,'norm') for norm,x in zip(self.coarse_normalizers,raw_scales[1:])]
        batch,_,channels=normalized[0].shape
        seasons,trends=[],[]
        for encoder,x in zip(self.seasonals,normalized):
            observed=x.transpose(1,2).reshape(batch*channels,encoder.length,1)
            seasonal,trend=self.initial_decomposition(observed)
            seasons.append(F.linear(seasonal,encoder.value_embedding.weight,None))
            trends.append(encoder.value_embedding(trend))
        seasons=self.encode_seasonal_features(seasons)
        for pdm in self.pdm_blocks:
            seasons,trends=pdm(seasons,trends)
        return normalized,(seasons,trends)
