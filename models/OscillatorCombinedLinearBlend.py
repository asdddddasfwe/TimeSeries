"""组合神经模型与单独拟合线性模型的固定等权推理适配器。"""
import torch
from layers.LinearForecastBlend import LinearForecastBlend
from models.OscillatorLinearBlend import Model as OriginalBlend
from models.OscillatorLowRankLocalTimeMix import Model as CombinedNeural


class Model(OriginalBlend):
    def __init__(self, configs):
        if getattr(configs, "is_training", 0):
            raise ValueError("组合模型采用两阶段拟合；本适配器只支持冻结推理")
        neural = CombinedNeural(configs)
        coefficients = torch.zeros(configs.enc_in, configs.seq_len + 1, configs.pred_len)
        LinearForecastBlend.__init__(self, neural, coefficients, neural_weight=.5)
        self._components_loaded = False
        for parameter in self.neural.parameters():
            parameter.requires_grad_(False)
        self.eval()
