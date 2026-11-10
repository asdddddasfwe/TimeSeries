"""最细尺度局部驱动与秩16时间投影的组合。"""
import torch
from torch import nn
from models.OscillatorLocalDriveTimeMix import Model as LocalDrive


class Model(LocalDrive):
    def __init__(self, configs):
        super().__init__(configs)
        dense = self.seasonals[0].temporal_projection
        if min(dense.in_features, dense.out_features) < 16:
            raise ValueError("秩16投影要求输入和输出长度至少为16")
        self.seasonals[0].temporal_projection = nn.Sequential(
            nn.Linear(dense.in_features, 16), nn.Linear(16, dense.out_features))
        self.initialize_projection(dense.weight, dense.bias)

    @torch.no_grad()
    def initialize_projection(self, weight, bias):
        # 平衡分解来自同一随机稠密矩阵，不读取数据或已训练参数。
        u, s, vh = torch.linalg.svd(weight.double(), full_matrices=False)
        root = s[:16].sqrt()
        first, second = self.seasonals[0].temporal_projection
        first.weight.copy_((root[:, None] * vh[:16]).to(first.weight))
        first.bias.zero_()
        second.weight.copy_((u[:, :16] * root[None, :]).to(second.weight))
        second.bias.copy_(bias)

    @torch.no_grad()
    def initialize_from_control(self, control):
        source = control.state_dict()
        own = self.state_dict()
        prefix = "seasonals.0.temporal_projection."
        for name in own:
            if not name.startswith(prefix):
                own[name].copy_(source[name])
        self.load_state_dict(own, strict=True)
        self.initialize_projection(source[prefix + "weight"], source[prefix + "bias"])
        for name, value in self.state_dict().items():
            if not name.startswith(prefix):
                torch.testing.assert_close(value, source[name], rtol=0, atol=0)
