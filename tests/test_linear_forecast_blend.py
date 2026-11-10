import unittest
import numpy as np
import torch
from torch import nn
from layers.LinearForecastBlend import LinearForecastBlend
from scripts.diagnose_timemix_linear_reference import predict


class HistoryOnly(nn.Module):
    def __init__(self,ms=False):
        super().__init__();self.features='MS' if ms else 'M';self.c_out=1 if ms else 3
    def forward(self,x,*unused,**kwargs):
        out=x[:,-1:].expand(-1,4,-1)
        return out[...,-1:] if self.features=='MS' else out


class BlendTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2021);self.x=torch.randn(5,8,3);self.w=torch.randn(3,9,4)
    def test_matches_independently_fitted_numpy_formula(self):
        model=LinearForecastBlend(HistoryOnly(),self.w).eval()
        expected=predict(self.x.double().numpy(),self.w.double().numpy())
        np.testing.assert_allclose(model.linear_forecast(self.x).numpy(),expected,rtol=1e-5,atol=2e-6)
        torch.testing.assert_close(model(self.x),.5*(HistoryOnly()(self.x)+model.linear_forecast(self.x)))
    def test_batch_independence_and_future_isolation(self):
        model=LinearForecastBlend(HistoryOnly(),self.w).eval()
        torch.testing.assert_close(model(self.x),torch.cat([model(x[None]) for x in self.x]))
        torch.testing.assert_close(model(self.x,None,torch.full((5,4,3),float('nan'))),model(self.x))
    def test_ms_and_strict_roundtrip(self):
        model=LinearForecastBlend(HistoryOnly(ms=True),self.w,.5).eval()
        self.assertEqual(tuple(model(self.x).shape),(5,4,1))
        restored=LinearForecastBlend(HistoryOnly(ms=True),torch.zeros_like(self.w),.9).eval()
        restored.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(restored(self.x),model(self.x),rtol=0,atol=0)
    def test_only_own_channel_enters_linear_map(self):
        model=LinearForecastBlend(HistoryOnly(),self.w).eval();changed=self.x.clone();changed[:,:,1]+=10
        torch.testing.assert_close(model.linear_forecast(self.x)[:,:,[0,2]],model.linear_forecast(changed)[:,:,[0,2]])
    def test_constant_history_finite_and_shape_guard(self):
        model=LinearForecastBlend(HistoryOnly(),self.w).eval()
        self.assertTrue(torch.isfinite(model(torch.ones_like(self.x))).all())
        with self.assertRaises(ValueError):model(torch.ones(5,7,3))


if __name__=='__main__':unittest.main()
