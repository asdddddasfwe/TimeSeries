import unittest
import torch
from test_oscillator_mixer import config
from models.OscillatorTimeMix import Model as Neural
from models.OscillatorLinearBlend import Model
from layers.LinearForecastBlend import LinearForecastBlend
from exp.exp_basic import Exp_Basic,LazyModelDict


class FrameworkBlendTests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(1);torch.manual_seed(2021)
    def tearDown(self):torch.set_num_threads(self.threads)
    def test_discovery_and_real_checkpoint_forward(self):
        cfg=config(is_training=0)
        direct=LinearForecastBlend(Neural(cfg),torch.randn(2,17,11),.5).eval()
        model=Model(cfg)
        with self.assertRaises(RuntimeError):model(torch.randn(2,16,2))
        model.load_state_dict(direct.state_dict(),strict=True)
        x=torch.randn(2,16,2)
        with torch.no_grad():
            # Frozen parameters may choose a different FP32 attention kernel.
            torch.testing.assert_close(model(x),direct(x),rtol=1e-6,atol=2e-7)
            torch.testing.assert_close(model(x,None,torch.full((2,11,2),float('nan')),None),model(x),rtol=0,atol=0)
        mapping=Exp_Basic._scan_models_directory(None)
        self.assertIs(LazyModelDict(mapping)['OscillatorLinearBlend'],Model)
    def test_training_and_partial_load_are_rejected(self):
        with self.assertRaises(ValueError):Model(config(is_training=1))
        model=Model(config())
        with self.assertRaises(ValueError):model.train()
        with self.assertRaises(ValueError):model.load_state_dict({},strict=False)
    def test_ms_strict_load_and_batch_independence(self):
        cfg=config(is_training=0,features='MS',c_out=1)
        direct=LinearForecastBlend(Neural(cfg),torch.randn(2,17,11)).eval()
        model=Model(cfg);model.load_state_dict(direct.state_dict())
        x=torch.randn(3,16,2)
        with torch.no_grad():
            output=model(x);self.assertEqual(tuple(output.shape),(3,11,1))
            torch.testing.assert_close(output,torch.cat([model(v[None]) for v in x]),rtol=1e-5,atol=1e-6)


if __name__=='__main__':unittest.main()
