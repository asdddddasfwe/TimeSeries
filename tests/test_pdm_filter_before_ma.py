"""Small deterministic invariants for observed-only low-pass preprocessing."""
import unittest,torch,math
from models.OscillatorPDMStatePatchFilteredMA import ObservedLowPass
class FilterBeforeMATests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.manual_seed(71);self.f=ObservedLowPass();self.x=torch.randn(3,96,1);self.c=torch.full((3,1),.25)
 def test_shape_finite(self):
  z=self.f(self.x,self.c);self.assertEqual(z.shape,self.x.shape);self.assertTrue(torch.isfinite(z).all())
 def test_observed_window_mean_preserved(self):
  for n,spacing in [(96,1),(48,2),(24,4),(12,8)]:
   x=self.x[:,:n];z=self.f(x,self.c,spacing);torch.testing.assert_close(z.mean(1),x.mean(1),atol=1e-7,rtol=1e-6)
 def test_dc_constant_preserved(self):
  x=torch.ones(3,96,1)*3.25;torch.testing.assert_close(self.f(x,self.c),x,atol=1e-6,rtol=0)
 def test_strength_endpoints_and_interpolation(self):
  low=self.f(self.x,self.c);torch.testing.assert_close(self.x-0*(self.x-low),self.x,atol=0,rtol=0);torch.testing.assert_close(self.x-1*(self.x-low),low)
  torch.testing.assert_close(self.x-.1*(self.x-low),.9*self.x+.1*low)
 def test_cutoff_and_strength_gradients(self):
  c=self.c.clone().requires_grad_();a=torch.tensor(-4.59512,requires_grad=True);low=self.f(self.x,c);z=self.x-a.sigmoid()*(self.x-low);z.square().mean().backward()
  self.assertTrue(torch.isfinite(c.grad).all() and c.grad.abs().max()>0);self.assertTrue(torch.isfinite(a.grad) and a.grad.abs()>0)
 def test_high_frequency_more_attenuated(self):
  t=torch.arange(96,dtype=torch.float32);lo=torch.sin(2*math.pi*t/32)[None,:,None];hi=torch.sin(2*math.pi*t/3)[None,:,None];c=torch.tensor([[.15]])
  self.assertGreater(float(self.f(lo,c).square().mean()/lo.square().mean()),float(self.f(hi,c).square().mean()/hi.square().mean()))
 def test_lowpass_stronger_for_lower_cutoff(self):
  a=self.f(self.x,torch.full_like(self.c,.1));b=self.f(self.x,torch.full_like(self.c,.4));self.assertLess(float(a.var()),float(b.var()))
 def test_batch_variable_independence(self):
  y=self.x.clone();y[1:]=100*y[1:];torch.testing.assert_close(self.f(y,self.c)[0],self.f(self.x,self.c)[0],atol=0,rtol=0)
 def test_invalid_input(self):
  with self.assertRaises(ValueError):self.f(torch.zeros(2,1,1),torch.zeros(2,1))
  with self.assertRaises(ValueError):self.f(self.x,torch.zeros(1,1))
if __name__=='__main__':unittest.main()