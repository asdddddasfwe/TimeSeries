import unittest
import torch
from utils.stable_gradient_clip import clip_finite_gradients


class StableGradientClipTests(unittest.TestCase):
    def test_large_fp32_gradients_do_not_overflow_adam_state(self):
        p=torch.nn.Parameter(torch.tensor([0.,0.]));p.grad=torch.tensor([3e30,4e30])
        norm,factor=clip_finite_gradients([p],1.)
        self.assertGreater(norm,4.9e30);self.assertLess(factor,1e-29)
        torch.testing.assert_close(p.grad,torch.tensor([.6,.8]))
        optimizer=torch.optim.Adam([p],lr=.001);optimizer.step()
        self.assertTrue(all(torch.isfinite(v).all() for v in optimizer.state[p].values()))

    def test_small_and_zero_gradients_preserved(self):
        for gradient in (torch.zeros(2),torch.tensor([.1,.2])):
            p=torch.nn.Parameter(torch.zeros(2));p.grad=gradient.clone()
            _,factor=clip_finite_gradients([p],1.)
            self.assertEqual(factor,1.);self.assertTrue(torch.equal(p.grad,gradient))

    def test_nonfinite_gradient_rejected(self):
        p=torch.nn.Parameter(torch.zeros(1));p.grad=torch.tensor([float('inf')])
        with self.assertRaises(FloatingPointError):clip_finite_gradients([p])


if __name__=='__main__':unittest.main()
