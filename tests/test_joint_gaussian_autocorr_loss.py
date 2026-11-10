import unittest
from types import SimpleNamespace
import torch
from utils.joint_gaussian_autocorr_loss import correlation_matrix, joint_gaussian_autocorr_kl_loss as loss
from utils.joint_gaussian_autocov_loss import joint_gaussian_autocov_kl_loss as cov_loss
from scripts.train_timemix_autocorr import training_losses, loss_configuration, ARMS, NORMALIZED, selection_score


def independent_matrix(sequence, offsets, floor=.01):
    z=sequence-sequence.mean(1,keepdim=True)
    n=z.shape[1]
    rows=[]
    for i in offsets:
        cols=[]
        for j in offsets:
            lag=abs(i-j)
            cols.append((z[:,:n-lag]*z[:,lag:]).sum(1)/n)
        rows.append(torch.stack(cols,-1))
    c=torch.stack(rows,-2)
    v=c.diagonal(dim1=-2,dim2=-1)
    scale=v.clamp_min(floor)
    return c/(scale.sqrt().unsqueeze(-1)*scale.sqrt().unsqueeze(-2))+torch.diag_embed(1-v/scale)


class AutocorrTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(43)
        self.h=torch.randn(2,9,2,dtype=torch.double)
        self.p=torch.randn(2,8,2,dtype=torch.double)
        self.y=torch.randn_like(self.p)
        self.offsets=(0,1,2,4,8)

    def test_independent_gram_and_distribution_kl(self):
        seq=torch.cat((self.h,self.p),1)
        torch.testing.assert_close(correlation_matrix(seq,self.offsets),independent_matrix(seq,self.offsets))
        a=independent_matrix(torch.cat((self.h,self.y),1),self.offsets)+.01*torch.eye(5)
        b=independent_matrix(seq,self.offsets)+.01*torch.eye(5)
        zero=torch.zeros(2,2,5,dtype=torch.double)
        expected=torch.distributions.kl_divergence(torch.distributions.MultivariateNormal(zero,a),torch.distributions.MultivariateNormal(zero,b))/5
        torch.testing.assert_close(loss(self.h,self.p,self.y,offsets=self.offsets,reduction='none'),expected)

    def test_unit_diagonal_psd_and_floor(self):
        for scale in (0.,1e-5,.05,1.,100.):
            c=correlation_matrix(torch.cat((self.h,self.p),1)*scale,self.offsets)
            torch.testing.assert_close(c.diagonal(dim1=-2,dim2=-1),torch.ones(2,2,5,dtype=torch.double))
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(c).min()),-1e-12)

    def test_whole_sequence_positive_scaling_invariance(self):
        seq=torch.cat((self.h,self.p),1)
        torch.testing.assert_close(correlation_matrix(seq,self.offsets),correlation_matrix(seq*3+4,self.offsets))
        torch.testing.assert_close(loss(self.h,self.p,self.y,offsets=self.offsets),loss(self.h*3,self.p*3,self.y*3,offsets=self.offsets))

    def test_future_only_scaling_not_invariant(self):
        v=loss(self.h,self.p,self.y,offsets=self.offsets)
        changed=loss(self.h,self.p*3,self.y,offsets=self.offsets)
        self.assertGreater(float((v-changed).abs()),1e-5)

    def test_only_prediction_gradient(self):
        h=self.h.requires_grad_();p=self.p.requires_grad_();y=self.y.requires_grad_()
        loss(h,p,y,offsets=self.offsets).backward()
        self.assertIsNone(h.grad);self.assertIsNone(y.grad)
        self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0)

    def test_identical_zero(self):
        p=self.y.clone().requires_grad_()
        v=loss(self.h,p,self.y,offsets=self.offsets)
        self.assertLess(float(v.detach()),1e-12)
        v.backward();self.assertLess(float(p.grad.abs().max()),1e-12)

    def test_gradcheck(self):
        p=self.p.clone().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda x:loss(self.h,x,self.y,offsets=self.offsets), (p,),eps=1e-6,atol=1e-5,rtol=1e-3))

    def test_constant_and_near_constant_finite(self):
        for scale in (0.,1e-6):
            p=(self.p*scale+2).requires_grad_()
            v=loss(self.h*scale+2,p,self.y*scale+2,offsets=self.offsets)
            v.backward()
            self.assertTrue(torch.isfinite(v));self.assertTrue(torch.isfinite(p.grad).all())

    def test_invalid_inputs(self):
        for args in ({'variance_floor':0},{'jitter':float('nan')},{'offsets':(0,0,1)},{'offsets':(0,17)},{'offsets':(0,1.5)},{'reduction':'sum'}):
            with self.assertRaises(ValueError):loss(self.h,self.p,self.y,**args)
        with self.assertRaises(ValueError):loss(self.h,self.p*float('nan'),self.y,offsets=self.offsets)
        with self.assertRaises(ValueError):loss(self.h,self.p[:,:0],self.y[:,:0],offsets=self.offsets)

    def test_training_objective_and_gradients(self):
        cfg=loss_configuration(SimpleNamespace(variance_floor=.01,kl_jitter=.01));cfg['offsets']=self.offsets
        self.assertTrue(all(NORMALIZED.values()))
        for arm in ARMS:
            p=self.p.clone().requires_grad_()
            parts=training_losses(p,self.y,arm,cfg,self.h)
            expected=(p-self.y).square().mean()
            if arm=='covariance_kl':expected=expected+cov_loss(self.h,p,self.y,offsets=self.offsets)
            if arm=='correlation_kl':expected=expected+loss(self.h,p,self.y,offsets=self.offsets)
            torch.testing.assert_close(parts['loss'],expected)
            g=torch.autograd.grad(parts['loss'],p,retain_graph=True)[0]
            e=torch.autograd.grad(expected,p)[0]
            torch.testing.assert_close(g,e)
            torch.testing.assert_close(parts['cov_kl'],cov_loss(self.h,p,self.y,offsets=self.offsets))
            torch.testing.assert_close(parts['corr_kl'],loss(self.h,p,self.y,offsets=self.offsets))
            self.assertFalse(parts['cov_kl'].requires_grad);self.assertFalse(parts['corr_kl'].requires_grad)
            if arm=='control':self.assertEqual(float(parts['auxiliary_kl']),0)
        self.assertEqual(selection_score(dict(neural_mse=.4,kl=100)),.4)

    def test_mixed_precision_cpu(self):
        with torch.autocast('cpu',dtype=torch.bfloat16):
            v=loss(self.h.float().bfloat16(),self.p.float().bfloat16(),self.y.float().bfloat16(),offsets=self.offsets)
        self.assertEqual(v.dtype,torch.float32);self.assertTrue(torch.isfinite(v))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_full_gpu_shape_backward(self):
        h=torch.randn(32,96,7,device='cuda');y=torch.randn_like(h);p=torch.randn_like(h,requires_grad=True)
        with torch.autocast('cuda',dtype=torch.float16):v=loss(h,p,y)
        v.backward()
        self.assertTrue(torch.isfinite(v));self.assertTrue(torch.isfinite(p.grad).all())


if __name__=='__main__':unittest.main()
