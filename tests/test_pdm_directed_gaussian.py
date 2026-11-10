"""Independent likelihood, sampling, gradient and retained-PDM contracts."""
import copy
import math
import unittest
import numpy as np
import torch
from layers.DirectedGaussianHead import DirectedGaussianHead
from utils.directed_gaussian import gaussian_nll, innovations, predictive_covariance, sample_paths, transition_matrix
from models.OscillatorPDMDirectedGaussian import Model
from models.OscillatorPDMTrendCondition import Model as PointModel
from test_oscillator_mixer import config


class GaussianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def parameters(self):
        torch.manual_seed(7)
        mean=torch.randn(2,7,2,dtype=torch.double)
        target=torch.randn_like(mean)
        coeff=torch.randn(2,7,2,2,dtype=torch.double)*.15
        variance=.3+torch.rand_like(mean)
        return mean,target,coeff,variance,(1,3)
    def test_nll_and_covariance_against_numpy_dense(self):
        m,y,b,d,lags=self.parameters();cov=predictive_covariance(m,b,d,lags)
        total=0.
        for n in range(2):
            for c in range(2):
                a=np.eye(7)
                for t in range(7):
                    for j,k in enumerate(lags):
                        if t>=k:a[t,t-k]=-b[n,t,c,j].item()
                inverse=np.linalg.inv(a);sigma=inverse@np.diag(d[n,:,c].numpy())@inverse.T
                np.testing.assert_allclose(cov[n,c].numpy(),sigma,rtol=1e-12,atol=1e-12)
                residual=(y-m)[n,:,c].numpy()
                total+=.5*(7*np.log(2*np.pi)+np.linalg.slogdet(sigma)[1]+residual@np.linalg.solve(sigma,residual))
        self.assertAlmostEqual(gaussian_nll(m,y,b,d,lags).item(),total/m.numel(),places=12)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(cov)>0))
    def test_gradcheck_and_target_detach(self):
        m,y,b,d,lags=self.parameters()
        m.requires_grad_();b.requires_grad_();d.requires_grad_();y.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda a,c,v:gaussian_nll(a,y,c,v,lags),(m,b,d),atol=1e-5))
        gaussian_nll(m,y,b,d,lags).backward();self.assertIsNone(y.grad)
        for v in (m,b,d):self.assertTrue(torch.isfinite(v.grad).all())
    def test_no_reverse_time_dependence_and_diagonal_case(self):
        m,y,b,d,lags=self.parameters();before=innovations(m,y,b,lags)
        y2=y.clone();y2[:,4:]+=123.;after=innovations(m,y2,b,lags)
        torch.testing.assert_close(before[:,:4],after[:,:4],rtol=0,atol=0)
        matrix=transition_matrix(b,lags);self.assertEqual(torch.count_nonzero(matrix.triu()).item(),0)
        empty=b[...,:0]
        expected=.5*((y-m).square()/d+d.log()+math.log(2*math.pi))
        torch.testing.assert_close(gaussian_nll(m,y,empty,d,(),reduction='none'),expected)
        bad=d.clone();bad[0,0,0]=0
        with self.assertRaises(ValueError):gaussian_nll(m,y,b,bad,lags)
        with self.assertRaises(ValueError):gaussian_nll(m,y,b,d,(0,3))
    def test_ancestral_sampling_matches_joint_moments(self):
        m,y,b,d,lags=self.parameters();m,b,d=m[:1],b[:1],d[:1]
        samples=sample_paths(m,b,d,lags,samples=40000,generator=torch.Generator().manual_seed(812))[:,0]
        cov=predictive_covariance(m,b,d,lags)[0]
        self.assertLess((samples.mean(0)-m[0]).abs().max().item(),.022)
        for c in range(2):
            torch.testing.assert_close(torch.cov(samples[:,:,c].T),cov[c],rtol=.08,atol=.022)
    def test_head_mask_bound_initialization_and_restore(self):
        torch.manual_seed(1);head=DirectedGaussianHead(3,32);x=torch.randn(2,2,3)
        b,d=head(x);self.assertEqual(torch.count_nonzero(b).item(),0);torch.testing.assert_close(d,torch.ones_like(d))
        with torch.no_grad():head.edges.weight.normal_(0,20);head.edges.bias.normal_(0,20)
        b,d=head(x);self.assertTrue((b.abs().sum(-1)<.95).all())
        for j,k in enumerate(head.lags):self.assertEqual(torch.count_nonzero(b[:,:k,:,j]).item(),0)
        other=DirectedGaussianHead(3,32);other.load_state_dict(head.state_dict())
        for a,c in zip(head(x),other(x)):torch.testing.assert_close(a,c,rtol=0,atol=0)


class PDMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def pair(self,features='M'):
        cfg=config(seq_len=96,pred_len=32,moving_avg=25,e_layers=1,dropout=.2,down_sampling_layers=3,down_sampling_window=2)
        cfg.seasonal_normalize=False;cfg.features=features;cfg.c_out=1 if features=='MS' else 2
        torch.manual_seed(2021);point=PointModel(cfg);rng=torch.get_rng_state()
        torch.manual_seed(2021);prob=Model(cfg)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        return point,prob
    def test_exact_mean_common_state_normalization_and_single_encoding(self):
        point,prob=self.pair();base=point.state_dict();new=prob.state_dict()
        for key,value in base.items():torch.testing.assert_close(new[key],value,rtol=0,atol=0)
        self.assertTrue(all(k.startswith('probability_head.') for k in set(new)-set(base)))
        x=torch.randn(2,96,2)
        for training in [False,True]:
            point.train(training);prob.train(training);torch.manual_seed(42);expected=point(x);rng=torch.get_rng_state()
            torch.manual_seed(42);out=prob.forward_distribution(x)
            torch.testing.assert_close(out['mean'],expected,rtol=0,atol=0);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            torch.testing.assert_close(out['variance'],prob.normalize.stdev.square().expand_as(out['variance']))
            self.assertEqual(len(prob.pdm_blocks[-1]._forward_hooks),0)
        point.eval();prob.eval();torch.testing.assert_close(prob(x),point(x),rtol=0,atol=0)
        _,ms=self.pair('MS');self.assertEqual(ms.forward_distribution(x)['mean'].shape,(2,32,1))
    def test_frozen_and_joint_backward(self):
        _,prob=self.pair();prob.eval();x=torch.randn(2,96,2);y=torch.randn(2,32,2)
        for frozen in [True,False]:
            prob.zero_grad(set_to_none=True);out=prob.forward_distribution(x,detach_backbone=frozen)
            gaussian_nll(out['mean'],y,out['coefficients'],out['variance'],out['lags']).backward()
            for n,p in prob.named_parameters():
                if frozen and not n.startswith('probability_head.'):self.assertIsNone(p.grad,n)
                elif p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),n)
            self.assertGreater(prob.probability_head.edges.weight.grad.abs().sum().item(),0)
            self.assertGreater(prob.probability_head.variance.weight.grad.abs().sum().item(),0)
            if not frozen:self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in prob.named_parameters() if not n.startswith('probability_head.')))
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_loss_gradient_parity_and_pdm_update(self):
        m,y,b,d,lags=GaussianTests().parameters();m,y,b,d=[v.float() for v in (m,y,b,d)]
        results=[]
        for device in ['cpu','cuda']:
            inputs=[v.to(device).detach().requires_grad_() for v in (m,b,d)]
            loss=gaussian_nll(inputs[0],y.to(device),inputs[1],inputs[2],lags)
            results.append((loss.detach().cpu(),[v.cpu() for v in torch.autograd.grad(loss,inputs)]))
        torch.testing.assert_close(results[0][0],results[1][0],rtol=2e-6,atol=2e-6)
        for a,c in zip(results[0][1],results[1][1]):torch.testing.assert_close(a,c,rtol=2e-5,atol=2e-6)
        _,prob=self.pair();prob.cuda().eval();opt=torch.optim.Adam(prob.probability_head.parameters(),lr=.001)
        x=torch.randn(2,96,2,device='cuda');y=torch.randn(2,32,2,device='cuda')
        for _ in range(2):
            opt.zero_grad();out=prob.forward_distribution(x,detach_backbone=True)
            loss=gaussian_nll(out['mean'],y,out['coefficients'],out['variance'],out['lags']);loss.backward();opt.step()
            self.assertTrue(torch.isfinite(loss))

if __name__=='__main__':unittest.main()
