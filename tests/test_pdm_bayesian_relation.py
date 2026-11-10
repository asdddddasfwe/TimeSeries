"""Bayesian AR posterior and Bregman score: independent numerical checks."""
import unittest
import numpy as np
import torch
from utils.bayesian_relation_bregman import posterior, normal_nodes, features, relation, objective


def numpy_features(x, q):
    z=x.transpose(0,2,1)
    f=np.stack([z[...,q-j:z.shape[-1]-j] for j in range(1,q+1)],-1)
    return z[...,q:],np.concatenate((f,np.ones_like(f[...,:1])),-1)


def numpy_score(p,y,x,q=4,nodes=None):
    d,z=numpy_features(x,q)
    precision=np.swapaxes(z,-1,-2)@z+np.eye(q+1)
    cov=np.linalg.solve(precision,np.broadcast_to(np.eye(q+1),precision.shape))
    mean=np.linalg.solve(precision,(np.swapaxes(z,-1,-2)@d[...,None]))[...,0]
    pd,pf=numpy_features(p,q);yd,yf=numpy_features(y,q)
    ef=pf-yf;delta=pd-yd-(ef*mean[...,None,:]).sum(-1)
    mq=.5*np.mean(delta**2)
    uq=.5*np.mean(np.einsum('bcrd,bcde,bcre->bcr',ef,cov,ef))
    nl=0.
    if nodes is not None:
        theta=mean[...,None,:]+np.einsum('bcij,sj->bcsi',np.linalg.cholesky(cov),nodes)
        rp=pd[...,None,:]-np.einsum('bcrd,bcsd->bcsr',pf,theta)
        ry=yd[...,None,:]-np.einsum('bcrd,bcsd->bcsr',yf,theta)
        nl=float(np.mean(np.logaddexp(ry,-ry)-np.logaddexp(rp,-rp)-np.tanh(rp)*(ry-rp)))
    return mean,cov,dict(mean_quadratic=float(mq),uncertainty_quadratic=float(uq),logcosh_divergence=nl)


class BayesianRelationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(2021)
        self.x=torch.randn(2,12,2,dtype=torch.double,requires_grad=True)
        self.p=torch.randn(2,12,2,dtype=torch.double,requires_grad=True)
        self.y=torch.randn_like(self.p,requires_grad=True)

    def test_numpy_posterior_scores_gradient_and_detach(self):
        post=posterior(self.x)
        m,v,parts=numpy_score(self.p.detach().numpy(),self.y.detach().numpy(),self.x.detach().numpy(),nodes=normal_nodes().numpy())
        np.testing.assert_allclose(post[0],m,atol=1e-12)
        np.testing.assert_allclose(post[1],v,atol=1e-12)
        for arm,keys in [('mean',['mean_quadratic']),('quadratic',['mean_quadratic','uncertainty_quadratic']),('bregman',list(parts))]:
            loss,_=relation(self.p,self.y,post,arm)
            self.assertAlmostEqual(float(loss.detach()),sum(parts[k] for k in keys),places=11)
        out=objective(self.p,self.y,self.x,'bregman',dict(weight=.1))
        g,gy,gx=torch.autograd.grad(out['loss'],(self.p,self.y,self.x),allow_unused=True)
        self.assertIsNone(gy);self.assertIsNone(gx);self.assertTrue(torch.isfinite(g).all())
        # Independent scatter of exact expected quadratic and sampled logcosh gradient.
        p=self.p.detach().numpy();y=self.y.detach().numpy();pd,pf=numpy_features(p,4);yd,yf=numpy_features(y,4)
        ed,ef=pd-yd,pf-yf;re=ed-(ef*m[...,None,:]).sum(-1);den=re.size
        expected=np.zeros_like(p.transpose(0,2,1));expected[...,4:]+=re/den
        for j in range(4):
            expected[...,3-j:11-j]+=(-m[...,j,None]*re+np.einsum('bcrd,bcd->bcr',ef,v[...,j,:]))/den
        theta=m[...,None,:]+np.einsum('bcij,sj->bcsi',np.linalg.cholesky(v),normal_nodes().numpy())
        rp=pd[...,None,:]-np.einsum('bcrd,bcsd->bcsr',pf,theta)
        ry=yd[...,None,:]-np.einsum('bcrd,bcsd->bcsr',yf,theta)
        gr=(1-np.tanh(rp)**2)*(rp-ry)/(den*theta.shape[-2])
        expected[...,4:]+=gr.sum(-2)
        for j in range(4):expected[...,3-j:11-j]-=(gr*theta[...,j,None]).sum(-2)
        np.testing.assert_allclose(g,2*(p-y)/p.size+.1*expected.transpose(0,2,1),atol=1e-11)
        self.assertTrue(torch.autograd.gradcheck(lambda z:relation(z,self.y,post,'bregman')[0],(self.p,)))

    def test_posterior_quadratic_exact_cubature_identity(self):
        post=posterior(self.x);m,v,chol=post
        # Spherical points reproduce Gaussian first two moments exactly.
        eps=torch.cat((torch.eye(5),-torch.eye(5))).double()*5**.5
        theta=m.unsqueeze(-2)+torch.einsum('bcij,sj->bcsi',chol,eps)
        pd,pf=features(self.p);yd,yf=features(self.y.detach())
        ed,ef=pd-yd,pf-yf
        self.assertTrue((ef[...,-1]==0).all())  # Intercept cancels in residual differences.
        sampled=.5*(ed.unsqueeze(-2)-torch.einsum('bcrd,bcsd->bcsr',ef,theta)).square().mean()
        exact,_=relation(self.p,self.y,post,'quadratic')
        torch.testing.assert_close(exact,sampled,atol=1e-12,rtol=1e-12)

    def test_population_mean_and_node_rng_independence(self):
        normal_nodes.cache_clear();before=torch.get_rng_state().clone();nodes=normal_nodes()
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(torch.equal(nodes[:32],-nodes[32:]))
        y0=self.y.detach();y1=torch.randn_like(y0);mean=(y0+y1)/2
        for arm in ('mean','quadratic','bregman'):
            p=mean.clone().requires_grad_()
            def loss(z,y):return objective(z,y,self.x,arm,dict(weight=.1))['loss']
            risk=(loss(p,y0)+loss(p,y1))/2
            self.assertLess(float(torch.autograd.grad(risk,p)[0].abs().max()),1e-12)
            trial=mean+.2*torch.randn_like(mean)
            gap=(loss(trial,y0)+loss(trial,y1))/2-risk.detach()
            torch.testing.assert_close(gap,loss(trial,mean),atol=1e-11,rtol=1e-10)

    def test_extremes_and_zero_error(self):
        for scale in (0.,1e-4,1.,1e3):
            x=self.x.detach()*scale;p=(self.p.detach()*scale).requires_grad_();y=self.y.detach()*scale
            post=posterior(x)
            for mode in ('mean','quadratic','bregman'):
                v,_=relation(p,y,post,mode);self.assertGreaterEqual(float(v.detach()),-1e-10)
                self.assertTrue(torch.isfinite(torch.autograd.grad(v,p)[0]).all())
                zero,_=relation(p,p.detach(),post,mode);self.assertEqual(float(zero.detach()),0.)

    def test_main_model_full_gradient_and_capture(self):
        from scripts import train_timemix_pdm_bayesian_relation as study
        from test_oscillator_mixer import config
        from utils.pdm_residual_ma_diagnostics import SelectedValidationCapture
        from types import SimpleNamespace
        from pathlib import Path
        import tempfile
        cfg=config(seq_len=96,pred_len=96,moving_avg=25,e_layers=1,dropout=.2,down_sampling_layers=3,down_sampling_window=2)
        cfg.seasonal_normalize=False;models=study.initialize_pair(cfg)
        lc=study.loss_configuration(SimpleNamespace(relation_weight=.1));self.assertEqual(lc['auxiliary_parameters'],0)
        x,y=torch.randn(2,96,2),torch.randn(2,96,2)
        with tempfile.TemporaryDirectory() as directory:
            for arm,model in models.items():
                for k,v in models['mse'].state_dict().items():self.assertTrue(torch.equal(v,model.state_dict()[k]))
                objective(model(x),y,x,arm,lc)['loss'].backward()
                for name,p in model.named_parameters():
                    self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
                model.eval()
                with torch.no_grad():
                    before=model(x).numpy()
                    with SelectedValidationCapture(model,Path(directory),arm,2) as cap:after=model(x).numpy()
                np.testing.assert_array_equal(before,after)
                self.assertLess(cap.report(after,y.numpy())['reconstruction_max_abs_error'],2e-5)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cpu_cuda_loss_gradient(self):
        pc=self.p.detach().float().requires_grad_();pg=pc.detach().cuda().requires_grad_()
        for arm in ('mean','quadratic','bregman'):
            a=objective(pc,self.y.float(),self.x.float(),arm,dict(weight=.1))['loss']
            b=objective(pg,self.y.float().cuda(),self.x.float().cuda(),arm,dict(weight=.1))['loss']
            torch.testing.assert_close(a,b.cpu(),atol=1e-6,rtol=1e-6)
            ga=torch.autograd.grad(a,pc)[0];gb=torch.autograd.grad(b,pg)[0].cpu()
            torch.testing.assert_close(ga,gb,atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
