import unittest
import torch
from models.OscillatorPDMStatePatchSpectralCutoff import SoftSpectralSplit

torch.set_num_threads(2)
class SpectralSplitTests(unittest.TestCase):
    def test_reconstruction_all_scales(self):
        split=SoftSpectralSplit()
        for scale,length in enumerate([96,48,24,12]):
            x=torch.randn(6,length,1,dtype=torch.double)
            s,t,m=split(x,torch.full((6,1),.04,dtype=torch.double),2**scale)
            torch.testing.assert_close(s+t,x,atol=1e-14,rtol=0)
            self.assertTrue(torch.isfinite(m).all())

    def test_dc_entirely_in_trend(self):
        x=torch.full((3,96,1),2.3,dtype=torch.double)
        s,t,m=SoftSpectralSplit()(x,torch.tensor([[.02],[.04],[.06]],dtype=torch.double))
        torch.testing.assert_close(s,torch.zeros_like(s),atol=1e-14,rtol=0)
        torch.testing.assert_close(t,x,atol=1e-14,rtol=0)
        self.assertTrue(torch.equal(m[:,0],torch.ones(3,dtype=torch.double)))

    def test_offset_equivariance(self):
        x=torch.randn(2,24,1,dtype=torch.double);c=torch.full((2,1),.04,dtype=torch.double);split=SoftSpectralSplit()
        s,t,_=split(x,c,4);ss,tt,_=split(x+5,c,4)
        torch.testing.assert_close(ss,s,atol=1e-14,rtol=0)
        torch.testing.assert_close(tt,t+5,atol=1e-14,rtol=0)

    def test_mask_monotonic_and_dynamic_per_sample(self):
        x=torch.randn(2,96,1,dtype=torch.double)
        _,_,m=SoftSpectralSplit()(x,torch.tensor([[.02],[.06]],dtype=torch.double))
        self.assertTrue((m[:,1:]<=m[:,:-1]).all())
        self.assertTrue((m[1,1:]>m[0,1:]).all())

    def test_physical_frequency_units(self):
        x=torch.randn(1,24,1,dtype=torch.double);c=torch.tensor([[.04]],dtype=torch.double);split=SoftSpectralSplit()
        _,_,mask=split(x,c,4)
        f=torch.fft.rfftfreq(70,d=4,dtype=torch.double)
        torch.testing.assert_close(mask[:,1:],torch.sigmoid((c-f[None,1:])/(1/96)))

    def test_frequency_selectivity(self):
        n=torch.arange(96,dtype=torch.double)
        lo=torch.cos(torch.pi*n/95)[None,:,None];hi=torch.cos(32*torch.pi*n/95)[None,:,None]
        split=SoftSpectralSplit();c=torch.tensor([[.04]],dtype=torch.double)
        _,tl,_=split(lo,c);_,th,_=split(hi,c)
        self.assertGreater(float(tl.square().mean()/lo.square().mean()),.7)
        self.assertLess(float(th.square().mean()/hi.square().mean()),.02)

    def test_differentiable_cutoff_and_input(self):
        x=torch.randn(1,12,1,dtype=torch.double,requires_grad=True)
        c=torch.tensor([[.04]],dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:SoftSpectralSplit()(a,b,8)[1],(x,c),eps=1e-6,atol=1e-5,rtol=1e-4))

    def test_no_cross_sample_or_channel_leak(self):
        x=torch.randn(4,24,1,dtype=torch.double);c=torch.full((4,1),.04,dtype=torch.double);split=SoftSpectralSplit()
        s,t,_=split(x,c,4);xx=x.clone();xx[0]+=torch.randn_like(xx[0]);ss,tt,_=split(xx,c,4)
        self.assertTrue(torch.equal(s[1:],ss[1:]) and torch.equal(t[1:],tt[1:]))

if __name__=='__main__':unittest.main()