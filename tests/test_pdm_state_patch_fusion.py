"""Independent PDM fusion equation, cross-stream gradients and A24 semantics."""
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from test_pdm_direct_patch import cfg
from scripts import train_timemix_pdm_state_patch_fusion as study
from scripts import train_timemix_pdm_state_patch_disconnected as old_study
from models import OscillatorPDMStatePatch as mean_module


class FusionTests(unittest.TestCase):
    def check_models(self,device):
        torch.set_num_threads(2);study.seed_all(2021)
        models=study.initialize_pair(cfg());rng=torch.get_rng_state().clone()
        old_study.seed_all(2021);old=old_study.initialize_pair(cfg())['disconnected']
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for arm,m in models.items():
            self.assertEqual(sum(p.numel() for p in m.parameters()),132450)
            self.assertEqual(set(m.state_dict()),set(old.state_dict()))
            self.assertFalse(any('trend_condition' in k or 'trend_context_norm' in k for k in m.state_dict()))
            for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,old.state_dict()[k]),k)
        models={a:m.to(device) for a,m in models.items()};old.to(device)
        x=torch.randn(2,96,7,device=device)
        for training in (False,True):
            outputs=[];states=[]
            for m in [old,*models.values()]:
                study.seed_all(100);m.train(training);outputs.append(m(x))
                states.append(torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state())
            self.assertTrue(torch.equal(outputs[0],outputs[1]))
            self.assertGreater(float((outputs[2]-outputs[1]).abs().max()),1e-6)
            self.assertTrue(all(torch.equal(states[0],v) for v in states))
        for m in models.values():
            m.eval();d=m.seasonals[0].decoder
            h=torch.randn(2,96,32,device=device,requires_grad=True);t=torch.randn_like(h,requires_grad=True)
            a,hi,*_=d.dynamics(h,t);b,hj,*_=d.dynamics(h,t*100+12)
            self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(hi,hj))
            self.assertIsNone(torch.autograd.grad(a.sum()+hi.sum(),t,allow_unused=True)[0])
        seasons=[torch.randn(2,n,32,device=device,requires_grad=True) for n in [96,48,24,12]]
        trends=[torch.randn_like(s,requires_grad=True) for s in seasons]
        block=models['fusion'].pdm_blocks[0]
        actual_s,actual_t=block(seasons,trends)
        ms=block.mixing_multi_scale_season([s.transpose(1,2) for s in seasons])
        mt=block.mixing_multi_scale_trend([t.transpose(1,2) for t in trends])
        for s,t,a,b,u,v,k in zip(seasons,trends,ms,mt,actual_s,actual_t,[25,13,7,3]):
            merged=s+t+block.out_cross_layer(a+b)
            avg=F.avg_pool1d(F.pad(merged.transpose(1,2),(k//2,k//2),mode='replicate'),k,stride=1).transpose(1,2)
            torch.testing.assert_close(v,avg,rtol=1e-6,atol=1e-6)
            torch.testing.assert_close(u,merged-avg,rtol=1e-6,atol=1e-6)
            torch.testing.assert_close(u+v,merged,rtol=1e-6,atol=1e-6)
        g=torch.autograd.grad(actual_s[0].square().mean(),trends,retain_graph=True,allow_unused=True)
        self.assertGreater(sum(float(v.norm()) for v in g if v is not None),0.)
        separated=models['disconnected'].pdm_blocks[0](seasons,trends)[0]
        self.assertTrue(all(v is None for v in torch.autograd.grad(separated[0].square().mean(),trends,allow_unused=True)))
        for arm,m in models.items():
            captures=[];scan=mean_module.complex_oscillator_scan
            def record(a,b,h):
                result=scan(a,b,h);captures.append((a.detach(),b.detach(),h.detach(),result.detach()));return result
            with patch.object(mean_module,'complex_oscillator_scan',record):parts=m.predict_components(x)
            self.assertEqual(parts.shape,(2,96,7,8));self.assertEqual(len(captures),4)
            for a,b,h,result in captures:
                self.assertEqual(result.shape,(14,24,8,2));serial=[];v=h
                for i in range(24):
                    ar,ai=a[:,i].unbind(-1);vr,vi=v.unbind(-1)
                    v=torch.stack((ar*vr-ai*vi,ar*vi+ai*vr),-1)+b[:,i];serial.append(v)
                torch.testing.assert_close(result,torch.stack(serial,1),rtol=2e-5,atol=2e-6)
            self.assertEqual(torch.count_nonzero(parts[...,4:]),0)
    def test_cpu(self):self.check_models('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda(self):self.check_models('cuda')

if __name__=='__main__':unittest.main()
