"""CPU/CUDA semantic tests for removing conditioning and independent trend recurrence."""
import unittest
from unittest.mock import patch
import torch
from test_pdm_direct_patch import cfg
from scripts import train_timemix_pdm_state_patch_disconnected as study
from scripts import train_timemix_pdm_state_patch as baseline
from models import OscillatorPDMStatePatch as mean_module

class DisconnectedTests(unittest.TestCase):
    def check_models(self,device):
        torch.set_num_threads(2);study.seed_all(2021)
        models=study.initialize_pair(cfg());rng=torch.get_rng_state().clone()
        study.seed_all(2021);old=baseline.initialize_pair(cfg())['mse']
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        print('PARAMETER_COUNTS', {a:sum(p.numel() for p in m.parameters()) for a,m in models.items()})
        for m in models.values():
            self.assertFalse(any('trend_condition' in k for k in m.state_dict()))
            for k,v in m.state_dict().items():
                if k in old.state_dict():self.assertTrue(torch.equal(v,old.state_dict()[k]),k)
        dual=models['dual']
        self.assertTrue({p.data_ptr() for p in dual.seasonals.parameters()}.isdisjoint({p.data_ptr() for p in dual.feature_trend_heads.parameters()}))
        self.assertFalse(any('blocks' in k or 'value_embedding' in k for k in dual.feature_trend_heads.state_dict()))
        x=torch.randn(2,96,7,device=device)
        models={a:m.to(device) for a,m in models.items()};old=old.to(device)
        for training in (False,True):
            outputs=[]
            for m in [old,*models.values()]:
                study.seed_all(100);m.train(training);outputs.append(m(x))
            for p in outputs[1:]:self.assertTrue(torch.equal(p,outputs[0]))
        for m in models.values():
            m.eval()
            e=m.seasonals[0];h=torch.randn(14,e.length,32,device=device,requires_grad=True)
            t=torch.randn_like(h,requires_grad=True)
            a,initial,_,_=e.decoder.dynamics(h,t)
            b,initial_b,_,_=e.decoder.dynamics(h,t*100+123)
            self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(initial,initial_b))
            g=torch.autograd.grad(a.sum()+initial.sum(),(h,t),allow_unused=True)
            self.assertIsNone(g[1]);self.assertTrue(torch.isfinite(g[0]).all())
        for arm,m in models.items():
            captures=[];scan=mean_module.complex_oscillator_scan
            def record(a,b,h):
                result=scan(a,b,h);captures.append((a.detach(),b.detach(),h.detach(),result.detach()));return result
            with patch.object(mean_module,'complex_oscillator_scan',record):components=m.predict_components(x)
            self.assertEqual(components.shape,(2,96,7,8));self.assertEqual(len(captures),4 if arm=='disconnected' else 8)
            for a,b,h,result in captures:
                self.assertEqual(result.shape,(14,24,8,2));serial=[];v=h
                for i in range(24):
                    ar,ai=a[:,i].unbind(-1);vr,vi=v.unbind(-1)
                    v=torch.stack((ar*vr-ai*vi,ar*vi+ai*vr),-1)+b[:,i];serial.append(v)
                torch.testing.assert_close(result,torch.stack(serial,1),rtol=2e-5,atol=2e-6)
            self.assertTrue(torch.equal(components[...,4:],torch.zeros_like(components[...,4:])))
    def test_cpu(self):self.check_models('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda(self):self.check_models('cuda')

if __name__=='__main__':unittest.main()
