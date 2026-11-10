"""Paired PE-only intervention, objective, optimizer and harmless probes."""
import unittest
from unittest.mock import patch
import torch
from test_pdm_direct_patch import cfg
from scripts import train_timemix_pdm_state_patch_rope as study
from scripts import train_timemix_pdm_state_patch as baseline
from models import OscillatorPDMStatePatch as mean_module

class RotaryStudyTests(unittest.TestCase):
    def check_pair(self,device):
        torch.set_num_threads(2);study.seed_all(2021)
        models=study.initialize_pair(cfg());rng=torch.get_rng_state().clone()
        study.seed_all(2021);old=baseline.initialize_pair(cfg())['patch_covariance']
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for arm,m in models.items():
            self.assertEqual(sum(p.numel() for p in m.parameters()),136546)
            for k,v in old.state_dict().items():self.assertTrue(torch.equal(v,m.state_dict()[k]),k)
            self.assertTrue(all(e.future_positions.shape==(96,32) for e in m.seasonals))
        x=torch.randn(2,96,7,device=device);y=torch.randn_like(x,requires_grad=True)
        models={k:m.to(device) for k,m in models.items()};old=old.to(device)
        for training in [False,True]:
            outputs=[]
            for m in [*models.values(),old]:
                m.train(training);study.seed_all(123);outputs.append(m(x))
            self.assertTrue(torch.equal(outputs[0],outputs[2]))
            self.assertFalse(torch.equal(outputs[0],outputs[1]))
            self.assertTrue(all(v.shape==(2,96,7) and torch.isfinite(v).all() for v in outputs))
        m=models['rope'].eval();captures=[];scan=mean_module.complex_oscillator_scan
        def record(a,b,h):
            states=scan(a,b,h);captures.append((a.detach(),b.detach(),h.detach(),states.detach()));return states
        with patch.object(mean_module,'complex_oscillator_scan',record):m.predict_components(x)
        self.assertEqual(len(captures),4)
        for a,b,h,states in captures:
            self.assertEqual(states.shape,(14,24,8,2));serial=[];v=h
            for i in range(24):
                ar,ai=a[:,i].unbind(-1);vr,vi=v.unbind(-1)
                v=torch.stack((ar*vr-ai*vi,ar*vi+ai*vr),-1)+b[:,i];serial.append(v)
            torch.testing.assert_close(states,torch.stack(serial,1),rtol=2e-5,atol=2e-6)
        for arm,m in models.items():
            m.train();m.zero_grad(set_to_none=True);study.seed_all(456)
            # Activate existing zero gates to test finite nonzero Q/K gradients.
            with torch.no_grad():
                for e in m.seasonals:e.blocks[0].row.gamma.fill_(.1);e.blocks[0].column.gamma.fill_(.1)
            r=study.objective(m(x),y,x,arm,study.loss_configuration(None))
            self.assertTrue(torch.equal(r['loss'],r['mse']+r['relation_for_gradient']))
            before=torch.autograd.grad(r['loss'],tuple(m.parameters()),retain_graph=True)
            diag=study.gradient_comparison(m,r,1.)
            self.assertGreater(diag['history_attention']['mse_norm'],0.)
            self.assertTrue(all(p.grad is None for p in m.parameters()))
            r['loss'].backward();self.assertIsNone(y.grad)
            for (n,p),g in zip(m.named_parameters(),before):
                self.assertTrue(torch.equal(p.grad,g));self.assertTrue(torch.isfinite(g).all())
                if '.query_conv.weight' in n or '.key_conv.weight' in n:self.assertGreater(float(g.norm()),0.)
            groups=study.parameter_groups(m,arm);ids=[id(p) for g in groups for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in m.parameters()})

    def test_pair_cpu(self):self.check_pair('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_pair_cuda(self):self.check_pair('cuda')

if __name__=='__main__':unittest.main()
