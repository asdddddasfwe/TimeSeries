"""Actual MSE+D training gradients, optimizer policy and probe isolation."""
import unittest
import torch
from test_pdm_direct_patch import cfg
from scripts import train_timemix_pdm_state_patch_fusion as study

class FusionStudyTests(unittest.TestCase):
    def check_training(self,device):
        torch.set_num_threads(2);study.seed_all(2021)
        models=study.initialize_pair(cfg());x=torch.randn(2,96,7,device=device);y=torch.randn_like(x,requires_grad=True)
        for arm,m in models.items():
            m.to(device).train();groups=study.parameter_groups(m,arm);opt=study.build_optimizer(groups,arm)
            ids=[id(p) for g in groups for p in g['params']];self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in m.parameters()})
            wd={id(p):g['weight_decay'] for g in groups for p in g['params']}
            for n,p in m.named_parameters():
                if 'log_decay' in n or 'frequency_logits' in n or '.condition.' in n:self.assertEqual(wd[id(p)],0.,n)
            for step in range(2):
                opt.zero_grad(set_to_none=True);r=study.objective(m(x),y,x,arm,study.loss_configuration(None))
                self.assertTrue(torch.equal(r['loss'],r['mse']+r['relation_for_gradient']))
                params=tuple(m.parameters());g=torch.autograd.grad(r['loss'],params,retain_graph=True)
                rng=torch.get_rng_state().clone();probe=study.gradient_comparison(m,r,1.)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(all(p.grad is None for p in params))
                r['loss'].backward();self.assertIsNone(y.grad)
                for p,expected in zip(params,g):self.assertTrue(torch.equal(p.grad,expected));self.assertTrue(torch.isfinite(p.grad).all())
                opt.step()
    def test_cpu(self):self.check_training('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda(self):self.check_training('cuda')
if __name__=='__main__':unittest.main()
