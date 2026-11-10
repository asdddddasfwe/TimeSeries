"""Three-arm initial functions, optimizer and same-forward gradient probes."""
import unittest
import torch
from test_pdm_direct_patch import cfg
from scripts import train_timemix_pdm_patch_cka_lag as study
from scripts import train_timemix_pdm_state_patch as mean_study


class PatchCKALagStudyTests(unittest.TestCase):
    def check_pair(self, device):
        torch.set_num_threads(2);study.seed_all(2021)
        models = study.initialize_pair(cfg())
        study.seed_all(2021)
        reference = mean_study.initialize_pair(cfg())['mse'].to(device)
        models = {k:v.to(device) for k,v in models.items()}
        for model in models.values():
            for key,value in reference.state_dict().items():
                self.assertTrue(torch.equal(value,model.state_dict()[key]),key)
            self.assertEqual(sum(p.numel() for p in model.parameters()),136546)
        x = torch.randn(2,96,7,device=device)
        target = torch.randn_like(x,requires_grad=True)
        for training in (False,True):
            outputs,rngs=[],[]
            for model in [*models.values(),reference]:
                model.train(training);study.seed_all(123)
                outputs.append(model(x))
                rngs.append(torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state())
            self.assertTrue(all(torch.equal(outputs[0],v) for v in outputs))
            self.assertTrue(all(torch.equal(rngs[0],v) for v in rngs))
        for arm in ('cka','patch_lag'):
            model=models[arm]
            def decay_map(model,groups):
                by_id={id(p):g['weight_decay'] for g in groups for p in g['params']}
                return {k:by_id[id(p)] for k,p in model.named_parameters()}
            self.assertEqual(decay_map(model,study.parameter_groups(model,arm)),
                             decay_map(reference,mean_study.parameter_groups(reference,'mse')))
            result=study.objective(model(x),target,x,arm,study.loss_configuration(None))
            torch.testing.assert_close(result['loss'],result['mse']+result['relation_for_gradient'])
            parameters=tuple(model.parameters())
            before=torch.autograd.grad(result['loss'],parameters,retain_graph=True)
            diag=study.gradient_comparison(model,result,1.)
            self.assertEqual(diag['state_patch_heads']['parameter_tensors'],8)
            self.assertGreater(diag['state_patch_heads']['auxiliary_norm'],0.)
            self.assertTrue(all(p.grad is None for p in parameters))
            result['loss'].backward();self.assertIsNone(target.grad)
            for parameter,expected in zip(parameters,before):
                self.assertTrue(torch.equal(parameter.grad,expected))
                self.assertTrue(torch.isfinite(expected).all())
        prediction=models['mse'](x)
        pure=study.objective(prediction,target,x,'mse',study.loss_configuration(None))
        self.assertTrue(torch.equal(pure['loss'],(prediction-target.detach()).square().mean()))
        self.assertEqual(study.joint_stale_count(4,True),0)
        self.assertEqual(study.joint_stale_count(4,False),5)

    def test_pair_cpu(self):self.check_pair('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_pair_cuda(self):self.check_pair('cuda')


if __name__=='__main__':unittest.main()
