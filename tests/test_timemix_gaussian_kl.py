import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader, TensorDataset
from test_oscillator_mixer import config
from scripts.train_timemix_gaussian_kl import (
    training_losses, initialize_pair, parameter_groups, build_optimizer,
    auxiliary_weight, evaluate_validation, selection_score, dataset, loss_configuration,
)
from scripts.run_timemix_gaussian_kl_experiment import check_test_result
from utils.gaussian_autocov_loss import gaussian_autocov_kl_loss


class GaussianKLTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(2021)
        self.cfg=loss_configuration(SimpleNamespace(kl_weight=.001, kl_warmup_epochs=2.,
                                                   variance_floor=.01, kl_jitter=.01))

    def test_control_gradient_is_exact_pure_mse(self):
        p=torch.randn(2,96,2,requires_grad=True);y=torch.randn_like(p,requires_grad=True)
        result=training_losses(p,y,'control',1000,10,self.cfg);result['loss'].backward()
        torch.testing.assert_close(p.grad,2*(p.detach()-y.detach())/p.numel())
        self.assertIsNone(y.grad);self.assertEqual(result['kl_weight'],0.)

    def test_candidate_gradient_and_loss_include_kl(self):
        p=torch.randn(2,96,2,requires_grad=True);y=torch.randn_like(p)
        parts=training_losses(p,y,'gaussian_kl',20,10,self.cfg)
        expected=(p-y).square().mean()+.001*gaussian_autocov_kl_loss(p,y)
        actual_grad=torch.autograd.grad(parts['loss'],p,retain_graph=True)[0]
        expected_grad=torch.autograd.grad(expected,p)[0]
        torch.testing.assert_close(actual_grad,expected_grad)
        torch.testing.assert_close(parts['loss'].detach(),parts['mse'].detach()+parts['weighted_kl'])

    def test_schedule_is_update_based_and_resume_stateless(self):
        self.assertEqual(auxiliary_weight('gaussian_kl',0,10,self.cfg),0)
        self.assertEqual(auxiliary_weight('gaussian_kl',10,10,self.cfg),.0005)
        self.assertEqual(auxiliary_weight('gaussian_kl',20,10,self.cfg),.001)
        self.assertEqual(auxiliary_weight('gaussian_kl',999,10,self.cfg),.001)
        for u in range(30):self.assertEqual(auxiliary_weight('control',u,10,self.cfg),0)

    def test_initial_weights_predictions_rng_and_lr_match(self):
        models=initialize_pair(config(seq_len=96,pred_len=96,dropout=.2))
        x=torch.randn(2,96,2)
        for train_mode in (False,True):
            outputs=[];states=[]
            for model in models.values():
                model.train(train_mode);torch.manual_seed(55)
                outputs.append(model(x));states.append(torch.get_rng_state())
            torch.testing.assert_close(*outputs,atol=0,rtol=0)
            self.assertTrue(torch.equal(*states))
        for name,model in models.items():
            opt=build_optimizer(parameter_groups(model,name),name)
            self.assertEqual(opt.param_groups[0]['lr'],.001)

    def test_model_backward_reaches_all_parameters(self):
        model=initialize_pair(config(seq_len=96,pred_len=96,dropout=.2))['gaussian_kl']
        x,y=torch.randn(2,96,2),torch.randn(2,96,2)
        parts=training_losses(model(x),y,'gaussian_kl',20,10,self.cfg)
        parts['loss'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_validation_only_scores_neural_output(self):
        class Constant(torch.nn.Module):
            def forward(self,x):return torch.full_like(x,2)
        x=torch.zeros(3,96,2);loader=DataLoader(TensorDataset(x,x,x,x),batch_size=2)
        with patch('scripts.train_timemix_gaussian_kl.gaussian_autocov_kl_loss',side_effect=AssertionError('No validation KL')):
            result=evaluate_validation(Constant(),loader,'cpu')
        self.assertEqual(result,dict(neural_mse=4.,neural_mae=2.))
        self.assertEqual(selection_score(result),4.)
        with self.assertRaises(ValueError):dataset(None,'test',0)

    def test_receipts_reject_partial_or_mismatched_test(self):
        result=dict(status='completed',split='test',smoke=False,windows=11425,official_windows=11425,
                    checkpoint_sha256='ck',model_config_sha256='cfg',source_hashes_verified=True)
        receipt=dict(checkpoint_sha256='ck');manifest=dict(checkpoint_sha256='ck',model_config_sha256='cfg')
        check_test_result(result,receipt,manifest)
        for field,value in [('windows',32),('smoke',True),('model_config_sha256','bad')]:
            with self.assertRaises(RuntimeError):check_test_result(dict(result,**{field:value}),receipt,manifest)


if __name__=='__main__':unittest.main()
