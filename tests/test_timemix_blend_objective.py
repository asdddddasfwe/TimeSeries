"""验证融合目标梯度、固定线性边界、配对初始化及选轮口径。"""
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader, TensorDataset
from test_oscillator_mixer import config
from layers.LinearForecastBlend import LinearForecastBlend
from scripts.train_timemix_blend_objective import training_losses, initialize_pair, evaluate_pair, parameter_groups, dataset
from scripts.run_timemix_blend_objective_experiment import ADAPTERS, MODELS


class BlendObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2021)

    def test_actual_blend_gradient_and_detached_linear(self):
        neural=torch.randn(3,96,2,requires_grad=True)
        linear=torch.randn_like(neural,requires_grad=True)
        target=torch.randn_like(neural)
        loss,nm,bm=training_losses(neural,linear,target,'blend_loss')
        expected=.5*neural.detach()+.5*linear.detach()-target
        loss.backward()
        torch.testing.assert_close(neural.grad,expected/neural.numel())
        self.assertIsNone(linear.grad)
        torch.testing.assert_close(loss,bm,atol=0,rtol=0)
        self.assertNotEqual(float(nm.detach()),float(bm.detach()))
        with self.assertRaises(ValueError): training_losses(neural,linear,target,'unknown')

    def test_control_gradient_keeps_neural_objective(self):
        neural=torch.randn(2,96,2,requires_grad=True)
        linear=torch.randn_like(neural,requires_grad=True)
        target=torch.randn_like(neural)
        loss,_,_=training_losses(neural,linear,target,'control')
        loss.backward()
        torch.testing.assert_close(neural.grad,2*(neural.detach()-target)/neural.numel())
        self.assertIsNone(linear.grad)

    def test_linear_component_cannot_enter_optimizer(self):
        cfg=config(seq_len=96,pred_len=96,dropout=.2)
        models=initialize_pair(cfg)
        linear=LinearForecastBlend(torch.nn.Identity(),torch.randn(2,97,96)*.01)
        self.assertEqual(list(linear.parameters()),[])
        before=linear.linear_coefficients.clone()
        x,y=torch.randn(2,96,2),torch.randn(2,96,2)
        model=models['blend_loss']
        optimizer=torch.optim.Adam(parameter_groups(model,'blend_loss'),lr=.001)
        with torch.no_grad(): lp=linear.linear_forecast(x)
        loss,_,_=training_losses(model(x),lp,y,'blend_loss')
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        torch.testing.assert_close(before,linear.linear_coefficients,atol=0,rtol=0)
        self.assertIsNone(linear.linear_coefficients.grad)

    def test_exact_initial_weights_outputs_and_dropout_rng(self):
        models=initialize_pair(config(seq_len=96,pred_len=96,dropout=.2))
        x=torch.randn(2,96,2)
        for mode in (False,True):
            outputs,states=[],[]
            for model in models.values():
                model.train(mode)
                torch.manual_seed(55)
                outputs.append(model(x))
                states.append(torch.get_rng_state())
            torch.testing.assert_close(*outputs,atol=0,rtol=0)
            self.assertTrue(torch.equal(*states))
        for k,v in models['control'].state_dict().items():
            torch.testing.assert_close(v,models['blend_loss'].state_dict()[k],atol=0,rtol=0)

    def test_validation_retains_opposing_neural_and_blend_ranking(self):
        class Constant(torch.nn.Module):
            def __init__(self,value): super().__init__(); self.value=value
            def forward(self,x): return torch.full_like(x,self.value)
        class FixedLinear:
            def linear_forecast(self,x): return torch.full_like(x,2)
        x=torch.zeros(3,96,2)
        loader=DataLoader(TensorDataset(x,x,x,x),batch_size=2)
        first=evaluate_pair(Constant(0),FixedLinear(),loader,'cpu')
        second=evaluate_pair(Constant(-2),FixedLinear(),loader,'cpu')
        self.assertLess(first['neural_mse'],second['neural_mse'])
        self.assertGreater(first['blend_mse'],second['blend_mse'])
        self.assertEqual(second['blend_mse'],0)

    def test_identical_frozen_adapter_and_no_training_test_access(self):
        self.assertEqual(len(set(MODELS.values())),1)
        self.assertEqual(set(ADAPTERS.values()),{'OscillatorCombinedLinearBlend'})
        with self.assertRaises(ValueError): dataset(None,'test',0)


if __name__=='__main__': unittest.main()
