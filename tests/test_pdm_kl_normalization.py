import copy
import unittest
from types import SimpleNamespace
import torch
from scripts.pdm_kl_normalization_study import auxiliary,calibrate,gradient_diagnostic,calibrated_weights
from scripts.train_timemix_pdm_kl_normalization import loss_configuration,training_losses,initialize_pair,ARMS,dataset,selection_score,joint_stale_count
from test_oscillator_mixer import config

class KLNormalizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(2021)
        self.c=loss_configuration(SimpleNamespace(variance_floor=.01,kl_jitter=.01))
        self.c['kl_weight_by_arm']={'mse':0.,'correlation_kl':.2,'covariance_kl':.03}
    def test_initial_three_models_equal(self):
        cfg=config(seq_len=96,pred_len=96,moving_avg=25,dropout=.2);cfg.e_layers=2;cfg.seasonal_normalize=False
        models=initialize_pair(cfg);states=[m.state_dict() for m in models.values()]
        for s in states[1:]:
            for k,v in states[0].items():self.assertTrue(torch.equal(v,s[k]))
        x=torch.randn(1,96,cfg.enc_in)
        for mode in [True,False]:
            outputs=[]
            for model in models.values():
                model.train(mode);torch.manual_seed(7)
                with torch.no_grad():outputs.append(model(x))
            self.assertTrue(all(torch.equal(outputs[0],p) for p in outputs))
    def test_weighted_gradient_and_detach(self):
        for arm in ARMS:
            p=torch.randn(2,96,2,dtype=torch.double,requires_grad=True);h=torch.randn_like(p,requires_grad=True);y=torch.randn_like(p,requires_grad=True)
            actual=training_losses(p,y,arm,self.c,h);expected=(p-y.detach()).square().mean()
            if arm!='mse':expected=expected+self.c['kl_weight_by_arm'][arm]*auxiliary(h,p,y,arm,self.c)
            ga=torch.autograd.grad(actual['loss'],(p,h,y),retain_graph=True,allow_unused=True)
            ge=torch.autograd.grad(expected,p)[0]
            torch.testing.assert_close(ga[0],ge);self.assertIsNone(ga[1]);self.assertIsNone(ga[2])
    def test_calibration_restores_rng_mode_state_and_fixed_ratio(self):
        model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.2)).eval()
        data=[(torch.randn(96,2),torch.randn(96,2),None,None) for _ in range(40)]
        rng=torch.get_rng_state();before=copy.deepcopy(model.state_dict())
        a=calibrate(model,data,self.c,2021,'cpu')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for kind in ['correlation_kl','covariance_kl']:
            m=sum(q['mse_squared_norm'] for q in a['rows']);k=sum(q[kind]['kl_squared_norm'] for q in a['rows'])
            self.assertAlmostEqual(a['weights'][kind]*(k/m)**.5,.1,places=12)
        self.assertEqual(a,calibrate(model,data,self.c,2021,'cpu'))
    def test_diagnostics_do_not_change_final_gradient(self):
        m=torch.nn.Linear(2,2).double();x=torch.randn(2,96,2,dtype=torch.double);y=torch.randn_like(x);p=m(x)
        total=training_losses(p,y,'covariance_kl',self.c,x)['loss']
        expected=torch.autograd.grad(total,list(m.parameters()),retain_graph=True)
        rng=torch.get_rng_state();d=gradient_diagnostic(m,x,p,y,self.c)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(all(p.grad is None for p in m.parameters()))
        total.backward()
        for p,g in zip(m.parameters(),expected):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        self.assertEqual(set(d),{'correlation_kl','covariance_kl'})
    def test_invalid_calibration_and_validation_selection(self):
        with self.assertRaises(RuntimeError):calibrated_weights([{'mse_squared_norm':0}],.1)
        with self.assertRaises(ValueError):dataset(None,'test',0)
        self.assertEqual(selection_score({'neural_mse':.3}),.3)
        self.assertEqual(joint_stale_count(4,True),0);self.assertEqual(joint_stale_count(4,False),5)

if __name__=='__main__':unittest.main()
