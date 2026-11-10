"""Joint patch readout semantics, function-preserving initialization and gradients."""
import copy
import unittest
import torch
from test_oscillator_mixer import config
from layers.OscillatorGatedMLPMixer import GELUMLPReadout
from models.OscillatorPDMJointPatch import JointPatchReadout
from scripts import train_timemix_pdm_joint_patch as study
from scripts.train_timemix_pdm_mixing_depth import initialize_pair as old_pair


def cfg():
    return config(seq_len=96,pred_len=96,enc_in=7,c_out=7,moving_avg=25,
        d_model=32,d_ff=64,ssm_modes=8,top_k=5,dropout=.2,e_layers=1,
        down_sampling_layers=3,down_sampling_window=2,seasonal_normalize=False)


class JointPatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2021)

    def test_head_rng_zero_and_joint_locality(self):
        pointwise=torch.nn.Linear(16,1)
        with torch.no_grad():pointwise.weight.zero_();pointwise.bias.zero_()
        rng=torch.get_rng_state().clone();head=JointPatchReadout(pointwise)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        x=torch.randn(2,8,16,requires_grad=True)
        self.assertTrue(torch.equal(head(x),pointwise(x)))
        with torch.no_grad():head.joint_projection.weight[0,16+3]=2.
        out=head(x);g=torch.autograd.grad(out[0,0,0],x)[0]
        self.assertEqual(float(g[0,1,3]),2.)
        self.assertEqual(int(torch.count_nonzero(g)),1)
        self.assertEqual(float(out[0,0,0].detach()),float(2*x[0,1,3].detach()))
        self.assertEqual(float(out[0,4,0].detach()),float(2*x[0,5,3].detach()))
        for shape in [(2,7,16),(2,8,15),(8,16)]:
            with self.assertRaises(ValueError):head(torch.zeros(shape))

    def check_model(self,device):
        c=cfg();study.seed_all(2021);old=old_pair(c)['depth1']
        study.seed_all(2021);models=study.initialize_pair(c)
        self.assertEqual(sum(p.numel() for p in old.parameters()),136150)
        for model in models.values():
            self.assertEqual(sum(p.numel() for p in model.parameters()),137190)
            for k,v in old.state_dict().items():
                self.assertTrue(torch.equal(v,model.state_dict()[k.replace('.decoder.readout.','.decoder.readout.pointwise.')]),k)
        x=torch.randn(2,96,7,device=device);y=torch.randn_like(x)
        old=old.to(device);models={k:v.to(device) for k,v in models.items()}
        for train in [False,True]:
            outputs=[];rngs=[]
            for model in [old,*models.values()]:
                model.train(train);study.seed_all(123)
                outputs.append(model(x));rngs.append((torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone() if device=='cuda' else None))
            for output,rng in zip(outputs[1:],rngs[1:]):
                self.assertTrue(torch.equal(outputs[0],output))
                self.assertTrue(torch.equal(rngs[0][0],rng[0]))
                if device=='cuda':self.assertTrue(torch.equal(rngs[0][1],rng[1]))
        model=models['patch_covariance'];model.zero_grad(set_to_none=True)
        r=study.objective(model(x),y,x,'patch_covariance',study.loss_configuration(None))
        expected=torch.autograd.grad(r['loss'],tuple(model.parameters()),retain_graph=True)
        probe=study.gradient_comparison(model,r,1.)
        self.assertEqual(probe['joint_patch_heads']['parameter_tensors'],8)
        self.assertGreater(probe['joint_patch_heads']['mse_norm'],0.)
        self.assertGreater(probe['joint_patch_heads']['auxiliary_norm'],0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        r['loss'].backward()
        for (n,p),g in zip(model.named_parameters(),expected):
            self.assertTrue(torch.equal(p.grad,g),n)
            self.assertTrue(torch.isfinite(p.grad).all(),n)
            if '.joint_projection.weight' in n:self.assertGreater(float(p.grad.norm()),0.,n)
        groups=study.parameter_groups(model,'patch_covariance')
        ids=[id(p) for group in groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        wd={id(p):group['weight_decay'] for group in groups for p in group['params']}
        for n,p in model.named_parameters():
            if '.joint_projection.' in n:self.assertEqual(wd[id(p)],0. if n.endswith('.bias') else .0001)

    def test_model_cpu(self):self.check_model('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_model_cuda(self):self.check_model('cuda')

    def test_pure_mse_and_joint_stop(self):
        p=torch.randn(2,96,7,requires_grad=True);y=torch.randn_like(p,requires_grad=True)
        r=study.objective(p,y,p.detach(),'mse',study.loss_configuration(None))
        self.assertTrue(torch.equal(r['loss'],(p-y.detach()).square().mean()))
        self.assertIsNone(torch.autograd.grad(r['loss'],y,allow_unused=True)[0])
        stale=0
        for improve,want in [(True,0),(False,1),(True,0),(False,1),(False,2),(False,3),(False,4),(False,5)]:
            stale=study.joint_stale_count(stale,improve);self.assertEqual(stale,want)

if __name__=='__main__':unittest.main()
