"""Single-state four-value readout, original recurrence and paired training."""
import unittest
from unittest.mock import patch
import torch
from test_oscillator_mixer import config
from models import OscillatorPDMStatePatch as model_module
from scripts import train_timemix_pdm_state_patch as study
from scripts.train_timemix_pdm_mixing_depth import initialize_pair as old_pair


def cfg():
    return config(seq_len=96,pred_len=96,enc_in=7,c_out=7,moving_avg=25,
        d_model=32,d_ff=64,ssm_modes=8,top_k=5,dropout=.2,e_layers=1,
        down_sampling_layers=3,down_sampling_window=2,seasonal_normalize=False)


class StatePatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(2021)

    def test_single_state_four_independent_outputs(self):
        model=study.initialize_pair(cfg())['mse']
        head=model.seasonals[0].decoder.readout
        x=torch.randn(2,24,16,requires_grad=True)
        with torch.no_grad():
            head.output_projection.weight.zero_()
            head.output_projection.weight[:,0]=torch.arange(1.,5.)
            head.output_projection.bias.copy_(torch.arange(4.))
        y=head(x);self.assertEqual(y.shape,(2,24,4))
        for j in range(4):
            g=torch.autograd.grad(y[0,3,j],x,retain_graph=True)[0]
            self.assertGreater(float(g[0,3].norm()),0.)
            g[0,3]=0;self.assertEqual(int(torch.count_nonzero(g)),0)
        self.assertEqual(len(torch.unique(y[0,3])),4)

    def check_model(self,device):
        c=cfg();study.seed_all(2021);old=old_pair(c)['depth1']
        study.seed_all(2021);models=study.initialize_pair(c)
        expected=model_module.convert_pointwise_state(old.state_dict())
        for model in models.values():
            self.assertEqual(sum(p.numel() for p in model.parameters()),136546)
            for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,expected[k]),k)
        x=torch.randn(2,96,7,device=device);target=torch.randn_like(x)
        models={k:v.to(device) for k,v in models.items()}
        for train in (False,True):
            outputs=[];rngs=[]
            for model in models.values():
                model.train(train);study.seed_all(123);outputs.append(model(x))
                rngs.append((torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone() if device=='cuda' else None))
            self.assertEqual(outputs[0].shape,(2,96,7));self.assertTrue(torch.equal(*outputs))
            self.assertTrue(torch.equal(rngs[0][0],rngs[1][0]))
            if device=='cuda':self.assertTrue(torch.equal(rngs[0][1],rngs[1][1]))
        model=models['patch_covariance'];model.eval()
        captures=[];scan=model_module.complex_oscillator_scan
        def recording(a,b,h):
            result=scan(a,b,h);captures.append((a.detach(),b.detach(),h.detach(),result.detach()))
            return result
        with patch.object(model_module,'complex_oscillator_scan',recording):components=model.predict_components(x)
        self.assertEqual(components.shape,(2,96,7,8));self.assertEqual(len(captures),4)
        scales,(seasons,trends)=model.encode_multiscale(model.downsample(x));alpha,beta=model.branch_gains()
        for enc,history,trend,(a,b,h,states) in zip(model.seasonals,seasons,trends,captures):
            self.assertEqual(states.shape,(14,24,8,2));self.assertEqual(enc.decoder.horizon,24)
            original_a,original_h,_,_=enc.decoder.dynamics(history,trend)
            self.assertTrue(torch.equal(a,original_a[:,None].expand_as(a)))
            self.assertTrue(torch.equal(h,original_h*alpha.repeat(2)[:,None,None]))
            features=enc.temporal_projection(history.transpose(1,2)).transpose(1,2)+enc.future_positions.to(history)
            pooled=torch.stack([features[:,i:i+4].mean(1) for i in range(0,96,4)],1)
            drive=enc.decoder.future_drive(pooled,history).reshape(14,24,8,2)*beta.repeat(2)[:,None,None,None]
            self.assertTrue(torch.equal(b,drive))
            state=h;serial=[]
            for t in range(24):
                ar,ai=a[:,t].unbind(-1);hr,hi=state.unbind(-1)
                state=torch.stack((ar*hr-ai*hi,ar*hi+ai*hr),-1)+b[:,t];serial.append(state)
            torch.testing.assert_close(states,torch.stack(serial,1),rtol=2e-5,atol=2e-6)
        model.train();model.zero_grad(set_to_none=True)
        r=study.objective(model(x),target,x,'patch_covariance',study.loss_configuration(None))
        gradients=torch.autograd.grad(r['loss'],tuple(model.parameters()),retain_graph=True)
        probe=study.gradient_comparison(model,r,1.)
        self.assertEqual(probe['state_patch_heads']['parameter_tensors'],8)
        self.assertGreater(probe['state_patch_heads']['mse_norm'],0.)
        self.assertGreater(probe['state_patch_heads']['auxiliary_norm'],0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        r['loss'].backward()
        for (n,p),g in zip(model.named_parameters(),gradients):
            self.assertTrue(torch.equal(p.grad,g),n);self.assertTrue(torch.isfinite(p.grad).all(),n)
            if '.readout.output_projection.weight' in n:
                self.assertGreater(float(p.grad.norm()),0.)
                self.assertFalse(torch.equal(p.grad[0],p.grad[1]))
        groups=study.parameter_groups(model,'patch_covariance')
        ids=[id(p) for group in groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        wd={id(p):group['weight_decay'] for group in groups for p in group['params']}
        for n,p in model.named_parameters():
            if '.readout.output_projection.' in n:self.assertEqual(wd[id(p)],0. if n.endswith('.bias') else .0001)

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
