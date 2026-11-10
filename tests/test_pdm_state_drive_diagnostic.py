"""Controlled drive-input tests; no forecasting data, fitting or official tests."""
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from models.OscillatorPDMStatePatchDriveDiagnostic import Model, phase_drives, driven_substates
from models.OscillatorPDMStatePatchReadoutDiagnostic import Model as Old, constant_drive_substates
from layers.OscillatorTimeMix import complex_oscillator_scan
import json
ROOT=Path(__file__).resolve().parents[1]


class DriveDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg=json.loads((ROOT/"tmp/mafirst_readout_substate_ettm1_fold0_h720_20261030_group/plan.json").read_text())["models"]["mafirst_season"]["config"]

    def build(self, policy):
        c=SimpleNamespace(**(self.cfg|{"state_drive_policy":policy,"state_readout_policy":"substate"}))
        torch.manual_seed(2021)
        return Model(c)

    def test_constructor_initial_state_parameters_and_rng(self):
        a=self.build("mean");after=torch.get_rng_state().clone()
        b=self.build("phase")
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in a.parameters()),345786)
        self.assertEqual((len(list(a.parameters())),len(list(a.buffers())),len(a.state_dict())),(212,32,244))
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        torch.manual_seed(2021)
        old=Old(SimpleNamespace(**self.cfg))
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,old.state_dict()[k]) for k,v in a.state_dict().items()))

    def test_identical_features_recover_drives_and_shared_dropout(self):
        m=self.build("mean");d=m.seasonals[0].decoder;d.train()
        h=torch.randn(2,8,d.drive_projection.in_features)
        x=torch.randn(2,5,1,h.shape[-1]).expand(-1,-1,4,-1)
        rng=torch.get_rng_state()
        a=phase_drives(d,x,h,"mean");after=torch.get_rng_state()
        torch.set_rng_state(rng)
        b=phase_drives(d,x,h,"phase")
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(b[:,:,0],b[:,:,3]))
        torch.set_rng_state(rng)
        legacy=d.future_drive(x[:,:,0],h).reshape(2,5,d.modes,2)
        self.assertTrue(torch.allclose(a[:,:,0],legacy,atol=2e-7,rtol=2e-6))

    def test_constant_drive_composition_states_and_gradients(self):
        r=torch.tensor([[.2,.8,.999]],dtype=torch.float64,requires_grad=True)
        f=torch.tensor([[.1,1.2,3.0]],dtype=torch.float64,requires_grad=True)
        t=torch.stack((r*f.cos(),r*f.sin()),-1)
        g=torch.randn(1,5,3,2,dtype=torch.float64,requires_grad=True)
        initial=torch.randn(1,3,2,dtype=torch.float64,requires_grad=True)
        ends=complex_oscillator_scan(t[:,None].expand(-1,5,-1,-1),g,initial)
        old,_=constant_drive_substates(t,r,f,g,initial,ends)
        states,rebuilt,new,effective=driven_substates(t,r,f,g[:,:,None].expand(-1,-1,4,-1,-1),initial)
        self.assertIs(states[-1],new)
        for x,y in zip(states,old):self.assertTrue(torch.allclose(x,y,rtol=1e-10,atol=1e-10))
        self.assertTrue(torch.allclose(rebuilt,new,rtol=1e-10,atol=1e-10))
        ga=torch.autograd.grad(sum(v.square().sum() for v in states),(r,f,g,initial),retain_graph=True)
        gb=torch.autograd.grad(sum(v.square().sum() for v in old),(r,f,g,initial))
        for x,y in zip(ga,gb):self.assertTrue(torch.allclose(x,y,rtol=1e-8,atol=1e-8))

    def test_zero_drive_uses_original_block_clock(self):
        r=torch.tensor([[.8]],dtype=torch.float64);f=torch.tensor([[.6]],dtype=torch.float64)
        t=torch.stack((r*f.cos(),r*f.sin()),-1)
        initial=torch.tensor([[[1.,0.]]],dtype=torch.float64)
        g=torch.zeros(1,1,4,1,2,dtype=torch.float64)
        states,_,end,_=driven_substates(t,r,f,g,initial)
        expected=complex_oscillator_scan(t[:,None],g[:,:,0],initial)
        self.assertTrue(torch.allclose(end,expected,atol=1e-12,rtol=1e-12))
        self.assertTrue(torch.allclose(states[0],torch.view_as_real(torch.polar(r.pow(.25),f*.25))[:,None]))

    def test_zero_sum_position_information_enters_drive_and_state(self):
        d=self.build("phase").seasonals[0].decoder;d.eval()
        width=d.drive_projection.in_features
        h=torch.randn(2,8,width)
        base=torch.randn(2,5,1,width).expand(-1,-1,4,-1).clone()
        z=torch.randn(2,5,width)
        changed=base.clone();changed[:,:,0]+=z;changed[:,:,1]-=z
        self.assertTrue(torch.allclose(base.mean(2),changed.mean(2),atol=3e-7))
        ma=phase_drives(d,base,h,"mean");mb=phase_drives(d,changed,h,"mean")
        pa=phase_drives(d,base,h,"phase");pb=phase_drives(d,changed,h,"phase")
        self.assertTrue(torch.allclose(ma,mb,atol=1e-6,rtol=1e-5))
        self.assertGreater(float((pa-pb).abs().max()),1e-4)
        r=torch.full((2,d.modes),.8);f=torch.full_like(r,.6)
        t=torch.stack((r*f.cos(),r*f.sin()),-1);initial=torch.zeros(2,d.modes,2)
        sa=driven_substates(t,r,f,pa,initial)[0]
        sb=driven_substates(t,r,f,pb,initial)[0]
        self.assertGreater(float((sa[0]-sb[0]).abs().max()),1e-4)

    def test_full_backbone_mean_matches_previous_substate_with_tolerance(self):
        m=self.build("mean");old=Old(SimpleNamespace(**self.cfg));old.load_state_dict(m.state_dict())
        m.eval();old.eval();x=torch.randn(2,96,7)
        a=m(x);b=old(x)
        self.assertTrue(torch.allclose(a,b,atol=3e-6,rtol=3e-5),float((a-b).abs().max()))

    def test_full_loss_all_gradients_finite_and_random_stream_matched(self):
        a=self.build("mean");b=self.build("phase")
        a.train();b.train();x=torch.randn(2,96,7);y=torch.randn(2,720,7)
        rng=torch.get_rng_state();pa=a(x);after=torch.get_rng_state()
        torch.set_rng_state(rng);pb=b(x)
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertFalse(torch.equal(pa,pb))
        from tmp.mafirst_fullbatch_mmd_recompute_20261028 import parts
        for model,pred in [(a,pa),(b,pb)]:
            before=torch.get_rng_state().clone()
            loss=(pred-y).square().mean()+parts(pred,y)[0];loss.backward()
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            for name,p in model.named_parameters():
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertEqual(list(model.initial_decomposition.kernels.values()),[9]*4)
            self.assertTrue(all(d["selected_endpoint_bitwise_exact"] and d["substate_reconstruction_relative"]<1e-5 for d in model.last_state_readout_diagnostics))

    def test_phase_order_retained_and_no_added_state(self):
        m=self.build("phase");d=m.seasonals[0].decoder;d.eval()
        h=torch.randn(1,8,d.drive_projection.in_features)
        x=torch.randn(1,3,4,h.shape[-1])
        a=phase_drives(d,x,h,"phase");b=phase_drives(d,x.flip(2),h,"phase")
        self.assertTrue(torch.equal(a.flip(2),b))
        self.assertFalse(any("correction" in k or "residual_head" in k for k in m.state_dict()))


if __name__=="__main__":unittest.main()
