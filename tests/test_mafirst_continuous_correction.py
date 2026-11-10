import unittest
import torch
from models.OscillatorPDMStatePatchMAFirstContinuousCorrection import correction_scan
from layers.OscillatorTimeMix import complex_oscillator_scan

class CorrectionTests(unittest.TestCase):
    def test_scan_and_gradients(self):
        torch.manual_seed(55)
        vals=[torch.randn(2,13,3,2,dtype=torch.float64)*.2,
              torch.randn(2,13,3,2,dtype=torch.float64),
              torch.randn(2,3,2,dtype=torch.float64),
              torch.randn(2,13,3,2,dtype=torch.float64),
              torch.rand(2,13,3,1,dtype=torch.float64)]
        vals=[v.requires_grad_() for v in vals]
        a,u,h,anchor,gate=vals
        actual=correction_scan(a,u,h,anchor,gate)
        states=[];prev=h
        for k in range(u.shape[1]):
            z=a[:,k];p=prev
            rotated=torch.stack((z[...,0]*p[...,0]-z[...,1]*p[...,1],
                                 z[...,1]*p[...,0]+z[...,0]*p[...,1]),-1)
            prev=(1-gate[:,k])*(rotated+u[:,k])+gate[:,k]*anchor[:,k]
            states.append(prev)
        expected=torch.stack(states,1)
        torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-12)
        ga=torch.autograd.grad(actual.square().sum(),vals,retain_graph=True)
        gb=torch.autograd.grad(expected.square().sum(),vals)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-11)

    def test_gate_endpoints_and_radius(self):
        torch.manual_seed(9)
        a=torch.randn(2,17,3,2)*.2;u=torch.randn_like(a);h=torch.randn(2,3,2)
        anchor=torch.randn_like(u);zero=torch.zeros(2,17,3,1)
        self.assertTrue(torch.equal(correction_scan(a,u,h,anchor,zero),complex_oscillator_scan(a,u,h)))
        torch.testing.assert_close(correction_scan(a,u,h,anchor,zero+1),anchor,rtol=0,atol=0)
        gate=torch.rand_like(zero)
        self.assertTrue(torch.all(((1-gate)*a).norm(dim=-1)<=a.norm(dim=-1)+1e-7))

    def test_full_models_and_original_initial(self):
        from tmp.run_mafirst_continuous_correction_20261020 import fresh,optimizer_groups
        for arm in ('centered_recurrence','gated_anchor'):
            model,cfg=fresh(720,arm,'/tmp/mafirst_continuous_correction_unit_logs')
            model.eval();x=torch.randn(2,96,7)
            parts=model.predict_components(x)
            y=model.normalize(parts[...,:4].sum(-1)+parts[...,4:].sum(-1),'denorm')
            self.assertEqual(tuple(y.shape),(2,720,7));self.assertTrue(torch.isfinite(y).all())
            if arm=='centered_recurrence':
                self.assertLess(float(parts[...,:4].mean(1).abs().max()),2e-6)
                self.assertEqual(sum(v.numel() for v in model.parameters()),320450)
            else:
                for gate in model.last_gates:torch.testing.assert_close(gate,torch.full_like(gate,.05))
                self.assertEqual(sum(v.numel() for v in model.parameters()),323618)
            grads=torch.autograd.grad(y.square().mean(),list(model.parameters()))
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            optimizer_groups(model)

    def test_centering_jacobian(self):
        raw=torch.randn(2,720,7,4,dtype=torch.float64,requires_grad=True)
        centered=raw-raw.mean(1,keepdim=True)
        self.assertLess(float(centered.mean(1).abs().max()),1e-14)
        g=torch.autograd.grad(centered.sum(),raw)[0]
        self.assertLess(float(g.abs().max()),1e-14)
if __name__=='__main__':unittest.main()
