import unittest
import numpy as np
import torch
from utils.patch_relation_kl_loss import parts, patch_log_probabilities
from scripts.audit_pdm_patch_relation_kl import numpy_relation_parts


class RelationKLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2021)

    def check_formula(self, device):
        p = torch.randn(2, 96, 3, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn_like(p, requires_grad=True)
        rng = torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state()
        loss, diag = parts(p, y)
        ref = numpy_relation_parts(p.detach().cpu().numpy(), y.detach().cpu().numpy())
        for k, v in ref.items(): self.assertAlmostEqual(diag[k], v, places=11)
        lp = patch_log_probabilities(p.transpose(1,2).unfold(-1,4,4))
        self.assertEqual(lp.shape, (2,3,24,23))
        torch.testing.assert_close(lp.exp().sum(-1), torch.ones_like(lp[...,0]))
        loss.backward()
        self.assertIsNone(y.grad)
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.norm()), 0.)
        current = torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state()
        self.assertTrue(torch.equal(rng, current))
        self.assertAlmostEqual(diag['relation_kl'], diag['relation_cross_entropy']-diag['target_relation_entropy'], places=12)

    def test_numpy_cpu(self): self.check_formula('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_numpy_cuda(self): self.check_formula('cuda')

    def test_gradcheck(self):
        p = torch.randn(1,12,1,dtype=torch.double,requires_grad=True)
        y = torch.randn_like(p)
        self.assertTrue(torch.autograd.gradcheck(lambda v: parts(v,y)[0], (p,), atol=1e-5,rtol=1e-4))

    def test_zero_and_invariances_and_alignment(self):
        p=torch.randn(2,96,2,dtype=torch.double)
        y=torch.randn_like(p)
        self.assertLess(float(parts(y,y)[0]),1e-13)
        self.assertLess(float(parts(y+3,y)[0]),1e-13)
        self.assertLess(float(parts(-y,y)[0]),1e-13)
        order=torch.randperm(24)
        perm=lambda v:v.reshape(2,24,4,2)[:,order].reshape(2,96,2)
        torch.testing.assert_close(parts(p,y)[0],parts(perm(p),perm(y))[0])
        self.assertGreater(float(parts(perm(y),y)[0]),1e-3)
        self.assertGreater(abs(float(parts(p,y)[0]-parts(y,p)[0])),1e-4)

    def test_isolation_and_exact_reduction(self):
        p=torch.randn(2,96,2,dtype=torch.double);y=torch.randn_like(p)
        pieces=[parts(p[b:b+1,:,c:c+1],y[b:b+1,:,c:c+1])[0] for b in range(2) for c in range(2)]
        torch.testing.assert_close(parts(p,y)[0],torch.stack(pieces).mean())

    def test_manual_three_patches(self):
        p=torch.tensor([[[0.],[1.],[3.]]],dtype=torch.double)
        y=torch.tensor([[[0.],[2.],[4.]]],dtype=torch.double)
        total=0.
        for i in range(3):
            ids=[j for j in range(3) if j!=i]
            pk=np.array([np.exp(-.5*float(p[0,i,0]-p[0,j,0])**2) for j in ids]);pk/=pk.sum()
            qk=np.array([np.exp(-.5*float(y[0,i,0]-y[0,j,0])**2) for j in ids]);qk/=qk.sum()
            total+=(qk*np.log(qk/pk)).sum()/3
        self.assertAlmostEqual(float(parts(p,y,length=1)[0]),total,places=12)

    def test_extreme_and_degenerate(self):
        p=(torch.randn(2,96,2)*1e4).requires_grad_();y=torch.randn_like(p)*1e4
        v,_=parts(p,y);v.backward()
        self.assertTrue(torch.isfinite(v));self.assertTrue(torch.isfinite(p.grad).all())
        z=torch.zeros(1,96,1)
        self.assertEqual(float(parts(z,z)[0]),0.)
        with self.assertRaises(ValueError):parts(z,z,bandwidth=0)
        with self.assertRaises(ValueError):parts(z[:,:95],z[:,:95])
        with self.assertRaises(ValueError):parts(z+float('nan'),z)

if __name__=='__main__': unittest.main()
