"""Independent kernel/gradient/injectivity/protocol tests before freeze."""
import unittest,tempfile,json
from pathlib import Path
import numpy as np,torch
from utils.amplitude_weighted_patch_kernel_20261025 import *
from tmp.run_mafirst_h720_ampkernel_20261025 import R,G,PLAN,read,fixed_data,data_for,hook_calls,small_passed,groups_audit,batches,kernel_scores
from tmp.mafirst_h720_ampkernel_20261025_factory import fresh,optimizer_groups
from tmp.evaluate_mafirst_h720_ampkernel_20261025 import evaluate

def numpy_k(a):
    b,h,c=a.shape;q=a.astype(float).transpose(0,2,1).reshape(b*c,h//4,4);sp=np.logaddexp(0,q);base=np.exp(-.5*((q[:,:,None]-q[:,None,:])**2).sum(-1));return np.stack([base*sp[:,:,r,None]*sp[:,None,:,r] for r in range(4)],1)
class AmpKernelTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(47)
    def test01_independent_numpy_kernel(self):
        x=torch.randn(2,12,2);np.testing.assert_allclose(kernel_matrices(x).numpy(),numpy_k(x.numpy()),atol=1e-12,rtol=1e-12)
    def test02_PSD_symmetric_and_nonunit_diagonal(self):
        x=torch.randn(1,40,2,dtype=torch.double);k=kernel_matrices(x).numpy();np.testing.assert_allclose(k,k.swapaxes(-1,-2),atol=1e-12);self.assertGreater(np.linalg.eigvalsh(k).min(),-1e-10);self.assertFalse(np.allclose(np.diagonal(k,axis1=-2,axis2=-1),1.))
    def test03_diagonals_reconstruct_all_coordinates(self):
        x=torch.randn(2,24,3,dtype=torch.double)*2;k=kernel_matrices(x);d=k.diagonal(dim1=-2,dim2=-1).transpose(1,2);recovered=torch.log(torch.expm1(torch.sqrt(d)));torch.testing.assert_close(recovered,patch_values(x),atol=1e-12,rtol=1e-12)
    def test04_translation_detected(self):
        x=torch.randn(1,24,2);self.assertGreater(float(parts(x+2,x)['diag_mse']),0.1)
    def test05_sign_flip_detected(self):
        x=torch.randn(1,24,2);self.assertGreater(float(parts(-x,x)['diag_mse']),0.1)
    def test06_common_fourphase_spike_detected(self):
        x=torch.randn(1,24,2);delta=torch.tensor([1.,-1.,1.,-1.]).repeat(6)[None,:,None];kp,_=base_and_amplitude(x);kq,_=base_and_amplitude(x+delta);torch.testing.assert_close(kp,kq,atol=1e-6,rtol=1e-6);self.assertGreater(float(parts(x+delta,x)['diag_mse']),.1)
    def test07_channels_not_averaged_before_loss(self):
        x=torch.randn(1,24,2,dtype=torch.double);q=x.transpose(1,2).reshape(1,2,6,4);z=q[...,[1,0,2,3]].reshape(1,2,24).transpose(1,2);a,b=kernel_matrices(x),kernel_matrices(z);torch.testing.assert_close(a.mean(1),b.mean(1),atol=1e-12,rtol=1e-12);self.assertGreater(float(parts(z,x)['diag_mse']),.01)
    def test08_all_repeated_patches_still_distinguished(self):
        x=torch.tensor([1.,2.,3.,4.],dtype=torch.double).repeat(6)[None,:,None];z=torch.tensor([4.,3.,2.,1.],dtype=torch.double).repeat(6)[None,:,None];self.assertGreater(float(parts(z,x)['diag_mse']),1.)
    def test09_target_detached(self):
        p=torch.randn(1,12,2,requires_grad=True);y=torch.randn_like(p,requires_grad=True);gp,gy=torch.autograd.grad(objective(p,y),(p,y),allow_unused=True);self.assertIsNone(gy);self.assertTrue(torch.isfinite(gp).all());self.assertGreater(float(gp.norm()),0.)
    def test10_flat_prediction_diagonal_gradient_nonzero(self):
        p=torch.zeros(1,12,2,dtype=torch.double,requires_grad=True);y=torch.ones_like(p);g=torch.autograd.grad(parts(p,y)['diag_mse'],p)[0];self.assertTrue(torch.all(g!=0))
    def test11_FP64_gradient_check(self):
        p=(torch.randn(1,8,1,dtype=torch.double)*.3).requires_grad_();y=torch.randn_like(p);self.assertTrue(torch.autograd.gradcheck(lambda z:objective(z,y),(p,),atol=2e-5,rtol=2e-4))
    def test12_exact_truth_zero(self):
        p=torch.randn(1,24,2);v=parts(p,p,metadata=True);self.assertEqual(float(v['diag_mse']),0.);self.assertEqual(float(v['offdiag_mse']),0.)
    def test13_independent_numpy_loss_means(self):
        p=torch.randn(2,16,2);y=torch.randn_like(p);a,b=numpy_k(p.numpy()),numpy_k(y.numpy());e=(a-b)**2;mask=~np.eye(4,dtype=bool);v=parts(p,y);self.assertAlmostEqual(float(v['offdiag_mse']),float(e[:,:,mask].mean()),places=5);self.assertAlmostEqual(float(v['diag_mse']),float(np.diagonal(e,axis1=-2,axis2=-1).mean()),places=5)
    def test14_fixed_weights_and_three_terms(self):
        p=torch.randn(1,12,2);y=torch.randn_like(p);t=training_terms(p,y);self.assertEqual(tuple(t),tuple(WEIGHTS));self.assertEqual(WEIGHTS,{'point_mse':1.,'amp_kernel_offdiag_mse':.1,'amp_kernel_diag_mse':.1});self.assertTrue(torch.equal(objective(p,y),sum(WEIGHTS[n]*v for n,v in t.items())))
    def test15_reference_optimizer_unchanged_F(self):
        with tempfile.TemporaryDirectory(dir=R/'tmp') as t:
            m,c=fresh(Path(t));self.assertEqual((sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers()))),(906416,227,259,32));old=read(R/'tmp/mafirst_kernel_h720_pointonly_20261025_h720_point_only_full_seed2021/optimizer_groups.json');self.assertEqual(groups_audit(m,optimizer_groups(m)),old)
    def test16_real_model_backward_and_raw_three_gradients(self):
        with tempfile.TemporaryDirectory(dir=R/'tmp') as t:
            m,c=fresh(Path(t));m.eval();x,y=fixed_data();fw,h=hook_calls(m);p=fw(x[:1]);terms=training_terms(p,y[:1]);gg=gradient_summary(terms,m);loss=sum(WEIGHTS[n]*v for n,v in terms.items());loss.backward();h.remove();self.assertEqual(len(gg['cosines']),3);self.assertTrue(all(np.isfinite(v['raw_norm']) for v in gg['norms'].values()));self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters()));self.assertGreater(float(m.global_seasonal_delta.grad.norm()),0.)
    def test17_data_scaler_sampler_match_F(self):
        train,val,sc,data=data_for();self.assertEqual((len(train),len(val)),(33745,10801));f=R/'tmp/mafirst_kernel_h720_pointonly_20261025_h720_point_only_full_seed2021';d=read(f/'data_manifest.json');np.testing.assert_array_equal(sc.mean_,d['scaler_mean']);np.testing.assert_array_equal(sc.scale_,d['scaler_scale']);h=__import__('hashlib').sha256(np.asarray(batches(train,1).dataset.indices,dtype=np.int64).tobytes()).hexdigest();self.assertEqual(h,read(f/'curves.json')[0]['order_sha256'])
    def test18_gate_requires_point_and_both_kernel_errors(self):
        p=read(PLAN)['small_sample'];v=dict(mse=1e-5,max_channel_mse=1e-4,max_abs_error=.01,amp_kernel_offdiag_mse=1e-4,amp_kernel_diag_mse=1e-4);self.assertTrue(small_passed(v,p))
        for key,bad in [('mse',.001),('max_channel_mse',.01),('max_abs_error',.3),('amp_kernel_offdiag_mse',.01),('amp_kernel_diag_mse',.01)]:self.assertFalse(small_passed({**v,key:bad},p))
    def test19_single_forward_diagnostic_saved_kernel(self):
        with tempfile.TemporaryDirectory(dir=R/'tmp') as t:
            folder=Path(t);m,c=fresh(folder/'logs');x,y=fixed_data()
            class TrainingWindows:
                def __len__(self):return 3
                def __getitem__(self,i):return x[i],y[i]
            v=evaluate(m,TrainingWindows(),'cpu',folder/'saved');p=np.load(folder/'saved/prediction.npy');truth=np.load(folder/'saved/truth.npy');a,b=numpy_k(p),numpy_k(truth);mask=~np.eye(180,dtype=bool);e=(a-b)**2;self.assertAlmostEqual(v['amp_kernel_offdiag_mse'],float(e[:,:,mask].mean()),places=5);self.assertAlmostEqual(v['amp_kernel_diag_mse'],float(np.diagonal(e,axis1=-2,axis2=-1).mean()),places=5)
            for i in range(3):
                z=np.load(folder/'saved'/f'amp_kernel_{i}.npz');np.testing.assert_allclose(z['prediction'],a.reshape(3,7,4,180,180)[i],atol=1e-12,rtol=1e-12)
    def test20_invalid_hyperparameters_and_shapes(self):
        x=torch.randn(1,12,2)
        for sigma,scale in [(0,1),(1,0),(-1,1),(1,float('inf'))]:
            with self.assertRaises(ValueError):kernel_matrices(x,sigma,scale)
        with self.assertRaises(ValueError):parts(x,x[:,:,:1])
if __name__=='__main__':unittest.main()
