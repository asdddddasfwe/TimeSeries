"""Meaningful CPU invariants for the one matched H720 affine follow-up."""
import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
import torch,numpy as np
from tmp.mafirst_kernel_h720_localcs_20261023_factory import fresh,optimizer_groups,R,G,PLAN,B,OLDG,C720,AFFINE_NAMES,REMOVED,ADDED
from models.MAFirstKernelTransferAblation import Model as OffModel,kernel_supervision,value_gram,parallel_future_states
from utils.time_position_patch_mmd import parts,numpy_parts
from utils.local_offdiagonal_gram_cs import parts as align_parts, numpy_parts as align_numpy
torch.set_num_threads(2)
class AffineTests(unittest.TestCase):
    def model(self):return fresh(720,'h720_localcs',tempfile.gettempdir()+'/affine_followup_unit')[0]
    def xy(self):
        g=torch.Generator().manual_seed(31);return torch.randn(2,96,7,generator=g),torch.randn(2,720,7,generator=g)
    def test_01_counts_shared255_and_new3_direct_head(self):
        from layers.StandardNorm import Normalize
        m=self.model();self.assertEqual((sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers()))),(388016,226,258,32))
        self.assertEqual(sum(isinstance(n,Normalize) for n in m.modules()),1);self.assertFalse(hasattr(m,'coarse_normalizers'))
        ref=torch.load(G/'reference_initial/h720.pt',map_location='cpu',weights_only=True);old=torch.load(OLDG/'reference_initial/h720.pt',map_location='cpu',weights_only=True)
        self.assertEqual(len(ref),258);shared=set(ref)&set(old);self.assertEqual(len(shared),258);self.assertEqual(set(old)-set(ref),set(REMOVED));self.assertEqual(set(ref)-set(old),set(ADDED))
        self.assertTrue(all(torch.equal(v,ref[k]) for k,v in m.state_dict().items()));self.assertTrue(all(torch.equal(old[k],ref[k]) for k in shared))
        original=torch.load(C720/'initial.pt',map_location='cpu',weights_only=True)['time_position_mmd'];self.assertEqual(len(original),236);self.assertTrue(all(torch.equal(v,ref[k]) for k,v in original.items()))

    def test_02_affine_unchanged_retained_optimizer_and_new_rows(self):
        m=self.model();names=dict(m.named_parameters());self.assertEqual({n for n in names if n.endswith(('affine_weight','affine_bias'))},set(AFFINE_NAMES))
        for n in AFFINE_NAMES:self.assertTrue(torch.equal(names[n],torch.ones(7) if n.endswith('weight') else torch.zeros(7)))
        groups=optimizer_groups(m);byid={id(p):n for n,p in names.items()};actual={byid[id(p)]:g['weight_decay'] for g in groups for p in g['params']};prior={n:g['weight_decay'] for g in json.loads((B/'optimizer_groups.json').read_text()) for n in g['names']}
        self.assertEqual(len(actual),226);self.assertEqual(sum(len(g['params']) for g in groups),226)
        self.assertTrue(all(wd==prior[n] for n,wd in actual.items() if n in prior));self.assertEqual(set(actual)-set(prior),set(ADDED));self.assertTrue(all(actual[n]==(0. if n.endswith('bias') else 1e-4) for n in ADDED));self.assertTrue(all(actual[n]==0 for n in AFFINE_NAMES))

    def test_03_full_revin_forward_norm_and_denorm_once(self):
        from unittest.mock import patch
        m=self.model();m.eval();x,_=self.xy()
        with patch.object(m.normalize,'forward',wraps=m.normalize.forward) as full,patch.object(m.normalize,'_get_statistics',wraps=m.normalize._get_statistics) as stats,patch.object(m.normalize,'_normalize',wraps=m.normalize._normalize) as affine:
            m(x)
            self.assertEqual([c.args[1] for c in full.call_args_list],['norm','denorm']);self.assertEqual(stats.call_count,1);self.assertEqual(affine.call_count,1)
            self.assertTrue(torch.equal(full.call_args_list[0].args[0],x));self.assertEqual(stats.call_args.args[0].shape,(2,96,7))
        scales=m.downsample(x);self.assertEqual([v.shape[1] for v in scales],[96,48,24,12])
        with patch.object(m.normalize,'forward',side_effect=AssertionError('No normalization after pooling')):m.encode_multiscale(scales)
        expected=(x-x.mean(1,keepdim=True))/(x.var(1,keepdim=True,unbiased=False)+1e-5).sqrt()
        for i,z in enumerate(scales):
            if i:expected=torch.nn.functional.avg_pool1d(expected.transpose(1,2),2,2).transpose(1,2)
            self.assertTrue(torch.equal(z,expected))

    def test_04_nonidentity_full_revin_inverse_and_head_scale_input(self):
        from models.MAFirstKernelH720DirectGram import direct_similarity
        m=self.model();m.eval();x,_=self.xy()
        with torch.no_grad():m.normalize.affine_weight.copy_(torch.linspace(.7,1.4,7));m.normalize.affine_bias.copy_(torch.linspace(-.3,.2,7))
        p=m(x);components=m.predict_components(x);scale,offset=m.inverse_coordinates()
        expected_scale=m.normalize.stdev/(m.normalize.affine_weight+m.normalize.eps**2);expected_offset=x.mean(1,keepdim=True)-m.normalize.affine_bias*expected_scale
        self.assertTrue(torch.equal(scale,expected_scale));self.assertTrue(torch.equal(offset,expected_offset));self.assertTrue(torch.equal(m.normalize.mean,x.mean(1,keepdim=True)))
        raw=components*scale.unsqueeze(-1);self.assertLess(float((raw[...,:4].sum(-1)+raw[...,4:].sum(-1)+offset-p).detach().abs().max()),2e-5)
        _,(season,trend)=m.encode_multiscale(m.downsample(x));h=m.kernel_head;joint=torch.cat((season[0],trend[0]),-1);future=h.temporal(joint.transpose(1,2)).transpose(1,2);rawscale=scale.transpose(1,2).reshape(14,1,1)
        logits=h.rows(torch.cat((future,rawscale.expand(-1,180,1)),-1));self.assertTrue(torch.equal(h(season[0],trend[0],rawscale),direct_similarity(logits)))

    def test_05_batch_channel_patch_order_targetdetach(self):
        m=self.model();m.eval();x,y=self.xy();p=m(x);target=y.clone().requires_grad_();aux,gold=kernel_supervision(m.predicted_gram,target)
        expected=value_gram(y.transpose(1,2).reshape(14,180,4));self.assertTrue(torch.equal(gold,expected));self.assertFalse(gold.requires_grad)
        self.assertIsNone(torch.autograd.grad(aux,target,allow_unused=True)[0])
        self.assertTrue(torch.equal(p,m(x)))
        for i in range(2):self.assertTrue(torch.allclose(p[i:i+1],m(x[i:i+1]),atol=2e-6,rtol=1e-5))
    def test_06_all226_finite_gradients_and_affine_updates(self):
        m=self.model();m.eval();x,y=self.xy();opt=torch.optim.Adam(optimizer_groups(m),lr=.001)
        initial={n:p.detach().clone() for n,p in m.named_parameters() if n in AFFINE_NAMES}
        for _ in range(3):
            opt.zero_grad();p=m(x);aux,gold=kernel_supervision(m.predicted_gram,y);loss=(p-y).square().mean()+parts(p,y)[0]+aux+align_parts(m.predicted_gram,gold)[0];loss.backward()
            self.assertEqual(len(list(m.parameters())),226);self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()));opt.step()
        self.assertTrue(all(not torch.equal(initial[n],p) for n,p in m.named_parameters() if n in AFFINE_NAMES))
    def test_07_both_feedback_interventions_effective(self):
        m=self.model();m.eval();x,y=self.xy();opt=torch.optim.Adam(optimizer_groups(m),lr=.001)
        # Initial decoder readout is zero: establish nonzero signal before testing routing.
        for _ in range(4):
            opt.zero_grad();pred=m(x);aux,gold=kernel_supervision(m.predicted_gram,y);loss=(pred-y).square().mean()+parts(pred,y)[0]+aux+align_parts(m.predicted_gram,gold)[0];loss.backward();opt.step()
        with torch.no_grad():
            for layer in [*m.kernel_season_feedback,*m.kernel_trend_feedback]:layer.weight.copy_(torch.randn_like(layer.weight)*.05)
        baseline=m.predict_components(x);handle=m.kernel_head.register_forward_hook(lambda mod,args,out:torch.eye(180).expand_as(out))
        try:altered=m.predict_components(x)
        finally:handle.remove()
        self.assertFalse(torch.equal(baseline[...,:4],altered[...,:4]));self.assertFalse(torch.equal(baseline[...,4:],altered[...,4:]))
    def test_08_time_mmd_numpy_and_detached_target(self):
        g=torch.Generator().manual_seed(91);p=torch.randn(2,12,3,dtype=torch.float64,generator=g,requires_grad=True);y=torch.randn(p.shape,dtype=p.dtype,generator=g,requires_grad=True)
        a,_=parts(p,y);self.assertAlmostEqual(float(a.detach()),numpy_parts(p.detach().numpy(),y.detach().numpy())['time_position_mmd'],places=12)
        self.assertIsNone(torch.autograd.grad(a,y,allow_unused=True)[0]);self.assertTrue(torch.autograd.gradcheck(lambda z:parts(z,y)[0],(p,),eps=1e-6,atol=1e-5))
    def test_09_inherited_gradient_capture_and_gram_properties(self):
        from tmp.run_mafirst_kernel_h720_localcs_20261023 import gradient_capture
        from models import MAFirstKernelTransferAblation as actual_globals
        original=actual_globals.parallel_future_states;m=self.model();x,y=self.xy()
        with tempfile.TemporaryDirectory() as d:
            result=gradient_capture(m,x,y,Path(d),'snapshot');self.assertEqual(len(result),4)
            z=np.load(Path(d)/'snapshot.npz')
            for i in range(4):self.assertTrue(np.array_equal(z[f's{i}_adjoint_rms'][:135],np.zeros(135)))
        self.assertIs(actual_globals.parallel_future_states,original)
        k=value_gram(torch.randn(2,180,4,dtype=torch.float64));self.assertTrue(torch.equal(k,k.transpose(-1,-2)));self.assertGreater(float(torch.linalg.eigvalsh(k).min()),-1e-10);self.assertTrue(torch.equal(k.diagonal(dim1=-2,dim2=-1),torch.ones(2,180,dtype=k.dtype)))
    def test_10_dataset_scaler_boundaries(self):
        from tmp.run_mafirst_kernel_h720_localcs_20261023 import data_for
        j=json.loads(PLAN.read_text())['jobs'][0];tr,va,sc,data=data_for(j)
        self.assertEqual((len(tr),len(va),int(np.ceil(len(tr)/32))),(33745,10801,1055))
        manifest=json.loads((B/'data_manifest.json').read_text());self.assertTrue(np.array_equal(sc.mean_,manifest['scaler_mean']));self.assertTrue(np.array_equal(sc.scale_,manifest['scaler_scale']))
        self.assertTrue(np.array_equal(tr[0][0],data[:96]));self.assertTrue(np.array_equal(tr[len(tr)-1][1],data[33840:34560]));self.assertTrue(np.array_equal(va[0][1],data[34560:35280]));self.assertTrue(np.array_equal(va[len(va)-1][1],data[45360:46080]))
    def test_11_nonidentity_sole_affine_before_all_pooling(self):
        from torch.nn import functional as F
        from unittest.mock import patch
        m=self.model();x,_=self.xy()
        with torch.no_grad():m.normalize.affine_weight.copy_(torch.linspace(.6,1.4,7));m.normalize.affine_bias.copy_(torch.linspace(-.3,.4,7))
        with patch.object(m.normalize,'forward',wraps=m.normalize.forward) as full:
            scales=m.downsample(x);self.assertEqual(full.call_count,1);self.assertEqual(full.call_args.args[1],'norm')
        expected=((x-m.normalize.mean)/m.normalize.stdev)*m.normalize.affine_weight+m.normalize.affine_bias
        for i,z in enumerate(scales):
            if i:expected=F.avg_pool1d(expected.transpose(1,2),2,2).transpose(1,2)
            self.assertTrue(torch.equal(z,expected))
        reconstructed=m.normalize(scales[0],'denorm');self.assertTrue(torch.allclose(reconstructed,x,atol=1e-6,rtol=1e-6))

    def test_12_normalization_inherited_exact_and_point_scratch_identity(self):
        from models.MAFirstKernelH720DirectGram import Model as Prior
        m=self.model();cfg=json.loads((B/'model_config.json').read_text());cfg.update(log_dir=tempfile.gettempdir(),tensorboard=False);old=Prior(SimpleNamespace(**cfg));old.load_state_dict(torch.load(OLDG/'reference_initial/h720.pt',map_location='cpu',weights_only=True))
        self.assertIs(m.downsample.__func__,old.downsample.__func__);self.assertIs(m.encode_multiscale.__func__,old.encode_multiscale.__func__)
        x,_=self.xy();m.eval();old.eval();self.assertTrue(all(torch.equal(a,b) for a,b in zip(m.downsample(x),old.downsample(x))));self.assertTrue(torch.equal(m(x),old(x)))
        m.train();old.train();torch.manual_seed(889);rng=torch.get_rng_state();a=m(x);torch.set_rng_state(rng);b=old(x);self.assertTrue(torch.equal(a,b))

    def test_13_direct_logits_numpy_symmetry_range_diagonal_gradient(self):
        from models.MAFirstKernelH720DirectGram import direct_similarity
        z=torch.randn(2,9,9,dtype=torch.float64,requires_grad=True);g=direct_similarity(z);a=z.detach().numpy();expected=1/(1+np.exp(-(a+a.transpose(0,2,1))/2));expected[:,np.arange(9),np.arange(9)]=1
        self.assertTrue(np.allclose(g.detach().numpy(),expected,atol=1e-15,rtol=1e-15));self.assertTrue(torch.equal(g,g.transpose(-1,-2)));self.assertTrue(torch.equal(g.diagonal(dim1=-2,dim2=-1),torch.ones(2,9,dtype=g.dtype)));self.assertTrue(((g>=0)&(g<=1)).all())
        grad=torch.autograd.grad(g.sum(),z)[0];self.assertTrue(torch.isfinite(grad).all());self.assertTrue(torch.equal(grad.diagonal(dim1=-2,dim2=-1),torch.zeros(2,9,dtype=grad.dtype)));self.assertGreater(float(grad.abs().sum()),0)
    def test_14_direct_similarity_does_not_enforce_PSD(self):
        from models.MAFirstKernelH720DirectGram import direct_similarity
        matrix=torch.tensor([[.5,.99,.99],[.99,.5,.01],[.99,.01,.5]],dtype=torch.float64);z=torch.log(matrix/(1-matrix));g=direct_similarity(z)
        self.assertLess(float(torch.linalg.eigvalsh(g).min()),-.1);self.assertTrue(torch.equal(g.diagonal(),torch.ones(3,dtype=g.dtype)))
    def test_15_history_scale_is_available_and_differentiable(self):
        m=self.model();h=m.kernel_head;s=torch.randn(2,96,32);t=torch.randn_like(s);scale=torch.tensor([.7,1.3]).reshape(2,1,1).requires_grad_();a=h(s,t,scale);b=h(s,t,scale*2)
        self.assertFalse(torch.equal(a,b));grad=torch.autograd.grad(a.sum(),scale)[0];self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)
    def test_16_predict_head_never_calls_distance_RBF(self):
        from unittest.mock import patch
        from models import MAFirstKernelTransferAblation as original
        m=self.model();m.eval();x,_=self.xy()
        with patch.object(original,'value_gram',side_effect=AssertionError('Predicted Gram must be direct logits, not a distance kernel')):m(x)
        g=m.predicted_gram;self.assertEqual(g.shape,(14,180,180));self.assertTrue(torch.equal(g,g.transpose(-1,-2)));self.assertTrue(((g>=0)&(g<=1)).all())

    def test_17_localCS_numpy_gradcheck_and_targetdetach(self):
        gen=torch.Generator().manual_seed(241);p=torch.rand(2,5,5,dtype=torch.float64,generator=gen,requires_grad=True);y=torch.rand(2,5,5,dtype=torch.float64,generator=gen,requires_grad=True)
        loss,meta=align_parts(p,y);self.assertAlmostEqual(float(loss.detach()),align_numpy(p.detach().numpy(),y.detach().numpy())['alignment_loss'],places=12)
        self.assertIsNone(torch.autograd.grad(loss,y,allow_unused=True,retain_graph=True)[0]);self.assertTrue(torch.autograd.gradcheck(lambda z:align_parts(z,y)[0],(p,),eps=1e-6,atol=1e-5))
    def test_18_original_diagonal_ignored_in_every_window(self):
        p=torch.rand(2,18,18,dtype=torch.float64,requires_grad=True);y=torch.rand_like(p);a,meta=align_parts(p,y);b,other=align_parts(p+torch.eye(18)*9,y-torch.eye(18)*7)
        self.assertTrue(torch.equal(a,b));self.assertEqual(meta['local_windows_per_matrix'],4)
        g=torch.autograd.grad(a,p)[0];self.assertTrue(torch.equal(g.diagonal(dim1=-2,dim2=-1),torch.zeros(2,18,dtype=g.dtype)))
        self.assertEqual(meta['local_prediction_std'],other['local_prediction_std']);self.assertEqual(meta['local_target_std'],other['local_target_std'])
    def test_19_identity_offset_and_contrast_shrink_penalty(self):
        y=torch.linspace(.1,.9,49,dtype=torch.float64).reshape(1,7,7);a=.05
        self.assertLess(float(align_parts(y,y)[0]),1e-12);self.assertLess(float(align_parts(y+.2,y)[0]),1e-12)
        p=.7+a*(y-y.mean());loss,_=align_parts(p,y);v=y[:,~torch.eye(7,dtype=torch.bool)].var(-1,unbiased=False);expected=1-(2*a*v+.03**2)/((a*a+1)*v+.03**2)
        self.assertAlmostEqual(float(loss),float(expected.mean()),places=12);self.assertGreater(float(loss),.8)
        self.assertGreater(float(align_parts(1-y,y)[0]),1.8)
    def test_20_flat_targets_included_and_constant_prediction_gradient(self):
        y=torch.ones(2,5,5,dtype=torch.float64)*.7;p=torch.rand_like(y,requires_grad=True);a,meta=align_parts(p,y)
        self.assertGreater(float(a.detach()),0);self.assertEqual(meta['target_degenerate_fraction'],1.);self.assertTrue(meta['target_flat_windows_included'])
        g=torch.autograd.grad(a,p)[0];self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().sum()),0)
        self.assertLess(float(align_parts(torch.full_like(y,.2),y)[0]),1e-12)
        p=torch.full((2,5,5),.5,dtype=torch.float64,requires_grad=True);y=torch.linspace(.1,.9,50,dtype=torch.float64).reshape(2,5,5);a,meta=align_parts(p,y);g=torch.autograd.grad(a,p)[0]
        self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().sum()),0);self.assertEqual(meta['prediction_degenerate_fraction'],1.)
    def test_21_batch_variable_isolation_and_shapes(self):
        p=torch.rand(2,3,5,5,dtype=torch.float64);y=torch.rand_like(p);a=align_parts(p,y)[0];b=torch.stack([align_parts(p[i,j],y[i,j])[0] for i in range(2) for j in range(3)]).mean()
        self.assertAlmostEqual(float(a),float(b),places=12)
        with self.assertRaises(ValueError):align_parts(torch.ones(2,2),torch.ones(2,2))
    def test_22_structure_loss_reaches_direct_head_and_shared_encoder(self):
        m=self.model();m.eval();x,y=self.xy();m(x);_,gold=kernel_supervision(m.predicted_gram,y);a,_=align_parts(m.predicted_gram,gold)
        names=list(m.named_parameters());grads=torch.autograd.grad(a,[p for n,p in names],allow_unused=True)
        self.assertTrue(all(g is None or torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for (n,p),g in zip(names,grads) if g is not None and n.startswith('kernel_head.rows.')),0)
        self.assertGreater(sum(float(g.abs().sum()) for (n,p),g in zip(names,grads) if g is not None and ('.blocks.' in n or n.startswith('pdm_blocks.'))),0)
    def test_23_known_matrix_alignment_optimization(self):
        from models.MAFirstKernelH720DirectGram import direct_similarity
        torch.manual_seed(81);z=torch.randn(1,8,8,dtype=torch.float64,requires_grad=True);y=value_gram(torch.randn(1,8,4,dtype=torch.float64));opt=torch.optim.Adam([z],lr=.05)
        first=float(align_parts(direct_similarity(z),y)[0].detach())
        for _ in range(30):opt.zero_grad();a,_=align_parts(direct_similarity(z),y);a.backward();opt.step()
        self.assertLess(float(align_parts(direct_similarity(z),y)[0].detach()),first-.1)
    def test_24_four_loss_gradient_sum_and_normalization_receipt(self):
        m=self.model();m.eval();x,y=self.xy();p=m(x);numeric,gold=kernel_supervision(m.predicted_gram,y);structure,_=align_parts(m.predicted_gram,gold)
        terms=[(p-y).square().mean(),parts(p,y)[0],numeric,structure];params=list(m.parameters());raw=[torch.autograd.grad(t,params,retain_graph=True,allow_unused=True) for t in terms]
        total=torch.autograd.grad(sum(terms),params,allow_unused=True)
        for i,(param,g) in enumerate(zip(params,total)):
            expected=sum((row[i] if row[i] is not None else torch.zeros_like(param)) for row in raw)
            self.assertIsNotNone(g);self.assertTrue(torch.isfinite(g).all());self.assertTrue(torch.allclose(g,expected,atol=2e-6,rtol=1e-5))


    def test_25_actual_evaluator_localCS_matches_numpy(self):
        from tmp.run_mafirst_kernel_h720_localcs_20261023 import evaluate
        from torch.utils.data import TensorDataset
        m=self.model();m.eval();x,y=self.xy()
        with torch.no_grad():
            m(x);numeric,gold=kernel_supervision(m.predicted_gram,y);expected=align_numpy(m.predicted_gram.numpy(),gold.numpy())
        measured=evaluate(m,TensorDataset(x,y),'cpu')
        self.assertAlmostEqual(measured['kernel_mse'],float(numeric),places=7)
        self.assertAlmostEqual(measured['kernel_local_cs_loss'],expected['alignment_loss'],places=7)
        self.assertAlmostEqual(measured['kernel_offdiagonal_correlation'],expected['mean_correlation'],places=12)
        self.assertAlmostEqual(measured['kernel_local_cs_similarity'],expected['local_cs_similarity'],places=10)
        self.assertAlmostEqual(measured['kernel_local_correlation'],expected['local_mean_correlation'],places=10)
        self.assertEqual(measured['windows'],2)
    def test_26_formal180_all841_windows_numpy(self):
        gen=torch.Generator().manual_seed(991);p=torch.rand(1,180,180,dtype=torch.float64,generator=gen);y=torch.rand(1,180,180,dtype=torch.float64,generator=gen)
        a,m=align_parts(p,y);n=align_numpy(p.numpy(),y.numpy());self.assertEqual(m['local_windows_per_matrix'],841)
        for k in ('local_cs_loss','local_cs_similarity','local_mean_correlation','local_prediction_std','local_target_std'):self.assertAlmostEqual(m[k],n[k],places=12)
    def test_27_final_boundary_coverage_numpy(self):
        p=torch.rand(2,19,19,dtype=torch.float64);y=torch.rand_like(p);a,m=align_parts(p,y);n=align_numpy(p.numpy(),y.numpy())
        self.assertEqual(m['local_windows_per_matrix'],9);self.assertAlmostEqual(float(a),n['alignment_loss'],places=12)
        z=p.clone();z[:,-1,-2]+=.7;self.assertNotEqual(float(a),float(align_parts(z,y)[0]))
    def test_28_joint_reversal_local_windows_consistent(self):
        p=torch.rand(2,24,24,dtype=torch.float64);y=torch.rand_like(p)
        self.assertAlmostEqual(float(align_parts(p,y)[0]),float(align_parts(p.flip((-2,-1)),y.flip((-2,-1)))[0]),places=12)
    def test_29_nearly_constant_matrices_finite_and_dtype(self):
        p=torch.full((2,18,18),.7,dtype=torch.float64,requires_grad=True);y=torch.ones_like(p)*.6+torch.arange(18,dtype=torch.float64)[None,:,None]*1e-10
        a,m=align_parts(p,y);self.assertTrue(torch.isfinite(a));self.assertTrue(torch.isfinite(torch.autograd.grad(a,p)[0]).all());self.assertGreaterEqual(float(a.detach()),0)
        self.assertEqual(align_parts(p.float(),y.float())[0].dtype,torch.float32)
    def test_30_locality_changes_under_arbitrary_joint_permutation(self):
        gen=torch.Generator().manual_seed(121);y=torch.rand(1,24,24,dtype=torch.float64,generator=gen);p=.2*y+.3;idx=torch.randperm(24,generator=gen)
        a=align_parts(p,y)[0];b=align_parts(p[:,idx][:,:,idx],y[:,idx][:,:,idx])[0]
        self.assertGreater(abs(float(a-b)),1e-8)

if __name__=='__main__':unittest.main()
