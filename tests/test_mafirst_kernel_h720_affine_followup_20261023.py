"""Meaningful CPU invariants for the one matched H720 affine follow-up."""
import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
import torch,numpy as np
from tmp.mafirst_kernel_h720_affine_followup_20261023_factory import fresh,optimizer_groups,R,G,PLAN,B,OLDG,C720,AFFINE_NAMES
from models.MAFirstKernelTransferAblation import Model as OffModel,kernel_supervision,value_gram,parallel_future_states
from utils.time_position_patch_mmd import parts,numpy_parts
torch.set_num_threads(2)
class AffineTests(unittest.TestCase):
    def model(self):return fresh(720,'h720_affine_on',tempfile.gettempdir()+'/affine_followup_unit')[0]
    def xy(self):
        g=torch.Generator().manual_seed(31);return torch.randn(2,96,7,generator=g),torch.randn(2,720,7,generator=g)
    def test_01_scope_counts_and_all264_old256_old236(self):
        plan=json.loads(PLAN.read_text());self.assertEqual([j['id'] for j in plan['jobs']],['h720_affine_on']);self.assertTrue(plan['affine'])
        m=self.model();self.assertEqual((sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers()))),(385226,232,264,32))
        for path,nested,count in [(G/'reference_initial/h720.pt',False,264),(OLDG/'reference_initial/h720.pt',False,256),(C720/'initial.pt',True,236)]:
            ref=torch.load(path,map_location='cpu',weights_only=True);ref=ref['time_position_mmd'] if nested else ref
            self.assertEqual(len(ref),count);self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in ref.items()))
    def test_02_identity_affine_storage_optimizer232(self):
        m=self.model();names=dict(m.named_parameters());self.assertEqual(len(AFFINE_NAMES),8)
        self.assertEqual(len({names[n].data_ptr() for n in AFFINE_NAMES}),8)
        for n in AFFINE_NAMES:self.assertTrue(torch.equal(names[n],torch.ones(7) if n.endswith('weight') else torch.zeros(7)))
        groups=optimizer_groups(m);byid={id(p):n for n,p in names.items()};actual={byid[id(p)]:g['weight_decay'] for g in groups for p in g['params']}
        expected={n:g['weight_decay'] for g in json.loads((B/'optimizer_groups.json').read_text()) for n in g['names']}
        self.assertEqual(len(actual),232);self.assertEqual({n:actual[n] for n in expected},expected);self.assertTrue(all(actual[n]==0 for n in AFFINE_NAMES))
    def test_03_scratch_point_eval_train_dropout_rng_exact(self):
        m=self.model();cfg=json.loads((B/'model_config.json').read_text());cfg.update(log_dir=tempfile.gettempdir(),tensorboard=False)
        b=OffModel(SimpleNamespace(**cfg));b.load_state_dict(torch.load(OLDG/'reference_initial/h720.pt',map_location='cpu',weights_only=True))
        x,_=self.xy()
        for training in (False,True):
            outs=[];rngs=[]
            for v in (m,b):v.train(training);torch.manual_seed(2021);outs.append(v(x));rngs.append(torch.get_rng_state())
            self.assertTrue(torch.equal(*outs));self.assertTrue(torch.equal(*rngs));self.assertEqual(outs[0].shape,(2,720,7))
    def test_04_nonidentity_affine_inverse_real_mean_and_units(self):
        m=self.model();m.eval();x,_=self.xy()
        with torch.no_grad():
            for i,n in enumerate([m.normalize,*m.coarse_normalizers]):n.affine_weight.copy_(torch.linspace(.7,1.4,7)+i*.02);n.affine_bias.copy_(torch.linspace(-.3,.2,7))
        p=m(x);c=m.predict_components(x);scale,offset=m.inverse_coordinates()
        expected_scale=m.normalize.stdev/(m.normalize.affine_weight+m.normalize.eps**2)
        expected_offset=x.mean(1,keepdim=True)-m.normalize.affine_bias*expected_scale
        self.assertTrue(torch.equal(scale,expected_scale));self.assertTrue(torch.equal(offset,expected_offset));self.assertTrue(torch.equal(m.normalize.mean,x.mean(1,keepdim=True)))
        raw=c*scale.unsqueeze(-1);reconstructed=raw[...,:4].sum(-1)+raw[...,4:].sum(-1)+offset
        self.assertLess(float((p-reconstructed).detach().abs().max()),2e-5)
        self.assertFalse(torch.equal(offset,m.normalize.mean))
        # Independent four-coordinate head calculation in raw data units.
        _,(season,trend)=m.encode_multiscale(m.downsample(x));h=m.kernel_head
        joint=torch.cat((season[0],trend[0]),-1);coords=h.coordinates(h.temporal(joint.transpose(1,2)).transpose(1,2))
        rawcoords=coords*expected_scale.transpose(1,2).reshape(14,1,1)
        self.assertTrue(torch.equal(h(season[0],trend[0],expected_scale.transpose(1,2).reshape(14,1,1)),value_gram(rawcoords)))
        # Translation of all four coordinates leaves the distance Gram unchanged.
        self.assertTrue(torch.allclose(value_gram(rawcoords),value_gram(rawcoords+2.0),atol=1e-6,rtol=1e-6))
    def test_05_batch_channel_patch_order_targetdetach(self):
        m=self.model();m.eval();x,y=self.xy();p=m(x);target=y.clone().requires_grad_();aux,gold=kernel_supervision(m.predicted_gram,target)
        expected=value_gram(y.transpose(1,2).reshape(14,180,4));self.assertTrue(torch.equal(gold,expected));self.assertFalse(gold.requires_grad)
        self.assertIsNone(torch.autograd.grad(aux,target,allow_unused=True)[0])
        self.assertTrue(torch.equal(p,m(x)))
        for i in range(2):self.assertTrue(torch.allclose(p[i:i+1],m(x[i:i+1]),atol=2e-6,rtol=1e-5))
    def test_06_all232_finite_gradients_and_affine_updates(self):
        m=self.model();m.eval();x,y=self.xy();opt=torch.optim.Adam(optimizer_groups(m),lr=.001)
        initial={n:p.detach().clone() for n,p in m.named_parameters() if n in AFFINE_NAMES}
        for _ in range(3):
            opt.zero_grad();p=m(x);aux,_=kernel_supervision(m.predicted_gram,y);loss=(p-y).square().mean()+parts(p,y)[0]+aux;loss.backward()
            self.assertEqual(len(list(m.parameters())),232);self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()));opt.step()
        self.assertTrue(all(not torch.equal(initial[n],p) for n,p in m.named_parameters() if n in AFFINE_NAMES))
    def test_07_both_feedback_interventions_effective(self):
        m=self.model();m.eval();x,y=self.xy();opt=torch.optim.Adam(optimizer_groups(m),lr=.001)
        # Initial decoder readout is zero: establish nonzero signal before testing routing.
        for _ in range(4):
            opt.zero_grad();pred=m(x);aux,_=kernel_supervision(m.predicted_gram,y);loss=(pred-y).square().mean()+parts(pred,y)[0]+aux;loss.backward();opt.step()
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
        from tmp.run_mafirst_kernel_h720_affine_followup_20261023 import gradient_capture
        from models import MAFirstKernelTransferAblation as actual_globals
        original=actual_globals.parallel_future_states;m=self.model();x,y=self.xy()
        with tempfile.TemporaryDirectory() as d:
            result=gradient_capture(m,x,y,Path(d),'snapshot');self.assertEqual(len(result),4)
            z=np.load(Path(d)/'snapshot.npz')
            for i in range(4):self.assertTrue(np.array_equal(z[f's{i}_adjoint_rms'][:135],np.zeros(135)))
        self.assertIs(actual_globals.parallel_future_states,original)
        k=value_gram(torch.randn(2,180,4,dtype=torch.float64));self.assertTrue(torch.equal(k,k.transpose(-1,-2)));self.assertGreater(float(torch.linalg.eigvalsh(k).min()),-1e-10);self.assertTrue(torch.equal(k.diagonal(dim1=-2,dim2=-1),torch.ones(2,180,dtype=k.dtype)))
    def test_10_dataset_scaler_boundaries(self):
        from tmp.run_mafirst_kernel_h720_affine_followup_20261023 import data_for
        j=json.loads(PLAN.read_text())['jobs'][0];tr,va,sc,data=data_for(j)
        self.assertEqual((len(tr),len(va),int(np.ceil(len(tr)/32))),(33745,10801,1055))
        manifest=json.loads((B/'data_manifest.json').read_text());self.assertTrue(np.array_equal(sc.mean_,manifest['scaler_mean']));self.assertTrue(np.array_equal(sc.scale_,manifest['scaler_scale']))
        self.assertTrue(np.array_equal(tr[0][0],data[:96]));self.assertTrue(np.array_equal(tr[len(tr)-1][1],data[33840:34560]));self.assertTrue(np.array_equal(va[0][1],data[34560:35280]));self.assertTrue(np.array_equal(va[len(va)-1][1],data[45360:46080]))
if __name__=='__main__':unittest.main()
