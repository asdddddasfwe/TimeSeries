"""Twelve meaningful checks for H720 transfer and two H336 gates."""
import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
import torch,numpy as np
from tmp.mafirst_kernel_transfer_ablation_20261022_factory import fresh,optimizer_groups,R,G,PLAN,B336,C336,C720
from models.MAFirstKernelTransferAblation import kernel_supervision,value_gram,parallel_future_states
from utils.time_position_patch_mmd import parts,numpy_parts

torch.set_num_threads(2)
class StudyTests(unittest.TestCase):
    def jobs(self):return json.loads(PLAN.read_text())['jobs']
    def model(self,j):return fresh(j['horizon'],j['id'],tempfile.gettempdir()+'/mafirst_transfer_unit')[0]
    def xy(self,H):
        g=torch.Generator().manual_seed(31);return torch.randn(2,96,7,generator=g),torch.randn(2,H,7,generator=g)
    def test_01_order_and_scope(self):
        self.assertEqual([j['id'] for j in self.jobs()],['h720_full','h336_aux_only','h336_feedback_only'])
        self.assertEqual([(j['feedback_enabled'],j['aux_weight']) for j in self.jobs()],[(True,1.),(False,1.),(True,0.)])
    def test_02_full256_reference_and_old236_exact(self):
        for j in self.jobs():
            m=self.model(j);ref=torch.load(G/'reference_initial'/f"h{j['horizon']}.pt",map_location='cpu',weights_only=True)
            self.assertEqual(len(ref),256);self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in ref.items()))
            old=torch.load((C720 if j['horizon']==720 else C336)/'initial.pt',map_location='cpu',weights_only=True)['time_position_mmd']
            self.assertEqual(len(old),236);self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in old.items()))
            self.assertFalse(m.initialization_audit['trained_or_preflight_weights_loaded'])
    def test_03_counts_and_all_optimizer_rules(self):
        for j in self.jobs():
            m=self.model(j);self.assertEqual((sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers()))),(j['parameters'],224,256,32))
            names={id(p):n for n,p in m.named_parameters()};current={names[id(p)]:g['weight_decay'] for g in optimizer_groups(m) for p in g['params']}
            source=json.loads(((C720 if j['horizon']==720 else B336)/'optimizer_groups.json').read_text());source=source['time_position_mmd'] if isinstance(source,dict) else source
            expected={n:g['weight_decay'] for g in source for n in g['names']};self.assertEqual({k:current[k] for k in expected},expected)
            self.assertEqual(len(current),224)
    def test_04_h720_scratch_matches_original_parallel_point_forward_rng(self):
        from models.MAFirstParallelDecoder import Model
        cfg=json.loads((C720/'protocol.json').read_text())['model_config'];cfg.update(log_dir=tempfile.gettempdir(),tensorboard=False)
        b=Model(SimpleNamespace(**cfg));b.load_state_dict(torch.load(C720/'initial.pt',map_location='cpu',weights_only=True)['time_position_mmd']);m=self.model(self.jobs()[0]);x,_=self.xy(720)
        for training in (False,True):
            outputs=[];rng=[]
            for a in (b,m):a.train(training);torch.manual_seed(2021);outputs.append(a(x));rng.append(torch.get_rng_state())
            self.assertTrue(torch.equal(*outputs));self.assertTrue(torch.equal(*rng));self.assertEqual(outputs[0].shape,(2,720,7))
    def test_05_h336_initial_matches_current_parallel(self):
        from models.MAFirstKernelValueOnly336 import Model
        cfg=json.loads((B336/'model_config.json').read_text());cfg.update(study_parallel=True,log_dir=tempfile.gettempdir(),tensorboard=False)
        b=Model(SimpleNamespace(**cfg));b.load_state_dict(torch.load(B336/'initial.pt',map_location='cpu',weights_only=True));b.eval();x,_=self.xy(336)
        for j in self.jobs()[1:]:
            m=self.model(j);m.eval();self.assertTrue(torch.equal(m(x),b(x)))
    def test_06_time_mmd_numpy_targetdetach(self):
        g=torch.Generator().manual_seed(91);p=torch.randn(2,12,3,dtype=torch.float64,generator=g,requires_grad=True);y=torch.randn(p.shape,dtype=p.dtype,generator=g,requires_grad=True)
        a,_=parts(p,y);self.assertAlmostEqual(float(a.detach()),numpy_parts(p.detach().numpy(),y.detach().numpy())['time_position_mmd'],places=12)
        self.assertIsNone(torch.autograd.grad(a,y,allow_unused=True)[0]);self.assertTrue(torch.autograd.gradcheck(lambda v:parts(v,y)[0],(p,),eps=1e-6,atol=1e-5))
    def test_07_gate_functional_interventions_and_total_gradients(self):
        for j in self.jobs():
            m=self.model(j);m.eval();x,y=self.xy(j['horizon']);opt=torch.optim.Adam(optimizer_groups(m),lr=.001)
            for _ in range(4):
                opt.zero_grad();pred=m(x);aux,_=kernel_supervision(m.predicted_gram,y);loss=(pred-y).square().mean()+parts(pred,y)[0]+j['aux_weight']*aux;loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
                gate=[p for n,p in m.named_parameters() if n.startswith(('kernel_season_feedback.','kernel_trend_feedback.'))]
                if not j['feedback_enabled']:self.assertTrue(all(torch.count_nonzero(p.grad)==0 for p in gate))
                opt.step()
            baseline=m.predict_components(x);handle=m.kernel_head.register_forward_hook(lambda mod,args,out:torch.eye(out.shape[-1]).expand_as(out))
            try:altered=m.predict_components(x)
            finally:handle.remove()
            for sl in (slice(None,4),slice(4,None)):
                self.assertEqual(torch.equal(baseline[...,sl],altered[...,sl]),not j['feedback_enabled'])
    def test_08_aux_only_has_no_point_head_gradient(self):
        j=self.jobs()[1];m=self.model(j);m.eval();x,y=self.xy(336);p=m(x);head=list(m.kernel_head.parameters())
        a=torch.autograd.grad((p-y).square().mean(),head,retain_graph=True)
        self.assertTrue(all(torch.count_nonzero(v)==0 for v in a))
        aux,_=kernel_supervision(m.predicted_gram,y);b=torch.autograd.grad(aux,head);self.assertGreater(sum(float(v.detach().square().sum()) for v in b),0.)
    def test_09_feedback_only_loss_gradient_and_later_head_learning(self):
        j=self.jobs()[2];m=self.model(j);m.eval();x,y=self.xy(336);opt=torch.optim.Adam(optimizer_groups(m),lr=.001)
        for step in range(3):
            opt.zero_grad();p=m(x);aux,_=kernel_supervision(m.predicted_gram,y);main=(p-y).square().mean()+parts(p,y)[0]
            named=list(m.named_parameters());a=torch.autograd.grad(main,[v for _,v in named],retain_graph=True);b=torch.autograd.grad(main+0*aux,[v for _,v in named],retain_graph=True)
            self.assertTrue(all(torch.equal(u,v) for u,v in zip(a,b)))
            headsum=sum(float(v.detach().square().sum()) for (n,_),v in zip(named,a) if n.startswith('kernel_head.'))
            if step==0:self.assertEqual(headsum,0.)
            else:self.assertGreater(headsum,0.)
            main.backward();opt.step()
    def test_10_no_future_input_and_reconstruction(self):
        for j in self.jobs():
            m=self.model(j);m.eval();x,y=self.xy(j['horizon']);p=m(x);target=y.clone().requires_grad_();aux,gold=kernel_supervision(m.predicted_gram,target)
            self.assertFalse(gold.requires_grad);self.assertIsNone(torch.autograd.grad(aux,target,allow_unused=True)[0]);self.assertTrue(torch.equal(p,m(x)))
            c=m.predict_components(x);scale,offset=m.inverse_coordinates();physical=c*scale.unsqueeze(-1)
            self.assertLess(float((physical[...,:4].sum(-1)+physical[...,4:].sum(-1)+offset-m.normalize(c[...,:4].sum(-1)+c[...,4:].sum(-1),'denorm')).detach().abs().max()),2e-5)
    def test_11_parallel_early_zero_and_gram_psd(self):
        a=torch.randn(2,3,2,dtype=torch.float64);u=torch.randn(2,12,3,2,dtype=torch.float64,requires_grad=True);h=torch.randn_like(a,requires_grad=True)
        out=parallel_future_states(a,u,h);g=torch.autograd.grad(out[:,-3:].square().sum(),u)[0];self.assertEqual(torch.count_nonzero(g[:,:9]),0)
        K=value_gram(torch.randn(2,180,4,dtype=torch.float64));self.assertTrue(torch.equal(K,K.transpose(-1,-2)));self.assertGreater(float(torch.linalg.eigvalsh(K).min()),-1e-10);self.assertTrue(torch.equal(K.diagonal(dim1=-2,dim2=-1),torch.ones(2,180,dtype=K.dtype)))
    def test_12_dataset_and_positions(self):
        from tmp.run_mafirst_kernel_transfer_ablation_20261022 import data_for
        for j in self.jobs():
            tr,va,sc,data=data_for(j);self.assertEqual((len(tr),len(va)),(j['train_windows'],j['val_test_windows']));self.assertEqual(tuple(tr[0][1].shape),(j['horizon'],7));self.assertEqual(int(np.ceil(len(tr)/32)),j['steps'])
            m=self.model(j);old=torch.load((C720 if j['horizon']==720 else C336)/'initial.pt',map_location='cpu',weights_only=True)['time_position_mmd']
            for n,v in old.items():
                if 'position' in n:self.assertTrue(torch.equal(v,m.state_dict()[n]))
if __name__=='__main__':unittest.main()
