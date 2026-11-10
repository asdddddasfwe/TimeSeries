"""No trained weights: horizon-specific scratch/reference CPU-exact initialization."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,torch
from models.MAFirstKernelTransferAblation import Model
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
R=Path(__file__).resolve().parents[1];G=R/'tmp/mafirst_kernel_transfer_ablation_group_20261022'
PLAN=R/'tmp/mafirst_kernel_transfer_ablation_plan_20261022.json'
B336=R/'tmp/mafirst_affine_kernel_20261022_kernel_h336_seed2021'
C720=R/'tmp/mafirst_mmd_ettm1_20261016_h720_seed2021_max30_patience5'
C336=R/'tmp/mafirst_mmd_ettm1_20261016_h336_seed2021_max30_patience5'
def fresh(H,job_id,logdir,verify_reference=True):
    plan=json.loads(PLAN.read_text());job=next(v for v in plan['jobs'] if v['id']==job_id);assert H==job['horizon']
    if H==720:
        path=C720/'initial.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()=='e6a685e3c5bd855db6b1bdec6172a6b172a3a5eb415e1ed4fd9212118807dcc4'
        cfg=json.loads((C720/'protocol.json').read_text())['model_config']
        old=torch.load(path,map_location='cpu',weights_only=True)['time_position_mmd'];assert len(old)==236
    else:
        assert H==336;path=B336/'initial.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()=='e242ae5c57fdc7a43b52676823f59e0898d407f9928eaaa4b6003355fc87b7d2'
        cfg=json.loads((B336/'model_config.json').read_text());old=torch.load(C336/'initial.pt',map_location='cpu',weights_only=True)['time_position_mmd']
    cfg.update(model='MAFirstKernelTransferAblation',study_arm='kernel',study_affine=False,study_parallel=True,study_feedback_enabled=job['feedback_enabled'],study_aux_weight=job['aux_weight'],log_dir=str(logdir),tensorboard=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2021);m=Model(SimpleNamespace(**cfg))
    if H==720:
        state=m.state_dict();assert old.keys()<=state.keys();state.update(old);m.load_state_dict(state,strict=True)
    else:m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    assert all(torch.equal(v,m.state_dict()[k]) and v.data_ptr()!=m.state_dict()[k].data_ptr() for k,v in old.items())
    if verify_reference:
        gold=torch.load(G/'reference_initial'/f'h{H}.pt',map_location='cpu',weights_only=True)
        assert m.state_dict().keys()==gold.keys() and all(torch.equal(v,gold[k]) for k,v in m.state_dict().items())
    count=(sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers())))
    assert count==((385170,224,256,32) if H==720 else (240786,224,256,32)),count
    m.initialization_audit=dict(status='verified',horizon=H,job=job_id,source_initial_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),old236_exact=True,all256_reference_exact=verify_reference,complete256_h336_kernel_scratch_exact=H==336,new20_h720_fresh_seed2021=H==720,independent_storage=True,trained_or_preflight_weights_loaded=False,parameters=count[0],parameter_tensors=count[1],state_tensors=count[2],buffers=count[3],feedback_enabled=job['feedback_enabled'],aux_weight=job['aux_weight'],parallel=True,affine=False)
    return m,SimpleNamespace(**cfg)
def optimizer_groups(m):return norm_bias_no_decay_groups(m,1e-4)
