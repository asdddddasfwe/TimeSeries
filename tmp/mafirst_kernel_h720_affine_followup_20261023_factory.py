"""H720 affine-only factory: all old256 scratch states exact, new8 identity affine."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,torch
from models.MAFirstKernelH720AffineFollowup import Model
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
R=Path(__file__).resolve().parents[1]
G=R/'tmp/mafirst_kernel_h720_affine_followup_group_20261023'
PLAN=R/'tmp/mafirst_kernel_h720_affine_followup_plan_20261023.json'
B=R/'tmp/mafirst_kernel_transfer_ablation_20261022_h720_full_seed2021'
OLDG=R/'tmp/mafirst_kernel_transfer_ablation_group_20261022'
C720=R/'tmp/mafirst_mmd_ettm1_20261016_h720_seed2021_max30_patience5'
AFFINE_NAMES=tuple(f'{prefix}.{name}' for prefix in ('normalize','coarse_normalizers.0','coarse_normalizers.1','coarse_normalizers.2') for name in ('affine_weight','affine_bias'))
def fresh(H,job_id,logdir,verify_reference=True):
    plan=json.loads(PLAN.read_text());job=next(v for v in plan['jobs'] if v['id']==job_id)
    assert H==job['horizon']==720 and job['feedback_enabled'] and job['aux_weight']==1
    path=OLDG/'reference_initial/h720.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='3c0c0a7ca301def47cd5d9a1d7fe4c9b6b3483279272347cfd18f9efb4ae7ebf'
    old=torch.load(path,map_location='cpu',weights_only=True);assert len(old)==256
    source=C720/'initial.pt';assert hashlib.sha256(source.read_bytes()).hexdigest()=='e6a685e3c5bd855db6b1bdec6172a6b172a3a5eb415e1ed4fd9212118807dcc4'
    original=torch.load(source,map_location='cpu',weights_only=True)['time_position_mmd'];assert len(original)==236
    cfg=json.loads((B/'model_config.json').read_text())
    cfg.update(model='MAFirstKernelH720AffineFollowup',study_arm='kernel',study_affine=True,study_parallel=True,study_feedback_enabled=True,study_aux_weight=1.0,log_dir=str(logdir),tensorboard=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2021);m=Model(SimpleNamespace(**cfg))
    state=m.state_dict();assert set(state)-set(old)==set(AFFINE_NAMES)
    state.update(old);m.load_state_dict(state,strict=True)
    assert all(torch.equal(v,m.state_dict()[k]) and v.data_ptr()!=m.state_dict()[k].data_ptr() for k,v in old.items())
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in original.items())
    assert all(torch.equal(m.state_dict()[n],torch.ones(7) if n.endswith('weight') else torch.zeros(7)) for n in AFFINE_NAMES)
    if verify_reference:
        gold=torch.load(G/'reference_initial/h720.pt',map_location='cpu',weights_only=True)
        assert m.state_dict().keys()==gold.keys() and all(torch.equal(v,gold[k]) for k,v in m.state_dict().items())
    count=(sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers())))
    assert count==(385226,232,264,32),count
    m.initialization_audit=dict(status='verified',horizon=720,job=job_id,source_reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),old236_exact=True,old256_reference_exact=True,all264_reference_exact=verify_reference,new8_identity_affine=True,independent_storage=True,trained_or_preflight_weights_loaded=False,parameters=count[0],parameter_tensors=count[1],state_tensors=count[2],buffers=count[3],feedback_enabled=True,aux_weight=1.,parallel=True,affine=True)
    return m,SimpleNamespace(**cfg)
def optimizer_groups(m):
    groups=norm_bias_no_decay_groups(m,1e-4)
    affine={id(p) for n,p in m.named_parameters() if n in AFFINE_NAMES};assert len(affine)==8
    result=[];protected=[]
    for g in groups:
        regular=[p for p in g['params'] if id(p) not in affine]
        protected.extend(p for p in g['params'] if id(p) in affine)
        if regular:result.append({**g,'params':regular})
    result.append(dict(params=protected,weight_decay=0.,group_name='followup_affine_no_decay'))
    names={id(p):n for n,p in m.named_parameters()};current={names[id(p)]:g['weight_decay'] for g in result for p in g['params']}
    expected={n:g['weight_decay'] for g in json.loads((B/'optimizer_groups.json').read_text()) for n in g['names']}
    assert len(expected)==224 and len(current)==232 and all(current[n]==wd for n,wd in expected.items())
    assert all(current[n]==0 for n in AFFINE_NAMES)
    actual=[id(p) for g in result for p in g['params']];assert len(actual)==len(set(actual))==232
    return result
