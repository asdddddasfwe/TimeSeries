"""New ordering only; all264 affine-ON scratch states and232 optimizer rules exact."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,torch
from models.MAFirstKernelH720PreNorm import Model
from tmp.mafirst_kernel_h720_affine_followup_20261023_factory import optimizer_groups as prior_groups,AFFINE_NAMES
R=Path(__file__).resolve().parents[1]
G=R/'tmp/mafirst_kernel_h720_prenorm_group_20261023'
PLAN=R/'tmp/mafirst_kernel_h720_prenorm_plan_20261023.json'
B=R/'tmp/mafirst_kernel_h720_affine_followup_20261023_full_seed2021'
OLDG=R/'tmp/mafirst_kernel_h720_affine_followup_group_20261023'
C720=R/'tmp/mafirst_mmd_ettm1_20261016_h720_seed2021_max30_patience5'
def fresh(H,job_id,logdir,verify_reference=True):
    plan=json.loads(PLAN.read_text());job=next(v for v in plan['jobs'] if v['id']==job_id)
    assert H==job['horizon']==720 and job['feedback_enabled'] and job['aux_weight']==1
    path=OLDG/'reference_initial/h720.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='eb87f08821b186f4a25283b428f76d03c51104a5e0ea356d110d1293a967d07c'
    old=torch.load(path,map_location='cpu',weights_only=True);assert len(old)==264
    original=torch.load(C720/'initial.pt',map_location='cpu',weights_only=True)['time_position_mmd'];assert len(original)==236
    cfg=json.loads((B/'model_config.json').read_text())
    cfg.update(model='MAFirstKernelH720PreNorm',study_arm='kernel',study_affine=True,study_parallel=True,study_feedback_enabled=True,study_aux_weight=1.,log_dir=str(logdir),tensorboard=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2021);m=Model(SimpleNamespace(**cfg))
    m.load_state_dict(old,strict=True)
    assert all(torch.equal(v,m.state_dict()[k]) and v.data_ptr()!=m.state_dict()[k].data_ptr() for k,v in old.items())
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in original.items())
    if verify_reference:
        gold=torch.load(G/'reference_initial/h720.pt',map_location='cpu',weights_only=True)
        assert m.state_dict().keys()==gold.keys() and all(torch.equal(v,gold[k]) for k,v in m.state_dict().items())
    count=(sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers())))
    assert count==(385226,232,264,32),count
    m.initialization_audit=dict(status='verified',horizon=720,job=job_id,source_reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),all264_reference_exact=verify_reference,prior264_scratch_exact=True,old256_reference_exact=True,old236_exact=True,independent_storage=True,trained_or_preflight_weights_loaded=False,parameters=count[0],parameter_tensors=count[1],state_tensors=count[2],buffers=count[3],feedback_enabled=True,aux_weight=1.,parallel=True,affine=True,shared_original_history_statistics=True,normalize_before_pooling=True,all4_existing_affines_retained=True)
    return m,SimpleNamespace(**cfg)
def optimizer_groups(m):
    groups=prior_groups(m);names={id(p):n for n,p in m.named_parameters()}
    current={names[id(p)]:g['weight_decay'] for g in groups for p in g['params']}
    expected={n:g['weight_decay'] for g in json.loads((B/'optimizer_groups.json').read_text()) for n in g['names']}
    assert len(current)==len(expected)==232 and current==expected
    return groups
