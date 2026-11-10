"""Single full RevIN; 258 retained scratch states, 226 active parameter tensors."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,torch
from models.MAFirstKernelH720SingleRevIN import Model
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
R=Path(__file__).resolve().parents[1]
G=R/'tmp/mafirst_kernel_h720_singlerevin_group_20261023'
PLAN=R/'tmp/mafirst_kernel_h720_singlerevin_plan_20261023.json'
B=R/'tmp/mafirst_kernel_h720_prenorm_20261023_full_seed2021'
OLDG=R/'tmp/mafirst_kernel_h720_affine_followup_group_20261023'
C720=R/'tmp/mafirst_mmd_ettm1_20261016_h720_seed2021_max30_patience5'
AFFINE_NAMES=('normalize.affine_weight','normalize.affine_bias')
REMOVED=tuple(f'coarse_normalizers.{i}.{n}' for i in range(3) for n in ('affine_weight','affine_bias'))
def fresh(H,job_id,logdir,verify_reference=True):
    plan=json.loads(PLAN.read_text());job=next(v for v in plan['jobs'] if v['id']==job_id)
    assert H==job['horizon']==720 and job['feedback_enabled'] and job['aux_weight']==1
    path=OLDG/'reference_initial/h720.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()=='eb87f08821b186f4a25283b428f76d03c51104a5e0ea356d110d1293a967d07c'
    source=torch.load(path,map_location='cpu',weights_only=True);assert len(source)==264
    kept={k:v for k,v in source.items() if k not in REMOVED};assert len(kept)==258 and set(source)-set(kept)==set(REMOVED)
    original=torch.load(C720/'initial.pt',map_location='cpu',weights_only=True)['time_position_mmd'];assert len(original)==236
    cfg=json.loads((B/'model_config.json').read_text());cfg.update(model='MAFirstKernelH720SingleRevIN',study_arm='kernel',study_affine=True,study_parallel=True,study_feedback_enabled=True,study_aux_weight=1.,log_dir=str(logdir),tensorboard=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2021);m=Model(SimpleNamespace(**cfg))
    m.load_state_dict(kept,strict=True)
    assert all(torch.equal(v,m.state_dict()[k]) and v.data_ptr()!=m.state_dict()[k].data_ptr() for k,v in kept.items())
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in original.items())
    if verify_reference:
        gold=torch.load(G/'reference_initial/h720.pt',map_location='cpu',weights_only=True)
        assert m.state_dict().keys()==gold.keys() and all(torch.equal(v,gold[k]) for k,v in m.state_dict().items())
    count=(sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers())))
    assert count==(385184,226,258,32),count
    assert not hasattr(m,'coarse_normalizers') and [n for n,p in m.named_parameters() if n.endswith(('affine_weight','affine_bias'))]==list(AFFINE_NAMES)
    m.initialization_audit=dict(status='verified',horizon=720,job=job_id,source_reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),all258_reference_exact=verify_reference,shared258_prior_scratch_exact=True,old256_reference_exact=True,old236_exact=True,independent_storage=True,trained_or_preflight_weights_loaded=False,parameters=count[0],parameter_tensors=count[1],state_tensors=count[2],buffers=count[3],feedback_enabled=True,aux_weight=1.,parallel=True,affine=True,complete_single_revin=True,one_affine_pair=True,removed6_coarse_affine_tensors=list(REMOVED),removed_scalar_parameters=42)
    return m,SimpleNamespace(**cfg)
def optimizer_groups(m):
    groups=norm_bias_no_decay_groups(m,1e-4);names={id(p):n for n,p in m.named_parameters()};affine={id(p) for n,p in m.named_parameters() if n in AFFINE_NAMES};assert len(affine)==2
    result=[];protected=[]
    for g in groups:
        regular=[p for p in g['params'] if id(p) not in affine];protected.extend(p for p in g['params'] if id(p) in affine)
        if regular:result.append({**g,'params':regular})
    result.append(dict(params=protected,weight_decay=0.,group_name='single_revin_affine_no_decay'))
    current={names[id(p)]:g['weight_decay'] for g in result for p in g['params']}
    expected={n:g['weight_decay'] for g in json.loads((B/'optimizer_groups.json').read_text()) for n in g['names'] if n not in REMOVED}
    assert len(current)==len(expected)==226 and current==expected and all(current[n]==0 for n in AFFINE_NAMES)
    actual=[id(p) for g in result for p in g['params']];assert len(actual)==len(set(actual))==226
    return result
