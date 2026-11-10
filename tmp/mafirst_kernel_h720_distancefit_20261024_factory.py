"""Original258 common scratch exact; newzero delta720x720, no trained weights."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,torch
from models.MAFirstKernelH720GlobalJoint import Model
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
R=Path(__file__).resolve().parents[1]
G=R/'tmp/mafirst_kernel_h720_distancefit_group_20261024'
PLAN=R/'tmp/mafirst_kernel_h720_distancefit_plan_20261024.json'
B=R/'tmp/mafirst_kernel_h720_dropoutgrammse_20261024_h720_drop_output_mse_full_seed2021'
SOURCE=R/'tmp/mafirst_kernel_h720_dropoutgrammse_group_20261024/reference_initial/h720.pt'
C720=R/'tmp/mafirst_mmd_ettm1_20261016_h720_seed2021_max30_patience5'
AFFINE_NAMES=('normalize.affine_weight','normalize.affine_bias')
ADDED=('global_seasonal_delta',);REMOVED=()
def fresh(H,job_id,logdir,verify_reference=True):
    job=next(j for j in json.loads(PLAN.read_text())['jobs'] if j['id']==job_id)
    assert H==job['horizon']==720 and job['feedback_enabled'] and job['aux_weight']==job['alignment_weight']==job['output_structure_weight']==1
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='1b63d64665cc614329efdeb9c7880cf73b5f44cdf556b2b80fa713092151a5aa'
    source=torch.load(SOURCE,map_location='cpu',weights_only=True);assert len(source)==258
    cfg=json.loads((B/'model_config.json').read_text());cfg.update(model='MAFirstKernelH720GlobalJoint',study_arm='kernel',study_affine=True,study_parallel=True,study_feedback_enabled=True,study_aux_weight=1.,log_dir=str(logdir),tensorboard=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2021);m=Model(SimpleNamespace(**cfg))
    result=m.load_state_dict(source,strict=False);assert result.missing_keys==list(ADDED) and result.unexpected_keys==[]
    assert set(m.state_dict())==set(source)|set(ADDED) and torch.count_nonzero(m.global_seasonal_delta)==0
    assert all(torch.equal(v,m.state_dict()[k]) and v.data_ptr()!=m.state_dict()[k].data_ptr() for k,v in source.items())
    original=torch.load(C720/'initial.pt',map_location='cpu',weights_only=True)['time_position_mmd'];assert len(original)==236 and all(torch.equal(v,m.state_dict()[k]) for k,v in original.items())
    if verify_reference:
        reference=torch.load(G/'reference_initial/h720.pt',map_location='cpu',weights_only=True)
        assert len(reference)==259 and m.state_dict().keys()==reference.keys() and all(torch.equal(v,reference[k]) for k,v in m.state_dict().items())
    count=(sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers())))
    assert count==(906416,227,259,32),count
    assert not hasattr(m,'coarse_normalizers') and [n for n,p in m.named_parameters() if n.endswith(('affine_weight','affine_bias'))]==list(AFFINE_NAMES)
    m.initialization_audit=dict(status='verified',horizon=720,job=job_id,source_reference_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),all259_reference_exact=verify_reference,all258_shared_scratch_exact=True,old236_exact=True,one_new_zero_delta=False,new_parameters=0,parameter_budget_matched=True,same_architecture_as_D=True,independent_storage=True,trained_or_preflight_weights_loaded=False,parameters=count[0],parameter_tensors=count[1],state_tensors=count[2],buffers=count[3],feedback_enabled=True,aux_weight=1.,alignment_weight=1.,parallel=True,affine=True,complete_single_revin=True,one_affine_pair=True,direct_matrix_head=True,predicted_PSD_enforced=False,raw_scale_conditioning=True,global_joint_output=True,effective_initial_identity=True)
    return m,SimpleNamespace(**cfg)
def optimizer_groups(m):
    groups=norm_bias_no_decay_groups(m,1e-4);names={id(p):n for n,p in m.named_parameters()};affine={id(p) for n,p in m.named_parameters() if n in AFFINE_NAMES};assert len(affine)==2
    result=[];protected=[]
    for g in groups:
        regular=[p for p in g['params'] if id(p) not in affine];protected.extend(p for p in g['params'] if id(p) in affine)
        if regular:result.append({**g,'params':regular})
    result.append(dict(params=protected,weight_decay=0.,group_name='single_revin_affine_no_decay'))
    current={names[id(p)]:g['weight_decay'] for g in result for p in g['params']};prior={n:g['weight_decay'] for g in json.loads((B/'optimizer_groups.json').read_text()) for n in g['names']}
    assert len(current)==227 and len(prior)==226 and {n:v for n,v in current.items() if n not in ADDED}==prior
    assert current['global_seasonal_delta']==1e-4 and all(current[n]==0 for n in AFFINE_NAMES)
    ids=[id(p) for g in result for p in g['params']];assert len(ids)==len(set(ids))==227
    return result
