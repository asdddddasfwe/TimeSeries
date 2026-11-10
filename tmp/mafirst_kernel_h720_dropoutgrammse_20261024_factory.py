"""Identical direct-Gram model; all258 scratch and all226 optimizer rules retained."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,torch
from models.MAFirstKernelH720DirectGram import Model
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
R=Path(__file__).resolve().parents[1]
G=R/'tmp/mafirst_kernel_h720_dropoutgrammse_group_20261024'
PLAN=R/'tmp/mafirst_kernel_h720_dropoutgrammse_plan_20261024.json'
B=R/'tmp/mafirst_kernel_h720_localcs_20261023_full_seed2021'
OLDG=R/'tmp/mafirst_kernel_h720_localcs_group_20261023'
C720=R/'tmp/mafirst_mmd_ettm1_20261016_h720_seed2021_max30_patience5'
AFFINE_NAMES=('normalize.affine_weight','normalize.affine_bias')
REMOVED=();ADDED=()
def fresh(H,job_id,logdir,verify_reference=True):
    plan=json.loads(PLAN.read_text());job=next(v for v in plan['jobs'] if v['id']==job_id)
    assert H==job['horizon']==720 and job['feedback_enabled'] and job['aux_weight']==job['alignment_weight']==job['output_structure_weight']==1
    path=OLDG/'reference_initial/h720.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()=='1b63d64665cc614329efdeb9c7880cf73b5f44cdf556b2b80fa713092151a5aa'
    source=torch.load(path,map_location='cpu',weights_only=True);assert len(source)==258
    cfg=json.loads((B/'model_config.json').read_text());cfg.update(model='MAFirstKernelH720DirectGram',study_arm='kernel',study_affine=True,study_parallel=True,study_feedback_enabled=True,study_aux_weight=1.,log_dir=str(logdir),tensorboard=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2021);m=Model(SimpleNamespace(**cfg))
    assert m.state_dict().keys()==source.keys();m.load_state_dict(source,strict=True)
    assert all(torch.equal(v,m.state_dict()[k]) and v.data_ptr()!=m.state_dict()[k].data_ptr() for k,v in source.items())
    original=torch.load(C720/'initial.pt',map_location='cpu',weights_only=True)['time_position_mmd'];assert len(original)==236 and all(torch.equal(v,m.state_dict()[k]) for k,v in original.items())
    if verify_reference:
        reference=torch.load(G/'reference_initial/h720.pt',map_location='cpu',weights_only=True);assert len(reference)==258 and m.state_dict().keys()==reference.keys() and all(torch.equal(v,reference[k]) for k,v in m.state_dict().items())
    count=(sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers())))
    assert count==(388016,226,258,32),count
    assert not hasattr(m,'coarse_normalizers') and [n for n,p in m.named_parameters() if n.endswith(('affine_weight','affine_bias'))]==list(AFFINE_NAMES)
    m.initialization_audit=dict(status='verified',horizon=720,job=job_id,source_reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),all258_reference_exact=verify_reference,all258_shared_scratch_exact=True,old236_exact=True,no_added_or_removed_parameters=True,independent_storage=True,trained_or_preflight_weights_loaded=False,parameters=count[0],parameter_tensors=count[1],state_tensors=count[2],buffers=count[3],feedback_enabled=True,aux_weight=1.,alignment_weight=1.,parallel=True,affine=True,complete_single_revin=True,one_affine_pair=True,direct_matrix_head=True,predicted_PSD_enforced=False,raw_scale_conditioning=True)
    return m,SimpleNamespace(**cfg)
def optimizer_groups(m):
    groups=norm_bias_no_decay_groups(m,1e-4);names={id(p):n for n,p in m.named_parameters()};affine={id(p) for n,p in m.named_parameters() if n in AFFINE_NAMES};assert len(affine)==2
    result=[];protected=[]
    for g in groups:
        regular=[p for p in g['params'] if id(p) not in affine];protected.extend(p for p in g['params'] if id(p) in affine)
        if regular:result.append({**g,'params':regular})
    result.append(dict(params=protected,weight_decay=0.,group_name='single_revin_affine_no_decay'))
    current={names[id(p)]:g['weight_decay'] for g in result for p in g['params']};prior={n:g['weight_decay'] for g in json.loads((B/'optimizer_groups.json').read_text()) for n in g['names']}
    assert len(current)==len(prior)==226 and current==prior and all(current[n]==0 for n in AFFINE_NAMES)
    ids=[id(p) for g in result for p in g['params']];assert len(ids)==len(set(ids))==226
    return result
