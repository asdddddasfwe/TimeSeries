"""Load only verified H336 original SCRATCH; additions use isolated seed2021."""
from pathlib import Path
from types import SimpleNamespace
import hashlib, json
import torch
from models.MAFirstAffineKernel336 import Model
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
R=Path(__file__).resolve().parents[1]
CONTROL=R/'tmp/mafirst_mmd_ettm1_20261016_h336_seed2021_max30_patience5'
G=R/'tmp/mafirst_affine_kernel_group_20261022_h336_seed2021'
INITIAL_SHA256='df6714f6f16f144ccbd25286c40c813642066ca530737c8cc2877f8ee3b657b0'

def fresh(arm, affine, logdir, verify_reference=True):
    initial=CONTROL/'initial.pt'
    assert hashlib.sha256(initial.read_bytes()).hexdigest()==INITIAL_SHA256
    cfg=json.loads((CONTROL/'protocol.json').read_text())['model_config']
    cfg.update(model='MAFirstAffineKernel336',study_arm=arm,study_affine=bool(affine),log_dir=str(logdir),tensorboard=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2021);m=Model(SimpleNamespace(**cfg))
    source=torch.load(initial,map_location='cpu',weights_only=True)['time_position_mmd']
    assert len(source)==236
    state=m.state_dict();assert source.keys()<=state.keys()
    state.update(source);m.load_state_dict(state,strict=True)
    assert all(torch.equal(v,m.state_dict()[k]) and v.data_ptr()!=m.state_dict()[k].data_ptr() for k,v in source.items())
    ref=G/'reference_initial'/f'{arm}_affine{int(affine)}.pt'
    if verify_reference and ref.exists():
        gold=torch.load(ref,map_location='cpu',weights_only=True)
        assert gold.keys()==m.state_dict().keys()
        assert all(torch.equal(v,gold[k]) for k,v in m.state_dict().items())
    m.initialization_audit=dict(status='verified',source_initial_sha256=INITIAL_SHA256,old236_exact=True,
        new_seed=2021,arm=arm,affine=bool(affine),parameters=sum(p.numel() for p in m.parameters()),
        parameter_tensors=len(list(m.parameters())),state_tensors=len(m.state_dict()),buffers=len(list(m.buffers())),
        added_state_names=sorted(set(state)-set(source)),trained_or_preflight_weights_loaded=False)
    return m,SimpleNamespace(**cfg)

def optimizer_groups(m):
    groups=norm_bias_no_decay_groups(m,1e-4)
    affine={id(p) for n,p in m.named_parameters() if n.endswith(('affine_weight','affine_bias'))}
    result=[]
    for group in groups:
        regular=[p for p in group['params'] if id(p) not in affine]
        protected=[p for p in group['params'] if id(p) in affine]
        if regular:result.append({**group,'params':regular})
        if protected:result.append({**group,'params':protected,'weight_decay':0.,'group_name':'revin_affine_no_decay'})
    ids=[id(p) for g in result for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(m.parameters()))
    return result
