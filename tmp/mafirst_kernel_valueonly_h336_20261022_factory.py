"""Each fresh run loads only all256 registered kernel SCRATCH states, never best/preflight."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,torch
from models.MAFirstKernelValueOnly336 import Model
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
R=Path(__file__).resolve().parents[1]
G=R/'tmp/mafirst_kernel_valueonly_group_20261022_h336_seed2021'
CONTROL=R/'tmp/mafirst_mmd_ettm1_20261016_h336_seed2021_max30_patience5'
BASE=R/'tmp/mafirst_affine_kernel_20261022_kernel_h336_seed2021'
KERNEL_INITIAL_SHA='e242ae5c57fdc7a43b52676823f59e0898d407f9928eaaa4b6003355fc87b7d2'

def fresh(arm,logdir,verify_reference=True):
    assert arm in ('recurrent','parallel')
    path=BASE/'initial.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==KERNEL_INITIAL_SHA
    cfg=json.loads((BASE/'model_config.json').read_text());cfg.update(model='MAFirstKernelValueOnly336',study_arm='kernel',study_affine=False,study_parallel=arm=='parallel',log_dir=str(logdir),tensorboard=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2021);m=Model(SimpleNamespace(**cfg))
    scratch=torch.load(path,map_location='cpu',weights_only=True);assert len(scratch)==256
    m.load_state_dict(scratch,strict=True)
    assert all(torch.equal(v,m.state_dict()[k]) and v.data_ptr()!=m.state_dict()[k].data_ptr() for k,v in scratch.items())
    old=torch.load(CONTROL/'initial.pt',map_location='cpu',weights_only=True)['time_position_mmd']
    assert len(old)==236 and all(torch.equal(v,m.state_dict()[k]) for k,v in old.items())
    if verify_reference:
        ref=G/'reference_initial/shared_initial.pt';assert ref.exists()
        gold=torch.load(ref,map_location='cpu',weights_only=True);assert gold.keys()==m.state_dict().keys() and all(torch.equal(v,gold[k]) for k,v in m.state_dict().items())
    m.initialization_audit=dict(status='verified',source_initial_sha256=KERNEL_INITIAL_SHA,old236_exact=True,all256_kernel_scratch_exact=True,arm=arm,affine=False,parallel=arm=='parallel',parameters=sum(p.numel() for p in m.parameters()),parameter_tensors=len(list(m.parameters())),state_tensors=len(m.state_dict()),buffers=len(list(m.buffers())),trained_or_preflight_weights_loaded=False)
    assert (m.initialization_audit['parameters'],m.initialization_audit['parameter_tensors'],m.initialization_audit['state_tensors'],m.initialization_audit['buffers'])==(240786,224,256,32)
    return m,SimpleNamespace(**cfg)

def optimizer_groups(m):return norm_bias_no_decay_groups(m,1e-4)
