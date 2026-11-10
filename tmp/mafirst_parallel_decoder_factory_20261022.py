"""Same 236 original SCRATCH state tensors for all new runs, never best/preflight."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import torch
from models.MAFirstParallelDecoder import Model
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
INITIAL_SHA256="6a572882d3161da56009437cad7a6251682f2688492e68dc4bd05d3cfc2c8086"

def load_original_scratch(root,logdir):
    root=Path(root);control=root/'tmp/mafirst_memory_factorial_20261017_full_h720_control_seed2021'
    initial=control/'initial.pt';assert hashlib.sha256(initial.read_bytes()).hexdigest()==INITIAL_SHA256
    cfg=json.loads((control/'model_config.json').read_text())
    cfg.update(model="MAFirstParallelDecoder",log_dir=str(logdir),tensorboard=False,
               factor_slow_memory=False,factor_readout_no_decay=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20211014);m=Model(SimpleNamespace(**cfg))
    state=torch.load(initial,map_location="cpu",weights_only=True);m.load_state_dict(state,strict=True)
    assert len(m.state_dict())==236 and len(list(m.parameters()))==204
    assert sum(p.numel() for p in m.parameters())==320450
    assert all(torch.equal(v,state[k]) and v.data_ptr()!=state[k].data_ptr() for k,v in m.state_dict().items())
    registered=root/'tmp/mafirst_parallel_decoder_group_20261022_h720_seed2021/reference_initial/initial.pt'
    if registered.exists():
        gold=torch.load(registered,map_location="cpu",weights_only=True)
        assert gold.keys()==m.state_dict().keys()
        assert all(torch.equal(v,gold[k]) for k,v in m.state_dict().items())
    m.initialization_audit=dict(status="verified",source_initial_sha256=INITIAL_SHA256,
        copied=list(state),copied_tensors=236,retained_exact=True,parameters=320450,
        parameter_tensors=204,buffers=32,state_tensors=236,new_parameters=0,
        trained_or_preflight_weights_loaded=False,initial_function_equal_to_control=False)
    return m,cfg

def optimizer_groups(m):
    return norm_bias_no_decay_groups(m,1e-4)
