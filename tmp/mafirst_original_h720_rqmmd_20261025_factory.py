"""Original model, common untrained scratch, original optimizer; no new parameters."""
import torch
from types import SimpleNamespace
from tmp.mafirst_original_h720_rqmmd_20261025_common import *
from scripts import run_mafirst_mmd_ettm1_h720_seed2021 as original
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
def fresh(logdir):
 cfg=SimpleNamespace(**read(BASE/'protocol.json')['model_config']);cfg.log_dir=str(logdir);cfg.root_path=str(G/'train_val_data')
 model=original.initialize_pair(cfg)['time_position_mmd']
 ref=torch.load(G/'reference_initial.pt',map_location='cpu',weights_only=True)
 assert len(ref)==len(model.state_dict())==236
 assert all(torch.equal(v,ref[n]) and v.data_ptr()!=ref[n].data_ptr() for n,v in model.state_dict().items())
 assert sum(p.numel() for p in model.parameters())==320450 and len(list(model.parameters()))==204 and len(list(model.buffers()))==32
 assert not model.normalize.affine and all(not n.affine for n in model.coarse_normalizers)
 return model,cfg
def optimizer_groups(m):return norm_bias_no_decay_groups(m,1e-4)
def groups_audit(m,groups):
 names={id(p):n for n,p in m.named_parameters()}
 result=[dict(group_name=g['group_name'],weight_decay=g['weight_decay'],names=[names[id(p)] for p in g['params']]) for g in groups]
 assert result==read(BASE/'optimizer_groups.json')['time_position_mmd']
 return result
