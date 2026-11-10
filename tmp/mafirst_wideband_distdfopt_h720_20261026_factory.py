"""Original model, common untrained scratch, original optimizer; no new parameters."""
import torch
from types import SimpleNamespace
from tmp.mafirst_wideband_distdfopt_h720_20261026_common import *
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
def groups_audit(m,groups):
 names={id(p):n for n,p in m.named_parameters()}
 assert len(groups)==1 and len(groups[0]['params'])==204
 g=groups[0];assert g['weight_decay']==0 and g['eps']==1e-8 and g['betas'][1]==.999 and not g['amsgrad']
 return [dict(group_name='official_all_parameters',weight_decay=0.,names=[names[id(p)] for p in g['params']])]
