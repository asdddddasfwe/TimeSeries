"""Original untrained scratch for common tensors; independent new seasonal encoder."""
import torch
from types import SimpleNamespace
from tmp.mafirst_season_sliding_distdfopt_h720_20261027_common import *
from scripts import run_mafirst_mmd_ettm1_h720_seed2021 as original
from models.OscillatorPDMStatePatchMAFirstSeasonSliding import Model
MODEL_NAME='OscillatorPDMStatePatchMAFirstSeasonSliding'
def configuration(logdir):
 cfg=SimpleNamespace(**read(BASE/'protocol.json')['model_config'])
 cfg.model=MODEL_NAME;cfg.log_dir=str(logdir);cfg.root_path=str(G/'train_val_data')
 assert cfg.d_model==32 and cfg.d_ff==64 and not hasattr(cfg,'cf_dim')
 return cfg
def untrained(logdir):
 cfg=configuration(logdir)
 source=original.initialize_pair(cfg)['time_position_mmd']
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(20211014);model=Model(cfg)
 state=model.state_dict();common={n:v for n,v in source.state_dict().items() if n in state and v.shape==state[n].shape}
 state.update({n:v.clone() for n,v in common.items()});model.load_state_dict(state)
 assert all(torch.equal(model.state_dict()[n],v) and model.state_dict()[n].data_ptr()!=v.data_ptr() for n,v in common.items())
 assert not model.normalize.affine and all(not n.affine for n in model.coarse_normalizers)
 return model,cfg
def fresh(logdir):
 model,cfg=untrained(logdir);ref=torch.load(G/'reference_initial.pt',map_location='cpu',weights_only=True);card=read(G/'expected_model_card.json')
 assert set(ref)==set(model.state_dict()) and all(torch.equal(v,ref[n]) and v.data_ptr()!=ref[n].data_ptr() for n,v in model.state_dict().items())
 assert sum(p.numel() for p in model.parameters())==card['parameters'] and len(list(model.parameters()))==card['parameter_tensors']
 return model,cfg
def groups_audit(m,groups):
 names={id(p):n for n,p in m.named_parameters()};assert len(groups)==1 and len(groups[0]['params'])==len(names)
 g=groups[0];assert g['weight_decay']==0 and g['eps']==1e-8 and g['betas'][1]==.999 and not g['amsgrad']
 assert {id(p) for p in g['params']}==set(names)
 return [dict(group_name='official_all_parameters',weight_decay=0.,names=[names[id(p)] for p in g['params']])]
