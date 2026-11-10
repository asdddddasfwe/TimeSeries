"""Official H720 width and native weights; matched local training configuration."""
import torch
from types import SimpleNamespace
from tmp.distdf_officialwidth_h720_20261026_common import *
from scripts import run_distdf_fredformer_matched as source
from utils.distdf_joint_wasserstein import distdf_loss
def config():return SimpleNamespace(**read(PLAN)['distdf']['config'])
def fresh(logdir):
 c=config();m=source.Model(c);ref=torch.load(G/'reference_initial/distdf.pt',map_location='cpu',weights_only=True)
 assert m.state_dict().keys()==ref.keys()
 assert all(torch.equal(v,ref[n]) and v.data_ptr()!=ref[n].data_ptr() for n,v in m.state_dict().items())
 return m,c
def optimizer_groups(m):return source.optimizer_groups(m)
def groups_audit(m,groups):
 names={id(p):n for n,p in m.named_parameters()}
 a=[dict(group_name=g['group_name'],weight_decay=g['weight_decay'],names=[names[id(p)] for p in g['params']]) for g in groups]
 assert a==read(G/'reference_initial/distdf_optimizer.json');return a
