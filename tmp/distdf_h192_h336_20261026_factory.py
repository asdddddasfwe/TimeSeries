"""Horizon-specific official Fredformer configs; independent fresh scratch."""
import torch,numpy as np
from types import SimpleNamespace
from torch.utils.data import DataLoader,Subset
from data_provider.data_loader import Dataset_ETT_minute
from tmp.distdf_h192_h336_20261026_common import *
from scripts import run_distdf_fredformer_matched as source
from utils.distdf_joint_wasserstein import distdf_loss
seed=source.seed
def config(arm,root=None):
 c=SimpleNamespace(**read(PLAN)['arms'][arm]['config'])
 if root is not None:c.root_path=str(root)
 return c
def fresh(arm,logdir):
 c=config(arm);m=source.Model(c);ref=torch.load(G/'reference_initial'/(arm+'.pt'),map_location='cpu',weights_only=True)
 assert m.state_dict().keys()==ref.keys() and all(torch.equal(v,ref[n]) and v.data_ptr()!=ref[n].data_ptr() for n,v in m.state_dict().items())
 return m,c
def optimizer_groups(m):return source.optimizer_groups(m)
def groups_audit(arm,m,groups):
 names={id(p):n for n,p in m.named_parameters()};a=[dict(group_name=g['group_name'],weight_decay=g['weight_decay'],names=[names[id(p)] for p in g['params']]) for g in groups];assert a==read(G/'reference_initial'/(arm+'_optimizer.json'));return a
def dataset(cfg,split,limit=0):
 h=cfg.pred_len;data=Dataset_ETT_minute(cfg,cfg.root_path,flag=split,size=[96,48,h],features='M',data_path='ETTm1.csv',timeenc=1,freq='h')
 assert len(data)==(34560-96-h+1 if split=='train' else 11520-h+1)
 if limit:data=Subset(data,np.unique(np.linspace(0,len(data)-1,min(limit,len(data)),dtype=int)).tolist())
 return data
def loader(data,epoch=None):
 if epoch is not None:data=Subset(data,torch.randperm(len(data),generator=torch.Generator().manual_seed(2021*1000+epoch)).tolist())
 return DataLoader(data,batch_size=32,shuffle=False,num_workers=0,drop_last=False)
