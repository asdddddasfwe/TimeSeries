"""Official Adam and Scheduler functions extracted unchanged from archived source."""
import ast,math
import numpy as np
import torch
from torch import optim
from types import SimpleNamespace
from tmp.mafirst_wideband_dmodel24_distdfopt_h720_20261027_common import *
def extracted(path,name,cls=None):
 tree=ast.parse(path.read_text())
 if cls:tree=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
 node=next(n for n in tree.body if n.name==name) if cls else next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
 ns=dict(optim=optim,math=math,np=np)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),ns)
 return ns[name]
OfficialScheduler=extracted(G/'official_sources/tools.py','Scheduler')
official_optimizer=extracted(G/'official_sources/exp_basic.py','_select_optimizer','Exp_Basic')
def make_optimizer(model):
 return official_optimizer(SimpleNamespace(model=model,args=SimpleNamespace(learning_rate=.0001,optim_type='adam')))
def make_scheduler(opt,steps):
 args=SimpleNamespace(lradj='TST',step_size=1,lr_decay=.5,min_lr=0.,mode=0,train_epochs=100,pct_start=.4)
 return OfficialScheduler(opt,args,steps)
def training_loader(ds,epoch):
 from torch.utils.data import Subset,DataLoader
 order=torch.randperm(len(ds),generator=torch.Generator().manual_seed(2021*1000+epoch)).tolist()
 return DataLoader(Subset(ds,order),batch_size=128,shuffle=False,drop_last=True,num_workers=0)
