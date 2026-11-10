"""New untrained MAFirst instances; real batch128, same architecture and loss."""
import random
from types import SimpleNamespace
import numpy as np,torch
from torch.utils.data import DataLoader,Subset
from tmp.mafirst_future_hidden_only_five_datasets_full100_20261109_common import *
from scripts.run_distdf_fredformer_official_recipe import seed, loader as distdf_loader
def configuration(arm):
 c=read(G/'plan.json')['models'][arm]['config']
 return SimpleNamespace(**c)
def untrained(arm):
 assert arm=='mafirst_season'
 seed(2021);cfg=configuration(arm)
 from importlib import import_module
 from models.OscillatorPDMStatePatchSpectralTimevaryFutureHiddenHorizons20261107 import Model
 m=Model(cfg)
 ref=torch.load(G/'reference_initial/mafirst_season.pt',map_location='cpu',weights_only=True)
 assert m.state_dict().keys()==ref.keys()
 m.load_state_dict(ref,strict=True)
 return m,cfg
def fresh(arm):
 m,c=untrained(arm);ref=torch.load(G/'reference_initial'/f'{arm}.pt',map_location='cpu',weights_only=True)
 assert m.state_dict().keys()==ref.keys()
 assert all(torch.equal(v,ref[n]) and v.data_ptr()!=ref[n].data_ptr() for n,v in m.state_dict().items())
 return m,c
def dataset(arm,split,limit=0):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from data_provider.data_loader import Dataset_Custom
    from utils.timefeatures import time_features
    if split not in ('train','val','test'):
        raise ValueError('Unknown benchmark split')
    o=OUT[arm]
    if split=='test':
        assert (G/'test_once.json').is_file() and (o/'freeze.json').is_file()
        assert read(o/'completion.json')['status']=='training_completed_best_frozen' and not read(o/'freeze.json')['test_opened']
    cfg=configuration(arm)
    source=R/'dataset'/CSV if split=='test' else Path(cfg.root_path)/cfg.data_path
    input_start,target_start,stop={
        'train':(0,96,TRAIN_END),'val':(TRAIN_END-96,TRAIN_END,VAL_END),
        'test':(VAL_END-96,VAL_END,TEST_END)}[split]

    class FixedSplit(Dataset_Custom):
        # Reuse the benchmark loader's window indexing with explicit full-data split borders.
        def __read_data__(self):
            frame=pd.read_csv(source,nrows=stop)
            assert len(frame)==stop and frame.columns[0]=='date' and frame.columns[-1]=='OT'
            assert frame.shape[1]==CHANNELS+1
            values=frame.iloc[:,1:].to_numpy()
            self.scaler=StandardScaler().fit(values[:TRAIN_END])
            self.data_x=self.data_y=self.scaler.transform(values[input_start:stop])
            dates=pd.to_datetime(frame.iloc[input_start:stop,0].values)
            self.data_stamp=time_features(dates,freq=FREQ).transpose(1,0)

    ds=FixedSplit(cfg,str(source.parent),flag=split,size=[96,cfg.label_len,HORIZON],
                  features='M',data_path=source.name,timeenc=1,freq=FREQ)
    ds.official_support=dict(input_start=input_start,target_start=target_start,
                             end_exclusive=stop,train_scaler_rows=TRAIN_END)
    assert len(ds)=={'train':TRAIN_WINDOWS,'val':VAL_WINDOWS,'test':EVAL_WINDOWS}[split]
    if limit:
        ds=Subset(ds,np.unique(np.linspace(0,len(ds)-1,min(limit,len(ds)),dtype=int)).tolist())
    return ds

def training_prediction(model,x):
    if not LARGE:
        return model(x)
    from torch.utils.checkpoint import checkpoint
    return checkpoint(model,x,use_reentrant=False,preserve_rng_state=True)

def tiled_parts(p,y):
    from utils.time_position_patch_mmd_memory import _terms
    from torch.utils.checkpoint import checkpoint
    # Tile independent channels and sequences of the exact same Gram calculation.
    # Reassemble before the existing global mean to keep the reduction order.
    p=p.transpose(1,2).unfold(-1,4,4).double()
    y=y.detach().transpose(1,2).unfold(-1,4,4).double()
    i=torch.arange(p.shape[-2],device=p.device,dtype=torch.float64)
    kernel=torch.exp(-0.5*((i[:,None]-i[None,:])/24.)**2)
    def evaluate(a,b):
        terms=[]
        for pa,pb in zip(a.split(32,dim=1),b.split(32,dim=1)):
            values=checkpoint(_terms,pa,pb,kernel,2.,use_reentrant=False,preserve_rng_state=False) if torch.is_grad_enabled() and pa.requires_grad else _terms(pa,pb,kernel,2.)
            terms.append(values[0]+values[1]-2*values[2])
        return torch.cat(terms,dim=1)
    raw=torch.cat([evaluate(a,b) for a,b in zip(p.split(4),y.split(4))])
    if not torch.isfinite(raw).all() or raw.min() < -1e-10:
        raise RuntimeError('Invalid tiled time-position MMD squared')
    return raw.clamp_min(0).mean()
def loader(ds, epoch=None, batch=32):
 if epoch is not None:
  return distdf_loader(ds, 'train')
 return DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False, num_workers=0)
def optimizer(m):
 return torch.optim.Adam(m.parameters(),lr=.0001,weight_decay=0.,betas=(.9,.999),eps=1e-8)
def scheduler(opt,steps):
 return torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[.0001],steps_per_epoch=steps,epochs=100,pct_start=.4,div_factor=25.,final_div_factor=10000.,cycle_momentum=True,base_momentum=.85,max_momentum=.95)
def loss(arm,p,y,x):
 mse=(p-y.detach()).square().mean()
 if arm=='mafirst_season':
  from utils.time_position_patch_mmd_memory import parts
  regularizer=tiled_parts(p,y).to(p.dtype) if LARGE else parts(p,y)[0];total=mse+regularizer
 else:
  from utils.distdf_joint_wasserstein import distdf_loss
  regularizer=distdf_loss(p,y,x,var_weight=.002,mean_weight=1.,reg=.005,eps=1e-8);total=.8*mse+.2*regularizer
 return total,mse,regularizer

from tmp.mafirst_distdf_selection_aligned_20261028 import selection_loader,IndexedValidationDataset
