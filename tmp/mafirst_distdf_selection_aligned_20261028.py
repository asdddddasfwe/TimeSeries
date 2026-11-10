"""Official DistDF-style checkpoint selection for MAFirst; no training-loss or model change."""
import hashlib,numpy as np,torch
from torch.utils.data import DataLoader,Dataset
class IndexedValidationDataset(Dataset):
 def __init__(self,ds):self.ds=ds
 def __len__(self):return len(self.ds)
 def __getitem__(self,i):return (i,*self.ds[i])
def selection_loader(ds,workers=10):
 return DataLoader(IndexedValidationDataset(ds),batch_size=128,shuffle=True,drop_last=True,num_workers=workers)
def selection_validation(arm,m,ds,device,out,epoch,horizon,atomic):
 m.eval();losses=[];indices=[]
 with torch.no_grad():
  for ids,x,yy,_,_ in selection_loader(ds):
   x=x.float().to(device);y=yy[:,-horizon:].float().to(device);p=m(x)
   assert torch.isfinite(p).all() and p.shape==y.shape
   losses.append(torch.nn.functional.mse_loss(p.detach(),y.detach()));indices.extend(ids.tolist())
 value=torch.mean(torch.stack(losses)).item()
 assert len(indices)==len(ds)//128*128 and len(indices)==len(set(indices))
 result=dict(mse=value,windows=len(indices),batch_size=128,shuffle=True,drop_last=True,workers=10,FP32_batch_MSE_mean=True,indices_sha256=hashlib.sha256(np.asarray(indices,dtype=np.int64).tobytes()).hexdigest())
 atomic(out/'selection_validation'/f'epoch{epoch:03d}.json',dict(**result,epoch=epoch,indices=indices))
 m.train();return result
