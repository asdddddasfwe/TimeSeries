"""One official-test forward pass after best is sealed. No fitting or optimizer."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import sys,json,argparse,importlib,time,traceback,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tmp.mafirst_future_hidden_only_five_datasets_full100_20261109_common import *
def main(d):
 guard();torch.set_num_threads(2);s=stem(d,0,'season');c=importlib.import_module('tmp.'+s+'_common');f=importlib.import_module('tmp.'+s+'_factory');o=output(d,0,'season')
 receipt=read(c.G/'execution/formal_train/log.exit.json');assert receipt['real_wait'] and receipt['exit_code']==0
 done=read(o/'completion.json');seal=read(o/'freeze.json')
 assert done['status']=='training_completed_best_frozen' and done['test_evaluations']==0 and not seal['test_opened']
 assert sha(o/'best.pt')==done['checkpoint_sha256']==seal['checkpoint_sha256'] and done['selected_epoch']==seal['selected_epoch']
 with locked():
  with (c.G/'test_once.json').open('x') as h:json.dump(dict(process=identity(),dataset=d,selected_epoch=seal['selected_epoch'],checkpoint_sha256=seal['checkpoint_sha256'],frozen_recipe_sha256=sha(c.G/'plan.json'),created=now()),h)
 torch.cuda.set_per_process_memory_fraction(.9);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
 m,cfg=f.fresh('mafirst_season');m.load_state_dict(torch.load(o/'best.pt',map_location='cpu',weights_only=True),strict=True);m.cuda().eval()
 ds=f.dataset('mafirst_season','test');assert len(ds)==EVAL_WINDOWS and ds.official_support==TEST_SUPPORT
 root=o/'test';root.mkdir(exist_ok=False)
 arrays={n:np.lib.format.open_memmap(root/(n+'.npy'),mode='w+',dtype='float32',shape=(EVAL_WINDOWS,HORIZON,CHANNELS)) for n in ['prediction','truth']}
 count=elements=0;sq=ab=0.;start=time.monotonic()
 with torch.inference_mode():
  for x,y,_,_ in f.loader(ds,batch=32):
   x=x.float().cuda();y=y[:,-HORIZON:].float().cuda();p=m(x);assert p.shape==y.shape and torch.isfinite(p).all()
   err=p.double()-y.double();sq+=float(err.square().sum());ab+=float(err.abs().sum());elements+=err.numel()
   arrays['prediction'][count:count+len(x)]=p.cpu().numpy();arrays['truth'][count:count+len(x)]=y.cpu().numpy();count+=len(x)
   if count%1024==0:atomic(c.G/'test_status.json',dict(status='official_test_running',windows=count,total=EVAL_WINDOWS,process=identity(),updated=now()))
 for a in arrays.values():a.flush()
 assert count==EVAL_WINDOWS and sha(o/'best.pt')==seal['checkpoint_sha256'];guard()
 result=dict(status='completed_verified',dataset=d,model='MAFirst combination '+c.TAG+'',seed=2021,input=96,horizon=HORIZON,features='M',variables=CHANNELS,parameters=read(c.G/'expected_model_cards.json')['mafirst_season']['parameters'],mse=sq/elements,mae=ab/elements,windows=count,elements=elements,metric_space='standardized by official train-prefix StandardScaler; all variables pooled; FP64 sum',official_test_evaluations=1,test_batch=32,shuffle=False,drop_last=False,checkpoint_sha256=seal['checkpoint_sha256'],selected_epoch=seal['selected_epoch'],epochs=done['epochs'],selection_validation_mse=done['selection_validation_mse'],full_validation=done['validation'],no_test_fitting_or_reselection=True,prediction_sha256=sha(root/'prediction.npy'),truth_sha256=sha(root/'truth.npy'),seconds=time.monotonic()-start,created=now())
 atomic(o/'official_test_result.json',result);atomic(c.G/'test_status.json',dict(status='completed_verified',windows=count,process=identity(),updated=now()))
 print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args()
 try:main(a.dataset)
 except BaseException as e:
  atomic(group(a.dataset,0,'season')/'test_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,created=now()));raise
