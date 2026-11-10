"""Independent saved-test-array verification and compact figures; no model forward."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,argparse,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tmp.mafirst_future_hidden_only_five_datasets_full100_20261109_common import *
def main(d):
 guard();g=group(d,0,'season');o=output(d,0,'season');r=read(o/'official_test_result.json')
 assert read(g/'execution/official_test/log.exit.json')['exit_code']==0 and r['official_test_evaluations']==1
 a,b=[np.load(o/'test'/(n+'.npy'),mmap_mode='r') for n in ['prediction','truth']]
 assert a.shape==b.shape==(EVAL_WINDOWS,HORIZON,CHANNELS)
 sq=np.zeros(CHANNELS);ab=np.zeros(CHANNELS)
 chunk=max(1,min(37,2_000_000//(HORIZON*CHANNELS)))
 for i in range(0,len(a),chunk):
  e=np.asarray(a[i:i+chunk],np.float64)-np.asarray(b[i:i+chunk],np.float64);assert np.isfinite(e).all();sq+=np.square(e).sum((0,1));ab+=np.abs(e).sum((0,1))
 sq/=EVAL_WINDOWS*HORIZON;ab/=EVAL_WINDOWS*HORIZON;assert abs(sq.mean()-r['mse'])<1e-10 and abs(ab.mean()-r['mae'])<1e-10
 assert sha(o/'test/prediction.npy')==r['prediction_sha256'] and sha(o/'test/truth.npy')==r['truth_sha256']
 curves=read(o/'curves.json');best=float('inf');stale=chosen=0
 for row in curves:
  if row['val']['mse']<=best:best,chosen,stale=row['val']['mse'],row['epoch'],0
  else:stale+=1
  assert row['stale']==stale and row['chosen']==chosen and row['updates']==row['epoch']*UPDATES_PER_EPOCH and row['model_and_Adam_all_finite']
 assert chosen==r['selected_epoch'] and (len(curves)==100 and curves[-1]['updates']==100*UPDATES_PER_EPOCH) and sha(o/'best.pt')==r['checkpoint_sha256']
 for row in curves:
  saved=read(o/'selection_validation'/f"epoch{row['epoch']:03d}.json")
  ids=np.asarray(saved['indices'],dtype=np.int64)
  assert len(ids)==len(set(ids))==row['val']['windows'] and ids.min()>=0 and ids.max()<VAL_WINDOWS
  assert hashlib.sha256(ids.tobytes()).hexdigest()==row['val']['indices_sha256']==saved['indices_sha256']
 labels=read(g/'plan.json')['variable_names']
 shown=np.argsort(sq)[-min(12,CHANNELS):] if CHANNELS>12 else np.arange(CHANNELS)
 import matplotlib.pyplot as plt
 figdir=g/'figures';figdir.mkdir(exist_ok=False)
 fig,ax=plt.subplots(1,2,figsize=(12,4));ax[0].plot([x['train_mse'] for x in curves],label='train MSE');ax[0].plot([x['val']['mse'] for x in curves],label='official val selection MSE');ax[0].axvline(chosen-1,color='gray',ls='--');ax[0].legend();ax[1].bar([labels[k] for k in shown],sq[shown]);ax[1].tick_params(axis='x',labelrotation=35);ax[1].set_title('official test MSE by variable' if CHANNELS<=12 else '12 variables with highest test MSE');fig.tight_layout();fig.savefig(figdir/'curves_metrics.png',dpi=140);plt.close(fig)
 fig,axes=plt.subplots(3,3,figsize=(15,8))
 for col,idx in enumerate([0,EVAL_WINDOWS//2,EVAL_WINDOWS-1]):
  for row,ch in enumerate([0,CHANNELS//2,CHANNELS-1]):
   axes[row,col].plot(b[idx,:,ch],label='truth');axes[row,col].plot(a[idx,:,ch],label='prediction');axes[row,col].set_title(labels[ch]+' test window '+str(idx))
 axes[0,0].legend();fig.tight_layout();fig.savefig(figdir/'predictions.png',dpi=140);plt.close(fig)
 record=dict(status='verified',independent_saved_only=True,no_model_or_forward=True,official_test_evaluations=1,recomputed_mse=float(sq.mean()),recomputed_mae=float(ab.mean()),val_indices_verified_epochs=len(curves),per_variable_mse=sq.tolist(),per_variable_mae=ab.tolist(),result_sha256=sha(o/'official_test_result.json'),figures={p.name:sha(p) for p in figdir.glob('*.png')},requires_actual_image_QA=True,created=now())
 atomic(g/'offline_review.json',record);print(json.dumps(record))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);main(p.parse_args().dataset)
