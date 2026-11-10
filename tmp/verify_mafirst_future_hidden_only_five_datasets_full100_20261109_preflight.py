"""Saved-only verifier of full preflight states and arrays; no model forward."""
from pathlib import Path
import sys,argparse
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from tmp.mafirst_future_hidden_only_five_datasets_full100_20261109_common import *
import torch,numpy as np

def exact(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b,(a,b)
root=G/'preflight/mafirst_season'
left,right=root/'continuous',root/'resumed'
x,y=[torch.load(p/'last.pt',map_location='cpu',weights_only=True) for p in [left,right]];exact(x,y)
for n in ['curves.json','gradients.jsonl','gate_trace.jsonl']:assert sha(left/n)==sha(right/n)
for p in (left/'selection_validation').glob('*.json'):assert read(p)==read(right/'selection_validation'/p.name)
files=list((left/'selected_val').glob('*.npy'));assert len(files)==7
for p in files:assert np.array_equal(np.load(p,mmap_mode='r'),np.load(right/'selected_val'/p.name,mmap_mode='r'))
atomic(root/'verification.json',dict(status='verified',model_Adam_scheduler_all_RNG_curves_gradients_gate_trace_actual_indices_7_arrays_exact=True,epochs=3,updates=12,arrays=7,development_weights_discarded=True,test_evaluations=0,execution_version='mafirst_future_hidden_only_five_datasets_full100_20261109',created=now()))
print('verified saved preflight state, optimizer, scheduler, RNG, curves, gradients, gate traces, indices, 7 arrays')
