"""Unique preparation of eight fresh experiments; no trained weights copied."""
import ast,hashlib,random,sys,json
from pathlib import Path
import torch,numpy as np,pandas as pd
from types import SimpleNamespace
from tmp.mafirst_single_unbounded_ettm1_ettm2_full100_20261107 import ROOT,BATCH,NEW,HS,DATASETS,ARM,MODEL,COUNTS,source,write,read,sha,now
from scripts.run_distdf_fredformer_official_recipe import seed
from models.OscillatorPDMStatePatchSpectralTimevaryHorizons20261105 import Model as Original
from models.OscillatorPDMStatePatchSingleUnboundedHorizons20261107 import Model

def prepare_batch():
 torch.set_num_threads(2);assert not BATCH.exists(),'Unique fresh experiment root must not exist'
 previous=ROOT/'tmp/mafirst_unbounded_period_angle_distdf_ettm1_h720_seed2021_20261105/spectral_timevary_unbounded_period_angle/group/frozen_sources.json';frozen=read(previous)
 for rel,digest in frozen.items():assert sha(ROOT/rel)==digest,rel
 extras=[ROOT/'tmp'/(NEW+'.py'),ROOT/'tmp'/(NEW+'_prepare.py'),ROOT/'models'/(MODEL+'.py')]+[ROOT/'tmp'/x.format(NEW) for x in ['{}_common.py','{}_factory.py','run_{}.py','test_{}.py','review_{}.py','verify_{}_preflight.py']]
 extras += [ROOT/'models/OscillatorPDMStatePatchSpectralTimevaryHorizons20261105.py',ROOT/'utils/time_position_patch_mmd_memory.py',ROOT/'data_provider/data_loader.py',ROOT/'utils/timefeatures.py',ROOT/'utils/augmentation.py']
 # Freeze static local import closure of the actual model and training entry points.
 pending=list(extras);seen=set()
 while pending:
  p=pending.pop();rel=p.relative_to(ROOT).as_posix()
  if rel in seen:continue
  seen.add(rel);tree=ast.parse(p.read_text(encoding='utf-8'))
  for node in ast.walk(tree):
   mods=[]
   if isinstance(node,ast.Import):mods=[x.name for x in node.names]
   elif isinstance(node,ast.ImportFrom) and node.level==0 and node.module:mods=[node.module]
   for mod in mods:
    if mod.split('.')[0] not in ('models','layers','utils','data_provider','scripts','tmp'):continue
    path=ROOT/Path(*mod.split('.')).with_suffix('.py')
    if not path.exists():path=ROOT/Path(*mod.split('.'))/'__init__.py'
    if path.exists() and path.relative_to(ROOT).as_posix() not in seen:pending.append(path)
 for rel in seen:frozen[rel]=sha(ROOT/rel)
 protected={}
 for path in [ROOT/'dataset/ETT-small/ETTm1.csv',ROOT/'dataset/ETT-small/ETTm2.csv',previous]:protected[path.relative_to(ROOT).as_posix()]=sha(path)
 for path in (ROOT/'docs/experiments').glob('MAFirst*.html'):protected[path.relative_to(ROOT).as_posix()]=sha(path)
 report=ROOT/'docs/experiments/DistDFFullDecayAllHorizonsETTm1_20261106.html';protected[report.relative_to(ROOT).as_posix()]=sha(report)
 for h in HS:
  bases=[source(h),ROOT/f'tmp/mafirst_future_hidden_unbounded_full100_ettm1_all_horizons_seed2021_20261107/h{h}']
  if h in (96,192):bases += [ROOT/f'tmp/mafirst_full_decay_distdf_ettm1_h{h}_seed2021_20261106/unbounded']
  if h==720:bases += [previous.parent.parent]
  for base in bases:
   candidates=[base/'group/plan.json',base/'group/reference_initial/mafirst_season.pt']
   candidates += [base/'output'/name for name in ['best.pt','last.pt','curves.json','epoch_receipts.jsonl','freeze.json','gradients.jsonl','gate_trace.jsonl','official_test_result.json','completion.json','test/prediction.npy','test/truth.npy']]
   candidates += list((base/'output/selection_validation').glob('*.json'))
   for p in candidates:
    if p.exists():protected[p.relative_to(ROOT).as_posix()]=sha(p)
 details={};BATCH.mkdir();write(BATCH/'protected_historical_records.json',protected)
 for d in DATASETS:
  official=ROOT/'dataset/ETT-small'/(d+'.csv');data=pd.read_csv(official,nrows=46080)
  assert len(data)==46080 and list(data.columns)==['date','HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'];assert np.isfinite(data.iloc[:,1:].values).all();assert pd.to_datetime(data['date']).is_monotonic_increasing
  prefix=b''.join(official.read_bytes().splitlines(keepends=True)[:46081])
  for h in HS:
   oldg=source(h)/'group';g=BATCH/d.lower()/f'h{h}/group';(g/'reference_initial').mkdir(parents=True);(g/'train_val_data').mkdir()
   oldplan=read(oldg/'plan.json');cfg=dict(oldplan['models'][ARM]['config']);cfg.update(model=MODEL,data=d,data_path=d+'.csv',root_path=str(g/'train_val_data'),log_dir=str(g/'logs'),tensorboard_dir=str(g/'tensorboard'),train_epochs=100,patience=None,seed=2021)
   origin=oldg/'reference_initial/mafirst_season.pt';assert sha(origin)==oldplan['models'][ARM]['reference_sha256'];base=torch.load(origin,map_location='cpu',weights_only=True);ref={**base,'learnable_period_offsets':torch.zeros(2)}
   seed(2021);control=Original(SimpleNamespace(**cfg));crng=torch.get_rng_state().clone();prng=random.getstate();nrng=np.random.get_state();seed(2021);m=Model(SimpleNamespace(**cfg));n2=np.random.get_state()
   assert torch.equal(crng,torch.get_rng_state()) and prng==random.getstate();assert nrng[0]==n2[0] and np.array_equal(nrng[1],n2[1]) and nrng[2:]==n2[2:]
   assert m.state_dict().keys()==ref.keys() and all(torch.equal(v,m.state_dict()[k]) for k,v in control.state_dict().items())
   m.load_state_dict(ref,strict=True);assert all(torch.equal(v,m.state_dict()[k]) for k,v in ref.items());count=sum(p.numel() for p in m.parameters());assert count==COUNTS[h]
   rp=g/'reference_initial/mafirst_season.pt';torch.save(ref,rp);dp=g/'train_val_data'/(d+'.csv');dp.write_bytes(prefix);assert pd.read_csv(dp).equals(data)
   if d=='ETTm1':assert pd.read_csv(oldg/'train_val_data/ETTm1.csv').equals(data)
   cards={ARM:dict(parameters=count,parameter_tensors=len(list(m.parameters())),states=len(m.state_dict()),buffers=len(list(m.buffers())),inactive_parameters=[])};write(g/'expected_model_cards.json',cards)
   plan=dict(name=f'{d}_H{h}_single_unbounded_period_fresh100',created=now(),dataset=d,horizon=h,input=96,variables=7,seed=2021,total_epochs=100,total_updates=100*((34560-96-h+1)//128),early_stopping=False,patience=None,trained_weights_loaded=False,reference_source=origin.relative_to(ROOT).as_posix(),models={ARM:dict(config=cfg,parameters=count,reference_sha256=sha(rp),original_untrained_reference_sha256=sha(origin))},recipe='actual128 FP32 10workers continuous RNG MSE+MMD2 original Adam OneCycle100 maxlr1e-4 pct_start.4',model_specification='Original single-unbounded InheritedCombination forward, exported for all four horizons; no future-hidden frequency conditioning',dynamic_frequency_condition='history mean and history last, shared coefficients across forecast blocks',future_hidden_frequency_conditioning=False,existing_future_hidden_additive_drive_retained=True,frequency_correction='.25*min(angle,pi/stride-angle)*tanh(history_coefficients * physical_sin_cos_basis)',basis='t=4*(k+.5), sin/cos for P=[96,384]*exp(eta)',initial_periods=[96,384],learnable_period_offsets_global_shared_across_scales=True,period_bounds=None,period_clamp=False,cumulative_phase_cap=False,extra_period_regularization=None,separate_frequency_lr=None,preparation_weights_discarded=True,official_support=dict(train_end=34560,val_end=46080,test_end=57600,scaler_fit_train_prefix_only=True),train_val_prefix_rows=46080,after_seen_prior_ETTm1_tests=True,ETTm2_uses_same_recipe_no_dataset_specific_tuning=True,no_trained_dataset_or_horizon_transfer=True)
   write(g/'plan.json',plan);write(g/'frozen_sources.json',frozen);write(g/'protected_records.json',protected)
   local=[rp,dp]+[g/name for name in ['plan.json','expected_model_cards.json','frozen_sources.json','protected_records.json']];write(g/'registration.json',dict(status='verified',hashes={p.relative_to(ROOT).as_posix():sha(p) for p in local}))
   details[f'{d}_h{h}']=dict(parameters=count,reference_sha256=sha(rp),original_untrained_reference_sha256=sha(origin),construction_CPU_RNG_sha256=hashlib.sha256(crng.numpy().tobytes()).hexdigest(),common_tensor_and_CPU_Python_NumPy_RNG_matches_original=True,reference_loaded_exact=True,zero_eta=True,train_windows=34560-96-h+1,updates_per_epoch=(34560-96-h+1)//128,full_val_test_windows=11520-h+1,elements=(11520-h+1)*h*7,frozen_sources=len(frozen),protected_records=len(protected),train_val_prefix_sha256=sha(dp));print('prepared',d,h,flush=True)
 write(BATCH/'design_receipt.json',dict(status='verified',user_request='Rerun single unbounded periods for all forecast horizons without early stopping, also ETTm2',datasets=list(DATASETS),horizons=list(HS),fresh100_each=True,no_early_stopping=True,total_cases=8,details=details,model=MODEL,frequency_head_condition='history_mean_and_last',no_future_hidden_frequency_conditioning=True,official_test_once_after_best_validation_freeze=True,full_val_and_test_batch=32,no_test_tuning=True,loss='MSE+MMD2',learning_rate='unchanged OneCycle100 maxlr1e-4 pct_start.4',created=now()))
 write(BATCH/'prepared.json',dict(status='verified',datasets=list(DATASETS),horizons=list(HS),cases=8,fresh100_each=True,early_stopping=False,created=now()));print(json.dumps(read(BATCH/'design_receipt.json')),flush=True)
