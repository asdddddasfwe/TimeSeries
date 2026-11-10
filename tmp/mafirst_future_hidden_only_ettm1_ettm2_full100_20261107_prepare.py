"""Lightweight reference registration; actual CPU/model checks deferred until predecessor ends."""
import ast,csv,io,shutil
from pathlib import Path
from tmp.mafirst_future_hidden_only_ettm1_ettm2_full100_20261107 import ROOT,BATCH,NEW,HS,DATASETS,ARM,MODEL,COUNTS,source,write,read,sha,now

def prepare_batch():
 assert not BATCH.exists(),'Never overwrite an existing experiment'
 prior=ROOT/'tmp/mafirst_single_unbounded_ettm1_ettm2_full100_20261107';pause=read(prior/'priority_future_hidden_supervisor_pause_receipt.json');assert pause['priority_root']==BATCH.relative_to(ROOT).as_posix()
 frozen=read(prior/'ettm1/h96/group/frozen_sources.json')
 for rel,digest in frozen.items():assert sha(ROOT/rel)==digest
 extras=[Path(__file__).resolve(),ROOT/'tmp'/(NEW+'.py'),ROOT/'models'/(MODEL+'.py'),ROOT/'tmp/mafirst_priority_future_hidden_queue_control_20261107.py']+[ROOT/'tmp'/x.format(NEW) for x in ['{}_common.py','{}_factory.py','run_{}.py','test_{}.py','review_{}.py','verify_{}_preflight.py']]
 seen=set();pending=list(extras)
 while pending:
  p=pending.pop();rel=p.relative_to(ROOT).as_posix()
  if rel in seen:continue
  seen.add(rel)
  for node in ast.walk(ast.parse(p.read_text(encoding='utf8'))):
   mods=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.level==0 and node.module else []
   for mod in mods:
    if mod.split('.')[0] not in ('models','layers','utils','data_provider','scripts','tmp'):continue
    q=ROOT/Path(*mod.split('.')).with_suffix('.py')
    if not q.exists():q=ROOT/Path(*mod.split('.'))/'__init__.py'
    if q.exists() and q.relative_to(ROOT).as_posix() not in seen:pending.append(q)
 for rel in seen:frozen[rel]=sha(ROOT/rel)
 protected=read(prior/'protected_historical_records.json');protected['tmp/mafirst_single_unbounded_ettm1_ettm2_full100_20261107/priority_future_hidden_supervisor_pause_receipt.json']=sha(prior/'priority_future_hidden_supervisor_pause_receipt.json')
 BATCH.mkdir();details={};write(BATCH/'protected_historical_records.json',protected)
 for d in DATASETS:
  official=ROOT/'dataset/ETT-small'/(d+'.csv')
  with official.open('rb') as f:prefix=b''.join(f.readline() for _ in range(46081))
  rows=list(csv.reader(io.StringIO(prefix.decode('utf8'))));assert len(rows)==46081 and rows[0]==['date','HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'];assert all(len(row)==8 for row in rows)
  for h in HS:
   oldg=source(h)/'group';g=BATCH/d.lower()/f'h{h}/group';(g/'reference_initial').mkdir(parents=True);(g/'train_val_data').mkdir();oldplan=read(oldg/'plan.json');origin=oldg/'reference_initial/mafirst_season.pt';assert sha(origin)==oldplan['models'][ARM]['reference_sha256']
   ref=g/'reference_initial/mafirst_season.pt';shutil.copyfile(origin,ref);assert sha(ref)==sha(origin)
   dp=g/'train_val_data'/(d+'.csv');dp.write_bytes(prefix);assert sha(dp)==read(prior/'design_receipt.json')['details'][f'{d}_h{h}']['train_val_prefix_sha256']
   cfg=dict(oldplan['models'][ARM]['config']);cfg.update(model=MODEL,data=d,data_path=d+'.csv',root_path=str(g/'train_val_data'),log_dir=str(g/'logs'),tensorboard_dir=str(g/'tensorboard'),train_epochs=100,patience=None,seed=2021)
   cards=read(oldg/'expected_model_cards.json');assert cards[ARM]['parameters']==COUNTS[h];write(g/'expected_model_cards.json',cards)
   plan=dict(name=f'{d}_H{h}_future_hidden_only_fresh100',created=now(),dataset=d,horizon=h,input=96,variables=7,seed=2021,total_epochs=100,total_updates=100*((34560-96-h+1)//128),early_stopping=False,patience=None,trained_weights_loaded=False,reference_source=origin.relative_to(ROOT).as_posix(),models={ARM:dict(config=cfg,parameters=COUNTS[h],reference_sha256=sha(ref),original_untrained_reference_sha256=sha(origin))},recipe='actual128 FP32 10workers continuous RNG MSE+MMD2 original Adam OneCycle100 maxlr1e-4 pct_start.4',model_specification='Frozen future-hidden V3 forward extended only to four horizons',dynamic_frequency_condition='per-future-block mean+last of projected hidden features from observed input',future_hidden_frequency_conditioning=True,learnable_periods=False,learnable_period_offsets_present=False,basis='fixed normalized four cosines cos(pi*(k+.5)*j/steps), j=1..4',basis_horizon_normalized=True,per_step_angle_margin_retained=True,extra_frequency_regularization=None,separate_frequency_lr=None,preparation_weights_discarded=True,official_support=dict(train_end=34560,val_end=46080,test_end=57600,scaler_fit_train_prefix_only=True),train_val_prefix_rows=46080,after_seen_prior_ETTm1_tests=True,ETTm2_uses_same_recipe_no_dataset_specific_tuning=True,no_trained_dataset_or_horizon_transfer=True,CPU_and_CUDA_actual_checks_pending=True)
   write(g/'plan.json',plan);write(g/'frozen_sources.json',frozen);write(g/'protected_records.json',protected);local=[ref,dp]+[g/name for name in ['plan.json','expected_model_cards.json','frozen_sources.json','protected_records.json']];write(g/'registration.json',dict(status='registered',hashes={p.relative_to(ROOT).as_posix():sha(p) for p in local}))
   details[f'{d}_h{h}']=dict(parameters=COUNTS[h],reference_sha256=sha(ref),original_untrained_reference_sha256=sha(origin),reference_file_copy_exact=True,trained_weights_loaded=False,constructor_and_RNG_actual_checks_pending=True,train_windows=34560-96-h+1,updates_per_epoch=(34560-96-h+1)//128,full_val_test_windows=11520-h+1,elements=(11520-h+1)*h*7,frozen_sources=len(frozen),protected_records=len(protected),train_val_prefix_sha256=sha(dp));print('registered',d,h,flush=True)
 write(BATCH/'design_receipt.json',dict(status='registered_pending_CPU_verification',user_request='Add single future-hidden groups, all forecast horizons without early stopping, put first',datasets=list(DATASETS),horizons=list(HS),fresh100_each=True,no_early_stopping=True,total_cases=8,details=details,model=MODEL,frequency_head_condition='future_block_mean_and_last',fixed_normalized_four_cosine_basis=True,no_learnable_periods=True,official_test_once_after_best_validation_freeze=True,full_val_and_test_batch=32,no_test_tuning=True,loss='MSE+MMD2',learning_rate='unchanged OneCycle100 maxlr1e-4 pct_start.4',priority_order='running single-unbounded ETTm1 H96 finishes, then these8, then remaining7 single-unbounded',current_prior_scientific_arm_untouched=True,actual_constructor_prechecks_deferred_until_current_group_exits_to_avoid_resource_overlap=True,created=now()))
 write(BATCH/'queued_prepared.json',dict(status='registered_pending_CPU_verification',cases=8,created=now()));print('registered 8 future-only cases; no CPU/model/forward/test claim yet',flush=True)
