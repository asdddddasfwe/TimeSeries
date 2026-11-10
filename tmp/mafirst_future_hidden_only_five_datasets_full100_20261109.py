"""Serial single future-hidden forecasting queue for five new datasets."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tmp import mafirst_future_hidden_only_five_datasets_full100_20261109_stages as stages
from tmp.mafirst_future_hidden_only_five_datasets_full100_20261109_common import SPECS, PY
from tmp.mafirst_future_hidden_only_five_datasets_full100_20261109_stages import (
    ARM, BATCH, DATASETS, HS, MODEL, NEW, STAGES, alive, common, identity, locked, now, read, sha, write,
)

PRIOR=ROOT/'tmp/mafirst_future_hidden_only_ettm1_ettm2_full100_20261107'
PREDECESSOR=ROOT/'tmp/mafirst_single_unbounded_distdf_ettm1_h720_20261109'
WINDOW=96


def prepare():
    import torch
    from types import SimpleNamespace
    from models.OscillatorPDMStatePatchSpectralTimevaryFutureHiddenHorizons20261107 import Model
    from scripts.run_distdf_fredformer_official_recipe import seed

    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2)
    predecessor=read(PREDECESSOR/'launch.json')['coordinator']
    assert predecessor['command'][-1]=='supervise'
    base_group=PRIOR/'ettm1/h96/group'
    frozen=read(base_group/'frozen_sources.json')
    for rel,digest in frozen.items():
        assert sha(ROOT/rel)==digest,rel
    new_sources=sorted((ROOT/'tmp').glob(f'*{NEW}*.py'))
    assert len(new_sources)==8,new_sources
    for p in new_sources:
        frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    protected_paths=[PRIOR/'results.json',PRIOR/'successful_stage_completion.json',
                     PREDECESSOR/'launch.json']
    for h in HS:
        g=PRIOR/f'ettm1/h{h}/group'
        protected_paths.extend([g/'plan.json',g/'reference_initial/mafirst_season.pt'])
    protected={p.relative_to(ROOT).as_posix():sha(p) for p in protected_paths}
    BATCH.mkdir(exist_ok=False)
    details={}
    for d in DATASETS:
        path,channels,train_end,val_end,test_end,freq=SPECS[d]
        source=ROOT/'dataset'/path
        prefix=BATCH/'train_val_prefix'/d.lower()/source.name
        prefix.parent.mkdir(parents=True,exist_ok=False)
        with source.open('rb') as stream,prefix.open('xb') as out:
            for _ in range(val_end+1):
                row=stream.readline()
                assert row
                out.write(row)
        with prefix.open(encoding='utf-8-sig',newline='') as stream:
            header=next(csv.reader(stream))
            assert header[0]=='date' and header[-1]=='OT' and len(header)==channels+1
        full_sha=sha(source)
        dataset_protected={**protected,source.relative_to(ROOT).as_posix():full_sha}
        if channels!=7:
            assert train_end==int(test_end*.7) and val_end==test_end-int(test_end*.2)
        for h in HS:
            c=common(d,h)
            g=c.G
            (g/'reference_initial').mkdir(parents=True,exist_ok=False)
            template=read(PRIOR/f'ettm1/h{h}/group/plan.json')['models'][ARM]
            cfg={**template['config'],'data':d if channels==7 else 'custom','data_path':source.name,
                 'root_path':str(prefix.parent),'log_dir':str(g/'logs'),'tensorboard_dir':str(g/'tensorboard'),
                 'enc_in':channels,'dec_in':channels,'c_out':channels,'freq':freq}
            seed(2021)
            model=Model(SimpleNamespace(**cfg))
            state=model.state_dict()
            ref=g/'reference_initial/mafirst_season.pt'
            if channels==7:
                origin=PRIOR/f'ettm1/h{h}/group/reference_initial/mafirst_season.pt'
                stored=torch.load(origin,map_location='cpu',weights_only=True)
                assert stored.keys()==state.keys() and all(state[n].shape==stored[n].shape and state[n].dtype==stored[n].dtype for n in state)
                assert sha(origin)==template['reference_sha256']
                shutil.copyfile(origin,ref)
            else:
                torch.save(state,ref)
            count=sum(p.numel() for p in model.parameters())
            cards=dict(parameters=count,inactive_parameters=[])
            case=dict(dataset=d,horizon=h,variables=channels,variable_names=header[1:],
                      input=WINDOW,seed=2021,total_epochs=100,early_stopping=False,
                      train_windows=c.TRAIN_WINDOWS,val_windows=c.VAL_WINDOWS,
                      test_windows=c.EVAL_WINDOWS,updates_per_epoch=c.UPDATES_PER_EPOCH,
                      total_updates=c.UPDATES_PER_EPOCH*100,models={ARM:dict(config=cfg,parameters=count,reference_sha256=sha(ref))},
                      train_end=train_end,val_end=val_end,test_end=test_end,train_val_prefix_sha256=sha(prefix),
                      full_dataset_sha256=full_sha,trained_weights_loaded=False,
                      initialization_policy='exact original untrained reference bytes' if channels==7 else 'native channel shape seeded untrained constructor',frequency_condition='future_block_mean_and_last',
                      learnable_periods=False,loss='MSE + MMD squared',actual_train_batch=128,
                      large_memory_optimization=c.LARGE,eval_batch=32,official_test_once_after_best_freeze=True)
            write(g/'expected_model_cards.json',{ARM:cards})
            write(g/'plan.json',case)
            write(g/'frozen_sources.json',frozen)
            write(g/'protected_records.json',dataset_protected)
            local=[ref,prefix]+[g/n for n in ('plan.json','expected_model_cards.json','frozen_sources.json','protected_records.json')]
            write(g/'registration.json',dict(status='registered',hashes={p.relative_to(ROOT).as_posix():sha(p) for p in local}))
            details[f'{d}_h{h}']={k:case[k] for k in ('variables','train_windows','val_windows','test_windows','updates_per_epoch','total_updates','large_memory_optimization')}
            details[f'{d}_h{h}']['parameters']=count
        print('registered',d,flush=True)
    write(BATCH/'design_receipt.json',dict(status='registered',dataset_order=list(DATASETS),horizons=list(HS),
          cases=details,model=MODEL,predecessor=predecessor,actual128_required=True,
          high_channel_gates='bitwise checkpoint and loss tiling equivalence, actual batch128 updates and eval32 fit',
          created=now()))
    write(BATCH/'status.json',dict(status='prepared',cases=len(details),updated=now()))


def wait_predecessor():
    prior=read(BATCH/'design_receipt.json')['predecessor']
    assert read(PREDECESSOR/'launch.json')['coordinator']==prior
    while True:
        if any(PREDECESSOR.glob('failure*.json')) or (PREDECESSOR/'group/failure.json').exists():
            raise RuntimeError('Predecessor recorded failure')
        if not alive(prior):
            assert read(PREDECESSOR/'status.json')['status']=='completed'
            result=read(PREDECESSOR/'results.json')
            assert result['status']=='verified' and set(result['real_stage_waits'])==set(STAGES)
            assert all(e['real_wait'] and e['exit_code']==0 for e in result['real_stage_waits'].values())
            return
        write(BATCH/'status.json',dict(status='waiting_predecessor',process=identity(),predecessor=prior,updated=now()))
        time.sleep(30)


def wait_resources(c):
    # Saved prediction and truth are FP32 and both must fit on the physical volume.
    needed=30*2**30+8*c.EVAL_WINDOWS*c.HORIZON*c.CHANNELS
    while True:
        memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],text=True).splitlines()[0])
        jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                    '--format=csv,noheader,nounits'],text=True).strip()
        disk=min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/mnt/d').free)
        if not jobs and memory>=2500 and disk>=needed:
            return
        c.atomic(c.G/'status.json',dict(status='waiting_resources',gpu_free_mib=memory,compute_pids=jobs,
                 disk_free_bytes=disk,required_disk_bytes=needed,updated=now()))
        time.sleep(30)


def case_process(d,h):
    stages.wait_resources=wait_resources
    stages.arm(d,h)


def supervise():
    with locked(),(BATCH/'supervisor_once.json').open('x') as f:
        json.dump(identity(),f)
    wait_predecessor()
    results={}
    blocked={}
    for d in DATASETS:
        for h in HS:
            c=common(d,h)
            name=f'{d}_h{h}'
            write(BATCH/'status.json',dict(status='running',dataset=d,horizon=h,process=identity(),updated=now()))
            cmd=[PY,'-B','-u',str(Path(__file__).resolve()),'case','--dataset',d,'--horizon',str(h)]
            with (c.G.parent/'arm.log').open('xb') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                code=process.wait()
            write(c.G.parent/'arm_exit.json',dict(exit_code=code,real_wait=True,created=now()))
            if code:
                phase=read(c.G/'status.json')['status']
                exe=c.G/'execution'/phase
                text=(exe/'log.txt').read_text(errors='replace') if (exe/'log.txt').exists() else ''
                if c.LARGE and phase in ('CUDA_route','actual128_memory') and 'out of memory' in text.lower():
                    blocked[name]=dict(status='actual128_memory_blocked',phase=phase,actual128_retained=True,
                                       formal_training_started=False,log=str(exe/'log.txt'))
                    c.atomic(c.G/'status.json',blocked[name])
                else:
                    raise RuntimeError(f'{name} failed at {phase}; see {c.G.parent / "arm.log"}')
            else:
                waits={s:read(c.G/'execution'/s/'log.exit.json') for s in STAGES}
                assert all(e['real_wait'] and e['exit_code']==0 for e in waits.values())
                results[name]=dict(result=read(c.OUT[ARM]/'official_test_result.json'),real_stage_waits=waits)
            write(BATCH/'progress_results.json',dict(status='in_progress',completed=results,blocked=blocked,updated=now()))
    write(BATCH/'results.json',dict(status='verified_with_resource_blocks' if blocked else 'verified',
                                   completed=results,blocked=blocked,created=now()))
    write(BATCH/'status.json',dict(status='completed_with_resource_blocks' if blocked else 'completed',
                                  completed=len(results),blocked=len(blocked),updated=now()))


def launch():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')!=''
    assert read(BATCH/'status.json')['status']=='prepared'
    with locked(),(BATCH/'dispatch_once.json').open('x') as f:
        json.dump(identity(),f)
    command=[PY,'-B','-u',str(Path(__file__).resolve()),'supervise']
    with (BATCH/'supervisor.log').open('xb') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL,start_new_session=True)
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError(f'Supervisor exited at launch: {process.returncode}')
            try:
                rec=identity(process.pid)
                if rec['command']==command:
                    break
            except OSError:
                pass
            time.sleep(.02)
        else:
            raise RuntimeError('Supervisor command did not match')
    write(BATCH/'launch.json',dict(coordinator=rec,source_sha256=sha(Path(__file__)),created=now()))
    print(json.dumps(read(BATCH/'launch.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','launch','supervise','case','verify'))
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--horizon',type=int,choices=HS)
    parser.add_argument('--device',choices=('cpu','cuda'))
    args=parser.parse_args()
    try:
        if args.mode=='verify':
            if args.device=='cpu':
                os.environ['CUDA_VISIBLE_DEVICES']=''
            stages.verify(args.dataset,args.horizon,args.device)
        elif args.mode=='case':
            case_process(args.dataset,args.horizon)
        else:
            globals()[args.mode]()
    except BaseException as exc:
        if BATCH.is_dir() and args.mode in ('prepare','supervise'):
            write(BATCH/'failure.json',dict(phase=args.mode,error=repr(exc),traceback=traceback.format_exc(),
                                           process=identity(),automatic_retry=False,created=now()))
        raise
