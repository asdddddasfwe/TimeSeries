"""MA1 pure-MSE schedule comparison, validation audit/freeze then default full test."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,json,hashlib,shutil,subprocess,sys,math,fcntl
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.train_timemix_pdm_schedule import atomic_json,ARMS,SchedulePolicy
from scripts.evaluate_frozen_timemix import verify_artifact
from scripts.diagnose_timemix_linear_reference import score
now=lambda:datetime.now(timezone.utc).isoformat()
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def phase(out,status,**extra):
    atomic_json(out/'status.json',dict(status=status,updated=now(),**extra));print(status,extra,flush=True)
def child_run(out,cmd,log,identity):
    with log.open('x') as f:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        proc=Path('/proc')/str(child.pid)
        atomic_json(identity,dict(pid=child.pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],command=cmd))
        if child.wait()!=0:raise RuntimeError('Child failed: '+str(log))
def finite(x):
    if isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,(list,tuple)):
        for v in x:finite(v)
def equal_state(x,y):
    assert x.keys()==y.keys()
    assert all(torch.equal(x[k].cpu(),y[k].cpu()) for k in x)

def freeze(out,row,tp,smoke):
    a=out/(row['variant']+'_artifact');a.mkdir()
    shutil.copyfile(row['checkpoint'],a/'checkpoint.pth')
    atomic_json(a/'model_config.json',dict(row['model_config'],is_training=0))
    identity=dict(checkpoint_sha256=sha(a/'checkpoint.pth'),model_config_sha256=sha(a/'model_config.json'))
    atomic_json(a/'manifest.json',dict(kind=row['model_config']['model'],selection_status='smoke_validation_only' if smoke else 'frozen_validation_diagnostic',**identity,neural_training_source_sha256=tp['source_sha256'],seed=row['seed'],best_epoch=row['best_epoch'],variant=row['variant'],selection='Own complete validation MSE, independent stopping',loss_config=tp['loss_config']))
    atomic_json(a/'freeze.json',dict(**identity,validation_evidence=row,reason='Both training policies finished, selected only by validation MSE; test has not been opened.',created=now()))
    verify_artifact(a,'val');return str(a)

def validation_audit(out,tr,tp,smoke):
    f=out/'training/2021';last=torch.load(f/'last.pt',map_location='cpu',weights_only=True)
    initial=torch.load(f/'initial.pth',map_location='cpu',weights_only=True);equal_state(initial['cosine'],initial['plateau'])
    ia=read(f/'initialization_audit.json');assert set(ia['parameter_counts'].values())=={136150}
    assert ia['eval_all_rng_equal'] and ia['train_all_rng_equal']
    assert not any(ia['eval_max_prediction_difference'].values()) and not any(ia['train_max_prediction_difference'].values())
    opt=read(f/'optimizer_groups.json');assert opt['cosine']==opt['plateau']
    for rel,digest in tp['source_sha256'].items():assert sha(ROOT/rel)==sha(out/'training/source'/rel)==digest
    prefix=Path(tp['model_args']['root_path'])/'ETTm1.csv'
    assert sha(prefix)==tp['options']['train_val_csv_sha256']
    with prefix.open('rb') as file:assert sum(1 for _ in file)==46081
    finite(read(f/'branch_traces.json'))
    result=dict(status='verified',created=now(),smoke=smoke,test_accessed=False,runs={},source_count=len(tp['source_sha256']),initialization=ia)
    reftruth=None
    for row in tr['runs']:
        arm=row['variant'];policy=SchedulePolicy(arm,tp['options']['epochs']);curves=[v for v in last['curves'] if v['variant']==arm]
        assert [v['epoch'] for v in curves]==list(range(1,len(curves)+1))
        for c in curves:
            finite(c);assert c['learning_rate']==policy.lr
            assert c['train_loss']==c['train_mse'] and c['train_relation']==c['train_auxiliary_relation']==c['validation_relation']==0
            policy.step(c['validation_mse']);assert c['next_learning_rate']==policy.lr and c['stale']==policy.stale
        assert policy.state_dict()==last['policies'][arm]==row['policy'] and policy.stopped
        assert row['best_epoch']==policy.best_epoch and row['validation_mse']==policy.best
        updates=policy.epoch*math.ceil(tr['train_windows']/32)
        assert last['completed_updates'][arm]==row['completed_updates']==updates
        for st in last['optimizers'][arm]['state'].values():assert int(st['step'])==updates
        selected=torch.load(row['checkpoint'],map_location='cpu',weights_only=True)
        equal_state(selected,last['best_models'][arm])
        equal_state(selected,torch.load(f/f'epochs/{arm}_epoch{policy.best_epoch:02d}.pth',map_location='cpu',weights_only=True))
        with np.load(f/f'{arm}_selected_validation.npz') as z:p,y,x=[z[k].astype('float64') for k in ('prediction','truth','history')]
        assert p.shape==y.shape==x.shape==(tr['validation_windows'],96,7)
        assert all(np.isfinite(v).all() for v in (p,y,x))
        if reftruth is None:reftruth=y.copy()
        else:assert np.array_equal(y,reftruth)
        metric=score(p,y)
        for k in ('mse','mae','window_offset_mse','low4_mse'):assert abs(metric[k]-row['selected_validation'][k])<1e-7
        parts=np.load(f/f'{arm}_validation_components.npy',mmap_mode='r');means=np.load(f/f'{arm}_validation_history_mean.npy',mmap_mode='r')
        assert parts.shape==(*p.shape,8)
        err=0.
        for i in range(0,len(p),128):
            c=parts[i:i+128].astype(float);m=means[i:i+128].astype(float);assert np.isfinite(c).all() and np.isfinite(m).all()
            err=max(err,float(abs(c.sum(-1)+m-p[i:i+128]).max()))
        assert err<2e-5
        amp=float(((p.std(1)-y.std(1))**2).mean());metric['amplitude_error_mse']=amp;metric['shape_alignment_mse']=metric['mse']-metric['window_offset_mse']-amp
        per={name:score(p[:,:,j:j+1],y[:,:,j:j+1]) for j,name in enumerate(['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'])}
        qtime=[score(p[ix],y[ix]) for ix in np.array_split(np.arange(len(p)),4)]
        qstep=[score(p[:,i:i+24],y[:,i:i+24]) for i in range(0,96,24)]
        exact_previous=False
        if arm=='cosine' and not smoke:
            old=ROOT/'tmp/lag_study_timemix_pdm_bayesian_relation_20261009_seed2021_earlystop5/training/2021'
            oldprogress=read(old/'progress.json')
            for c in curves:
                oldpath=old/f"epochs/mse_epoch{c['epoch']:02d}.pth"
                if oldpath.exists():equal_state(torch.load(f/f"epochs/cosine_epoch{c['epoch']:02d}.pth",map_location='cpu',weights_only=True),torch.load(oldpath,map_location='cpu',weights_only=True))
            if oldprogress['best_epoch']['mse']==policy.best_epoch:
                equal_state(selected,torch.load(old/'mse_best.pth',map_location='cpu',weights_only=True))
                with np.load(old/'mse_selected_validation.npz') as z:assert np.array_equal(p,z['prediction']) and np.array_equal(y,z['truth'])
                exact_previous=True
        result['runs'][arm]=dict(metrics=metric,per_channel=per,time_quarters=qtime,horizon_quarters=qstep,policy=policy.state_dict(),completed_updates=updates,selected_updates=policy.best_epoch*math.ceil(tr['train_windows']/32),checkpoint_sha256=sha(row['checkpoint']),prediction_sha256=sha(f/f'{arm}_selected_validation.npz'),reconstruction_max_abs_error=err,exact_previous_mse=exact_previous)
    atomic_json(out/'validation_audit.json',result);return result

def test_all(out,tr,tp,device,fraction):
    t=out/'test';t.mkdir()
    atomic_json(t/'authorization.json',dict(user_request='做完实验默认再运行test',experiment_request='帮我进行实验',created=now(),selection='Both policy arms selected and frozen by own validation MSE; full test once; no re-selection'))
    full=ROOT/'dataset/ETT-small/ETTm1.csv';prefix=out/'training/train_val_data/ETTm1.csv'
    with full.open('rb') as f:assert b''.join(next(f) for _ in range(46081))==prefix.read_bytes()
    atomic_json(t/'data_identity.json',dict(full_csv_sha256=sha(full),prefix_sha256=sha(prefix)))
    for arm in ARMS:
        src=out/(arm+'_artifact');manifest,cfg=verify_artifact(src,'val');dst=t/(arm+'_artifact');dst.mkdir()
        shutil.copyfile(src/'checkpoint.pth',dst/'checkpoint.pth');atomic_json(dst/'model_config.json',dict(cfg,root_path=str(full.parent)))
        identity=dict(checkpoint_sha256=sha(dst/'checkpoint.pth'),model_config_sha256=sha(dst/'model_config.json'))
        atomic_json(dst/'manifest.json',dict(manifest,selection_status='frozen_for_test',**identity,validation_manifest_sha256=sha(src/'manifest.json'),config_changes={'root_path':str(full.parent)}))
        atomic_json(dst/'freeze.json',dict(**identity,validation_evidence=read(src/'freeze.json')['validation_evidence'],reason='User default test after validation freeze. Same checkpoint, only CSV root changes.',created=now()))
        verify_artifact(dst,'test')
    results={};truth=None
    for arm in ARMS:
        phase(out,'test',variant=arm)
        code="import runpy,sys,torch;torch.set_num_threads(2);"+(f"torch.cuda.set_per_process_memory_fraction({fraction});" if device=='cuda' else '')+"sys.argv=['scripts/evaluate_frozen_timemix.py']+sys.argv[1:];runpy.run_path('scripts/evaluate_frozen_timemix.py',run_name='__main__')"
        cmd=[sys.executable,'-u','-c',code,'--artifact',str(t/(arm+'_artifact')),'--split','test','--output',str(t/arm),'--device',device]
        child_run(out,cmd,t/(arm+'_launch.log'),out/'test_process.json')
        report=read(t/arm/'report.json');assert report['status']=='completed' and report['windows']==11425 and not report['smoke']
        m,cfg=verify_artifact(t/(arm+'_artifact'),'test');assert report['checkpoint_sha256']==m['checkpoint_sha256']
        with np.load(t/arm/'predictions.npz') as z:p,y=[z[k].astype(float) for k in ('prediction','truth')]
        assert p.shape==y.shape==(11425,96,7) and np.isfinite(p).all()
        if truth is None:truth=y.copy()
        else:assert np.array_equal(y,truth)
        metrics=score(p,y)
        for k in ('mse','mae'):assert abs(metrics[k]-report['metrics'][k])<1e-12
        results[arm]=dict(metrics=metrics,per_channel={name:score(p[:,:,j:j+1],y[:,:,j:j+1]) for j,name in enumerate(['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'])},checkpoint_sha256=m['checkpoint_sha256'],prediction_sha256=sha(t/arm/'predictions.npz'),selected_epoch=m['best_epoch'])
    atomic_json(t/'audit.json',dict(status='verified',runs=results,created=now()));return results

def publish(out,report):
    a=report['validation_audit'];test=report.get('test_audit',{})
    lines=['# MA1纯MSE训练日程对照结果','', '比较cosine早停5与平台降率早停8两套完整训练策略；两组独立停止，训练预算可能不同。单种子2021，重复使用验证集，不能视为跨种子稳定收益。','', '| 组 | 停止轮 | 选轮 | 更新数 | 验证MSE | 验证MAE | test MSE | test MAE |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    for arm in ARMS:
        r=a['runs'][arm];m=r['metrics'];te=test.get(arm,{}).get('metrics',{})
        lines.append(f"| {arm} | {r['policy']['epoch']} | {r['policy']['best_epoch']} | {r['completed_updates']} | {m['mse']:.10f} | {m['mae']:.10f} | {te.get('mse','未运行')} | {te.get('mae','未运行')} |")
    lines+=['','平台组降率记录：'+json.dumps(a['runs']['plateau']['policy']['reductions'],ensure_ascii=False),'','## 完整核验','','两组初值/初始函数/RNG、136150参数、优化器分组、CSV前缀46081行、原Scaler/Normalize、每轮学习率/早停重放、优化器步数、选轮检查点和8分量重构均已验证。两组纯MSE，没有关系/NLL/额外分支监督。test在两组验证冻结和审计后运行，使用同一已选权重。','',f"详细验证及变量/时间/预测步指标：`{out/'validation_audit.json'}`。",f"完整报告：`{out/'report.json'}`。"]
    for split,runs in [('validation',a['runs']),('test',test)]:
        if not runs:continue
        lines += ['',f'## {split} 指标变化','', '| 指标 | plateau减cosine | 相对变化 |','| --- | --- | --- |']
        for k in ['mse','mae','low4_mse','window_offset_mse','centered_amplitude_ratio']:
            x=runs['cosine']['metrics'][k];y=runs['plateau']['metrics'][k]
            lines.append(f'| {k} | {y-x:.10f} | {(y/x-1)*100:.4f}% |')
        lines += ['','| 变量 | MSE变化% | MAE变化% |','| --- | --- | --- |']
        for n in ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']:
            x=runs['cosine']['per_channel'][n];y=runs['plateau']['per_channel'][n]
            lines.append(f"| {n} | {(y['mse']/x['mse']-1)*100:.4f}% | {(y['mae']/x['mae']-1)*100:.4f}% |")
    doc=ROOT/'docs/experiments/PDMSchedule_results_20261010.md';doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for file,link in [(ROOT/'README.md','docs/experiments/PDMSchedule_results_20261010.md'),(ROOT/'docs/experiments/README.md','PDMSchedule_results_20261010.md')]:
        s=file.read_text();entry=f'\n- [MA1纯MSE日程对照结果 20261010]({link})\n'
        if link not in s:file.write_text(s+entry,encoding='utf-8')
    record=dict(experiment='pdm_schedule_results',status='completed_verified',output=str(out),document=str(doc.relative_to(ROOT)),completed_utc=now(),validation={k:v['metrics'] for k,v in a['runs'].items()},test={k:v['metrics'] for k,v in test.items()})
    # Lock read-modify-write to preserve other experiment state and history.
    with (ROOT/'tmp/pdm_schedule_registration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=ROOT/'tmp/timemix_0308_goal_state.json';state=read(path);entry=state.setdefault('pdm_schedule_results',{})
        if entry.get('current_run',{}).get('output')==str(out):entry['current_run'].update(status='completed_verified',completion_audit=str(out/'completion_audit.json'),results_document=str(doc))
        entry.update(status='completed_verified',results=record);entry.setdefault('run_history',[]).append(record);atomic_json(path,state)
        path=ROOT/'experiments/experiment_index_20261008.json';index=read(path);index.setdefault('supplemental_records',[]).append(record);atomic_json(path,index)
    atomic_json(out/'registration.json',record)

def main():
    os.chdir(ROOT);p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--epochs',type=int,default=30);p.add_argument('--gpu-memory-fraction',type=float,default=.5);p.add_argument('--train-limit',type=int,default=0);p.add_argument('--val-limit',type=int,default=0);opts=p.parse_args()
    if opts.epochs!=30 and not (opts.train_limit and opts.val_limit):p.error('Formal horizon fixed30')
    out=opts.output.resolve();out.mkdir(parents=True,exist_ok=False)
    lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'pid.txt').write_text(str(os.getpid())+'\n');torch.set_num_threads(2)
    smoke=bool(opts.train_limit or opts.val_limit);report=dict(status='running',smoke=smoke,test_accessed=False)
    atomic_json(out/'pipeline_options.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(opts).items()})
    atomic_json(out/'report.json',report)
    try:
        phase(out,'training')
        cmd=[sys.executable,'-u','scripts/train_timemix_pdm_schedule.py','--output',str(out/'training'),'--device',opts.device,'--epochs',str(opts.epochs),'--gpu-memory-fraction',str(opts.gpu_memory_fraction),'--train-limit',str(opts.train_limit),'--val-limit',str(opts.val_limit)]
        child_run(out,cmd,out/'training_launch.log',out/'process.json')
        tr=read(out/'training/report.json');tp=read(out/'training/protocol.json');assert tr['status']=='completed'
        if not smoke:assert (tr['train_windows'],tr['validation_windows'])==(34369,11425)
        phase(out,'validation_audit');audit=validation_audit(out,tr,tp,smoke);report.update(runs=tr['runs'],validation_audit=audit)
        for row in tr['runs']:
            phase(out,'freezing',variant=row['variant']);row['artifact']=freeze(out,row,tp,smoke)
        atomic_json(out/'report.json',report)
        if not smoke:
            report['test_audit']=test_all(out,tr,tp,opts.device,opts.gpu_memory_fraction);report['test_accessed']=True
        report.update(status='completed_verified',completed_utc=now())
        atomic_json(out/'completion_audit.json',dict(status='verified',validation=audit,test=report.get('test_audit'),test_accessed=report['test_accessed']))
        atomic_json(out/'report.json',report)
        if not smoke:publish(out,report)
        phase(out,'completed_verified');print('SCHEDULE EXPERIMENT COMPLETE',out,flush=True)
    except BaseException as e:
        report.update(status='failed',error=repr(e));atomic_json(out/'report.json',report);phase(out,'failed',error=repr(e));raise
if __name__=='__main__':main()
