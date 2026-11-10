"""MA1 disjoint patch covariance comparison, audit/freeze then full test."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,json,hashlib,shutil,subprocess,sys,math,fcntl
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.train_timemix_pdm_state_patch_concat import atomic_json,ARMS,WEIGHTS,learning_rate
from scripts.audit_pdm_patch_covariance import numpy_covariance_parts
from scripts.evaluate_frozen_timemix import verify_artifact
from scripts.diagnose_timemix_linear_reference import score
now=lambda:datetime.now(timezone.utc).isoformat()
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

REFERENCE=ROOT/'tmp/lag_study_timemix_pdm_state_patch_20261011_seed2021_earlystop5'
POINT_REFERENCE=ROOT/'tmp/lag_study_timemix_pdm_patch_covariance_20261011_seed2021_earlystop5'

JOINT_REFERENCE=ROOT/'tmp/lag_study_timemix_pdm_joint_patch_20261011_seed2021_earlystop5'

def reference_evidence():
    evidence={}
    for ref in (REFERENCE,JOINT_REFERENCE,POINT_REFERENCE):
        assert read(ref/'status.json')['status']=='completed_verified'
        assert read(ref/'completion_audit.json')['status']=='verified'
        files=['report.json','completion_audit.json','validation_audit.json','test/audit.json','training/2021/initial.pth','training/2021/progress.json','training/2021/optimizer_groups.json']
        for arm in ARMS:
            vm,_=verify_artifact(ref/(arm+'_artifact'),'val')
            tm,_=verify_artifact(ref/'test'/(arm+'_artifact'),'test')
            assert vm['checkpoint_sha256']==tm['checkpoint_sha256']
            te=read(ref/'test'/arm/'report.json')
            assert te['status']=='completed' and te['windows']==11425
            files.extend([arm+'_artifact/checkpoint.pth',arm+'_artifact/manifest.json','test/'+arm+'/report.json','test/'+arm+'/predictions.npz'])
        for name in files:evidence[os.path.relpath(ref/name,REFERENCE)]=sha(ref/name)
    return evidence

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
    atomic_json(a/'manifest.json',dict(kind=row['model_config']['model'],selection_status='smoke_validation_only' if smoke else 'frozen_validation_diagnostic',**identity,neural_training_source_sha256=tp['source_sha256'],seed=row['seed'],best_epoch=row['best_epoch'],variant=row['variant'],selection='Own complete validation MSE, joint patience5',loss_config=tp['loss_config']))
    atomic_json(a/'freeze.json',dict(**identity,validation_evidence=row,reason='Both disjoint patch covariance arms finished, selected only by validation MSE; test has not been opened.',created=now()))
    verify_artifact(a,'val');return str(a)

def validation_audit(out,tr,tp,smoke):
    f=out/'training/2021';last=torch.load(f/'last.pt',map_location='cpu',weights_only=True)
    initial=torch.load(f/'initial.pth',map_location='cpu',weights_only=True)
    for arm in ARMS:equal_state(initial[ARMS[0]],initial[arm])
    ia=read(f/'initialization_audit.json');assert set(ia['parameter_counts'].values())=={153058}
    assert ia['eval_all_rng_equal'] and ia['train_all_rng_equal']
    assert not any(ia['eval_max_prediction_difference'].values()) and not any(ia['train_max_prediction_difference'].values())
    opt=read(f/'optimizer_groups.json');assert all(opt[ARMS[0]]==opt[a] for a in ARMS)
    for rel,digest in tp['source_sha256'].items():assert sha(ROOT/rel)==sha(out/'training/source'/rel)==digest
    prefix=Path(tp['model_args']['root_path'])/'ETTm1.csv'
    assert sha(prefix)==tp['options']['train_val_csv_sha256']
    with prefix.open('rb') as file:assert sum(1 for _ in file)==46081
    finite(read(f/'branch_traces.json'))
    result=dict(status='verified',created=now(),smoke=smoke,test_accessed=False,runs={},source_count=len(tp['source_sha256']),initialization=ia)
    reftruth=None
    best={a:float('inf') for a in ARMS};best_epoch={a:0 for a in ARMS};stale=0
    for epoch in range(1,last['epoch']+1):
        assert stale < tp['options']['patience']
        improved=False
        rows=[c for c in last['curves'] if c['epoch']==epoch]
        assert {c['variant'] for c in rows}==set(ARMS)
        for c in rows:
            arm=c['variant'];finite(c)
            assert c['learning_rate']==learning_rate(arm,epoch-1,tp['options']['epochs'])
            assert abs(c['train_loss']-c['train_mse']-WEIGHTS[arm]*c['train_relation'])<2e-6
            assert c['train_auxiliary_relation']==0
            if c['validation_mse']<best[arm]:best[arm]=c['validation_mse'];best_epoch[arm]=epoch;improved=True
        stale=0 if improved else stale+1
    assert stale==last['stale'] and best==last['best'] and best_epoch==last['best_epoch']
    assert stale>=tp['options']['patience'] or last['epoch']==tp['options']['epochs']
    for row in tr['runs']:
        arm=row['variant'];curves=[v for v in last['curves'] if v['variant']==arm]
        assert [v['epoch'] for v in curves]==list(range(1,last['epoch']+1))
        assert row['best_epoch']==best_epoch[arm] and row['validation_mse']==best[arm]
        assert row['trained_epochs']==last['epoch'] and row['joint_stale']==stale
        updates=last['epoch']*math.ceil(tr['train_windows']/32)
        assert last['completed_updates']==updates
        for st in last['optimizers'][arm]['state'].values():assert int(st['step'])==updates
        selected=torch.load(row['checkpoint'],map_location='cpu',weights_only=True)
        equal_state(selected,last['best_models'][arm])
        head_norms={k:float(v.double().norm()) for k,v in selected.items() if '.readout.output_projection.weight' in k}
        assert len(head_norms)==4 and all(math.isfinite(v) and v>0 for v in head_norms.values())
        equal_state(selected,torch.load(f/f'epochs/{arm}_epoch{best_epoch[arm]:02d}.pth',map_location='cpu',weights_only=True))
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
        relation=numpy_covariance_parts(p,y,tp['loss_config']['bandwidth'],tp['loss_config']['patch_length'])
        for key,value in relation.items():assert abs(value-row['selected_validation'][key])<1e-6
        amp=float(((p.std(1)-y.std(1))**2).mean());metric['amplitude_error_mse']=amp;metric['shape_alignment_mse']=metric['mse']-metric['window_offset_mse']-amp
        per={name:score(p[:,:,j:j+1],y[:,:,j:j+1]) for j,name in enumerate(['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'])}
        qtime=[score(p[ix],y[ix]) for ix in np.array_split(np.arange(len(p)),4)]
        qstep=[score(p[:,i:i+24],y[:,i:i+24]) for i in range(0,96,24)]
        exact_previous=False # New readout architecture: do not require old same-epoch weights.
        result['runs'][arm]=dict(state_output_weight_norms=head_norms,metrics=metric,boundary=patch_boundary_metrics(p,y),per_channel=per,time_quarters=qtime,horizon_quarters=qstep,policy=dict(epoch=last['epoch'],best_epoch=best_epoch[arm],stale=stale,kind='joint'),relation=relation,lambda_weight=WEIGHTS[arm],completed_updates=updates,selected_updates=best_epoch[arm]*math.ceil(tr['train_windows']/32),checkpoint_sha256=sha(row['checkpoint']),prediction_sha256=sha(f/f'{arm}_selected_validation.npz'),reconstruction_max_abs_error=err,old_same_epoch_weight_comparison_applicable=exact_previous)
    reference_initial=torch.load(REFERENCE/'training/2021/initial.pth',map_location='cpu',weights_only=True)['mse']
    oldopt=read(REFERENCE/'training/2021/optimizer_groups.json')['mse']
    for arm in ARMS:
        extra=set(initial[arm])-set(reference_initial)
        assert len(extra)==8 and all('.patch_driver_projection.' in k for k in extra)
        for k,v in reference_initial.items():assert torch.equal(initial[arm][k],v),k
        for k in extra:
            v=initial[arm][k]
            expected=torch.zeros_like(v) if k.endswith('.bias') else torch.eye(32).repeat(1,4)/4
            assert torch.equal(v,expected),k
            assert opt[arm][k]==(0. if k.endswith('.bias') else .0001)
        assert {k:v for k,v in opt[arm].items() if k in oldopt}==oldopt
        selected=torch.load(next(r['checkpoint'] for r in tr['runs'] if r['variant']==arm),map_location='cpu',weights_only=True)
        changes={k:float((selected[k]-initial[arm][k]).double().norm()) for k in extra if k.endswith('.weight')}
        assert len(changes)==4 and all(math.isfinite(v) and v>0 for v in changes.values())
        result['runs'][arm]['projection_change_from_mean_initialization']=changes
    result['reference_common_initial_and_optimizer_exact']=True
    result['projection_initialized_to_mean_mapping']=True
    result['initial_function_matches_mean_pool_within_tolerance_not_bitwise']=True
    atomic_json(out/'validation_audit.json',result);return result

def test_all(out,tr,tp,device,fraction):
    t=out/'test';t.mkdir()
    atomic_json(t/'authorization.json',dict(user_request='做完实验默认再运行test',experiment_request='用户要求按顺序拼接后投影并启动训练：原A递推，单隐状态MLP输出4点，纯MSE与MSE加片段核协方差两组',created=now(),selection='Both disjoint patch covariance arms selected and frozen by own validation MSE; full test once; no re-selection'))
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
        metrics['amplitude_error_mse']=float(((p.std(1)-y.std(1))**2).mean())
        metrics['shape_alignment_mse']=metrics['mse']-metrics['window_offset_mse']-metrics['amplitude_error_mse']
        receipt=read(t/(arm+'_artifact')/'test_result.json');assert receipt['checkpoint_sha256']==m['checkpoint_sha256']
        assert m['checkpoint_sha256']==verify_artifact(out/(arm+'_artifact'),'val')[0]['checkpoint_sha256']
        relation=numpy_covariance_parts(p,y,tp['loss_config']['bandwidth'],tp['loss_config']['patch_length'])
        results[arm]=dict(boundary=patch_boundary_metrics(p,y),relation=relation,time_quarters=[score(p[ix],y[ix]) for ix in np.array_split(np.arange(len(p)),4)],horizon_quarters=[score(p[:,i:i+24],y[:,i:i+24]) for i in range(0,96,24)],metrics=metrics,per_channel={name:score(p[:,:,j:j+1],y[:,:,j:j+1]) for j,name in enumerate(['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'])},checkpoint_sha256=m['checkpoint_sha256'],prediction_sha256=sha(t/arm/'predictions.npz'),selected_epoch=m['best_epoch'])
    atomic_json(t/'audit.json',dict(status='verified',runs=results,created=now()));return results

def patch_boundary_metrics(p,y):
    e=np.diff(p.astype(float)-y.astype(float),axis=1)
    boundary=(np.arange(e.shape[1])+1)%4==0
    return dict(boundary_increment_error_mse=float((e[:,boundary]**2).mean()),interior_increment_error_mse=float((e[:,~boundary]**2).mean()),boundary_count=int(boundary.sum()),interior_count=int((~boundary).sum()))


def publish(out,report):
    import html
    a=report['validation_audit']['runs'];test=report['test_audit']
    def table(headers,rows):
        return '<table><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    lines=['<!doctype html><meta charset="utf-8"><title>MA1顺序拼接投影单状态四点读出完整实验</title><style>body{max-width:1200px;margin:30px auto;font:16px/1.6 system-ui}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}</style><h1>MA1顺序拼接投影单状态四点读出两组完整实验</h1>',
    '<p>四个连续32维未来投影特征（含位置编码）按时间顺序拼接为128维，经可学习Linear128到32聚合为一个驱动。四尺度各一层，初始化为四块I/4和零偏置；相对均值聚合增加16512参数，共153058。保留原A每片段递推一次，共24次，每个16维状态经GELU16到32到4输出四值；无重叠、接缝损失、输出反馈或额外平滑。历史扫描、趋势96点、归一化保持。新两组同scratch初值；与旧均值聚合共有参数精确相同、初始函数仅数值容差内一致。固定cosine30最多30轮、两组联合patience5，各自valMSE选轮，两组冻结后各完整test一次。原值长度4步长4、lag0核协方差损失sigma1/lambda1保持。旧对照只读，新旧参数数量和停止预算可能不同；单种子反复使用val/test，不能称独立显著性、跨种子稳定或唯一因果。</p>']
    rows=[]
    for arm in ARMS:
        x=a[arm];y=test[arm];rows.append([arm,x['policy']['epoch'],x['policy']['best_epoch'],x['completed_updates'],x['metrics']['mse'],x['metrics']['mae'],y['metrics']['mse'],y['metrics']['mae']])
    lines.append(table(['组','停止轮','选轮','更新数','val MSE','val MAE','test MSE','test MAE'],rows))
    for split,runs in [('val',a),('test',test)]:
        lines.append('<h2>'+split+'共同定义辅助项与误差分解</h2>')
        keys=['mse','mae','low4_mse','window_offset_mse','amplitude_error_mse','shape_alignment_mse']
        lines.append(table(['组','预测核协方差范数平方','不重叠片段核协方差']+keys,[[arm,*[runs[arm]['relation'][k] for k in ('prediction_covariance_norm2','patch_covariance')],*[runs[arm]['metrics'][k] for k in keys]] for arm in ARMS]))
        for arm in ARMS[1:]:
            dm=runs[arm]['metrics']['mse']/runs['mse']['metrics']['mse']-1
            da=runs[arm]['metrics']['mae']/runs['mse']['metrics']['mae']-1
            lines.append(f'<p>{arm}相对mse：MSE {dm:+.2%}，MAE {da:+.2%}。'+('MAE恶化。' if da>0 else '')+'</p>')
        lines.append(table(['组','变量','MSE','MAE','相对mse的MSE变化'],[[arm,c,v['mse'],v['mae'],v['mse']/runs['mse']['per_channel'][c]['mse']-1] for arm in ARMS for c,v in runs[arm]['per_channel'].items()]))
        for key in ('time_quarters','horizon_quarters'):
            lines.append('<h3>'+key+'</h3>');lines.append(table(['组','段','MSE','MAE'],[[arm,i+1,v['mse'],v['mae']] for arm in ARMS for i,v in enumerate(runs[arm][key])]))
    reference=read(REFERENCE/'report.json')
    lines.append('<h2>只读对照旧均值聚合单状态四点实验</h2><p>旧结果不重训不重测。新旧参数数不同、联合早停分组独立，最终更新数可能不同；不能把新旧差异唯一归因于核损失。</p>')
    for split,new,old in [('val',a,reference['validation_audit']['runs']),('test',test,reference['test_audit'])]:
        lines.append('<h3>'+split+'</h3>'+table(['架构','损失','MSE','MAE','选轮','更新数'],[[label,arm,runs[arm]['metrics']['mse'],runs[arm]['metrics']['mae'],runs[arm].get('policy',{}).get('best_epoch',runs[arm].get('selected_epoch')),(report if label=='state_patch_concat' else reference)['validation_audit']['runs'][arm]['completed_updates']] for label,runs in [('mean_pool',old),('state_patch_concat',new)] for arm in ARMS]))
        boundaries=[]
        for arm in ARMS:
            path=REFERENCE/(f'training/2021/{arm}_selected_validation.npz' if split=='val' else f'test/{arm}/predictions.npz')
            with np.load(path) as z:old_boundary=patch_boundary_metrics(z['prediction'],z['truth'])
            for label,b in [('mean_pool',old_boundary),('state_patch_concat',new[arm]['boundary'])]:
                boundaries.append([label,arm,b['boundary_increment_error_mse'],b['interior_increment_error_mse']])
        lines.append(table(['架构','损失','跨片段边界增量误差MSE','片段内部增量误差MSE'],boundaries))
    joint=read(JOINT_REFERENCE/'report.json')
    lines.append('<h2>只读对照上一轮四状态联合残差读出</h2><p>上一轮为四状态拼接加联合残差。本轮是顺序拼接投影驱动、一个状态直接输出四值。相对四状态残差模型还改变递推次数与读出，不能唯一归因于聚合。</p>')
    for split,new,old in [('val',a,joint['validation_audit']['runs']),('test',test,joint['test_audit'])]:
        lines.append('<h3>'+split+'</h3>'+table(['架构','损失','MSE','MAE','边界增量误差','内部增量误差'],[[label,arm,*[runs[arm]['metrics'][k] for k in ('mse','mae')],*[runs[arm]['boundary'][k] for k in ('boundary_increment_error_mse','interior_increment_error_mse')]] for label,runs in [('four_state_residual',old),('one_state_four_values',new)] for arm in ARMS]))
    curves=read(out/'training/2021/progress.json')['curves']
    reference_curves=read(REFERENCE/'training/2021/progress.json')['curves']
    lines.append('<h2>同轮训练与验证</h2>');lines.append(table(['组','轮','train MSE','train auxiliary','val MSE','val MAE','val prediction covariance norm2','val covariance distance'],[[r[k] for k in ('variant','epoch','train_mse','train_relation','validation_mse','validation_mae','validation_prediction_norm2','validation_covariance')] for r in curves]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for label,cs in [('state_patch_concat',curves),('mean_pool',reference_curves)]:
        for arm in ARMS:
            rows=[r for r in cs if r['variant']==arm]
            for ax,key in zip(axes,('train_mse','validation_mse')):ax.plot([r['epoch'] for r in rows],[r[key] for r in rows],label=label+'/'+arm,linestyle='-' if label=='state_patch_concat' else '--');ax.set_title(key);ax.set_xlabel('Epoch');ax.grid(alpha=.3)
    axes[0].legend();fig.tight_layout();fig.savefig(out/'curves.png',dpi=150);plt.close(fig)
    lines.append('<img style="max-width:100%" src="../../'+str(out.relative_to(ROOT))+'/curves.png">')
    traces=read(out/'training/2021/branch_traces.json');rows=[]
    for arm in ARMS:
        for epoch in sorted({r['epoch'] for r in traces}):
            for group in ('all','fine_trend_head','state_patch_heads','patch_driver_projection'):
                gs=[r['gradient_comparison'][group] for r in traces if r['variant']==arm and r['epoch']==epoch]
                rows.append([arm,epoch,group,float(np.median([g['weighted_ratio'] for g in gs])),float(np.median([g['cosine'] for g in gs])),sum(g['cosine']<0 for g in gs),len(gs)])
    lines.append('<h2>同次forward分项参数梯度</h2>'+table(['组','轮','范围','加权范数比中位数','余弦中位数','冲突数','采样数'],rows))
    lines.append('<p>梯度不是Adam实际更新方向。固定lambda不限制梯度范数比；lag0协方差对片段集合排列不敏感，片段内部顺序通过核输入保留。协方差距离下降不等于点预测成功；形状项也受幅值影响。完整数值、权重SHA、11425窗口与test回执见completion_audit.json和test/audit.json。</p>')
    doc=ROOT/'docs/experiments/PDMStatePatchConcat_results_20261012.html';doc.write_text('\n'.join(lines))
    record=dict(experiment='pdm_state_patch_concat_results',status='completed_verified',output=str(out),document=str(doc.relative_to(ROOT)),completed_utc=now(),validation={k:v['metrics'] for k,v in a.items()},test={k:v['metrics'] for k,v in test.items()})
    with (ROOT/'tmp/pdm_state_patch_concat_registration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=ROOT/'tmp/timemix_0308_goal_state.json';state=read(path);entry=state.setdefault('pdm_state_patch_concat_results',{})
        if entry.get('current_run',{}).get('output')==str(out):entry['current_run'].update(status='completed_verified',results_document=str(doc))
        entry.update(status='completed_verified',results=record);entry.setdefault('run_history',[]).append(record);atomic_json(path,state)
        path=ROOT/'experiments/experiment_index_20261008.json';index=read(path);index.setdefault('supplemental_records',[]).append(record);atomic_json(path,index)
        for file,link in [(ROOT/'README.md','docs/experiments/PDMStatePatchConcat_results_20261012.html'),(ROOT/'docs/experiments/README.md','PDMStatePatchConcat_results_20261012.html')]:
            text=file.read_text()
            if link not in text:file.write_text(text+f'\n- [MA1顺序拼接投影单状态四点读出两组完整实验 20261012]({link})\n')
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
        atomic_json(out/'reference_evidence.json', reference_evidence())
        phase(out,'training')
        cmd=[sys.executable,'-u','scripts/train_timemix_pdm_state_patch_concat.py','--output',str(out/'training'),'--device',opts.device,'--epochs',str(opts.epochs),'--gpu-memory-fraction',str(opts.gpu_memory_fraction),'--train-limit',str(opts.train_limit),'--val-limit',str(opts.val_limit)]
        child_run(out,cmd,out/'training_launch.log',out/'process.json')
        tr=read(out/'training/report.json');tp=read(out/'training/protocol.json');assert tr['status']=='completed'
        if not smoke:assert (tr['train_windows'],tr['validation_windows'])==(34369,11425)
        phase(out,'validation_audit');audit=validation_audit(out,tr,tp,smoke);report.update(runs=tr['runs'],validation_audit=audit)
        for row in tr['runs']:
            phase(out,'freezing',variant=row['variant']);row['artifact']=freeze(out,row,tp,smoke)
        atomic_json(out/'report.json',report)
        if not smoke:
            report['test_audit']=test_all(out,tr,tp,opts.device,opts.gpu_memory_fraction);report['test_accessed']=True
        assert read(out/'reference_evidence.json')==reference_evidence(), 'Reference evidence changed'
        report.update(status='completed_verified',completed_utc=now())
        atomic_json(out/'completion_audit.json',dict(status='verified',validation=audit,test=report.get('test_audit'),test_accessed=report['test_accessed']))
        atomic_json(out/'report.json',report)
        if not smoke:publish(out,report)
        phase(out,'completed_verified');print('STATE PATCH EXPERIMENT COMPLETE',out,flush=True)
    except BaseException as e:
        report.update(status='failed',error=repr(e));atomic_json(out/'report.json',report);phase(out,'failed',error=repr(e));raise
if __name__=='__main__':main()
