"""Sequential screening and conditional confirmation; validation only."""
import os,sys,json,subprocess,fcntl
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_timemix_pdm_trend_decay import atomic_json,ARMS
from scripts.run_timemix_pdm_coarse_ma_experiment import make_artifact,audit_predictions
from scripts.evaluate_frozen_timemix import verify_artifact
import torch

def decision(rows):
    by={r['variant']:r for r in rows};a,b=[by[k] for k in ARMS]
    da,db=[r['mechanism_diagnostics'] for r in (a,b)]
    def gain(d):return sum(d['trend_path_directional_gain'][f'layer{i}_scale2'] for i in [0,1])/2
    def wave(d):return sum(d['per_channel_waveform_mse'][i] for i in [0,2])/2
    delta=dict(validation_mse=b['validation_mse']-a['validation_mse'],
        hufl_mufl_waveform_mse=wave(db)-wave(da),
        lufl_coarse_harm=db['coarse_mse_excess_per_channel'][4]-da['coarse_mse_excess_per_channel'][4],
        trend_path_gain=gain(db)-gain(da))
    checks={k:(v>0 if k=='trend_path_gain' else v<0) for k,v in delta.items()}
    return dict(delta=delta,checks=checks,passed=all(checks.values()))

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--patience',type=int,default=5)
    p.add_argument('--train-limit',type=int,default=0);p.add_argument('--val-limit',type=int,default=0)
    opts=p.parse_args();os.chdir(ROOT);torch.set_num_threads(2)
    out=opts.output.resolve();out.mkdir(parents=True,exist_ok=False)
    lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'pid.txt').write_text(str(os.getpid()))
    report=dict(status='running',test_accessed=False,smoke=bool(opts.train_limit or opts.val_limit),
        plan_seeds=[2021,2022,2023],runs=[],seed_decisions={},started_at=datetime.now(timezone.utc).isoformat())
    def save(status,**kw):
        report.update(status=status,**kw);atomic_json(out/'report.json',report)
        atomic_json(out/'status.json',dict(status=status,updated=datetime.now(timezone.utc).isoformat(),active_seed=report.get('active_seed')))
    save('screening',active_seed=2021)
    try:
        for seed in [2021,2022,2023]:
            if seed!=2021 and (report['smoke'] or not report['seed_decisions']['2021']['passed']):break
            stage=out/f'seed{seed}';stage.mkdir()
            save('screening' if seed==2021 else 'confirmation',active_seed=seed)
            command=[sys.executable,'-u','scripts/train_timemix_pdm_trend_decay.py','--output',str(stage/'training'),
                '--device',opts.device,'--seeds',str(seed),'--epochs',str(opts.epochs),'--patience',str(opts.patience),
                '--gpu-memory-fraction','1.0','--train-limit',str(opts.train_limit),'--val-limit',str(opts.val_limit)]
            atomic_json(stage/'command.json',command)
            with (stage/'training_launch.log').open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(stage/'process.json',dict(pid=child.pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],command=command))
                code=child.wait()
            if code:raise RuntimeError(f'seed {seed} trainer exit {code}; see {stage}/training_launch.log')
            tr=json.loads((stage/'training/report.json').read_text())
            protocol=json.loads((stage/'training/protocol.json').read_text())
            assert tr['status']=='completed' and len(tr['runs'])==2
            if not report['smoke']:assert (tr['train_windows'],tr['validation_windows'])==(34369,11425)
            rows=[]
            for source in tr['runs']:
                row=dict(source);name=row['variant'];folder=stage/'training'/str(seed)
                row['validation_audit']=audit_predictions(folder/(name+'_selected_validation.npz'),row['selected_validation'])
                row['artifact']=make_artifact(stage/(name+'_artifact'),SimpleNamespace(**protocol['model_args']),protocol,row)
                verify_artifact(row['artifact'],'val')
                rows.append(row)
            assert rows[0]['trained_epochs']==rows[1]['trained_epochs']
            report['runs'].extend(rows)
            report['seed_decisions'][str(seed)]=decision(rows)
            save('seed_completed',active_seed=seed)
        decisions=list(report['seed_decisions'].values())
        confirmed=(len(decisions)==3 and all(x['delta']['validation_mse']<0 for x in decisions)
            and sum(x['delta']['hufl_mufl_waveform_mse'] for x in decisions)<0
            and sum(x['delta']['lufl_coarse_harm'] for x in decisions)<0
            and sum(x['delta']['trend_path_gain'] for x in decisions)>0)
        save('completed',confirmed_in_planned_seeds=confirmed,active_seed=None,
            conclusion='supported_in_three_seeds' if confirmed else 'not_confirmed',
            finished_at=datetime.now(timezone.utc).isoformat())
        lines=['# PDM趋势路径衰减实验结果','',f'结论：{report["conclusion"]}；仅验证集，未访问test。','',
            '|seed|组|完成轮|所选轮|验证MSE|验证MAE|','|---|---|---:|---:|---:|---:|']
        for row in report['runs']:
            lines.append(f'|{row["seed"]}|{row["variant"]}|{row["trained_epochs"]}|{row["best_epoch"]}|{row["validation_mse"]:.9f}|{row["selected_validation"]["mae"]:.9f}|')
        lines+=['','预注册机制门槛与每个种子差值：','```json',json.dumps(report['seed_decisions'],indent=2),'```',
            '仅提高权重/响应不是成功。单种子筛查或三个种子一致都不是独立数据集或统计显著性证明。']
        (out/'results_zh.md').write_text('\n'.join(lines))
        print('COMPLETE',out,report['conclusion'],flush=True)
    except BaseException as exc:
        save('failed',error=repr(exc));raise
    finally:lock.close()
if __name__=='__main__':main()
