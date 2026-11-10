"""Controlled corrected-covariance study plus train-only ETTm1 feasibility screen."""
import argparse, os, sys, json, hashlib, copy, math
from pathlib import Path
from datetime import datetime, timezone
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from utils.predictable_autocov_loss import DEFAULT_LAGS, sample_autocovariance, expected_centered_autocovariance, predictable_autocov_loss
H=96;LAGS=DEFAULT_LAGS;ARMS=('mse','raw_cov','oracle_corrected','estimated_corrected','biased_corrected');SEED=2021;COEF=.025

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def cov_np(z):
    z=z-z.mean(1,keepdims=True)
    return np.stack([(z[:,:H-k]*z[:,k:]).sum(1)/H for k in LAGS],-1)

def correction_np(s):
    p=np.eye(H)-np.ones((H,H))/H;s=p@s@p
    return np.array([np.diag(s,k).sum()/H for k in LAGS])

def make_synthetic(scenario):
    rng=np.random.default_rng(SEED);features=np.c_[np.ones(10240),rng.normal(size=(10240,4))]
    t=np.arange(H);wave=np.sin(2*np.pi*t/24);cos=np.cos(2*np.pi*t/24);ramp=(t-(H-1)/2)/H
    weights=np.stack([.6*wave,.4*wave,.4*cos,.2*ramp,.15*np.ones(H)])
    mean=features@weights;regime=(features[:,-1]>0).astype(int);truth=mean.copy();sigmas=[]
    for g in range(2):
        rho=(.85,.35)[g];std=(.8,1.2)[g]
        sigma=std**2*rho**np.abs(np.arange(H)[:,None]-np.arange(H)[None,:])
        if scenario=='deterministic':sigma=np.zeros((H,H))
        ids=np.flatnonzero(regime==g)
        if scenario!='deterministic':truth[ids]+=rng.normal(size=(len(ids),H))@np.linalg.cholesky(sigma).T
        if scenario=='random_phase':
            v=.9*np.sin(2*np.pi*t/32)
            truth[ids]+=rng.choice([-1.,1.],size=(len(ids),1))*v[None]
            sigma=sigma+np.outer(v,v)
        sigmas.append(sigma)
    return features,mean[...,None],truth[...,None],regime,weights,np.stack(sigmas)

def metrics(pred,truth,mean):
    p=pred.astype(np.float64);y=truth.astype(np.float64);m=mean.astype(np.float64)
    return dict(mse=float(np.mean((p-y)**2)),mae=float(np.mean(np.abs(p-y))),conditional_mean_mse=float(np.mean((p-m)**2)),
                predictable_covariance_mse=float(np.mean((cov_np(p)-cov_np(m))**2)),sample_target_covariance_mse=float(np.mean((cov_np(p)-cov_np(y))**2)),
                centered_energy=float(np.mean((p-p.mean(1,keepdims=True))**2)),true_mean_centered_energy=float(np.mean((m-m.mean(1,keepdims=True))**2)))

def synthetic(out):
    result={};torch.set_num_threads(2)
    for scenario in ('deterministic','correlated_ar','random_phase'):
        folder=out/scenario;folder.mkdir();features,mean,y,regime,true_weights,sigmas=make_synthetic(scenario)
        reference={};estimates={};oracle=np.stack([correction_np(s) for s in sigmas])[:,None,:]
        for name,cols in [('estimated_corrected',[0,1,2,3,4]),('biased_corrected',[0,1,3,4])]:
            w=np.linalg.lstsq(features[:2048,cols],y[:2048,:,0],rcond=None)[0]
            residual=y[2048:4096]-(features[2048:4096,cols]@w)[...,None]
            c=cov_np(residual);estimate=np.stack([c[regime[2048:4096]==g].mean(0) for g in range(2)])
            estimates[name]=estimate;reference[name]=dict(columns=cols,calibration_mse=float(np.mean(residual**2)),correction=estimate.tolist(),oracle_error_l2=float(np.linalg.norm(estimate-oracle)))
            np.savez_compressed(folder/(name+'_reference.npz'),weights=w,correction=estimate)
        scale=np.maximum(y[:8192].var(axis=(0,1))*1.01,1e-8)
        np.savez_compressed(folder/'data.npz',features=features,truth=y,conditional_mean=mean,regime=regime,true_weights=true_weights,noise_covariance=sigmas,oracle_correction=oracle,channel_variance=scale)
        # Population amplitude risk: omit only prediction-independent target-variance constants.
        point=features[0];m=point@true_weights;c=cov_np(m[None,:,None])[0,0];noise=oracle[regime[0],0];energy=float(np.mean(m*m))
        amplitude={}
        for arm,nc in [('raw_cov',np.zeros_like(noise)),('oracle_corrected',noise)]:
            def risk(a):return energy*(a-1)**2+COEF*np.mean(((a*a*c-(c+noise-nc))/scale[0])**2)
            fit=minimize_scalar(risk,bounds=(0,4),method='bounded',options={'xatol':1e-12})
            assert fit.success
            amplitude[arm]=dict(optimal_amplitude=float(fit.x),risk_without_constant=float(fit.fun),conditional_mean_mse=float(energy*(fit.x-1)**2))
        assert abs(amplitude['oracle_corrected']['optimal_amplitude']-1)<1e-5
        models={a:torch.nn.Parameter(torch.zeros(5,H)) for a in ARMS};optimizers={a:torch.optim.Adam([models[a]],lr=.01) for a in ARMS}
        initial={a:models[a].detach().clone() for a in ARMS};torch.save(initial,folder/'initial.pt')
        xf=torch.tensor(features,dtype=torch.float32);yf=torch.tensor(y,dtype=torch.float32)
        corrections={'mse':np.zeros_like(oracle),'raw_cov':np.zeros_like(oracle),'oracle_corrected':oracle,**estimates}
        correction_tensors={a:torch.tensor(v,dtype=torch.float32) for a,v in corrections.items()}
        curves=[];best={a:float('inf') for a in ARMS};best_epoch={};best_weights={};gradient_samples=[]
        for epoch in range(1,41):
            order=np.random.default_rng(SEED+epoch).permutation(np.arange(4096,8192))
            totals={a:np.zeros(3) for a in ARMS}
            for step,start in enumerate(range(0,len(order),128),1):
                ids=order[start:start+128];x=xf[ids];target=yf[ids]
                for arm in ARMS:
                    optimizers[arm].zero_grad();pred=(x@models[arm])[...,None]
                    parts=predictable_autocov_loss(pred,target,correction_tensors[arm][regime[ids]],scale)
                    loss=parts['mse'] if arm=='mse' else parts['mse']+COEF*parts['structure']
                    if step==1:
                        gm=torch.autograd.grad(parts['mse'],pred,retain_graph=True)[0];gc=torch.autograd.grad(parts['structure'],pred,retain_graph=True)[0]*COEF
                        gradient_samples.append(dict(epoch=epoch,variant=arm,mse_norm=float(gm.norm()),weighted_structure_norm=float(gc.norm()),ratio=float(gc.norm()/gm.norm().clamp_min(1e-30)),cosine=float((gc*gm).sum()/(gc.norm()*gm.norm()).clamp_min(1e-30))))
                    assert torch.isfinite(loss);loss.backward();assert torch.isfinite(models[arm].grad).all();optimizers[arm].step()
                    totals[arm]+=np.array([parts['mse'].item(),parts['structure'].item(),loss.item()])*len(ids)
            for arm in ARMS:
                with torch.no_grad():prediction=(xf[8192:]@models[arm]).numpy()[...,None]
                met=metrics(prediction,y[8192:],mean[8192:]);curve=dict(epoch=epoch,variant=arm,train_mse=float(totals[arm][0]/4096),train_structure=float(totals[arm][1]/4096),train_loss=float(totals[arm][2]/4096),validation=met);curves.append(curve)
                if met['mse']<best[arm]:best[arm]=met['mse'];best_epoch[arm]=epoch;best_weights[arm]=models[arm].detach().clone()
            write(out/'status.json',dict(status='synthetic_training',scenario=scenario,epoch=epoch,max_epochs=40))
            if epoch%10==0:print(scenario,epoch,{a:round(best[a],6) for a in ARMS},flush=True)
        runs={}
        for arm in ARMS:
            with torch.no_grad():prediction=(xf[8192:]@best_weights[arm]).numpy()[...,None]
            runs[arm]=dict(selected_epoch=best_epoch[arm],updates=40*32,metrics=metrics(prediction,y[8192:],mean[8192:]))
            np.savez_compressed(folder/(arm+'_selected_validation.npz'),prediction=prediction,truth=y[8192:],conditional_mean=mean[8192:])
        torch.save(dict(initial=initial,best=best_weights,final={a:p.detach().clone() for a,p in models.items()},optimizers={a:z.state_dict() for a,z in optimizers.items()},best_epoch=best_epoch,seed=SEED),folder/'checkpoints.pt')
        record=dict(runs=runs,curves=curves,output_gradient_samples=gradient_samples,reference=reference,oracle_correction=oracle.tolist(),population_amplitude=amplitude,scale=scale.tolist())
        write(folder/'report.json',record);result[scenario]=record
    return result

def design(x,kind):
    mean=x.mean(1,keepdims=True);std=np.sqrt(x.var(1,keepdims=True)+1e-5);z=x-mean
    if kind=='block12':z=z.reshape(len(x),12,8,7).mean(2)
    return np.concatenate([z,std],axis=1).transpose(0,2,1),mean

def windows(data,start,end,step):
    ids=np.arange(start,end-2*H+1,step)
    return np.stack([data[i:i+H] for i in ids]),np.stack([data[i+H:i+2*H] for i in ids]),ids

def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-8))

def real_screen(out):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    old=json.loads((ROOT/'tmp/lag_study_timemix_pdm_innovation_20261006_seed2021_earlystop5/training/protocol.json').read_text())
    cfg=old['model_args'];datafile=Path(cfg['root_path'])/cfg['data_path']
    if not datafile.is_absolute():datafile=ROOT/datafile
    frame=pd.read_csv(datafile,nrows=34560);assert len(frame)==34560
    names=list(frame.columns[1:]);assert len(names)==7;raw=frame.iloc[:,1:].values;scaler=StandardScaler().fit(raw);data=scaler.transform(raw)
    directory=out/'ettm1_train_screen';directory.mkdir();fits={};stats={};saved={}
    xtrain,ytrain,ids=windows(data,0,17280,1);threshold=np.median(xtrain.std(1),axis=0)
    for kind in ['linear96','block12']:
        z,mu=design(xtrain,kind);target=(ytrain-mu).transpose(0,2,1)
        gram=np.einsum('nci,ncj->cij',z,z,optimize=True)/len(z);rhs=np.einsum('nci,nch->cih',z,target,optimize=True)/len(z)
        penalty=np.eye(z.shape[-1]);penalty[-1,-1]=0
        weights=np.linalg.solve(gram+.001*penalty[None],rhs);fits[kind]=weights;stats[kind]={}
        for split,start,end in [('calibration',17280,25920),('check',25920,34560)]:
            x,y,indices=windows(data,start,end,96);zz,mean=design(x,kind);pred=mean+np.einsum('nci,cih->nhc',zz,weights,optimize=True)
            residual=y-pred;cov=cov_np(residual);state=(x.std(1)>threshold).astype(int)
            stats[kind][split]=dict(mse=float(np.mean(residual**2)),per_channel_mse=np.mean(residual**2,axis=(0,1)).tolist(),cells={})
            for c,name in enumerate(names):
                for g in range(2):
                    mask=state[:,c]==g;key=f'{name}:{g}';count=int(mask.sum())
                    stats[kind][split]['cells'][key]=dict(count=count,covariance=cov[mask,c].mean(0).tolist() if count else None,mse=float(np.mean(residual[mask,:,c]**2)) if count else None)
            saved[kind+'_'+split]=dict(prediction=pred,truth=y,history=x,residual_covariance=cov,state=state,indices=indices)
            np.savez_compressed(directory/(kind+'_'+split+'.npz'),**saved[kind+'_'+split])
        np.savez_compressed(directory/(kind+'_weights.npz'),weights=weights)
    gates={};channel_ok={}
    for c in names:
        channel_ok[c]=False
        for g in range(2):
            key=f'{c}:{g}';a=stats['linear96']['calibration']['cells'][key];b=stats['linear96']['check']['cells'][key];other=stats['block12']['calibration']['cells'][key]
            countok=min(a['count'],b['count'],other['count'])>=20
            drift=relative(np.array(a['covariance']),np.array(b['covariance'])) if a['count'] and b['count'] else None
            sensitivity=relative(np.array(a['covariance']),np.array(other['covariance'])) if a['count'] and other['count'] else None
            ok=countok and drift<=.25 and sensitivity<=.25
            reasons=[]
            if not countok:reasons.append('fewer_than_20_windows')
            if drift is None or drift>.25:reasons.append('temporal_drift')
            if sensitivity is None or sensitivity>.25:reasons.append('reference_sensitivity')
            gates[key]=dict(passed=ok,calibration_count=a['count'],check_count=b['count'],relative_drift=drift,relative_reference_difference=sensitivity,reasons=reasons)
            channel_ok[c]|=ok
    record=dict(status='screen_passed' if all(channel_ok.values()) else 'screen_failed',all_channels_have_eligible_state=all(channel_ok.values()),channels=names,channel_eligible=channel_ok,gates=gates,references=stats,
                boundaries=dict(reference=[0,17280],calibration=[17280,25920],check=[25920,34560]),scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),history_std_threshold=threshold.tolist(),official_validation_accessed=False,test_accessed=False,rows_read=34560,reference_training_windows=len(xtrain),data_prefix_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),warning='Residual covariance includes reference bias. Passing necessary gates does not identify irreducible noise; no PDM effect claim.')
    write(directory/'report.json',record);return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);os.chdir(ROOT)
    sources=['utils/predictable_autocov_loss.py','tests/test_predictable_autocov_loss.py','scripts/run_predictable_autocov_study.py','docs/experiments/PredictableAutocov_protocol_20261007.md']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}
    for f in sources:
        q=out/'source'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/f).read_bytes())
    config=dict(seed=SEED,horizon=H,lags=LAGS,coefficient=COEF,arms=ARMS,epochs=40,batch_size=128,adam_lr=.01,training_rows=8192,validation_rows=2048,main_training_slice=[4096,8192],source_sha256=hashes,started_utc=datetime.now(timezone.utc).isoformat(),cuda_available=torch.cuda.is_available(),torch_version=torch.__version__)
    write(out/'protocol.json',config)
    old=json.loads((ROOT/'tmp/lag_study_timemix_pdm_innovation_20261006_seed2021_earlystop5/training/protocol.json').read_text())['source_sha256']
    assert len(old)==226 and all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in old.items())
    # Actual small CUDA optimizer smoke, in addition to CPU/CUDA loss gradient tests.
    if torch.cuda.is_available():
        torch.manual_seed(SEED);model=torch.nn.Linear(4,H).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        for _ in range(2):
            optimizer.zero_grad();x=torch.randn(8,4,device='cuda');y=torch.randn(8,H,1,device='cuda');pred=model(x)[...,None]
            parts=predictable_autocov_loss(pred,y,torch.zeros(1,len(LAGS),device='cuda'),torch.ones(1,device='cuda'));loss=parts['mse']+COEF*parts['structure'];loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in model.parameters());optimizer.step()
        del model,optimizer;torch.cuda.empty_cache()
    results=synthetic(out)
    write(out/'status.json',dict(status='train_only_feasibility_screen'))
    screening=real_screen(out)
    assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in old.items())
    report=dict(status='completed',synthetic=results,real_screen=screening,prior_frozen_226_sources_unchanged=True,model_normalization_modified=False,official_test_accessed=False,real_official_validation_accessed=False,formal_pdm_training_started=False,finished_utc=datetime.now(timezone.utc).isoformat())
    write(out/'report.json',report);write(out/'status.json',dict(status='completed',real_screen=screening['status']))
    print('COMPLETE',screening['status'],screening['channel_eligible'],flush=True)
if __name__=='__main__':main()
