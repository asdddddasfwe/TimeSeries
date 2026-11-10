"""Saved-validation-output gradients only; no forecast model or test forward."""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from tmp.mafirst_kernel_timeparallel_h336_20261022_common import G,PLAN,read,atomic,sha,now

def kernel_output_gradient(prediction,target,timed=True):
    """Exact derivative of each B,C sequence objective, without B*C averaging.
    Returns B,C objective; B,H,C gradient. sigmaValue1, raw4, /N².
    Autograd unit tests multiply mean-objective gradient by B*C for equality.
    """
    p=np.asarray(prediction,dtype=np.float64);y=np.asarray(target,dtype=np.float64)
    B,H,C=p.shape;assert y.shape==p.shape and H%4==0 and H>=8
    N=H//4;a=p.transpose(0,2,1).reshape(B,C,N,4);b=y.transpose(0,2,1).reshape(B,C,N,4)
    pp=a[:,:,:,None,:]-a[:,:,None,:,:];py=a[:,:,:,None,:]-b[:,:,None,:,:];yy=b[:,:,:,None,:]-b[:,:,None,:,:]
    kpp=np.exp(-.5*np.square(pp).sum(-1));kpy=np.exp(-.5*np.square(py).sum(-1));kyy=np.exp(-.5*np.square(yy).sum(-1))
    W=np.exp(-.5*((np.arange(N)[:,None]-np.arange(N)[None,:])/2.)**2) if timed else np.ones((N,N))
    raw=(W*(kpp+kyy-2*kpy)).mean((-1,-2));assert raw.min()>=-1e-10 and np.isfinite(raw).all()
    grad=(2/N**2)*((W*kpy)[...,None]*py-(W*kpp)[...,None]*pp).sum(-2)
    # Exact clamp derivative only matters at zero. Equality inputs have zero analytic gradient.
    grad=np.where((raw>0)[...,None,None],grad,0.)
    gradient=grad.reshape(B,C,H).transpose(0,2,1)
    dist=np.sqrt(np.square(py).sum(-1))
    stats={'cross_kernel_mean':kpy.mean((-1,-2)),'cross_kernel_fraction_below_1e6':(kpy<1e-6).mean((-1,-2)),
           'cross_distance_mean':dist.mean((-1,-2)),
           'cross_radial_derivative_mean':(dist*kpy).mean((-1,-2)),
           'time_weighted_cross_radial_derivative_mean':(W*dist*kpy).mean((-1,-2))}
    return np.maximum(raw,0),gradient,stats

def summarize(values,mask=None):
    result={}
    for k,a in values.items():
        z=np.asarray(a)[mask] if mask is not None else np.asarray(a).reshape(-1)
        finite=z[np.isfinite(z)];entry={'count':int(z.size),'finite_count':int(finite.size)}
        if finite.size:
            entry.update(mean=float(finite.mean()),median=float(np.median(finite)),p10=float(np.quantile(finite,.1)),p90=float(np.quantile(finite,.9)))
            if 'cosine' in k:entry['negative_fraction']=float((finite<0).mean())
        result[k]=entry
    return result

def diagnose(label,output):
    target=G/f'output_gradient_{label}.json'
    if target.exists():
        cached=read(target);assert cached['status']=='verified' and cached['sample_count']==512
        assert sha(R/output/'selected_val/prediction.npy')==cached['source_prediction_sha256']
        assert sha(R/output/'selected_val/truth.npy')==cached['source_truth_sha256']
        return cached
    folder=R/output/'selected_val';P=np.load(folder/'prediction.npy',mmap_mode='r');Y=np.load(folder/'truth.npy',mmap_mode='r')
    assert P.shape==Y.shape==(11185,336,7)
    indices=np.unique(np.linspace(0,len(P)-1,512,dtype=int));assert len(indices)==512
    # Full-validation target amplitudes define per-channel descriptive strata, no fitting/tuning.
    amps=np.concatenate([np.asarray(Y[j:j+64],dtype=float).std(1) for j in range(0,len(Y),64)])
    cuts=np.quantile(amps,[.25,.5,.75],axis=0);amplitude_bin=(amps[indices,:,None]>cuts.T[None,:,:]).sum(-1)
    bags={key:{} for key in ('time','value')};statsbags={key:{} for key in bags}
    for begin in range(0,len(indices),8):
        ix=indices[begin:begin+8];p=np.asarray(P[ix],dtype=float);y=np.asarray(Y[ix],dtype=float)
        gm=2*(p-y)/p.shape[1];center=p-p.mean(1,keepdims=True)
        mse=np.square(p-y).mean(1)
        for timed,key in [(True,'time'),(False,'value')]:
            obj,g,stats=kernel_output_gradient(p,y,timed)
            fields={'mse':mse,'kernel_objective':obj,'mse_amplitude_derivative':(gm*center).sum(1),'mmd_amplitude_derivative':(g*center).sum(1),'mse_mean_derivative':gm.sum(1),'mmd_mean_derivative':g.sum(1)}
            fields['amplitude_derivative_opposite_sign']=(fields['mse_amplitude_derivative']*fields['mmd_amplitude_derivative']<0).astype(float)
            for q,sl in [('all',slice(None))]+[(f'q{i+1}',slice(i*84,(i+1)*84)) for i in range(4)]:
                a=gm[:,sl];b=g[:,sl];an=np.sqrt(np.square(a).sum(1));bn=np.sqrt(np.square(b).sum(1));den=an*bn
                fields[q+'_mse_norm']=an;fields[q+'_mmd_norm']=bn
                fields[q+'_norm_ratio']=np.divide(bn,an,out=np.full_like(an,np.nan),where=an>1e-30)
                fields[q+'_cosine']=np.divide((a*b).sum(1),den,out=np.full_like(den,np.nan),where=den>1e-30)
            for n,v in fields.items():bags[key].setdefault(n,[]).append(v)
            for n,v in stats.items():statsbags[key].setdefault(n,[]).append(v)
        if begin%128==0:print(json.dumps({'diagnosis':label,'processed':begin+len(ix),'of':len(indices)}),flush=True)
    result={'status':'verified','label':label,'source_output':output,'split':'selected_val','sample_count':512,'total_validation_windows':11185,'selection':'uniform unique linspace indices, fixed before training; no model forward or test','indices':indices.tolist(),'source_prediction_sha256':sha(folder/'prediction.npy'),'source_truth_sha256':sha(folder/'truth.npy'),'per_channel_amplitude_quartile_thresholds':cuts.T.tolist(),'normalization':'gradient per individual H-length sequence, no batch/channel mean; both objectives same normalization','interpretation':'output-space gradient relationship only; not shared-parameter gradient or update-causality evidence; auxiliary Gram is not a function of saved final forecast','objective_definitions':{'time':'raw4 sigmaValue1 * W sigmaTime2, all three terms /84², biased MMD²','value':'same without W, no mass or norm compensation'},'kernels':{},'created':now()}
    arrays={'indices':indices,'amplitude_bins':amplitude_bin}
    channels=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
    for key in bags:
        v={n:np.concatenate(z) for n,z in bags[key].items()};v.update({n:np.concatenate(z) for n,z in statsbags[key].items()})
        result['kernels'][key]={'overall':summarize(v),'per_channel':{c:summarize({n:a[:,i] for n,a in v.items()}) for i,c in enumerate(channels)},'amplitude_quartiles':{str(i+1):summarize(v,amplitude_bin==i) for i in range(4)}}
        arrays.update({key+'_'+n:a for n,a in v.items()})
    np.savez_compressed(G/f'output_gradient_{label}.npz',**arrays);result['details_sha256']=sha(G/f'output_gradient_{label}.npz')
    atomic(target,result);return result

def render(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    labels=list(data);x=np.arange(len(labels));width=.35
    for j,key in enumerate(('time','value')):
        axes[0].bar(x+(j-.5)*width,[data[n]['kernels'][key]['overall']['all_norm_ratio']['median'] for n in labels],width,label=key)
        axes[1].bar(x+(j-.5)*width,[data[n]['kernels'][key]['overall']['all_cosine']['negative_fraction'] for n in labels],width,label=key)
    for ax in axes:ax.set_xticks(x,labels,rotation=20);ax.legend()
    axes[0].set(ylabel='Median output-gradient norm ratio MMD / MSE',yscale='log');axes[1].set(ylabel='Fraction cosine < 0',ylim=(0,1))
    fig.savefig(G/'output_gradient_diagnosis.png',dpi=150);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--include-new',action='store_true');args=ap.parse_args()
    plan=read(PLAN);c=plan['comparison'];sources={'kernel_time_recurrent':c['baseline_kernel_time_recurrent'],'kernel_value_recurrent':c['valueonly_recurrent'],'kernel_value_parallel':c['valueonly_parallel']}
    if args.include_new:sources['kernel_time_parallel']=plan['jobs'][0]['output']
    data={label:diagnose(label,path) for label,path in sources.items()};render(data)
    result={'status':'verified' if args.include_new else 'old_validation_diagnosis_verified','old_models':3,'new_model_included':args.include_new,'only_saved_validation_arrays':True,'no_forecast_model_forward':True,'sampling_frozen':True,'test_used':False,'runs':data,'plot_sha256':sha(G/'output_gradient_diagnosis.png'),'created':now()}
    atomic(G/'output_gradient_diagnosis.json',result)
    print(json.dumps({'status':result['status'],'runs':list(data),'plot':str(G/'output_gradient_diagnosis.png')}),flush=True)
if __name__=='__main__':main()
