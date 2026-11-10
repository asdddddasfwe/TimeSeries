"""Streaming NumPy audits for variable forecast horizons; no torch/model imports."""
import numpy as np

def metrics(p,y,chunk=64):
    count=len(p);totals={};H=p.shape[1];basis=np.cos(np.pi/H*(np.arange(H)[:,None]+.5)*np.arange(4)[None])*np.sqrt(2/H);basis[:,0]/=np.sqrt(2)
    boundary=(np.arange(H-1)+1)%4==0
    for start in range(0,count,chunk):
        a,b=[np.asarray(v[start:start+chunk],dtype=np.float64) for v in (p,y)]
        assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all();e=a-b;low=np.einsum('hk,nkc->nhc',basis,np.einsum('hk,nhc->nkc',basis,e));level=float(np.mean(e.mean(1)**2));amp=float(np.mean((a.std(1)-b.std(1))**2));mse=float(np.mean(e**2))
        row=dict(mse=mse,mae=float(np.mean(abs(e))),low4_mse=float(np.mean(low**2)),remaining_mse=float(np.mean((e-low)**2)),window_offset_mse=level,amplitude_error_mse=amp,shape_alignment_mse=mse-level-amp)
        assert abs(mse-row['low4_mse']-row['remaining_mse'])<2e-12
        da,db=np.diff(a,axis=1),np.diff(b,axis=1)
        for label,mask in [('boundary',boundary),('interior',~boundary)]:
            for k,v in [('error_mse',float(np.mean((da[:,mask]-db[:,mask])**2))),('prediction_energy',float(np.mean(da[:,mask]**2))),('truth_energy',float(np.mean(db[:,mask]**2)))]:row[label+'_'+k]=v
        for k,v in row.items():totals[k]=totals.get(k,0.)+v*len(a)
    out={k:v/count for k,v in totals.items()};out['remaining_mode_count']=H-4;out['increments']={label:dict(count=int(mask.sum()),**{k:out.pop(label+'_'+k) for k in ('error_mse','prediction_energy','truth_energy')}) for label,mask in [('boundary',boundary),('interior',~boundary)]};return out

def numpy_relations(p,y,chunk=16):
    assert p.shape==y.shape and p.ndim==3 and p.shape[1]%4==0
    N=p.shape[1]//4;time=np.exp(-(np.arange(N)[:,None]-np.arange(N)[None,:])**2/8.);total=np.zeros(8);count=0
    for start in range(0,len(p),chunk):
        a,b=[np.asarray(v[start:start+chunk],dtype=np.float64).transpose(0,2,1).reshape(-1,p.shape[2],N,4) for v in (p,y)]
        assert np.isfinite(a).all() and np.isfinite(b).all()
        def gram(x,z):
            sq=np.zeros((*x.shape[:-2],N,N),dtype=np.float64)
            for j in range(4):sq+=(x[...,j,None]-z[...,None,:,j])**2
            return np.exp(-sq/2.)
        aa,bb,ab=gram(a,a),gram(b,b),gram(a,b)
        mp,my,mc=[(time*g).mean((-1,-2)) for g in (aa,bb,ab)];raw=mp+my-2*mc;assert raw.min()>=-1e-10
        def center(g):return g-g.mean(-1,keepdims=True)-g.mean(-2,keepdims=True)+g.mean((-1,-2),keepdims=True)
        dp,dy,dc=[np.mean(center(g)**2,axis=(-1,-2)) for g in (aa,bb,ab)];dr=dp+dy-2*dc;assert dr.min()>=-1e-10
        total+=np.array([np.maximum(raw,0).mean(),mp.mean(),my.mean(),mc.mean(),np.maximum(dr,0).mean(),dp.mean(),dy.mean(),dc.mean()])*len(a);count+=len(a)
    return dict(zip(('time_position_mmd','mmd_prediction_self','mmd_target_self','mmd_cross','patch_covariance','prediction_covariance_norm2','target_covariance_norm2','cross_covariance_inner_product'),map(float,total/count)))
