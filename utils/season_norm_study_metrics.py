"""Offline, pure-network shape metrics; no model or dataset access."""
import numpy as np


def shape_metrics(prediction, truth):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 3 or p.shape[1] != 96:
        raise ValueError("Expected matching [N,96,C] arrays")
    pc, yc = p-p.mean(1, keepdims=True), y-y.mean(1, keepdims=True)
    pv, yv = np.mean(pc**2, 1), np.mean(yc**2, 1)
    valid = yv >= .01
    acf = {}
    for lag in [1, 2, 4, 8, 12, 24]:
        pa = np.mean(pc[:, lag:]*pc[:, :-lag], 1)*(96-lag)/96/np.maximum(pv, 1e-12)
        ya = np.mean(yc[:, lag:]*yc[:, :-lag], 1)*(96-lag)/96/np.maximum(yv, 1e-12)
        err = (pa-ya)**2
        acf[str(lag)] = dict(mse=float(err[valid].mean()) if valid.any() else None,
            per_channel_mse=[float(err[:, j][valid[:, j]].mean()) if valid[:, j].any() else None for j in range(p.shape[2])])
    # A descriptive alignment diagnostic, never used for epoch selection.
    # 5-point VALID moving average, positive lag means prediction is delayed.
    ps = sum(p[:, i:i+92] for i in range(5))/5
    ys = sum(y[:, i:i+92] for i in range(5))/5
    lags = np.arange(-12, 13);corrs=[]
    for lag in lags:
        a, b = (ps[:, lag:], ys[:, :92-lag]) if lag > 0 else ((ps[:, :92+lag], ys[:, -lag:]) if lag < 0 else (ps, ys))
        a, b = a-a.mean(1,keepdims=True), b-b.mean(1,keepdims=True)
        corrs.append(np.sum(a*b,1)/np.maximum(np.sqrt(np.sum(a*a,1)*np.sum(b*b,1)),1e-12))
    corr=np.stack(corrs,-1);best=np.argmax(corr,-1);lag=lags[best]
    ok=valid & (pv>=.01);peak=np.take_along_axis(corr,best[...,None],-1)[...,0]
    robust=ok & (peak>=.5) & (peak-corr[...,12]>=.05) & (np.abs(lag)>=2)
    return dict(definition='Biased future ACF, truth variance>=.01; degenerate prediction ACF=0. Smooth5 valid correlation lag[-12,12], positive=prediction delayed; robust requires both variances>=.01, peak>=.5, gain>=.05, abs(lag)>=2. Descriptive only.',
        valid_acf_pairs=int(valid.sum()),future_acf=acf,
        phase=dict(eligible=int(ok.sum()),robust_shift_fraction=float(robust.sum()/max(ok.sum(),1)),
            mean_abs_best_lag=float(np.abs(lag[ok]).mean()) if ok.any() else None,
            median_best_lag=float(np.median(lag[ok])) if ok.any() else None,
            centered_smoothed_correlation=float(corr[...,12][ok].mean()) if ok.any() else None))


def factorial_contrasts(rows, split):
    by={r['variant']:r for r in rows}
    def metric(name,key):
        return by[name]['selected_validation' if split=='val' else 'test']['metrics'][key] if split!='val' else by[name]['selected_validation'][key]
    out={}
    for key in ['mse','mae','low4_mse','window_offset_mse']:
        a,b,c,d=[metric(n,key) for n in ['control','original_kl','season_norm_mse','season_norm_kl']]
        out[key]=dict(original_mse=a,original_kl=b,normalized_mse=c,normalized_kl=d,
            normalization_effect_mse=c-a,normalization_effect_mse_pct=100*(c-a)/a,
            normalization_effect_kl=d-b,normalization_effect_kl_pct=100*(d-b)/b,
            original_kl_penalty=b-a,normalized_kl_penalty=d-c,
            interaction=(d-c)-(b-a),interpretation='negative interaction means KL penalty reduced; does not alone mean absolute forecasts improve')
    return out
