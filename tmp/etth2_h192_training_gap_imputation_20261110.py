"""Training-only reconstruction for the registered ETTh2 rule flags."""
import numpy as np


def runs(active):
    edges = np.flatnonzero(np.diff(np.r_[False, active, False]))
    return zip(edges[::2], edges[1::2])


def hourly_medians(values, mask, hours):
    clean = np.where(mask, np.nan, values)
    donors = np.array([np.nanmedian(clean[hours == hour], axis=0) for hour in range(24)])
    assert np.isfinite(donors).all()
    return donors[hours]


def cross_channel(values, mask, hours, output):
    """Regress on the channels observed at that hour and align to recent clean history."""
    width = values.shape[1]
    harmonics = np.column_stack([f(2*np.pi*hours*k/24) for k in (1,2) for f in (np.sin,np.cos)])
    partial = (mask.sum(1) > 0) & (mask.sum(1) < width)
    patterns = np.unique(mask[partial], axis=0)
    for pattern in patterns:
        rows = np.flatnonzero(np.all(mask == pattern, axis=1))
        available = ~pattern
        usable = ~mask[:, available].any(1)
        covariates = np.column_stack((values[:, available], harmonics))
        for channel in np.flatnonzero(pattern):
            observed = usable & ~mask[:, channel]
            features = covariates[observed]
            assert len(features) >= 128
            center = features.mean(0)
            scale = features.std(0)
            scale[scale == 0] = 1
            matrix = (features - center) / scale
            targets = values[observed, channel]
            level = targets.mean()
            weights = np.linalg.solve(matrix.T @ matrix + .01*len(matrix)*np.eye(matrix.shape[1]),
                                      matrix.T @ (targets - level))
            prediction = level + ((covariates-center)/scale) @ weights
            for start, stop in runs(mask[:, channel]):
                positions = rows[(rows >= start) & (rows < stop)]
                if not len(positions):
                    continue
                anchor = np.flatnonzero(observed[max(0,start-96):start])[-24:] + max(0,start-96)
                offset = np.median(values[anchor,channel]-prediction[anchor]) if len(anchor) else 0
                adjusted = prediction[positions] + offset * np.exp(-(positions-start)/24)
                output[positions,channel] = np.clip(adjusted, targets.min(), targets.max())


def similar_days(values, mask, hours, output, feature_scale):
    """Choose clean historical day trajectories by their preceding 24-hour context."""
    width = values.shape[1]
    n = len(values)
    candidate_starts = np.flatnonzero(hours == 0)
    clean = ~mask.any(1)
    donors = np.array([i for i in candidate_starts if 24<=i<=n-24 and clean[i-24:i+24].all()])
    # Keep the fallback hourly estimates if no earlier comparable context exists.
    limits_min = np.nanmin(np.where(mask,np.nan,values),axis=0)
    limits_max = np.nanmax(np.where(mask,np.nan,values),axis=0)
    rows_matched = np.zeros(n,dtype=bool)
    for start, stop in runs(mask.all(1)):
        for i in range(start,stop,24):
            length = min(24,stop-i)
            earlier = donors[donors+24<=i]
            if i<24 or not len(earlier):
                continue
            idx = earlier[:,None] + np.arange(-24,0)
            observed = ~mask[i-24:i]
            if observed.sum()<48:
                continue
            query = values[i-24:i]
            difference = (query[None] - values[idx]) / feature_scale
            distance = np.square(difference[:,observed]).mean(1)
            nearest = np.argsort(distance,kind="stable")[:5]
            weights = 1 / np.maximum(np.sqrt(distance[nearest]),1e-3)
            weights /= weights.sum()
            candidates = values[earlier[nearest,None]+np.arange(length)].copy()
            for ch in range(width):
                use = np.flatnonzero(observed[:,ch])[-6:]
                if len(use):
                    shift = np.median(query[use,ch]-values[idx[nearest][:,use],ch],axis=1)
                    candidates[:,:,ch] += shift[:,None]
            estimate = np.sum(candidates*weights[:,None,None],axis=0)
            output[i:i+length] = np.clip(estimate,limits_min,limits_max)
            rows_matched[i:i+length] = True
    return rows_matched


def alternatives(values, mask, hours):
    median = hourly_medians(values,mask,hours)
    output = np.where(mask,median,values)
    cross_channel(values,mask,hours,output)
    scale = np.nanstd(np.where(mask,np.nan,values),axis=0)
    assert np.isfinite(scale).all() and (scale>0).all()
    analog_rows = similar_days(values,mask,hours,output,scale)
    assert np.array_equal(output[~mask],values[~mask]) and np.isfinite(output).all()
    return median,output,scale,analog_rows


def evaluate_and_fill(values, mask, hours):
    """Validate methods by hiding observed training segments, then refit on the training prefix."""
    candidates = {}
    for length in (24,48):
        candidates[length] = [i for i in np.flatnonzero(hours==0)
                              if i>=14*24 and i+length<=len(values) and not mask[i-24:i+length].any()]
        assert len(candidates[length]) >= 12
    scenarios = [("partial",24,1,6),("partial",48,2,6),("all_channels",24,7,6)]
    tests = []
    summaries = {}
    for route,length,width,count in scenarios:
        # Cover the available train chronology and select reproducible channels.
        positions = np.linspace(0,len(candidates[length])-1,count,dtype=int)
        errors = {method:[] for method in ("hourly_median","model")}
        for case,pos in enumerate(positions):
            start = candidates[length][pos]
            channels = [(3*case+j+length//24)%7 for j in range(width)]
            injected = np.zeros_like(mask)
            injected[start:start+length,channels] = True
            assert not (mask & injected).any()
            damaged = mask | injected
            m,r,scale,matched = alternatives(values,damaged,hours)
            if route=="all_channels":
                assert matched[start:start+length].all()
            tests.append(dict(route=route,start=int(start),hours=length,channels=channels))
            for key,estimate in (("hourly_median",m),("model",r)):
                error=(estimate-values)/scale
                errors[key].extend(error[injected].tolist())
        key=f"{route}_{length}h_{width}ch"
        summaries[key] = dict(route=route,scenarios=count,elements=len(errors["model"]),metrics={
            k:dict(mae=float(np.mean(np.abs(e))),mse=float(np.mean(np.square(e)))) for k,e in errors.items()})
    decisions={}
    for route in ("partial","all_channels"):
        relevant=[x for x in summaries.values() if x["route"]==route]
        averages={m:float(np.average([x["metrics"][m]["mae"] for x in relevant],weights=[x["elements"] for x in relevant]))
                  for m in ("hourly_median","model")}
        decisions[route] = dict(weighted_mae=averages,selected=min(averages,key=averages.get))
    median,model,scale,analog_rows=alternatives(values,mask,hours)
    output=values.copy()
    for route,rows in (("partial",~mask.all(1)),("all_channels",mask.all(1))):
        use=rows[:,None] & mask
        estimate=model if decisions[route]["selected"]=="model" else median
        output[use]=estimate[use]
    assert np.array_equal(output[~mask],values[~mask]) and np.isfinite(output).all()
    return output,dict(validation_scope="masked originally valid training observations only",cases=tests,
                       synthetic_metrics=summaries,decisions=decisions,all_channel_model_matched_hours=int(analog_rows.sum()))
