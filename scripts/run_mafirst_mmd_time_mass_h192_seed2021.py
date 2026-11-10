"""Original MAFirst+MMD seed2021; max30, patience5, cosine horizon30."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import copy
import fcntl
import html
import json
import math
from pathlib import Path
import shutil
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from scripts.train_timemix_pdm_state_patch import initialize_pair as old_initialize_pair
from torch.utils.data import DataLoader, Subset
from utils.pdm_state_patch_ma_linear_diagnostics import FeatureStatistics, gradient_probe
from models.OscillatorPDMStatePatchSeparateMLP import Model as SeparateMLP
from models.OscillatorPDMStatePatchMAFirst import Model as PositionOnly
from models.OscillatorPDMStatePatchMAFirstSeparateLinear import Model as SeparateLinear
from models.OscillatorPDMStatePatchPreDecompSliding import Model as Sliding
from scripts import run_timemix_dual_encoder_experiment as shared
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts
from utils.time_position_patch_mmd import parts as mmd_parts
from utils.mafirst_mmd_horizon_audit import metrics as saved_metrics, numpy_relations

ARMS = ('time_position_mmd',)
CLASSES = dict(time_position_mmd=PositionOnly)
COUNTS = {}  # Filled from the frozen protocol; optimizer names audit every tensor.
MODEL_NAMES = {a: cls.__module__.split('.')[-1] for a, cls in CLASSES.items()}
from utils.time_position_mmd_scale import mmd_weight, time_kernel_mass
SEED = 2021
HORIZON = 192
MMD_WEIGHT = mmd_weight(HORIZON)
TRAIN_WINDOWS = 34273
VAL_WINDOWS = 11329
STATE_KEY = 'pdm_state_patch_mmd_time_mass_h192_seed2021_results'
PROTOCOL = ROOT/'docs/experiments/PDMStatePatchMMDTimeMassH192Seed2021_protocol_20261016.json'
REPORT = ROOT/'docs/experiments/PDMStatePatchMMDTimeMassH192Seed2021_results_20261016.html'
# Reuse implementation utilities, with this study's own snapshot protocol and
# registration namespace. None of the completed study's files are edited.
shared.PROTOCOL, shared.STATE_KEY = PROTOCOL, STATE_KEY
if PROTOCOL.exists():
    COUNTS.update(json.loads(PROTOCOL.read_text())['parameters'])


def arm_config(cfg, arm):
    return SimpleNamespace(**{**vars(cfg), 'model': MODEL_NAMES[arm]})


def scratch_reference(cfg):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        original = old_initialize_pair(cfg)['patch_covariance']
        with torch.random.fork_rng(devices=[]):
            source = SeparateMLP(cfg)
        state = source.state_dict()
        state.update({n: v.clone() for n, v in original.state_dict().items()})
        source.load_state_dict(state, strict=True)
        source.clone_trend_mlps()
        return source


def initialize_pair(cfg):
    source = scratch_reference(cfg)
    result = {}
    for arm in ARMS:
        with torch.random.fork_rng(devices=[]):
            # Deterministic new attention parameters, isolated from old scratch.
            torch.manual_seed(20211014)
            model = CLASSES[arm](arm_config(cfg, arm))
        state = model.state_dict()
        state.update({n: v.clone() for n, v in source.state_dict().items()
                      if n in state and state[n].shape == v.shape})
        model.load_state_dict(state, strict=True)
        if hasattr(model,'clone_embeddings'):
            model.clone_embeddings(source)
        if hasattr(model, 'clone_new_heads'):
            model.clone_new_heads()
        result[arm] = model
    return result


def control_reference(cfg):
    source=scratch_reference(cfg)
    with torch.random.fork_rng(devices=[]):
        model=PositionOnly(cfg)
    model.load_state_dict(source.state_dict(),strict=True)
    return model


def initialization_audit(models):
    model=models['time_position_mmd'];device=next(model.parameters()).device
    source=control_reference(config(ROOT/'dataset/ETT-small')).to(device);base=source.state_dict();state=model.state_dict()
    assert set(base)==set(state) and len(state)==236
    assert all(torch.equal(v,state[n]) and v.data_ptr()!=state[n].data_ptr() for n,v in base.items())
    assert sum(p.numel() for p in model.parameters())==COUNTS['time_position_mmd'] and len(list(model.parameters()))==204
    result=dict(parameter_counts=COUNTS,states=236,parameter_tensors=204,buffers=32,fresh_seed=2021,horizon=HORIZON,fresh236_regeneration_exact=True,fresh_independent_storage=True,old_trained_weights_loaded=False,initialization_rule_unchanged=True,horizon_specific_shapes_and_RNG_consumption=True)
    x=torch.randn(2,96,7,generator=torch.Generator().manual_seed(101)).to(device)
    for training in (False,True):
        outputs=[];rng=[]
        for m in (source,model):
            m.train(training)
            with torch.random.fork_rng(devices=[device.index or 0] if x.is_cuda else []):
                shared.seed_all(314)
                with torch.no_grad():outputs.append(m.predict_components(x))
                rng.append(torch.cuda.get_rng_state(device) if x.is_cuda else torch.get_rng_state())
        assert outputs[0].shape==(2,HORIZON,7,8) and torch.equal(outputs[0],outputs[1]) and torch.equal(rng[0],rng[1])
        result['train' if training else 'eval']=dict(component_max_difference=0.,rng_equal=True)
    return result


OLD = ROOT/'tmp/lag_study_timemix_pdm_state_patch_20261011_seed2021_earlystop5'
MA_OLD = ROOT/'tmp/lag_study_timemix_pdm_state_patch_ma_decoupled_20261014_seed2021_earlystop5'
MLP_OLD = ROOT/'tmp/lag_study_timemix_pdm_state_patch_separate_mlp_20261014_seed2021_earlystop5'


def config(root_path):
    values = dict(shared.read(OLD/'training/protocol.json')['model_args'])
    values.update(root_path=str(root_path), model=MODEL_NAMES['time_position_mmd'],
                  log_dir=str(Path(root_path).parent/'logs'), tensorboard=False, seed=SEED, seq_len=96, pred_len=HORIZON, label_len=0)
    return SimpleNamespace(**values)


def loader(data, epoch=None):
    if epoch is not None:
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(SEED*1000+epoch)).tolist()
        data = Subset(data, order)
    return DataLoader(data, batch_size=32, shuffle=False, num_workers=0, drop_last=False)



def dataset(cfg,split,limit=0):
    from data_provider.data_loader import Dataset_ETT_minute
    data=Dataset_ETT_minute(cfg,cfg.root_path,flag=split,size=[96,0,HORIZON],features='M',data_path='ETTm1.csv',timeenc=1,freq='h')
    expected=TRAIN_WINDOWS if split=='train' else VAL_WINDOWS;assert len(data)==expected,(split,len(data),expected)
    if limit:data=Subset(data,np.unique(np.linspace(0,len(data)-1,min(limit,len(data)),dtype=int)).tolist())
    return data


shared.dataset = dataset
shared.config = config
shared.loader = loader


def evaluate(model, data, device, folder, label, capture=False):
    model.eval()
    statistics = FeatureStatistics(model) if capture else None
    sums = dict(mse=0., mae=0., patch_covariance=0., prediction_covariance_norm2=0.,
                target_covariance_norm2=0., cross_covariance_inner_product=0., time_position_mmd=0., mmd_prediction_self=0., mmd_target_self=0., mmd_cross=0.)
    squared = absolute = count = 0
    arrays = {}
    if capture:
        destination = folder/label
        destination.mkdir(exist_ok=False)
        for key, tail in [('prediction', (HORIZON, 7)), ('truth', (HORIZON, 7)), ('history', (96, 7)),
                          ('components', (HORIZON, 7, 8)), ('history_mean', (1, 7))]:
            arrays[key] = np.lib.format.open_memmap(destination/(key+'.npy'), mode='w+',
                dtype='float32', shape=(len(data), *tail))
    offset = 0
    try:
        with torch.no_grad():
            for batch_index, (x, target, _, _) in enumerate(loader(data)):
                x, target = x.float().to(device), target[:, -HORIZON:].float().to(device)
                components = model.predict_components(x)
                norm = model.normalize
                assert not norm.affine and not norm.non_norm and not norm.subtract_last
                prediction = norm(components[..., :4].sum(-1)+components[..., 4:].sum(-1), 'denorm')
                if not torch.isfinite(prediction).all():
                    raise RuntimeError('Nonfinite prediction')
                _, diagnostic = parts(prediction, target)
                _, mmd_diagnostic = mmd_parts(prediction,target)
                diagnostic.update(mmd_diagnostic)
                if diagnostic['minimum_unclipped_distance'] < -1e-10:
                    raise RuntimeError('Negative kernel distance')
                n = len(x)
                error = prediction.double()-target.double()
                squared += float(error.square().sum()); absolute += float(error.abs().sum()); count += target.numel()
                for key in sums.keys()-{'mse', 'mae'}:
                    sums[key] += diagnostic[key]*n
                if capture:
                    values = dict(prediction=prediction, truth=target, history=x,
                        components=components*norm.stdev.unsqueeze(-1), history_mean=norm.mean)
                    for key, value in values.items():
                        arrays[key][offset:offset+n] = value.cpu().numpy()
                offset += n
                if batch_index % 50 == 0:
                    shared.phase(folder, label, windows=offset, total_windows=len(data))
    finally:
        if statistics:
            statistics.close()
    for array in arrays.values():
        array.flush()
    result = {key: value/offset for key, value in sums.items()}
    result.update(mse=squared/count, mae=absolute/count)
    if statistics:
        shared.atomic_json(folder/(label+'_features.json'), statistics.summary())
    return result


def verify_policy(last, train_windows, maximum):
    best = {a: math.inf for a in ARMS}; chosen = {a: 0 for a in ARMS}; stale = 0
    for epoch in range(1, last['epoch']+1):
        improved = False
        rows = [row for row in last['curves'] if row['epoch'] == epoch]
        assert {row['arm'] for row in rows} == set(ARMS)
        for row in rows:
            assert row['learning_rate'] == shared.learning_rate(epoch)
            assert abs(row['train_loss']-row['train_mse']-MMD_WEIGHT*row['train_mmd']) < 2e-6
            arm = row['arm']
            if row['val']['mse'] < best[arm]:
                best[arm], chosen[arm], improved = row['val']['mse'], epoch, True
        stale = 0 if improved else stale+1
    assert best == last['best'] and chosen == last['best_epoch'] and stale == last['stale']
    assert 1 <= last['epoch'] <= maximum
    assert last['epoch'] == maximum or stale == 5
    running_best = math.inf; running_stale = 0
    for row in last['curves'][:-1]:
        if row['val']['mse'] < running_best:
            running_best, running_stale = row['val']['mse'], 0
        else:
            running_stale += 1
        assert running_stale < 5, 'Training continued past patience5'
    updates = last['epoch']*math.ceil(train_windows/32)
    assert last['updates_per_arm'] == updates
    for arm in ARMS:
        assert all(int(v['step']) == updates for v in last['optimizers'][arm]['state'].values())
    return dict(epochs=last['epoch'], selected_epochs=chosen, single_stale_diagnostic=stale, updates_per_arm=updates, stopping='max30, strict improvement, patience5; cosine horizon30 unchanged')


def references():
    result = {}
    for rel in ('report.json', 'completion_audit.json', 'offline_comparison.json',
                'training/protocol.json', 'training/2021/initial.pth', 'training/2021/optimizer_groups.json',
                'patch_covariance_artifact/checkpoint.pth', 'patch_covariance_artifact/manifest.json',
                'test/patch_covariance/report.json'):
        path = OLD/rel
        result[str(path.relative_to(ROOT))] = shared.sha(path)
    for rel in ('initial.pt','optimizer_groups.json','report.json','completion_audit.json','sources_manifest.json',
                'separate_mlp/artifact/checkpoint.pt','separate_mlp/artifact/model_config.json','separate_mlp/test_receipt.json'):
        path = MLP_OLD/rel
        result[str(path.relative_to(ROOT))] = shared.sha(path)
    assert shared.read(MLP_OLD/'status.json')['status'] == 'completed_verified'
    assert shared.read(OLD/'status.json')['status'] == 'completed_verified'
    for rel in ('initial.pt','optimizer_groups.json','completion_audit.json','sources_manifest.json','offline_final_review.json','offline_ma_review.json','position_only/artifact/checkpoint.pt','position_only/test_receipt.json','fully_decoupled/artifact/checkpoint.pt','fully_decoupled/test_receipt.json'):
        path=MA_OLD/rel;result[str(path.relative_to(ROOT))]=shared.sha(path)
    return result


def mmd_gradient_probe(model,mse,mmd):
    raw=gradient_probe(model,mse,mmd)
    return {k:dict(parameter_tensors=v['parameter_tensors'],mse_norm=v['mse_norm'],mmd_norm=v['d_norm'],mmd_mse_ratio=v['weighted_ratio'],mmd_mse_cosine=v['cosine']) for k,v in raw.items()}


def output_gradient_probe(prediction,mse,mmd,old_distance):
    gradients=[torch.autograd.grad(loss,prediction,retain_graph=True)[0].detach().double() for loss in (mse,mmd,old_distance)]
    h=prediction.shape[1];t=torch.arange(h,device=prediction.device,dtype=torch.float64);k=torch.arange(4,device=prediction.device,dtype=torch.float64)
    basis=torch.cos(torch.pi/h*(t[:,None]+.5)*k[None,:]);basis[:,0]*=(1/h)**.5;basis[:,1:] *= (2/h)**.5
    def project(g):return torch.einsum('bkc,tk->btc',torch.einsum('btc,tk->bkc',g,basis),basis)
    result={}
    for label,values in [('all',gradients),('low4',[project(g) for g in gradients]),('remaining'+str(HORIZON-4),[g-project(g) for g in gradients]),('level',[g.mean(1,keepdim=True).expand_as(g) for g in gradients])]:
        a,b,c=[v.flatten() for v in values];na,nb,nc=[float(v.norm()) for v in (a,b,c)]
        result[label]=dict(mse_norm=na,mmd_norm=nb,old_d_norm=nc,mmd_mse_ratio=nb/max(na,1e-30),mmd_mse_cosine=float(a@b)/max(na*nb,1e-30),old_d_mse_ratio=nc/max(na,1e-30),old_d_mse_cosine=float(a@c)/max(na*nc,1e-30))
    return result


def audit_capture(folder,label,measured):
    p,y=[np.load(folder/label/(n+'.npy'),mmap_mode='r') for n in ('prediction','truth')];c=np.load(folder/label/'components.npy',mmap_mode='r');mean=np.load(folder/label/'history_mean.npy',mmap_mode='r');history=np.load(folder/label/'history.npy',mmap_mode='r');assert p.shape==y.shape==(len(p),HORIZON,7) and c.shape==(len(p),HORIZON,7,8) and history.shape==(len(p),96,7)
    maximum=0.;group32max=0.
    for i in range(0,len(p),32):
        cc=np.asarray(c[i:i+32]);mm=np.asarray(mean[i:i+32]);assert np.isfinite(cc).all() and np.isfinite(mm).all()
        maximum=max(maximum,float(np.max(abs(cc.astype(float).sum(-1)+mm.astype(float)-p[i:i+32]))));rec=cc[...,:4].sum(-1,dtype=np.float32)+cc[...,4:].sum(-1,dtype=np.float32)+mm;group32max=max(group32max,float(np.max(abs(rec-p[i:i+32]))))
    assert maximum<2e-5 and group32max<2e-5
    relation=numpy_relations(p,y);metric=saved_metrics(p,y)
    assert all(abs(v-measured[k])<1e-6 for k,v in relation.items())
    assert all(abs(metric[k]-measured[k])<1e-7 for k in ('mse','mae'))
    result=dict(status='verified',windows=len(p),history_length=96,prediction_length=HORIZON,metrics=metric,covariance={k:v for k,v in relation.items() if 'mmd' not in k},time_position_mmd={k:v for k,v in relation.items() if 'mmd' in k},reconstruction_max_abs=maximum,fp32_group_reconstruction_maxabs=group32max,objective=f'MSE+{MMD_WEIGHT}*raw_time_position_mmd_squared; oldD diagnostic0',files={f.name:shared.sha(f) for f in (folder/label).glob('*.npy')})
    shared.atomic_json(folder/label/'report.json',result);return result


def publish(out,report):
    page='<!doctype html><meta charset="utf-8"><h1>MAFirst＋时间位置MMD：ETTm1 96→192 seed2021 时间核质量补偿</h1><p>原初始化规则/架构族、batch32/lr.001/cosine30/max30/patience5；自身完整val选轮后只本组一次test。H变化使参数尺寸、窗口数、MMD patch数变化；公式仍每样本每变量/N²、sigmaTime2patch、lambda=1.9327232596390695，以96步时间核质量为锚补偿。结果等待完整离线核验。</p><pre>'+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre>'
    with (ROOT/'tmp/pdm_state_patch_rope_registration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);t=REPORT.with_suffix('.html.tmp');t.write_text(page);t.replace(REPORT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.4)
    parser.add_argument('--stop-after-epochs', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.epochs <= 30 or min(args.train_limit, args.val_limit) < 0 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error('Invalid experiment budget')
    smoke = bool(args.train_limit or args.val_limit)
    if smoke and not (args.train_limit >= 4 and args.val_limit >= 4):
        parser.error('Preflight requires both limits >=4')
    if not smoke and args.epochs != 30:
        parser.error('Formal experiment uses max30 epochs, patience5, cosine horizon30')
    if args.resume and not smoke:
        parser.error('No automatic formal resume')
    out = args.output.resolve()
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    lock = (out/'pipeline.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    shared.atomic_json(out/'process.json', shared.process_record())
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    torch.set_num_threads(2)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    shared.seed_all(SEED)
    try:
        if not args.resume:
            shared.phase(out, 'preparing')
            prefix = out/'train_val_data'; prefix.mkdir()
            with (ROOT/'dataset/ETT-small/ETTm1.csv').open('rb') as src, (prefix/'ETTm1.csv').open('wb') as dst:
                for _ in range(46081):
                    line = src.readline(); assert line; dst.write(line)
            cfg = shared.config(prefix)
            shared.atomic_json(out/'data_manifest.json', dict(train_val_csv_sha256=shared.sha(prefix/'ETTm1.csv'),
                full_csv_sha256=shared.sha(ROOT/'dataset/ETT-small/ETTm1.csv'), scaler_fit_rows=34560, train_val_rows=46080, history_length=96, prediction_length=HORIZON, train_windows=TRAIN_WINDOWS, val_test_windows=VAL_WINDOWS))
            shared.atomic_json(out/'reference_evidence.json', references())
            shared.atomic_json(out/'protocol.json', dict(model_config=vars(cfg), options={**vars(args), 'output': str(out)},
                formal_protocol=shared.read(PROTOCOL), smoke=smoke, started=shared.now()))
            manifest = shared.snapshot_sources(out)
            shutil.copyfile(ROOT/'scripts/monitor_mafirst_mmd_time_mass_h192_seed2021.py', out/'monitor_check.py')
            for arm in ARMS:
                (out/arm/'epochs').mkdir(parents=True)
            if not smoke:
                shared.register(out, 'running', **shared.process_record(), source_count=len(manifest),
                    protocol=str(PROTOCOL), parameter_counts=COUNTS, arms=list(ARMS), launch_log=str(out)+'_launch.log')
        else:
            saved = shared.read(out/'protocol.json')
            assert saved['smoke'] and shared.read(out/'status.json')['status'] == 'controlled_stop'
            for key in ('epochs', 'train_limit', 'val_limit', 'device'):
                assert getattr(args, key) == saved['options'][key]
            cfg = SimpleNamespace(**saved['model_config'])
        shared.verify_sources(out)
        assert shared.sha(Path(cfg.root_path)/'ETTm1.csv') == shared.read(out/'data_manifest.json')['train_val_csv_sha256']
        train, val = shared.dataset(cfg, 'train', args.train_limit), shared.dataset(cfg, 'val', args.val_limit)
        models = {a: m.to(args.device) for a, m in initialize_pair(cfg).items()}
        assert {a: sum(p.numel() for p in m.parameters()) for a, m in models.items()} == COUNTS
        optimizers = {a: torch.optim.Adam(norm_bias_no_decay_groups(m, 1e-4), lr=.001) for a, m in models.items()}
        if not args.resume:
            initial_audit = initialization_audit(models)
            shared.atomic_json(out/'initialization_audit.json', initial_audit)
            initial = {a: m.state_dict() for a, m in models.items()}
            shared.atomic_torch(out/'initial.pt', initial)
            optimizer_audit = {}
            old_decay = shared.read(OLD/'training/2021/optimizer_groups.json')['patch_covariance']
            for arm in ARMS:
                names = {id(p): n for n, p in models[arm].named_parameters()}
                optimizer_audit[arm] = [dict(weight_decay=g['weight_decay'], group_name=g['group_name'],
                    names=[names[id(p)] for p in g['params']]) for g in optimizers[arm].param_groups]
                current_decay = {n:g['weight_decay'] for g in optimizer_audit[arm] for n in g['names']}
                assert set(current_decay) == set(names.values())
                assert all(current_decay[n] == old_decay[n] for n in set(current_decay)&set(old_decay))
                for n in current_decay:
                    if '.trend_out_cross_layer.' in n:
                        assert current_decay[n] == current_decay[n.replace('.trend_out_cross_layer.','.out_cross_layer.')]
            shared.atomic_json(out/'optimizer_groups.json', optimizer_audit)

            curves, best, chosen, stale, updates, start = [], {a: math.inf for a in ARMS}, {a: 0 for a in ARMS}, 0, 0, 1
        else:
            last = torch.load(out/'last.pt', map_location=args.device, weights_only=True)
            for arm in ARMS:
                models[arm].load_state_dict(last['models'][arm], strict=True)
                optimizers[arm].load_state_dict(last['optimizers'][arm])
            curves, best, chosen = last['curves'], last['best'], last['best_epoch']
            stale, updates, start = last['stale'], last['updates_per_arm'], last['epoch']+1
        for epoch in range(start, args.epochs+1):
            for model in models.values():
                model.train()
            for opt in optimizers.values():
                for group in opt.param_groups:
                    group['lr'] = shared.learning_rate(epoch)
            totals = {a: np.zeros(4) for a in ARMS}; windows = 0
            batches=shared.loader(train, epoch)
            for step, (x, target, _, _) in enumerate(batches, 1):
                x, target = x.float().to(args.device), target[:, -HORIZON:].float().to(args.device)
                sampled = step in (1, 200, 400, 600, 800, 1000)
                losses = {}
                for arm in ARMS:
                    shared.seed_all(SEED*1000000+epoch*10000+step-1)
                    model, opt = models[arm], optimizers[arm]
                    statistics = FeatureStatistics(model) if sampled else None
                    opt.zero_grad(set_to_none=True)
                    prediction = model(x)
                    mse = (prediction-target.detach()).square().mean()
                    old_distance, diagnostic = parts(prediction, target)
                    distance,mmd_diagnostic=mmd_parts(prediction,target)
                    loss = mse+MMD_WEIGHT*distance
                    assert torch.isfinite(loss)
                    assert diagnostic['minimum_unclipped_distance'] >= -1e-10
                    if sampled:
                        row = dict(epoch=epoch, step=step, arm=arm, scopes=mmd_gradient_probe(model,mse,distance),
                                   old_covariance_scopes=gradient_probe(model,mse,old_distance),
                                   output_gradients=output_gradient_probe(prediction,mse,distance,old_distance),
                                   mse=float(mse.detach()), time_position_mmd=float(distance.detach()), old_covariance_diagnostic=float(old_distance.detach()), diagnostics=diagnostic,mmd_diagnostics=mmd_diagnostic)
                        row['mmd_weight']=MMD_WEIGHT
                        row['weighted_time_position_mmd']=MMD_WEIGHT*row['time_position_mmd']
                        row['weighted_scopes']={k:{'mmd_norm':MMD_WEIGHT*v['mmd_norm'],'mmd_mse_ratio':MMD_WEIGHT*v['mmd_mse_ratio']} for k,v in row['scopes'].items()}
                        row['features'] = statistics.summary()
                        statistics.close()
                        with (out/'gradients.jsonl').open('a') as file:
                            file.write(json.dumps(row)+'\n')
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                            raise RuntimeError('Missing/nonfinite gradient: '+arm+'/'+name)
                    opt.step()
                    totals[arm] += np.array([float(loss.detach()), float(mse.detach()), float(distance.detach()),float(old_distance.detach())])*len(x)
                    losses[arm] = float(loss.detach())
                windows += len(x); updates += 1
                if step == 1 or step%50 == 0:
                    shared.phase(out, 'training', epoch=epoch, step=step, steps=math.ceil(len(train)/32),
                                 losses=losses, completed_epochs=epoch-1)
            improved = False
            for arm in ARMS:
                shared.phase(out, 'validation', arm=arm, epoch=epoch)
                validation = evaluate(models[arm], val, args.device, out/arm, 'validation')
                if validation['mse'] < best[arm]:
                    best[arm], chosen[arm], improved = validation['mse'], epoch, True
                    shared.atomic_torch(out/arm/'best.pt', models[arm].state_dict())
                curves.append(dict(epoch=epoch, arm=arm, train_loss=float(totals[arm][0]/windows),
                    train_mse=float(totals[arm][1]/windows), train_mmd=float(totals[arm][2]/windows),train_d=float(totals[arm][3]/windows),mmd_weight=MMD_WEIGHT,train_weighted_mmd=float(MMD_WEIGHT*totals[arm][2]/windows),old_covariance_weight=0.,
                    val=validation, learning_rate=shared.learning_rate(epoch), training_windows=windows, training_indices_sha256=__import__('hashlib').sha256(np.asarray(batches.dataset.indices,dtype=np.int64).tobytes()).hexdigest()))
                shared.atomic_torch(out/arm/'epochs'/f'epoch{epoch:02d}.pt', models[arm].state_dict())
            stale = 0 if improved else stale+1
            last = dict(epoch=epoch, models={a: m.state_dict() for a, m in models.items()},
                optimizers={a: o.state_dict() for a, o in optimizers.items()}, curves=curves,
                best=best, best_epoch=chosen, stale=stale, updates_per_arm=updates)
            shared.atomic_torch(out/'last.pt', last)
            shared.atomic_json(out/'progress.json', {k:v for k,v in last.items() if k not in ('models', 'optimizers')})
            if args.stop_after_epochs and epoch >= args.stop_after_epochs:
                shared.phase(out, 'controlled_stop', completed_epochs=epoch)
                return
            if stale >= 5:
                break
        shared.phase(out, 'freezing', completed_epochs=last['epoch'])
        policy = verify_policy(last, len(train), args.epochs)
        shared.verify_sources(out)
        for rel, digest in shared.read(out/'reference_evidence.json').items():
            assert shared.sha(ROOT/rel) == digest, rel
        digests = {}
        for arm in ARMS:
            artifact = out/arm/'artifact'; artifact.mkdir()
            shutil.copyfile(out/arm/'best.pt', artifact/'checkpoint.pt')
            digests[arm] = shared.sha(artifact/'checkpoint.pt')
            selected = torch.load(artifact/'checkpoint.pt', map_location=args.device, weights_only=True)
            epoch_state = torch.load(out/arm/'epochs'/f'epoch{chosen[arm]:02d}.pt', map_location=args.device, weights_only=True)
            assert all(torch.equal(v, epoch_state[k]) for k, v in selected.items())
            models[arm].load_state_dict(selected, strict=True)
            shared.atomic_json(artifact/'model_config.json', vars(arm_config(cfg, arm)))
        shared.atomic_json(out/'freeze.json', dict(checkpoint_sha256=digests, policy=policy,
            selection='New single arm complete; saved controls untouched; own validation MSE only, no epoch0; test not opened', created=shared.now(), smoke=smoke))
        report = dict(status='preflight_verified' if smoke else 'validation_frozen', parameter_counts=COUNTS,
                      initialization=shared.read(out/'initialization_audit.json'), policy=policy, arms={})
        for arm in ARMS:
            shared.phase(out, 'capture_val', arm=arm)
            measured = evaluate(models[arm], val, args.device, out/arm, 'val', capture=True)
            assert abs(measured['mse']-best[arm]) < 1e-7
            shared.phase(out, 'validation_audit', arm=arm)
            report['arms'][arm] = dict(validation=audit_capture(out/arm, 'val', measured))
        if not smoke:
            assert shared.sha(ROOT/'dataset/ETT-small/ETTm1.csv') == shared.read(out/'data_manifest.json')['full_csv_sha256']
            with (out/'test_started.json').open('x') as file:
                json.dump(dict(created=shared.now(), checkpoint_sha256=digests, reason='New arm val artifact frozen/audited; saved controls not tested again'), file)
            test = shared.dataset(shared.config(ROOT/'dataset/ETT-small'), 'test')
            for arm in ARMS:
                shared.phase(out, 'test', arm=arm)
                with (out/arm/'test_started.json').open('x') as file:
                    json.dump(dict(checkpoint_sha256=digests[arm], created=shared.now()), file)
                measured = evaluate(models[arm], test, args.device, out/arm, 'test', capture=True)
                assert shared.sha(out/arm/'artifact/checkpoint.pt') == digests[arm]
                shared.atomic_json(out/arm/'test_receipt.json', dict(status='completed', windows=VAL_WINDOWS,
                    checkpoint_sha256=digests[arm], selected_epoch=chosen[arm], evaluations=1, created=shared.now()))
                shared.phase(out, 'test_audit', arm=arm)
                report['arms'][arm]['test'] = audit_capture(out/arm, 'test', measured)
            report['status'] = 'completed_verified'
        shared.verify_sources(out)
        shared.atomic_json(out/'report.json', report)
        shared.atomic_json(out/'completion_audit.json', dict(status='verified', smoke=smoke, policy=policy,
            checkpoint_sha256=digests, test_windows_per_arm=0 if smoke else VAL_WINDOWS,
            source_count=len(shared.read(out/'sources_manifest.json')), created=shared.now()))
        if not smoke:
            publish(out, report)
            shared.register(out, 'completed_verified', report=str(REPORT), policy=policy,
                test_metrics={a: {k: report['arms'][a]['test']['metrics'][k] for k in ('mse', 'mae')} for a in ARMS})
        shared.phase(out, report['status'], **policy)
    except BaseException as error:
        shared.atomic_json(out/'failure.json', dict(created=shared.now(), error=repr(error), traceback=traceback.format_exc()))
        shared.phase(out, 'failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
