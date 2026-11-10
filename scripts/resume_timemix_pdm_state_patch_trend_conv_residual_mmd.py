"""User-authorized manual recovery of group2; original frozen runner remains unchanged."""
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
from utils.pdm_state_patch_trend_temporal_conv_no_decay_diagnostics import FeatureStatistics, gradient_probe, weight_statistics
from models.OscillatorPDMStatePatchSeparateMLP import Model as SeparateMLP
from models.OscillatorPDMStatePatchMAFirst import Model as PositionOnly
from models.OscillatorPDMStatePatchTrendTemporalConv import Model as TrendTemporalConv
from models.OscillatorPDMStatePatchTrendTemporalInnerResidual import Model as InnerResidual
from utils.time_position_patch_mmd import parts as mmd_parts, numpy_parts as numpy_mmd_parts
from models.OscillatorPDMStatePatchPreDecompSliding import Model as Sliding
from scripts import run_timemix_dual_encoder_experiment as shared
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts

ALL_ARMS = ('trend_temporal_conv_inner_residual', 'trend_temporal_conv_inner_residual_mmd', 'trend_temporal_conv_no_decay_mmd')
ARMS = ALL_ARMS
CLASSES = dict(zip(ALL_ARMS, (InnerResidual, InnerResidual, TrendTemporalConv)))
EXPERIMENT_OLD = ROOT/'tmp/lag_study_timemix_pdm_state_patch_trend_temporal_conv_20261015_seed2021_fixed11'
NO_DECAY_OLD = ROOT/'tmp/lag_study_timemix_pdm_state_patch_trend_temporal_conv_no_decay_20261015_seed2021_fixed11'
MMD_OLD = ROOT/'tmp/lag_study_timemix_pdm_state_patch_time_position_mmd_20261015_seed2021_fixed11'
MODEL_NAMES = {a: cls.__module__.split('.')[-1] for a, cls in CLASSES.items()}
PROTOCOL = ROOT/'docs/experiments/PDMStatePatchTrendConvInnerResidualMMDRecovery_protocol_20261015.json'
COUNTS = {a:145986 for a in ARMS}
STATE_KEY = None
REPORT = None

def configure(arm):
    global ARMS, COUNTS, STATE_KEY, REPORT
    assert arm in ALL_ARMS
    ARMS = (arm,)
    COUNTS = {arm:145986}
    STATE_KEY = 'pdm_state_patch_'+arm+'_results'
    REPORT = ROOT/'docs/experiments/PDMStatePatch_trend_temporal_conv_inner_residual_mmd_recovery1_results_20261015.html'
    shared.PROTOCOL, shared.STATE_KEY = PROTOCOL, STATE_KEY

def uses_mmd(arm):
    return arm.endswith('_mmd')


def conv_weight_decay(arm):
    """Internal-residual arms retain original Adam L2; arm3 stays no-decay."""
    return 0. if arm == ALL_ARMS[2] else 1e-4


def optimizer_groups(model, arm=None):
    """Original policy for arms1/2; only16 convolution tensors lose L2 in arm3."""
    arm = arm or ARMS[0]
    protected={id(p) for n,p in model.named_parameters() if n.startswith('trend_temporal_encoders.')}
    assert len(protected)==16
    groups=[]
    for original in norm_bias_no_decay_groups(model,1e-4):
        keep=[p for p in original['params'] if id(p) not in protected]
        conv=[p for p in original['params'] if id(p) in protected]
        if keep:groups.append({**original,'params':keep})
        if conv:groups.append({**original,'params':conv,'weight_decay':conv_weight_decay(arm),'group_name':'trend_temporal_encoder'})
    ids=[id(p) for g in groups for p in g['params']]
    assert len(ids)==len(set(ids))==220 and set(ids)=={id(p) for p in model.parameters()}
    return groups


def arm_config(cfg, arm):
    return SimpleNamespace(**{**vars(cfg), 'model': MODEL_NAMES[arm]})


def scratch_reference(cfg):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2021)
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
    saved = torch.load(NO_DECAY_OLD/'initial.pt',map_location='cpu',weights_only=True)['trend_temporal_conv_no_decay']
    result = {}
    for arm, model in models.items():
        device = next(model.parameters()).device
        state = model.state_dict()
        assert set(state)==set(saved) and len(state)==252
        assert len(list(model.parameters()))==220 and sum(p.numel() for p in model.parameters())==145986
        for n,v in state.items():
            assert torch.equal(v.cpu(),saved[n]) and v.data_ptr()!=saved[n].data_ptr(), n
        for encoder in model.trend_temporal_encoders:
            assert torch.count_nonzero(encoder.output.weight)==0
        # Fresh reference function only, never a trained control or held-out test.
        with torch.random.fork_rng(devices=[]):
            reference = TrendTemporalConv(config(ROOT/'dataset/ETT-small')).to(device)
        reference.load_state_dict({n:v.detach().clone() for n,v in state.items()})
        audit = dict(states=252,parameter_tensors=220,buffers=32,parameters=145986,
            all252_saved_no_decay_initial_exact=True,storage_independent=True,
            added_states=[],removed_states=[],zero_output=True,old_trained_weights_loaded=False,
            initialization_seed_actual=20211014,internal_residual=arm!=ALL_ARMS[2],convolution_weight_decay=conv_weight_decay(arm),
            objective='MSE+time_position_mmd' if uses_mmd(arm) else 'MSE+covarianceD')
        x = torch.randn(2,96,7,generator=torch.Generator().manual_seed(101)).to(device)
        for training in (False,True):
            outputs, rng = [], []
            for m in (reference,model):
                m.train(training)
                with torch.random.fork_rng(devices=[device.index or 0] if x.is_cuda else []):
                    shared.seed_all(314)
                    with torch.no_grad():outputs.append(m.predict_components(x))
                    rng.append(torch.cuda.get_rng_state(device) if x.is_cuda else torch.get_rng_state())
            assert torch.equal(outputs[0],outputs[1]) and torch.equal(rng[0],rng[1])
            audit['train' if training else 'eval']=dict(component_max_difference=0.,rng_equal=True)
        result[arm]=audit
    return result


OLD = ROOT/'tmp/lag_study_timemix_pdm_state_patch_20261011_seed2021_earlystop5'
MA_OLD = ROOT/'tmp/lag_study_timemix_pdm_state_patch_ma_decoupled_20261014_seed2021_earlystop5'
MLP_OLD = ROOT/'tmp/lag_study_timemix_pdm_state_patch_separate_mlp_20261014_seed2021_earlystop5'


def config(root_path):
    values = dict(shared.read(OLD/'training/protocol.json')['model_args'])
    values.update(root_path=str(root_path), model=MODEL_NAMES[ARMS[0]],
                  log_dir=str(Path(root_path).parent/'logs'), tensorboard=False)
    return SimpleNamespace(**values)


def loader(data, epoch=None):
    if epoch is not None:
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(2021*1000+epoch)).tolist()
        data = Subset(data, order)
    return DataLoader(data, batch_size=32, shuffle=False, num_workers=0, drop_last=False)


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
        for key, tail in [('prediction', (96, 7)), ('truth', (96, 7)), ('history', (96, 7)),
                          ('components', (96, 7, 8)), ('history_mean', (1, 7))]:
            arrays[key] = np.lib.format.open_memmap(destination/(key+'.npy'), mode='w+',
                dtype='float32', shape=(len(data), *tail))
    offset = 0
    try:
        with torch.no_grad():
            for batch_index, (x, target, _, _) in enumerate(loader(data)):
                x, target = x.float().to(device), target[:, -96:].float().to(device)
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
        shared.atomic_json(folder/(label+'_encoder_features.json'), statistics.encoder_summary())
    return result


def verify_policy(last, train_windows, maximum):
    best = {a: math.inf for a in ARMS}; chosen = {a: 0 for a in ARMS}; stale = 0
    for epoch in range(1, last['epoch']+1):
        improved = False
        rows = [row for row in last['curves'] if row['epoch'] == epoch]
        assert {row['arm'] for row in rows} == set(ARMS)
        for row in rows:
            assert row['learning_rate'] == shared.learning_rate(epoch)
            assert abs(row['train_loss']-row['train_mse']-row['train_mmd' if uses_mmd(row['arm']) else 'train_d']) < 2e-6
            arm = row['arm']
            if row['val']['mse'] < best[arm]:
                best[arm], chosen[arm], improved = row['val']['mse'], epoch, True
        stale = 0 if improved else stale+1
    assert best == last['best'] and chosen == last['best_epoch'] and stale == last['stale']
    assert last['epoch'] == maximum
    updates = last['epoch']*math.ceil(train_windows/32)
    assert last['updates_per_arm'] == updates
    for arm in ARMS:
        assert all(int(v['step']) == updates for v in last['optimizers'][arm]['state'].values())
    return dict(epochs=last['epoch'], selected_epochs=chosen, single_stale_diagnostic=stale, updates_per_arm=updates, stopping='fixed11 matched saved controls; cosine30 unchanged')


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
    for rel in ('initial.pt','optimizer_groups.json','report.json','completion_audit.json','sources_manifest.json','offline_final_review.json','offline_ma_review.json','encoder_collapse_diagnosis_20261015.json','trend_temporal_conv/artifact/checkpoint.pt','trend_temporal_conv/test_receipt.json'):
        path=EXPERIMENT_OLD/rel;result[str(path.relative_to(ROOT))]=shared.sha(path)
    for folder, arm in ((NO_DECAY_OLD,'trend_temporal_conv_no_decay'),(MMD_OLD,'time_position_mmd')):
        for rel in ('initial.pt','optimizer_groups.json','report.json','completion_audit.json','sources_manifest.json','offline_final_review.json','offline_ma_review.json',arm+'/artifact/checkpoint.pt',arm+'/test_receipt.json'):
            path=folder/rel;result[str(path.relative_to(ROOT))]=shared.sha(path)
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
    for label,values in [('all',gradients),('low4',[project(g) for g in gradients]),('remaining92',[g-project(g) for g in gradients]),('level',[g.mean(1,keepdim=True).expand_as(g) for g in gradients])]:
        a,b,c=[v.flatten() for v in values];na,nb,nc=[float(v.norm()) for v in (a,b,c)]
        result[label]=dict(mse_norm=na,mmd_norm=nb,old_d_norm=nc,mmd_mse_ratio=nb/max(na,1e-30),mmd_mse_cosine=float(a@b)/max(na*nb,1e-30),old_d_mse_ratio=nc/max(na,1e-30),old_d_mse_cosine=float(a@c)/max(na*nc,1e-30))
    return result


def audit_capture(folder,label,measured):
    result=shared.audit_capture(folder,label,measured)
    p,y=[np.load(folder/label/(name+'.npy'),mmap_mode='r') for name in ('prediction','truth')]
    relation=numpy_mmd_parts(p,y)
    assert all(abs(v-measured[k])<1e-6 for k,v in relation.items())
    result['time_position_mmd']=relation;result['objective']='MSE+1*time_position_mmd; covarianceD diagnostic only' if uses_mmd(ARMS[0]) else 'MSE+1*covarianceD; MMD diagnostic only'
    shared.atomic_json(folder/label/'report.json',result)
    return result


def publish(out,report):
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+ARMS[0]+'</title><h1>'+ARMS[0]+'：训练完成，等待完整离线复核</h1><pre>'+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre></html>'
    with (ROOT/'tmp/pdm_state_patch_rope_registration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        t=REPORT.with_suffix('.html.tmp');t.write_text(page);t.replace(REPORT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ALL_ARMS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=11)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.4)
    parser.add_argument('--stop-after-epochs', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    configure(args.arm)
    if not 1 <= args.epochs <= 30 or min(args.train_limit, args.val_limit) < 0 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error('Invalid experiment budget')
    smoke = bool(args.train_limit or args.val_limit)
    if smoke and not (args.train_limit >= 4 and args.val_limit >= 4):
        parser.error('Preflight requires both limits >=4')
    if not smoke and args.epochs != 11:
        parser.error('Formal new arm must match saved11-epoch budget; cosine remains30')
    if not args.resume or args.arm != 'trend_temporal_conv_inner_residual_mmd':
        parser.error('Only explicitly authorized group2 continuation is supported')
    if not smoke:
        authorization=shared.read(PROTOCOL)['recovery']
        assert args.output.resolve()==Path(authorization['output'])
        assert shared.read(args.output/'manual_recovery_authorization.json')['authorized_by_user']
    out = args.output.resolve()
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    lock = (out/'pipeline.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    shared.atomic_json(out/'process.json', shared.process_record())
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    torch.set_num_threads(2)
    # New CUDA convolutions require a reproducible kernel for controlled resume.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    shared.seed_all(2021)
    try:
        if not args.resume:
            shared.phase(out, 'preparing')
            prefix = out/'train_val_data'; prefix.mkdir()
            with (ROOT/'dataset/ETT-small/ETTm1.csv').open('rb') as src, (prefix/'ETTm1.csv').open('wb') as dst:
                for _ in range(46081):
                    line = src.readline(); assert line; dst.write(line)
            cfg = shared.config(prefix)
            shared.atomic_json(out/'data_manifest.json', dict(train_val_csv_sha256=shared.sha(prefix/'ETTm1.csv'),
                full_csv_sha256=shared.sha(ROOT/'dataset/ETT-small/ETTm1.csv'), scaler_fit_rows=34560, train_val_rows=46080))
            shared.atomic_json(out/'reference_evidence.json', references())
            shared.atomic_json(out/'protocol.json', dict(model_config=vars(cfg), options={**vars(args), 'output': str(out)},
                formal_protocol=shared.read(PROTOCOL), smoke=smoke, started=shared.now()))
            manifest = shared.snapshot_sources(out)
            shutil.copyfile(ROOT/'scripts/monitor_timemix_pdm_state_patch_trend_conv_residual_mmd.py', out/'monitor_check.py')
            for arm in ARMS:
                (out/arm/'epochs').mkdir(parents=True)
            if not smoke:
                shared.register(out, 'running', **shared.process_record(), source_count=len(manifest),
                    protocol=str(PROTOCOL), parameter_counts=COUNTS, arms=list(ARMS), launch_log=str(out)+'_launch.log')
        else:
            saved = shared.read(out/'protocol.json')
            assert shared.read(out/'status.json')['status'] in ('controlled_stop','manual_recovery_prepared')
            assert saved['smoke']==smoke
            if not smoke:
                assert shared.sha(out/'last.pt')==shared.read(PROTOCOL)['recovery']['parent_checkpoint_sha256']
                assert shared.read(out/'manual_recovery_authorization.json')['authorized_by_user']
            for key in ('arm', 'epochs', 'train_limit', 'val_limit', 'device'):
                assert getattr(args, key) == saved['options'][key]
            cfg = SimpleNamespace(**saved['model_config'])
            if not smoke:
                shared.register(out,'running',**shared.process_record(),source_count=len(shared.read(out/'sources_manifest.json')),protocol=str(PROTOCOL),parameter_counts=COUNTS,arms=list(ARMS),launch_log=str(out)+'_launch.log',manual_recovery=True,parent_output=shared.read(PROTOCOL)['recovery']['parent_output'],resume_epoch=2)
        shared.verify_sources(out)
        assert shared.sha(Path(cfg.root_path)/'ETTm1.csv') == shared.read(out/'data_manifest.json')['train_val_csv_sha256']
        train, val = shared.dataset(cfg, 'train', args.train_limit), shared.dataset(cfg, 'val', args.val_limit)
        models = {a: m.to(args.device) for a, m in initialize_pair(cfg).items()}
        assert {a: sum(p.numel() for p in m.parameters()) for a, m in models.items()} == COUNTS
        optimizers = {a: torch.optim.Adam(optimizer_groups(m,a), lr=.001) for a, m in models.items()}
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
                original_policy={n:g['weight_decay'] for g in shared.read(EXPERIMENT_OLD/'optimizer_groups.json')['trend_temporal_conv'] for n in g['names']}
                changed={n for n in current_decay if current_decay[n]!=original_policy[n]}
                expected_changes={n for n in current_decay if n.startswith('trend_temporal_encoders.')} if conv_weight_decay(arm)==0 else set()
                assert changed==expected_changes
                assert all(original_policy[n]==1e-4 and current_decay[n]==0 for n in changed)
                no_decay_policy={n:g['weight_decay'] for g in shared.read(NO_DECAY_OLD/'optimizer_groups.json')['trend_temporal_conv_no_decay'] for n in g['names']}
                for n in no_decay_policy:
                    if n.startswith('trend_temporal_encoders.'):no_decay_policy[n]=conv_weight_decay(arm)
                assert current_decay==no_decay_policy
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
            for step, (x, target, _, _) in enumerate(shared.loader(train, epoch), 1):
                x, target = x.float().to(args.device), target[:, -96:].float().to(args.device)
                sampled = step in (1, 200, 400, 600, 800, 1000)
                losses = {}
                for arm in ARMS:
                    shared.seed_all(2021*1000000+epoch*10000+step-1)
                    model, opt = models[arm], optimizers[arm]
                    statistics = FeatureStatistics(model) if sampled else None
                    opt.zero_grad(set_to_none=True)
                    prediction = model(x)
                    mse = (prediction-target.detach()).square().mean()
                    old_distance, diagnostic = parts(prediction, target)
                    mmd, mmd_diagnostic = mmd_parts(prediction,target)
                    distance = mmd if uses_mmd(arm) else old_distance
                    loss = mse+distance
                    assert torch.isfinite(loss)
                    assert diagnostic['minimum_unclipped_distance'] >= -1e-10
                    if sampled:
                        row = dict(epoch=epoch,step=step,arm=arm,
                            scopes=mmd_gradient_probe(model,mse,mmd) if uses_mmd(arm) else gradient_probe(model,mse,old_distance),
                            old_covariance_scopes=gradient_probe(model,mse,old_distance),
                            mmd_scopes=mmd_gradient_probe(model,mse,mmd),
                            output_gradients=output_gradient_probe(prediction,mse,mmd,old_distance),
                            mse=float(mse.detach()),time_position_mmd=float(mmd.detach()),
                            distance=float(distance.detach()),old_covariance_diagnostic=float(old_distance.detach()),
                            diagnostics=diagnostic,mmd_diagnostics=mmd_diagnostic,
                            objective='MSE+time_position_mmd' if uses_mmd(arm) else 'MSE+covarianceD')
                        row['features'] = statistics.summary()
                        row['encoder_features'] = statistics.encoder_summary()
                        row['encoder_weights'] = weight_statistics(model)
                        statistics.close()
                        with (out/'gradients.jsonl').open('a') as file:
                            file.write(json.dumps(row)+'\n')
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                            raise RuntimeError('Missing/nonfinite gradient: '+arm+'/'+name)
                    opt.step()
                    totals[arm] += np.array([float(loss.detach()),float(mse.detach()),float(old_distance.detach()),float(mmd.detach())])*len(x)
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
                    train_mse=float(totals[arm][1]/windows),train_d=float(totals[arm][2]/windows),train_mmd=float(totals[arm][3]/windows),mmd_weight=float(uses_mmd(arm)),old_covariance_weight=float(not uses_mmd(arm)),
                    val=validation, learning_rate=shared.learning_rate(epoch)))
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
            selection='Independent arm complete; saved controls untouched; own validation MSE only, no epoch0; test not opened', created=shared.now(), smoke=smoke))
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
                shared.atomic_json(out/arm/'test_receipt.json', dict(status='completed', windows=11425,
                    checkpoint_sha256=digests[arm], selected_epoch=chosen[arm], evaluations=1, created=shared.now()))
                shared.phase(out, 'test_audit', arm=arm)
                report['arms'][arm]['test'] = audit_capture(out/arm, 'test', measured)
            report['status'] = 'completed_verified'
        shared.verify_sources(out)
        shared.atomic_json(out/'report.json', report)
        shared.atomic_json(out/'completion_audit.json', dict(status='verified', smoke=smoke, policy=policy,
            checkpoint_sha256=digests, test_windows_per_arm=0 if smoke else 11425,
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
