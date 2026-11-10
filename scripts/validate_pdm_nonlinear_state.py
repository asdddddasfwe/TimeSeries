"""CPU implementation validation, not an ETTm1 effectiveness experiment.

Uses four training windows from an existing train/val-only prefix, two optimizer
steps, saved round-trip config/weights, and isolated recurrence diagnostics.
Does not load prior trained weights, access test, or use the occupied CUDA device.
"""
import argparse
import copy
import importlib
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_timemix_pdm_learning_rate import initialize_pair, dataset, parameter_groups
from models.OscillatorPDMNonlinearState import Model, prepare_recurrence
from layers.NonlinearOscillatorScan import (
    sequential_nonlinear_scan, parallel_nonlinear_scan, recurrence_residual,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    opts = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(2021)
    protocol = json.loads(opts.protocol.read_text())
    out = opts.output.resolve();out.mkdir(parents=True, exist_ok=False)
    cfg = SimpleNamespace(**protocol['model_args'])
    cfg.use_gpu = False
    cfg.nonlinear_solver = 'parallel'
    cfg.nonlinear_iterations = 4
    cfg.nonlinear_hidden = 32
    csv = Path(cfg.root_path)/cfg.data_path
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == protocol['options']['train_val_csv_sha256']
    with csv.open('rb') as f:assert sum(1 for _ in f) == 46081
    train = dataset(cfg, 'train', 0)
    assert len(train) == 34369
    indices = [0, 1000, 16000, 30000]
    x = torch.stack([torch.as_tensor(train[i][0]).float() for i in indices])
    y = torch.stack([torch.as_tensor(train[i][1][-96:]).float() for i in indices])
    baseline = initialize_pair(cfg)['lr1e3'].eval()
    cfg.model = 'OscillatorPDMNonlinearState'
    model = Model(cfg).eval()
    merged = model.state_dict()
    for name,value in baseline.state_dict().items():merged[name]=value
    model.load_state_dict(merged,strict=True)
    initial = []
    for training in (False, True):
        values=[];rng=[]
        for m in (baseline,model):
            m.train(training);torch.manual_seed(41)
            with torch.no_grad():values.append(m(x))
            rng.append(torch.get_rng_state())
        torch.testing.assert_close(*values,rtol=0,atol=0)
        assert torch.equal(*rng)
        initial.append(dict(mode='train' if training else 'eval',max_abs_difference=float((values[0]-values[1]).abs().max()),rng_equal=True))
    groups=parameter_groups(model,'lr1e3')
    optimizer=torch.optim.Adam(groups,lr=.001)
    steps=[];model.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);torch.manual_seed(2021+step)
        prediction=model(x[step*2:(step+1)*2]);loss=(prediction-y[step*2:(step+1)*2]).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        norms={}
        for name,param in model.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(),name
            if '.state_correction.' in name:norms[name]=float(param.grad.norm())
        steps.append(dict(step=step+1,mse=float(loss.detach()),correction_gradient_norms=norms))
        optimizer.step()
    assert any(v>0 for n,v in steps[1]['correction_gradient_norms'].items() if '.input.' in n)
    model.eval()
    torch.save(model.state_dict(),out/'smoke_state.pth')
    (out/'model_config.json').write_text(json.dumps(vars(cfg),indent=2)+'\n')
    saved_cfg=SimpleNamespace(**json.loads((out/'model_config.json').read_text()))
    restored=importlib.import_module('models.'+saved_cfg.model).Model(saved_cfg).eval()
    restored.load_state_dict(torch.load(out/'smoke_state.pth',weights_only=True,map_location='cpu'),strict=True)
    diagnostics=[]
    with torch.no_grad():
        actual=model(x[:2]);torch.testing.assert_close(actual,restored(x[:2]),atol=0,rtol=0)
        restored.nonlinear_solver='sequential';serial_prediction=restored(x[:2])
        prediction_error=float((actual-serial_prediction).abs().max())
        parts=model.predict_components(x[:2])
        reconstructed=model.normalize(parts[...,:4].sum(-1)+parts[...,4:].sum(-1),'denorm')
        torch.testing.assert_close(actual,reconstructed,atol=0,rtol=0)
        scales,(seasons,trends)=model.encode_multiscale(model.downsample(x[:2]))
        alpha,beta=model.branch_gains()
        for scale,(encoder,xx,season,trend) in enumerate(zip(model.seasonals,scales,seasons,trends)):
            args=prepare_recurrence(encoder,xx,alpha,beta,season,trend)
            correction=encoder.decoder.state_correction
            reference=sequential_nonlinear_scan(*args,correction)
            rows=[]
            for k in (0,1,2,4,8,16,96):
                values=parallel_nonlinear_scan(*args,correction,k)
                residual=recurrence_residual(values,*args,correction)
                assert torch.isfinite(values).all() and torch.isfinite(residual).all()
                error=float((values-reference).abs().max())
                rows.append(dict(iterations=k,max_abs_state_error=error,max_abs_recurrence_residual=float(residual.abs().max())))
                if k==96:torch.testing.assert_close(values,reference,atol=1e-5,rtol=1e-5)
            # Include nonzero state coupling beyond the two tiny optimizer steps.
            saved=correction.output.weight.clone()
            correction.output.weight.normal_(0,.03)
            reference=sequential_nonlinear_scan(*args,correction)
            stress=[]
            for k in (1,2,4,8,16,96):
                values=parallel_nonlinear_scan(*args,correction,k)
                assert torch.isfinite(values).all() and torch.isfinite(reference).all()
                stress.append(dict(iterations=k,max_abs_state_error=float((values-reference).abs().max()),max_abs_recurrence_residual=float(recurrence_residual(values,*args,correction).abs().max())))
            # Float32 round-off can accumulate in a strongly coupled recurrence.
            # Verify mathematical solver equivalence separately in float64, while
            # retaining the measured FP32 errors instead of calling them exact.
            double_args=tuple(t.double() for t in args)
            double_correction=copy.deepcopy(correction).double()
            double_serial=sequential_nonlinear_scan(*double_args,double_correction)
            double_parallel=parallel_nonlinear_scan(*double_args,double_correction,96)
            torch.testing.assert_close(double_parallel,double_serial,atol=1e-10,rtol=1e-10)
            double_error=float((double_parallel-double_serial).abs().max())
            correction.output.weight.copy_(saved)
            times={}
            for name,fn in [('sequential',lambda:sequential_nonlinear_scan(*args,correction)),('parallel4',lambda:parallel_nonlinear_scan(*args,correction,4))]:
                fn();samples=[]
                for _ in range(5):
                    start=time.perf_counter();fn();samples.append(time.perf_counter()-start)
                times[name]=statistics.median(samples)
            diagnostics.append(dict(scale=scale,history_length=encoder.length,after_two_steps=rows,synthetic_nonzero_output_weights_std=.03,synthetic_stress=stress,stress_float64_horizon_rounds_max_abs_error=double_error,cpu_forward_median_seconds=times))
    # Existing launch snapshot is untouched; newly added modules are separate.
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    counts=dict(baseline=sum(p.numel() for p in baseline.parameters()),nonlinear=sum(p.numel() for p in model.parameters()))
    result=dict(status='passed',seed=2021,device='cpu',cpu_threads=1,cuda_used=False,test_accessed=False,train_window_indices=indices,training_steps=steps,initial_function=initial,parameters=counts,default_iterations=4,prediction_shape=list(actual.shape),after_two_steps_parallel4_vs_serial_prediction_max_abs=prediction_error,config_checkpoint_roundtrip_exact=True,component_reconstruction_exact=True,source_snapshot_unchanged=len(protocol['source_sha256']),scales=diagnostics,caveat='Implementation smoke only; no validation/test effect or GPU speed claim; finite iteration approximates nonlinear recurrence and must be monitored during training.')
    files=['layers/NonlinearOscillatorScan.py','models/OscillatorPDMNonlinearState.py','tests/test_nonlinear_oscillator_state.py','scripts/validate_pdm_nonlinear_state.py']
    result['implementation_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('training_steps','scales','implementation_sha256')},indent=2))
    for row in diagnostics:print('SCALE',row['scale'],'after_steps_k4',row['after_two_steps'][3],'stress_k4',row['synthetic_stress'][2],'times',row['cpu_forward_median_seconds'])


if __name__=='__main__':main()
