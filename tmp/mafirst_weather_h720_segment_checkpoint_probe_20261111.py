"""Isolated Weather-H720 FP32 segmented activation checkpoint capacity probe."""
import functools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[1]
sys.path.insert(0, str(REPO))
os.environ["MAFIRST_FUTURE_HIDDEN_ONLY_DATASET_20261107"] = "Weather"
os.environ["MAFIRST_FUTURE_HIDDEN_ONLY_HORIZON_20261107"] = "720"

import torch
from torch.utils.checkpoint import checkpoint
from models.OscillatorPDMStatePatchSpectralTimevaryFutureHiddenV320261105 import (
    SpectralModel, complex_oscillator_scan, future_rotation_delta, spectral_dynamics,
)
from tmp import mafirst_future_hidden_only_five_datasets_full100_20261109_common as common
from tmp import mafirst_future_hidden_only_five_datasets_full100_20261109_factory as factory
from tmp import run_mafirst_future_hidden_only_five_datasets_full100_20261109 as runner

SOURCE = common.G
ARM = common.ARMS[0]
ROOT = SCRIPT.with_suffix("")
GROUP = ROOT / "group"
OUTPUT = ROOT / "probe"
PREDECESSOR = REPO / "tmp/mafirst_etth2_boundary_input_other_horizons_20261111"
RECORDS = ("plan.json", "expected_model_cards.json", "reference_initial/mafirst_season.pt")


def checkpoint_branch(fn, *args):
    return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=True)


def recurrent_branch(model, index, batch, channels, history, trend, spectrum, alpha, beta):
    encoder = model.seasonals[index]
    decoder = encoder.decoder
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    n, horizon, width = future.shape
    steps = horizon // 4
    future_blocks = future.reshape(n, steps, 4, width)
    future = future_blocks.mean(2)
    shift = model.spectral_conditions[index](spectrum)
    transition, initial, _, angle = spectral_dynamics(decoder, history, trend, shift)
    delta = future_rotation_delta(
        decoder, future_blocks, angle[:, None], math.pi / decoder.stride,
        model.position_frequency_basis, model.position_frequency_heads[index],
    )
    co, si = delta.cos(), delta.sin()
    real, imag = transition.unbind(-1)
    sequence = torch.stack((real[:, None] * co - imag[:, None] * si,
                            real[:, None] * si + imag[:, None] * co), -1)
    drive = decoder.future_drive(future, history).reshape(n, steps, decoder.modes, 2)
    initial = initial * alpha.repeat(batch)[:, None, None]
    drive = drive * beta.repeat(batch)[:, None, None, None]
    states = complex_oscillator_scan(sequence, drive, initial)
    outputs = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    return (outputs.reshape(batch, channels, horizon).transpose(1, 2),
            delta.detach().abs().max(), shift.detach().abs().max())


def segmented_components(model, x):
    assert x.ndim == 3 and x.shape[1:] == (96, 21)
    assert not model.direct and model.spectral_enabled and model.timevary_enabled
    scales, (seasons, trends) = model.encode_multiscale(model.downsample(x))
    spectrum = model.spectral_encoder(SpectralModel.spectrum_features(model, scales[0]))
    alpha, beta = model.branch_gains()
    batch, _, channels = x.shape
    recurrent, diagnostics = [], []
    for index, (history, trend) in enumerate(zip(seasons, trends, strict=True)):
        branch = functools.partial(recurrent_branch, model, index, batch, channels)
        output, delta_max, shift_max = checkpoint_branch(branch, history, trend, spectrum, alpha, beta)
        recurrent.append(output)
        diagnostics.append(dict(context_error=0., trend_delta_energy=0., is_season_tensor=True,
                                history_length=history.shape[1], direct_generation=False,
                                spectral_before_history_scan=True, time_varying_rotation=True,
                                frequency_delta_max_abs=delta_max, future_hidden_conditioning=True,
                                spectral_shift_max_abs=shift_max))
    feature = [checkpoint_branch(head, trend).reshape(batch, channels, model.pred_len).transpose(1, 2)
               for head, trend in zip(model.feature_trend_heads, trends, strict=True)]
    model.last_gate_context_diagnostics = diagnostics
    return torch.stack(recurrent + feature, -1)


def training_prediction(model, x):
    with patch.object(model, "predict_components", functools.partial(segmented_components, model)):
        return model(x)


def verify_sources():
    common.guard()
    plan = common.read(SOURCE / "plan.json")
    assert (plan["dataset"], plan["horizon"], plan["variables"], plan["actual_train_batch"]) == (
        "Weather", 720, 21, 128,
    )
    return {name: common.sha(SOURCE / name) for name in RECORDS}


def configure_trial(hashes):
    for name, digest in hashes.items():
        assert common.sha(GROUP / name) == digest, name
    for module in (common, factory, runner):
        module.G = GROUP
        module.OUT = {ARM: OUTPUT}
        module.LARGE = True
    runner.training_prediction = training_prediction


def prepare():
    hashes = verify_sources()
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT / "preparation.json").exists() and not (ROOT / "launch.json").exists()
    for name in RECORDS:
        destination = GROUP / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with destination.open("xb") as output, (SOURCE / name).open("rb") as source:
                shutil.copyfileobj(source, output)
    configure_trial(hashes)
    runner.configure("cpu")
    model, _ = factory.fresh(ARM)
    model.train()
    x = torch.randn(2, 96, 21)
    before = torch.get_rng_state()
    expected = model(x)
    expected.square().mean().backward()
    expected_diag = model.last_gate_context_diagnostics
    after = torch.get_rng_state()
    grads = {name: value.grad.detach().clone() for name, value in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(before)
    actual = training_prediction(model, x)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), after)
    assert model.last_gate_context_diagnostics == expected_diag
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, grads[name], atol=0, rtol=0, msg=name)
    receipt("preparation", status="verified", cpu_batch=2, fp32=True, output_gradients_rng_bitwise_exact=True,
            parameter_gradients=len(grads), diagnostics_exact=True, source_hashes=hashes,
            script_sha256=common.sha(SCRIPT), predecessor=common.read(PREDECESSOR / "launch.json")["supervisor"])
    print((ROOT / "preparation.json").read_text(), flush=True)


def receipt(name, **values):
    target = ROOT / (name + ".json")
    pending = ROOT / f".{name}.{os.getpid()}.pending"
    pending.write_text(json.dumps(dict(created=common.now(), **values), indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
    pending.replace(target)


def gpu_probe():
    prepared = common.read(ROOT / "preparation.json")
    assert verify_sources() == prepared["source_hashes"]
    configure_trial(prepared["source_hashes"])
    original_step = runner.backward_step

    def step_with_probes(arm, model, opt, sched, xx, yy, device, epoch, step, probe=None):
        print(f"Actual batch128 step {step}/8", flush=True)
        probe = OUTPUT / "gradients.jsonl" if step in (1, 8) else None
        return original_step(arm, model, opt, sched, xx, yy, device, epoch, step, probe)

    runner.backward_step = step_with_probes
    phase = "cuda_equivalence_and_training_capacity"
    try:
        runner.memory(ARM, "cuda")
        evidence = common.read(GROUP / "memory" / ARM / "verification.json")
        assert (evidence["steps"], evidence["actual_loss_batch"], evidence["accumulations"]) == (8, 128, 1)
        phase = "validation128_capacity"
        torch.cuda.empty_cache()
        model, _ = factory.fresh(ARM)
        model.to("cuda").eval()
        val = factory.dataset(ARM, "val")
        x = next(iter(factory.loader(val, batch=128)))[0].float().to("cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            prediction = model(x)
            assert prediction.shape == (128, 720, 21) and torch.isfinite(prediction).all()
        torch.cuda.synchronize()
        receipt("result", status="verified", train=evidence, eval32_verified=True, eval128_verified=True,
                eval128_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                eval128_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                formal_training_started=False, test_evaluations=0)
    except Exception as exc:
        receipt("failure", phase=phase, error=repr(exc), traceback=traceback.format_exc(),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                formal_training_started=False, test_evaluations=0)
        raise


def supervise():
    prepared = common.read(ROOT / "preparation.json")
    assert prepared["script_sha256"] == common.sha(SCRIPT)
    predecessor = prepared["predecessor"]
    receipt("status", status="waiting_for_etth2", predecessor=predecessor)
    while common.alive(predecessor):
        time.sleep(20)
    previous = common.read(PREDECESSOR / "status.json")
    assert previous["status"] in ("completed", "completed_with_failures"), previous
    assert set(previous["completed"] + previous["failed"]) == {96, 336, 720}, previous
    receipt("predecessor_exit", predecessor=predecessor, terminal_status=previous)
    receipt("status", status="waiting_for_gpu_capacity")
    deadline = time.monotonic() + 3600
    while True:
        jobs = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True,
        ).strip()
        free = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True,
        ).strip()
        if not jobs and int(free) >= 7300:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU availability gate timed out, free MiB: {free}, jobs: {jobs!r}")
        time.sleep(30)
    with (ROOT / "gpu_probe.log").open("xb") as log:
        child = subprocess.Popen([sys.executable, "-B", "-u", str(SCRIPT), "gpu_probe"],
                                 cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        receipt("status", status="cuda_probe", process=common.identity(child.pid), free_mib=int(free))
        code = child.wait()
    receipt("probe_exit", returncode=code, real_parent_wait=True)
    assert code == 0 and common.read(ROOT / "result.json")["status"] == "verified"
    receipt("status", status="completed")


def launch():
    prepared = common.read(ROOT / "preparation.json")
    assert prepared["status"] == "verified" and prepared["script_sha256"] == common.sha(SCRIPT)
    with (ROOT / "supervisor.log").open("xb") as log:
        child = subprocess.Popen([sys.executable, "-B", "-u", str(SCRIPT), "supervise"],
                                 cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        receipt("launch", supervisor=common.identity(child.pid), script_sha256=common.sha(SCRIPT))
    print((ROOT / "launch.json").read_text(), flush=True)


if __name__ == "__main__":
    command = sys.argv[1]
    if command not in ("prepare", "launch", "supervise", "gpu_probe"):
        raise SystemExit("Expected prepare, launch, supervise or gpu_probe")
    try:
        globals()[command]()
    except Exception as exc:
        if ROOT.exists() and command == "supervise":
            receipt("supervisor_failure", error=repr(exc), traceback=traceback.format_exc())
            receipt("status", status="failed")
        raise

