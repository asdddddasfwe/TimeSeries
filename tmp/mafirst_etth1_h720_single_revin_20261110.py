"""Run one RevIN versus the saved ETTh1-H720 per-scale RevIN control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.update(MAFIRST_FUTURE_HIDDEN_ONLY_DATASET_20261107="ETTh1",
                  MAFIRST_FUTURE_HIDDEN_ONLY_HORIZON_20261107="720",
                  OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MPLBACKEND="Agg")

from tmp import mafirst_future_hidden_only_five_datasets_full100_20261109_common as common

BASELINE = common.G.parent
PREDECESSOR = BASELINE.parents[1]
CASE = REPO / "tmp/mafirst_etth1_h720_single_revin_20261110"
GROUP = CASE / "group"
OUTPUT = CASE / "output"
ARM = "mafirst_season"
REF = Path("reference_initial/mafirst_season.pt")
MODEL = "OscillatorPDMStatePatchFutureHiddenSingleRevIN20261110"
ROUTE = "RevIN on observed 96 points, then four average-pooled scales"
common.G = common.B = GROUP
common.OUT = {ARM: OUTPUT}
common.TAG = "future_hidden_only_single_revin"


def write(name, value):
    path = CASE / f"{name}.json"
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)
    assert common.read(path) == value


def status(phase, **details):
    write("status", dict(status=phase, updated=common.now(), **details))


def prepare():
    source = BASELINE / "group"
    plan = common.read(source / "plan.json")
    card = common.read(source / "expected_model_cards.json")
    completion = common.read(BASELINE / "output/completion.json")
    assert plan["actual_train_batch"] == 128 and plan["total_updates"] == 6100
    assert (completion["epochs"], completion["updates"], completion["selected_epoch"]) == (100, 6100, 41)
    assert common.read(BASELINE / "arm_exit.json")["exit_code"] == 0
    assert common.read(BASELINE / "arm_exit.json")["real_wait"]
    assert common.sha(source / REF) == plan["models"][ARM]["reference_sha256"]
    config = plan["models"][ARM]["config"]
    assert config["use_norm"] == 1 and not config["seasonal_normalize"]
    config.update(model=MODEL, log_dir=str(GROUP / "logs"), tensorboard_dir=str(GROUP / "tensorboard"))
    plan.update(input_normalization=ROUTE, revin_affine=False,
                initialization_policy="exact original untrained reference bytes",
                comparison=str(BASELINE))
    frozen = common.read(source / "frozen_sources.json")
    for path in (Path(__file__), REPO / "models" / f"{MODEL}.py"):
        frozen[path.relative_to(REPO).as_posix()] = common.sha(path)
    protected = common.read(source / "protected_records.json")
    for path in (source / "plan.json", source / REF, BASELINE / "arm_exit.json",
                 *(BASELINE / "output" / name for name in
                   ("completion.json", "official_test_result.json", "optimization_receipt.json", "curves.json"))):
        protected[path.relative_to(REPO).as_posix()] = common.sha(path)
    for path, digest in {**frozen, **protected}.items():
        assert common.sha(REPO / path) == digest, path
    CASE.mkdir(exist_ok=False)
    (GROUP / REF).parent.mkdir(parents=True)
    shutil.copyfile(source / REF, GROUP / REF)
    assert common.sha(GROUP / REF) == plan["models"][ARM]["reference_sha256"]
    for name, data in (("plan", plan), ("expected_model_cards", card),
                       ("frozen_sources", frozen), ("protected_records", protected)):
        common.atomic(GROUP / f"{name}.json", data)
    files = [REF, *(Path(f"{name}.json") for name in
                    ("plan", "expected_model_cards", "frozen_sources", "protected_records"))]
    common.atomic(GROUP / "registration.json", dict(hashes={
        (GROUP / file).relative_to(REPO).as_posix(): common.sha(GROUP / file) for file in files}))
    common.atomic(GROUP / "prepared.json", common.guard())
    status("prepared")


def install_factory():
    import torch
    from models.OscillatorPDMStatePatchFutureHiddenSingleRevIN20261110 import Model
    from tmp import mafirst_future_hidden_only_five_datasets_full100_20261109_factory as factory

    def fresh(arm):
        factory.seed(2021)
        config = factory.configuration(arm)
        model = Model(config)
        weights = torch.load(GROUP / REF, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        assert all(torch.equal(value, weights[name]) for name, value in model.state_dict().items())
        return model, config

    factory.fresh = fresh
    return factory


def verify(device):
    import random
    import numpy as np
    import torch
    from torch.nn import functional as F
    from models.OscillatorPDMStatePatchSpectralTimevaryFutureHiddenHorizons20261107 import Model as Control

    torch.set_num_threads(2)
    common.guard()
    factory = install_factory()
    factory.seed(2021)
    control = Control(factory.configuration(ARM))
    cpu_rng = torch.get_rng_state()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    model, _ = factory.fresh(ARM)
    assert torch.equal(cpu_rng, torch.get_rng_state()) and python_rng == random.getstate()
    for left, right in zip(numpy_rng, np.random.get_state()):
        assert np.array_equal(left, right)
    weights = torch.load(BASELINE / "group" / REF, map_location="cpu", weights_only=True)
    control.load_state_dict(weights, strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == sum(p.numel() for p in control.parameters()) == 359258
    assert model.predict_components.__func__ is control.predict_components.__func__
    assert model.forward.__func__ is control.forward.__func__
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.9)
    model.to(device).eval()
    control.to(device).eval()
    train = factory.dataset(ARM, "train")
    assert len(train) == 7825 and len(factory.dataset(ARM, "val")) == 2161
    windows = [train[i] for i in (0, 240, 1000, 2000)]
    history = torch.as_tensor(np.stack([row[0] for row in windows]), device=device, dtype=torch.float32)
    truth = torch.as_tensor(np.stack([row[1][-720:] for row in windows]), device=device, dtype=torch.float32)
    calls = []
    hook = model.normalize.register_forward_hook(lambda _, args, __: calls.append(args[1]))
    with torch.no_grad():
        prediction = model(history)
    hook.remove()
    assert calls == ["norm", "denorm"]
    assert prediction.shape == (4, 720, 7) and torch.isfinite(prediction).all()
    with torch.no_grad():
        baseline = control(history)
        difference = float((prediction - baseline).abs().max())
        assert difference > 0 and torch.isfinite(baseline).all()
        actual = model.downsample(history)
        expected = (history - history.mean(1, keepdim=True)) / (
            history.var(1, unbiased=False, keepdim=True) + model.normalize.eps).sqrt()
        for scale, length in zip(actual, (96, 48, 24, 12)):
            assert scale.shape[1] == length and torch.equal(scale, expected)
            if length > 12:
                expected = F.avg_pool1d(expected.transpose(1, 2), 2, 2).transpose(1, 2)
    loss, _, _ = factory.loss(ARM, model(history), truth, history)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    common.atomic(GROUP / f"{device}_verification.json", dict(
        status="verified", parameters=parameters, normalization_calls=calls,
        shared_fine_statistics_exact_at_four_scales=True, all_initial_tensors_exact=True,
        constructor_cpu_python_numpy_rng_exact=True, initialized_prediction_max_abs_difference=difference,
        finite_loss_and_gradients=True, development_weights_discarded=True, test_evaluations=0))


def audit():
    import numpy as np
    from tmp import review_mafirst_future_hidden_only_five_datasets_full100_20261109 as review

    review.main("ETTh1")
    baseline = common.read(BASELINE / "output/official_test_result.json")
    result = common.read(OUTPUT / "official_test_result.json")
    assert common.sha(OUTPUT / "test/truth.npy") == result["truth_sha256"] == baseline["truth_sha256"]
    original = common.read(BASELINE / "output/curves.json")
    curves = common.read(OUTPUT / "curves.json")
    assert len(curves) == len(original) == 100
    hashes = {}
    for row in curves:
        saved = common.read(OUTPUT / "selection_validation" / f"epoch{row['epoch']:03d}.json")
        digest = hashlib.sha256(np.asarray(saved["indices"], dtype="<i8").tobytes()).hexdigest()
        assert digest == saved["indices_sha256"] == row["val"]["indices_sha256"]
        hashes[str(row["epoch"])] = digest
    schedule = common.read(OUTPUT / "full_decay_completion_audit.json")
    assert (schedule["source_epochs"], schedule["total_epochs"], schedule["total_updates"]) == (0, 100, 6100)
    common.atomic(GROUP / "paired_audit.json", dict(
        status="verified", baseline=baseline, experiment=result,
        baseline_full_validation=common.read(BASELINE / "output/completion.json")["validation"],
        experiment_full_validation=common.read(OUTPUT / "completion.json")["validation"],
        actual_validation_indices_sha256=hashes,
        matching_training_orders=sum(a["order_sha256"] == b["order_sha256"] for a, b in zip(curves, original)),
        matching_selection_indices=sum(a["val"]["indices_sha256"] == b["val"]["indices_sha256"]
                                       for a, b in zip(curves, original)),
        test_delta={metric: result[metric] - baseline[metric] for metric in ("mse", "mae")},
        independent_review=common.read(GROUP / "offline_review.json"), full_decay_audit=schedule))


def stage(name, *arguments, environment=None):
    status(name)
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if environment is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = environment
        common.run(name, [common.PY, "-B", "-u", str(Path(__file__).resolve()), *arguments], root=GROUP)
    finally:
        if environment is not None:
            if previous is None:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous


def wait_predecessor():
    identity = common.read(PREDECESSOR / "launch.json")["coordinator"]
    status("waiting_predecessor", predecessor=identity, cpu_route_verified=True)
    while common.alive(identity):
        time.sleep(30)
    prior_status = common.read(PREDECESSOR / "status.json")["status"]
    assert prior_status in ("completed", "completed_with_resource_blocks"), prior_status
    result = common.read(PREDECESSOR / "results.json")
    assert result["status"] in ("verified", "verified_with_resource_blocks")
    assert len(result["completed"]) + len(result["blocked"]) == 20
    write("predecessor_receipt", dict(identity=identity, prior_status=prior_status,
                                      result_sha256=common.sha(PREDECESSOR / "results.json")))


def wait_resources():
    query = ["--format=csv,noheader,nounits"]
    required = 30 * 2**30 + 8 * common.EVAL_WINDOWS * common.HORIZON * common.CHANNELS
    while True:
        memory = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", *query],
                                             text=True).splitlines()[0])
        jobs = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", *query], text=True).strip()
        disk = min(shutil.disk_usage(REPO).free, shutil.disk_usage("/mnt/d").free)
        if not jobs and memory >= 2500 and disk >= required:
            return
        status("waiting_resources", gpu_free_mib=memory, compute_pids=jobs,
               disk_free_bytes=disk, required_disk_bytes=required)
        time.sleep(30)


def supervise():
    with (CASE / "supervisor_once.json").open("x") as stream:
        json.dump(common.identity(), stream)
    try:
        common.guard()
        stage("CPU_route", "cpu", environment="")
        wait_predecessor()
        wait_resources()
        stage("CUDA_route", "cuda")
        stage("actual128_memory", "train", "--phase", "memory")
        preflight = GROUP / "preflight" / ARM
        for name, destination, extra in (("continuous", "continuous", ()),
                                         ("stop1", "resumed", ("--stop-after", "1")),
                                         ("resume", "resumed", ("--resume",))):
            stage("preflight_" + name, "train", "--phase", "preflight",
                  "--output", str(preflight / destination), *extra)
        stage("preflight_saved_exact", "preflight_verify")
        stage("formal_train", "train", "--phase", "full")
        stage("official_test", "test")
        stage("saved_review", "audit")
        exits = {path.parent.name: common.read(path) for path in (GROUP / "execution").glob("*/log.exit.json")}
        assert len(exits) == 10 and all(item["real_wait"] and item["exit_code"] == 0 for item in exits.values())
        common.guard()
        write("results", dict(status="verified", stages=exits, result=common.read(GROUP / "paired_audit.json")))
        status("completed")
    except BaseException as error:
        write("failure", dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        status("failed", error=repr(error))
        raise


def launch():
    assert common.read(CASE / "status.json")["status"] == "prepared"
    with (CASE / "launch_once.json").open("x") as stream:
        json.dump(common.identity(), stream)
    command = [common.PY, "-B", "-u", str(Path(__file__).resolve()), "supervise"]
    with (CASE / "supervisor.log").open("xb") as log:
        process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError((CASE / "supervisor.log").read_text())
        try:
            identity = common.identity(process.pid)
            if identity["command"] == command:
                break
        except FileNotFoundError:
            pass
        time.sleep(.02)
    else:
        raise RuntimeError("Supervisor identity could not be established")
    write("launch", dict(supervisor=identity, source_sha256=common.sha(Path(__file__)), created=common.now()))
    print(json.dumps(common.read(CASE / "launch.json")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "launch", "supervise", "cpu", "cuda",
                                        "train", "preflight_verify", "test", "audit"))
    args, trailing = parser.parse_known_args()
    if args.mode in ("prepare", "launch", "supervise", "audit"):
        globals()[args.mode]()
    elif args.mode in ("cpu", "cuda"):
        verify(args.mode)
    elif args.mode == "preflight_verify":
        from tmp import verify_mafirst_future_hidden_only_five_datasets_full100_20261109_preflight
    elif args.mode == "train":
        install_factory()
        from tmp import run_mafirst_future_hidden_only_five_datasets_full100_20261109 as runner
        sys.argv = [sys.argv[0], "--arm", ARM, *trailing]
        runner.main()
    else:
        install_factory()
        from tmp import test_mafirst_future_hidden_only_five_datasets_full100_20261109 as tester
        tester.main("ETTh1")


if __name__ == "__main__":
    main()
