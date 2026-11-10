"""Extend the registered ETTh2 input-only correction to H96, H336 and H720."""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[1]
sys.path.insert(0, str(REPO))
from tmp import mafirst_etth2_h192_boundary_imputation_queue_20261110 as boundary

common = boundary.common
history = boundary.history
ARM = boundary.arm
ROOT = REPO / "tmp/mafirst_etth2_boundary_input_other_horizons_20261111"
BASE = boundary.original.PREDECESSOR / "etth2"
SOURCE = boundary.roots["input"]
HORIZONS = (96, 336, 720)
MASK = Path("train_anomaly_flags.npy")
VALUES = Path("boundary_values.npz")
IDS = Path("train_window_ids.npy")
REF = boundary.original.REF


def write(path, data):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)
    assert common.read(path) == data


def configure(horizon, prepared):
    os.environ["MAFIRST_FUTURE_HIDDEN_ONLY_HORIZON_20261107"] = str(horizon)
    importlib.reload(common)
    original = BASE / f"h{horizon}"
    assert common.G == original / "group" and common.HORIZON == horizon
    if not prepared:
        common.guard()
    root = ROOT / f"h{horizon}"
    group = root / "group"
    out = root / "output"
    common.G = common.B = group
    common.OUT = {ARM: out}
    common.TAG = "future_hidden_fixed_cosine_boundary_corrected_input"
    boundary.variant, boundary.group = "input", group
    history.root, history.group, history.output = root, group, out
    if prepared:
        bind_counts(common.read(group / "plan.json")["train_windows"])
        common.guard()
    return original, root, group, out


def bind_counts(count):
    boundary.train_count = history.train_count = count
    boundary.updates = history.updates_per_epoch = count // 128


def prepare():
    ROOT.mkdir(exist_ok=False)
    source_group = SOURCE / "group"
    sealed = common.read(source_group / "registration.json")["hashes"]
    for path, digest in sealed.items():
        assert common.sha(REPO / path) == digest, path
    mask = np.load(source_group / MASK, allow_pickle=False)
    assert mask.shape == (8640, 7) and mask.dtype == np.bool_
    assert np.array_equal(history.target_valid_indices(mask), np.load(source_group / IDS, allow_pickle=False))
    source_plan = common.read(source_group / "plan.json")
    counts = {}
    for horizon in HORIZONS:
        original, root, group, _ = configure(horizon, prepared=False)
        original_group = original / "group"
        plan = common.read(original_group / "plan.json")
        assert plan["train_end"] == len(mask) and plan["train_val_prefix_sha256"] == source_plan["train_val_prefix_sha256"]
        ids = history.target_valid_indices(mask)
        count = len(ids)
        assert ids.dtype == np.int64 and count >= 512 and count < plan["train_windows"]
        plan.update(train_windows=count, updates_per_epoch=count // 128, total_updates=count // 128 * 100,
                    training_imputation=source_plan["training_imputation"])
        plan["training_imputation"]["detection"]["original_case"] = str(original)
        plan["models"][ARM]["config"].update(log_dir=str(group/"logs"), tensorboard_dir=str(group/"tensorboard"))
        group.mkdir(parents=True)
        (group / REF).parent.mkdir()
        copies = (REF, Path("expected_model_cards.json"), MASK, VALUES)
        for name in copies:
            parent = source_group if name in (MASK, VALUES) else original_group
            shutil.copyfile(parent / name, group / name)
            assert common.sha(parent / name) == common.sha(group / name)
        np.save(group / IDS, ids, allow_pickle=False)
        frozen = common.read(original_group / "frozen_sources.json")
        for file in (SCRIPT, Path(boundary.__file__), Path(history.__file__)):
            frozen[file.relative_to(REPO).as_posix()] = common.sha(file)
        protected = common.read(original_group / "protected_records.json")
        protected.update(sealed)
        for file in (original_group/"registration.json", original/"output/official_test_result.json"):
            protected[file.relative_to(REPO).as_posix()] = common.sha(file)
        for name, data in (("plan", plan), ("protected_records", protected), ("frozen_sources", frozen)):
            common.atomic(group / f"{name}.json", data)
        files = (*copies, IDS, Path("plan.json"), Path("protected_records.json"), Path("frozen_sources.json"))
        common.atomic(group/"registration.json", dict(hashes={
            (group / p).relative_to(REPO).as_posix(): common.sha(group / p) for p in files}))
        common.guard()
        counts[str(horizon)] = dict(train_windows=count, val_windows=common.VAL_WINDOWS,
                                    test_windows=common.EVAL_WINDOWS, updates_per_epoch=count//128,
                                    total_updates=count//128*100)
    write(ROOT/"preparation.json", dict(status="verified", source=str(SOURCE), original_h192_indices_match=True,
                                       source_mask_sha256=common.sha(source_group/MASK),
                                       source_values_sha256=common.sha(source_group/VALUES), cases=counts))
    print(json.dumps(counts), flush=True)


def stage(name, *args, cpu=False):
    horizon = str(common.HORIZON)
    history.status(name)
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        common.run(name, [common.PY, "-B", "-u", str(SCRIPT), "stage", horizon, *args], root=common.G)
    finally:
        if cpu and old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        elif cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = old


def audit(original, group, output):
    from tmp import review_mafirst_future_hidden_only_five_datasets_full100_20261109 as review

    review.UPDATES_PER_EPOCH = history.updates_per_epoch
    review.main("ETTh2")
    base = common.read(original/"output/official_test_result.json")
    result = common.read(output/"official_test_result.json")
    assert result["truth_sha256"] == base["truth_sha256"]
    assert result["windows"] == base["windows"] == common.EVAL_WINDOWS
    ours, ref = (common.read(p/"curves.json") for p in (output, original/"output"))
    assert len(ours) == len(ref) == 100
    common.atomic(group/"paired_audit.json", dict(status="verified", result=result, baseline=base,
        mse_delta=result["mse"]-base["mse"], mae_delta=result["mae"]-base["mae"],
        validation_index_sha_matches=sum(a["val"]["indices_sha256"] == b["val"]["indices_sha256"] for a,b in zip(ours,ref)),
        train_order_sha_matches=sum(a["order_sha256"] == b["order_sha256"] for a,b in zip(ours,ref)),
        common_full_test_truth_sha=True, saved_only_review=common.read(group/"offline_review.json")))


def run_stage(horizon, mode, args):
    original, _, group, output = configure(horizon, prepared=True)
    if mode in ("cpu", "cuda"):
        boundary.verify(mode)
    elif mode == "train":
        boundary.install_loader()
        from tmp import run_mafirst_future_hidden_only_five_datasets_full100_20261109 as runner

        runner.TRAIN_WINDOWS = history.train_count
        runner.UPDATES_PER_EPOCH = history.updates_per_epoch
        sys.argv = [str(SCRIPT), "--arm", ARM, *args]
        runner.main()
    elif mode == "preflight_verify":
        from tmp import verify_mafirst_future_hidden_only_five_datasets_full100_20261109_preflight
    elif mode == "test":
        from tmp import test_mafirst_future_hidden_only_five_datasets_full100_20261109 as tester

        tester.main("ETTh2")
    elif mode == "audit":
        audit(original, group, output)
    else:
        raise ValueError(mode)


def supervise():
    results = {}
    failures = {}
    for horizon in HORIZONS:
        _, root, _, _ = configure(horizon, prepared=True)
        write(ROOT/"status.json", dict(status="running", horizon=horizon, process=common.identity(), updated=common.now()))
        history.stage = stage
        try:
            history.experiment()
            result = common.read(root/"results.json")
            assert result["status"] == "verified"
            history.status("completed")
            results[str(horizon)] = result
        except Exception as error:
            history.write("failure", dict(error=repr(error), phase=common.read(root/"status.json")["status"],
                                         traceback=traceback.format_exc(), automatic_retry=False))
            failures[str(horizon)] = common.read(root/"failure.json")
            history.status("failed")
        write(ROOT/"progress.json", dict(completed=results, failed=failures, updated=common.now()))
    write(ROOT/"status.json", dict(status="completed_with_failures" if failures else "completed",
                                  completed=list(results), failed=list(failures), updated=common.now()))


def launch():
    assert common.read(ROOT/"preparation.json")["status"] == "verified"
    with (ROOT/"supervisor.log").open("xb") as log:
        child = subprocess.Popen([common.PY,"-B","-u",str(SCRIPT),"supervise"], cwd=REPO,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    history.until(lambda: (ROOT/"status.json").is_file() or child.poll() is not None, 30, "supervisor did not start")
    assert child.poll() is None, (ROOT/"supervisor.log").read_text()
    record = common.read(ROOT/"status.json")["process"]
    assert record == common.identity(child.pid)
    write(ROOT/"launch.json", dict(supervisor=record, script_sha256=common.sha(SCRIPT), created=common.now()))
    print(json.dumps(common.read(ROOT/"launch.json")), flush=True)


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "stage":
        run_stage(int(args[0]), args[1], args[2:])
    else:
        {"prepare": prepare, "launch": launch, "supervise": supervise}[mode]()
