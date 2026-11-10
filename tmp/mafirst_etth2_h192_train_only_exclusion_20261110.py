"""ETTh2-H192 rules-only training-window exclusion with full validation and test."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tmp import mafirst_etth2_h192_window_exclusion_20261110 as original
from tmp.mafirst_etth2_h192_window_exclusion_priority_20261110 import (
    check_stage_waits, child_ids, fields, inspect, matches_arm, until,
)

common = original.common
repo = original.REPO
root = repo / "tmp/mafirst_etth2_h192_train_only_exclusion_20261110"
group = root / "group"
output = root / "output"
arm = original.MODEL_ARM
mask_path = group / "train_anomaly_flags.npy"
indices_path = group / "train_window_ids.npy"
train_count = 4704
updates_per_epoch = train_count // 128


def activate():
    common.G = common.B = group
    common.OUT = {arm: output}
    common.TAG = "future_hidden_fixed_cosine_rules_exclusion_train_only"


def write(name, record):
    path = root / (name + ".json")
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)
    assert common.read(path) == record


def status(value, **details):
    write("status", dict(status=value, updated=common.now(), **details))


def prepare():
    import numpy as np

    source = original.ROOT / "rules/group"
    registered = common.read(source / "registration.json")["hashes"]
    for filename in (original.MASK, Path("train_window_ids.npy"), Path("plan.json"), original.REF):
        path = source / filename
        assert common.sha(path) == registered[path.relative_to(repo).as_posix()]
    mask = np.load(source / original.MASK, allow_pickle=False)[:common.TRAIN_END]
    expected = np.load(source / "train_window_ids.npy", allow_pickle=False)
    ids = original.eligible_starts(mask, 0, common.TRAIN_END)
    assert mask.shape == (8640, 7) and mask.dtype == np.bool_
    assert ids.dtype == np.int64 and len(ids) == train_count and np.array_equal(expected, ids)
    baseline = common.read(original.BASE / "output/official_test_result.json")
    assert baseline["official_test_evaluations"] == 1 and baseline["windows"] == common.EVAL_WINDOWS
    root.mkdir(exist_ok=False)
    (group / original.REF).parent.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / original.REF, group / original.REF)
    np.save(mask_path, mask, allow_pickle=False)
    np.save(indices_path, ids, allow_pickle=False)
    plan = common.read(source / "plan.json")
    detection = plan.pop("window_exclusion")["detection"]
    assert detection["lof_enabled"] is False and plan["train_windows"] == train_count
    plan.update(val_windows=common.VAL_WINDOWS, test_windows=common.EVAL_WINDOWS,
                updates_per_epoch=updates_per_epoch, total_updates=100 * updates_per_epoch)
    plan["models"][arm]["config"].update(log_dir=str(group / "logs"), tensorboard_dir=str(group / "tensorboard"))
    plan["window_exclusion"] = dict(detection=detection, affected_splits=["train"],
        any_channel_excludes_entire_hour=True, train_requires_all_96_input_and_192_target_hours=True,
        imputation=False, standard_scaler="original full training prefix",
        validation="unfiltered original split", test="unfiltered original split")
    frozen = common.read(source / "frozen_sources.json")
    script = Path(__file__)
    helper = script.with_name("mafirst_etth2_h192_window_exclusion_priority_20261110.py")
    for path in (script, helper):
        frozen[path.relative_to(repo).as_posix()] = common.sha(path)
    protected = common.read(source / "protected_records.json")
    for path in (source / original.MASK, source / "train_window_ids.npy", source / "plan.json"):
        protected[path.relative_to(repo).as_posix()] = common.sha(path)
    activate()
    for name, data in (("plan", plan), ("expected_model_cards", common.read(source / "expected_model_cards.json")),
                       ("frozen_sources", frozen), ("protected_records", protected)):
        common.atomic(group / (name + ".json"), data)
    sealed = [group / original.REF, mask_path, indices_path,
              *(group / (n + ".json") for n in ("plan","expected_model_cards","frozen_sources","protected_records"))]
    common.atomic(group / "registration.json", dict(hashes={
        p.relative_to(repo).as_posix(): common.sha(p) for p in sealed}))
    write("preparation", dict(status="verified", training_mask_train_prefix_only=True,
          removed_train_hours=int(mask.any(1).sum()), train_windows=train_count,
          val_windows=common.VAL_WINDOWS, test_windows=common.EVAL_WINDOWS,
          training_updates=100 * updates_per_epoch, baseline=baseline, test_not_opened=True))
    status("prepared")
    print(json.dumps(common.read(root / "preparation.json")), flush=True)


def install_loader():
    import numpy as np
    from torch.utils.data import Subset
    from tmp import mafirst_future_hidden_only_five_datasets_full100_20261109_factory as factory

    native = factory.dataset

    def dataset(model_arm, split, limit=0):
        if split != "train":
            return native(model_arm, split, limit)
        ds = native(model_arm, split)
        ids = np.load(indices_path, allow_pickle=False)
        assert len(ids) == train_count and ids.dtype == np.int64
        assert ids[0] >= 0 and ids[-1] < len(ds) and np.all(np.diff(ids) > 0)
        if limit:
            ids = ids[np.linspace(0, len(ids)-1, min(limit, len(ids)), dtype=np.int64)]
        return Subset(ds, ids.tolist())

    factory.dataset = dataset
    return factory, native


def verify(device):
    import numpy as np
    import torch

    common.guard()
    factory, native = install_loader()
    from tmp import run_mafirst_etth2_h192_window_exclusion_20261110 as runner

    runner.configure(device)
    ds = factory.dataset(arm, "train")
    source = native(arm, "train")
    ids = original.eligible_starts(np.load(mask_path, allow_pickle=False), 0, common.TRAIN_END)
    assert np.array_equal(ds.indices, ids)
    for i in np.linspace(0, len(ds)-1, 31, dtype=int):
        assert all(np.array_equal(a,b) for a,b in zip(ds[int(i)], source[int(ids[i])]))
    validation = factory.dataset(arm, "val")
    raw_validation = native(arm, "val")
    assert len(validation) == len(raw_validation) == common.VAL_WINDOWS
    for key in ("data_x", "data_y", "data_stamp"):
        assert np.array_equal(getattr(validation,key), getattr(raw_validation,key))
    model, _ = factory.fresh(arm)
    model.to(device).train()
    opt = factory.optimizer(model)
    scheduler = factory.scheduler(opt, updates_per_epoch)
    losses = []
    for step, ids in enumerate(np.array_split(np.linspace(0, len(ds)-1, 256, dtype=int), 2), 1):
        batch = [ds[int(i)] for i in ids]
        x = torch.from_numpy(np.stack([row[0] for row in batch]))
        y = torch.from_numpy(np.stack([row[1] for row in batch]))
        terms, _, _ = runner.backward_step(arm, model, opt, scheduler, x, y, device, 1, step)
        runner.finite(model, opt)
        losses.append(terms.tolist())
    common.atomic(group / (device + "_verification.json"), dict(status="verified",
        raw_retained_training_windows=True, validation_bitwise_equal=True,
        actual_batch=128, pilot_updates=2, discarded_weights=True, losses=losses))


def audit():
    import numpy as np
    from tmp import review_mafirst_future_hidden_only_five_datasets_full100_20261109 as review

    review.UPDATES_PER_EPOCH = updates_per_epoch
    review.main("ETTh2")
    baseline = common.read(original.BASE / "output/official_test_result.json")
    candidate = common.read(output / "official_test_result.json")
    assert candidate["truth_sha256"] == baseline["truth_sha256"]
    assert candidate["windows"] == baseline["windows"] == common.EVAL_WINDOWS
    curves = common.read(output / "curves.json")
    baseline_curves = common.read(original.BASE / "output/curves.json")
    assert len(curves) == len(baseline_curves) == 100 and curves[-1]["updates"] == 3600
    record = dict(status="verified", result=candidate, baseline=baseline, common_full_test_truth_sha=True,
        mse_delta=candidate["mse"]-baseline["mse"], mae_delta=candidate["mae"]-baseline["mae"],
        validation_index_sha_matches=sum(x["val"]["indices_sha256"]==y["val"]["indices_sha256"]
                                         for x,y in zip(curves,baseline_curves)),
        validation_rng_trajectory_may_differ_due_to_fewer_training_batches=True,
        saved_only_review=common.read(group / "offline_review.json"))
    prior = original.ROOT / "rules/group/test_window_ids.npy"
    if prior.is_file() and (original.ROOT / "rules/group/paired_audit.json").is_file():
        ids = np.load(prior, allow_pickle=False)
        p = np.load(output / "test/prediction.npy", mmap_mode="r")[ids].astype(np.float64)
        y = np.load(output / "test/truth.npy", mmap_mode="r")[ids].astype(np.float64)
        filtered = common.read(original.ROOT / "rules/group/paired_audit.json")["result"]
        prior_audit = common.read(original.ROOT / "rules/group/paired_audit.json")
        assert len(ids)==filtered["windows"] and common.sha(prior)==prior_audit["test_ids_file_sha256"]
        filtered_truth = np.load(original.ROOT / "rules/output/test/truth.npy", mmap_mode="r")
        assert np.array_equal(y, filtered_truth)
        record["prior_rules_test_subset"] = dict(windows=len(ids), train_only_mse=float(np.square(p-y).mean()),
             train_only_mae=float(np.abs(p-y).mean()), train_and_eval_filtered=filtered)
    common.atomic(group / "paired_audit.json", record)


def stage(name, *args, cpu=False):
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        status(name)
        common.run(name, [common.PY, "-B", "-u", str(Path(__file__).resolve()), *args], root=group)
    finally:
        if cpu and env is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        elif cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = env


def wait_resources():
    while True:
        jobs = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
        memory = int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],
                                            text=True).splitlines()[0])
        disk = min(shutil.disk_usage(repo).free, shutil.disk_usage("/mnt/d").free)
        if not jobs and memory >= 2500 and disk >= 30 * 2**30 + 8 * common.EVAL_WINDOWS * common.HORIZON * common.CHANNELS:
            return
        status("waiting_resources", gpu_free_mib=memory, free_bytes=disk)
        time.sleep(30)


def experiment():
    wait_resources()
    common.guard()
    stage("CPU_route", "cpu", cpu=True)
    stage("CUDA_route", "cuda")
    stage("actual128_memory", "train", "--phase", "memory")
    pre = group / "preflight" / arm
    for label, folder, options in (("continuous","continuous",()),("stop1","resumed",("--stop-after","1")),
                                   ("resume","resumed",("--resume",))):
        stage("preflight_" + label, "train", "--phase", "preflight", "--output", str(pre/folder), *options)
    stage("preflight_saved_exact", "preflight_verify")
    stage("formal_train", "train", "--phase", "full")
    stage("official_test", "test")
    stage("saved_review", "audit")
    common.guard()
    write("results", dict(status="verified", stages=check_stage_waits(root),
                          audit=common.read(group / "paired_audit.json")))


def supervise():
    from select import poll, POLLIN

    prior = original.ROOT
    previous = common.read(prior / "priority_launch.json")["supervisor"]
    dispatcher = common.read(original.PREDECESSOR / "launch.json")["coordinator"]
    boundary = common.read(prior / "priority_boundary_receipt.json")
    assert boundary["dispatcher"] == dispatcher and boundary["draining_case"] is None
    assert inspect(dispatcher) in ("T","t") and inspect(previous) not in ("T","t","Z")
    with (root / "supervisor_once.json").open("x") as f:
        json.dump(common.identity(), f)
    write("launch", dict(supervisor=common.identity(), previous=previous, dispatcher=dispatcher,
                         source_sha256=common.sha(Path(__file__)), created=common.now()))
    status("waiting_two_exclusion_groups", previous=previous, dispatcher=dispatcher)
    stopped = False
    draining = None
    fd = os.pidfd_open(dispatcher["pid"])
    prior_fd = os.pidfd_open(previous["pid"])
    try:
        tail = prior / "rules_lof/group/execution/saved_review/log.exit.json"
        failure = prior / "priority_failure.json"
        resume = prior / "priority_resume_receipt.json"
        while not (tail.is_file() or failure.is_file() or resume.is_file()):
            assert common.alive(previous), "two-group supervisor disappeared before releasing dispatcher"
            time.sleep(.1)
        # The earlier supervisor publishes its resume receipt without taking this
        # lock. The five-dataset dispatcher must acquire it before publishing a new arm.
        with common.locked():
            until(resume.is_file, 180, "previous supervisor did not release dispatcher")
            release = common.read(resume)
            assert release["dispatcher"] == dispatcher and release["resumed"]
            assert inspect(dispatcher) not in ("T","t","Z")
            signal.pidfd_send_signal(fd, signal.SIGSTOP)
            stopped = True
            until(lambda: inspect(dispatcher) in ("T","t"), 5, "dispatcher did not pause")
            current = common.read(original.PREDECESSOR / "status.json")
            assert current["status"]=="running" and current["process"]==dispatcher
            ids = child_ids(dispatcher)
            if ids:
                child, = ids
                pid = int(child)
                dataset, horizon = current["dataset"], current["horizon"]
                folder = original.PREDECESSOR / dataset.lower() / f"h{horizon}"
                until(lambda: common.identity(pid)["command"]==dispatcher["command"][:-1]
                      + ["case","--dataset",dataset,"--horizon",str(horizon)]
                      or fields(pid)[0]=="Z", 5, "could not identify dispatched case")
                identity = common.identity(pid)
                matches_arm(identity,pid,dispatcher,dataset,horizon)
                draining = (identity,folder,dataset,horizon)
            write("priority_boundary_receipt", dict(dispatcher=dispatcher, previous_release=release,
                  draining_case=list(draining[2:]) if draining else None, scientific_processes_signalled=False))
        previous_exit = poll()
        previous_exit.register(prior_fd, POLLIN)
        assert previous_exit.poll(120000), "two-group supervisor did not exit after release"
        if not failure.exists():
            assert common.read(prior / "status.json")["status"]=="completed"
            assert common.read(prior / "results.json")["status"]=="verified"
        if draining:
            identity,folder,dataset,horizon=draining
            status("waiting_already_dispatched_case", dataset=dataset, horizon=horizon)
            while True:
                assert inspect(dispatcher) in ("T","t")
                state=matches_arm(identity,identity["pid"],dispatcher,dataset,horizon)
                if state[0]=="Z":
                    break
                time.sleep(2)
            assert int(state[49])==0
            write("priority_case_drain_receipt", dict(arm=identity, kernel_exit_status=0, stages=check_stage_waits(folder)))
        experiment()
    except BaseException as error:
        write("failure", dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        status("failed", error=repr(error))
        raise
    finally:
        try:
            if stopped:
                assert inspect(dispatcher) in ("T","t")
                signal.pidfd_send_signal(fd, signal.SIGCONT)
                write("priority_resume_receipt", dict(dispatcher=dispatcher,resumed=True,created=common.now()))
        finally:
            os.close(fd)
            os.close(prior_fd)
    if draining:
        identity,folder,*_ = draining
        until(lambda:(folder/"arm_exit.json").is_file(),120,"dispatched arm was not collected")
        receipt=common.read(folder/"arm_exit.json")
        assert receipt["real_wait"] and receipt["exit_code"]==0 and not common.alive(identity)
        write("priority_case_parent_wait",receipt)
    status("completed")


def launch():
    assert common.read(root / "status.json")["status"]=="prepared"
    with (root/"launch_once.json").open("x") as f:
        json.dump(common.identity(),f)
    cmd=[sys.executable,"-B","-u",str(Path(__file__).resolve()),"supervise"]
    with (root/"supervisor.log").open("xb") as log:
        child=subprocess.Popen(cmd,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    until(lambda:(root/"launch.json").is_file() or child.poll() is not None,20,"supervisor did not register")
    if not (root/"launch.json").is_file():
        raise RuntimeError((root/"supervisor.log").read_text())
    print(json.dumps(common.read(root/"launch.json")),flush=True)


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    mode, *arguments = sys.argv[1:]
    if mode == "prepare":
        prepare()
        return
    activate()
    if mode == "launch":
        launch()
    elif mode == "supervise":
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        supervise()
    elif mode in ("cpu","cuda"):
        verify(mode)
    elif mode == "preflight_verify":
        from tmp import verify_mafirst_future_hidden_only_five_datasets_full100_20261109_preflight
    elif mode == "audit":
        audit()
    elif mode == "train":
        install_loader()
        from tmp import run_mafirst_etth2_h192_window_exclusion_20261110 as runner

        runner.TRAIN_WINDOWS = train_count
        runner.UPDATES_PER_EPOCH = updates_per_epoch
        sys.argv = [sys.argv[0], "--arm", arm, *arguments]
        runner.main()
    elif mode == "test":
        from tmp import test_mafirst_future_hidden_only_five_datasets_full100_20261109 as tester

        tester.main("ETTh2")
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
