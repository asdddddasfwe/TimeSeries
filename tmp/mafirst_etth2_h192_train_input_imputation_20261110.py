"""ETTh2-H192 training-only input imputation with original labels and full evaluation."""
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
    check_stage_waits, until,
)

common = original.common
repo = original.REPO
root = repo / "tmp/mafirst_etth2_h192_train_input_imputation_20261110"
group = root / "group"
output = root / "output"
arm = original.MODEL_ARM
mask_path = group / "train_anomaly_flags.npy"
indices_path = group / "train_window_ids.npy"
filled_path = group / "imputed_train_inputs.npy"
train_count = 5364
updates_per_epoch = train_count // 128


def activate():
    common.G = common.B = group
    common.OUT = {arm: output}
    common.TAG = "future_hidden_fixed_cosine_imputed_train_history"


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
    import pandas as pd
    from tmp.etth2_h192_training_gap_imputation_20261110 import evaluate_and_fill

    previous = repo / "tmp/mafirst_etth2_h192_train_only_exclusion_20261110"
    source = previous / "group"
    registered = common.read(source / "registration.json")["hashes"]
    for filename in (Path("train_anomaly_flags.npy"), Path("train_window_ids.npy"), Path("plan.json"), original.REF):
        path = source / filename
        assert common.sha(path) == registered[path.relative_to(repo).as_posix()]
    mask = np.load(source / "train_anomaly_flags.npy", allow_pickle=False)
    assert mask.shape == (8640, 7) and mask.dtype == np.bool_
    prior_ids = np.load(source / "train_window_ids.npy", allow_pickle=False)
    ids = target_valid_indices(mask)
    assert len(ids) == train_count and len(prior_ids) == 4704
    assert len(np.setdiff1d(ids, prior_ids)) == 660 and np.isin(prior_ids, ids).all()
    values, dates = original.source_frame(common.TRAIN_END)
    plan = common.read(source / "plan.json")
    cfg = plan["models"][arm]["config"]
    frozen_raw = pd.read_csv(Path(cfg["root_path"]) / cfg["data_path"], nrows=common.TRAIN_END)
    assert np.array_equal(values, frozen_raw.iloc[:, 1:].to_numpy())
    assert dates.equals(pd.to_datetime(frozen_raw.date))
    filled, diagnostic = evaluate_and_fill(values, mask, dates.dt.hour.to_numpy())
    assert np.array_equal(values[~mask], filled[~mask])
    baseline = common.read(original.BASE / "output/official_test_result.json")
    strict = common.read(previous / "output/official_test_result.json")
    assert baseline["truth_sha256"] == strict["truth_sha256"] and baseline["windows"] == strict["windows"] == 2689
    root.mkdir(exist_ok=False)
    (group / original.REF).parent.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / original.REF, group / original.REF)
    assert common.sha(group / original.REF) == plan["models"][arm]["reference_sha256"]
    np.save(mask_path, mask, allow_pickle=False)
    np.save(indices_path, ids, allow_pickle=False)
    np.save(filled_path, filled, allow_pickle=False)
    detection = plan.pop("window_exclusion")["detection"]
    assert detection["lof_enabled"] is False
    plan.update(train_windows=train_count, updates_per_epoch=updates_per_epoch, total_updates=100*updates_per_epoch)
    plan["models"][arm]["config"].update(log_dir=str(group / "logs"), tensorboard_dir=str(group / "tensorboard"))
    plan["training_imputation"] = dict(detection=detection, affected_splits=["train"], retained_original_labels=True,
        reject_windows_with_any_flagged_forecast_target=True, training_fit_only=True, no_test_or_validation_in_imputer=True,
        other_observed_channels_unchanged=True, standard_scaler="original raw training prefix",
        choices=diagnostic["decisions"], validation="unfiltered original split", test="unfiltered original split")
    frozen = common.read(source / "frozen_sources.json")
    script = Path(__file__)
    for path in (script, script.with_name("etth2_h192_training_gap_imputation_20261110.py")):
        frozen[path.relative_to(repo).as_posix()] = common.sha(path)
    protected = common.read(source / "protected_records.json")
    for path in (source / "train_anomaly_flags.npy", source / "train_window_ids.npy", source / "plan.json",
                 previous / "output/official_test_result.json", previous / "delivery_verification.json"):
        protected[path.relative_to(repo).as_posix()] = common.sha(path)
    activate()
    for name,data in (("plan",plan), ("expected_model_cards",common.read(source/"expected_model_cards.json")),
                      ("frozen_sources",frozen), ("protected_records",protected), ("imputation_diagnostics",diagnostic)):
        common.atomic(group / (name + ".json"),data)
    sealed = [group / original.REF, mask_path, indices_path, filled_path,
              *(group / (n + ".json") for n in ("plan","expected_model_cards","frozen_sources","protected_records","imputation_diagnostics"))]
    common.atomic(group / "registration.json",dict(hashes={p.relative_to(repo).as_posix():common.sha(p) for p in sealed}))
    recovered = np.setdiff1d(ids,prior_ids)
    touched = np.zeros(len(values), dtype=bool)
    for start in recovered:
        touched[start:start+96] = True
    write("preparation",dict(status="verified", train_windows=train_count, recovered_windows=len(recovered),
        val_windows=common.VAL_WINDOWS, test_windows=common.EVAL_WINDOWS, training_updates=100*updates_per_epoch,
        flagged_cells=int(mask.sum()), filled_cells_changed=int(np.count_nonzero(values!=filled)),
        changed_cells_in_recovered_histories=int(np.count_nonzero((values!=filled)&touched[:,None])),
        choices=diagnostic["decisions"], baseline=baseline, strict_exclusion=strict, evaluation_data_not_opened=True))
    status("prepared")
    print(json.dumps(common.read(root/"preparation.json")),flush=True)


def target_valid_indices(mask):
    import numpy as np

    bad = np.r_[0,mask.any(axis=1).cumsum(dtype=np.int64)]
    indices = np.arange(len(mask)-96-common.HORIZON+1,dtype=np.int64)
    return indices[bad[indices+96+common.HORIZON]==bad[indices+96]]


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
        values = np.load(filled_path, allow_pickle=False)
        assert values.shape == ds.data_x.shape == ds.data_y.shape
        ds.data_x = ds.scaler.transform(values)
        if limit:
            ids = ids[np.linspace(0,len(ids)-1,min(limit,len(ids)),dtype=np.int64)]
        return Subset(ds,ids.tolist())

    factory.dataset = dataset
    return factory,native


def verify(device):
    import numpy as np
    import torch

    common.guard()
    factory, native = install_loader()
    from tmp import run_mafirst_etth2_h192_window_exclusion_20261110 as runner

    runner.configure(device)
    ds = factory.dataset(arm, "train")
    source = native(arm, "train")
    mask = np.load(mask_path, allow_pickle=False)
    ids = target_valid_indices(mask)
    assert np.array_equal(ds.dataset.data_y,source.data_y)
    assert np.array_equal(ds.dataset.data_stamp,source.data_stamp)
    assert np.array_equal(ds.dataset.data_x[~mask],source.data_x[~mask])
    assert np.any(ds.dataset.data_x[mask] != source.data_x[mask])
    assert np.array_equal(ds.indices, ids)
    for i in np.linspace(0, len(ds)-1, 31, dtype=int):
        repaired, *original_parts = ds[int(i)]
        raw, *expected_parts = source[int(ids[i])]
        start = int(ids[i])
        flags = mask[start:start+96]
        assert np.array_equal(repaired[~flags],raw[~flags])
        assert all(np.array_equal(a,b) for a,b in zip(original_parts,expected_parts))
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
        no_imputed_supervision=True, original_observed_inputs_exact=True, validation_bitwise_equal=True,
        actual_batch=128, pilot_updates=2, discarded_weights=True, losses=losses))


def audit():
    from tmp import review_mafirst_future_hidden_only_five_datasets_full100_20261109 as review

    review.UPDATES_PER_EPOCH = updates_per_epoch
    review.main("ETTh2")
    previous = repo / "tmp/mafirst_etth2_h192_train_only_exclusion_20261110"
    baseline = common.read(original.BASE / "output/official_test_result.json")
    strict = common.read(previous / "output/official_test_result.json")
    candidate = common.read(output / "official_test_result.json")
    assert candidate["truth_sha256"] == baseline["truth_sha256"] == strict["truth_sha256"]
    assert candidate["windows"] == baseline["windows"] == strict["windows"] == common.EVAL_WINDOWS
    curves = common.read(output / "curves.json")
    assert len(curves) == 100 and curves[-1]["updates"] == 4100
    comparisons = {}
    for name,folder,reference in (("unmodified",original.BASE,baseline),("strict_exclusion",previous,strict)):
        history = common.read(folder / "output/curves.json")
        assert len(history) == 100
        comparisons[name] = dict(reference=reference,
            mse_delta=candidate["mse"]-reference["mse"], mae_delta=candidate["mae"]-reference["mae"],
            validation_index_sha_matches=sum(x["val"]["indices_sha256"]==y["val"]["indices_sha256"]
                                             for x,y in zip(curves,history)))
    common.atomic(group / "paired_audit.json",dict(status="verified", result=candidate, comparisons=comparisons,
        common_full_test_truth_sha=True, no_imputed_supervision=True,
        saved_only_review=common.read(group / "offline_review.json"),
        imputation_diagnostics_sha256=common.sha(group / "imputation_diagnostics.json")))


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
    predecessor = common.read(original.PREDECESSOR / "launch.json")["coordinator"]
    with (root / "supervisor_once.json").open("x") as f:
        json.dump(common.identity(),f)
    write("launch",dict(supervisor=common.identity(),predecessor=predecessor,
         source_sha256=common.sha(Path(__file__)),created=common.now()))
    status("waiting_five_dataset_queue", predecessor=predecessor, cpu_pilots_deferred=True)
    try:
        while common.alive(predecessor):
            time.sleep(30)
        prior = common.read(original.PREDECESSOR / "status.json")
        assert prior["status"] in ("completed","completed_with_resource_blocks"),prior
        record = common.read(original.PREDECESSOR / "results.json")
        assert record["status"] in ("verified","verified_with_resource_blocks")
        assert len(record["completed"]) + len(record["blocked"]) == 20
        write("predecessor_receipt",dict(original_supervisor_inactive=True,status=prior["status"],
              results_sha256=common.sha(original.PREDECESSOR/"results.json")))
        experiment()
        status("completed")
    except BaseException as error:
        write("failure",dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        status("failed",error=repr(error))
        raise


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
