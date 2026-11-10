"""ETTh2-H192 paired complete-window exclusion ablation."""
import argparse
from hashlib import sha256
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
os.environ.update(MAFIRST_FUTURE_HIDDEN_ONLY_DATASET_20261107="ETTh2",
                  MAFIRST_FUTURE_HIDDEN_ONLY_HORIZON_20261107="192",
                  OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MPLBACKEND="Agg")
from tmp import mafirst_future_hidden_only_five_datasets_full100_20261109_common as common

BASE = common.G.parent
PREDECESSOR = BASE.parents[1]
PRIORITY = REPO / "tmp/mafirst_etth1_h720_single_revin_20261110"
ROOT = REPO / "tmp/mafirst_etth2_h192_window_exclusion_20261110"
ARMS = ("rules", "rules_lof")
MODEL_ARM = "mafirst_season"
REF = Path("reference_initial/mafirst_season.pt")
MASK = Path("input_flags_trainval.npy")
EXPECTED_TEST_WINDOWS = {"rules": 1167, "rules_lof": 1081}


def activate(variant):
    global GROUP, OUTPUT
    GROUP, OUTPUT = ROOT / variant / "group", ROOT / variant / "output"
    common.G = common.B = GROUP
    common.OUT = {MODEL_ARM: OUTPUT}
    common.TAG = f"future_hidden_fixed_cosine_window_exclusion_{variant}"


def write(name, value):
    target = ROOT / f"{name}.json"
    temp = target.with_suffix(".pending")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(target)
    assert common.read(target) == value


def source_frame(stop):
    import pandas as pd

    source = REPO / "dataset" / common.CSV
    frame = pd.read_csv(source, nrows=stop)
    dates = pd.to_datetime(frame.pop("date"))
    assert len(frame) == stop and dates.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    return frame.to_numpy(dtype="float64"), dates


from tmp.mafirst_etth2_h192_input_cleaning_20261110 import cleaned_inputs


def eligible_starts(mask, start, stop):
    import numpy as np

    excluded = mask.any(axis=1)
    prefix = np.r_[0, excluded.cumsum(dtype=np.int64)]
    span = 96 + common.HORIZON
    candidates = np.arange(start, stop - span + 1, dtype=np.int64)
    return (candidates[prefix[candidates + span] == prefix[candidates]] - start).astype(np.int64)


def prepare():
    import numpy as np

    source = BASE / "group"
    previous = REPO / "tmp/mafirst_etth2_h192_input_cleaning_20261110"
    assert common.read(previous / "user_switch_to_window_exclusion_receipt.json")["supervisor_exited"]
    baseline = common.read(BASE / "output/official_test_result.json")
    assert baseline["epochs"] == 100 and baseline["official_test_evaluations"] == 1
    assert common.read(BASE / "arm_exit.json")["real_wait"] and common.read(BASE / "arm_exit.json")["exit_code"] == 0
    ROOT.mkdir(exist_ok=False)
    records = {}
    for name in ARMS:
        activate(name)
        old_group = previous / name / "group"
        old_mask = old_group / MASK
        old_registration = common.read(old_group / "registration.json")["hashes"]
        assert common.sha(old_mask) == old_registration[old_mask.relative_to(REPO).as_posix()]
        mask = np.load(old_mask, allow_pickle=False)
        assert mask.shape == (common.VAL_END, common.CHANNELS) and mask.dtype == np.bool_
        bounds = {"train": (0, common.TRAIN_END), "val": (common.TRAIN_END-96, common.VAL_END)}
        starts = {split: eligible_starts(mask, *pair) for split, pair in bounds.items()}
        # Verify counts independently by enumerating contiguous unflagged segments.
        for split, (start, stop) in bounds.items():
            edges = np.flatnonzero(np.diff(np.r_[False, ~mask[start:stop].any(axis=1), False]))
            lengths = edges[1::2] - edges[::2]
            assert len(starts[split]) == np.maximum(lengths - (95 + common.HORIZON), 0).sum()
        counts = {key: len(ids) for key, ids in starts.items()}
        assert counts == {"train": {"rules": 4704, "rules_lof": 3217}[name], "val": 2067}
        counts["test"] = EXPECTED_TEST_WINDOWS[name]
        steps = counts["train"] // 128
        (GROUP / REF).parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source / REF, GROUP / REF)
        shutil.copyfile(old_mask, GROUP / MASK)
        for split, ids in starts.items():
            np.save(GROUP / f"{split}_window_ids.npy", ids, allow_pickle=False)

        plan = common.read(source / "plan.json")
        assert common.sha(GROUP / REF) == plan["models"][MODEL_ARM]["reference_sha256"]
        plan.update(train_windows=counts["train"], val_windows=counts["val"], test_windows=counts["test"],
                    updates_per_epoch=steps, total_updates=100 * steps)
        plan["models"][MODEL_ARM]["config"].update(log_dir=str(GROUP / "logs"),
                                                   tensorboard_dir=str(GROUP / "tensorboard"))
        detection = common.read(old_group / "plan.json")["input_cleaning"]
        for key in ("replacement", "target_policy"):
            detection.pop(key)
        plan["window_exclusion"] = dict(detection=detection, any_channel_excludes_entire_hour=True,
            train_val_test_require_all_96_input_and_192_target_hours=True, imputation=False,
            original_csv_scaler_values_timestamps_and_split_borders=True, original_unfiltered_test_windows=2689,
            expected_test_count_from_prior_read_only_count=True, retained_test_ids_computed_after_best_freeze=True)
        frozen = common.read(source / "frozen_sources.json")
        for item in (Path(__file__), previous.with_suffix(".py"), Path(__file__).with_name("run_" + Path(__file__).name)):
            frozen[item.relative_to(REPO).as_posix()] = common.sha(item)
        protected = common.read(source / "protected_records.json")
        for item in (source / "plan.json", source / REF, BASE / "output/curves.json",
                     BASE / "output/official_test_result.json", old_mask, old_group / "plan.json",
                     previous / "user_switch_to_window_exclusion_receipt.json"):
            protected[item.relative_to(REPO).as_posix()] = common.sha(item)
        for filename, payload in (("plan", plan), ("expected_model_cards", common.read(source / "expected_model_cards.json")),
                                  ("frozen_sources", frozen), ("protected_records", protected)):
            common.atomic(GROUP / f"{filename}.json", payload)
        sealed = (REF, MASK, Path("train_window_ids.npy"), Path("val_window_ids.npy"),
                  *(Path(n + ".json") for n in ("plan", "expected_model_cards", "frozen_sources", "protected_records")))
        common.atomic(GROUP / "registration.json", dict(hashes={
            (GROUP / n).relative_to(REPO).as_posix(): common.sha(GROUP / n) for n in sealed}))
        records[name] = dict(windows=counts, updates_per_epoch=steps, updates=steps*100,
            train_excluded_hours=int(mask[:common.TRAIN_END].any(axis=1).sum()),
            val_excluded_hours=int(mask[common.TRAIN_END:].any(axis=1).sum()))
    write("preparation", dict(status="verified", records=records, baseline=baseline,
                              no_imputation=True, test_data_not_opened_by_preparation=True))
    write("status", dict(status="prepared", created=common.now()))
    print(json.dumps(records), flush=True)


def install_loader(variant):
    import numpy as np
    from torch.utils.data import Subset
    from tmp import mafirst_future_hidden_only_five_datasets_full100_20261109_factory as factory

    base_dataset = factory.dataset
    plan = common.read(GROUP / "plan.json")

    def dataset(arm, split, limit=0):
        ds = base_dataset(arm, split)
        if split == "test":
            assert (GROUP / "test_once.json").is_file()
            values, dates = source_frame(common.TEST_END)
            _, all_masks = cleaned_inputs(values, dates)
            mask = all_masks[variant]
            assert np.array_equal(mask[:common.VAL_END], np.load(GROUP / MASK, allow_pickle=False))
            ids = eligible_starts(mask, ds.official_support["input_start"], ds.official_support["end_exclusive"])
            with (GROUP / "test_window_ids.npy").open("xb") as f:
                np.save(f, ids, allow_pickle=False)
        else:
            ids = np.load(GROUP / f"{split}_window_ids.npy", allow_pickle=False)
        plan_key = {"train": "train_windows", "val": "val_windows", "test": "test_windows"}[split]
        assert ids.dtype == np.int64 and len(ids) == plan[plan_key]
        assert ids[0] >= 0 and ids[-1] < len(ds) and np.all(np.diff(ids) > 0)
        if limit:
            ids = ids[np.linspace(0, len(ids)-1, min(limit, len(ids)), dtype=np.int64)]
        selected = Subset(ds, ids.tolist())
        selected.official_support = ds.official_support
        return selected

    factory.dataset = dataset
    return factory, base_dataset


def verify(variant, device):
    import numpy as np
    import torch

    common.guard()
    torch.set_num_threads(2)
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.9)
    factory, base_dataset = install_loader(variant)
    model, _ = factory.fresh(MODEL_ARM)
    model.to(device).train()
    cleaned = factory.dataset(MODEL_ARM, "train")
    plan = common.read(GROUP / "plan.json")
    mask = np.load(GROUP / MASK, allow_pickle=False)
    for split, key in (("train", "train_windows"), ("val", "val_windows")):
        left = factory.dataset(MODEL_ARM, split)
        right = base_dataset(MODEL_ARM, split)
        support = right.official_support
        expected = eligible_starts(mask, support["input_start"], support["end_exclusive"])
        assert np.array_equal(left.indices, expected) and len(left) == plan[key]
        assert np.array_equal(left.dataset.scaler.mean_, right.scaler.mean_)
        assert np.array_equal(left.dataset.scaler.scale_, right.scaler.scale_)
        for i in np.linspace(0, len(left)-1, 31, dtype=int):
            assert all(np.array_equal(a, b) for a, b in zip(left[int(i)], right[int(expected[i])]))
    ids = np.arange(len(cleaned), dtype=np.int64)
    optimizer = factory.optimizer(model)
    losses = []
    for picked in np.array_split(ids[np.linspace(0,len(ids)-1,256,dtype=int)],2):
        assert len(picked) == 128
        samples = [cleaned[int(i)] for i in picked]
        x = torch.as_tensor(np.stack([s[0] for s in samples]), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.stack([s[1][-192:] for s in samples]), dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        assert pred.shape == y.shape == (128,192,7)
        loss, _, _ = factory.loss(MODEL_ARM, pred, y, x)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        losses.append(float(loss.detach()))
    common.atomic(GROUP / f"{device}_verification.json", dict(status="verified", raw_labels_exact=True,
        raw_train_scaler_exact=True, time_features_exact=True, eligible_original_windows_exact=True, no_imputation=True,
        actual_batch=128, pilot_Adam_updates=2,
        losses=losses, pilot_weights_discarded=True, official_test_evaluations=0))


def audit():
    import numpy as np
    from tmp import review_mafirst_future_hidden_only_five_datasets_full100_20261109 as review

    plan = common.read(GROUP / "plan.json")
    review.UPDATES_PER_EPOCH = plan["updates_per_epoch"]
    review.VAL_WINDOWS = plan["val_windows"]
    review.EVAL_WINDOWS = plan["test_windows"]
    review.main("ETTh2")
    baseline = common.read(BASE / "output/official_test_result.json")
    result = common.read(OUTPUT / "official_test_result.json")
    assert result["official_test_evaluations"] == 1
    ids_path = GROUP / "test_window_ids.npy"
    ids = np.load(ids_path, allow_pickle=False)
    assert ids.dtype == np.int64 and len(ids) == plan["test_windows"] and np.all(np.diff(ids) > 0)
    assert ids[0] >= 0 and ids[-1] < baseline["windows"]
    original_test = BASE / "output/test"
    for key in ("prediction", "truth"):
        assert common.sha(original_test / (key + ".npy")) == baseline[key + "_sha256"]
    original_truth = np.load(original_test / "truth.npy", mmap_mode="r")
    original_prediction = np.load(original_test / "prediction.npy", mmap_mode="r")
    truth = np.load(OUTPUT / "test/truth.npy", mmap_mode="r")
    assert np.array_equal(original_truth[ids], truth)
    err = original_prediction[ids].astype(np.float64) - truth.astype(np.float64)
    reference_mse = float(np.square(err).mean())
    reference_mae = float(np.abs(err).mean())
    common.atomic(GROUP / "paired_audit.json", dict(status="verified", result=result,
        baseline_full_unfiltered=baseline,
        baseline_same_retained_test_windows=dict(mse=reference_mse, mae=reference_mae, windows=len(ids)),
        test_truth_exact_on_subset=True, test_ids_file_sha256=common.sha(ids_path),
        mse_delta_vs_baseline_same_subset=result["mse"]-reference_mse,
        mae_delta_vs_baseline_same_subset=result["mae"]-reference_mae,
        subset_uses_test_target_anomaly_flags=True, test_flags_never_used_for_fit_or_selection=True,
        saved_only_review=common.read(GROUP / "offline_review.json")))


def stage(name, variant, *arguments, cpu=False):
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        common.run(name, [common.PY,"-B","-u",str(Path(__file__).resolve()),*arguments,"--variant",variant],
                   root=GROUP)
    finally:
        if cpu:
            if env is None:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = env


def wait_for_gpu():
    threshold = 30 * 2**30 + 8 * common.EVAL_WINDOWS * common.HORIZON * common.CHANNELS
    while True:
        jobs = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],
                                       text=True).strip()
        memory = int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],
                                             text=True).splitlines()[0])
        disk = min(shutil.disk_usage(REPO).free, shutil.disk_usage("/mnt/d").free)
        if not jobs and memory >= 2500 and disk >= threshold:
            return
        write("status", dict(status="waiting_resources", gpu_free_mib=memory, free_bytes=disk, updated=common.now()))
        time.sleep(30)


def supervise():
    with (ROOT / "supervisor_once.json").open("x") as stream:
        json.dump(common.identity(), stream)
    try:
        predecessor = common.read(PREDECESSOR / "launch.json")["coordinator"]
        priority = common.read(PRIORITY / "priority_launch.json")["supervisor"]
        write("status", dict(status="waiting_predecessors", predecessor=predecessor, priority=priority,
                             cpu_pilots_deferred=True, updated=common.now()))
        while common.alive(predecessor) or common.alive(priority):
            time.sleep(30)
        prior = common.read(PREDECESSOR / "status.json")["status"]
        assert prior in ("completed", "completed_with_resource_blocks"), prior
        prior_result = common.read(PREDECESSOR / "results.json")
        assert prior_result["status"] in ("verified", "verified_with_resource_blocks")
        assert len(prior_result["completed"]) + len(prior_result["blocked"]) == 20
        write("predecessor_receipt", dict(five_datasets_status=prior, original_supervisors_inactive=True,
                                        five_datasets_result_sha256=common.sha(PREDECESSOR / "results.json")))
        for name in ARMS:
            activate(name)
            stage("CPU_route",name,"cpu",cpu=True)
        results = {}
        for name in ARMS:
            activate(name)
            wait_for_gpu()
            write("status", dict(status="running", variant=name, updated=common.now()))
            stage("CUDA_route",name,"cuda")
            stage("actual128_memory",name,"train","--phase","memory")
            pre = GROUP / "preflight" / MODEL_ARM
            for label, folder, extra in (("continuous","continuous",()),("stop1","resumed",("--stop-after","1")),
                                          ("resume","resumed",("--resume",))):
                stage("preflight_"+label,name,"train","--phase","preflight","--output",str(pre/folder),*extra)
            stage("preflight_saved_exact",name,"preflight_verify")
            stage("formal_train",name,"train","--phase","full")
            stage("official_test",name,"test")
            stage("saved_review",name,"audit")
            exits={p.parent.name:common.read(p) for p in (GROUP/"execution").glob("*/log.exit.json")}
            assert len(exits)==10 and all(e["real_wait"] and e["exit_code"]==0 for e in exits.values())
            common.guard()
            results[name]=dict(stages=exits, audit=common.read(GROUP/"paired_audit.json"))
            write("progress_results",dict(status="in_progress",variants=results))
        write("results",dict(status="verified",variants=results))
        write("status",dict(status="completed",updated=common.now()))
    except BaseException as error:
        write("failure",dict(error=repr(error),traceback=traceback.format_exc()))
        write("status",dict(status="failed",error=repr(error),updated=common.now()))
        raise


def launch():
    assert common.read(ROOT / "status.json")["status"] == "prepared"
    with (ROOT / "launch_once.json").open("x") as stream:
        json.dump(common.identity(), stream)
    command = [common.PY,"-B","-u",str(Path(__file__).resolve()),"supervise"]
    with (ROOT/"supervisor.log").open("xb") as stream:
        child = subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=stream,
                                 stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(100):
        if child.poll() is not None:
            raise RuntimeError((ROOT/"supervisor.log").read_text())
        identity=common.identity(child.pid)
        if identity["command"]==command:
            break
        time.sleep(.02)
    else:
        raise RuntimeError("could not verify supervisor identity")
    write("launch",dict(supervisor=identity,source_sha256=common.sha(Path(__file__)),created=common.now()))
    print(json.dumps(common.read(ROOT/"launch.json")),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","launch","supervise","cpu","cuda","train","preflight_verify","test","audit"))
    parser.add_argument("--variant",choices=ARMS)
    args, trailing=parser.parse_known_args()
    if args.mode in ("prepare","launch","supervise"):
        globals()[args.mode]()
        return
    assert args.variant is not None
    activate(args.variant)
    if args.mode in ("cpu","cuda"):
        verify(args.variant,args.mode)
    elif args.mode=="preflight_verify":
        from tmp import verify_mafirst_future_hidden_only_five_datasets_full100_20261109_preflight
    elif args.mode=="audit":
        audit()
    else:
        install_loader(args.variant)
        if args.mode=="train":
            from tmp import run_mafirst_etth2_h192_window_exclusion_20261110 as runner
            sys.argv=[sys.argv[0],"--arm",MODEL_ARM,*trailing]
            plan = common.read(GROUP / "plan.json")
            runner.TRAIN_WINDOWS = plan["train_windows"]
            runner.UPDATES_PER_EPOCH = plan["updates_per_epoch"]
            runner.main()
        else:
            from tmp import test_mafirst_future_hidden_only_five_datasets_full100_20261109 as tester
            tester.EVAL_WINDOWS = common.read(GROUP / "plan.json")["test_windows"]
            tester.main("ETTh2")


if __name__=="__main__":
    main()
