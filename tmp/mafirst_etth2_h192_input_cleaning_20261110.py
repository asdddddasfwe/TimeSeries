"""ETTh2-H192 paired, input-only data cleaning ablation."""
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
ROOT = REPO / "tmp/mafirst_etth2_h192_input_cleaning_20261110"
ARMS = ("rules", "rules_lof")
MODEL_ARM = "mafirst_season"
REF = Path("reference_initial/mafirst_season.pt")
DATA = Path("cleaned_trainval_inputs.npy")


def activate(variant):
    global GROUP, OUTPUT
    GROUP, OUTPUT = ROOT / variant / "group", ROOT / variant / "output"
    common.G = common.B = GROUP
    common.OUT = {MODEL_ARM: OUTPUT}
    common.TAG = f"future_hidden_fixed_cosine_input_cleaning_{variant}"


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


def cleaned_inputs(values, dates):
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import RobustScaler

    assert np.isfinite(values).all()
    n, channels = values.shape
    train = common.TRAIN_END
    day31 = dates.dt.day.to_numpy() == 31
    hours = dates.dt.hour.to_numpy()
    # Derive the calendar pattern exclusively from complete days in the training prefix.
    first = np.flatnonzero(day31[:train] & (hours[:train] == 0))
    assert len(first) >= 2 and all(i + 24 <= train and np.all(values[i:i+24] == values[i]) for i in first)

    changes = np.vstack((np.ones(channels, dtype=bool), values[1:] != values[:-1]))
    i = np.arange(n)[:, None]
    run = i - np.maximum.accumulate(np.where(changes, i, 0), axis=0) + 1
    rules = np.broadcast_to(day31[:, None], values.shape).copy()
    rules |= (run >= 48) & (values != 0)

    hourly = np.empty((24, channels))
    for hour in range(24):
        x = values[:train][hours[:train] == hour].copy()
        x[rules[:train][hours[:train] == hour]] = np.nan
        hourly[hour] = np.nanmedian(x, axis=0)
    assert np.isfinite(hourly).all()
    donor = hourly[hours]
    residual = values - donor
    additional = np.zeros_like(rules)
    for ch in range(channels):
        col = values[:, ch]
        history = sliding_window_view(col, 24)
        features = np.column_stack((residual[23:, ch], np.median(history, axis=1) - donor[23:, ch],
                                    np.median(np.abs(np.diff(history, axis=1)), axis=1),
                                    np.ptp(history, axis=1), (history == 0).mean(axis=1),
                                    np.log1p(np.minimum(run[23:, ch], 48))))
        train_rows = np.arange(23, n) < train
        references = train_rows & ~rules[23:, ch] & (run[23:, ch] < 12)
        scaler = RobustScaler().fit(features[references])
        estimator = LocalOutlierFactor(n_neighbors=35, novelty=True)
        estimator.fit(scaler.transform(features[references]))
        cutoff = np.quantile(estimator.negative_outlier_factor_, .01)
        candidate = np.flatnonzero(~rules[23:, ch] & (run[23:, ch] >= 12) & (col[23:] != 0))
        if len(candidate):
            scores = estimator.score_samples(scaler.transform(features[candidate]))
            additional[candidate[scores < cutoff] + 23, ch] = True
    return {"rules": np.where(rules, donor, values), "rules_lof": np.where(rules | additional, donor, values)}, {
        "rules": rules, "rules_lof": rules | additional,
    }


def prepare():
    import numpy as np

    source = BASE / "group"
    baseline = common.read(BASE / "output/official_test_result.json")
    assert baseline["epochs"] == 100 and baseline["official_test_evaluations"] == 1
    assert common.read(BASE / "arm_exit.json")["real_wait"] and common.read(BASE / "arm_exit.json")["exit_code"] == 0
    values, dates = source_frame(common.VAL_END)
    original_plan = common.read(source / "plan.json")
    cfg = original_plan["models"][MODEL_ARM]["config"]
    import pandas as pd

    frozen_values = pd.read_csv(Path(cfg["root_path"]) / cfg["data_path"], nrows=common.VAL_END)
    assert np.array_equal(values, frozen_values.iloc[:, 1:].to_numpy())
    assert dates.equals(pd.to_datetime(frozen_values.date))
    data, masks = cleaned_inputs(values, dates)
    # Check that a shorter observation prefix yields identical results at every retained timestamp.
    shorter = common.VAL_END - 96
    previous, previous_masks = cleaned_inputs(values[:shorter], dates.iloc[:shorter])
    assert all(np.array_equal(data[name][:shorter], previous[name]) and
               np.array_equal(masks[name][:shorter], previous_masks[name]) for name in ARMS)
    assert np.any(data["rules_lof"] != data["rules"])
    ROOT.mkdir(exist_ok=False)
    metrics = {}
    for name in ARMS:
        activate(name)
        (GROUP / REF).parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source / REF, GROUP / REF)
        np.save(GROUP / DATA, data[name], allow_pickle=False)
        np.save(GROUP / "input_flags_trainval.npy", masks[name], allow_pickle=False)
        plan = common.read(source / "plan.json")
        assert plan["actual_train_batch"] == 128 and plan["total_updates"] == 6500
        assert common.sha(GROUP / REF) == plan["models"][MODEL_ARM]["reference_sha256"]
        plan["models"][MODEL_ARM]["config"].update(log_dir=str(GROUP / "logs"),
                                                   tensorboard_dir=str(GROUP / "tensorboard"))
        plan["input_cleaning"] = dict(variant=name, train_only_fit=True, calendar_day31=True,
            nonzero_repeat_rule_hours=48, replacement="per-channel training median by hour",
            lof_enabled=name == "rules_lof", lof_neighbors=35, lof_training_reference_quantile=.01,
            lof_min_nonzero_repeat_hours=12, lof_feature_history_hours=24, causal_features=True,
            target_policy="raw labels for training, validation and test; original raw training scaler",
            original_case=str(BASE), retrospective_test_diagnostics_informed_problem_choice=True)
        frozen = common.read(source / "frozen_sources.json")
        frozen[Path(__file__).relative_to(REPO).as_posix()] = common.sha(Path(__file__))
        protected = common.read(source / "protected_records.json")
        for item in (source / "plan.json", source / REF, BASE / "output/curves.json",
                     BASE / "output/official_test_result.json"):
            protected[item.relative_to(REPO).as_posix()] = common.sha(item)
        for filename, payload in (("plan", plan), ("expected_model_cards", common.read(source / "expected_model_cards.json")),
                                  ("frozen_sources", frozen), ("protected_records", protected)):
            common.atomic(GROUP / f"{filename}.json", payload)
        sealed = [REF, DATA, Path("input_flags_trainval.npy"), *(Path(n+".json") for n in
                  ("plan", "expected_model_cards", "frozen_sources", "protected_records"))]
        common.atomic(GROUP / "registration.json", dict(hashes={
            (GROUP / n).relative_to(REPO).as_posix(): common.sha(GROUP / n) for n in sealed}))
        common.guard()
        changed = data[name] != values
        metrics[name] = {part: dict(flagged=int(masks[name][lo:hi].sum()), changed=int(changed[lo:hi].sum()),
                            changed_by_channel=dict(zip(frozen_values.columns[1:], changed[lo:hi].sum(0).tolist())))
                         for part, lo, hi in (("train",0,common.TRAIN_END),("validation",common.TRAIN_END,common.VAL_END))}
    write("preparation", dict(status="verified", baseline=baseline, records=metrics,
                             val_prefix_invariance=True, test_not_loaded=True))
    write("status", dict(status="prepared", created=common.now()))
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


def install_loader(variant):
    import numpy as np
    from torch.utils.data import Subset
    from tmp import mafirst_future_hidden_only_five_datasets_full100_20261109_factory as factory

    base_dataset = factory.dataset

    def dataset(arm, split, limit=0):
        ds = base_dataset(arm, split)
        prepared = np.load(GROUP / DATA, allow_pickle=False)
        if split == "test":
            values, dates = source_frame(common.TEST_END)
            inputs, _ = cleaned_inputs(values, dates)
            source = inputs[variant]
            assert np.array_equal(source[:common.VAL_END], prepared)
        else:
            source = prepared
        start = ds.official_support["input_start"]
        stop = ds.official_support["end_exclusive"]
        ds.data_x = ds.scaler.transform(source[start:stop])
        assert ds.data_x.shape == ds.data_y.shape
        if limit:
            return Subset(ds, np.unique(np.linspace(0, len(ds)-1, min(limit,len(ds)), dtype=int)).tolist())
        return ds

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
    raw = base_dataset(MODEL_ARM, "train")
    for split, left, right in [("train", cleaned, raw),
            ("val", factory.dataset(MODEL_ARM, "val"), base_dataset(MODEL_ARM, "val"))]:
        assert len(left) == len(right) and np.array_equal(left.data_y, right.data_y)
        assert np.array_equal(left.data_stamp, right.data_stamp)
        assert np.array_equal(left.scaler.mean_, right.scaler.mean_)
        assert np.array_equal(left.scaler.scale_, right.scaler.scale_)
        assert np.isfinite(left.data_x).all(), split
    changed = np.flatnonzero((cleaned.data_x != raw.data_x).any(1))
    ids = np.unique(np.clip(changed - 32, 0, len(cleaned)-1))
    assert len(ids) >= 128
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
        raw_train_scaler_exact=True, time_features_exact=True, actual_batch=128, pilot_Adam_updates=2,
        losses=losses, pilot_weights_discarded=True, official_test_evaluations=0))


def audit():
    import numpy as np
    from tmp import review_mafirst_future_hidden_only_five_datasets_full100_20261109 as review

    review.main("ETTh2")
    control = common.read(BASE / "output/official_test_result.json")
    result = common.read(OUTPUT / "official_test_result.json")
    assert result["truth_sha256"] == control["truth_sha256"]
    assert result["official_test_evaluations"] == 1
    curves = common.read(OUTPUT / "curves.json")
    original = common.read(BASE / "output/curves.json")
    assert len(curves) == len(original) == 100 and curves[-1]["updates"] == 6500
    report = {}
    for label, out in (("raw", BASE / "output"), ("cleaned", OUTPUT)):
        pred = np.load(out / "test/prediction.npy", mmap_mode="r")
        true = np.load(out / "test/truth.npy", mmap_mode="r")
        delta = np.abs(np.diff(pred.astype(np.float64), axis=1))
        boundary = np.arange(delta.shape[1]) % 4 == 3
        y = true[:, :, 3].astype(np.float64)
        errors = pred[:, :, 3].astype(np.float64) - y
        shift = np.abs(y[:, 24:] - y[:, :-24]) > 1
        report[label] = dict(boundary_ratio=float(delta[:, boundary].mean()/delta[:, ~boundary].mean()),
            mull_24h_change_threshold=1., mull_shift_elements=int(shift.sum()),
            mull_shift_mse=float(np.mean(errors[:, 24:][shift]**2)),
            mull_shift_mae=float(np.mean(np.abs(errors[:, 24:][shift]))))
    common.atomic(GROUP / "paired_audit.json", dict(status="verified", baseline=control, result=result,
        matched_train_orders=sum(a["order_sha256"] == b["order_sha256"] for a,b in zip(curves,original)),
        matched_validation_indices=sum(a["val"]["indices_sha256"] == b["val"]["indices_sha256"]
                                       for a,b in zip(curves,original)), diagnostics=report,
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
        for name in ARMS:
            activate(name)
            stage("CPU_route",name,"cpu",cpu=True)
        predecessor = common.read(PREDECESSOR / "launch.json")["coordinator"]
        priority = common.read(PRIORITY / "priority_launch.json")["supervisor"]
        write("status", dict(status="waiting_predecessors", predecessor=predecessor, priority=priority,
                             cpu_pilot_passed=True, updated=common.now()))
        while common.alive(predecessor) or common.alive(priority):
            time.sleep(30)
        prior = common.read(PREDECESSOR / "status.json")["status"]
        assert prior in ("completed", "completed_with_resource_blocks"), prior
        prior_result = common.read(PREDECESSOR / "results.json")
        assert prior_result["status"] in ("verified", "verified_with_resource_blocks")
        assert len(prior_result["completed"]) + len(prior_result["blocked"]) == 20
        write("predecessor_receipt", dict(five_datasets_status=prior, original_supervisors_inactive=True,
                                        five_datasets_result_sha256=common.sha(PREDECESSOR / "results.json")))
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
            from tmp import run_mafirst_future_hidden_only_five_datasets_full100_20261109 as runner
            sys.argv=[sys.argv[0],"--arm",MODEL_ARM,*trailing]
            runner.main()
        else:
            from tmp import test_mafirst_future_hidden_only_five_datasets_full100_20261109 as tester
            tester.main("ETTh2")


if __name__=="__main__":
    main()
