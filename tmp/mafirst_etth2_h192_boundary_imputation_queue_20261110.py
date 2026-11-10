"""Window-aware boundary correction for the two pending ETTh2-H192 imputations."""
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tmp import mafirst_etth2_h192_train_input_imputation_20261110 as history

common = history.common
original = history.original
repo = history.repo
arm = history.arm
script = Path(__file__).resolve()
old = {
    "input": history.root,
    "both": history.root.with_name("mafirst_etth2_h192_train_input_target_imputation_20261110"),
}
roots = {key: path.with_name(path.name + "_boundary") for key, path in old.items()}
gate = roots["input"] / "replacement_receipt.json"
roles = ("input", "both")
variant = None
history_hours = 96
taper_hours = 12


def activate(key):
    global variant, root, group, output, train_count, updates
    variant = key
    root = roots[key]
    group = root / "group"
    output = root / "output"
    train_count = 5364 if key == "input" else 8353
    updates = train_count // 128
    history.root, history.group, history.output = root, group, output
    history.train_count, history.updates_per_epoch = train_count, updates
    history.stage = stage
    history.activate()
    common.TAG = "future_hidden_fixed_cosine_boundary_corrected_" + key


def correction(values, mask, original_fill):
    import numpy as np

    left = original_fill.copy()
    both = original_fill.copy()
    right_anchor = np.full(mask.shape, -1, dtype=np.int64)
    thresholds = []
    for channel in range(mask.shape[1]):
        valid_pairs = ~mask[:-1, channel] & ~mask[1:, channel]
        threshold = np.quantile(np.abs(np.diff(values[:, channel]))[valid_pairs], .90)
        thresholds.append(float(threshold))
        edges = np.flatnonzero(np.diff(np.r_[False, mask[:, channel], False]))
        for start, end in zip(edges[::2], edges[1::2]):
            length = min(taper_hours, (end - start) // 2)
            if length == 0:
                continue
            weights = .5 + .5 * np.cos(np.pi * np.arange(length) / length)
            if start > 0:
                delta = values[start-1, channel] - left[start, channel]
                left[start:start+length, channel] += np.sign(delta) * max(0, abs(delta)-threshold) * weights
                both[start:start+length, channel] = left[start:start+length, channel]
            if end < len(values):
                delta = values[end, channel] - both[end-1, channel]
                excess = np.sign(delta) * max(0, abs(delta)-threshold)
                if excess:
                    positions = np.arange(end-1, end-1-length, -1)
                    both[positions, channel] += excess * weights
                    right_anchor[positions, channel] = end
    return left, both, right_anchor, thresholds


class CorrectedWindows:
    def __init__(self, source, indices, left, both, right_anchor):
        self.source = source
        self.indices = indices
        self.left = left
        self.both = both
        self.right_anchor = right_anchor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        import numpy as np

        start = int(self.indices[index])
        stop = start + history_hours
        _, y, x_mark, y_mark = self.source[start]
        anchors = self.right_anchor[start:stop]
        visible = (anchors >= start) & (anchors < stop)
        x = np.where(visible, self.both[start:stop], self.left[start:stop])
        return x, y, x_mark, y_mark


def prepare():
    import numpy as np

    source = old[variant] / "group"
    registered = common.read(source / "registration.json")["hashes"]
    for rel, digest in registered.items():
        assert common.sha(repo / rel) == digest, rel
    mask = np.load(source / "train_anomaly_flags.npy", allow_pickle=False)
    filled = np.load(source / "imputed_train_inputs.npy", allow_pickle=False)
    ids = np.load(source / "train_window_ids.npy", allow_pickle=False)
    values, _ = original.source_frame(common.TRAIN_END)
    assert np.array_equal(filled[~mask], values[~mask])
    assert len(ids) == train_count and ids.dtype == np.int64
    expected = history.target_valid_indices(mask) if variant == "input" else np.arange(common.TRAIN_WINDOWS)
    assert np.array_equal(ids, expected)
    left, both, anchors, thresholds = correction(values, mask, filled)
    assert np.isfinite(left).all() and np.isfinite(both).all()
    assert np.array_equal(left[~mask], values[~mask]) and np.array_equal(both[~mask], values[~mask])
    assert np.array_equal(both[anchors == -1], left[anchors == -1])
    times, channels = np.nonzero(anchors >= 0)
    assert (anchors[times, channels] > times).all() and not mask[anchors[times, channels], channels].any()
    uses_right = np.zeros(len(ids), dtype=bool)
    hides_right = np.zeros(len(ids), dtype=bool)
    for i, start in enumerate(ids):
        end = start + history_hours
        observed = anchors[start:end]
        uses_right[i] = ((observed >= start) & (observed < end)).any()
        hides_right[i] = (observed >= end).any()
    assert uses_right.any() and hides_right.any()

    plan = common.read(source / "plan.json")
    assert plan["train_windows"] == train_count and plan["total_updates"] == updates*100
    plan["models"][arm]["config"].update(log_dir=str(group/"logs"), tensorboard_dir=str(group/"tensorboard"))
    plan["training_imputation"]["boundary_correction"] = dict(
        maximum_taper_hours_per_side=taper_hours, cosine_taper=True, permitted_jump_quantile=.90,
        raw_training_channel_thresholds=thresholds, input_right_anchor_must_be_inside_history=True,
        training_target_uses_both_sides=variant == "both", unchanged_observations_exact=True,
        source_fill_sha256=common.sha(source / "imputed_train_inputs.npy"),
    )
    frozen = common.read(source / "frozen_sources.json")
    frozen[script.relative_to(repo).as_posix()] = common.sha(script)
    protected = common.read(source / "protected_records.json")
    protected.update(registered)
    root.mkdir(exist_ok=False)
    (group / original.REF).parent.mkdir(parents=True, exist_ok=False)
    copies = (original.REF, Path("train_anomaly_flags.npy"), Path("train_window_ids.npy"),
              Path("expected_model_cards.json"), Path("imputation_diagnostics.json"))
    for name in copies:
        shutil.copyfile(source / name, group / name)
        assert common.sha(group / name) == registered[(source / name).relative_to(repo).as_posix()]
    np.savez(group / "boundary_values.npz", left=left, both=both, right_anchor=anchors)
    for name, record in (("plan", plan), ("frozen_sources", frozen), ("protected_records", protected)):
        common.atomic(group / (name + ".json"), record)
    sealed = (*copies, Path("boundary_values.npz"), Path("plan.json"),
              Path("frozen_sources.json"), Path("protected_records.json"))
    common.atomic(group / "registration.json", dict(
        hashes={(group / p).relative_to(repo).as_posix(): common.sha(group / p) for p in sealed},
    ))
    common.guard()
    history.write("preparation", dict(status="verified", train_windows=train_count, updates=100*updates,
        training_eval_splits_opened=False, val_windows=common.VAL_WINDOWS, test_windows=common.EVAL_WINDOWS,
        corrected_raw_cells=int(np.count_nonzero(both != filled)),
        windows_using_visible_right_anchor=int(uses_right.sum()),
        windows_hiding_unavailable_right_anchor=int(hides_right.sum()),
        train_targets_are_corrected=variant == "both"))
    history.status("prepared")
    print(json.dumps(common.read(root/"preparation.json")), flush=True)


def install_loader():
    import numpy as np
    from tmp import mafirst_future_hidden_only_five_datasets_full100_20261109_factory as factory

    native = factory.dataset

    def dataset(model_arm, split, limit=0):
        if split != "train":
            return native(model_arm, split, limit)
        ds = native(model_arm, split)
        ids = np.load(group / "train_window_ids.npy", allow_pickle=False)
        assert len(ids) == train_count
        with np.load(group / "boundary_values.npz", allow_pickle=False) as arrays:
            left = ds.scaler.transform(arrays["left"])
            both = ds.scaler.transform(arrays["both"])
            anchors = arrays["right_anchor"]
        if variant == "both":
            ds.data_y = both
        if limit:
            ids = ids[np.linspace(0, len(ids)-1, min(limit, len(ids)), dtype=np.int64)]
        return CorrectedWindows(ds, ids, left, both, anchors)

    factory.dataset = dataset
    return factory, native


def verify_windows(factory, native):
    import numpy as np

    ds = factory.dataset(arm, "train")
    raw = native(arm, "train")
    mask = np.load(group / "train_anomaly_flags.npy", allow_pickle=False)
    assert np.array_equal(ds.left[~mask], raw.data_x[~mask])
    assert np.array_equal(ds.both[~mask], raw.data_x[~mask])
    assert np.array_equal(ds.source.data_stamp, raw.data_stamp)
    expected_ids = history.target_valid_indices(mask) if variant == "input" else np.arange(len(raw))
    assert np.array_equal(ds.indices, expected_ids)
    validation = factory.dataset(arm, "val")
    raw_val = native(arm, "val")
    assert len(validation) == len(raw_val) == common.VAL_WINDOWS
    for key in ("data_x", "data_y", "data_stamp"):
        assert np.array_equal(getattr(validation, key), getattr(raw_val, key))

    visible_count = hidden_count = 0
    examples = []
    for i, start in enumerate(ds.indices):
        stop = start + history_hours
        anchors = ds.right_anchor[start:stop]
        visible = (anchors >= start) & (anchors < stop)
        hidden = anchors >= stop
        if not (visible.any() or hidden.any()):
            continue
        x, y, *marks = ds[i]
        _, raw_y, *raw_marks = raw[int(start)]
        assert np.array_equal(x[visible], ds.both[start:stop][visible])
        assert np.array_equal(x[~visible], ds.left[start:stop][~visible])
        assert np.array_equal(x[~mask[start:stop]], raw.data_x[start:stop][~mask[start:stop]])
        assert all(np.array_equal(a,b) for a,b in zip(marks,raw_marks))
        label = y[-common.HORIZON:]
        raw_label = raw_y[-common.HORIZON:]
        label_flags = mask[stop:stop+common.HORIZON]
        if variant == "both":
            assert np.array_equal(label, ds.both[stop:stop+common.HORIZON])
            assert np.array_equal(label[~label_flags], raw_label[~label_flags])
        else:
            assert np.array_equal(label, raw_label) and not label_flags.any()
        visible_count += int(visible.any())
        hidden_count += int(hidden.any())
        if visible.any() and hidden.any() and len(examples)<3:
            examples.append(int(start))
    assert visible_count and hidden_count
    if variant == "both":
        assert np.any(ds.source.data_y[mask] != raw.data_y[mask])
    else:
        assert np.array_equal(ds.source.data_y, raw.data_y)
    return ds, dict(visible_right_windows=visible_count, hidden_right_windows=hidden_count,
                    evaluation_validation_bitwise_equal=True, example_start_hours=examples)


def verify(device):
    import numpy as np
    import torch

    common.guard()
    factory, native = install_loader()
    from tmp import run_mafirst_etth2_h192_window_exclusion_20261110 as runner

    ds, evidence = verify_windows(factory, native)
    runner.configure(device)
    model, _ = factory.fresh(arm)
    model.to(device).train()
    opt = factory.optimizer(model)
    scheduler = factory.scheduler(opt, updates)
    eligible = np.arange(len(ds))
    if variant == "both":
        eligible = np.setdiff1d(eligible, history.target_valid_indices(
            np.load(group/"train_anomaly_flags.npy", allow_pickle=False)), assume_unique=True)
    sample = eligible[np.linspace(0,len(eligible)-1,256,dtype=int)]
    losses = []
    for step, indices in enumerate(np.array_split(sample,2),1):
        batch = [ds[int(i)] for i in indices]
        x, y = (torch.from_numpy(np.stack([row[col] for row in batch])) for col in (0,1))
        terms, _, _ = runner.backward_step(arm, model, opt, scheduler, x, y, device, 1, step)
        runner.finite(model, opt)
        losses.append(terms.tolist())
    common.atomic(group / (device + "_verification.json"), dict(
        status="verified", actual_batch=128, pilot_updates=2, discarded_weights=True,
        training_targets_are_corrected=variant == "both", losses=losses, **evidence))


def audit():
    from tmp import review_mafirst_future_hidden_only_five_datasets_full100_20261109 as review

    review.UPDATES_PER_EPOCH = updates
    review.main("ETTh2")
    result = common.read(output/"official_test_result.json")
    curves = common.read(output/"curves.json")
    assert len(curves) == 100 and curves[-1]["updates"] == updates*100
    strict = old["input"].with_name("mafirst_etth2_h192_train_only_exclusion_20261110")
    references = {"unmodified":original.BASE, "strict_exclusion":strict}
    if variant == "both":
        references["input_boundary_corrected"] = roots["input"]
    comparisons = {}
    for name, folder in references.items():
        reference = common.read(folder/"output/official_test_result.json")
        assert result["truth_sha256"] == reference["truth_sha256"]
        assert result["windows"] == reference["windows"] == common.EVAL_WINDOWS
        prior = common.read(folder/"output/curves.json")
        assert len(prior) == 100
        comparisons[name] = dict(reference=reference, mse_delta=result["mse"]-reference["mse"],
            mae_delta=result["mae"]-reference["mae"],
            validation_index_sha_matches=sum(a["val"]["indices_sha256"] == b["val"]["indices_sha256"]
                                             for a,b in zip(curves,prior)),
            train_order_sha_matches=sum(a["order_sha256"] == b["order_sha256"] for a,b in zip(curves,prior)))
    common.atomic(group/"paired_audit.json",dict(status="verified", result=result, comparisons=comparisons,
        boundary_corrected=True, train_targets_are_corrected=variant == "both", common_full_test_truth_sha=True,
        saved_only_review=common.read(group/"offline_review.json")))


def stage(name, *args, cpu=False):
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        history.status(name)
        common.run(name,[common.PY,"-B","-u",str(script),variant,*args],root=group)
    finally:
        if cpu and env is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES",None)
        elif cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = env


def supervise():
    predecessor_root = original.PREDECESSOR if variant == "input" else roots["input"]
    key = "coordinator" if variant == "input" else "supervisor"
    predecessor = common.read(predecessor_root/"launch.json")[key]
    with (root/"supervisor_once.json").open("x") as file:
        json.dump(common.identity(),file)
    history.write("launch",dict(supervisor=common.identity(),predecessor=predecessor,
                                source_sha256=common.sha(script),created=common.now()))
    history.status("waiting_replacement_gate",predecessor=predecessor)
    try:
        while not gate.is_file():
            time.sleep(5)
        receipt = common.read(gate)
        assert receipt["status"] == "superseded_waiters_exited_without_dispatch"
        assert all(not common.alive(rec) for rec in receipt["retired"].values())
        history.status("waiting_five_dataset_queue" if variant=="input" else "waiting_corrected_input",
                       predecessor=predecessor,cpu_pilots_deferred=True)
        while common.alive(predecessor):
            time.sleep(30)
        prior = common.read(predecessor_root/"status.json")
        result = common.read(predecessor_root/"results.json")
        if variant == "input":
            assert prior["status"] in ("completed","completed_with_resource_blocks"), prior
            assert result["status"] in ("verified","verified_with_resource_blocks")
            assert len(result["completed"])+len(result["blocked"]) == 20
        else:
            assert prior["status"] == "completed" and result["status"] == "verified"
            history.check_stage_waits(predecessor_root)
        history.write("predecessor_receipt",dict(original_supervisor_inactive=True,status=prior["status"],
                            results_sha256=common.sha(predecessor_root/"results.json")))
        history.experiment()
        history.status("completed")
    except BaseException as error:
        history.write("failure",dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        history.status("failed",error=repr(error))
        raise


def launch():
    common.guard()
    assert common.read(root/"status.json")["status"] == "prepared"
    with (root/"launch_once.json").open("x") as file:
        json.dump(common.identity(),file)
    cmd = [sys.executable,"-B","-u",str(script),variant,"supervise"]
    with (root/"supervisor.log").open("xb") as log:
        child = subprocess.Popen(cmd,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,
                                 stderr=subprocess.STDOUT,start_new_session=True)
    history.until(lambda:(root/"launch.json").is_file() or child.poll() is not None,20,
                  "supervisor did not register")
    assert (root/"launch.json").is_file(),(root/"supervisor.log").read_text()
    print(json.dumps(common.read(root/"launch.json")),flush=True)


def replace():
    expected_status = {"input":"waiting_five_dataset_queue","both":"waiting_input_only_imputation"}
    predecessor = common.read(original.PREDECESSOR/"launch.json")["coordinator"]
    retired = {key:common.read(old[key]/"launch.json")["supervisor"] for key in roles}
    incoming = {key:common.read(roots[key]/"launch.json")["supervisor"] for key in roles}
    assert not gate.exists() and all(common.alive(rec) for rec in incoming.values())
    assert all(common.read(roots[key]/"status.json")["status"] == "waiting_replacement_gate" for key in roles)
    assert common.alive(predecessor)
    assert common.read(old["input"]/"launch.json")["predecessor"] == predecessor
    assert common.read(old["both"]/"launch.json")["predecessor"] == retired["input"]
    descriptors = {}
    try:
        for key in roles:
            rec = retired[key]
            assert common.alive(rec)
            fd = os.pidfd_open(rec["pid"])
            descriptors[key] = fd
            assert common.alive(rec)
            assert common.read(old[key]/"status.json")["status"] == expected_status[key]
            assert not (old[key]/"group/execution").exists()
            assert not (Path("/proc")/str(rec["pid"])/"task"/str(rec["pid"])/"children").read_text().strip()
        for key in reversed(roles):
            assert common.alive(predecessor)
            assert common.read(old[key]/"status.json")["status"] == expected_status[key]
            signal.pidfd_send_signal(descriptors[key],signal.SIGTERM)
            assert select.select([descriptors[key]],[],[],15)[0],key
            assert not common.alive(retired[key])
            assert not (old[key]/"group/execution").exists()
        receipt = dict(status="superseded_waiters_exited_without_dispatch",retired=retired,
                       incoming=incoming,main_coordinator=predecessor,created=common.now())
        for key in roles:
            path = old[key]/"superseded_by_boundary_correction.json"
            with path.open("x") as file:
                json.dump(dict(replacement=str(roots[key]),previous=retired[key],new=incoming[key],
                               science_stages_dispatched=False,created=receipt["created"]),file,indent=2)
        assert root == roots["input"]
        history.write("replacement_receipt",receipt)
        print(json.dumps(receipt),flush=True)
    finally:
        for fd in descriptors.values():
            os.close(fd)


def main():
    key, mode, *args = sys.argv[1:]
    assert key in roles
    activate(key)
    if mode == "prepare":
        prepare()
    elif mode == "launch":
        launch()
    elif mode == "replace":
        replace()
    elif mode == "supervise":
        signal.signal(signal.SIGTERM,history.interrupted)
        signal.signal(signal.SIGINT,history.interrupted)
        supervise()
    elif mode in ("cpu","cuda"):
        verify(mode)
    elif mode == "data_verify":
        common.guard()
        _, evidence = verify_windows(*install_loader())
        common.atomic(group/"data_verification.json",dict(status="verified",**evidence))
        print(json.dumps(evidence),flush=True)
    elif mode == "train":
        install_loader()
        from tmp import run_mafirst_etth2_h192_window_exclusion_20261110 as runner

        runner.TRAIN_WINDOWS = train_count
        runner.UPDATES_PER_EPOCH = updates
        sys.argv = [sys.argv[0],"--arm",arm,*args]
        runner.main()
    elif mode == "preflight_verify":
        from tmp import verify_mafirst_future_hidden_only_five_datasets_full100_20261109_preflight
    elif mode == "test":
        from tmp import test_mafirst_future_hidden_only_five_datasets_full100_20261109 as tester

        tester.main("ETTh2")
    elif mode == "audit":
        audit()
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
