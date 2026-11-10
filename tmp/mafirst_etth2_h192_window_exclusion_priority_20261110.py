"""Prioritize the registered ETTh2 exclusions at the RevIN handoff."""
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tmp import mafirst_etth2_h192_window_exclusion_20261110 as run

common = run.common
root = run.ROOT
prior = run.PRIORITY
weather = run.PREDECESSOR / "weather/h96"
proc = Path("/proc")


def status(phase, **details):
    run.write("status", dict(status=phase, updated=common.now(), **details))


def inspect(rec):
    assert common.identity(rec["pid"]) == rec
    return fields(rec["pid"])[0]


def fields(pid):
    return (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()


def child_ids(rec):
    pid = str(rec["pid"])
    return (proc / pid / "task" / pid / "children").read_text().split()


def until(predicate, timeout, message):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise RuntimeError(message)
        time.sleep(.02)


def matches_arm(arm, pid, dispatcher, dataset, horizon):
    stat = fields(pid)
    assert stat[1] == str(dispatcher["pid"]) and stat[19] == arm["start_ticks"]
    if stat[0] != "Z":
        assert common.identity(pid) == arm
        assert arm["command"] == dispatcher["command"][:-1] + ["case", "--dataset", dataset, "--horizon", str(horizon)]
    return stat


def wait_for_weather_wait(revin):
    path = weather / "arm_exit.json"
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    try:
        # arm_exit.json is published by atomic rename. Check again after installing the watch.
        if libc.inotify_add_watch(fd, os.fsencode(weather), 0x80) < 0:  # IN_MOVED_TO
            raise OSError(ctypes.get_errno(), "inotify_add_watch")
        while not path.is_file():
            assert common.alive(revin), "RevIN supervisor exited before the Weather-H96 wait receipt"
            if select.select([fd], [], [], 2)[0]:
                os.read(fd, 65536)
    finally:
        os.close(fd)
    receipt = common.read(path)
    assert receipt["real_wait"] and receipt["exit_code"] == 0
    return receipt


def check_stage_waits(folder):
    exits = {p.parent.name: common.read(p) for p in (folder / "group/execution").glob("*/log.exit.json")}
    assert len(exits) == 10 and all(e["real_wait"] and e["exit_code"] == 0 for e in exits.values())
    return exits


def run_exclusions():
    for variant in run.ARMS:
        run.activate(variant)
        common.guard()
        status("CPU_route", variant=variant)
        run.stage("CPU_route", variant, "cpu", cpu=True)
    results = {}
    for variant in run.ARMS:
        run.activate(variant)
        run.wait_for_gpu()
        status("running", variant=variant)
        run.stage("CUDA_route", variant, "cuda")
        run.stage("actual128_memory", variant, "train", "--phase", "memory")
        preflight = run.GROUP / "preflight" / run.MODEL_ARM
        for label, output, args in (("continuous", "continuous", ()), ("stop1", "resumed", ("--stop-after", "1")),
                                    ("resume", "resumed", ("--resume",))):
            run.stage("preflight_" + label, variant, "train", "--phase", "preflight",
                      "--output", str(preflight / output), *args)
        run.stage("preflight_saved_exact", variant, "preflight_verify")
        run.stage("formal_train", variant, "train", "--phase", "full")
        run.stage("official_test", variant, "test")
        run.stage("saved_review", variant, "audit")
        common.guard()
        results[variant] = dict(stages=check_stage_waits(run.GROUP.parent),
                                audit=common.read(run.GROUP / "paired_audit.json"))
        run.write("progress_results", dict(status="in_progress", variants=results))
    return results


def supervise():
    original = common.read(root / "launch.json")["supervisor"]
    dispatcher = common.read(run.PREDECESSOR / "launch.json")["coordinator"]
    revin = common.read(prior / "priority_launch.json")["supervisor"]
    caught = False
    replaced = False
    draining = None
    pidfd = os.pidfd_open(dispatcher["pid"])
    try:
        with common.locked():
            state = common.read(root / "status.json")
            assert state["status"] == "waiting_predecessors" and state["cpu_pilots_deferred"]
            assert state["predecessor"] == dispatcher and state["priority"] == revin
            assert inspect(dispatcher) in ("T", "t") and inspect(revin) not in ("T", "t", "Z")
            assert inspect(original) not in ("T", "t", "Z") and not child_ids(original)
            assert all(not (root / v / "group/execution").exists() for v in run.ARMS)
            with os.fdopen(os.pidfd_open(original["pid"]), "rb") as waiter:
                inspect(original)
                signal.pidfd_send_signal(waiter.fileno(), signal.SIGTERM)
                poll = select.poll()
                poll.register(waiter.fileno(), select.POLLIN)
                assert poll.poll(5000), "old exclusion waiter did not exit"
            replaced = True
            run.write("priority_launch", dict(supervisor=common.identity(), original_waiter=original,
                original_waiter_exited=True, revin=revin, dispatcher=dispatcher,
                source_sha256=common.sha(Path(__file__)), created=common.now()))
            status("waiting_revin_boundary", revin=revin, dispatcher=dispatcher, cpu_pilots_deferred=True)

        # Let the RevIN controller resume the original parent and collect the genuine
        # Weather-H96 wait receipt. Reacquire the same lock at that dispatch boundary.
        weather_wait = wait_for_weather_wait(revin)

        with common.locked():
            assert inspect(dispatcher) not in ("T", "t", "Z")
            signal.pidfd_send_signal(pidfd, signal.SIGSTOP)
            caught = True
            until(lambda: inspect(dispatcher) in ("T", "t"), 5, "dispatcher did not pause")
            current = common.read(run.PREDECESSOR / "status.json")
            assert current["status"] == "running" and current["process"] == dispatcher
            children = child_ids(dispatcher)
            if children:
                child, = children
                dataset, horizon = current["dataset"], current["horizon"]
                folder = run.PREDECESSOR / dataset.lower() / f"h{horizon}"
                until(lambda: common.identity(int(child))["command"] == dispatcher["command"][:-1]
                      + ["case", "--dataset", dataset, "--horizon", str(horizon)]
                      or fields(int(child))[0] == "Z", 5, "could not identify the already dispatched case")
                arm = common.identity(int(child))
                state = matches_arm(arm, int(child), dispatcher, dataset, horizon)
                assert state[0] != "Z" or int(state[49]) == 0
                draining = (int(child), arm, folder, dataset, horizon)
            until(lambda: (prior / "priority_resume_receipt.json").is_file(), 10, "RevIN release receipt is missing")
            release = common.read(prior / "priority_resume_receipt.json")
            assert release["dispatcher"] == dispatcher and release["resumed"] is True
            run.write("priority_boundary_receipt", dict(dispatcher=dispatcher, revin_release=release,
                weather_h96_parent_wait=weather_wait, draining_case=list(draining[3:]) if draining else None,
                scientific_processes_signalled=False, created=common.now()))

        # The prior controller finishes independently once it sees the actual parent wait.
        until(lambda: not common.alive(revin), 120, "RevIN priority controller did not exit")
        if (prior / "results.json").exists():
            assert common.read(prior / "status.json")["status"] == "completed"
            assert common.read(prior / "results.json")["status"] == "verified"
            assert common.read(prior / "priority_original_parent_wait.json")["original_parent_wait"] == weather_wait
        else:
            assert (prior / "priority_failure.json").is_file(), "RevIN missing terminal evidence"
        if draining:
            pid, arm, folder, dataset, horizon = draining
            status("waiting_already_dispatched_case", dataset=dataset, horizon=horizon)
            while True:
                state = matches_arm(arm, pid, dispatcher, dataset, horizon)
                assert inspect(dispatcher) in ("T", "t")
                if state[0] == "Z":
                    break
                time.sleep(2)
            assert int(state[49]) == 0
            assert common.read(folder / "group/status.json")["status"] == "completed"
            run.write("priority_case_drain_receipt", dict(arm=arm, kernel_exit_status=0, stage_waits=check_stage_waits(folder)))

        run.wait_for_gpu()
        results = run_exclusions()
        run.write("results", dict(status="verified", variants=results))
    except BaseException as exc:
        run.write("priority_failure", dict(error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False))
        if replaced:
            status("failed", error=repr(exc))
        raise
    finally:
        try:
            if caught:
                assert inspect(dispatcher) in ("T", "t")
                signal.pidfd_send_signal(pidfd, signal.SIGCONT)
                run.write("priority_resume_receipt", dict(dispatcher=dispatcher, resumed=True, created=common.now()))
        finally:
            os.close(pidfd)
    if draining:
        folder = draining[2]
        until(lambda: (folder / "arm_exit.json").is_file(), 120, "dispatcher did not collect the drained case")
        receipt = common.read(folder / "arm_exit.json")
        assert receipt["real_wait"] and receipt["exit_code"] == 0 and not common.alive(draining[1])
        run.write("priority_case_parent_wait", receipt)
    status("completed")


def launch():
    assert common.read(root / "status.json")["status"] == "waiting_predecessors"
    with (root / "priority_dispatch_once").open("x") as marker:
        marker.write(str(os.getpid()) + "\n")
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "supervise"]
    with (root / "priority_supervisor.log").open("xb") as log:
        child = subprocess.Popen(command, cwd=run.REPO, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    until(lambda: (root / "priority_launch.json").exists() or child.poll() is not None,
          20, "priority scheduler did not register")
    if not (root / "priority_launch.json").exists():
        raise RuntimeError((root / "priority_supervisor.log").read_text())
    print(json.dumps(common.read(root / "priority_launch.json")), flush=True)


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    {"launch": launch, "supervise": supervise}[sys.argv[1]]()
