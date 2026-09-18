"""Console and per-run file logging without changing existing print calls."""

import json
import logging
import os
import re
import shlex
import sys
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

_active_session = None

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class _PlainFormatter(logging.Formatter):
    def format(self, record):
        return _ANSI_ESCAPE.sub("", super().format(record))


class _TeeStream:
    """Keep terminal output unchanged and log complete lines to the file."""

    def __init__(self, stream, logger, level):
        self.stream = stream
        self.logger = logger
        self.level = level
        self._pending = ""
        self.lock = threading.RLock()

    def write(self, text):
        with self.lock:
            result = self.stream.write(text)
            self._pending += text.replace("\r", "\n")
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                self._emit(line)
            return result

    def _emit(self, line):
        if line.strip():
            self.logger.log(self.level, line, extra={"from_stream": True})

    def flush(self):
        with self.lock:
            self.stream.flush()
            if self._pending:
                self._emit(self._pending)
                self._pending = ""

    def __getattr__(self, name):
        return getattr(self.stream, name)



def get_logger(name=None):
    """Return the shared application logger, usable from any model or task."""
    return logging.getLogger("tslib" + (f".{name}" if name else ""))


class _Session:
    def __init__(self, tensorboard_dir):
        self.tensorboard_dir = tensorboard_dir
        self.writer = None
        self.pid = os.getpid()

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def start_experiment(args, iteration=0):
    """Start a separate TensorBoard run for each repetition of an experiment."""
    session = _active_session
    if session is None or session.pid != os.getpid():
        return
    session.close()
    get_logger().info("Experiment iteration=%s | resolved arguments: %s",
                      iteration, json.dumps(vars(args), default=str, sort_keys=True))
    if session.tensorboard_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        directory = session.tensorboard_dir / f"iteration_{iteration}"
        session.writer = SummaryWriter(log_dir=str(directory), flush_secs=10)
        session.writer.add_text(
            "config/arguments",
            "\x60\x60\x60json\n" + json.dumps(vars(args), default=str, indent=2) + "\n\x60\x60\x60",
            0,
        )
        session.writer.flush()
        get_logger().info("TensorBoard directory: %s", directory)


def _flatten_metrics(metrics, prefix=""):
    for name, value in metrics.items():
        tag = f"{prefix}/{name}" if prefix else str(name)
        if isinstance(value, Mapping):
            yield from _flatten_metrics(value, tag)
        else:
            yield tag, float(value)


def log_metrics(metrics, step=0, prefix=""):
    """Write scalar metrics (including nested mappings) to text and TensorBoard.

    Call from the experiment's main process, not DataLoader workers. With no
    active logging context, this still emits a normal Python logging record.
    """
    session = _active_session
    if session is not None and session.pid != os.getpid():
        return
    values = dict(_flatten_metrics(metrics, prefix))
    get_logger().info("step=%s | %s", step, " | ".join(
        f"{name}={value:.7g}" for name, value in values.items()))
    if session is not None and session.writer is not None:
        for name, value in values.items():
            session.writer.add_scalar(name, value, global_step=step)
        session.writer.flush()


def _safe_name(value):
    return re.sub(r"[^\w.-]+", "_", str(value))[:40] or "experiment"


@contextmanager
def experiment_logging(args):
    """Capture Python stdout/stderr and failures for one run.py invocation.

    New code can use logging.getLogger("tslib") or a child such as
    logging.getLogger("tslib.training"). Native file-descriptor writes are
    not captured. Streams and logger configuration are restored on exit.
    """
    global _active_session
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    name = "_".join(_safe_name(getattr(args, key))
                    for key in ("task_name", "model", "model_id"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = log_dir / (
        f"{name}_sl{args.seq_len}_pl{args.pred_len}_{stamp}_{os.getpid()}.log"
    )
    file_handler = logging.FileHandler(path, mode="x", encoding="utf-8")
    formatter = _PlainFormatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    stdout, stderr = sys.stdout, sys.stderr
    console_handler = logging.StreamHandler(stdout)
    console_handler.setFormatter(formatter)
    # Captured print/stderr output has already been written to the terminal.
    console_handler.addFilter(lambda record: not getattr(record, "from_stream", False))
    logger = logging.getLogger("tslib")
    old_state = (logger.handlers[:], logger.level, logger.propagate, logger.disabled)
    logger.handlers = [console_handler, file_handler]
    logger.setLevel(args.log_level)
    logger.propagate = False
    logger.disabled = False
    out = _TeeStream(stdout, logger, logging.INFO)
    err = _TeeStream(stderr, logger, logging.WARNING)
    sys.stdout, sys.stderr = out, err
    started = time.monotonic()
    previous_session = _active_session
    tb_dir = (Path(args.tensorboard_dir).expanduser().resolve() / path.stem
              if getattr(args, "tensorboard", False) else None)
    session = _Session(tb_dir)
    _active_session = session
    try:
        logger.info("Log file: %s", path)
        logger.info("Command: %s", shlex.join([sys.executable] + sys.argv))
        logger.info("Working directory: %s", Path.cwd())
        logger.info("Arguments: %s", json.dumps(vars(args), ensure_ascii=False, sort_keys=True))
        yield logger
    except KeyboardInterrupt:
        out.flush()
        err.flush()
        logger.warning("Run interrupted by user after %.2fs", time.monotonic() - started)
        raise
    except SystemExit as exc:
        out.flush()
        err.flush()
        if exc.code not in (None, 0):
            logger.error("Run exited with status %s", exc.code)
        raise
    except BaseException:
        out.flush()
        err.flush()
        # The interpreter prints the re-raised traceback to the restored stderr.
        logger.exception("Run failed", extra={"from_stream": True})
        raise
    else:
        out.flush()
        err.flush()
        logger.info("Run completed in %.2fs", time.monotonic() - started)
    finally:
        try:
            out.flush()
            err.flush()
            session.close()
        finally:
            _active_session = previous_session
            sys.stdout, sys.stderr = stdout, stderr
            logger.handlers, logger.level, logger.propagate, logger.disabled = old_state
            file_handler.close()
            console_handler.close()
