"""Run with: python -m unittest discover -s tests -p 'test_*.py'."""

import io
import logging
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from utils.logger import experiment_logging, get_logger, log_metrics, start_experiment


class ExperimentLoggingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = Namespace(
            log_dir=str(self.root / "logs"), log_level="INFO",
            tensorboard=False, tensorboard_dir=str(self.root / "runs"),
            task_name="long_term_forecast", model="TimeMixer",
            model_id="../nested/中文", seq_len=96, pred_len=96,
        )

    def read_log(self):
        return next((self.root / "logs").glob("*.log")).read_text(encoding="utf-8")

    def test_print_stderr_and_shared_logger_are_not_duplicated(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with experiment_logging(self.args):
                print("\033[1m中文 epoch\033[0m")
                sys.stdout.write("partial")
                sys.stdout.write(" line\n")
                print("stderr diagnostic", file=sys.stderr)
                get_logger("TimeMixer").info("model message")
                sys.stdout.write("final fragment")
        content = self.read_log()
        for message in ["中文 epoch", "partial line", "model message", "final fragment"]:
            self.assertEqual(content.count(message), 1)
            self.assertEqual(out.getvalue().count(message), 1)
        self.assertIn("stderr diagnostic", content)
        self.assertEqual(err.getvalue().count("stderr diagnostic"), 1)
        self.assertNotIn("\033[", content)

    def test_failure_traceback_and_stream_restoration(self):
        out, err = io.StringIO(), io.StringIO()
        logger = get_logger()
        handlers, level = logger.handlers[:], logger.level
        with redirect_stdout(out), redirect_stderr(err):
            with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
                with experiment_logging(self.args):
                    raise RuntimeError("deliberate failure")
            self.assertIs(sys.stdout, out)
            self.assertIs(sys.stderr, err)
        content = self.read_log()
        self.assertIn("Traceback", content)
        self.assertIn("RuntimeError: deliberate failure", content)
        self.assertNotIn("Run completed", content)
        self.assertEqual(logger.handlers, handlers)
        self.assertEqual(logger.level, level)

    def test_interruption_is_logged_and_propagated(self):
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                with experiment_logging(self.args):
                    raise KeyboardInterrupt()
        self.assertIn("Run interrupted by user", self.read_log())
        self.assertNotIn("Run completed", self.read_log())

    def test_repeated_sessions_use_unique_files(self):
        with redirect_stdout(io.StringIO()):
            for _ in range(2):
                with experiment_logging(self.args):
                    get_logger().info("unique message")
        paths = list((self.root / "logs").glob("*.log"))
        self.assertEqual(len(paths), 2)
        for path in paths:
            self.assertEqual(path.read_text().count("unique message"), 1)

    def test_disabled_tensorboard_keeps_text_metrics(self):
        with redirect_stdout(io.StringIO()):
            with experiment_logging(self.args):
                start_experiment(self.args)
                log_metrics({"loss/train": 0.25}, step=1)
        self.assertIn("loss/train=0.25", self.read_log())
        self.assertFalse((self.root / "runs").exists())

    def test_real_tensorboard_events_and_iteration_isolation(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        self.args.tensorboard = True
        self.args.log_level = "WARNING"
        with redirect_stdout(io.StringIO()):
            with experiment_logging(self.args):
                for iteration in range(2):
                    start_experiment(self.args, iteration)
                    log_metrics({"loss": {"train": 0.25 + iteration}}, step=1)
                    log_metrics({"mae": 0.1 + iteration}, prefix="test")
        events = sorted((self.root / "runs").rglob("events.out.tfevents.*"))
        self.assertEqual(len(events), 2)
        for iteration, event in enumerate(events):
            accumulator = EventAccumulator(str(event.parent)).Reload()
            scalar = accumulator.Scalars("loss/train")[0]
            self.assertEqual(scalar.step, 1)
            self.assertAlmostEqual(scalar.value, 0.25 + iteration)
            self.assertAlmostEqual(accumulator.Scalars("test/mae")[0].value,
                                   0.1 + iteration, places=6)
            self.assertIn("config/arguments/text_summary", accumulator.Tags()["tensors"])

    def test_cli_import_failure_is_saved_and_help_still_works(self):
        import subprocess

        root = Path(__file__).resolve().parents[1]
        script = """
import importlib.abc
import sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch':
            raise ImportError('simulated torch import failure')
sys.meta_path.insert(0, BlockTorch())
import run
run.main()
"""
        help_result = subprocess.run(
            [sys.executable, "-c", script, "--help"], cwd=root,
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--tensorboard_dir", help_result.stdout)
        result = subprocess.run(
            [sys.executable, "-c", script, "--task_name", "long_term_forecast",
             "--is_training", "1", "--model_id", "failure", "--model", "TimeMixer",
             "--data", "ETTm1", "--log_dir", self.args.log_dir, "--no_tensorboard"],
            cwd=root, capture_output=True, text=True, timeout=15,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stderr.count("Traceback"), 1)
        self.assertIn("ImportError: simulated torch import failure", self.read_log())


if __name__ == "__main__":
    unittest.main()
