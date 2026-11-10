"""Training must not replace a valid checkpoint with non-finite results."""

import tempfile
import unittest
from pathlib import Path

import torch

from utils.tools import EarlyStopping
from utils.training import require_finite_loss, require_finite_gradients


class TrainingStabilityTests(unittest.TestCase):
    def test_loss_guard_rejects_nan_and_inf(self):
        require_finite_loss(torch.tensor(0.5), "epoch=1 batch=2")
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(FloatingPointError, "epoch=1 batch=2"):
                require_finite_loss(torch.tensor(value), "epoch=1 batch=2")

    def test_gradient_guard_catches_bad_backward_with_finite_loss(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        loss = model(torch.ones(1, 1)).sqrt().sum()
        require_finite_loss(loss, "batch=3")
        loss.backward()
        before = model.weight.detach().clone()
        with self.assertRaisesRegex(FloatingPointError, "weight"):
            require_finite_gradients(model, "batch=3")
        torch.testing.assert_close(model.weight, before)
        model.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        require_finite_gradients(model, "batch=4")

    def test_nonfinite_validation_preserves_best_checkpoint_and_state(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as path:
                model = torch.nn.Linear(2, 1)
                stopping = EarlyStopping()
                stopping(0.5, model, path)
                checkpoint = Path(path) / "checkpoint.pth"
                original = checkpoint.read_bytes()
                with torch.no_grad():
                    model.weight.fill_(float("nan"))
                with self.assertRaisesRegex(FloatingPointError, "not overwritten"):
                    stopping(value, model, path)
                self.assertEqual(checkpoint.read_bytes(), original)
                self.assertEqual(stopping.best_score, -0.5)
                self.assertEqual(stopping.val_loss_min, 0.5)
                self.assertEqual(stopping.counter, 0)

    def test_nonfinite_first_validation_creates_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as path:
            stopping = EarlyStopping()
            with self.assertRaises(FloatingPointError):
                stopping(float("nan"), torch.nn.Linear(2, 1), path)
            self.assertIsNone(stopping.best_score)
            self.assertFalse((Path(path) / "checkpoint.pth").exists())


if __name__ == "__main__":
    unittest.main()
