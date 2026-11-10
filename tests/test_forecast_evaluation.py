"""Stable validation batches, dataset-wide MSE, and optional regularization."""

from argparse import Namespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


class IndexedDataset(Dataset):
    def __init__(self, **kwargs):
        pass

    def __len__(self):
        return 7

    def __getitem__(self, index):
        return index


class ZeroForecast(torch.nn.Module):
    def forward(self, x, *args):
        return torch.zeros_like(x[:, -1:])


class ForecastEvaluationTests(unittest.TestCase):
    def test_only_training_shuffles_and_validation_batch_groups_repeat(self):
        args = Namespace(data="fixture", embed="timeF", batch_size=3, freq="h",
                         task_name="long_term_forecast", num_workers=0,
                         root_path="", data_path="", seq_len=1, label_len=0,
                         pred_len=1, features="M", target="OT", seasonal_patterns="Monthly")
        with patch.dict("data_provider.data_factory.data_dict", fixture=IndexedDataset):
            for flag in ("train", "val", "test", "TEST"):
                with self.subTest(flag=flag):
                    _, loader = data_provider(args, flag)
                    sampler = RandomSampler if flag == "train" else SequentialSampler
                    self.assertIsInstance(loader.sampler, sampler)
                    if flag != "train":
                        expected = [[0, 1, 2], [3, 4, 5], [6]]
                        self.assertEqual([b.tolist() for b in loader], expected)
                        self.assertEqual([b.tolist() for b in loader], expected)

    def test_validation_mse_matches_concatenated_predictions_for_partial_batch(self):
        experiment = object.__new__(Exp_Long_Term_Forecast)
        experiment.args = Namespace(pred_len=1, label_len=0, features="M", use_amp=False)
        experiment.device = torch.device("cpu")
        experiment.model = ZeroForecast()
        batches, targets = [], []
        for size, value in ((3, 0.0), (1, 3.0)):
            x = torch.zeros(size, 1, 1)
            y = torch.full_like(x, value)
            mark = torch.zeros(size, 1, 4)
            batches.append((x, y, mark, mark))
            targets.append(y)
        mse = experiment.vali(None, batches, torch.nn.MSELoss())
        self.assertAlmostEqual(mse, torch.cat(targets).square().mean().item())
        self.assertAlmostEqual(mse, 2.25)
        with self.assertRaisesRegex(ValueError, "no target elements"):
            experiment.vali(None, [], torch.nn.MSELoss())

    def test_optional_adam_l2_decay_updates_with_zero_data_gradient(self):
        for decay in (None, 0.01):
            with self.subTest(decay=decay):
                experiment = object.__new__(Exp_Long_Term_Forecast)
                experiment.args = Namespace(learning_rate=0.001)
                if decay is not None:
                    experiment.args.weight_decay = decay
                experiment.model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    experiment.model.weight.fill_(1.0)
                optimizer = experiment._select_optimizer()
                experiment.model.weight.grad = torch.zeros_like(experiment.model.weight)
                optimizer.step()
                weight = experiment.model.weight.item()
                if decay is None:
                    self.assertEqual(weight, 1.0)
                else:
                    self.assertLess(weight, 1.0)


if __name__ == "__main__":
    unittest.main()
