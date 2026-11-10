"""Opt-in ETTm1 integration smoke; never evaluates full model accuracy."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import run
from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from models.SeasonTrendMixer import Model
from utils.logger import experiment_logging, start_experiment


def framework_args(argv):
    """Use the real CLI parser, stopping before its logging/training context."""
    captured = []
    class Parsed(Exception):
        pass
    def capture(args):
        captured.append(args)
        raise Parsed()
    with patch.object(sys, "argv", ["run.py"] + argv):
        with patch.object(run, "experiment_logging", side_effect=capture):
            try:
                run.main()
            except Parsed:
                pass
    return captured[0]


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class SmokeExperiment(Exp_Long_Term_Forecast):
    def _get_data(self, flag):
        if not hasattr(self, "smoke_data"):
            self.smoke_data = {}
        if flag not in self.smoke_data:
            dataset, _ = data_provider(self.args, flag)
            count = 32 if flag == "train" else 16
            subset = Subset(dataset, list(range(count)))
            loader = DataLoader(subset, batch_size=8, shuffle=flag == "train",
                                num_workers=0, drop_last=False)
            self.smoke_data[flag] = dataset, loader
        return self.smoke_data[flag]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    options = parser.parse_args()
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dataset_file = ROOT / "dataset/ETT-small/ETTm1.csv"
    if not dataset_file.is_file():
        raise FileNotFoundError(dataset_file)
    tag = "seasontrendmixer_patch_conv_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    artifact_dir = ROOT / "tmp" / tag
    artifact_dir.mkdir(parents=True, exist_ok=False)
    args = framework_args([
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--model", "SeasonTrendMixer", "--model_id", tag, "--data", "ETTm1",
        "--root_path", str(dataset_file.parent), "--data_path", dataset_file.name,
        "--features", "M", "--freq", "h", "--seq_len", "96", "--label_len", "0",
        "--pred_len", "96", "--enc_in", "7", "--dec_in", "7", "--c_out", "7",
        "--d_model", "32", "--d_ff", "64", "--n_heads", "4", "--e_layers", "2",
        "--top_k", "3", "--moving_avg", "25", "--dropout", "0.2",
        "--down_sampling_layers", "3", "--down_sampling_window", "2",
        "--down_sampling_method", "avg", "--channel_independence", "1", "--use_norm", "1",
        "--batch_size", "8", "--num_workers", "0", "--train_epochs", "1",
        "--patience", "10", "--learning_rate", "0.001", "--weight_decay", "0.0001",
        "--lradj", "cosine", "--checkpoints", str(artifact_dir / "checkpoints"),
        "--log_dir", str(artifact_dir / "logs"),
        "--tensorboard_dir", str(artifact_dir / "runs"),
        "--use_gpu" if options.device == "cuda" else "--no_use_gpu",
    ])
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.set_num_threads(2)
    with working_directory(ROOT), experiment_logging(args) as logger:
        args.device = torch.device(options.device)
        logger.info("SMOKE ONLY: actual seed=2021, train/val/test=32/16/16; "
                    "first windows from each standard ETTm1 split; torch=%s, device=%s",
                    torch.__version__, options.device)
        experiment = SmokeExperiment(args)
        start_experiment(args, iteration=0)
        for flag in ("train", "val", "test"):
            experiment._get_data(flag)
        temporal_names = [
            name for name, _ in experiment.model.named_parameters()
            if name.startswith(("season_linears.", "trend_linears.", "season_heads.", "trend_heads."))
            and name.endswith(".weight")
        ]
        assert len(temporal_names) == 14
        convolution_names = [name for name, _ in experiment.model.named_parameters()
                             if ".local_encoder." in name and name.endswith(".weight")]
        assert len(convolution_names) == 4
        before = {name: value.detach().clone() for name, value in experiment.model.named_parameters()
                  if name in temporal_names + convolution_names}
        # The existing trainer writes relative results and a summary text file.
        # Scope all of them to this fresh smoke directory.
        with working_directory(artifact_dir):
            experiment.train(tag)
            updated = [name for name, value in experiment.model.named_parameters()
                       if name in temporal_names and not torch.equal(before[name], value)]
            assert len(updated) == 14, updated
            convolutions_updated = [name for name, value in experiment.model.named_parameters()
                                    if name in convolution_names and not torch.equal(before[name], value)]
            assert len(convolutions_updated) == 4, convolutions_updated
            checkpoint = artifact_dir / "checkpoints" / tag / "checkpoint.pth"
            restored = Model(args).to(experiment.device)
            restored.load_state_dict(torch.load(checkpoint, map_location=experiment.device,
                                                weights_only=True), strict=True)
            restored.eval()
            experiment.model.eval()
            inputs = next(iter(experiment._get_data("test")[1]))[0].float().to(experiment.device)
            with torch.no_grad():
                expected = experiment.model(inputs)
                actual = restored(inputs)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            assert torch.isfinite(actual).all()
            experiment.test(tag)
        logger.info("SMOKE PASS: all 6 preprocessing and 8 prediction weights updated; "
                    "all 4 local convolution weights updated; "
                    "strict checkpoint restore gives identical predictions")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_files = list((artifact_dir / "runs").rglob("events.out.tfevents.*"))
    assert event_files, "No TensorBoard event file"
    events = EventAccumulator(str(event_files[0].parent)).Reload()
    required_tags = ("loss/train", "loss/val", "loss/test", "learning_rate", "test/mse", "test/mae")
    for name in required_tags:
        scalars = events.Scalars(name)
        assert scalars and all(np.isfinite(s.value) for s in scalars), name
    results = artifact_dir / "results" / tag
    prediction = np.load(results / "pred.npy")
    truth = np.load(results / "true.npy")
    metrics = np.load(results / "metrics.npy")
    assert prediction.shape == truth.shape == (16, 96, 7)
    assert np.isfinite(prediction).all() and np.isfinite(truth).all() and np.isfinite(metrics).all()
    logs = list((artifact_dir / "logs").glob("*.log"))
    assert logs and "six identity temporal preprocessors" in logs[0].read_text()
    assert "encoding=period_patch_conv" in logs[0].read_text()
    summary = dict(
        purpose="integration smoke only; subset metrics do not measure forecasting quality",
        model="SeasonTrendMixer", architecture="fft_period_patch_conv",
        dataset=str(dataset_file), seed=2021, device=options.device,
        torch_version=torch.__version__, train_windows=32, val_windows=16, test_windows=16,
        window_selection="first windows of the standard train/val/test splits",
        epochs=1, batch_size=8, num_workers=0, prediction_shape=list(prediction.shape),
        temporal_weight_matrices_updated=len(updated),
        convolution_weight_matrices_updated=len(convolutions_updated),
        strict_checkpoint_roundtrip=True,
        tensorboard_scalar_tags=events.Tags()["scalars"],
        losses={name: events.Scalars(name)[-1].value for name in required_tags},
        checkpoint=str(checkpoint), log=str(logs[0]), tensorboard=str(event_files[0].parent),
    )
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"SMOKE PASSED: {summary_path}")


if __name__ == "__main__":
    main()
