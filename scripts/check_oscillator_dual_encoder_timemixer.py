"""Small train/validation smoke check; never evaluates the test split."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from models.OscillatorDualEncoderTimeMixer import Model
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--csv', type=Path, default=ROOT / 'dataset/ETT-small/ETTm1.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp/dual_encoder_timemixer_smoke_20261013')
    parser.add_argument('--patch-length', type=int, default=4)
    parser.add_argument('--patch-stride', type=int, default=2)
    parser.add_argument('--patch-heads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.2)
        torch.cuda.manual_seed_all(2021)
    # Read only the established train+validation prefix; no test rows.
    frame = pd.read_csv(args.csv, nrows=46080)
    if frame.shape != (46080, 8):
        raise ValueError('Expected the ETTm1 train+validation prefix, 46080 rows and date+7 variables')
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(values[:34560])
    values = scaler.transform(values).astype(np.float32)

    def windows(start, count):
        x = np.stack([values[i:i+96] for i in range(start, start+count)])
        y = np.stack([values[i+96:i+192] for i in range(start, start+count)])
        return torch.from_numpy(x).to(args.device), torch.from_numpy(y).to(args.device)

    train_x, train_y = windows(0, 32)
    val_x, val_y = windows(34560-96, 16)
    config = SimpleNamespace(
        task_name='long_term_forecast', seq_len=96, pred_len=96, enc_in=7, c_out=7,
        features='M', moving_avg=25, d_model=32, d_ff=64, e_layers=1, dropout=.2,
        top_k=5, ssm_modes=8, trend_patch_length=args.patch_length,
        trend_patch_stride=args.patch_stride, trend_patch_heads=args.patch_heads)
    model = Model(config).to(args.device)
    optimizer = torch.optim.Adam(norm_bias_no_decay_groups(model, 1e-4), lr=.001)
    training = []
    for step in range(2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_x)
        mse = (prediction-train_y).square().mean()
        distance, diagnostic = parts(prediction, train_y)
        loss = mse + distance
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite smoke loss')
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError('Missing/nonfinite gradient: ' + name)
        gradients = {
            name: float(torch.stack([p.grad.square().sum() for p in module.parameters()]).sum().sqrt())
            for name, module in [('season_recurrence', model.season_oscillators),
                                 ('trend_recurrence', model.trend_oscillators),
                                 ('trend_attention', model.trend_encoders), ('pdm', model.pdm)]}
        optimizer.step()
        training.append(dict(step=step+1, mse=float(mse.detach()), **diagnostic,
                             gradient_norms=gradients))
    model.eval()
    with torch.no_grad():
        prediction = model(val_x)
        components = model.predict_components(val_x)
        reconstructed = model.normalizers[0](components.sum(-1), 'denorm')
        torch.testing.assert_close(prediction, reconstructed, rtol=0, atol=0)
        _, diagnostic = parts(prediction, val_y)
        validation = dict(mse=float((prediction-val_y).square().mean()),
                          mae=float((prediction-val_y).abs().mean()), **diagnostic)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / 'smoke_weights.pt'
    torch.save(model.state_dict(), checkpoint)
    restored = Model(config).to(args.device).eval()
    restored.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(val_x), prediction, rtol=0, atol=0)
    audit = dict(status='verified', scope='2 updates, first32 train / first16 validation windows; NOT full experiment',
                 device=args.device, seed=2021, configuration=vars(config),
                 parameters=sum(p.numel() for p in model.parameters()),
                 scaler_fit_rows=34560, data_rows_read=46080, test_windows_evaluated=0,
                 train=training, validation=validation,
                 output_shape=list(prediction.shape), components_shape=list(components.shape),
                 checkpoint_reload_exact=True, component_reconstruction_exact=True,
                 checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
