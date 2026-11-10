"""Low-frequency LagARFormer diagnostics; never change optimizer or model state."""
import math

import torch


def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def parameter_metrics(model, gradient_scale=1.0):
    """First-batch norms; divide out AMP scale and retain tiny weights in float64."""
    model = unwrap(model)
    result = {}
    for i, encoder in enumerate(model.seasonals):
        modules = {
            "relation": [encoder.lag_embedding] + [b.lag_projection for b in encoder.message_blocks],
            "source": [b.source_projection for b in encoder.message_blocks],
        }
        for group, members in modules.items():
            parameters = [p for module in members for p in module.parameters()]
            result[f"scale{i}/{group}_parameter_l2"] = math.sqrt(sum(
                p.detach().double().square().sum().item() for p in parameters))
            result[f"scale{i}/{group}_gradient_l2"] = math.sqrt(sum(
                (p.grad.detach().double() / gradient_scale).square().sum().item()
                for p in parameters if p.grad is not None))
    return result


def validation_probe(model, dataset, count=16):
    """Fixed uniform validation windows, FP32, no gradients or DataLoader RNG.

    Restore training flags and transient RevIN state even on failure.
    """
    if count < 1 or len(dataset) < 1:
        raise ValueError("ACF probe requires a nonempty validation set and positive count")
    model = unwrap(model)
    indices = torch.linspace(0, len(dataset) - 1, min(count, len(dataset))).long().tolist()
    samples = [dataset[index] for index in indices]
    device = next(model.parameters()).device
    x = torch.stack([torch.as_tensor(row[0]) for row in samples]).float().to(device)
    y = torch.stack([torch.as_tensor(row[1])[-model.pred_len:] for row in samples]).float().to(device)
    if model.features == "MS":
        y = y[..., -1:]
    modes = {module: module.training for module in model.modules()}
    sentinel = object()
    norm_state = {key: getattr(model.normalize, key, sentinel) for key in ("mean", "stdev", "last")}
    try:
        model.eval()
        with torch.no_grad():
            baseline = model.forecast(x)
            zero = model.forecast(x, zero_acf=True)
            if model.features == "MS":
                baseline, zero = baseline[..., -1:], zero[..., -1:]
            delta = (zero.double() - baseline.double()).square().mean().sqrt()
            metrics = {
                "windows": len(indices),
                "full_mse": (baseline.double() - y.double()).square().mean().item(),
                "zero_acf_mse": (zero.double() - y.double()).square().mean().item(),
                "prediction_change_rms": delta.item(),
            }
            metrics["mse_change"] = metrics["zero_acf_mse"] - metrics["full_mse"]
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Non-finite ACF validation diagnostic")
            return metrics
    finally:
        for module, mode in modes.items():
            module.training = mode
        for key, value in norm_state.items():
            if value is sentinel:
                if hasattr(model.normalize, key):
                    delattr(model.normalize, key)
            else:
                setattr(model.normalize, key, value)
