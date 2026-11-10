"""Numerical checks shared by training loops."""

import torch


def require_finite_loss(loss, context):
    """Stop before backward/updates when an objective contains NaN or Inf."""
    if not torch.isfinite(loss.detach()).all():
        raise FloatingPointError(
            f"{context}: non-finite loss ({loss.detach().item()}); optimizer step aborted. "
            "Check inputs, model activations and learning rate."
        )


def require_finite_gradients(model, context):
    """Check all gradients with one device synchronization on the normal path."""
    named = [(name, parameter.grad) for name, parameter in model.named_parameters()
             if parameter.grad is not None]
    if not named:
        return
    finite = torch.stack([torch.isfinite(grad).all() for _, grad in named])
    if not finite.all():
        failed = [name for (name, _), ok in zip(named, finite.tolist()) if not ok]
        raise FloatingPointError(
            f"{context}: non-finite gradients in {', '.join(failed[:8])}; "
            "optimizer step aborted. Enable autograd anomaly detection to locate the operation."
        )
