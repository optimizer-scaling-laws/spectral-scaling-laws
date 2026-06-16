"""Validation/evaluation helpers for training."""

from __future__ import annotations

import torch
import torch.distributed as dist


def estimate_validation_loss(model, val_loader, val_steps: int, autocast_ctx, device) -> float:
    """Estimate validation loss and average it across distributed ranks."""
    model.eval()
    val_loader.reset()
    val_loss = torch.tensor(0.0, device=device)
    for _ in range(val_steps):
        with torch.no_grad():
            x_val, y_val = val_loader.next_batch()
            with autocast_ctx:
                loss = model(x_val, y_val)
            val_loss += loss

    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    return val_loss.item() / val_steps
