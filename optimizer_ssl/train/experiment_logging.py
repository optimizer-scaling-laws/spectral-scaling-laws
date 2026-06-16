"""Lightweight logging helpers for distributed training runs.

This module intentionally keeps logging simple: rank-zero printing, run-name
construction, and optional Weights & Biases setup/logging. The core training
loop remains in ``train_lm.py``.
"""

from __future__ import annotations

import os
from typing import Any

import torch.distributed as dist

try:
    import wandb
except ImportError:  # WandB is optional when configs set no_wandb=true.
    wandb = None

_MASTER_PROCESS = True


def set_master_process(is_master: bool) -> None:
    """Set whether the current process should emit user-facing logs."""
    global _MASTER_PROCESS
    _MASTER_PROCESS = bool(is_master)


def is_master_process() -> bool:
    """Return True for global rank zero."""
    return _MASTER_PROCESS


def print0(*args: Any, **kwargs: Any) -> None:
    """Print only on global rank zero."""
    if _MASTER_PROCESS:
        print(*args, **kwargs)


def build_run_name(hp: Any, cli_args: Any) -> str:
    """Construct the human-readable run name used in logs and WandB."""
    run_name = f"({hp.optimizer}+{hp.scalar_opt})"
    if "dion" in hp.optimizer or "dion2" in hp.optimizer:
        run_name += f"frac={hp.rank_fraction}"
    if cli_args.dp_size is not None:
        run_name += (
            f"_dp={cli_args.dp_size}_fs={cli_args.fs_size}_tp={cli_args.tp_size}"
            f"_gradsync={cli_args.replicate_mesh_grad_sync}"
        )
    if cli_args.wandb_job_name:
        run_name += f"_{cli_args.wandb_job_name}"
    return run_name


def wandb_enabled(cli_args: Any) -> bool:
    """Return whether WandB logging should be active for this run."""
    return not cli_args.no_wandb and not cli_args.debug


def setup_wandb_if_enabled(hp: Any, cli_args: Any, checkpoint_manager: Any, run_name: str) -> None:
    """Initialize WandB on rank zero and synchronize the run id.

    WandB is skipped in debug mode or when ``--no_wandb`` is set;
    otherwise it is required.
    """
    if not wandb_enabled(cli_args):
        return

    if wandb is None:
        raise ImportError("wandb is required unless --no_wandb is set")
    assert hp.wandb_project_name, "wandb project name is required"

    if _MASTER_PROCESS:
        wandb_id = checkpoint_manager.wandb_id
        resume = "must" if wandb_id else "never"
        wandb.login(
            key=os.environ.get("WANDB_API_KEY"),
            host=os.environ.get("WANDB_HOST"),
            timeout=0,
        )
        wandb.init(
            project=hp.wandb_project_name,
            name=run_name,
            config=hp.__dict__,
            id=wandb_id,
            resume=resume,
        )
        checkpoint_manager.wandb_id = wandb.run.id

    obj_list = [checkpoint_manager.wandb_id]
    dist.broadcast_object_list(obj_list, src=0)
    checkpoint_manager.wandb_id = obj_list[0]


def log_validation_if_enabled(cli_args: Any, step: int, val_loss: float, training_time_ms: float) -> None:
    """Log validation metrics to WandB when enabled."""
    if _MASTER_PROCESS and wandb_enabled(cli_args):
        assert wandb is not None
        wandb.log(
            {
                "val/loss": val_loss,
                "step": step,
                "time/training_time_ms": training_time_ms,
            }
        )


def log_train_if_enabled(
    cli_args: Any,
    step: int,
    train_loss: float,
    grad_norm: float,
    training_time_ms: float,
) -> None:
    """Log training metrics to WandB when enabled."""
    if _MASTER_PROCESS and wandb_enabled(cli_args):
        assert wandb is not None
        wandb.log(
            {
                "train/loss": train_loss,
                "train/grad_norm": grad_norm,
                "step": step,
                "time/training_time_ms": training_time_ms,
            }
        )
