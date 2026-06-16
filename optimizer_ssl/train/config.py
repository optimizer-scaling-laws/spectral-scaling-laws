"""Configuration and CLI parsing for language-model training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml


@dataclass
class Hyperparameters:
    # Data directory
    data_dir: str = "data/fineweb10B"

    # Training config
    batch_size: int = 8 * 64  # global batch size (across devices)
    device_batch_size: int = 64  # per-device batch size
    sequence_length: int = 1024  # tokens per sequence
    num_iterations: int = 5000
    seed: int = 1337
    warmup_ratio: float = 0.01
    warmdown_ratio: float = 0.2

    # Model config
    model_dim: int = 768
    n_layer: int = 12
    n_head: int = 6
    ffn_mult: int = 8
    postln_frac: float = 0.0

    # Evaluation and logging
    val_loss_every: int = 125
    val_tokens: int = 10485760
    checkpoint_freq: int = 0
    checkpoint_dir: str | None = None
    wandb_project_name: str = "optimizer-spectral-scaling-laws"

    # Optimizer
    optimizer: str = "dion"
    scalar_opt: str = "lion"

    # Main optimizer hyperparameters
    lr: float = 0.02
    mu: float = 0.95
    weight_decay: float = 0.01
    rank_fraction: float = 0.125

    # Optimizer specific hyperparameters
    qr_method: str = "rcqr"
    cqr_warmup: float = 0.05
    rcqr_oversample: float = 1.25
    replicate_mesh_grad_sync: bool = False
    mixed_precision: bool = False
    adjust_lr: str = "spectral_norm"  # for Muon only

    # Spectral metrics parameters
    enable_eigen_metrics: bool = False
    eigen_metrics_dir: str | None = None
    eigen_log_steps: int = 100
    track_by_frequency: bool = False
    token_freq_file: str | None = None
    frequency_bucket_reduction: str = "rank0_local"
    spectral_error_policy: str = "warn"


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments and merge optional YAML config values.

    CLI values always override YAML values. Extra YAML metadata keys are retained
    on the returned namespace so analysis scripts can inspect them, but they are
    ignored by ``Hyperparameters`` unless the dataclass has a matching field.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML file whose keys match train.py flags "
        "(CLI values always override the YAML).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory that contains fineweb_train_*.bin and fineweb_val_*.bin",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to load and save checkpoints",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps, 0 to disable",
    )

    # ---------- optimizer ----------
    parser.add_argument(
        "--optimizer", type=str, default=None, help="Choice of optimizer algorithm"
    )
    parser.add_argument(
        "--scalar_opt", type=str, help="Optimizer for scalar parameters", default=None
    )
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate")
    parser.add_argument(
        "--adjust_lr",
        type=str,
        default=None,
        help="Adjust learning rate method for Muon",
    )
    parser.add_argument(
        "--inv_rank_fraction",
        type=int,
        default=None,
        help="1/r rank fraction for Dion",
    )
    parser.add_argument(
        "--qr_method", type=str, default=None, choices=["qr", "cqr", "rcqr"]
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", help="Use mixed precision for Dion"
    )

    # ---------- model ----------
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)

    # ---------- training hyperparameters ----------
    parser.add_argument(
        "--num_iterations", type=int, default=None, help="Number of training steps"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Global batch size"
    )
    parser.add_argument("--device_batch_size", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--warmdown_ratio", type=float, default=None)

    # ---------- wandb logging ----------
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--wandb_project_name", type=str, default=None, help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_job_name",
        type=str,
        default=None,
        help="Append custom text to wandb job name",
    )

    # ---------- distributed training ----------
    parser.add_argument(
        "--dp_size", type=int, default=None, help="Data Parallel size (no sharding)"
    )
    parser.add_argument(
        "--fs_size", type=int, default=None, help="Fully Sharded Data Parallel size"
    )
    parser.add_argument(
        "--tp_size", type=int, default=None, help="Tensor Parallel size"
    )
    parser.add_argument(
        "--replicate_mesh_grad_sync",
        action="store_true",
        help="Do data-parallel gradient sync inside Dion optimizer",
    )
    parser.add_argument(
        "--fast_fsdp",
        action="store_true",
        help="Optimizer FSDP for speed instead of memory efficiency",
    )

    # ---------- debugging ----------
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--no_compile", action="store_true", help="Disable torch.compile for model"
    )
    parser.add_argument(
        "--no_triton", action="store_true", help="Disable Triton kernels"
    )

    # ---------- spectral metrics ----------
    parser.add_argument(
        "--enable_eigen_metrics",
        action="store_true",
        help="Enable spectral/eigenvalue metrics tracking",
    )
    parser.add_argument(
        "--eigen_metrics_dir",
        type=str,
        default=None,
        help="Directory for spectral metrics logs. If unspecified, uses checkpoint_dir/eigen_metrics_logs.",
    )
    parser.add_argument(
        "--eigen_log_steps",
        type=int,
        default=None,
        help="Step interval for logging spectral metrics",
    )
    parser.add_argument(
        "--track_by_frequency",
        action="store_true",
        help="Enable frequency-bucketed spectral metrics (requires token_freq_file)",
    )
    parser.add_argument(
        "--token_freq_file",
        type=str,
        default=None,
        help="Path to precomputed token frequencies (.pt file) for frequency-bucketed analysis",
    )
    parser.add_argument(
        "--frequency_bucket_reduction",
        type=str,
        default=None,
        choices=["rank0_local", "distributed_covariance"],
        help=(
            "How to reduce HEAD/MID/TAIL bucket metrics across data-parallel ranks. "
            "rank0_local reproduces the submitted paper; distributed_covariance "
            "globally reduces bucket covariance statistics."
        ),
    )
    parser.add_argument(
        "--spectral_error_policy",
        type=str,
        default=None,
        choices=["warn", "raise", "nan"],
        help=(
            "How spectral telemetry handles recoverable metric failures. "
            "warn logs an explicit missing marker, raise fails fast, nan writes NaN markers."
        ),
    )

    cli_args = parser.parse_args()
    if cli_args.config:
        cfg_path = Path(cli_args.config)
        with cfg_path.open("r") as f:
            yaml_cfg = yaml.safe_load(f) or {}

        for key, value in yaml_cfg.items():
            if getattr(cli_args, key, None) is None:
                setattr(cli_args, key, value)

        # Store-true flags need manual handling because argparse defaults False.
        for flag in (
            "mixed_precision",
            "replicate_mesh_grad_sync",
            "fast_fsdp",
            "no_wandb",
            "no_compile",
            "no_triton",
            "debug",
            "enable_eigen_metrics",
            "track_by_frequency",
        ):
            if yaml_cfg.get(flag, False):
                setattr(cli_args, flag, True)

    return cli_args


def override_args_from_cli(
    hp: Hyperparameters,
    cli_args: argparse.Namespace,
    printer: Callable[..., None] | None = None,
) -> Hyperparameters:
    """Apply CLI/YAML namespace values to the hyperparameter dataclass."""
    for key, value in vars(cli_args).items():
        if value is not None and hasattr(hp, key):
            if printer is not None:
                printer(f"Setting hyperparameter {key}={value}")
            setattr(hp, key, value)
    return hp


def validate_hyperparameters(hp: Hyperparameters) -> None:
    """Validate cross-field hyperparameter constraints not tied to a GPU runtime."""
    if hp.checkpoint_freq > 0 and not hp.checkpoint_dir:
        raise ValueError("Must specify --checkpoint_dir to save checkpoints")
    if hp.eigen_log_steps <= 0:
        raise ValueError("eigen_log_steps must be positive")
    allowed_frequency_reductions = {"rank0_local", "distributed_covariance"}
    if hp.frequency_bucket_reduction not in allowed_frequency_reductions:
        raise ValueError(
            "frequency_bucket_reduction must be one of "
            f"{sorted(allowed_frequency_reductions)}, got {hp.frequency_bucket_reduction!r}"
        )
    if hp.track_by_frequency and not hp.token_freq_file:
        raise ValueError("track_by_frequency=True requires token_freq_file")
    allowed_error_policies = {"warn", "raise", "nan"}
    if hp.spectral_error_policy not in allowed_error_policies:
        raise ValueError(
            "spectral_error_policy must be one of "
            f"{sorted(allowed_error_policies)}, got {hp.spectral_error_policy!r}"
        )
    if not isinstance(hp.seed, int):
        raise ValueError("seed must be an integer")
