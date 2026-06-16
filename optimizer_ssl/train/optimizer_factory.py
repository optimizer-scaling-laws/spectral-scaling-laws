"""Optimizer construction for AdamW, Muon, NorMuon, and Dion variants."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.distributed.tensor import DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP

from optimizer_ssl.models.gpt_model import GPT
from optimizer_ssl.train.config import Hyperparameters
from optimizer_ssl.train.experiment_logging import print0

# Allow running from a source checkout before `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DION_VENDOR_ROOT = _REPO_ROOT / "third_party" / "dion"
if _DION_VENDOR_ROOT.exists() and str(_DION_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_DION_VENDOR_ROOT))

from dion import Dion, DionMixedPrecisionConfig  # noqa: E402
from dion import Dion2, DionReference, DionSimple  # noqa: E402
from dion import Muon, MuonReference, NorMuon  # noqa: E402


def init_optimizer(
    model: GPT,
    device_mesh: Optional[DeviceMesh],
    ddp_model: Optional[DDP],
    hp: Hyperparameters,
    cli_args,
):
    """Create the optimizer for a configured training run."""
    if hp.scalar_opt not in ["adamw", "lion"]:
        raise ValueError(f"Unrecognized scalar optimizer: {hp.scalar_opt}")

    matrix_params = list(model.transformer.h.parameters())
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())

    param_groups = [dict(params=matrix_params)]
    param_groups.append(
        dict(
            params=embedding_params,
            algorithm=hp.scalar_opt,
            lr=hp.lr,
            betas=(0.95, 0.98),
            weight_decay=0,
        )
    )
    param_groups.append(
        dict(
            params=lm_head_params,
            algorithm=hp.scalar_opt,
            lr=hp.lr / math.sqrt(hp.model_dim),
            betas=(0.95, 0.98),
            weight_decay=0,
        )
    )

    if device_mesh is not None:
        replicate_mesh = device_mesh["dp"]
        outer_shard_mesh = device_mesh["fs"]
        inner_shard_mesh = device_mesh["tp"] if device_mesh["tp"].size() > 1 else None
    else:
        assert ddp_model is not None
        replicate_mesh = ddp_model.process_group
        outer_shard_mesh = None
        inner_shard_mesh = None

    if hp.mixed_precision:
        dion_mixed_precision_config = DionMixedPrecisionConfig(
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            Q_dtype=torch.bfloat16,
        )
    else:
        dion_mixed_precision_config = None

    if hp.optimizer == "dion":
        print0(f"Dion rank fraction: {hp.rank_fraction}")
        print0(f"Dion mixed precision: {hp.mixed_precision}")
        print0(f"Compressed data-parallel gradient sync: {hp.replicate_mesh_grad_sync}")
        opt = Dion(
            param_groups,
            replicate_mesh=replicate_mesh,
            outer_shard_mesh=outer_shard_mesh,
            inner_shard_mesh=inner_shard_mesh,
            replicate_mesh_grad_sync=hp.replicate_mesh_grad_sync,
            rank_fraction=hp.rank_fraction,
            lr=hp.lr,
            mu=hp.mu,
            weight_decay=hp.weight_decay,
            qr_method=hp.qr_method,
            cqr_warmup_steps=round(hp.cqr_warmup * hp.num_iterations),
            rcqr_oversample=hp.rcqr_oversample,
            mixed_precision_config=dion_mixed_precision_config,
        )

    elif hp.optimizer == "dion_reference":
        print0(f"Dion rank fraction: {hp.rank_fraction}")
        print0(f"Dion QR method: {hp.qr_method}")
        print0(f"Dion mixed precision: {hp.mixed_precision}")
        print0(f"Compressed data-parallel gradient sync: {hp.replicate_mesh_grad_sync}")
        opt = DionReference(
            param_groups,
            replicate_mesh=replicate_mesh,
            outer_shard_mesh=outer_shard_mesh,
            inner_shard_mesh=inner_shard_mesh,
            replicate_mesh_grad_sync=hp.replicate_mesh_grad_sync,
            rank_fraction=hp.rank_fraction,
            lr=hp.lr,
            mu=hp.mu,
            weight_decay=hp.weight_decay,
            qr_method=hp.qr_method,
            cqr_warmup_steps=round(hp.cqr_warmup * hp.num_iterations),
            rcqr_oversample=hp.rcqr_oversample,
            mixed_precision_config=dion_mixed_precision_config,
        )

    elif hp.optimizer == "muon":
        if device_mesh is not None:
            if inner_shard_mesh is not None and inner_shard_mesh.size() > 1:
                raise ValueError("Tensor parallel is not supported by Muon.")
            distributed_mesh = (
                outer_shard_mesh if outer_shard_mesh.size() > 1 else replicate_mesh
            )
            comm_method = "all-to-all" if outer_shard_mesh.size() > 1 else "all-gather"
        else:
            assert ddp_model is not None
            distributed_mesh = ddp_model.process_group
            comm_method = "all-gather"
        print0(f"Muon LR adjust method: {hp.adjust_lr}")
        print0(f"Triton Newton-Schulz kernels: {not cli_args.no_triton}")
        print0(f"Distributed Muon using: {comm_method}")
        opt = Muon(
            param_groups,
            distributed_mesh=distributed_mesh,
            lr=hp.lr,
            mu=hp.mu,
            weight_decay=hp.weight_decay,
            nesterov=True,
            adjust_lr=hp.adjust_lr,
            use_triton=(not cli_args.no_triton),
        )

    elif hp.optimizer == "dion2":
        if device_mesh is not None:
            if inner_shard_mesh is not None and inner_shard_mesh.size() > 1:
                raise ValueError("Tensor parallel is not supported by dion2.")
            distributed_mesh = (
                outer_shard_mesh if outer_shard_mesh.size() > 1 else replicate_mesh
            )
            comm_method = "all-to-all" if outer_shard_mesh.size() > 1 else "all-gather"
        else:
            assert ddp_model is not None
            distributed_mesh = ddp_model.process_group
            comm_method = "all-gather"
        print0(f"LR adjust method: {hp.adjust_lr}")
        print0(f"Triton Newton-Schulz kernels: {not cli_args.no_triton}")
        print0(f"Distributed Dion2 using: {comm_method}")
        opt = Dion2(
            param_groups,
            distributed_mesh=distributed_mesh,
            lr=hp.lr,
            fraction=hp.rank_fraction,
            ef_decay=hp.mu,
            weight_decay=hp.weight_decay,
            adjust_lr=hp.adjust_lr,
            use_triton=(not cli_args.no_triton),
        )

    elif hp.optimizer == "normuon":
        if device_mesh is not None:
            if inner_shard_mesh is not None and inner_shard_mesh.size() > 1:
                raise ValueError("Tensor parallel is not supported by NorMuon.")
            distributed_mesh = (
                outer_shard_mesh if outer_shard_mesh.size() > 1 else replicate_mesh
            )
            comm_method = "all-to-all" if outer_shard_mesh.size() > 1 else "all-gather"
        else:
            assert ddp_model is not None
            distributed_mesh = ddp_model.process_group
            comm_method = "all-gather"
        print0(f"NorMuon LR adjust method: {hp.adjust_lr}")
        print0(f"Triton Newton-Schulz kernels: {not cli_args.no_triton}")
        print0(f"Distributed NorMuon using: {comm_method}")
        opt = NorMuon(
            param_groups,
            distributed_mesh=distributed_mesh,
            lr=hp.lr,
            mu=hp.mu,
            muon_beta2=0.95,
            weight_decay=hp.weight_decay,
            nesterov=True,
            adjust_lr=hp.adjust_lr,
            use_triton=(not cli_args.no_triton),
        )

    elif hp.optimizer == "dion_simple":
        assert device_mesh is None, f"{hp.optimizer} does not support device mesh"
        print0(f"Dion rank fraction: {hp.rank_fraction}")
        opt = DionSimple(
            param_groups,
            lr=hp.lr,
            mu=hp.mu,
            weight_decay=hp.weight_decay,
            rank=round(hp.rank_fraction * hp.model_dim),
            mixed_precision_config=dion_mixed_precision_config,
        )

    elif hp.optimizer == "muon_reference":
        print0(f"Muon LR adjust method: {hp.adjust_lr}")
        opt = MuonReference(
            param_groups,
            lr=hp.lr,
            mu=hp.mu,
            weight_decay=hp.weight_decay,
            nesterov=True,
            adjust_lr=hp.adjust_lr,
        )

    elif hp.optimizer == "adamw":
        print0("Using AdamW for all params, scalar optimizer will be ignored")
        print0("Setting all param groups to use unscaled base learning rate")
        for group in param_groups:
            group["lr"] = hp.lr
            group["betas"] = (0.9, 0.95)
        opt = torch.optim.AdamW(
            param_groups,
            lr=hp.lr,
            betas=(0.9, 0.95),
            weight_decay=hp.weight_decay,
        )

    else:
        raise ValueError(f"Unsupported optimizer: {hp.optimizer}")

    if hp.replicate_mesh_grad_sync and hp.optimizer not in ("dion", "dion_reference"):
        raise ValueError("replicate_mesh_grad_sync is set for non-Dion optimizer")
    if not hp.replicate_mesh_grad_sync and hp.optimizer in ("dion", "dion_reference"):
        print0("Warning: not using replicate_mesh_grad_sync for Dion optimizer")

    return opt
