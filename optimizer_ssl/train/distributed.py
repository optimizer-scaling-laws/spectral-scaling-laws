"""Distributed-runtime helpers for training."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DeviceMesh

from optimizer_ssl.train.experiment_logging import print0, set_master_process


def init_distributed(dp_size, fs_size, tp_size) -> Optional[DeviceMesh]:
    """Initialize DeviceMesh or ProcessGroup for distributed training.

    If all mesh dimensions are None, this defaults to ordinary DDP.
    """
    assert torch.cuda.is_available(), "CUDA must be available"
    assert torch.distributed.is_available(), "Distributed must be available"

    assert all(
        var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
    ), "This script must be launched using the 'torchrun' command."
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    set_master_process(rank == 0)

    mesh_dims = (dp_size, fs_size, tp_size)
    if all(d is None for d in mesh_dims):
        device_mesh = None
        dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)

        print0("=" * 80)
        print0("Distributed training initialized with DDP")
        print0(f"World size: {world_size}")
    else:
        assert all(
            d is not None for d in mesh_dims
        ), f"All mesh dimensions (dp_size, fs_size, tp_size) must be specified, but got ({dp_size}, {fs_size}, {tp_size})"

        total_gpus = dp_size * fs_size * tp_size
        assert world_size == total_gpus, (
            f"World size {world_size} does not match expected size {total_gpus} "
            f"(DP {dp_size}, FS {fs_size}, TP {tp_size})"
        )
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(dp_size, fs_size, tp_size),
            mesh_dim_names=("dp", "fs", "tp"),
        )

        print0("=" * 80)
        print0("Distributed training initialized with DeviceMesh")
        print0(f"World size: {world_size}")
        print0(f"DP size: {dp_size}")
        print0(f"FS size: {fs_size}")
        print0(f"TP size: {tp_size}")
        print0(device_mesh)

    return device_mesh


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
