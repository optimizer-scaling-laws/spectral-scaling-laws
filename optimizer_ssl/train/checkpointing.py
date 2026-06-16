"""Distributed checkpoint save/load helpers."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from optimizer_ssl.models.gpt_utils import DistributedDataLoader
from optimizer_ssl.train.experiment_logging import print0


class CheckpointManager:
    """Small wrapper around torch.distributed.checkpoint for training state."""

    def __init__(
        self,
        checkpoint_dir: str | None,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DistributedDataLoader,
        val_loader: DistributedDataLoader,
        wandb_id: Optional[str] = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wandb_id = wandb_id
        self.step = None
        self.DEFAULT_NAME = "checkpoint"

    def _get_state_dict(self) -> dict:
        model_state, opt_state = get_state_dict(self.model, self.optimizer)
        state_dict = {
            "model": model_state,
            "optimizer": opt_state,
            "train_loader": self.train_loader.state_dict(),
            "val_loader": self.val_loader.state_dict(),
            "step": self.step,
            "wandb_id": self.wandb_id,
        }
        return state_dict

    def save(self, name: Optional[str] = None, step: Optional[int] = None):
        """Save a sharded distributed checkpoint under checkpoint_dir/name/."""
        assert self.checkpoint_dir, "Checkpoint directory must be specified"
        self.step = step
        name = name or self.DEFAULT_NAME
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        print0(f"Saving checkpoint to {checkpoint_path}")

        tmpdir = None
        if dist.get_rank() == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            tmpdir = tempfile.mkdtemp(dir=self.checkpoint_dir)

        obj_list = [tmpdir]
        dist.broadcast_object_list(obj_list, src=0)
        tmpdir = obj_list[0]

        state_dict = self._get_state_dict()
        dcp.save(state_dict, checkpoint_id=tmpdir)
        dist.barrier()

        if dist.get_rank() == 0:
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)
            elif os.path.isdir(checkpoint_path):
                shutil.rmtree(checkpoint_path, ignore_errors=True)
            shutil.move(tmpdir, checkpoint_path)
        dist.barrier()

    def load(self, name: Optional[str] = None, allow_missing: bool = False):
        """Load a sharded distributed checkpoint from checkpoint_dir/name/."""
        assert self.checkpoint_dir, "Checkpoint directory must be specified"
        name = name or self.DEFAULT_NAME
        checkpoint_path = os.path.join(self.checkpoint_dir, name)

        if not os.path.isdir(checkpoint_path):
            if allow_missing:
                print0(f"Checkpoint {checkpoint_path} does not exist, skipping load")
                return
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")

        print0(f"Loading checkpoint from {checkpoint_path}")
        state_dict = self._get_state_dict()
        dcp.load(state_dict, checkpoint_id=checkpoint_path)

        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )

        self.train_loader.load_state_dict(state_dict["train_loader"])
        self.val_loader.load_state_dict(state_dict["val_loader"])

        self.step = state_dict["step"]
        self.wandb_id = state_dict["wandb_id"]
        dist.barrier()
