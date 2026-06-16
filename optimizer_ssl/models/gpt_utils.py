# Adapted from nanoGPT/llm.c-style GPT training code and modified for
# optimizer-induced spectral telemetry experiments. See NOTICE.md for
# attribution and license details.

"""Data-loading utilities for GPT training.

This file is adapted from nanoGPT/llm.c-style training utilities and modified
for the Optimizer-SSL experiments. See the top-level NOTICE.md for attribution.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

from optimizer_ssl.data.binary_shards import load_headered_token_shard


class DistributedDataLoader:
    """Simple strided data-parallel loader for headered token shards.

    Each data-parallel rank starts at a different offset and advances by the
    global data-parallel stride. Tensor-parallel ranks should share the same
    data-parallel rank so they consume identical batches.
    """

    def __init__(
        self,
        filename_pattern: str,
        B: int,
        T: int,
        dp_rank: int,
        dp_world_size: int,
        device: str | torch.device | None = None,
    ):
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.B = B
        self.T = T
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.files = sorted(glob.glob(filename_pattern))
        if len(self.files) < 1:
            raise FileNotFoundError(
                f"Could not find any files matching the pattern {filename_pattern!r}"
            )

        self.reset()

    def _load_tokens_for_current_shard(self) -> None:
        self.tokens = np.asarray(load_headered_token_shard(self.files[self.current_shard]))

    def reset(self) -> None:
        self.current_shard = 0
        self.current_position = self.dp_rank * self.B * self.T
        self._load_tokens_for_current_shard()

    def advance(self) -> None:
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.dp_rank * self.B * self.T
        self._load_tokens_for_current_shard()

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B * T * self.dp_world_size
        if self.current_position + (B * T * self.dp_world_size + 1) > len(self.tokens):
            self.advance()
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def state_dict(self) -> dict:
        return {
            f"current_shard_rank_{self.dp_rank}": self.current_shard,
            f"current_position_rank_{self.dp_rank}": self.current_position,
            "dataloader_world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict_world_size = state_dict.get("dataloader_world_size")
        if state_dict_world_size != self.dp_world_size:
            raise NotImplementedError(
                "DistributedDataLoader does not support redistributing checkpoints to a "
                "different world size. Current process has world size "
                f"{self.dp_world_size}, but checkpoint has size {state_dict_world_size}."
            )

        self.current_shard = state_dict[f"current_shard_rank_{self.dp_rank}"]
        self.current_position = state_dict[f"current_position_rank_{self.dp_rank}"]
        self._load_tokens_for_current_shard()
