import logging
from typing import Optional, Sequence

import torch
import torch.nn as nn


class BlockSwapManager:
    """Stream transformer blocks between CPU and GPU to reduce peak VRAM."""

    def __init__(self, blocks: Sequence[nn.Module], device, chunk_size: int = 1) -> None:
        if not blocks:
            raise ValueError("BlockSwapManager requires at least one block.")
        self.blocks = list(blocks)
        self.device = self._normalize_device(device)
        self.chunk_size = max(1, min(int(chunk_size), len(self.blocks)))
        self.enabled = False
        self._active_chunk_start: Optional[int] = None
        self._blocks_remaining_in_chunk: int = 0

    @staticmethod
    def _normalize_device(device) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            if device.isdigit():
                return torch.device(f"cuda:{device}")
            return torch.device(device)
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        raise TypeError(f"Unsupported device specifier: {device!r}")

    def configure(self, enabled: bool, chunk_size: int) -> None:
        chunk_size = max(1, int(chunk_size))
        chunk_size = min(chunk_size, len(self.blocks))
        if enabled and not torch.cuda.is_available():
            logging.warning("BlockSwap requested but CUDA is unavailable; disabling.")
            enabled = False
        state_changed = enabled != self.enabled or chunk_size != self.chunk_size
        self.chunk_size = chunk_size
        if not state_changed:
            return
        if enabled:
            self._move_all_to_cpu()
            self.enabled = True
        else:
            self._move_all_to_device()
            self.enabled = False
        self._active_chunk_start = None
        self._blocks_remaining_in_chunk = 0

    def disable(self) -> None:
        self.configure(enabled=False, chunk_size=self.chunk_size)

    def start_pass(self) -> None:
        if not self.enabled:
            return
        self._active_chunk_start = None
        self._blocks_remaining_in_chunk = 0

    def finish_pass(self) -> None:
        if not self.enabled:
            return
        if self._active_chunk_start is not None:
            self._move_chunk_to_cpu(self._active_chunk_start)
            self._active_chunk_start = None
        self._blocks_remaining_in_chunk = 0

    def activate(self, index: int) -> nn.Module:
        block = self.blocks[index]
        if not self.enabled:
            return block
        chunk_start = (index // self.chunk_size) * self.chunk_size
        if chunk_start != self._active_chunk_start:
            self._switch_to_chunk(chunk_start)
        return self.blocks[index]

    def release(self, index: int) -> None:
        if not self.enabled or self._active_chunk_start is None:
            return
        self._blocks_remaining_in_chunk -= 1
        if self._blocks_remaining_in_chunk <= 0:
            self._move_chunk_to_cpu(self._active_chunk_start)
            self._active_chunk_start = None

    # Internal helpers -------------------------------------------------
    def _switch_to_chunk(self, chunk_start: int) -> None:
        if self._active_chunk_start is not None:
            self._move_chunk_to_cpu(self._active_chunk_start)
        self._move_chunk_to_device(chunk_start)
        self._active_chunk_start = chunk_start
        chunk_end = min(chunk_start + self.chunk_size, len(self.blocks))
        self._blocks_remaining_in_chunk = chunk_end - chunk_start

    def _move_chunk_to_device(self, chunk_start: int) -> None:
        chunk_end = min(chunk_start + self.chunk_size, len(self.blocks))
        for idx in range(chunk_start, chunk_end):
            self.blocks[idx] = self.blocks[idx].to(device=self.device, non_blocking=True)

    def _move_chunk_to_cpu(self, chunk_start: int) -> None:
        chunk_end = min(chunk_start + self.chunk_size, len(self.blocks))
        for idx in range(chunk_start, chunk_end):
            self.blocks[idx] = self.blocks[idx].to("cpu", non_blocking=True)

    def _move_all_to_cpu(self) -> None:
        for idx, block in enumerate(self.blocks):
            self.blocks[idx] = block.to("cpu", non_blocking=True)

    def _move_all_to_device(self) -> None:
        for idx, block in enumerate(self.blocks):
            self.blocks[idx] = block.to(device=self.device, non_blocking=True)

