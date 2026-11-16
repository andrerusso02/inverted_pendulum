"""Observation histories without device-to-host checks in the stepping path."""

from collections.abc import Sequence

import torch


class ObservationHistoryBuffer:
    """Isaac Lab circular-buffer interface for this task's observation histories.

    Reset indices are already known on the host (including their count). Keep
    them until the next append instead of discovering first samples with a GPU
    reduction and boolean indexing on every step. The first sample still fills
    the entire history of each reset environment.
    """

    def __init__(self, max_len: int, batch_size: int, device: str):
        if max_len < 1:
            raise ValueError("History length must be at least one.")
        self.max_length = max_len
        self.batch_size = batch_size
        self.device = device
        self._all_indices = torch.arange(batch_size, device=device)
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._pointer = -1
        self._buffer = None
        self._reset_all = True
        self._reset_indices: list[torch.Tensor] = []

    @property
    def current_length(self) -> torch.Tensor:
        return self._num_pushes.clamp(max=self.max_length)

    @property
    def buffer(self) -> torch.Tensor:
        """Return an independent oldest-to-newest history, with batch first."""
        if self._buffer is None:
            raise RuntimeError("Append a sample before reading history.")
        # roll already allocates a snapshot; a preceding clone doubles the copy.
        return torch.roll(self._buffer, self.max_length - self._pointer - 1, dims=0).transpose(0, 1)

    def reset(self, batch_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if batch_ids is None:
            self._num_pushes.zero_()
            if self._buffer is not None:
                self._buffer.zero_()
            self._reset_all = True
            self._reset_indices.clear()
        elif len(batch_ids) > 0:
            ids = torch.as_tensor(batch_ids, dtype=torch.long, device=self.device)
            self._num_pushes.index_fill_(0, ids, 0)
            if self._buffer is not None:
                self._buffer.index_fill_(1, ids, 0)
            if not self._reset_all:
                # Own the indices even if a caller reuses its reset tensor.
                self._reset_indices.append(ids.clone())

    def append(self, data: torch.Tensor) -> None:
        if data.shape[0] != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} samples, received {data.shape[0]}.")
        data = data.to(self.device)
        if self._buffer is None:
            self._buffer = torch.empty((self.max_length, *data.shape), dtype=data.dtype, device=self.device)
        self._pointer = (self._pointer + 1) % self.max_length
        if self._reset_all:
            self._buffer.copy_(data.unsqueeze(0).expand_as(self._buffer))
            self._reset_all = False
        else:
            self._buffer[self._pointer].copy_(data)
            for ids in self._reset_indices:
                initial = data.index_select(0, ids).unsqueeze(0).expand(self.max_length, -1, *data.shape[1:])
                self._buffer.index_copy_(1, ids, initial)
        self._reset_indices.clear()
        self._num_pushes.add_(1)

    def __getitem__(self, lag: torch.Tensor) -> torch.Tensor:
        if len(lag) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} delays, received {len(lag)}.")
        if self._buffer is None or self._reset_all or self._reset_indices:
            raise RuntimeError("Append a sample after reset before reading delayed history.")
        valid_lag = torch.minimum(lag, self._num_pushes - 1)
        slots = (self._pointer - valid_lag) % self.max_length
        return self._buffer[slots, self._all_indices]
