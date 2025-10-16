"""
episode.py: container and utilities for trajectory data.

- Defines `Episode`, a dataclass that stores a single trajectory:
  observations, actions, rewards, terminal flags (`ends`), padding mask,
  and an optional `condition` (e.g., env identifier).
- On construction, it auto-truncates the trajectory at the first terminal step.
- Provides helpers to:
    • `merge` two consecutive chunks of the same episode,
    • take a `segment` (with optional left/right padding),
    • `compute_metrics` (length and total return),
    • `save` to disk.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class EpisodeMetrics:
    """
    Lightweight summary of an episode.
    - episode_length: number of time steps.
    - episode_return: sum of rewards over the episode.
    """
    episode_length: int
    episode_return: float


@dataclass
class Episode:
    """
    Stores a trajectory as tensors.

    Fields
    -------
    observations : torch.ByteTensor
        Shape (T, C, H, W). Raw frames (uint8) in channel-first format.
    actions : torch.LongTensor
        Shape (T,). Discrete action ids.
    rewards : torch.FloatTensor
        Shape (T,). Reward per time step.
    ends : torch.LongTensor
        Shape (T,). Binary flags; 1 at terminal step, else 0.
    mask_padding : torch.BoolTensor
        Shape (T,). True for real data; padding steps set to False in segments.
    condition : torch.IntTensor
        Arbitrary conditioning vector (e.g., one-hot env id), broadcast per step.
    """
    observations: torch.ByteTensor
    actions: torch.LongTensor
    rewards: torch.FloatTensor
    ends: torch.LongTensor
    mask_padding: torch.BoolTensor
    condition: torch.IntTensor

    def __post_init__(self):
        """
        Validate equal lengths across all time-series fields and
        truncate the episode at the first terminal (if any).

        After truncation, keeps tensors in sync so the first 1 in `ends`
        becomes the final step included in the episode.
        """
        assert len(self.observations) == len(self.actions) == len(self.rewards) == len(self.ends) == len(self.mask_padding)
        if self.ends.sum() > 0: # if there's at least one terminal step (ends=1)
            idx_end = torch.argmax(self.ends) + 1  # include the terminal step
            self.observations = self.observations[:idx_end]
            self.actions = self.actions[:idx_end]
            self.rewards = self.rewards[:idx_end]
            self.ends = self.ends[:idx_end]
            self.mask_padding = self.mask_padding[:idx_end]

    def __len__(self) -> int:
        """
        Number of time steps (T) in the episode after any truncation.
        """
        return self.observations.size(0)

    def merge(self, other: Episode) -> Episode:
        """
        Concatenate another episode chunk to the end of this one along time.

        Requires the same `condition`. Does not insert padding; assumes `other`
        directly follows `self` in time.

        Returns
        -------
        Episode
            New episode with fields concatenated along the first dimension.
        """
        assert torch.all(other.condition == self.condition)
        return Episode(
            torch.cat((self.observations, other.observations), dim=0),
            torch.cat((self.actions, other.actions), dim=0),
            torch.cat((self.rewards, other.rewards), dim=0),
            torch.cat((self.ends, other.ends), dim=0),
            torch.cat((self.mask_padding, other.mask_padding), dim=0),
            self.condition,
        )

    def segment(self, start: int, stop: int, should_pad: bool = False) -> Episode:
        """
        Extract a [start, stop) sub-trajectory, with optional zero-padding
        if the requested range goes out of bounds.

        Parameters
        ----------
        start : int
            Inclusive start index (can be negative if padding is enabled).
        stop : int
            Exclusive stop index (can exceed len(self) if padding is enabled).
        should_pad : bool
            If True, pads on the left/right to exactly match the requested
            length when indices fall outside episode bounds.

        Returns
        -------
        Episode
            A new episode segment with the same `condition`. The `mask_padding`
            is updated so padded steps are False while real steps remain True.
        """
        assert start < len(self) and stop > 0 and start < stop
        padding_length_right = max(0, stop - len(self))
        padding_length_left = max(0, -start)
        # Either both paddings are zero, or padding is explicitly allowed.
        assert padding_length_right == padding_length_left == 0 or should_pad

        def pad(x):
            """
            Build F.pad args for time dimension (last dim of shape (T, ...)):
            For an N-D tensor, F.pad expects paddings from last dim backward.
            We add zeros for non-time dims and pad only along time.
            """
            pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
            return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

        # Clip to valid range for slicing; padding (if requested) fixes length.
        start = max(0, start)
        stop = min(len(self), stop)
        segment = Episode(
            self.observations[start:stop],
            self.actions[start:stop],
            self.rewards[start:stop],
            self.ends[start:stop],
            self.mask_padding[start:stop],
            self.condition,
        )
        
        # Pad time dimension to requested length; update mask_padding accordingly.
        segment.observations = pad(segment.observations)
        segment.actions = pad(segment.actions)
        segment.rewards = pad(segment.rewards)
        segment.ends = pad(segment.ends)
        segment.mask_padding = torch.cat((torch.zeros(padding_length_left, dtype=torch.bool), segment.mask_padding, torch.zeros(padding_length_right, dtype=torch.bool)), dim=0)

        return segment

    def compute_metrics(self) -> EpisodeMetrics:
        """
        Compute simple episode metrics: length and return (sum of rewards).
        """
        return EpisodeMetrics(len(self), self.rewards.sum())

    def save(self, path: Path) -> None:
        """
        Persist the episode to disk as a PyTorch .pt file by saving the
        internal dictionary (`__dict__`). Use `Episode(**torch.load(path))`
        to restore.
        """
        torch.save(self.__dict__, path)
