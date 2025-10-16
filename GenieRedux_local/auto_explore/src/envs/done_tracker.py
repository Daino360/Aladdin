"""
done_tracker.py: tiny utility to track per-env 'done' status for vectorized envs.

State encoding per environment index:
- 0 -> not done yet (still running)
- 1 -> just became done on the current update
- 2 -> was already done on a previous update (post-terminal)

It provides counters and boolean masks derived from this state.
"""
import numpy as np


class DoneTrackerEnv:
    def __init__(self, num_envs: int) -> None:
        """Monitor env dones: 0 when not done, 1 when done, 2 when already done."""
        """
        Create a tracker for `num_envs` parallel environments.
        Initializes all slots to 0 (not done).
        """        
        self.num_envs = num_envs
        self.done_tracker = None
        self.reset_done_tracker()

    def reset_done_tracker(self) -> None:
        """Reset all environments to the 'not done' state (0)."""
        self.done_tracker = np.zeros(self.num_envs, dtype=np.uint8)

    def update_done_tracker(self, done: np.ndarray) -> None:
        """
        Update the tracker with a boolean/0-1 array `done` from the env step.

        Transition rule (per index): new_state = clamp(2*old_state + done, 0, 2)
        - 0 + 0 -> 0  (still running)
        - 0 + 1 -> 1  (just became done)
        - 1 + x -> 2  (moves to 'already done' thereafter)
        - 2 + x -> 2  (stays 'already done')
        """        
        self.done_tracker = np.clip(2 * self.done_tracker + done, 0, 2)

    @property
    def num_envs_done(self) -> int:
        """Number of envs that are done now or were done before (state > 0)."""
        return (self.done_tracker > 0).sum()

    @property
    def mask_dones(self) -> np.ndarray:
        """
        Boolean mask over all envs: True where the env is NOT done (state == 0),
        False where it is newly or already done (state 1 or 2).
        """
        return np.logical_not(self.done_tracker)

    @property
    def mask_new_dones(self) -> np.ndarray:
        """
        Boolean mask over the subset of envs with state <= 1
        (i.e., still running or just became done). Within that subset,
        True marks running (0), False marks newly done (1).
        Note: envs already done (state 2) are excluded from this array.
        """
        return np.logical_not(self.done_tracker[self.done_tracker <= 1])
