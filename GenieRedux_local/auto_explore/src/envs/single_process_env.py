from typing import Any, Tuple

import numpy as np

from .done_tracker import DoneTrackerEnv
"""
single_process_env.py: minimal wrapper to run a single env with 'done' tracking.

- Wraps one environment instance and exposes a vectorized-like API:
  * reset() returns observations with a leading batch dim (1, ...).
  * step() expects an action shaped for a batch of size 1 and returns
    (obs, reward, done, info) all with a leading batch dim.
- Inherits from DoneTrackerEnv to keep track of when the episode finishes.
"""

class SingleProcessEnv(DoneTrackerEnv):
    def __init__(self, env_fn):
        """
        Build a single environment and initialize 'done' tracking.

        Parameters
        ----------
        env_fn : Callable[[], gym.Env-like]
            Factory function that returns a fresh environment instance.
        """
        super().__init__(num_envs=1)
        self.env = env_fn()
        self.num_actions = self.env.action_space.n

    def should_reset(self) -> bool:
        """
        Whether this env should be reset (True once the single env is done).
        """
        return self.num_envs_done == 1

    def reset(self) -> np.ndarray:
        """
        Reset the environment and return the initial observation with a batch dim.

        Returns
        -------
        np.ndarray
            Observation shaped as (1, H, W, C) or (1, ...) depending on the env.
        """        
        self.reset_done_tracker()
        obs = self.env.reset()
        return obs[None, ...]

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : np.ndarray
            Action for the single env; expected shape (1, ...) to mimic batching.

        Returns
        -------
        obs : np.ndarray
            Next observation with leading batch dim (1, ...).
        reward : np.ndarray
            Reward array of shape (1,).
        done : np.ndarray
            Done flag array of shape (1,) with boolean/0-1.
        info : Any
            Placeholder; always None here.
        """        
        obs, reward, done, _ = self.env.step(action[0])  # action is supposed to be ndarray (1,)
        done = np.array([done])
        self.update_done_tracker(done)
        return obs[None, ...], np.array([reward]), done, None

    def render(self) -> None:
        """Render the underlying environment (passthrough)."""
        self.env.render()

    def close(self) -> None:
        """Close the underlying environment (passthrough)."""
        self.env.close()
