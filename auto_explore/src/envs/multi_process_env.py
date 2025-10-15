"""
multiprocess_env.py: vectorized environment runner using Python processes.

- Spawns one child process per env. Each child holds its own env instance and
  communicates with the parent via a Pipe using (MessageType, payload) pairs.
- Supports batched reset() / step() across all workers, optional per-frame
  transform, and returns observations in HWC format (N, H, W, C).
- Inherits from DoneTrackerEnv to track which envs are running/newly done/already done,
  and exposes should_reset() to tell callers when enough envs have finished to roll over.
"""

from dataclasses import astuple, dataclass
from enum import Enum
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Callable, Iterator, List, Optional, Tuple
from lovely_numpy import lo
import numpy as np
from einops import rearrange

from tools.logger import getLogger
log = getLogger(__name__)

from .done_tracker import DoneTrackerEnv
from .retrowrapper import RetroWrapper, set_retro_make
import pickle


class MessageType(Enum):
    """Types of parent↔child messages exchanged over the Pipes."""
    RESET = 0
    RESET_RETURN = 1
    STEP = 2
    STEP_RETURN = 3
    CLOSE = 4


@dataclass
class Message:
    """Lightweight (type, content) envelope for Pipe communication."""
    type: MessageType
    content: Optional[Any] = None

    def __iter__(self) -> Iterator:
        """Allow tuple-unpacking on the receiving side: 'msg_type, payload = msg'."""        
        return iter(astuple(self))


def child_env(child_id: int, env_fn: Callable, child_conn: Connection) -> None:
    """
    Worker process loop: owns a single env instance and responds to commands.

    - Seeds NumPy uniquely per child.
    - On RESET: env.reset() and return first observation.
    - On STEP: env.step(action), compute 'done' from terminated|truncated.
      If done, immediately reset so the next frame is a fresh start.
    - On CLOSE: cleanly close the Pipe and exit.
    """
    np.random.seed(child_id + np.random.randint(0, 2 ** 31 - 1))
    env = env_fn()
    while True:
        message_type, content = child_conn.recv()
        if message_type == MessageType.RESET:
            obs = env.reset()
            child_conn.send(Message(MessageType.RESET_RETURN, obs))
        elif message_type == MessageType.STEP:
            obs, rew, terminated, truncated, _ = env.step(content)
            done = terminated or truncated
            if terminated or truncated:
                obs = env.reset()
            child_conn.send(Message(MessageType.STEP_RETURN, (obs, rew, done, None)))
        elif message_type == MessageType.CLOSE:
            child_conn.close()
            return
        else:
            raise NotImplementedError

def process_obs_np(obs, transform):
    """
    Apply a callable 'transform' to each observation in a sequence (numpy arrays).
    Returns a Python list of transformed frames.
    """
    new_obs = []
    for ob in obs:
        ob = transform(ob)
        new_obs.append(ob)
    return new_obs

class MultiProcessEnv(DoneTrackerEnv):
    """
    Parent-side vectorized env wrapper backed by multiple worker processes.

    Parameters
    ----------
    env_fn : Callable[[], gym.Env-like]
        Factory that builds a *fresh* environment instance (no args).
    num_envs : int
        Number of parallel envs / worker processes.
    should_wait_num_envs_ratio : float
        Fraction of envs that must be done before should_reset() returns True.
    transform : callable or None
        Optional transform applied per frame; expected to output CHW, which
        is then rearranged back to HWC for the batch.
    """
    def __init__(self, env_fn: Callable, num_envs: int, should_wait_num_envs_ratio: float, transform: int|None =None) -> None:
        super().__init__(num_envs)
        self.transform = transform

        # Probe action space size once
        self.num_actions = env_fn().action_space.n
        # self.num_actions = env_fn().env.action_space.n
        self.should_wait_num_envs_ratio = should_wait_num_envs_ratio
        
        # Create Pipes and fork worker processes
        self.processes, self.parent_conns = [], []
        for child_id in range(num_envs):
            parent_conn, child_conn = Pipe()
            self.parent_conns.append(parent_conn)
            p = Process(target=child_env, args=(child_id, env_fn, child_conn), daemon=True)
            self.processes.append(p)
        for p in self.processes:
            p.start()

    def should_reset(self) -> bool:
        """
        Return True when the fraction of finished envs ≥ should_wait_num_envs_ratio.
        Upstream code can use this to decide when to flush/rotate episodes.
        """        
        return (self.num_envs_done / self.num_envs) >= self.should_wait_num_envs_ratio

    def _receive(self, check_type: Optional[MessageType] = None) -> List[Any]:
        """
        Blocking receive from all children. Optionally checks all message types match.

        Returns
        -------
        list
            List of .content payloads, one per child.
        """
        messages = [parent_conn.recv() for parent_conn in self.parent_conns]
        if check_type is not None:
            assert all([m.type == check_type for m in messages])
        return [m.content for m in messages]

    def reset(self) -> np.ndarray:
        """
        Reset all workers and return a stacked batch of initial observations (N, H, W, C).

        Notes
        -----
        - Some envs may return tuples; this extracts the first element per child.
        - If a transform is provided, it is applied per-frame and converted back to HWC.
        """
        self.reset_done_tracker()
        for parent_conn in self.parent_conns:
            parent_conn.send(Message(MessageType.RESET))
        content = self._receive(check_type=MessageType.RESET_RETURN)
        # Unwrap obs if returned as (obs, ...) tuples
        content = [c[0] for c in content]
        # Optional transform (expects CHW; convert back to HWC)
        if self.transform is not None:
            for idx in range(len(content)):
                temp = np.expand_dims(np.array(content[idx]), axis=0)
                temp = process_obs_np(temp, self.transform)
                content[idx] = rearrange(temp, 't c h w -> t h w c')[0]
        return np.stack(content)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Step all workers with the provided actions.

        Parameters
        ----------
        actions : np.ndarray
            One action per env. Shape is whatever the underlying env expects
            (e.g., one-hot vector).

        Returns
        -------
        obs : np.ndarray
            Batched observations (N, H, W, C), optionally transformed.
        rew : np.ndarray
            Per-env rewards, shape (N,).
        done : np.ndarray
            Per-env done flags, shape (N,) (True when terminated or truncated).
        info : Any
            Placeholder, currently always None.
        """
        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(Message(MessageType.STEP, action))
        content = self._receive(check_type=MessageType.STEP_RETURN)
        obs, rew, done, _ = zip(*content)
        # Normalize obs that may come wrapped in tuples per child
        if isinstance(obs, tuple):
            obs = list(obs)
        for idx in range(len(obs)):
            if isinstance(obs[idx], tuple):
                obs[idx] = obs[idx][0]
        done = np.stack(done)
        self.update_done_tracker(done)
        
        # Optional transform per observation; ensure HWC output
        new_obs = []
        if self.transform is not None:
            for idx in range(len(obs)):
                try:
                    temp = np.expand_dims(np.array(obs[idx]), axis=0)
                    temp = process_obs_np(temp, self.transform)
                    temp = rearrange(temp, 't c h w -> t h w c')[0]
                    new_obs.append(temp)
                except Exception as e:
                    log.e("Wut", idx, obs)
                    #save obs[idx] to file
                    with open(f'/home/nedko_savov/projects/ivg/external/open-genie/obs_{idx}.pkl', 'wb') as f:
                        pickle.dump(obs[idx], f)
                    for o in obs[idx]:
                        log.e("W", len(o))
                        log.e("H", len(o[0]))
                        
                    log.e(f"Error in step: {e}")
                    raise e
        else:
            log.e("No transform")
            new_obs = obs
        
        # Stack all env observations into a batch
        try:
            new_obs = np.stack(new_obs)
        except Exception as e:
            log.e("Stack fail", new_obs)
            
            raise Exception(e)
        return new_obs, np.stack(rew), done, None

    def close(self) -> None:
        """
        Cleanly shut down all workers: send CLOSE, join processes, and close Pipes.
        """
        for parent_conn in self.parent_conns:
            parent_conn.send(Message(MessageType.CLOSE))
        for p in self.processes:
            p.join()
        for parent_conn in self.parent_conns:
            parent_conn.close()
