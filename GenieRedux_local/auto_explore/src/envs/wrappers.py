"""
Credits to https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""
"""
wrappers.py: lightweight Atari-style wrappers to preprocess gymnasium environments.

- make_atari(): convenience factory that builds an env and stacks common wrappers:
  resize frames, optional reward clipping, time limit, random no-ops on reset,
  frame skipping with max-pool, and (optionally) episodic life handling.
- ResizeObsWrapper: resizes observations to (size, size) in HWC uint8.
- RewardClippingWrapper: maps rewards to {-1, 0, +1}.
- NoopResetEnv: does a random number of NOOP actions on reset for state randomization.
- EpisodicLifeEnv: treats life loss as an episode end (resets only on true game over).
- MaxAndSkipEnv: repeats the same action for K frames, sums rewards, and max-pools
  the last two frames to reduce flicker.
"""
from typing import Tuple

import gymnasium as gym
import numpy as np
from PIL import Image

from data_generation.generator.connector_retro_act import make_retro

def make_atari(id, size=64, max_episode_steps=None, noop_max=30, frame_skip=4, done_on_life_loss=False, clip_reward=False, full_action_space=True, repeat_action_probability=0, frameskip=1):
    env = gym.make(id, full_action_space=full_action_space, repeat_action_probability=repeat_action_probability, frameskip=frameskip)
    """
    Build a preprocessed Atari-like env.

    Pipeline
    --------
    gym.make -> ResizeObsWrapper -> (RewardClippingWrapper) -> (TimeLimit)
              -> (NoopResetEnv) -> MaxAndSkipEnv -> (EpisodicLifeEnv)

    Parameters
    ----------
    id : str
        Gymnasium env id (e.g., "ALE/Breakout-v5").
    size : int
        Target square resolution for observations (size x size).
    max_episode_steps : int or None
        If set, wrap with TimeLimit.
    noop_max : int or None
        If set, perform 1..noop_max NOOPs on reset.
    frame_skip : int
        Repeat action for this many frames in MaxAndSkipEnv.
    done_on_life_loss : bool
        If True, life loss triggers done=True (DeepMind-style).
    clip_reward : bool
        If True, clip rewards to {-1, 0, +1}.
    full_action_space, repeat_action_probability, frameskip
        Passed to gym.make for ALE settings.

    Returns
    -------
    gym.Env
        Wrapped environment.
    """
    # assert 'NoFrameskip' in env.spec.id or 'Frameskip' not in env.spec
    env = ResizeObsWrapper(env, (size, size))
    if clip_reward:
        env = RewardClippingWrapper(env)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if noop_max is not None:
        env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    if done_on_life_loss:
        env = EpisodicLifeEnv(env)
    return env


class ResizeObsWrapper(gym.ObservationWrapper):
    """Resize observations to a fixed (H, W) using bilinear interpolation (keeps 3 channels)."""
    def __init__(self, env: gym.Env, size: Tuple[int, int]) -> None:
        """
        Parameters
        ----------
        env : gym.Env
            Base environment.
        size : (int, int)
            Output (H, W) for resized observations.
        """
        gym.ObservationWrapper.__init__(self, env)
        self.size = tuple(size)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8)
        self.unwrapped.original_obs = None

    def resize(self, obs: np.ndarray):
        """Resize a single frame (HWC uint8) to target size."""
        img = Image.fromarray(obs)
        img = img.resize(self.size, Image.BILINEAR)
        return np.array(img)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Store original frame (for debugging) and return the resized frame."""
        self.unwrapped.original_obs = observation
        return self.resize(observation)


class RewardClippingWrapper(gym.RewardWrapper):
    """Clip rewards to their sign: -1, 0, or +1."""
    def reward(self, reward):
        """Return np.sign(reward) for stability across games."""
        return np.sign(reward)


class NoopResetEnv(gym.Wrapper):
    """
    On reset, perform a random number of NOOP (action 0) steps to randomize starts.
    Assumes action 0 is 'NOOP' in env.get_action_meanings().
    """
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        """
        Parameters
        ----------
        env : gym.Env
            Base environment.
        noop_max : int
            Upper bound of uniform [1, noop_max] NOOPs on reset.
        """        
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        """
        Reset the env, then do a random number of NOOP steps (1..noop_max).
        If a terminal happens during NOOPs, reset again.
        """        
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, terminated, truncated, _ = self.env.step(self.noop_action)
            done = terminated or truncated
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        """Pass through to the base env step."""
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make loss of life act like end of episode for value learning,
    but only reset the env on a true game over.
    """    
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        """Track lives and distinguish life-loss terminals from true terminals."""
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        """
        Step the env; convert a life loss into done=True (unless lives==0).
        Keep real termination state in was_real_done.
        """        
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        """
        Only reset on true game over; otherwise advance one NOOP step
        to leave the terminal life-loss state.
        """        
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat the same action for `skip` frames; sum rewards and max-pool last 2 frames."""

    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        """
        Parameters
        ----------
        env : gym.Env
            Base environment.
        skip : int
            Number of repeated steps per action.
        """        
        gym.Wrapper.__init__(self, env)
        assert skip > 0
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        """
        Repeat `action` for `skip` frames, accumulate reward, and
        return the max over the last two frames to reduce flicker.
        """        
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        self.max_frame = self._obs_buffer.max(axis=0)

        return self.max_frame, total_reward, done, info

    def reset(self, **kwargs):
        """Pass-through reset to the base env."""
        return self.env.reset(**kwargs)
