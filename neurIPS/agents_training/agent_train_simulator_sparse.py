#!/usr/bin/env python3
"""
PPO training directly inside Procgen CoinRun with a binary goal reward.

- Same CLI/monitoring ergonomics as agent_train_simulator.py.
- Flow odometry still tracks goal progress, but the agent only receives
  reward 10 when the goal is reached and 0 otherwise.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple
import time

import cv2
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import imageio
import numpy as np

from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecTransposeImage,
)

from coinrun_simulator import ActionRemapWrapper, obs_to_hwc_uint8, make_env

np.bool8 = bool  # SB3 / NumPy compat

# =========================
#  Run directory helper
# =========================
def make_run_dir(base_logdir: str, agent_name: str) -> str:
    agent_root = os.path.join(base_logdir, agent_name)
    os.makedirs(agent_root, exist_ok=True)
    idx = 1
    while True:
        run_dir = os.path.join(agent_root, f"{agent_name}_{idx:02d}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            os.makedirs(os.path.join(run_dir, "tb_snaps"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "tb_train"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "tb_vis"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
            return run_dir
        idx += 1


# =========================
#  Flow reward components
# =========================
def _downscale_to_longside(gray: np.ndarray, longside: int) -> np.ndarray:
    H, W = gray.shape
    if longside is None or max(H, W) == longside:
        return gray
    s = float(longside) / float(max(H, W))
    new_size = (max(8, int(W * s)), max(8, int(H * s)))
    return cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)


def _flow_residual(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    *,
    longside: int = 64,
    baseline: str = "sky",
) -> Tuple[np.ndarray, np.ndarray]:
    p = _downscale_to_longside(prev_gray, longside)
    n = _downscale_to_longside(next_gray, longside)
    flow = cv2.calcOpticalFlowFarneback(p, n, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    fx, fy = flow[..., 0], flow[..., 1]
    if baseline == "global":
        fx -= np.median(fx)
        fy -= np.median(fy)
    elif baseline == "sky":
        h = fx.shape[0]
        top = slice(0, max(1, int(0.2 * h)))
        fx -= np.median(fx[top, :])
        fy -= np.median(fy[top, :])
    elif baseline == "none":
        pass
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    return fx, fy


def _roi_weight(
    H: int,
    W: int,
    *,
    bottom_bias: float,
    roi: Tuple[float, float, float, float],
) -> np.ndarray:
    row = np.linspace(0, 1, H, dtype=np.float32)
    wv = 1.0 + (row**2) * (bottom_bias - 1.0)
    wv /= max(np.mean(wv), 1e-9)
    weight = np.repeat(wv[:, None], W, axis=1)
    y0, y1, x0, x1 = roi
    iy0, iy1 = int(y0 * H), int(y1 * H)
    ix0, ix1 = int(x0 * W), int(x1 * W)
    mask = np.zeros_like(weight, dtype=np.float32)
    mask[iy0:iy1, ix0:ix1] = 1.0
    return weight * mask


@dataclass
class BinaryGoalRewardConfig:
    roi: Tuple[float, float, float, float]
    longside: int
    bottom_bias: float
    flow_baseline: str
    goal_x: float
    goal_y: float
    goal_y_weight: float
    goal_radius: float
    dx_clip: float
    dx_smooth: float
    success_reward: float


def compute_binary_goal_reward(
    prev_rgb: np.ndarray,
    curr_rgb: np.ndarray,
    *,
    cfg: BinaryGoalRewardConfig,
    state: dict,
) -> Tuple[float, float]:
    p = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    c = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
    fx_res, fy_res = _flow_residual(
        p,
        c,
        longside=cfg.longside,
        baseline=cfg.flow_baseline,
    )
    Hf, Wf = fx_res.shape
    weight = _roi_weight(Hf, Wf, bottom_bias=cfg.bottom_bias, roi=cfg.roi)
    denom = weight.sum() + 1e-9

    dxf_px = float(((-fx_res) * weight).sum() / denom)
    dyf_px = float(((-fy_res) * weight).sum() / denom)
    dx = np.clip(dxf_px / max(Wf, 1), -cfg.dx_clip, cfg.dx_clip)
    dy = np.clip(dyf_px / max(Hf, 1), -cfg.dx_clip, cfg.dx_clip)
    dx = (1.0 - cfg.dx_smooth) * dx + cfg.dx_smooth * state["prev_dx"]
    dy = (1.0 - cfg.dx_smooth) * dy + cfg.dx_smooth * state["prev_dy"]
    state["prev_dx"], state["prev_dy"] = dx, dy

    x_prev, y_prev = state["xhat"], state["yhat"]
    state["xhat"] = max(0.0, x_prev + dx)
    state["yhat"] = float(np.clip(y_prev + dy, 0.0, 1.0))

    gx, gy = cfg.goal_x, cfg.goal_y
    prev_dist = math.sqrt(
        (gx - x_prev) ** 2 + (cfg.goal_y_weight * (gy - y_prev)) ** 2
    )
    dist = math.sqrt(
        (gx - state["xhat"]) ** 2
        + (cfg.goal_y_weight * (gy - state["yhat"])) ** 2
    )
    reward = 0.0
    goal_now = dist < cfg.goal_radius
    if goal_now and not state["goal_awarded"]:
        reward = cfg.success_reward
        state["goal_awarded"] = True
    return float(reward), float(dist)


class BinaryGoalRewardWrapper(gym.Wrapper):
    """Replace native Procgen reward with a binary goal reward."""

    def __init__(self, env: gym.Env, cfg: BinaryGoalRewardConfig):
        super().__init__(env)
        self.cfg = cfg
        self._state = None
        self._prev_rgb = None
        self._native_return = 0.0

    def reset(self, **kwargs):
        self._state = {
            "xhat": 0.0,
            "yhat": 0.5,
            "prev_dx": 0.0,
            "prev_dy": 0.0,
            "goal_awarded": False,
        }
        self._prev_rgb = None
        self._native_return = 0.0
        reset_out = self.env.reset(**kwargs)
        if isinstance(reset_out, tuple):
            obs, info = reset_out
        else:
            obs, info = reset_out, {}
        self._prev_rgb = obs_to_hwc_uint8(obs)
        return obs, info

    def step(self, action):
        step_out = self.env.step(action)
        if len(step_out) == 4:
            obs, reward_native, done, info = step_out
            terminated, truncated = bool(done), False
        else:
            obs, reward_native, terminated, truncated, info = step_out
        info = dict(info or {})

        curr_rgb = obs_to_hwc_uint8(obs)
        reward_custom, dist = compute_binary_goal_reward(
            self._prev_rgb,
            curr_rgb,
            cfg=self.cfg,
            state=self._state,
        )
        self._prev_rgb = curr_rgb
        self._native_return += float(reward_native)

        info["distance_to_goal"] = dist
        info["xhat"] = self._state["xhat"]
        info["yhat"] = self._state["yhat"]
        if terminated or truncated:
            info["native_return"] = self._native_return

        return obs, reward_custom, terminated, truncated, info


# =========================
#  Progress callback
# =========================
class TqdmProgressCallback(BaseCallback):
    """Console progress with steps/sec using tqdm."""

    def __init__(self, total_timesteps: int, refresh_sec: float = 0.5):
        super().__init__()
        self.total = int(total_timesteps)
        self.refresh = float(refresh_sec)
        self._pbar = None
        self._t0 = None
        self._last_t = None
        self._last_steps = 0

    def _on_training_start(self) -> None:
        from tqdm.auto import tqdm

        self._t0 = self._last_t = time.perf_counter()
        self._last_steps = 0
        self._pbar = tqdm(
            total=self.total,
            dynamic_ncols=True,
            smoothing=0.2,
            desc="PPO training",
        )
        self._pbar.update(0)

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if now - self._last_t < self.refresh:
            return True
        steps = int(self.model.num_timesteps)
        self._pbar.n = min(steps, self.total)
        dt_tot = now - self._t0
        sps_avg = steps / max(dt_tot, 1e-9)
        self._pbar.set_postfix(sps=f"{sps_avg:,.0f}")
        self._pbar.refresh()
        self._last_t = now
        self._last_steps = steps
        return True

    def _on_training_end(self) -> None:
        if self._pbar is not None:
            self._pbar.n = min(int(self.model.num_timesteps), self.total)
            self._pbar.close()


# =========================
#  Env / logging helpers
# =========================
def _parse_action_map_arg(spec: Optional[str]):
    if not spec:
        return None
    cleaned = spec.strip()
    if not cleaned:
        return None
    for ch in "[]()":
        cleaned = cleaned.replace(ch, " ")
    parts = [p.strip() for p in cleaned.replace(";", ",").split(",")]
    vals = [int(p) for p in parts if p]
    if not vals:
        raise ValueError(f"Could not parse action_map from '{spec}'")
    return tuple(vals)


def _make_env_thunk(
    rank: int,
    args,
    action_map: Optional[Sequence[int]],
    reward_cfg: BinaryGoalRewardConfig,
) -> Callable[[], gym.Env]:
    seed = None if args.seed is None else args.seed + rank

    def _init():
        env = make_env(
            start_level=args.start_level + rank,
            num_levels=args.num_levels,
            distribution_mode=args.distribution_mode,
        )
        if seed is not None:
            env.action_space.seed(seed)
        if action_map is not None:
            env = ActionRemapWrapper(env, action_map)
        env = BinaryGoalRewardWrapper(env, reward_cfg)
        if args.max_episode_steps:
            env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
        return env

    return _init


def build_vec_env(
    args,
    action_map: Optional[Sequence[int]],
    reward_cfg: BinaryGoalRewardConfig,
):
    thunks = [
        _make_env_thunk(i, args, action_map, reward_cfg)
        for i in range(args.num_envs)
    ]
    if args.vec_backend == "subproc" and args.num_envs > 1:
        vec = SubprocVecEnv(thunks)
    else:
        vec = DummyVecEnv(thunks)
    vec = VecMonitor(vec, info_keywords=("native_return", "distance_to_goal"))
    vec = VecTransposeImage(vec)
    return vec


def save_roi_preview_from_env(
    env: gym.Env,
    roi,
    bottom_bias: float,
    downscale: int,
    out_path: str,
):
    sample = env.reset()
    if isinstance(sample, tuple):
        frame = obs_to_hwc_uint8(sample[0])
    else:
        frame = obs_to_hwc_uint8(sample)
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    p = _downscale_to_longside(gray, downscale)
    Hf, Wf = p.shape
    weight = _roi_weight(Hf, Wf, bottom_bias=bottom_bias, roi=roi)
    wnorm = (weight - weight.min()) / (weight.max() - weight.min() + 1e-9)
    wimg = cv2.resize(wnorm, (W, H), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((wimg * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * heat + 0.4 * frame).astype(np.uint8)

    y0, y1, x0, x1 = roi
    iy0, iy1 = int(y0 * H), int(y1 * H)
    ix0, ix1 = int(x0 * W), int(x1 * W)
    cv2.rectangle(overlay, (ix0, iy0), (ix1, iy1), (0, 128, 255), 2)
    cv2.putText(
        overlay,
        f"ROI y=[{y0:.2f},{y1:.2f}] x=[{x0:.2f},{x1:.2f}] bottom_bias={bottom_bias}",
        (8, H - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    imageio.imwrite(out_path, overlay)
    env.close()


def seed_everything(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


# =========================
#  Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="runs/coinrun", help="base log dir")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--start_level", type=int, default=0)
    parser.add_argument("--num_levels", type=int, default=200)
    parser.add_argument("--distribution_mode", type=str, default="easy")
    parser.add_argument(
        "--vec_backend",
        choices=["subproc", "dummy"],
        default="subproc",
        help="VecEnv backend (subproc recommended for >1 env)",
    )
    parser.add_argument(
        "--action_map",
        type=str,
        default="4,7,1,5,8,2,3",
        help="Comma-separated mapping from policy actions to Procgen actions",
    )

    parser.add_argument(
        "--roi",
        type=str,
        default="0.20,0.95,0.05,0.95",
        help="normalized ROI y0,y1,x0,x1",
    )
    parser.add_argument(
        "--downscale", type=int, default=64, help="flow longside resolution"
    )
    parser.add_argument(
        "--bottom_bias",
        type=float,
        default=2.0,
        help="emphasize lower rows for odometry",
    )
    parser.add_argument(
        "--flow_baseline",
        type=str,
        choices=["sky", "global", "none"],
        default="sky",
    )
    parser.add_argument(
        "--goal_screens",
        type=float,
        default=1.0,
        help="absolute X goal in screen widths",
    )
    parser.add_argument(
        "--goal_xy",
        type=str,
        default=None,
        help="override using 'x_screens,y_frac'",
    )
    parser.add_argument(
        "--goal_y_weight",
        type=float,
        default=0.3,
        help="scales Y error inside distance computation",
    )
    parser.add_argument(
        "--goal_radius",
        type=float,
        default=0.05,
        help="success radius in screen widths",
    )
    parser.add_argument(
        "--success_reward",
        type=float,
        default=10.0,
        help="reward emitted once when the estimated goal is reached",
    )
    parser.add_argument("--dx_clip", type=float, default=0.1)
    parser.add_argument("--dx_smooth", type=float, default=0.5)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ppo", action="store_true", help="Train with PPO")
    group.add_argument(
        "--recurrent_ppo",
        action="store_true",
        help="Train with RecurrentPPO (CnnLstmPolicy)",
    )
    group.add_argument("--a2c", action="store_true", help="Train with A2C")

    args = parser.parse_args()

    if args.a2c:
        agent_name = "A2C"
    elif args.recurrent_ppo:
        agent_name = "RecurrentPPO"
    else:
        agent_name = "PPO"
    run_dir = make_run_dir(args.logdir, f"{agent_name}_CoinRun")
    print(f"[RUN]  run_dir = {os.path.abspath(run_dir)}")

    tb_train_dir = os.path.join(run_dir, "tb_train")
    print(f"[TB]   tb_train = {os.path.abspath(tb_train_dir)}")

    if args.goal_xy is not None:
        goal_x, goal_y = [float(v.strip()) for v in args.goal_xy.split(",")]
    else:
        goal_x, goal_y = float(args.goal_screens), 0.5
    roi = tuple(float(v.strip()) for v in args.roi.split(","))
    action_map = _parse_action_map_arg(args.action_map)
    seed_everything(args.seed)

    reward_cfg = BinaryGoalRewardConfig(
        roi=roi,
        longside=args.downscale,
        bottom_bias=args.bottom_bias,
        flow_baseline=args.flow_baseline,
        goal_x=goal_x,
        goal_y=goal_y,
        goal_y_weight=args.goal_y_weight,
        goal_radius=args.goal_radius,
        dx_clip=args.dx_clip,
        dx_smooth=args.dx_smooth,
        success_reward=args.success_reward,
    )

    # ROI preview for sanity checking
    preview_env = make_env(
        start_level=args.start_level,
        num_levels=args.num_levels,
        distribution_mode=args.distribution_mode,
    )
    if action_map is not None:
        preview_env = ActionRemapWrapper(preview_env, action_map)
    save_roi_preview_from_env(
        preview_env,
        roi,
        args.bottom_bias,
        args.downscale,
        os.path.join(run_dir, "roi_preview.png"),
    )

    vec = build_vec_env(args, action_map, reward_cfg)

    if args.recurrent_ppo:
        model = RecurrentPPO(
            "CnnLstmPolicy",
            vec,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=tb_train_dir,
            device=args.device,
        )
    elif args.a2c:
        model = A2C(
            "CnnPolicy",
            vec,
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=tb_train_dir,
            device=args.device,
        )
    else:
        model = PPO(
            "CnnPolicy",
            vec,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=tb_train_dir,
            device=args.device,
        )

    progress_cb = TqdmProgressCallback(
        total_timesteps=args.total_timesteps, refresh_sec=0.5
    )
    callback = CallbackList([progress_cb])

    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(os.path.join(run_dir, "models", f"{agent_name}_policy"))
    vec.close()


if __name__ == "__main__":
    main()
