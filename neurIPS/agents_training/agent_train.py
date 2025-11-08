"""
Train PPO (Stable-Baselines3) inside the Genie/GenieRedux world model
with optical-flow-based rewards.

Includes:
- Fast grayscale optical flow
- Residual-flow reward (with stuck→jump assist)
- Flow-odometry + goal (distance-to-goal reduction)
- Reward stride, downscale, ROI, bottom-bias
- VecFrameStack(4) and VecNormalize(norm_reward=True)
"""

from __future__ import annotations

import os, glob, argparse, random, time, sys
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, VecEnv
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from models import construct_model


# =========================
#  Optical Flow helpers
# =========================

def _downscale_to_longside(gray: np.ndarray, longside: int) -> np.ndarray:
    H, W = gray.shape
    if longside is None:
        return gray
    if max(H, W) == longside:
        return gray
    s = float(longside) / float(max(H, W))
    new_size = (max(8, int(W * s)), max(8, int(H * s)))
    return cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

def _flow_residual(prev_gray: np.ndarray, next_gray: np.ndarray, *, longside: int = 64):
    """Compute Farnebäck flow on grayscale frames and subtract median (camera pan)."""
    p = _downscale_to_longside(prev_gray, longside)
    n = _downscale_to_longside(next_gray, longside)

    flow = cv2.calcOpticalFlowFarneback(
        p, n, None, pyr_scale=0.5, levels=3,
        winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    fx = flow[..., 0]
    fy = flow[..., 1]
    fx -= np.median(fx)
    fy -= np.median(fy)
    return fx, fy  # residual flow at downscaled resolution

def _roi_weight(H: int, W: int, *, bottom_bias: float, roi: Tuple[float,float,float,float]) -> np.ndarray:
    """Bottom-biased vertical weights multiplied by an ROI mask."""
    row = np.linspace(0, 1, H, dtype=np.float32)
    wv  = 1.0 + (row ** 2) * (bottom_bias - 1.0)  # emphasize bottom
    wv /= wv.mean()
    weight = np.repeat(wv[:, None], W, axis=1)
    y0, y1, x0, x1 = roi
    iy0, iy1 = int(y0 * H), int(y1 * H)
    ix0, ix1 = int(x0 * W), int(x1 * W)
    mask = np.zeros_like(weight, dtype=np.float32)
    mask[iy0:iy1, ix0:ix1] = 1.0
    weight *= mask
    return weight


# -------------------------
# Residual optical-flow reward (fast GRAY version)
# -------------------------
def optical_flow_reward_gray(prev_gray: np.ndarray,
                             next_gray: np.ndarray,
                             *,
                             right_weight: float = 1.0,
                             jitter_penalty: float = 0.15,
                             bottom_bias: float = 2.0,
                             clip: float = 1.0,
                             downscale: int = 64,
                             roi: Tuple[float,float,float,float] = (0.55, 1.00, 0.25, 0.75),
                             stuck_eps: float = 0.010,
                             jump_bonus_w: float = 0.30,
                             stuck_jitter_penalty: float = 0.05) -> float:
    """
    Residual optical-flow reward with ROI, bottom bias, and a 'stuck' jump bonus.
    (Fast: expects grayscale inputs.)
    """
    fx_res, fy_res = _flow_residual(prev_gray, next_gray, longside=downscale)
    H, W = fx_res.shape
    weight = _roi_weight(H, W, bottom_bias=bottom_bias, roi=roi)
    denom = weight.sum() + 1e-9

    # Forward: background moving LEFT → -fx_res positive
    forward = float((np.maximum(-fx_res, 0.0) * weight).sum() / denom)
    # Jitter: residual magnitude
    jitter = float(np.sqrt(fx_res**2 + fy_res**2).mean())

    if forward < stuck_eps:
        up = float((np.maximum(-fy_res, 0.0) * weight).sum() / denom)  # upward ≈ -fy
        r = right_weight * forward + jump_bonus_w * up - stuck_jitter_penalty * jitter
    else:
        r = right_weight * forward - jitter_penalty * jitter

    return float(np.clip(r, -clip, clip))


def optical_flow_reward(prev_rgb: np.ndarray, next_rgb: np.ndarray, **kw) -> float:
    """Wrapper that converts RGB→GRAY then calls the fast gray reward."""
    prev = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    nxt  = cv2.cvtColor(next_rgb,  cv2.COLOR_RGB2GRAY)
    return optical_flow_reward_gray(prev, nxt, **kw)


# =========================
#  Small tensor/image utils
# =========================
def _to_chw(img_hwc: np.ndarray) -> torch.Tensor:
    if img_hwc.dtype not in (np.float32, np.float64):
        x = torch.from_numpy(img_hwc.astype(np.uint8)).permute(2, 0, 1).float() / 255.0
    else:
        x = torch.from_numpy(img_hwc).permute(2, 0, 1).float()
        if x.max() > 1.0:
            x = x / 255.0
    return x

def _resize_chw(x_chw: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x_chw.unsqueeze(0), size=size_hw, mode="bilinear", align_corners=False).squeeze(0)


# =========================
#  World model loading
# =========================
def load_world_model(cfg_path: str, ckpt_path: str, device: torch.device):
    cfg = OmegaConf.load(cfg_path)
    model = construct_model(cfg)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    try:
        image_size = model.image_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
    except Exception:
        image_size = (64, 64)

    tokens_per_frame = getattr(model.tokenizer, "image_num_tokens", None)
    maskgit = getattr(model.dynamics, "maskgit", None)
    max_seq_len = getattr(maskgit, "max_seq_len", None) if maskgit is not None else None
    action_dim = getattr(maskgit, "action_dim", None) if maskgit is not None else None

    if tokens_per_frame is None or max_seq_len is None or action_dim is None:
        raise RuntimeError("Model missing tokenizer.image_num_tokens, dynamics.maskgit.max_seq_len, or action_dim")

    max_frames = max(1, max_seq_len // tokens_per_frame)
    return model, (int(image_size[0]), int(image_size[1])), int(action_dim), int(max_frames)


# =========================
#  Prime provider
# =========================
class PrimeProvider:
    """Sample an initial (reset) frame from a pool of NPZ episodes, resized to WM size."""
    def __init__(self, npz_glob: str, image_hw: Tuple[int, int]):
        paths = sorted(glob.glob(npz_glob))
        if not paths:
            raise RuntimeError(f"No NPZs matched for primes: {npz_glob}")
        self.paths = paths
        self.Ht, self.Wt = image_hw

    def sample_prime(self) -> np.ndarray:
        p = random.choice(self.paths)
        with np.load(p, allow_pickle=True) as npz:
            frames = None
            for k in ["frames", "videos", "input_frames", "obs", "images", "x"]:
                if k in npz:
                    frames = npz[k]; break
            if frames is None:
                for k, v in npz.items():
                    if isinstance(v, np.ndarray) and v.ndim >= 3 and 3 in v.shape:
                        frames = v; break
            if frames is None:
                raise RuntimeError(f"No frames found in {p}")

            f = frames
            # normalize to T×H×W×C
            if f.ndim == 4:
                if f.shape[-1] in (1, 3):
                    pass
                elif f.shape[0] in (1, 3):
                    f = np.transpose(f, (1, 2, 3, 0))  # C T H W -> T H W C
                elif f.shape[1] in (1, 3):
                    f = np.transpose(f, (0, 2, 3, 1))  # T C H W -> T H W C
            elif f.ndim == 5 and f.shape[0] in (1, 3):
                f = np.transpose(f, (1, 2, 3, 0))
            else:
                raise RuntimeError(f"Unsupported frames shape in {p}: {frames.shape}")

            f0 = f[0]
            if f0.dtype != np.uint8:
                if np.max(f0) <= 1.0:
                    f0 = (np.clip(f0, 0, 1) * 255).astype(np.uint8)
                else:
                    f0 = np.clip(f0, 0, 255).astype(np.uint8)

            chw = _resize_chw(_to_chw(f0), (self.Ht, self.Wt))
            return (chw.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()


# =========================
#  Single SB3-compatible Gym Env (kept for check_env)
# =========================
class GenieReduxEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 wm,
                 device: torch.device,
                 image_hw: Tuple[int, int],
                 action_dim: int,
                 prime_provider: PrimeProvider,
                 *,
                 horizon: int = 256,
                 fp: int = 1,
                 inference_steps: int = 4,
                 sample_temperature: float = 1.0,
                 mask_schedule: str = "cosine"):
        super().__init__()
        self.wm = wm
        self.dev = device
        self.H, self.W = image_hw
        self.A = int(action_dim)
        self.horizon = int(horizon)
        self.fp = max(1, int(fp))
        self.inf_steps = int(inference_steps)
        self.temp = float(sample_temperature)
        self.mask_schedule = mask_schedule
        self.provider = prime_provider

        self.action_space = spaces.Discrete(self.A)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8)

        self._t = 0
        self._prime: Optional[torch.Tensor] = None
        self._last_obs: Optional[np.ndarray] = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        f0 = self.provider.sample_prime()
        chw = _resize_chw(_to_chw(f0), (self.H, self.W))
        self._prime = chw.unsqueeze(1).to(self.dev)
        self._last_obs = f0
        return np.array(f0, copy=True), {}

    @torch.no_grad()
    def step(self, action: int):
        prev = self._last_obs
        acts = torch.tensor([[int(action)]], device=self.dev, dtype=torch.long)
        preds = self.wm.sample(
            prime_frames=self._prime.unsqueeze(0),
            actions=acts,
            num_frames=1,
            inference_steps=self.inf_steps,
            sample_temperature=self.temp,
            mask_schedule=self.mask_schedule,
            return_recons_only=True,
        )  # (1,C,1,H,W)

        self._prime = preds[:, :, -self.fp:][0]  # (C,fp,H,W)
        nxt = (preds[:, :, 0].clamp(0,1) * 255).to(torch.uint8)[0].permute(1,2,0).cpu().numpy()
        reward = optical_flow_reward(prev, nxt)
        self._t += 1
        truncated = self._t >= self.horizon
        self._last_obs = nxt
        if truncated:
            obs, _ = self.reset()
            return obs, float(reward), False, True, {}
        return nxt, float(reward), False, False, {}


# =========================
#  Batched VecEnv (with goal shaping option)
# =========================
class GenieReduxBatchedVecEnv(VecEnv):
    """
    SB3 VecEnv that batches the world-model step and computes rewards:
      reward_mode:
        - "flow": residual-flow + stuck→jump (dense, myopic)
        - "goal": flow-odometry + distance-to-goal reduction (professor's idea)
        - "hybrid": goal + small progress − jitter
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 wm,
                 device: torch.device,
                 image_hw: Tuple[int, int],
                 action_dim: int,
                 prime_provider: PrimeProvider,
                 *,
                 num_envs: int = 128,
                 horizon: int = 256,
                 fp: int = 1,
                 inference_steps: int = 4,
                 sample_temperature: float = 1.0,
                 mask_schedule: str = "cosine",
                 use_amp: bool = True,
                 amp_dtype: torch.dtype = torch.bfloat16,
                 # reward config
                 reward_mode: str = "goal",         # "flow" | "goal" | "hybrid"
                 downscale: int = 64,
                 bottom_bias: float = 2.0,
                 roi: Tuple[float,float,float,float] = (0.55, 1.00, 0.25, 0.75),
                 reward_stride: int = 1,
                 # flow-reward (flow mode)
                 right_weight: float = 1.0,
                 jitter_penalty: float = 0.15,
                 stuck_eps: float = 0.010,
                 jump_bonus_w: float = 0.30,
                 stuck_jitter_penalty: float = 0.05,
                 # goal-reward (goal/hybrid)
                 goal_weight: float = 1.0,
                 progress_weight: float = 0.20,
                 jitter_weight: float = 0.10,
                 success_bonus: float = 1.0,
                 goal_radius: float = 0.05):
        H, W = image_hw
        self.H, self.W = H, W
        self.A = int(action_dim)
        self.N = int(num_envs)

        self.wm = wm
        self.dev = device
        self.provider = prime_provider
        self.horizon = int(horizon)
        self.fp = max(1, int(fp))
        self.inf_steps = int(inference_steps)
        self.temp = float(sample_temperature)
        self.mask_schedule = mask_schedule
        self.use_amp = use_amp and (device.type == "cuda")
        self.amp_dtype = amp_dtype

        # Reward config
        self.reward_mode = reward_mode
        self.downscale = downscale
        self.bottom_bias = bottom_bias
        self.roi = roi
        self.reward_stride = max(1, int(reward_stride))

        self.right_weight = right_weight
        self.jitter_penalty = jitter_penalty
        self.stuck_eps = stuck_eps
        self.jump_bonus_w = jump_bonus_w
        self.stuck_jitter_penalty = stuck_jitter_penalty

        self.goal_weight = goal_weight
        self.progress_weight = progress_weight
        self.jitter_weight = jitter_weight
        self.success_bonus = success_bonus
        self.goal_radius = goal_radius

        # Odometry state for goal shaping
        self._xhat = np.zeros(self.N, dtype=np.float32)      # accumulated coordinate
        self._goal = np.full(self.N, 1.0, dtype=np.float32)  # rightward goal (1 screen)
        self._prev_dx = np.zeros(self.N, dtype=np.float32)   # optional EMA of dx

        # Reward cache for striding
        self._rew_cache = np.zeros(self.N, dtype=np.float32)

        observation_space = spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8)
        action_space = spaces.Discrete(self.A)
        super().__init__(num_envs=self.N, observation_space=observation_space, action_space=action_space)

        self._t = np.zeros(self.N, dtype=np.int32)
        self._prime = None                  # (N,C,fp,H,W) on device
        self._last_obs = None               # (N,H,W,3) uint8
        self._actions_buffer = None

        self._reset_all_slots()

    # ----- VecEnv API -----
    def reset(self):
        self._reset_all_slots()
        return self._last_obs

    def step_async(self, actions):
        self._actions_buffer = np.asarray(actions, dtype=np.int64)

    def step_wait(self):
        assert self._actions_buffer is not None
        prev_obs = self._last_obs
        acts = torch.from_numpy(self._actions_buffer.reshape(self.N, 1)).to(self.dev, dtype=torch.long)

        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    preds = self.wm.sample(
                        prime_frames=self._prime,
                        actions=acts,
                        num_frames=1,
                        inference_steps=self.inf_steps,
                        sample_temperature=self.temp,
                        mask_schedule=self.mask_schedule,
                        return_recons_only=True,
                    )
            else:
                preds = self.wm.sample(
                    prime_frames=self._prime,
                    actions=acts,
                    num_frames=1,
                    inference_steps=self.inf_steps,
                    sample_temperature=self.temp,
                    mask_schedule=self.mask_schedule,
                    return_recons_only=True,
                )

        self._prime = preds[:, :, -self.fp:]
        nxt = (preds[:, :, 0].clamp(0,1) * 255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()

        # ---- Reward (stride-aware) ----
        recompute = (self._t % self.reward_stride) == 0
        if recompute.any():
            for i in range(self.N):
                if not recompute[i]:
                    continue
                if self.reward_mode == "flow":
                    # fast gray call
                    p = cv2.cvtColor(prev_obs[i], cv2.COLOR_RGB2GRAY)
                    c = cv2.cvtColor(nxt[i],      cv2.COLOR_RGB2GRAY)
                    r = optical_flow_reward_gray(
                        p, c,
                        right_weight=self.right_weight,
                        jitter_penalty=self.jitter_penalty,
                        bottom_bias=self.bottom_bias,
                        clip=1.0,
                        downscale=self.downscale,
                        roi=self.roi,
                        stuck_eps=self.stuck_eps,
                        jump_bonus_w=self.jump_bonus_w,
                        stuck_jitter_penalty=self.stuck_jitter_penalty,
                    )
                    self._rew_cache[i] = r

                else:
                    # goal or hybrid: compute residual flow once
                    p = cv2.cvtColor(prev_obs[i], cv2.COLOR_RGB2GRAY)
                    c = cv2.cvtColor(nxt[i],      cv2.COLOR_RGB2GRAY)
                    fx_res, fy_res = _flow_residual(p, c, longside=self.downscale)
                    H, W = fx_res.shape
                    weight = _roi_weight(H, W, bottom_bias=self.bottom_bias, roi=self.roi)
                    denom = weight.sum() + 1e-9

                    # signed forward increment (scene-left positive)
                    dxf_px = float(((-fx_res) * weight).sum() / denom)
                    dx_norm = np.clip(dxf_px / W, -0.1, 0.1)  # ~fraction of screen width
                    # optional EMA:
                    dx_norm = 0.8 * self._prev_dx[i] + 0.2 * dx_norm
                    self._prev_dx[i] = dx_norm

                    x_prev = self._xhat[i]
                    x_cur  = max(0.0, x_prev + dx_norm)
                    self._xhat[i] = x_cur

                    goal = self._goal[i]
                    dist_prev = abs(goal - x_prev)
                    dist_cur  = abs(goal - x_cur)
                    r_goal = dist_prev - dist_cur  # potential-based distance reduction

                    jitter = float(np.sqrt(fx_res**2 + fy_res**2).mean())

                    if self.reward_mode == "goal":
                        r = (
                            self.goal_weight * r_goal
                            + self.progress_weight * dx_norm
                            - self.jitter_weight * jitter
                        )
                    else:  # "hybrid"
                        r = (
                            self.goal_weight * r_goal
                            + (self.progress_weight * 0.5) * dx_norm
                            - (self.jitter_weight * 0.5) * jitter
                        )

                    if dist_cur < self.goal_radius:
                        r += self.success_bonus
                        self._goal[i] += 1.0  # curriculum: next screen

                    self._rew_cache[i] = float(np.clip(r, -1.0, 1.0))

        rew = self._rew_cache.copy()

        # Time / done
        self._t += 1
        done = (self._t >= self.horizon)
        infos = [{} for _ in range(self.N)]

        if np.any(done):
            for i in np.where(done)[0]:
                infos[i]["terminal_observation"] = nxt[i].copy()
                infos[i]["TimeLimit.truncated"] = True
                self._reset_slot(i)
                nxt[i] = self._last_obs[i]
                self._t[i] = 0

        self._last_obs = nxt
        self._actions_buffer = None
        return nxt, rew, done, infos

    def close(self):
        pass

    # ----- helpers -----
    def _reset_all_slots(self):
        primes, last = [], []
        for _ in range(self.N):
            f0 = self.provider.sample_prime()
            last.append(f0)
            primes.append(_resize_chw(_to_chw(f0), (self.H, self.W)))
        chws = torch.stack(primes, 0)
        self._prime = chws.unsqueeze(2).to(self.dev)
        self._last_obs = np.stack(last, 0)
        self._t[:] = 0
        self._xhat[:] = 0.0
        self._goal[:] = 1.0
        self._prev_dx[:] = 0.0
        self._rew_cache[:] = 0.0

    def _reset_slot(self, i: int):
        f0 = self.provider.sample_prime()
        self._last_obs[i] = f0
        chw = _resize_chw(_to_chw(f0), (self.H, self.W)).to(self.dev)
        self._prime[i, :, 0] = chw
        self._xhat[i] = 0.0
        self._goal[i] = 1.0
        self._prev_dx[i] = 0.0
        self._rew_cache[i] = 0.0

    def _indices_to_list(self, indices):
        if indices is None:
            return list(range(self.N))
        if isinstance(indices, (int, np.integer)):
            return [int(indices)]
        return list(indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        idxs = self._indices_to_list(indices)
        if hasattr(self, method_name) and callable(getattr(self, method_name)):
            res = getattr(self, method_name)(*method_args, **method_kwargs)
            return [res for _ in idxs]
        return [None for _ in idxs]

    def get_attr(self, attr_name, indices=None):
        idxs = self._indices_to_list(indices)
        val = getattr(self, attr_name, None)
        return [val for _ in idxs]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)
        idxs = self._indices_to_list(indices)
        return [None for _ in idxs]

    def env_is_wrapped(self, wrapper_class, indices=None):
        idxs = self._indices_to_list(indices)
        return [False for _ in idxs]


# ---- Progress bar + speedometer for SB3 ----
class TqdmProgressCallback(BaseCallback):
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
        self._pbar = tqdm(total=self.total, dynamic_ncols=True, smoothing=0.2, desc="PPO training")
        self._pbar.update(0)

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if now - self._last_t < self.refresh:
            return True
        steps = int(self.model.num_timesteps)
        self._pbar.n = min(steps, self.total)
        dt = now - self._last_t
        dt_tot = now - self._t0
        dsteps = steps - self._last_steps
        sps_inst = dsteps / max(dt, 1e-9)
        sps_avg = steps / max(dt_tot, 1e-9)
        eta_s = (self.total - steps) / max(sps_avg, 1e-9)
        self._pbar.set_postfix(sps=f"{sps_avg:,.0f}", eta=f"{eta_s/60:.1f}m")
        self._pbar.refresh()
        self.logger.record("time/sps_avg", sps_avg)
        self.logger.record("time/sps_inst", sps_inst)
        self._last_t = now
        self._last_steps = steps
        return True

    def _on_training_end(self) -> None:
        if self._pbar is not None:
            self._pbar.n = min(int(self.model.num_timesteps), self.total)
            self._pbar.close()


# =========================
#  Main: train PPO
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Hydra YAML for world model")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path")
    ap.add_argument("--init_npz_glob", required=True, help="Glob with NPZs to sample reset frames")
    ap.add_argument("--device", default="cuda")

    # PPO / vec env
    ap.add_argument("--n_steps", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--n_epochs", type=int, default=8)
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--total_timesteps", type=int, default=1_000_000)
    ap.add_argument("--learning_rate", type=float, default=5e-4)
    ap.add_argument("--ent_coef", type=float, default=0.02)
    ap.add_argument("--clip_range", type=float, default=0.2)

    # World-model sampling
    ap.add_argument("--inference_steps", type=int, default=2)
    ap.add_argument("--sample_temperature", type=float, default=1.0)
    ap.add_argument("--mask_schedule", type=str, default="cosine")

    # Reward config
    ap.add_argument("--reward_mode", type=str, default="goal", choices=["flow", "goal", "hybrid"])
    ap.add_argument("--downscale", type=int, default=64)
    ap.add_argument("--bottom_bias", type=float, default=2.0)
    ap.add_argument("--reward_stride", type=int, default=1)

    # Flow-mode knobs
    ap.add_argument("--stuck_eps", type=float, default=0.010)
    ap.add_argument("--jump_bonus_w", type=float, default=0.30)

    # Goal-mode knobs
    ap.add_argument("--goal_weight", type=float, default=1.0)
    ap.add_argument("--progress_weight", type=float, default=0.20)
    ap.add_argument("--jitter_weight", type=float, default=0.10)
    ap.add_argument("--goal_radius", type=float, default=0.05)

    # Stacking / normalization
    ap.add_argument("--stack", type=int, default=4)
    ap.add_argument("--norm_reward", action="store_true", default=True)
    ap.add_argument("--logdir", type=str, default="runs/genie_sb3_ppoflow")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")

    wm, image_hw, action_dim, _ = load_world_model(args.config, args.ckpt, device)
    provider = PrimeProvider(args.init_npz_glob, image_hw)

    # Sanity check on a single env
    check_env(Monitor(GenieReduxEnv(
        wm=wm, device=device, image_hw=image_hw, action_dim=action_dim,
        prime_provider=provider, horizon=args.horizon,
        fp=1, inference_steps=args.inference_steps,
        sample_temperature=args.sample_temperature,
        mask_schedule=args.mask_schedule
    )), warn=True, skip_render_check=True)

    # Batched env
    vec = GenieReduxBatchedVecEnv(
        wm=wm, device=device, image_hw=image_hw, action_dim=action_dim,
        prime_provider=provider,
        num_envs=args.num_envs, horizon=args.horizon,
        fp=1, inference_steps=args.inference_steps,
        sample_temperature=args.sample_temperature,
        mask_schedule=args.mask_schedule,
        use_amp=True, amp_dtype=torch.bfloat16,
        reward_mode=args.reward_mode,
        downscale=args.downscale,
        bottom_bias=args.bottom_bias,
        reward_stride=args.reward_stride,
        stuck_eps=args.stuck_eps,
        jump_bonus_w=args.jump_bonus_w,
        goal_weight=args.goal_weight,
        progress_weight=args.progress_weight,
        jitter_weight=args.jitter_weight,
        goal_radius=args.goal_radius,
    )

    # Wrappers: monitor → transpose(HWC→CHW) → frame stack → reward normalize
    vec = VecMonitor(vec)
    vec = VecTransposeImage(vec)
    if args.stack and args.stack > 1:
        vec = VecFrameStack(vec, n_stack=args.stack, channels_order="first")
    if args.norm_reward:
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=5.0)

    model = PPO(
        "CnnPolicy", vec,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=0.99, gae_lambda=0.95,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        verbose=1, tensorboard_log=args.logdir,
    )

    progress_cb = TqdmProgressCallback(total_timesteps=args.total_timesteps, refresh_sec=0.5)
    callback = CallbackList([progress_cb])

    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    os.makedirs(args.logdir, exist_ok=True)
    model.save(os.path.join(args.logdir, f"ppo_genie_{args.reward_mode}"))

if __name__ == "__main__":
    main()
