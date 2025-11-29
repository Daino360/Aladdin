#!/usr/bin/env python3
"""
Simplified PPO training inside a Genie/GenieRedux world model (CoinRun-like).

Design goals (per user request):
- **No frame stacking** (no 4-stacked observations).  
- **Single reward mode only**: absolute-goal via FLOW-odometry ("professor reward").
  We accumulate residual optical flow to estimate an absolute horizontal
  coordinate \hat{x} (in "screen widths"), then use reward r_t = -|goal - \hat{x}_t|.
  Optionally a small success bonus when within a radius.
- **Lightweight visuals** every `vis_interval` steps:  
  (1) a GIF for env #0 over `vis_len` steps, and  
  (2) a single PNG **contact sheet**: rows = levels (envs), columns = time steps.

This file intentionally removes: hybrid/flow modes, ROI preview, reward normalize,
extra wrappers, and other bells & whistles to keep things focused and readable.

Run (example):
    python train_in_WM_simple_goal.py \
      --config path/to/world_model.yaml \
      --ckpt path/to/world_model.pt \
      --init_npz_glob "dataset/*.npz" \
      --logdir runs/ppo_goal_flowodom \
      --num_envs 128 --n_steps 128 --total_timesteps 1000000 \
      --goal_screens 1.5 --goal_radius 0.05 --vis_interval 20000

Notes:
- The absolute coordinate \hat{x} integrates forward motion inferred from
  residual optical flow over a bottom-biased ROI. Units are approx "screen widths"
  because dx is normalized by the flow-resolution width.
- If Genie/Redux actions differ from your discrete action count, `action_dim`
  is read from the checkpoint config via `construct_model(cfg)`.
- The odometry is approximate; tune ROI, baseline, and smoothing if needed.
"""
from __future__ import annotations

from collections import Counter

import os, glob, argparse, random, sys, time
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.vec_env import VecEnv, VecMonitor, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from torch.utils.tensorboard import SummaryWriter
import imageio

# Project-root import path so we can import your WM constructor
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from models import construct_model

#=========================
# Directory structure
#=========================
def make_run_dir(base_logdir: str, agent_name: str) -> str:
    """
    Create runs/<Agent>/<Agent>_XX directory that never overwrites past runs.
    Returns the full path to the new run directory.
    """
    agent_root = os.path.join(base_logdir, agent_name)
    os.makedirs(agent_root, exist_ok=True)
    idx = 1
    while True:
        run_dir = os.path.join(agent_root, f"{agent_name}_{idx:02d}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            # standard subfolders
            os.makedirs(os.path.join(run_dir, "tb_snaps"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "tb_train"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "tb_vis"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
            return run_dir
        idx += 1



# =========================
#  Optical flow utilities
# =========================

def _downscale_to_longside(gray: np.ndarray, longside: int) -> np.ndarray:
    """Resize grayscale `gray` so max(H,W) == `longside` (keeps aspect).
    If `longside` is None or already matched, return input.
    """
    H, W = gray.shape
    if longside is None or max(H, W) == longside:
        return gray
    s = float(longside) / float(max(H, W))
    new_size = (max(8, int(W * s)), max(8, int(H * s)))
    return cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)


def _flow_residual(prev_gray: np.ndarray, next_gray: np.ndarray, *,
                   longside: int = 64,
                   baseline: str = "sky") -> Tuple[np.ndarray, np.ndarray]:
    """Compute Farnebäck dense flow at a small resolution and remove median
    flow ("baseline") to reduce camera/scroll bias.

    Args:
        prev_gray, next_gray: HxW uint8 grayscale frames.
        longside: resize long side before flow.
        baseline: "global" median subtraction, "sky" top-band median, or "none".
    Returns:
        (fx_res, fy_res): residual flow fields.
    """
    p = _downscale_to_longside(prev_gray, longside)
    n = _downscale_to_longside(next_gray, longside)
    flow = cv2.calcOpticalFlowFarneback(p, n, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    fx, fy = flow[..., 0], flow[..., 1]
    if baseline == "global":
        fx -= np.median(fx); fy -= np.median(fy)
    elif baseline == "sky":
        h = fx.shape[0]; top = slice(0, max(1, int(0.2 * h)))
        fx -= np.median(fx[top, :]); fy -= np.median(fy[top, :])
    elif baseline == "none":
        pass
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    return fx, fy


def _roi_weight(H: int, W: int, *, bottom_bias: float,
                roi: Tuple[float, float, float, float]) -> np.ndarray:
    """Build a per-pixel weight map with bottom emphasis + rectangular ROI.

    ROI is normalized (y0,y1,x0,x1) in [0,1]. Bottom bias gradually increases
    weights near the ground to favor motion cues relevant to the agent.
    """
    row = np.linspace(0, 1, H, dtype=np.float32)
    wv = 1.0 + (row ** 2) * (bottom_bias - 1.0)  # emphasize bottom
    wv /= max(np.mean(wv), 1e-9)
    weight = np.repeat(wv[:, None], W, axis=1)

    y0, y1, x0, x1 = roi
    iy0, iy1 = int(y0 * H), int(y1 * H)
    ix0, ix1 = int(x0 * W), int(x1 * W)
    mask = np.zeros_like(weight, dtype=np.float32)
    mask[iy0:iy1, ix0:ix1] = 1.0
    return weight * mask


# =========================
#  World model loading
# =========================

def load_world_model(cfg_path: str, ckpt_path: str, device: torch.device):
    """Load the Hydra-configured world model and extract key shapes.

    Returns:
        model:     torch.nn.Module in eval mode on `device`.
        image_hw:  (H, W) of frames expected by WM.
        action_dim: number of discrete actions.
        max_frames: max frames if the tokenizer/dynamics impose a limit.
    """
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
#  Prime frame provider
# =========================
class PrimeProvider:
    """Samples an initial reset frame from NPZ episodes and resizes to WM size."""
    def __init__(self, npz_glob: str, image_hw: Tuple[int, int]):
        paths = sorted(glob.glob(npz_glob))
        if not paths:
            raise RuntimeError(f"No NPZs matched for primes: {npz_glob}")
        self.paths = paths
        self.Ht, self.Wt = image_hw

    def _to_chw(self, img_hwc: np.ndarray) -> torch.Tensor:
        """HWC -> CHW float32 tensor in [0,1]."""
        x = torch.from_numpy(img_hwc.astype(np.uint8)).permute(2, 0, 1).float() / 255.0
        return x

    def _resize_chw(self, x_chw: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        """Bilinear resize single CHW tensor (keeps [0,1])."""
        return F.interpolate(x_chw.unsqueeze(0), size=size_hw, mode="bilinear", align_corners=False).squeeze(0)

    def sample_prime(self) -> np.ndarray:
        """Pick a random NPZ and return its first frame as uint8 HWC at target size."""
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
                raise RuntimeError(f"No frames array found in {p}")

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

            chw = self._resize_chw(self._to_chw(f0), (self.Ht, self.Wt))
            return (chw.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()


# =========================
#  VecEnv with GOAL reward (flow odometry)
# =========================
class GenieReduxBatchedVecEnv(VecEnv):
    """Batched VecEnv around the world model using **only** absolute-goal reward.

    Reward definition (per step):
        - Compute residual flow between prev and next frames in a bottom-biased ROI.
        - Estimate signed forward increment dx (scene moving left ⇒ agent right).
        - Accumulate \hat{x} ← max(0, \hat{x} + dx).
        - Reward r = -|goal - \hat{x}| (+ success bonus if within radius).

    The unit of \hat{x} is roughly "screen widths" at the flow resolution.
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
                 inference_steps: int = 2,
                 sample_temperature: float = 1.0,
                 mask_schedule: str = "cosine",
                 use_amp: bool = True,
                 amp_dtype: torch.dtype = torch.bfloat16,
                 # reward/odometry config
                 downscale: int = 64,
                 bottom_bias: float = 2.0,
                 roi: Tuple[float, float, float, float] = (0.25, 0.85, 0.15, 0.65),
                 flow_baseline: str = "sky",
                 dx_clip: float = 0.1,
                 dx_smooth: float = 0.5,
                 goal_x_screens: float = 1.0,
                 goal_y_frac: float = 0.5,
                 goal_y_weight: float = 1.0,
                 goal_radius: float = 0.05,
                 success_bonus: float = 1.0,
                 advance_goal_on_success: bool = False,
                 ):
        H, W = image_hw
        self.H, self.W = H, W
        self.A = int(action_dim)
        self.N = int(num_envs)

        self.wm = wm
        self.dev = device
        self.provider = prime_provider
        self.horizon = int(horizon)
        self.inf_steps = int(inference_steps)
        self.temp = float(sample_temperature)
        self.mask_schedule = mask_schedule
        self.use_amp = use_amp and (device.type == "cuda")
        self.amp_dtype = amp_dtype

        # Reward/odometry params
        self.downscale = int(downscale)
        self.bottom_bias = float(bottom_bias)
        self.roi = roi
        self.flow_baseline = flow_baseline
        self.dx_clip = float(dx_clip)
        self.dx_smooth = float(dx_smooth)
        self.goal_radius = float(goal_radius)
        self.goal_y_weight = float(goal_y_weight)
        self.success_bonus = float(success_bonus)
        self.advance_goal_on_success = bool(advance_goal_on_success)

        # Odometry state per env
        self._last_dx = np.zeros(self.N, dtype=np.float32)         # last dx for debug

        self._xhat = np.zeros(self.N, dtype=np.float32)            # accumulated X (screens)
        self._yhat = np.zeros(self.N, dtype=np.float32)            # accumulated Y (0..1)
        self._prev_dx = np.zeros(self.N, dtype=np.float32)         # EMA helper (x)
        self._prev_dy = np.zeros(self.N, dtype=np.float32)         # EMA helper (y)
        self._last_dx = np.zeros(self.N, dtype=np.float32)         # debug
        self._last_dy = np.zeros(self.N, dtype=np.float32)         # debug
        self._goal_x = np.full(self.N, float(goal_x_screens), np.float32)
        self._goal_y = np.full(self.N, float(goal_y_frac),    np.float32)


        observation_space = spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8)
        action_space = spaces.Discrete(self.A)
        super().__init__(num_envs=self.N, observation_space=observation_space, action_space=action_space)

        self._t = np.zeros(self.N, dtype=np.int32)
        self._prime = None                  # (N,C,1,H,W) on device
        self._last_obs = None               # (N,H,W,3) uint8
        self._actions_buffer = None

        self._reset_all_slots()

    # ---------- VecEnv API ----------
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
                        return_recons_only=False,
                    )
            else:
                preds = self.wm.sample(
                    prime_frames=self._prime,
                    actions=acts,
                    num_frames=1,
                    inference_steps=self.inf_steps,
                    sample_temperature=self.temp,
                    mask_schedule=self.mask_schedule,
                    return_recons_only=False,
                )

        # next observations
        self._prime = preds[:, :, -1:]
        nxt = (preds[:, :, 0].clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        # reward via FLOW-odometry → xhat → -distance
        rew = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            p = cv2.cvtColor(prev_obs[i], cv2.COLOR_RGB2GRAY)
            c = cv2.cvtColor(nxt[i],      cv2.COLOR_RGB2GRAY)
            fx_res, fy_res = _flow_residual(p, c, longside=self.downscale, baseline=self.flow_baseline)
            Hf, Wf = fx_res.shape
            weight = _roi_weight(Hf, Wf, bottom_bias=self.bottom_bias, roi=self.roi)
            denom  = weight.sum() + 1e-9

            # Flow → (dx, dy)
            dxf_px = float(((-fx_res) * weight).sum() / denom)       # scene-left ⇒ agent-right
            dyf_px = float(((-fy_res) * weight).sum() / denom)       # positive up
            dx = np.clip(dxf_px / max(Wf, 1), -self.dx_clip, self.dx_clip)
            dy = np.clip(dyf_px / max(Hf, 1), -self.dx_clip, self.dx_clip)

            # Smooth
            dx = (1.0 - self.dx_smooth) * dx + self.dx_smooth * self._prev_dx[i]
            dy = (1.0 - self.dx_smooth) * dy + self.dx_smooth * self._prev_dy[i]
            self._prev_dx[i] = dx
            self._prev_dy[i] = dy
            self._last_dx[i] = dx
            self._last_dy[i] = dy

            # Integrate absolute position
            x_prev = self._xhat[i]
            y_prev = self._yhat[i]
            x_cur  = max(0.0, x_prev + dx)
            y_cur  = float(np.clip(y_prev + dy, 0.0, 1.0))
            self._xhat[i] = x_cur
            self._yhat[i] = y_cur

            # Progress reward: r = prev_dist - dist  (positive if closer, negative if farther)
            gx, gy = self._goal_x[i], self._goal_y[i]
            dxerr_prev = gx - x_prev
            dyerr_prev = (gy - y_prev) * self.goal_y_weight
            prev_dist  = float(np.sqrt(dxerr_prev*dxerr_prev + dyerr_prev*dyerr_prev))

            dxerr = gx - x_cur
            dyerr = (gy - y_cur) * self.goal_y_weight
            dist  = float(np.sqrt(dxerr*dxerr + dyerr*dyerr))

            # Optionally scale progress if it’s tiny:
            progress_coef = 5.0   # try 2–10 based on your Δdist scale
            r = progress_coef * (prev_dist - dist)

            # Success bonus inside the goal radius
            if dist < self.goal_radius:
                r += self.success_bonus
                if self.advance_goal_on_success:
                    self._goal_x[i] += float(self.goal_radius)

            rew[i] = r


        # time & truncation
        self._t += 1
        done = (self._t >= self.horizon)
        infos = [{} for _ in range(self.N)]
        if np.any(done):
            for i in np.where(done)[0]:
                infos[i]["TimeLimit.truncated"] = True
                infos[i]["terminal_observation"] = nxt[i].copy()
                self._reset_slot(i)
                nxt[i] = self._last_obs[i]
                self._t[i] = 0

        self._last_obs = nxt
        self._actions_buffer = None
        return nxt, rew, done, infos

    def close(self):
        pass

    # ---- SB3 v2 required VecEnv API ----
    def get_attr(self, attr_name, indices=None):
        """Return attribute `attr_name` for each env index.
        We host all slots inside a single object, so we mirror the same value."""
        if indices is None:
            return [getattr(self, attr_name, None)] * self.N
        if isinstance(indices, int):
            return getattr(self, attr_name, None)
        return [getattr(self, attr_name, None) for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute on the shared container (no per-env objects)."""
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        """Best-effort method call to satisfy SB3's VecEnv API.
        We don't expose per-env objects; attempt to call a method on `self`.
        If not present, this is a no-op returning a list of None(s)."""
        if hasattr(self, method_name) and callable(getattr(self, method_name)):
            out = getattr(self, method_name)(*args, **kwargs)
            if indices is None:
                return [out] * self.N
            if isinstance(indices, int):
                return out
            return [out for _ in indices]
        n = self.N if indices is None else (1 if isinstance(indices, int) else len(indices))
        return [None] * n

    def env_is_wrapped(self, wrapper_class, indices=None):
        """We don't wrap per-env instances; always return False."""
        n = self.N if indices is None else (1 if isinstance(indices, int) else len(indices))
        return [False] * n

    # ---------- helpers ----------
    def _reset_all_slots(self):
        primes, last = [], []
        for _ in range(self.N):
            f0 = self.provider.sample_prime()
            last.append(f0)
            chw = torch.from_numpy(f0).permute(2, 0, 1).float() / 255.0
            chw = F.interpolate(chw.unsqueeze(0), size=(self.H, self.W), mode="bilinear", align_corners=False).squeeze(0)
            primes.append(chw)
        chws = torch.stack(primes, 0)
        self._prime = chws.unsqueeze(2).to(self.dev)  # (N,C,1,H,W)
        self._last_obs = np.stack(last, 0)
        self._t[:] = 0
        self._xhat[:] = 0.0
        self._yhat[:] = 0.5
        self._prev_dx[:] = 0.0
        self._prev_dy[:] = 0.0
        self._last_dx[:] = 0.0
        self._last_dy[:] = 0.0
        # goals are set in __init__

    def _reset_slot(self, i: int):
        f0 = self.provider.sample_prime()
        self._last_obs[i] = f0
        chw = torch.from_numpy(f0).permute(2, 0, 1).float() / 255.0
        chw = F.interpolate(chw.unsqueeze(0), size=(self.H, self.W), mode="bilinear", align_corners=False).squeeze(0)
        self._prime[i, :, 0] = chw.to(self.dev)
        self._xhat[i] = 0.0
        self._yhat[i] = 0.5
        self._prev_dx[i] = 0.0
        self._prev_dy[i] = 0.0
        self._last_dx[i] = 0.0
        self._last_dy[i] = 0.0


# =========================
#  Simple progress bar + visual logger
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


def _tile_contact_sheet(frames_nt_hw3: np.ndarray) -> np.ndarray:
    """Create a single PNG contact sheet.

    Args:
        frames_nt_hw3: array shaped (N_envs, T, H, W, 3) uint8.
    Returns:
        H_img x W_img x 3 uint8 image with N rows (envs) and T columns (time).
    """
    assert frames_nt_hw3.ndim == 5
    N, T, H, W, C = frames_nt_hw3.shape
    pad_y, pad_x = 4, 4
    grid = np.full((N * H + (N - 1) * pad_y, T * W + (T - 1) * pad_x, 3), 16, np.uint8)
    for r in range(N):
        for c in range(T):
            y0 = r * (H + pad_y)
            x0 = c * (W + pad_x)
            grid[y0:y0 + H, x0:x0 + W] = frames_nt_hw3[r, c]
    return grid


class VisualLoggerCallback(BaseCallback):
    """Every `vis_interval` steps (elapsed, not modulo), roll out a small eval vec and save:
       - GIF (env 0) over `vis_len` steps.
       - PNG contact sheet with rows = envs and cols = time.
       Also logs video + contact sheet to TensorBoard.
    """
    def __init__(self,
                 wm,
                 device,
                 image_hw,
                 action_dim,
                 provider: PrimeProvider,
                 run_dir: str,
                 *,
                 vis_interval: int = 20_000,
                 vis_len: int = 64,
                 n_envs: int = 4,
                 inference_steps: int = 2,
                 sample_temperature: float = 1.0,
                 mask_schedule: str = "cosine",
                 roi=(0.25, 0.85, 0.15, 0.65),
                 downscale: int = 64,
                 bottom_bias: float = 2.0,
                 flow_baseline: str = "sky",
                 goal_screens: float = 1.0,
                 goal_radius: float = 0.05,
                 goal_x: float = 1.0,
                 goal_y: float = 0.5,
                 goal_y_weight: float = 1.0,
                 success_bonus: float = 1.0):
        super().__init__()
        self.wm = wm
        self.device = device
        self.image_hw = image_hw
        self.action_dim = action_dim
        self.provider = provider
        self.run_dir = run_dir
        self.vis_interval = int(vis_interval)
        self.vis_len = int(vis_len)
        self.n_envs = int(n_envs)
        self.inf_steps = int(inference_steps)
        self.temp = float(sample_temperature)
        self.mask_schedule = mask_schedule
        self.roi = roi
        self.downscale = downscale
        self.bottom_bias = bottom_bias
        self.flow_baseline = flow_baseline
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_y_weight = goal_y_weight
        self.goal_radius = goal_radius
        self.success_bonus = success_bonus

        # elapsed-step scheduler (None = force first dump)
        self._last_logged = None

        self.snapdir = os.path.join(self.run_dir, "tb_snaps")
        os.makedirs(self.snapdir, exist_ok=True)
        self.tb = SummaryWriter(log_dir=os.path.join(self.run_dir, "tb_vis"))
        self.tb.add_text("status", "tb_vis_initialized", 0)

        # small eval env
        self.eval_vec = GenieReduxBatchedVecEnv(
            wm=self.wm, device=self.device, image_hw=self.image_hw, action_dim=self.action_dim,
            prime_provider=self.provider,
            num_envs=self.n_envs, horizon=self.vis_len,
            inference_steps=self.inf_steps,
            sample_temperature=self.temp,
            mask_schedule=self.mask_schedule,
            downscale=self.downscale,
            bottom_bias=self.bottom_bias,
            roi=self.roi,
            flow_baseline=self.flow_baseline,
            goal_x_screens=self.goal_x,
            goal_y_frac=self.goal_y,
            goal_y_weight=self.goal_y_weight,
            goal_radius=self.goal_radius,
            success_bonus=self.success_bonus,
        )
        self.eval_vec = VecMonitor(self.eval_vec)
        self.eval_vec = VecTransposeImage(self.eval_vec)

        # tell me where we write
        self._log(f"[VIS] snaps → {os.path.abspath(self.snapdir)}")
        self._log(f"[VIS] tb_vis → {os.path.abspath(os.path.join(self.run_dir, 'tb_vis'))}")

    # pretty printing that plays nice with tqdm
    def _log(self, msg: str):
        try:
            from tqdm.auto import tqdm
            tqdm.write(msg)
        except Exception:
            print(msg, flush=True)

    def _on_training_start(self) -> None:
        # ensure the first _on_step triggers a visual immediately
        self._last_logged = None

    def _should_vis(self) -> bool:
        steps = int(self.model.num_timesteps)
        if self._last_logged is None:
            return True
        return (steps - self._last_logged) >= self.vis_interval

    def _on_step(self) -> bool:
        if not self._should_vis():
            return True

        obs = self.eval_vec.reset()
        N = self.n_envs
        H, W = self.image_hw
        frames_all = np.zeros((N, self.vis_len, H, W, 3), dtype=np.uint8)

        gs = int(self.model.num_timesteps)
        debug_n = min(self.vis_len, 20)
        debug_lines, a_hist_first20, a_hist_all = [], [], []
        cum = 0.0

        state = None
        episode_start = np.ones((self.n_envs,), dtype=bool)
        for t in range(self.vis_len):
            actions, state = self.model.predict(
                obs, state=state, episode_start=episode_start, deterministic=False
            )
            obs, rews, dones, _ = self.eval_vec.step(actions)
            episode_start = np.array(dones, dtype=bool)

            # unwrap all wrappers to reach the core env
            core = self.eval_vec
            while hasattr(core, "venv"):
                core = core.venv
            last_obs_batch = core._last_obs  # (N,H,W,3)
            frames_all[:, t] = last_obs_batch.copy()

            # decode action for env0
            try:
                a0 = int(actions[0]) if np.ndim(actions) else int(actions)
            except Exception:
                a0 = int(np.array(actions).reshape(-1)[0])
            a_hist_all.append(a0)

            if t < debug_n:
                r0 = float(rews[0]) if np.ndim(rews) else float(rews)
                dx0 = float(getattr(core, "_last_dx", [0.0])[0])
                dy0 = float(getattr(core, "_last_dy", [0.0])[0])
                x0  = float(getattr(core, "_xhat",   [0.0])[0])
                y0  = float(getattr(core, "_yhat",   [0.0])[0])
                gx0 = float(getattr(core, "_goal_x", [0.0])[0])
                gy0 = float(getattr(core, "_goal_y", [0.0])[0])
                wy  = float(getattr(core, "goal_y_weight", 1.0))
                dist0 = float(np.sqrt((gx0-x0)**2 + (wy*(gy0-y0))**2))
                cum += r0
                line = (f"[DEBUG {gs:08d}] t={t:02d} a={a0} "
                        f"dx={dx0:+.4f} dy={dy0:+.4f} x={x0:+.3f} y={y0:+.3f} "
                        f"goal=({gx0:.2f},{gy0:.2f}) dist2d={dist0:+.3f} r={r0:+.3f} R={cum:+.3f}")
                self._log(line)
                debug_lines.append(line)
                a_hist_first20.append(a0)

        # summarize + save debug text
        if debug_lines:
            summary = f"[DEBUG {gs:08d}] sum_R(0..{debug_n-1}) = {cum:+.3f}"
            self._log(summary)
            debug_lines.append(summary)
            from collections import Counter as _Ctr
            hist20 = _Ctr(a_hist_first20)
            histAll = _Ctr(a_hist_all)
            h20_str  = " ".join(f"{k}:{v}" for k, v in sorted(hist20.items()))
            hall_str = " ".join(f"{k}:{v}" for k, v in sorted(histAll.items()))
            line20   = f"[DEBUG {gs:08d}] action_hist_first20  -> {h20_str}"
            lineAll  = f"[DEBUG {gs:08d}] action_hist_full_T={self.vis_len} -> {hall_str}"
            self._log(line20); self._log(lineAll)
            debug_lines.extend([line20, lineAll])
            with open(os.path.join(self.snapdir, f"debug_steps_{gs:08d}.txt"), "w") as f:
                f.write("\n".join(debug_lines) + "\n")

        # file outputs
        gif_path = os.path.join(self.snapdir, f"rollout_{gs:08d}.gif")
        imageio.mimsave(gif_path, list(frames_all[0]), duration=0.08)

        contact = _tile_contact_sheet(frames_all)
        png_path = os.path.join(self.snapdir, f"contact_{gs:08d}.png")
        imageio.imwrite(png_path, contact)

        # TB outputs
        vid = np.transpose(frames_all[0], (0, 3, 1, 2))[None].astype(np.float32) / 255.0
        self.tb.add_video("rollout/env0", vid, global_step=gs, fps=12)
        self.tb.add_image("contact/envs", np.transpose(contact, (2,0,1)), global_step=gs)
        self.tb.flush()

        # mark last time we logged
        self._last_logged = gs
        return True

    def _on_training_end(self) -> None:
        try:
            self.tb.flush(); self.tb.close()
        except Exception:
            pass



# =========================
#  ROI preview helper (saves a PNG overlay)
# =========================

def save_roi_preview(provider: PrimeProvider, roi, bottom_bias: float, downscale: int, out_path: str):
    """Render a single sample frame with the ROI + bottom-bias heat overlay and save to PNG.
    - ROI box is drawn in blue.
    - Heat overlay shows the per-pixel weight map used for flow (after resize).
    """
    frame = provider.sample_prime()  # HxWx3 uint8 at WM size
    H, W = frame.shape[:2]

    # Build weight map at flow resolution then upsample for visualization
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    p = _downscale_to_longside(gray, downscale)
    Hf, Wf = p.shape
    weight = _roi_weight(Hf, Wf, bottom_bias=bottom_bias, roi=roi)
    wnorm = (weight - weight.min()) / (weight.max() - weight.min() + 1e-9)
    wimg = cv2.resize(wnorm, (W, H), interpolation=cv2.INTER_LINEAR)

    # Colorize and blend overlay
    heat = cv2.applyColorMap((wimg * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * heat + 0.4 * frame).astype(np.uint8)

    # Draw ROI rectangle
    y0, y1, x0, x1 = roi
    iy0, iy1, ix0, ix1 = int(y0 * H), int(y1 * H), int(x0 * W), int(x1 * W)
    cv2.rectangle(overlay, (ix0, iy0), (ix1, iy1), (0, 128, 255), 2)
    cv2.putText(overlay, f"ROI y=[{y0:.2f},{y1:.2f}] x=[{x0:.2f},{x1:.2f}] bottom_bias={bottom_bias}",
                (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    imageio.imwrite(out_path, overlay)


# =========================
#  Main (commented line-by-line)
# =========================
def main():
    parser = argparse.ArgumentParser()
    # --- required world model inputs ---
    parser.add_argument("--config", default="/home/sdainelli/Aladdin/neurIPS/configs/config/guided_genie_config.yaml", help="Hydra YAML for the world model")  # path to WM config
    parser.add_argument("--ckpt", default="/home/sdainelli/Aladdin/neurIPS/checkpoints/GenieRedux_Guided_CoinRun_80mln_v1.0/model.pt", help="Checkpoint path for the world model")  # path to WM weights
    parser.add_argument("--init_npz_glob", default="/home/sdainelli/Aladdin/neurIPS/ground_truth_data/coinrun/myrun/all_frames/*.npz", help="Glob for NPZ episodes to sample reset frames")  # dataset

    # --- runtime/device/logging ---
    parser.add_argument("--device", default="cuda", help="torch device, e.g. 'cuda' or 'cpu'")  # device
    parser.add_argument("--logdir", default="runs", help="logging directory")  # output dir

    # --- PPO hyperparams ---
    parser.add_argument("--num_envs", type=int, default=256, help="# parallel WMs in the VecEnv")  # vectorized envs
    parser.add_argument("--horizon", type=int, default=256, help="episode length (TimeLimit)")  # rollout horizon
    parser.add_argument("--n_steps", type=int, default=128, help="PPO n_steps")  # PPO rollout steps
    parser.add_argument("--batch_size", type=int, default=2048, help="PPO batch size")  # PPO batch size
    parser.add_argument("--n_epochs", type=int, default=8, help="PPO epochs")  # PPO epochs per update
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="PPO learning rate")  # PPO LR
    parser.add_argument("--ent_coef", type=float, default=0.02, help="PPO entropy coeff")  # exploration bonus
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range")  # policy clip
    parser.add_argument("--log_interval", type=int, default=100, help="A2C learn() log interval")  # A2C logging
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="total training steps")  # total steps

    # --- WM sampling params ---
    parser.add_argument("--inference_steps", type=int, default=2, help="WM diffusion steps per frame")  # WM steps
    parser.add_argument("--sample_temperature", type=float, default=1.0, help="WM sampling temperature")  # WM temp
    parser.add_argument("--mask_schedule", type=str, default="cosine", help="WM mask schedule")  # WM schedule

    # --- ROI/flow odometry params ---
    parser.add_argument("--downscale", type=int, default=64, help="flow longside resolution")  # flow scale
    parser.add_argument("--bottom_bias", type=float, default=2.0, help="emphasize lower rows")  # bottom weight
    parser.add_argument("--roi", type=str, default="0.20,0.95,0.05,0.95", help="normalized ROI y0,y1,x0,x1")  # ROI
    parser.add_argument("--flow_baseline", type=str, choices=["global","sky","none"], default="sky",
                        help="residual-flow baseline subtraction")  # baseline

    # --- Goal reward params ---
    parser.add_argument("--goal_screens", type=float, default=1.0, help="(legacy) absolute X goal in screens")
    parser.add_argument("--goal_xy", type=str, default=None,
                        help="2D goal as 'x_screens,y_frac' (y in [0,1], 0=top, 1=bottom); overrides --goal_screens")
    parser.add_argument("--goal_y_weight", type=float, default=1.0,
                        help="weight for vertical component in distance (scales y error)")
    parser.add_argument("--goal_radius", type=float, default=0.05, help="success radius in screens")  # radius
    parser.add_argument("--success_bonus", type=float, default=0.0, help="bonus when within radius")  # bonus
    parser.add_argument("--advance_goal_on_success", action="store_true", help="shift goal forward on success")

    # --- Visual logging ---
    parser.add_argument("--vis_interval", type=int, default=20_000, help="steps between visual logs")  # visuals freq
    parser.add_argument("--vis_len", type=int, default=64, help="rollout length for visuals")  # video length
    parser.add_argument("--vis_envs", type=int, default=4, help="#envs for visual rollouts (rows in contact sheet)")

    # --- Choose agent ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ppo", action="store_true", help="Train with PPO")
    group.add_argument("--recurrent_ppo", action="store_true", help="Train with RecurrentPPO (CnnLstmPolicy)")
    group.add_argument("--a2c", action="store_true", help="Train with A2C (CnnPolicy)")

    args = parser.parse_args()

    if args.recurrent_ppo:
        agent_name = "RecurrentPPO"
    elif args.a2c:
        agent_name = "A2C"
    else:
        agent_name = "PPO"
    run_dir = make_run_dir(args.logdir, agent_name)

    print(f"[RUN]  run_dir = {os.path.abspath(run_dir)}")
    print(f"[SNAP] tb_snaps = {os.path.abspath(os.path.join(run_dir, 'tb_snaps'))}")
    print(f"[TB]   tb_train = {os.path.abspath(os.path.join(run_dir, 'tb_train'))}")
    print(f"[TB]   tb_vis   = {os.path.abspath(os.path.join(run_dir, 'tb_vis'))}")

    
    # 2D goal parse
    if args.goal_xy is not None:
        gx, gy = [float(v.strip()) for v in args.goal_xy.split(",")]
    else:
        gx, gy = float(args.goal_screens), 0.5  # default: target mid-height

    # Parse ROI string "y0,y1,x0,x1" -> tuple of floats
    y0, y1, x0, x1 = [float(v.strip()) for v in args.roi.split(",")]
    roi = (y0, y1, x0, x1)

    # Device selection (CUDA if available and requested)
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")

    # Load world model + prime provider
    wm, image_hw, action_dim, _ = load_world_model(args.config, args.ckpt, device)
    provider = PrimeProvider(args.init_npz_glob, image_hw)

    # Save a ROI preview PNG before training starts
    # run_dir = make_run_dir(args.logdir, agent_name) 
    save_roi_preview(provider, roi, args.bottom_bias, args.downscale, os.path.join(run_dir, "roi_preview.png"))


    # Build training VecEnv with GOAL reward only
    vec = GenieReduxBatchedVecEnv(
        wm=wm, device=device, image_hw=image_hw, action_dim=action_dim,
        prime_provider=provider,
        num_envs=args.num_envs, horizon=args.horizon,
        inference_steps=args.inference_steps,
        sample_temperature=args.sample_temperature,
        mask_schedule=args.mask_schedule,
        downscale=args.downscale,
        bottom_bias=args.bottom_bias,
        roi=roi,
        flow_baseline=args.flow_baseline,
        goal_x_screens=gx,
        goal_y_frac=gy,
        goal_y_weight=args.goal_y_weight,
        
        goal_radius=args.goal_radius,
        success_bonus=args.success_bonus,
        advance_goal_on_success=args.advance_goal_on_success,
    )
    vec = VecMonitor(vec)            # record episode stats
    vec = VecTransposeImage(vec)     # HWC -> CHW for CnnPolicy

    tb_train_dir = os.path.join(run_dir, "tb_train")
    print(f"[TB] Writing training TensorBoard logs to: {tb_train_dir}")

    if args.recurrent_ppo:
        model = RecurrentPPO(
            "CnnLstmPolicy", vec,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gamma=0.99, gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=tb_train_dir,
        )
    elif args.a2c:
        model = A2C(
            "CnnPolicy", vec,
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            gamma=0.99, gae_lambda=0.95,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=tb_train_dir,
        )
    else:
        model = PPO(
            "CnnPolicy", vec,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gamma=0.99, gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=tb_train_dir,
        )


    print("rundir:",run_dir)
    # Simple progress bar + visuals every vis_interval
    progress_cb = TqdmProgressCallback(total_timesteps=args.total_timesteps, refresh_sec=0.5)
    vis_cb = VisualLoggerCallback(
        wm=wm, device=device, image_hw=image_hw, action_dim=action_dim, provider=provider,
        run_dir=run_dir,
        vis_interval=args.vis_interval, vis_len=args.vis_len, n_envs=args.vis_envs,
        inference_steps=args.inference_steps, sample_temperature=args.sample_temperature,
        mask_schedule=args.mask_schedule,
        roi=roi, downscale=args.downscale, bottom_bias=args.bottom_bias,
        flow_baseline=args.flow_baseline,
        goal_x=gx, goal_y=gy, goal_y_weight=args.goal_y_weight,
        goal_radius=args.goal_radius, success_bonus=args.success_bonus
    )
    callback = CallbackList([progress_cb, vis_cb])

    # Train and save
    learn_kwargs = dict(total_timesteps=args.total_timesteps, callback=callback)
    if args.a2c:
        learn_kwargs["log_interval"] = args.log_interval
    model.learn(**learn_kwargs)
    os.makedirs(args.logdir, exist_ok=True)
    model.save(os.path.join(run_dir, "models", f"{agent_name}_policy"))


if __name__ == "__main__":
    main()
