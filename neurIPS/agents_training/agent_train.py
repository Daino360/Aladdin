#!/usr/bin/env python3
"""
agent_train.py
=================

Summary
-------
Train a PPO agent **inside a Genie/GenieRedux world model** (Tokenizer + Dynamics),
treating the world model as the environment. At each step:

    Agent (π) --a_t--> WorldModel.sample(...) --f_{t+1}--> Heuristic Reward --> PPO update

Observations are the world model's **predicted frames** (float HWC in [0,1]).
Rewards are computed **from the predicted frames** using a simple, robust heuristic:

- First estimate a horizontal position x ∈ [0,1] from an RGB frame via a modal-background
  segmentation trick (fast, dependency-free).
- Choose a reward mode:
    * 'delta': r_t = bucketize( x_{t+1} - x_t )   (recommended; more stationary)
    * 'abs'  : r_t = bucketize( x_{t+1} )         (can be okay, but less shaped)

Bucketization maps the (small) motion signal into {-2,-1,0,+1,+2} using two thresholds
(small, large) to reduce reward noise and make returns well-bounded.

This script:
- Loads Hydra config + checkpoint to build the Genie/GenieRedux model.
- Builds a small Gym-like Env that calls `model.sample(...)` each step.
- Vectorizes N copies of the Env (synchronous).
- Calls your `ppo.py` (PPOConfig + ppo_train) to actually train the policy.

You *are* using `dynamics.py` and `genie_redux.py`: they are used indirectly via
`construct_model(cfg)` and `model.sample(...)` during every environment step.
"""

from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # /home/.../neurIPS
sys.path.insert(0, str(ROOT))

# ---- stdlib
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import argparse
import glob
import json
import math
import os
import random

# ---- third-party
# import gym
import numpy as np
import torch
import torch.nn.functional as F

# ---- world model (Genie/GenieRedux)
from omegaconf import OmegaConf
from models import construct_model

# ---- your PPO driver (make sure ppo.py exposes these)
from ppo import PPOConfig, ppo_train


# ============================================================================
# Heuristic reward helpers
# ============================================================================

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@dataclass
class XExtractorConfig:
    """Config for foreground-vs-background heuristic to estimate horizontal position x ∈ [0,1]."""
    fg_thresh: float = 0.12  # foreground threshold vs modal background color (L2 distance in RGB)


def extract_x_naive(frame_hwc_uint8: np.ndarray, cfg: XExtractorConfig = XExtractorConfig()) -> float:
    """
    Estimate horizontal position x ∈ [0,1] from an RGB frame (H×W×3, uint8).
    Method:
      1) Downsample and compute the modal background color in a coarse 32^3 RGB histogram.
      2) Foreground = pixels far from that color (L2 distance > fg_thresh).
      3) Weighted horizontal center of mass (weights favor higher rows).
    This is fast and robust enough for "go right" reward shaping.
    """
    img = frame_hwc_uint8.astype(np.float32) / 255.0  # H×W×3, [0,1]
    H, W, _ = img.shape
    ds = img[::4, ::4]                # coarse grid to stabilize background
    if ds.size == 0:
        return 0.0

    # modal background in 32×32×32 histogram
    bins = (ds * 31).astype(np.int32)
    flat = bins.reshape(-1, 3)
    hvals = flat[:, 0] * 32 * 32 + flat[:, 1] * 32 + flat[:, 2]
    mode_hash = np.bincount(hvals).argmax()
    br = mode_hash // (32 * 32)
    bg = (mode_hash // 32) % 32
    bb = mode_hash % 32
    bg_color = np.array([br, bg, bb], dtype=np.float32) / 31.0

    # foreground mask = far from modal background
    dist = np.linalg.norm(img - bg_color[None, None, :], axis=-1)
    fg = dist > cfg.fg_thresh
    ys, xs = np.nonzero(fg)
    if xs.size == 0:
        return 0.0

    # vertical weighting: favor pixels higher in the image (less floor clutter)
    weights = 1.0 - (ys.astype(np.float32) / max(1, H - 1))
    x_cm = (xs.astype(np.float32) * weights).sum() / max(1e-6, weights.sum())
    return float(x_cm / max(1, W - 1))  # normalize to [0,1]


def bucketize_delta(delta: float, small: float, large: float) -> int:
    """
    Map Δx to a discrete reward in {-2,-1,0,+1,+2} using two thresholds:

      +2 if d ≥ +large
      +1 if +small ≤ d < +large
       0 if -small < d < +small
      -1 if -large < d ≤ -small
      -2 if d ≤ -large

    Suggested defaults: small=0.01, large=0.05. Tune to your frame scale & WM noise.
    """
    if delta >=  large: return +2
    if delta >=  small: return +1
    if delta >  -small: return 0
    if delta >  -large: return -1
    return -2


# ============================================================================
# Small tensor/image utils
# ============================================================================

def _to_chw(img_hwc: np.ndarray) -> torch.Tensor:
    """Convert HWC np.uint8/float array into CHW float tensor in [0,1]."""
    if img_hwc.dtype not in (np.float32, np.float64):
        x = torch.from_numpy(img_hwc.astype(np.uint8)).permute(2, 0, 1).float() / 255.0
    else:
        x = torch.from_numpy(img_hwc).permute(2, 0, 1).float()
        if x.max() > 1.0:
            x = x / 255.0
    return x


def _resize_chw(x_chw: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Bilinear resize a CHW tensor to the given (H, W)."""
    return F.interpolate(x_chw.unsqueeze(0), size=size_hw, mode="bilinear", align_corners=False).squeeze(0)


# ============================================================================
# World model loading
# ============================================================================

def load_world_model(cfg_path: str, ckpt_path: str, device: torch.device):
    """
    Load Hydra-configured Genie/GenieRedux model and weights.
    Returns:
      model:             the constructed model on device in eval() mode
      image_hw:          (H, W) the expected image size for the model
      action_dim:        number of discrete actions expected by the model
      max_frames_per_call: masking budget (for reference; we step 1 frame here)
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


# ============================================================================
# Action mapping (optional)
# ============================================================================

class ActionMapper:
    """
    Map policy action IDs → world model action IDs.
    Use when your policy’s action space differs from the model’s action_dim.
    You can pass a JSON mapping, or the preset 'coinrun15_to_7'.
    """
    def __init__(self, src_dim: int, dst_dim: int, mapping: Optional[str] = None):
        self.src_dim, self.dst_dim = int(src_dim), int(dst_dim)
        if mapping is None and src_dim == dst_dim:
            self.table = {i: i for i in range(src_dim)}
        elif mapping == "coinrun15_to_7":
            groups = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 2, 3, 0]
            self.table = {i: int(groups[i]) for i in range(len(groups))}
        else:
            try:
                tbl = json.loads(mapping) if mapping is not None else {}
                self.table = {int(k): int(v) for k, v in tbl.items()}
            except Exception as e:
                raise ValueError("Provide JSON mapping or preset 'coinrun15_to_7'") from e

    def __call__(self, a: int) -> int:
        v = self.table.get(int(a), 0)
        return max(0, min(self.dst_dim - 1, v))


# ============================================================================
# Prime provider (reset frames)
# ============================================================================

class PrimeProvider:
    """
    Samples an initial (reset) frame from a pool of NPZ episodes and resizes it to the WM size.
    We use only the first frame of a random NPZ as the starting prime.
    """
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
                    frames = npz[k]
                    break
            if frames is None:
                for k, v in npz.items():
                    if isinstance(v, np.ndarray) and v.ndim >= 3 and 3 in v.shape:
                        frames = v
                        break
            if frames is None:
                raise RuntimeError(f"No frames found in {p}")

            f = frames
            # normalize to T×H×W×C
            if f.ndim == 4:
                if f.shape[-1] in (1, 3):
                    pass
                elif f.shape[0] in (1, 3):  # C T H W -> T H W C
                    f = np.transpose(f, (1, 2, 3, 0))
                elif f.shape[1] in (1, 3):  # T C H W -> T H W C
                    f = np.transpose(f, (0, 2, 3, 1))
            elif f.ndim == 5 and f.shape[0] in (1, 3):  # C T H W (extra dim)
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


# ============================================================================
# WorldModel-backed Gym Env
# ============================================================================

class BatchedWMVecEnv:
    """
    Vectorized env that steps the world model for ALL envs in one batched call.

    Observation: (N, H, W, C) float32 in [0,1]
    Action:      (N,) ints in [0, policy_action_dim-1] → mapped to WM action_dim
    Reward:      bucketized Δx or absolute x based on `reward_mode`.
    Auto-resets any env that hits `horizon`, but still returns done=True for PPO.
    """
    def __init__(
        self,
        wm,                       # your Genie/GenieRedux model
        device: torch.device,
        image_hw,                 # (H, W)
        action_dim: int,          # WM action_dim (e.g., 7)
        num_envs: int,
        prime_provider,           # has .sample_prime() -> HWC uint8
        *,
        fp: int = 1,
        horizon: int = 128,
        inference_steps: int = 8,
        sample_temperature: float = 1.0,
        mask_schedule: str = "cosine",
        reward_mode: str = "delta",     # 'delta' or 'abs'
        bucket_small: float = 0.01,
        bucket_large: float = 0.05,
        action_mapper=None,             # optional callable: policy_id -> wm_id
    ):
        self.wm = wm
        self.dev = device
        self.H, self.W = image_hw
        self.N = int(num_envs)
        self.fp = max(1, int(fp))
        self.horizon = int(horizon)
        self.inf_steps = int(inference_steps)
        self.temp = float(sample_temperature)
        self.mask_schedule = mask_schedule
        self.reward_mode = reward_mode
        self.small = float(bucket_small)
        self.large = float(bucket_large)
        self.map_action = action_mapper or (lambda a: a)

        # state
        self._t = 0
        self._x_prev = np.zeros(self.N, dtype=np.float32)
        self._prime = None  # (N, C, fp, H, W) torch
        self._last_obs = None  # (N, H, W, C) float

        # build initial primes
        self._reset_all(prime_provider)

    # ----- public API expected by PPO -----
    @property
    def num_envs(self): return self.N

    def reset(self):
        # return last_obs (normalized floats)
        return self._last_obs

    @torch.no_grad()
    def step(self, actions_np: np.ndarray):
        """
        actions_np: (N,) ints in policy space
        Returns: obs(N,H,W,C) float32, rew(N,), done(N,), infos(list of dict)
        """
        # map to WM action ids
        a_wm = np.empty_like(actions_np)
        for i, a in enumerate(actions_np):
            a_wm[i] = int(self.map_action(int(a)))

        acts = torch.from_numpy(a_wm.reshape(self.N, 1)).to(self.dev, dtype=torch.long)

        # one batched WM call
        preds = self.wm.sample(
            prime_frames=self._prime,          # (N,C,fp,H,W)
            actions=acts,                      # (N,1)
            num_frames=1,
            inference_steps=self.inf_steps,
            sample_temperature=self.temp,
            mask_schedule=self.mask_schedule,
            return_recons_only=True,
        )  # (N, C, 1, H, W)

        # update prime window
        self._prime = preds[:, :, -self.fp :]

        # to CPU numpy once (N,H,W,C)
        chw = preds[:, :, 0]                           # (N,C,H,W)
        hwc = (chw.clamp(0,1) * 255.0).to(torch.uint8).permute(0,2,3,1).cpu().numpy()

        # rewards
        x_now = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            x_now[i] = extract_x_naive(hwc[i])

        if self.reward_mode == "delta":
            delta = x_now - self._x_prev
            rew = np.array([bucketize_delta(float(d), self.small, self.large) for d in delta], dtype=np.float32)
        else:  # 'abs'
            # you can also bucketize absolute x if you want; here we leave it continuous in [0,1]
            rew = x_now.astype(np.float32)

        self._x_prev = x_now

        # time & done
        self._t += 1
        done = (self._t >= self.horizon) * np.ones(self.N, dtype=bool)
        infos = [{"x": float(x_now[i])} for i in range(self.N)]

        # auto-reset done envs (like your old Vec wrapper)
        if done.any():
            for i in np.where(done)[0]:
                # re-seed prime with the ORIGINAL reset frame you used last time
                # simplest: just treat current predicted frame as new start
                # if you want real resets from NPZ, store the provider and call it here
                f0 = hwc[i]  # or: provider.sample_prime()
                chw0 = _resize_chw(_to_chw(f0), (self.H, self.W)).unsqueeze(0).unsqueeze(2).to(self.dev)
                self._prime[i:i+1] = chw0
                self._x_prev[i] = extract_x_naive(f0)
                hwc[i] = f0  # show reset obs right away

            self._t = 0  # whole batch shares a clock; simplest is to reset when any done
            # (or keep per-env counters if you prefer; PPO works fine either way)

        # store/return normalized floats
        self._last_obs = hwc.astype(np.float32) / 255.0
        return self._last_obs, rew, done, infos

    # ----- internal -----
    def _reset_all(self, provider):
        """Initialize all envs’ prime windows from primes, compute first obs/x."""
        primes_hwc = []
        for _ in range(self.N):
            f0 = provider.sample_prime()      # HWC uint8
            primes_hwc.append(f0)

        # stack to device tensor (N,C,fp,H,W)
        chws = []
        for f0 in primes_hwc:
            chws.append(_resize_chw(_to_chw(f0), (self.H, self.W)))
        chws = torch.stack(chws, 0)                       # (N,C,H,W)
        self._prime = chws.unsqueeze(2).to(self.dev)      # Fp=1 to start

        # x_prev and last_obs
        self._x_prev = np.array([extract_x_naive(f0) for f0 in primes_hwc], dtype=np.float32)
        self._last_obs = np.stack([(f.astype(np.float32) / 255.0) for f in primes_hwc], 0)
        self._t = 0 


# ============================================================================
# CLI / main
# ============================================================================

def main():
    """
    CLI entry:
      - Loads world model (Hydra YAML + checkpoint).
      - Builds WM-backed vectorized env with heuristic reward (delta/abs).
      - Runs PPO training by calling into your ppo.py (ppo_train).
    """
    ap = argparse.ArgumentParser("PPO in Genie/GenieRedux World Model (heuristic reward)")

    # World model
    ap.add_argument("--config", required=True, help="Hydra YAML for the world model")
    ap.add_argument("--model_ckpt", required=True, help="World model checkpoint (.pt/.pth)")
    ap.add_argument("--device", default="cuda")

    # Initial primes
    ap.add_argument("--init_npz_glob", required=True, help="Glob to NPZs to sample reset frames")

    # Reward settings
    ap.add_argument("--reward_mode", choices=["delta", "abs"], default="delta", help="delta: r=Δx; abs: r=x_{t+1} (both bucketized into {-2..+2})")
    ap.add_argument("--bucket_small", type=float, default=0.01, help="Δx/x threshold for ±1")
    ap.add_argument("--bucket_large", type=float, default=0.05, help="Δx/x threshold for ±2")

    # Env / WM sampling
    ap.add_argument("--num_envs", type=int, default=64, help="Parallel envs")
    ap.add_argument("--horizon", type=int, default=128, help="Episode length inside WM")
    ap.add_argument("--fp", type=int, default=1, help="Number of prime frames kept by WM")
    ap.add_argument("--inference_steps", type=int, default=4, help="MaskGIT inference steps per call")
    ap.add_argument("--sample_temperature", type=float, default=1.0)
    ap.add_argument("--mask_schedule", type=str, default="cosine")

    # Action mapping (if policy action_dim != wm.action_dim)
    ap.add_argument("--policy_action_dim", type=int, default=None, help="If set and != WM actions, map policy actions into WM actions")
    ap.add_argument("--action_map", type=str, default=None, help="JSON mapping or preset 'coinrun15_to_7'")

    # PPO hyperparameters
    ap.add_argument("--total_timesteps", type=int, default=200_000) #1_000_000
    ap.add_argument("--nsteps", type=int, default=64)
    ap.add_argument("--update_epochs", type=int, default=4)
    ap.add_argument("--num_minibatches", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_coef", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--clip_vloss", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_interval", type=int, default=50, help="Save every N PPO updates (set 0 to disable)")

    ap.add_argument("--out_dir", type=str, default="wm_ppo_runs")

    args = ap.parse_args()

    # Device
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")

    # Load world model once
    wm, image_hw, wm_action_dim, _ = load_world_model(args.config, args.model_ckpt, device)

    # Optional action mapper (policy -> WM)
    mapper = None
    policy_action_dim = args.policy_action_dim or wm_action_dim
    if policy_action_dim != wm_action_dim:
        mapper = ActionMapper(
            src_dim=policy_action_dim,
            dst_dim=wm_action_dim,
            mapping=args.action_map,       # "coinrun15_to_7" or JSON or None
        )

    # Build vectorized env over the WM
    prime_provider = PrimeProvider(args.init_npz_glob, image_hw)

    venv = BatchedWMVecEnv(
        wm=wm,
        device=device,
        image_hw=image_hw,
        action_dim=wm_action_dim,          # WM’s true action space (7)
        num_envs=args.num_envs,
        prime_provider=prime_provider,
        fp=args.fp,
        horizon=args.horizon,           # or args.max_steps if that’s your flag
        inference_steps=args.inference_steps,   # keep low: 4–8 is fine
        sample_temperature=args.sample_temperature,
        mask_schedule=args.mask_schedule,
        reward_mode=args.reward_mode,   # "delta" or "abs"
        bucket_small=args.bucket_small,
        bucket_large=args.bucket_large,
        action_mapper=mapper,           # None if policy_action_dim == action_dim
    )


    # Configure PPO and call your ppo.py
    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        nsteps=args.nsteps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        clip_vloss=args.clip_vloss,
    )


    os.makedirs(args.out_dir, exist_ok=True)

    steps_per_update = cfg.nsteps * venv.num_envs
    save_every_updates = args.save_interval  # e.g., 50

    def save_fn(step, state):
        # step is global env-steps consumed so far
        upd = step // steps_per_update
        if save_every_updates and (upd % save_every_updates == 0):
            path = os.path.join(args.out_dir, f"ppo_update{upd:05d}.pt")
            torch.save(state, path)
            print(f"[save] {path}")


    # PPO expects CHW in the policy; env delivers HWC to ppo_train which will do the conversion.
    C, H, W = 3, image_hw[0], image_hw[1]

    # >>> This is where we "call ppo.py" <<<
    policy = ppo_train(
        venv=venv,
        device=device,
        obs_shape=(C, H, W),
        action_dim=(policy_action_dim or wm_action_dim),
        cfg=cfg,
        seed=args.seed,
        save_fn=save_fn,   # optional; omit if you don’t want periodic saving
    )


if __name__ == "__main__":
    main()
