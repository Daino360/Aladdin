#!/usr/bin/env python3
"""
Debug + evaluate a saved SB3 PPO policy in the Genie world model, and export GIF + PNG strip.
This version auto-detects observation layout (HWC vs CHW), avoids double-transpose,
logs action stats, and guards against dtype/scale issues.
"""
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

import gymnasium as gym  # for spaces/introspection
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage

from agent_train import GenieReduxBatchedVecEnv, PrimeProvider, load_world_model

# ----------------- paths & quick knobs -----------------
config = "/home/sdainelli/Aladdin/neurIPS/configs/config/guided_genie_config.yaml"
ckpt   = "/home/sdainelli/Aladdin/neurIPS/checkpoints/GenieRedux_Guided_CoinRun_80mln_v1.0/model.pt"
logdir = "/home/sdainelli/Aladdin/neurIPS/agents_training/runs/PPO/PPO_01/models"
# Use the exact filename you saved, typically ends with .zip
model_path = f"{logdir}/PPO_policy.zip"
init_npz_glob = "/home/sdainelli/Aladdin/neurIPS/ground_truth_data/coinrun/myrun/all_frames/*.npz"

mode = "policy"      # "policy" or "random"
frames_to_record = 256
fps = 20
inference_steps = 8   # lower = faster; raise if visuals look too noisy
stochastic_eval = False  # if True, sample actions instead of greedy

out_dir = os.path.join(logdir, "preview")
os.makedirs(out_dir, exist_ok=True)
gif_path = os.path.join(out_dir, f"rollout_{mode}.gif")
png_path = os.path.join(out_dir, f"rollout_{mode}_strip.png")

# ----------------- build world model + env -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wm, image_hw, action_dim, _ = load_world_model(config, ckpt, device)
provider = PrimeProvider(init_npz_glob, image_hw)

# base env for clean recording (no wrappers yet)
base_env = GenieReduxBatchedVecEnv(
    wm=wm, device=device, image_hw=image_hw, action_dim=action_dim,
    prime_provider=provider,
    num_envs=1, horizon=256,
    inference_steps=inference_steps, sample_temperature=1.0,
    mask_schedule="cosine", use_amp=True,
)

# Decide whether to transpose based on base env observation layout
obs_shape = base_env.observation_space.shape
need_transpose = (len(obs_shape) == 3 and obs_shape[-1] in (1, 3))  # HWC -> CHW for SB3

if need_transpose:
    env = VecTransposeImage(base_env)
    layout_note = "HWC→CHW via VecTransposeImage"
else:
    env = base_env
    layout_note = "CHW already (no transpose)"

# Always good to monitor
env = VecMonitor(env)

print("[env] observation_space:", env.observation_space)
print("[env] action_space      :", env.action_space)
print("[env] layout            :", layout_note)

# ----------------- load policy (if in policy mode) -----------------
policy = None
if mode == "policy":
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}\nTip: confirm filename and extension (.zip)")
    policy = PPO.load(model_path, env=env, device=device)  # attach eval env for space checks

# ----------------- helpers -----------------

def to_vis_frame(obs_batch) -> np.ndarray:
    """Convert current observation batch to a single HxWx3 uint8 frame for saving.
    Works for obs in either CHW or HWC, float [0,1] or uint8 [0,255].
    """
    x = obs_batch[0]
    if x.ndim != 3:
        raise ValueError(f"Unexpected obs ndim: {x.ndim}, shape={x.shape}")

    # If float, assume [0,1] and scale
    if x.dtype != np.uint8:
        x = np.clip(x, 0.0, 1.0) * 255.0
        x = x.astype(np.uint8)

    # CHW -> HWC
    if x.shape[0] in (1, 3) and x.shape[1] == x.shape[2]:
        x = np.transpose(x, (1, 2, 0))

    # If single channel, tile to 3
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    return x

# ----------------- rollout & record -----------------
obs = env.reset()
frames, actions = [], []
sum_absdiff = 0.0
prev_frame = None

for t in range(frames_to_record):
    if mode == "policy":
        # stochastic_eval lets you see if the policy *can* jump even if greedy stays right
        action, _ = policy.predict(obs, deterministic=not stochastic_eval)
    else:
        # Random sanity check: if this shows motion/variety but the policy doesn't,
        # the issue is with the policy (weights, wrappers, spaces), not the WM env.
        action = np.array([env.action_space.sample()])

    obs, rewards, dones, infos = env.step(action)

    frame = to_vis_frame(obs)
    frames.append(frame)

    # Record scalar action for histogram (Discrete only)
    if isinstance(env.action_space, gym.spaces.Discrete):
        actions.append(int(action[0]))
    else:
        # For non-discrete, store a hashable repr
        actions.append(tuple(np.array(action[0]).tolist()))

    if prev_frame is not None:
        sum_absdiff += float(np.mean(np.abs(frame.astype(np.int16) - prev_frame.astype(np.int16))))
    prev_frame = frame

    if dones[0]:
        obs = env.reset()

# ----------------- diagnostics -----------------
if len(frames) == 0:
    raise RuntimeError("No frames recorded – check env reset/step and observation pipeline.")

# Action histogram
try:
    uniq, counts = np.unique(np.array(actions, dtype=object), return_counts=True)
    action_stats = dict(zip([int(u) if isinstance(u, (np.integer, int)) else u for u in uniq.tolist()], counts.tolist()))
except Exception:
    action_stats = {"samples": len(actions)}

print(f"[{mode}] actions chosen: {action_stats}")
print(f"[{mode}] avg per-frame absolute pixel diff: {sum_absdiff / max(1, len(frames)-1):.3f}")

# Heuristic hints
if isinstance(env.action_space, gym.spaces.Discrete) and len(action_stats) == 1:
    only = list(action_stats.keys())[0]
    print("\n[hint] Policy used a single action for the whole rollout -> likely wrapper/space mismatch or bad weights.")
    print("      - Confirm the env observation layout matches training (this script auto-detects HWC/CHW).")
    print("      - Confirm action set / mapping is identical to training (same action_dim, same ordering).")
    print("      - If training used VecNormalize/FrameStack, you must attach the same wrappers before loading.")

# ----------------- save GIF & PNG strip -----------------
imageio.mimsave(gif_path, frames, fps=fps)

H, W, _ = frames[0].shape
strip = Image.new("RGB", (W * len(frames), H))
for i, fr in enumerate(frames):
    strip.paste(Image.fromarray(fr), (i * W, 0))
strip.save(png_path, optimize=True)

print(f"Saved GIF  → {gif_path}")
print(f"Saved PNG  → {png_path}")
