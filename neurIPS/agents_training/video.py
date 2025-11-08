#!/usr/bin/env python3
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage

from agent_train import GenieReduxBatchedVecEnv, PrimeProvider, load_world_model

# --- config (edit paths as you like) ---
config = "/home/sdainelli/Aladdin/neurIPS/configs/config/guided_genie_config.yaml"
ckpt   = "/home/sdainelli/Aladdin/neurIPS/checkpoints/GenieRedux_Guided_CoinRun_80mln_v1.0/model.pt"
logdir = "/home/sdainelli/Aladdin/neurIPS/agents_training/runs/genie_sb3_ppoflow"
model_path = f"{logdir}/ppo_genie_optflow.zip"   # <- include .zip if you saved with PPO.save
init_npz_glob = "/home/sdainelli/Aladdin/neurIPS/ground_truth_data/coinrun/myrun/all_frames/*.npz"

frames_to_record = 256
fps = 20
out_dir = os.path.join(logdir, "preview")
os.makedirs(out_dir, exist_ok=True)
gif_path = os.path.join(out_dir, "rollout.gif")
png_path = os.path.join(out_dir, "rollout_strip.png")

# --- build world model + provider ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wm, image_hw, action_dim, _ = load_world_model(config, ckpt, device)
provider = PrimeProvider(init_npz_glob, image_hw)

# --- small eval env: num_envs=1 for clean recording ---
eval_vec = GenieReduxBatchedVecEnv(
    wm=wm, device=device, image_hw=image_hw, action_dim=action_dim,
    prime_provider=provider,
    num_envs=1,            # <-- single stream for recording
    horizon=256,
    fp=1,
    inference_steps=8,     # can bump up/down; 8 is fine for eval quality
    sample_temperature=1.0,
    mask_schedule="cosine",
    use_amp=True,
)
eval_vec = VecMonitor(eval_vec)
eval_vec = VecTransposeImage(eval_vec)  # policy expects CHW

# --- load policy ---
model = PPO.load(model_path, env=eval_vec, device=device)  # attach your small eval env here

# --- rollout & record: write GIF and a single PNG strip ---
frames = []
obs = eval_vec.reset()
for t in range(frames_to_record):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = eval_vec.step(action)

    # VecTransposeImage → obs is (N, C, H, W) uint8; convert to HWC for saving
    frame = np.transpose(obs[0], (1, 2, 0)).astype(np.uint8)
    frames.append(frame)

    if dones[0]:
        obs = eval_vec.reset()

# save GIF
imageio.mimsave(gif_path, frames, fps=fps)

# save horizontal PNG strip
H, W, _ = frames[0].shape
strip = Image.new("RGB", (W * len(frames), H))
for i, fr in enumerate(frames):
    strip.paste(Image.fromarray(fr), (i * W, 0))
strip.save(png_path, optimize=True)

print(f"Saved GIF  → {gif_path}")
print(f"Saved PNG  → {png_path}")
