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

# ----------------- paths & quick knobs -----------------
config = "/home/sdainelli/Aladdin/neurIPS/configs/config/guided_genie_config.yaml"
ckpt   = "/home/sdainelli/Aladdin/neurIPS/checkpoints/GenieRedux_Guided_CoinRun_80mln_v1.0/model.pt"
logdir = "/home/sdainelli/Aladdin/neurIPS/agents_training/runs/genie_sb3_ppoflow"
model_path = f"{logdir}/ppo_genie_goal"     # use the exact filename you saved
init_npz_glob = "/home/sdainelli/Aladdin/neurIPS/ground_truth_data/coinrun/myrun/all_frames/*.npz"

mode = "policy"     # "policy" or "random"
frames_to_record = 256
fps = 20
inference_steps = 8  # lower = faster; raise if visuals look too noisy

out_dir = os.path.join(logdir, "preview")
os.makedirs(out_dir, exist_ok=True)
gif_path = os.path.join(out_dir, f"rollout_{mode}.gif")
png_path = os.path.join(out_dir, f"rollout_{mode}_strip.png")

# ----------------- build world model + env -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wm, image_hw, action_dim, _ = load_world_model(config, ckpt, device)
provider = PrimeProvider(init_npz_glob, image_hw)

# 1 env for clean recording
env = GenieReduxBatchedVecEnv(
    wm=wm, device=device, image_hw=image_hw, action_dim=action_dim,
    prime_provider=provider,
    num_envs=1, horizon=256, fp=1,
    inference_steps=inference_steps, sample_temperature=1.0,
    mask_schedule="cosine", use_amp=True
)
env = VecMonitor(env)
env = VecTransposeImage(env)  # policy expects CHW; we'll transpose back for saving

# ----------------- load policy (if in policy mode) -----------------
policy = None
if mode == "policy":
    policy = PPO.load(model_path, env=env, device=device)  # attach your small eval env here


# ----------------- rollout & record -----------------
obs = env.reset()
frames, actions = [], []
sum_absdiff = 0.0
prev_frame = None

for t in range(frames_to_record):
    if mode == "policy":
            action, _ = policy.predict(obs, deterministic=True)
    else:  # random sanity check
        action = np.array([env.action_space.sample()], dtype=np.int64)

    obs, rewards, dones, infos = env.step(action)

    # VecTransposeImage → obs is (N,C,H,W). Convert to HWC for saving.
    frame = np.transpose(obs[0], (1, 2, 0)).astype(np.uint8)
    frames.append(frame)
    actions.append(int(action[0]))

    if prev_frame is not None:
        sum_absdiff += float(np.mean(np.abs(frame.astype(np.int16) - prev_frame.astype(np.int16))))
    prev_frame = frame

    if dones[0]:
        obs = env.reset()

# ----------------- diagnostics -----------------
uniq, counts = np.unique(actions, return_counts=True)
print(f"[{mode}] actions chosen: {dict(zip(uniq.tolist(), counts.tolist()))}")
print(f"[{mode}] avg per-frame absolute pixel diff: {sum_absdiff / max(1, frames_to_record-1):.3f}")

# If avg diff is ~0, you’re getting identical frames → either the WM call isn’t advancing
# or actions are effectively no-ops. In that case, rerun with mode='random'.
# If random shows motion but policy doesn’t, training likely didn’t learn (or outputs noop).

# ----------------- save GIF & PNG strip -----------------
imageio.mimsave(gif_path, frames, fps=fps)

H, W, _ = frames[0].shape
strip = Image.new("RGB", (W * len(frames), H))
for i, fr in enumerate(frames):
    strip.paste(Image.fromarray(fr), (i * W, 0))
strip.save(png_path, optimize=True)

print(f"Saved GIF  → {gif_path}")
print(f"Saved PNG  → {png_path}")
