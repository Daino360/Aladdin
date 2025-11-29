#!/usr/bin/env python3
import argparse, time, csv
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO
from typing import Optional

np.bool8 = bool  # SB3 / NumPy compat

# ----------------- ENV / OBS HELPERS -------------------

def make_env(start_level: int, num_levels: int = 1, distribution_mode: str = "easy"):
    return gym.make(
        "GymV21Environment-v0",
        env_id="procgen:procgen-coinrun-v0",
    )


class ActionRemapWrapper(gym.ActionWrapper):
    """Expose a reduced Discrete action space while remapping to the env space."""
    def __init__(self, env, action_map: tuple[int, ...]):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError("ActionRemapWrapper requires a Discrete action space.")
        if not action_map:
            raise ValueError("action_map must contain at least one entry.")
        arr = np.asarray(action_map, dtype=np.int64)
        if np.any(arr < 0):
            raise ValueError(f"action_map must be non-negative, got {action_map}")
        max_idx = int(np.max(arr))
        if max_idx >= env.action_space.n:
            raise ValueError(
                f"action_map references env action {max_idx}, but env only supports 0..{env.action_space.n - 1}"
            )
        self._action_map = tuple(int(v) for v in arr.tolist())
        self._policy_n = len(self._action_map)
        self.action_space = spaces.Discrete(self._policy_n)

    @staticmethod
    def _to_scalar(action) -> int:
        if isinstance(action, (int, np.integer)):
            return int(action)
        arr = np.asarray(action)
        if arr.size != 1:
            raise ValueError(f"Expected scalar action, got shape {arr.shape}")
        return int(arr.reshape(-1)[0])

    def action(self, act):
        idx = self._to_scalar(act)
        if idx < 0 or idx >= self._policy_n:
            raise ValueError(f"Policy action {idx} outside remapped range 0..{self._policy_n - 1}")
        return self._action_map[idx]

def obs_to_hwc_uint8(obs) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected obs ndim: {arr.shape}")
    if arr.shape[-1] in (1, 3):
        hwc = arr
    elif arr.shape[0] in (1, 3):
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Unexpected obs shape: {arr.shape}")
    if hwc.dtype == np.uint8:
        out = hwc
    else:
        mx = float(np.nanmax(hwc)) if hwc.size else 1.0
        hwc = hwc.astype(np.float32)
        if mx > 1.5:
            hwc = hwc / 255.0
        out = np.clip(np.round(hwc * 255.0), 0, 255).astype(np.uint8)
    return out

def obs_for_policy(obs, expects_nchw: bool, expects_uint8: bool):
    x = np.asarray(obs)
    # layout
    if expects_nchw:
        if x.ndim == 3 and x.shape[0] in (1,3):
            pass
        elif x.ndim == 3 and x.shape[-1] in (1,3):
            x = np.transpose(x, (2,0,1))
        else:
            raise ValueError(f"Bad obs shape for NCHW: {x.shape}")
    else:
        if x.ndim == 3 and x.shape[-1] in (1,3):
            pass
        elif x.ndim == 3 and x.shape[0] in (1,3):
            x = np.transpose(x, (1,2,0))
        else:
            raise ValueError(f"Bad obs shape for HWC: {x.shape}")
    # dtype/scale
    if expects_uint8:
        if x.dtype != np.uint8:
            if x.dtype.kind in "fc":
                x = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
            else:
                x = np.clip(x, 0, 255).astype(np.uint8)
    else:
        if x.dtype == np.uint8:
            x = (x.astype(np.float32) / 255.0)
        else:
            x = x.astype(np.float32)
            if np.nanmax(x) > 1.5:
                x = x / 255.0
    return x

# ---------------- FLOW-ODOMETRY REWARD -----------------

def _downscale_to_longside(gray: np.ndarray, longside: int) -> np.ndarray:
    H, W = gray.shape
    if longside is None or max(H, W) == longside:
        return gray
    s = float(longside) / float(max(H, W))
    new_size = (max(8, int(W * s)), max(8, int(H * s)))
    return cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

def _flow_residual(prev_gray: np.ndarray, next_gray: np.ndarray, *, longside: int = 64, baseline: str = "sky"):
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
                roi: tuple[float, float, float, float]) -> np.ndarray:
    row = np.linspace(0, 1, H, dtype=np.float32)
    wv = 1.0 + (row ** 2) * (bottom_bias - 1.0)
    wv /= max(np.mean(wv), 1e-9)
    weight = np.repeat(wv[:, None], W, axis=1)
    y0, y1, x0, x1 = roi
    iy0, iy1 = int(y0 * H), int(y1 * H)
    ix0, ix1 = int(x0 * W), int(x1 * W)
    mask = np.zeros_like(weight, dtype=np.float32)
    mask[iy0:iy1, ix0:ix1] = 1.0
    return weight * mask

def flow_goal_reward(prev_rgb: np.ndarray, curr_rgb: np.ndarray,
                     *, roi, longside: int, bottom_bias: float,
                     goal_x: float, goal_y: float, goal_y_weight: float,
                     goal_radius: float, dx_clip: float, dx_smooth: float,
                     state: dict) -> tuple[float, dict]:
    p = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    c = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
    fx_res, fy_res = _flow_residual(p, c, longside=longside, baseline="sky")
    Hf, Wf = fx_res.shape
    weight = _roi_weight(Hf, Wf, bottom_bias=bottom_bias, roi=roi)
    denom  = weight.sum() + 1e-9

    dxf_px = float(((-fx_res) * weight).sum() / denom)
    dyf_px = float(((-fy_res) * weight).sum() / denom)
    dx = np.clip(dxf_px / max(Wf, 1), -dx_clip, dx_clip)
    dy = np.clip(dyf_px / max(Hf, 1), -dx_clip, dx_clip)
    dx = (1.0 - dx_smooth) * dx + dx_smooth * state["prev_dx"]
    dy = (1.0 - dx_smooth) * dy + dx_smooth * state["prev_dy"]
    state["prev_dx"], state["prev_dy"] = dx, dy

    x_prev, y_prev = state["xhat"], state["yhat"]
    state["xhat"]  = max(0.0, x_prev + dx)
    state["yhat"]  = float(np.clip(y_prev + dy, 0.0, 1.0))

    gx, gy = goal_x, goal_y
    prev_dist = float(np.sqrt((gx - x_prev)**2 + (goal_y_weight*(gy - y_prev))**2))
    dist      = float(np.sqrt((gx - state["xhat"])**2 + (goal_y_weight*(gy - state["yhat"]))**2))
    r = 5.0 * (prev_dist - dist)
    if dist < goal_radius:
        r += 1.0
    return float(r), state

DEFAULT_MAX_GIFS = 25

def _save_gif(frames, out_path, fps=15):
    if not frames:
        return False
    dur = max(1, int(1000 / fps))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=dur,
        loop=0,
        optimize=False,
    )
    return True

def _save_strip(frames, out_path, max_strip_frames=128):
    if not frames:
        return False
    if len(frames) > max_strip_frames:
        step = max(1, len(frames) // max_strip_frames)
        frames = frames[::step][:max_strip_frames]
    H = frames[0].height
    W = sum(f.width for f in frames)
    canvas = Image.new("RGB", (W, H))
    x = 0
    for f in frames:
        canvas.paste(f, (x, 0))
        x += f.width
    canvas.save(out_path)
    return True

# --------------------------- MAIN ------------------------------------

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--start_level", type=int, default=1000)
    ap.add_argument("--num_levels", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--frame_h", type=int, default=128)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out_dir", default="simulator_eval_outputs")
    ap.add_argument("--transpose_nchw", action="store_true")
    ap.add_argument("--expects_uint8", action="store_true")
    # custom reward params (match your training)
    ap.add_argument("--roi", type=str, default="0.20,0.95,0.05,0.95")
    ap.add_argument("--goal_xy", type=str, default="1.0,0.2")
    ap.add_argument("--goal_y_weight", type=float, default=0.3)
    ap.add_argument("--goal_radius", type=float, default=0.05)
    ap.add_argument("--downscale", type=int, default=64)
    ap.add_argument("--bottom_bias", type=float, default=2.0)
    ap.add_argument("--dx_clip", type=float, default=0.1)
    ap.add_argument("--dx_smooth", type=float, default=0.5)
    ap.add_argument("--action_map", type=str, default="4,7,1,5,8,2,3", help="Comma-separated mapping from policy to Procgen actions (e.g., '0,1,2,5,7,8,11').")
    ap.add_argument("--algo", choices=["ppo", "recurrent_ppo", "a2c"], default="ppo", help="Policy algorithm to load.")
    ap.add_argument("--save_csv", action="store_true")
    ap.add_argument("--max_gifs", type=int, default=DEFAULT_MAX_GIFS, help="Maximum number of best-episode GIFs/strips to export (0 to disable).")
    args = ap.parse_args()

    if args.max_gifs < 0:
        ap.error("--max_gifs must be non-negative")
    max_gifs = args.max_gifs
    if max_gifs == 0:
        print("[info] GIF export disabled (--max_gifs=0)")

    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = (Path(args.out_dir).expanduser().resolve() / f"coinrun{run_stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    if args.algo == "recurrent_ppo":
        model = RecurrentPPO.load(args.model, device="auto")
    elif args.algo == "a2c":
        model = A2C.load(args.model, device="auto")
    else:
        model = PPO.load(args.model, device="auto")
    print(f"[info] loaded {args.model}")
    policy_action_n = getattr(getattr(model, "action_space", None), "n", None)

    gx, gy = [float(v.strip()) for v in args.goal_xy.split(",")]
    roi = tuple(float(v.strip()) for v in args.roi.split(","))

    csv_rows = []
    user_action_map = _parse_action_map_arg(args.action_map)
    if user_action_map is not None:
        if policy_action_n is not None and len(user_action_map) != policy_action_n:
            raise ValueError(
                f"Provided action_map has {len(user_action_map)} entries, but policy was trained with {policy_action_n} actions."
            )
        print(f"[info] using user-provided action remap: {user_action_map}")
    auto_warned = False
    auto_identity_map = tuple(range(policy_action_n)) if policy_action_n is not None else None
    
    all_custom_rewards = []
    all_native_rewards = []
    best_episodes = []

    for ep in range(args.episodes):
        level = args.start_level + ep
        env = make_env(start_level=level, num_levels=args.num_levels)
        env_action_n = getattr(getattr(env, "action_space", None), "n", None)
        action_map_to_use = user_action_map
        if (
            action_map_to_use is None
            and policy_action_n is not None
            and env_action_n is not None
            and env_action_n != policy_action_n
        ):
            if env_action_n < policy_action_n:
                raise ValueError(
                    f"Env action space ({env_action_n}) smaller than policy actions ({policy_action_n}); cannot remap."
                )
            action_map_to_use = auto_identity_map
            if not auto_warned:
                print(
                    "[warn] Env and policy action counts differ "
                    f"({env_action_n} vs {policy_action_n}). Defaulting to identity remap 0..{policy_action_n - 1}. "
                    "Pass --action_map to override."
                )
                auto_warned = True
        if action_map_to_use is not None:
            env = ActionRemapWrapper(env, action_map_to_use)

        print(f"[info] ep {ep+1}/{args.episodes} | level={level}")

        frames = []
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        im0 = Image.fromarray(obs_to_hwc_uint8(obs))
        if args.frame_h and im0.height != args.frame_h:
            w = int(im0.width * (args.frame_h / im0.height))
            im0 = im0.resize((w, args.frame_h), Image.NEAREST)
        frames.append(im0)

        st = {"xhat": 0.0, "yhat": 0.5, "prev_dx": 0.0, "prev_dy": 0.0}
        prev_rgb = obs_to_hwc_uint8(obs)

        native_return = 0.0
        custom_return = 0.0

        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)

        for t in range(args.max_steps):
            obs_fixed = obs_for_policy(obs,
                                       expects_nchw=args.transpose_nchw,
                                       expects_uint8=args.expects_uint8)
            # always stochastic for discrete tasks
            if args.algo == "recurrent_ppo":
                action, lstm_state = model.predict(
                    obs_fixed,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=False,
                )
            else:
                action, _ = model.predict(obs_fixed, deterministic=False)

            step_out = env.step(action)
            if len(step_out) == 4:
                obs, reward_native, done, info = step_out
                terminated, truncated = bool(done), False
            else:
                obs, reward_native, terminated, truncated, info = step_out

            curr_rgb = obs_to_hwc_uint8(obs)

            r_custom, st = flow_goal_reward(
                prev_rgb, curr_rgb,
                roi=roi, longside=args.downscale, bottom_bias=args.bottom_bias,
                goal_x=gx, goal_y=gy, goal_y_weight=args.goal_y_weight,
                goal_radius=args.goal_radius, dx_clip=args.dx_clip, dx_smooth=args.dx_smooth,
                state=st
            )
            native_return += float(reward_native)
            custom_return += float(r_custom)

            # capture every step
            im = Image.fromarray(curr_rgb)
            if args.frame_h and im.height != args.frame_h:
                w = int(im.width * (args.frame_h / im.height))
                im = im.resize((w, args.frame_h), Image.NEAREST)
            frames.append(im)

            prev_rgb = curr_rgb

            done_flag = terminated or truncated
            if args.algo == "recurrent_ppo":
                episode_start = np.array([done_flag], dtype=bool)
                if done_flag:
                    lstm_state = None
            if done_flag:
                # Stop the rollout once the env signals termination/truncation.
                break

        base = out_dir / f"lvl{level}"
        gif_path = base.with_suffix(".gif")
        png_path = base.with_name(base.name + "_strip.png")
        step_count = len(frames)
        print(f"[ret] native={native_return:.3f}  custom={custom_return:.3f}")
        candidate_native = np.isclose(native_return, 10.0)
        if candidate_native:
            if max_gifs == 0:
                print(f"[queue] lvl{level} native return met but GIF export disabled; frames discarded.")
                frames.clear()
            else:
                best_episodes.append({
                    "level": level,
                    "frames": frames,
                    "steps": step_count,
                    "native_return": native_return,
                    "custom_return": custom_return,
                    "gif_path": gif_path,
                    "png_path": png_path,
                    "episode_index": ep,
                })
                best_episodes.sort(key=lambda e: (e["steps"], e["episode_index"]))
                while len(best_episodes) > max_gifs:
                    removed = best_episodes.pop()
                    removed["frames"].clear()
                print(f"[queue] lvl{level} kept for GIF export (steps={step_count}, native={native_return:.3f}, queue={len(best_episodes)}/{max_gifs})")
        else:
            print(f"[skip] lvl{level} not eligible for GIF export (native={native_return:.3f}, steps={step_count})")
            frames.clear()
        
        all_custom_rewards.append(custom_return)
        all_native_rewards.append(native_return)
        
        if args.save_csv:
            csv_rows.append({
                "episode": f"lvl{level}",
                "native_return": native_return,
                "custom_return": custom_return,
                "steps": step_count,
                "goal_x": gx, "goal_y": gy
            })
        env.close()

    print("Average Native Reward:", np.mean(all_native_rewards))
    print("Average Custom Reward:", np.mean(all_custom_rewards))

    if best_episodes:
        print(f"[info] Saving GIFs/strips for {len(best_episodes)} best episodes (max {max_gifs}).")
        for entry in best_episodes:
            frames = entry["frames"]
            ok1 = _save_gif(frames, entry["gif_path"], fps=args.fps)
            ok2 = _save_strip(frames, entry["png_path"], max_strip_frames=128)
            gif_status = "ok" if ok1 and entry["gif_path"].exists() else "FAIL"
            strip_status = "ok" if ok2 and entry["png_path"].exists() else "FAIL"
            print(f"[save] gif:   {entry['gif_path']}  ({gif_status})")
            print(f"[save] strip: {entry['png_path']}  ({strip_status})")
            frames.clear()
    else:
        print("[info] No episodes reached native reward 10; skipping GIF/PNG export.")

    if args.save_csv and csv_rows:
        csv_path = out_dir / "simulator_eval_summary.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode","native_return","custom_return","steps","goal_x","goal_y"])
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)

            writer.writerow({
                "episode": "AVERAGE",
                "native_return": f"{np.mean(all_native_rewards):.3f}",
                "custom_return": f"{np.mean(all_custom_rewards):.3f}",
                "steps": "",
                "goal_x": "",
                "goal_y": "",
            })
            writer.writerow({
                "episode": "MAX",
                "native_return": f"{np.max(all_native_rewards):.3f}",
                "custom_return": f"{np.max(all_custom_rewards):.3f}",
                "steps": "",
                "goal_x": "",
                "goal_y": "",
            })
            writer.writerow({
                "episode": "MIN",
                "native_return": f"{np.min(all_native_rewards):.3f}",
                "custom_return": f"{np.min(all_custom_rewards):.3f}",
                "steps": "",
                "goal_x": "",
                "goal_y": "",
            })
        print(f"[save] {csv_path}")

if __name__ == "__main__":
    main()
