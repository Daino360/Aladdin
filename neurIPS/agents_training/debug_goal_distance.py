#!/usr/bin/env python3
"""Compute per-episode agent→goal distances using the FLOW odometry reward."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C, PPO

from agents_training import coinrun_simulator as sim


def parse_action_map(spec: Optional[str]) -> Optional[tuple[int, ...]]:
    if not spec:
        return None
    cleaned = spec.strip()
    if not cleaned:
        return None
    parts = [p.strip() for p in cleaned.split(",")]
    if not parts:
        return None
    try:
        vals = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"Invalid action map spec: {spec}") from exc
    return vals


def load_policy(algo: str, model_path: str):
    if algo == "ppo":
        return PPO.load(model_path, device="auto")
    if algo == "a2c":
        return A2C.load(model_path, device="auto")
    if algo == "recurrent_ppo":
        return RecurrentPPO.load(model_path, device="auto")
    raise ValueError(f"Unsupported algo: {algo}")


def compute_distance(state: dict, goal_x: float, goal_y: float, goal_y_weight: float) -> float:
    return float(
        math.sqrt(
            (goal_x - state["xhat"]) ** 2
            + (goal_y_weight * (goal_y - state["yhat"])) ** 2
        )
    )


def main():
    ap = argparse.ArgumentParser(description="Distance-only debug rollouts for CoinRun FLOW reward")
    ap.add_argument("--model", required=True, help="Path to SB3 policy .zip")
    ap.add_argument("--algo", choices=["ppo", "recurrent_ppo", "a2c"], default="ppo")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--start_level", type=int, default=1000)
    ap.add_argument("--num_levels", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--transpose_nchw", action="store_true")
    ap.add_argument("--expects_uint8", action="store_true")
    ap.add_argument("--goal_xy", type=str, default="1.0,0.2")
    ap.add_argument("--goal_y_weight", type=float, default=0.3)
    ap.add_argument("--goal_radius", type=float, default=0.05)
    ap.add_argument("--roi", type=str, default="0.20,0.95,0.05,0.95")
    ap.add_argument("--downscale", type=int, default=64)
    ap.add_argument("--bottom_bias", type=float, default=2.0)
    ap.add_argument("--dx_clip", type=float, default=0.1)
    ap.add_argument("--dx_smooth", type=float, default=0.5)
    ap.add_argument("--action_map", type=str, default="4,7,1,5,8,2,3")
    ap.add_argument("--output_csv", type=str, default="distance_debug.csv")
    args = ap.parse_args()

    gx, gy = [float(v.strip()) for v in args.goal_xy.split(",")]
    roi = tuple(float(v.strip()) for v in args.roi.split(","))
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    policy = load_policy(args.algo, args.model)
    policy_action_n = getattr(getattr(policy, "action_space", None), "n", None)

    user_action_map = parse_action_map(args.action_map)
    auto_warned = False
    auto_identity = tuple(range(policy_action_n)) if policy_action_n is not None else None

    rows = []
    min_dists = []
    final_dists = []

    print(f"[info] loaded {args.model}")
    print(f"[info] running {args.episodes} episode(s)")

    for ep in range(args.episodes):
        level = args.start_level + ep
        env = sim.make_env(start_level=level, num_levels=args.num_levels)
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
            action_map_to_use = auto_identity
            if not auto_warned:
                print(
                    "[warn] Env and policy action counts differ "
                    f"({env_action_n} vs {policy_action_n}). Defaulting to identity remap 0..{policy_action_n - 1}. "
                    "Pass --action_map to override."
                )
                auto_warned = True
        if action_map_to_use is not None:
            env = sim.ActionRemapWrapper(env, action_map_to_use)

        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        state = {"xhat": 0.0, "yhat": 0.5, "prev_dx": 0.0, "prev_dy": 0.0}
        prev_rgb = sim.obs_to_hwc_uint8(obs)
        initial_dist = compute_distance(state, gx, gy, args.goal_y_weight)
        distances = [initial_dist]
        goal_reached = False
        native_return = 0.0
        custom_return = 0.0
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)

        for t in range(args.max_steps):
            obs_fixed = sim.obs_for_policy(
                obs,
                expects_nchw=args.transpose_nchw,
                expects_uint8=args.expects_uint8,
            )
            if args.algo == "recurrent_ppo":
                action, lstm_state = policy.predict(
                    obs_fixed,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=False,
                )
            else:
                action, _ = policy.predict(obs_fixed, deterministic=False)

            step_out = env.step(action)
            if len(step_out) == 4:
                obs, reward_native, done, info = step_out
                terminated, truncated = bool(done), False
            else:
                obs, reward_native, terminated, truncated, info = step_out

            curr_rgb = sim.obs_to_hwc_uint8(obs)
            r_custom, state = sim.flow_goal_reward(
                prev_rgb,
                curr_rgb,
                roi=roi,
                longside=args.downscale,
                bottom_bias=args.bottom_bias,
                goal_x=gx,
                goal_y=gy,
                goal_y_weight=args.goal_y_weight,
                goal_radius=args.goal_radius,
                dx_clip=args.dx_clip,
                dx_smooth=args.dx_smooth,
                state=state,
            )
            prev_rgb = curr_rgb
            dist_to_goal = compute_distance(state, gx, gy, args.goal_y_weight)
            distances.append(dist_to_goal)
            native_return += float(reward_native)
            custom_return += float(r_custom)
            if dist_to_goal < args.goal_radius:
                goal_reached = True

            done_flag = terminated or truncated
            if args.algo == "recurrent_ppo":
                episode_start = np.array([done_flag], dtype=bool)
                if done_flag:
                    lstm_state = None
            if done_flag:
                break

        env.close()

        if distances:
            min_dist = float(np.min(distances))
            final_dist = float(distances[-1])
            mean_dist = float(np.mean(distances))
        else:
            min_dist = final_dist = mean_dist = float("nan")

        min_dists.append(min_dist)
        final_dists.append(final_dist)

        rows.append({
            "episode": ep,
            "level": level,
            "steps": len(distances) - 1,
            "native_return": native_return,
            "custom_return": custom_return,
            "min_dist": min_dist,
            "final_dist": final_dist,
            "mean_dist": mean_dist,
            "goal_reached": goal_reached,
        })

        print(
            f"[ep {ep+1}/{args.episodes}] level={level} steps={len(distances)-1} "
            f"native={native_return:.3f} custom={custom_return:.3f} "
            f"min_dist={min_dist:.4f} final_dist={final_dist:.4f}"
        )

    if min_dists:
        print(
            f"[summary] median_min={float(np.median(min_dists)):.4f} "
            f"median_final={float(np.median(final_dists)):.4f} "
            f"mean_min={float(np.mean(min_dists)):.4f}"
        )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "level",
                "steps",
                "native_return",
                "custom_return",
                "min_dist",
                "final_dist",
                "mean_dist",
                "goal_reached",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[save] wrote per-episode distances to {out_csv}")


if __name__ == "__main__":
    main()
