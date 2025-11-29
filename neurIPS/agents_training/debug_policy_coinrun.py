"""Script to debug a trained SB3 policy (PPO/RecurrentPPO/A2C) on Procgen CoinRun.
Saves per-step action distributions, values, entropies, and other stats to CSV.
Also saves a summary of the policy architecture and weight statistics.
Optionally saves a PNG bar chart of mean action probabilities across all steps."""
import argparse, csv, json
from pathlib import Path

import numpy as np
np.bool8 = bool  # workaround for NumPy 1.24+ and older SB3 versions
import torch
import gymnasium as gym
from stable_baselines3 import PPO, A2C

try:
    from sb3_contrib import RecurrentPPO
except ImportError:  # pragma: no cover - optional dependency at runtime
    RecurrentPPO = None

# -------------------------
# Helpers
# -------------------------
def make_env(start_level: int, num_levels: int = 1, distribution_mode: str = "easy"):
    # Procgen via Shimmy (Gym v21 wrapper exposed in Gymnasium)
    return gym.make(
        "GymV21Environment-v0",
        env_id="procgen:procgen-coinrun-v0",
    )

def to_hwc_uint8(obs) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3):  # HWC
        pass
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Unexpected obs shape: {arr.shape}")
    return arr.astype(np.uint8)

def policy_summary(model) -> str:
    lines = []
    lines.append("=== Policy Summary ===")
    lines.append(repr(model.policy))
    if hasattr(model.policy, "features_extractor"):
        lines.append("\n--- features_extractor ---")
        lines.append(repr(model.policy.features_extractor))
    if hasattr(model.policy, "mlp_extractor"):
        lines.append("\n--- mlp_extractor ---")
        lines.append(repr(model.policy.mlp_extractor))
    return "\n".join(lines)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to SB3 .zip")
    ap.add_argument("--algo", choices=["ppo", "recurrent_ppo", "a2c"], default="ppo",
                    help="Algorithm used while training the policy.")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--start_level", type=int, default=1000)    # base level seed
    ap.add_argument("--num_levels", type=int, default=1)        # 1=fixed deterministic level
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--deterministic", action="store_true", help="Use greedy actions")
    ap.add_argument("--out_dir", default="policy_debug")
    ap.add_argument("--plot", action="store_true", help="Save a PNG of mean action probs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] writing to {out_dir}")

    # Load model (weights/schedules deserialization warnings are fine for inference)
    if args.algo == "recurrent_ppo":
        if RecurrentPPO is None:
            raise ImportError("sb3-contrib is required for RecurrentPPO (--algo recurrent_ppo).")
        model = RecurrentPPO.load(args.model, device="auto")
    elif args.algo == "a2c":
        model = A2C.load(args.model, device="auto")
    else:
        model = PPO.load(args.model, device="auto")
    model.policy.eval()
    print(f"[info] loaded {args.model}")

    # Dump policy architecture
    (out_dir / "policy.txt").write_text(policy_summary(model))
    print(f"[save] {out_dir/'policy.txt'}")

    # Also dump lightweight param stats
    wstats = []
    for name, p in model.policy.named_parameters():
        if not p.requires_grad:
            continue
        x = p.detach().cpu().float().view(-1)
        wstats.append({
            "name": name,
            "shape": list(p.shape),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "min": float(x.min()),
            "max": float(x.max()),
            "l2": float(torch.linalg.vector_norm(x).item()),
            "n": int(x.numel()),
        })
    (out_dir / "weights_stats.json").write_text(json.dumps(wstats, indent=2))
    print(f"[save] {out_dir/'weights_stats.json'}")

    # Per-step CSV debug
    csv_path = out_dir / "steps.csv"
    with csv_path.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "episode", "step", "reward", "terminated", "truncated",
            "value", "entropy", "action", "log_prob", "max_prob",
            "probs_json", "logits_json"
        ])

        # Aggregate probs across all steps to compute mean later
        all_probs = []
        use_recurrent = args.algo == "recurrent_ppo"

        def _zeros_lstm_state():
            shape = getattr(model.policy, "lstm_hidden_state_shape", None)
            if shape is None:
                raise RuntimeError("Policy missing lstm_hidden_state_shape; expected recurrent weights.")
            h = torch.zeros(shape, dtype=torch.float32, device=model.policy.device)
            c = torch.zeros_like(h)
            return (h, c)

        def _actor_distribution(obs_tensor, actor_state, episode_flags):
            """Run actor LSTM (if any) and return distribution + new state."""
            features = model.policy.extract_features(obs_tensor, model.policy.pi_features_extractor)
            if actor_state is not None:
                latent, actor_state = model.policy._process_sequence(
                    features, actor_state, episode_flags, model.policy.lstm_actor
                )
            else:
                latent, actor_state = features, None
            latent = model.policy.mlp_extractor.forward_actor(latent)
            dist = model.policy._get_action_dist_from_latent(latent)
            if actor_state is not None:
                actor_state = tuple(s.detach() for s in actor_state)
            return dist, actor_state, latent

        def _critic_value(obs_tensor, critic_state, episode_flags, latent_pi=None):
            """Run critic pathway (LSTM if present) and return value + updated state."""
            if critic_state is not None and getattr(model.policy, "lstm_critic", None) is not None:
                vf_features = model.policy.extract_features(obs_tensor, model.policy.vf_features_extractor)
                latent_vf, critic_state = model.policy._process_sequence(
                    vf_features, critic_state, episode_flags, model.policy.lstm_critic
                )
                critic_state = tuple(s.detach() for s in critic_state)
            elif getattr(model.policy, "shared_lstm", False):
                if latent_pi is None:
                    raise RuntimeError("shared_lstm=True but latent_pi was not provided.")
                latent_vf = latent_pi.detach()
            else:
                vf_features = model.policy.extract_features(obs_tensor, model.policy.vf_features_extractor)
                latent_vf = model.policy.critic(vf_features)
            latent_vf = model.policy.mlp_extractor.forward_critic(latent_vf)
            value = model.policy.value_net(latent_vf)
            return value, critic_state

        for ep in range(args.episodes):
            level = args.start_level + ep
            env = make_env(start_level=level, num_levels=args.num_levels)
            print(f"[info] episode {ep+1}/{args.episodes} level={level}")

            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            actor_state = _zeros_lstm_state() if use_recurrent else None
            critic_state = _zeros_lstm_state() if use_recurrent and getattr(model.policy, "lstm_critic", None) is not None else None
            episode_start = np.array([True], dtype=bool) if use_recurrent else None

            for t in range(args.max_steps):
                with torch.no_grad():
                    # Convert obs to torch (SB3 helper adds batch dim & device)
                    obs_t, _ = model.policy.obs_to_tensor(obs)
                    if episode_start is not None:
                        ep_tensor = torch.as_tensor(episode_start.astype(np.float32), device=obs_t.device)
                        dist, actor_state, latent_pi = _actor_distribution(obs_t, actor_state, ep_tensor)
                        value_t, critic_state = _critic_value(obs_t, critic_state, ep_tensor, latent_pi=latent_pi)
                    else:
                        dist = model.policy.get_distribution(obs_t)
                        value_t = model.policy.predict_values(obs_t)

                    torch_dist = getattr(dist, "distribution", None)
                    if torch_dist is None:
                        raise RuntimeError("Could not access torch distribution from policy.")
                    probs_tensor = getattr(torch_dist, "probs", None)
                    logits_tensor = getattr(torch_dist, "logits", None)
                    if probs_tensor is None and logits_tensor is not None:
                        probs_tensor = torch.softmax(logits_tensor, dim=-1)
                    if probs_tensor is None:
                        raise RuntimeError("Policy distribution did not expose probs/logits.")

                    if args.deterministic:
                        action_tensor = torch.argmax(probs_tensor, dim=-1)
                    else:
                        action_tensor = torch_dist.sample()
                    if action_tensor.ndim == 0:
                        action_tensor = action_tensor.unsqueeze(0)
                    action_np = action_tensor.detach().cpu().numpy()
                    action_idx = int(action_np[0])
                    log_prob = float(torch_dist.log_prob(action_tensor).detach().cpu().numpy()[0])
                    entropy = float(torch_dist.entropy().detach().cpu().numpy()[0])
                    value = float(value_t.detach().cpu().numpy()[0])
                    probs = probs_tensor.detach().cpu().numpy()[0]
                    logits = logits_tensor.detach().cpu().numpy()[0] if logits_tensor is not None else np.log(probs + 1e-12)
                    all_probs.append(probs.tolist())
                # mark that episode has started once we used the initial hidden state
                if episode_start is not None:
                    episode_start[:] = False

                # Step the env
                step_out = env.step(action_idx)
                # GymV21 returns 4 items: obs, reward, done, info
                if len(step_out) == 4:
                    obs, reward, done, info = step_out
                    terminated, truncated = bool(done), False
                else:
                    obs, reward, terminated, truncated, info = step_out

                writer.writerow([
                    ep, t, float(reward), int(terminated), int(truncated),
                    value, entropy, action_idx, log_prob, float(np.max(probs)),
                    json.dumps([float(x) for x in probs]),
                    json.dumps([float(x) for x in logits]) if logits is not None else json.dumps([])
                ])

                if terminated or truncated:
                    if episode_start is not None:
                        episode_start[:] = True
                    break

            env.close()

    print(f"[save] {csv_path}")

    # Optional: save a simple PNG bar chart of mean action probabilities
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            P = np.array(all_probs)
            mean_probs = P.mean(axis=0) if P.size else np.array([])
            plt.figure(figsize=(8, 4))
            plt.bar(np.arange(len(mean_probs)), mean_probs)
            plt.xlabel("Action")
            plt.ylabel("Mean probability")
            plt.title("Policy mean action probabilities")
            png_path = out_dir / "mean_action_probs.png"
            plt.tight_layout(); plt.savefig(png_path, dpi=150)
            print(f"[save] {png_path}")
        except Exception as e:
            print(f"[warn] plotting failed: {e}")

if __name__ == "__main__":
    main()
