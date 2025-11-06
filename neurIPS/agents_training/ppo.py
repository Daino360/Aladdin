#!/usr/bin/env python3
"""
ppo.py
======

Pure PyTorch PPO (no external RL libs). Minimal, stable defaults. Works with any
Gym-like environment that returns HWC float observations in [0,1] and discrete actions.

Exposes:
- PPOConfig: hyperparameters
- ActorCritic: CNN policy/value for (C,H,W) obs
- RolloutBuffer: fixed-size storage
- ppo_train(...): run the PPO loop on a vectorized env (SimpleVecEnv-like)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# --------------------------- Policy/value ---------------------------

class ActorCritic(nn.Module):
    """Small CNN for 64×64-ish inputs. Adjust strides/filters if resolution changes."""
    def __init__(self, obs_shape: Tuple[int, int, int], action_dim: int):
        super().__init__()
        C, H, W = obs_shape
        self.body = nn.Sequential(
            nn.Conv2d(C, 32, 8, stride=4, padding=2), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(True),
        )
        with torch.no_grad():
            flat = self.body(torch.zeros(1, C, H, W)).view(1, -1).shape[1]
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, 512), nn.ReLU(True))
        self.pi = nn.Linear(512, action_dim)
        self.v  = nn.Linear(512, 1)

    def forward(self, x):
        z = self.head(self.body(x))
        return self.pi(z), self.v(z).squeeze(-1)

    def act(self, obs_t):
        logits, value = self.forward(obs_t)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), value, dist.entropy()

    def evaluate_actions(self, obs_t, actions):
        logits, v = self.forward(obs_t)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), v

# --------------------------- PPO core ---------------------------

@dataclass
class PPOConfig:
    total_timesteps: int = 1_000_000
    nsteps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_vloss: bool = True

class RolloutBuffer:
    """(T,N)-shaped storage for PPO."""
    def __init__(self, nsteps, nenvs, obs_shape, device):
        C,H,W = obs_shape
        self.obs       = torch.zeros(nsteps, nenvs, C, H, W, device=device)
        self.actions   = torch.zeros(nsteps, nenvs, dtype=torch.long, device=device)
        self.logprobs  = torch.zeros(nsteps, nenvs, device=device)
        self.rewards   = torch.zeros(nsteps, nenvs, device=device)
        self.dones     = torch.zeros(nsteps, nenvs, device=device)
        self.values    = torch.zeros(nsteps, nenvs, device=device)
        self.advantages= torch.zeros(nsteps, nenvs, device=device)
        self.returns   = torch.zeros(nsteps, nenvs, device=device)

def _flatten_time_env(x: torch.Tensor):
    return x.reshape(-1, *x.shape[2:])

def _compute_gae(buf: RolloutBuffer, last_value: torch.Tensor, gamma: float, lmbda: float):
    T, N = buf.rewards.shape
    gae = torch.zeros(N, device=buf.rewards.device)
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - buf.dones[t]
        next_value = last_value if t == T - 1 else buf.values[t + 1]
        delta = buf.rewards[t] + gamma * next_value * next_nonterminal - buf.values[t]
        gae = delta + gamma * lmbda * next_nonterminal * gae
        buf.advantages[t] = gae
    buf.returns = buf.advantages + buf.values

from tqdm import trange

def ppo_train(
    venv,                    # vectorized env with reset() and step()
    device: torch.device,
    obs_shape: Tuple[int,int,int],
    action_dim: int,
    cfg: PPOConfig,
    seed: int = 0,
    save_fn=None,            # callable(step, state_dict)
):
    """Run PPO on a vectorized env. venv returns HWC floats in [0,1]."""

    # -------- minor perf knobs --------
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_float32_matmul_precision("high")  # tiny win on Ampere+
    policy = ActorCritic(obs_shape, action_dim).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5)

    buf = RolloutBuffer(cfg.nsteps, venv.num_envs, obs_shape, device)
    obs = venv.reset()  # (N, H, W, C)

    global_step = 0
    num_updates = math.ceil(cfg.total_timesteps / (cfg.nsteps * venv.num_envs))

    outer = trange(num_updates, desc="PPO updates", dynamic_ncols=True)
    for upd in outer:
        policy.train()

        # -------- rollout collection --------
        # Use a lightweight inner progress bar (leave=False to keep the outer clean)
        # If you prefer only one bar, remove the inner bar and keep 'outer.set_postfix'.
        for t in trange(cfg.nsteps, leave=False, desc="collect", dynamic_ncols=True):
            # non_blocking transfer helps a bit if obs is pinned (it isn't here, but harmless)
            obs_t = torch.from_numpy(obs).permute(0,3,1,2).to(device, non_blocking=True)
            with torch.inference_mode():         # slightly faster than no_grad for fwd-only
                a, logp, v, _ = policy.act(obs_t)

            next_obs, rew, done, _ = venv.step(a.cpu().numpy())

            buf.obs[t].copy_(obs_t)
            buf.actions[t].copy_(a)
            buf.logprobs[t].copy_(logp)
            buf.values[t].copy_(v)
            buf.rewards[t].copy_(torch.from_numpy(rew).to(device))
            buf.dones[t].copy_(torch.from_numpy(done.astype(np.float32)).to(device))

            obs = next_obs
            global_step += venv.num_envs

        # -------- advantage / return bootstrap --------
        with torch.inference_mode():
            final_obs_t = torch.from_numpy(obs).permute(0,3,1,2).to(device, non_blocking=True)
            _, last_v = policy.forward(final_obs_t)
        _compute_gae(buf, last_v, cfg.gamma, cfg.gae_lambda)

        # -------- flatten buffers --------
        b_obs        = _flatten_time_env(buf.obs)
        b_actions    = _flatten_time_env(buf.actions)
        b_logprobs   = _flatten_time_env(buf.logprobs)
        b_advantages = _flatten_time_env(buf.advantages)
        b_returns    = _flatten_time_env(buf.returns)
        b_values     = _flatten_time_env(buf.values)

        # normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std(unbiased=False) + 1e-8)

        batch_size = b_obs.shape[0]
        minibatch  = max(1, batch_size // cfg.num_minibatches)

        approx_kl, clipfrac = 0.0, 0.0

        # -------- policy/value optimization --------
        for _ in trange(cfg.update_epochs, leave=False, desc="update", dynamic_ncols=True):
            idx = torch.randperm(batch_size, device=device)
            for s in range(0, batch_size, minibatch):
                mb = idx[s:s+minibatch]
                mb_obs, mb_act = b_obs[mb], b_actions[mb]
                mb_old_logp, mb_adv = b_logprobs[mb], b_advantages[mb]
                mb_ret, mb_val_old  = b_returns[mb], b_values[mb]

                new_logp, ent, v_new = policy.evaluate_actions(mb_obs, mb_act)
                ratio = (new_logp - mb_old_logp).exp()

                # policy loss (clipped)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # value loss (clipped)
                if cfg.clip_vloss:
                    v_clipped = mb_val_old + torch.clamp(v_new - mb_val_old, -cfg.clip_coef, +cfg.clip_coef)
                    v_loss = 0.5 * torch.max((v_new - mb_ret)**2, (v_clipped - mb_ret)**2).mean()
                else:
                    v_loss = 0.5 * (v_new - mb_ret).pow(2).mean()

                loss = pg_loss - cfg.ent_coef * ent.mean() + cfg.vf_coef * v_loss

                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optim.step()

                with torch.inference_mode():
                    approx_kl += (mb_old_logp - new_logp).mean().item()
                    clipfrac  += (torch.abs(ratio - 1.0) > cfg.clip_coef).float().mean().item()

        denom = max(1, cfg.update_epochs * math.ceil(batch_size / minibatch))
        outer.set_postfix(
            step=global_step,
            kl=f"{approx_kl/denom:.6f}",
            clipfrac=f"{clipfrac/denom:.3f}"
        )

        if save_fn is not None:
            save_fn(global_step, {"policy": policy.state_dict()})

    return policy

