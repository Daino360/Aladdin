"""
actor_critic.py: recurrent policy+value network for discrete control, with
optional IMPALA encoder and utilities to (a) roll out on given observations
(“imagine_none”) and (b) roll out in a real env while optionally probing a
world model (“imagine_gt”). Includes an A2C/GAE-style loss using λ-returns.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Union
import sys

from einops import rearrange
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data.data import TransformsGenerator
from data_generation.generator.connector_retro_act import make_retro
from auto_explore.src.envs.multi_process_env import MultiProcessEnv
from auto_explore.src.utils import dotdict
from models.genie_redux import GenieReduxGuided

from ..dataset import Batch
from ..utils import compute_lambda_returns, LossWithIntermediateLosses
from .impala import ImpalaCNN

from tools.logger import getLogger
log = getLogger(__name__)


@dataclass
class ActorCriticOutput:
    """Outputs of a single policy forward pass."""
    logits_actions: torch.FloatTensor  # (B, 1, A) unnormalized logits over discrete actions
    means_values: torch.FloatTensor    # (B, 1, 1) scalar state-value estimate


@dataclass
class ImagineOutput:
    """Container for imagined/rolled-out trajectories."""
    observations: torch.ByteTensor     # (B, T, C, H, W) uint8 frames
    actions: torch.LongTensor          # (B, T) sampled action ids
    logits_actions: torch.FloatTensor  # (B, T, A) policy logits per step
    values: torch.FloatTensor          # (B, T) value predictions
    rewards: torch.FloatTensor         # (B, T) rewards used for learning
    ends: torch.BoolTensor             # (B, T) done flags (True at terminal steps)


class ActorCritic(nn.Module):
    """
    Recurrent actor-critic with either a small ConvNet or IMPALA encoder + LSTM.

    - Encoder: 4-layer CNN (to 1024) or ImpalaCNN (to 512).
    - Core: LSTMCell(encoder_out -> 512).
    - Heads: policy logits (→ act_vocab_size) and scalar value.
    """    
    def __init__(self, act_vocab_size, use_original_obs: bool = False, use_impala:bool = False) -> None:
        """Configure encoder type and build recurrent core + policy/value heads."""
        super().__init__()
        self.use_original_obs = use_original_obs
        
        self.use_impala = use_impala
        

        if self.use_impala:
            self.impala = ImpalaCNN([64,64])
            output_size = 512
        else:
            self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
            self.maxp1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
            self.maxp2 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
            self.maxp3 = nn.MaxPool2d(2, 2)
            self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.maxp4 = nn.MaxPool2d(2, 2)
            output_size = 1024

        self.lstm_dim = 512
        self.lstm = nn.LSTMCell(output_size, self.lstm_dim)
        self.hx, self.cx = None, None

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, act_vocab_size)
        
    def __repr__(self) -> str:
        """Nice short name for logging/metrics prefixes."""
        return "actor_critic"

    def clear(self) -> None:
        """Forget LSTM hidden-state (e.g., at end of rollout/imagination)."""
        self.hx, self.cx = None, None

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None, mask_padding: Optional[torch.Tensor] = None) -> None:
        """
        Initialize LSTM state for a batch of size n; optionally “burn in”
        with a short sequence to warm up the recurrent state where mask_padding=True.
        """
        device = self.conv1.weight.device

        self.hx = torch.zeros(n, self.lstm_dim, device=device, dtype=torch.bfloat16)
        self.cx = torch.zeros(n, self.lstm_dim, device=device, dtype=torch.bfloat16)
        
        if burnin_observations is not None:
            #einops pack the burnin_observations to 64x64
            b= burnin_observations.size(0)
            burnin_observations = rearrange(burnin_observations, 'b t c h w -> (b t) c h w')
            burnin_observations = F.interpolate(burnin_observations, size=(64, 64), mode='bilinear', align_corners=False, antialias=True)
            burnin_observations = rearrange(burnin_observations, '(b t) c h w -> b t c h w', b=b)
            assert burnin_observations.ndim == 5 and burnin_observations.size(0) == n and mask_padding is not None and burnin_observations.shape[:2] == mask_padding.shape
            for i in range(burnin_observations.size(1)):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self(burnin_observations[:, i], mask_padding[:, i])

    def prune(self, mask: np.ndarray) -> None:
        """Keep only rows where mask is True in the LSTM states (e.g., after env pruning)."""
        self.hx = self.hx[mask]
        self.cx = self.cx[mask]

    def forward(self, inputs, mask_padding: Optional[torch.BoolTensor] = None) -> ActorCriticOutput:
        """
        One recurrent policy step.

        Expects inputs in [0,1], shape (B, 3, 64, 64) (resized here if needed).
        If mask_padding is provided, only those rows update the recurrent state.
        Returns (logits over actions, value).
        """        
        inputs = F.interpolate(inputs, size=(64, 64), mode='bilinear', align_corners=False, antialias=True)
        assert inputs.ndim == 4 and inputs.shape[1:] == (3, 64, 64)
        epsilon = 1e-6
        assert 0.0 - epsilon <= inputs.min().item() <= 1.0 + epsilon and 0.0 - epsilon <= inputs.max().item() <= 1.0 + epsilon
        assert mask_padding is None or (mask_padding.ndim == 1 and mask_padding.size(0) == inputs.size(0) and mask_padding.any())
        x = inputs[mask_padding] if mask_padding is not None else inputs

        x = x.mul(2).sub(1)
        if self.use_impala:
            x = self.impala(x)
        else:
            x = F.relu(self.maxp1(self.conv1(x)))
            x = F.relu(self.maxp2(self.conv2(x)))
            x = F.relu(self.maxp3(self.conv3(x)))
            x = F.relu(self.maxp4(self.conv4(x)))
            x = torch.flatten(x, start_dim=1)

        if mask_padding is None:
            self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        else:
            self.hx[mask_padding], self.cx[mask_padding] = self.lstm(x, (self.hx[mask_padding], self.cx[mask_padding]))

        logits_actions = rearrange(self.actor_linear(self.hx), 'b a -> b 1 a')
        means_values = rearrange(self.critic_linear(self.hx), 'b 1 -> b 1 1')

        return ActorCriticOutput(logits_actions, means_values)

    def compute_loss(self, batch: Batch, world_model: GenieReduxGuided, imagine_horizon: int, gamma: float, lambda_: float, entropy_weight: float, **kwargs: Any) -> LossWithIntermediateLosses:
        """
        A2C-style loss on a rollout “imagined” directly from batch observations.

        - Builds a trajectory with imagine_none (actions sampled from the policy).
        - Computes λ-returns (bootstrapped with predicted values).
        - Policy loss: advantage-weighted log-prob (negative).
        - Entropy regularization (encourages exploration).
        - Value loss: MSE(values, λ-returns).
        """        
        assert not self.use_original_obs
        
        outputs = self.imagine_none(batch, horizon=imagine_horizon)
        

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.values,
                ends=outputs.ends,
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]

        values = outputs.values[:, :-1]

        d = Categorical(logits=outputs.logits_actions[:, :-1])
        log_probs = d.log_prob(outputs.actions[:, :-1])
        loss_actions = -1 * (log_probs * (lambda_returns - values.detach())).mean()
        loss_entropy = - entropy_weight * d.entropy().mean()
        loss_values = F.mse_loss(values, lambda_returns)

        return LossWithIntermediateLosses(loss_actions=loss_actions, loss_values=loss_values, loss_entropy=loss_entropy)
    

    def imagine_none(self, batch: Batch, horizon: int, show_pbar: bool = False) -> ImagineOutput:
        """
        “Roll out” on the given ground-truth observation sequence (no env/world model).
        At each time step:
          - feed the current frame to the policy,
          - sample an action,
          - record logits and value (rewards/ends are taken from the batch).
        Note: the provided `horizon` is ignored; uses the batch length instead.
        """        
        assert not self.use_original_obs
        initial_observations = batch['observations']
        rewards = batch['rewards']
        dones = batch['ends']
        horizon = initial_observations.size(1)
        
        mask_padding = batch['mask_padding']
        assert initial_observations.ndim == 5
        assert mask_padding[:, -1].all()
        device = initial_observations.device

        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []

        
        self.reset(n=initial_observations.size(0), burnin_observations=None, mask_padding=None)

        #create a list from dim 1 of initial_observations
        
        for k in tqdm(range(horizon), disable=not show_pbar, desc='Imagination', file=sys.stdout):
            obs = initial_observations[:, k]
            all_observations.append(obs)

            outputs_ac = self(obs)
            action_token = Categorical(logits=outputs_ac.logits_actions).sample()
            # obs, reward, done, _ = wm_env.step(action_token, should_predict_next_obs=(k < horizon - 1))

            all_actions.append(action_token)
            all_logits_actions.append(outputs_ac.logits_actions)
            all_values.append(outputs_ac.means_values)
            # all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            # all_ends.append(torch.tensor(done).reshape(-1, 1))

        self.clear()

        return ImagineOutput(
            observations=torch.stack(all_observations, dim=1).mul(255).byte(),      # (B, T, C, H, W) in [0, 255]
            actions=torch.cat(all_actions, dim=1),                                  # (B, T)
            logits_actions=torch.cat(all_logits_actions, dim=1),                    # (B, T, #actions)
            values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),         # (B, T)
            rewards=rewards.to(device),                       # (B, T) torch.cat(all_rewards, dim=1)
            ends=dones.to(device) #torch.cat(all_ends, dim=1).to(device),                             # (B, T)
        )


    def imagine_gt(self, batch: Batch, world_model: GenieReduxGuided, horizon: int, show_pbar: bool = False) -> ImagineOutput:
        """
        Roll out actions in a *real* parallel env (here hardcoded to SMB) for `horizon`
        steps; preprocess observations; sample actions from the policy each step; and
        optionally probe the world model to derive an intrinsic reward signal.

        Returns an ImagineOutput with captured (obs, actions, logits, values) and
        rewards/ends measured during the env rollout (rewards here are set from
        a world-model entropy proxy if that block is reached).
        """
        assert not self.use_original_obs
        initial_observations = batch['observations']
        mask_padding = batch['mask_padding']
        assert initial_observations.ndim == 5 #and initial_observations.shape[2:] == (3, 64, 64)
        assert mask_padding[:, -1].all()
        device = initial_observations.device


        def create_env(cfg_env, num_envs):
            
            game = cfg_env.id
            max_episode_steps = cfg_env.max_episode_steps
            frame_skip = cfg_env.frame_skip
            valid_action_combos = ["UP", "DOWN", "RIGHT", "LEFT", "ACTION_JUMP"]

            env_fn = partial(
                make_retro,
                game=game,
                render_mode="rgb_array",
                valid_action_combos=valid_action_combos,
                skip_frames=frame_skip,
                max_episode_steps=max_episode_steps
            )
            return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) #if num_envs > 1 else SingleProcessEnv(env_fn)

        cfg_env = dotdict(
            id="SuperMarioBros-Nes",
            max_episode_steps=64,
            frame_skip=4
        )

        env =create_env(cfg_env, initial_observations.size(0))
        

        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []

        
        self.reset(n=initial_observations.size(0), burnin_observations=None, mask_padding=mask_padding[:, :-1])

        
        obs = env.reset()
        
        transforms = TransformsGenerator.get_final_transforms(world_model.image_size, None)
        transform = transforms["train"]

        def process_obs(obs, dtype, transform):
            """Apply transform per-frame and convert to device/dtype tensor."""
            new_obs = []
            for ob in obs:
                ob = transform(ob)
                new_obs.append(ob)
            obs = torch.tensor(np.array(new_obs))
            obs = obs.to(device)
            obs = obs.to(dtype)
            return obs
        obs = process_obs(obs, initial_observations.dtype, transform)

        
        n_input_frames = 2
        n_preds = 10
        
        was_training = world_model.training
        world_model.eval()
        for k in tqdm(range(horizon), disable=not show_pbar, desc='Imagination', file=sys.stdout):

            all_observations.append(obs)

            outputs_ac = self(obs)
            action = Categorical(logits=outputs_ac.logits_actions).sample()
            #turn action token to a one_hot on the last dim
            action_token = F.one_hot(action, num_classes=env.num_actions)
            action_token = action_token.squeeze(1) # B x 1 x num_actions -> B x num_actions
            #convert action token to numpy
            obs, reward, done, _ = env.step(action_token.cpu().numpy())
            obs = process_obs(obs, initial_observations.dtype, transform)

            all_actions.append(action)
            all_logits_actions.append(outputs_ac.logits_actions)
            all_values.append(outputs_ac.means_values)
            # all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))
            
            # Optionally use the world model to compute an entropy-based pseudo-reward
            if k+1 % (n_preds + n_input_frames) == 0:
                #convert all_observations to a single torch array with torch stack
                wm_observations = torch.stack(all_observations[-(n_input_frames + n_preds):-n_input_frames], dim=1)
                #do the same for actions
                wm_actions = torch.cat(all_actions[-n_input_frames:], dim=1)
                wm_observations = rearrange(wm_observations, 'b t c h w -> b c t h w')
                with torch.no_grad():
                    logits = world_model.sample(
                        prime_frames=wm_observations,
                        actions=wm_actions,
                        num_frames=n_preds,
                        inference_steps=25,
                        mask_schedule="exp",
                        sample_temperature=1.0,
                        return_logits=True,)
                #compute the entropy on the last dimension of logits
                logits = rearrange(logits, 'b (t d) c -> b t d c', t=10)
                
                entropy = -torch.sum(logits * torch.log(logits), dim=-1)
                #find the mean over the last dim
                entropy = torch.mean(entropy, dim=-1)
                #add an empty dimension to the end
                entropy = entropy.unsqueeze(-1)
                all_rewards.append(entropy)
        world_model.train(was_training)
        self.clear()

        return ImagineOutput(
            observations=torch.stack(all_observations, dim=1).mul(255).byte(),      # (B, T, C, H, W) in [0, 255]
            actions=torch.cat(all_actions, dim=1),                                  # (B, T)
            logits_actions=torch.cat(all_logits_actions, dim=1),                    # (B, T, #actions)
            values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),         # (B, T)
            rewards=torch.cat(all_rewards, dim=1).to(device),                       # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),                             # (B, T)
        )
