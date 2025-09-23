"""
AutoExploreAgent: thin inference wrapper around an ActorCritic policy (world model kept but unused here).

- Purpose: given an observation tensor, pick a discrete action.
- load(path, device, ...): loads ONLY the actor_critic weights from the checkpoint
  via extract_state_dict('actor_critic'); tokenizer/world_model flags are currently ignored.
- act(obs, should_sample=True, temperature=1.0):
    * Resizes obs to 64×64 (bilinear, antialiased), expecting shape B×C×H×W.
    * Forwards through actor_critic and takes the last-step logits ([:, -1]).
    * Applies temperature scaling; samples (Categorical) if should_sample else argmax.
    * Returns a LongTensor of action indices with shape (B,).
- device property reflects actor_critic.conv1.weight.device.

Notes:
- Caller must ensure obs is on the same device and correctly normalized.
- The stored world_model is not consulted during act().
"""

from pathlib import Path

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

from models.genie_redux import GenieReduxGuided

from .models.actor_critic import ActorCritic
from .utils import extract_state_dict



class AutoExploreAgent(nn.Module):
    def __init__(self, world_model: GenieReduxGuided, actor_critic: ActorCritic):
        super().__init__()
        self.world_model = world_model
        self.actor_critic = actor_critic

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def load(self, path_to_checkpoint: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True, load_actor_critic: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        # Tokenizer state is no longer used/loaded
        if load_actor_critic:
            self.actor_critic.load_state_dict(extract_state_dict(agent_state_dict, 'actor_critic'))

    @torch.no_grad()
    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        input_ac = obs
        input_ac = nn.functional.interpolate(input_ac, size=(64, 64), mode='bilinear', align_corners=False, antialias=True)
        logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        return act_token
