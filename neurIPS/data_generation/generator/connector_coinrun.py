from generator.connector_base import BaseConnector
# Use the module so we can patch its ppo_init that ppo_agent_generator calls
from coinrun import random_agent as ra
# Keep names for readability, but we will use ra.ppo_agent_generator internally
from coinrun.random_agent import (
    ppo_agent_generator,      # not strictly needed once we use ra., but fine to keep
    random_agent_generator,
    ppo_init,                 # original ref, not used after patch
)
import cv2

# --- NEW: imports for weights loading ---
from typing import Any, Dict
import torch


class CoinRunConnector(BaseConnector):
    def __init__(self, config=None):
        if config is None:
            config = {
                "name": "coinrun",
                "version": "0.1.0",
                "is_high_res": False,
                "is_high_difficulty": True,
                "should_paint_velocity": False,
                "agent_type": "ppo",
            }

        self.config = config
        self.name = config["name"]
        self.version = config["version"]
        self.is_high_res = config["is_high_res"]
        self.is_high_difficulty = config["is_high_difficulty"]
        self.should_paint_velocity = config["should_paint_velocity"]
        self.image_size = config["image_size"]
        self.agent_type = config["agent_type"]

        # --- NEW: optional checkpoint + device ---
        self.policy_pth = config.get("policy_pth", None)  # path to .pth state_dict
        req_device = str(config.get("device", "cpu"))
        want_cuda = ("cuda" in req_device) and torch.cuda.is_available()
        self.device = torch.device(req_device if want_cuda else "cpu")

        # IMPORTANT: use the module's generators so they see our patched ppo_init
        self.agent_generator = (
            ra.ppo_agent_generator if self.agent_type == "ppo" else ra.random_agent_generator
        )

        # If we’re PPO and a checkpoint is provided, patch ppo_init once
        self._ppo_patched = False
        if self.agent_type == "ppo" and self.policy_pth:
            self._patch_ppo_init()

    # -------------------- BaseConnector API --------------------

    def get_name(self):
        return "coinrun"

    def get_info(self):
        return {
            "action_space": [7],
            "observation_space": [512, 512] if self.is_high_res else [256, 256],
            "config": self.config,
            # optional, useful for provenance:
            "policy_checkpoint": self.policy_pth if self.policy_pth else None,
            "device": str(self.device),
        }

    # -------------------- Data generator --------------------

    def generator(self, instance_id, session_id, n_steps_max):

        for frame_id, (_obs, acts, rews, _dones, _infos, extras) in enumerate(
            self.agent_generator(
                num_envs=1,
                max_steps=n_steps_max,
                is_high_difficulty=self.is_high_difficulty,
                is_high_res=self.is_high_res,
                should_paint_velocity=self.should_paint_velocity,
                seed_ids=[instance_id],
            )
        ):
            frame = _obs[0]
            action = acts[0]
            session_end = frame_id == n_steps_max - 1
            if self.image_size is not None:
                frame = cv2.resize(frame, self.image_size)

            if _dones[0]:
                break

            yield {
                "src_frame_id": frame_id - 1,
                "tgt_frame_id": frame_id,
                "frame": frame,
                "action": int(action),
                "session_end": session_end,
                "extras": extras,
            }
            if rews[0] > 0:
                break

    # -------------------- NEW: ppo_init patching --------------------

    def _patch_ppo_init(self):
        """
        Patch coinrun.random_agent.ppo_init so that after it builds the PPO policy,
        we load self.policy_pth (a torch state_dict) into that policy.
        This way ppo_agent_generator will roll out with your trained weights.
        """
        if self._ppo_patched:
            return

        orig_ppo_init = ra.ppo_init  # keep original

        def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str = "module."):
            if not isinstance(sd, dict):
                return sd
            return { (k[len(prefix):] if isinstance(k, str) and k.startswith(prefix) else k): v
                     for k, v in sd.items() }

        def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
            """
            Accept either a raw state_dict or a dict with common keys.
            """
            if not isinstance(obj, dict):
                # raw state_dict-like (OrderedDict) also behaves like dict, but guard anyway
                if hasattr(obj, "keys"):
                    return obj  # assume it's already a state_dict
                raise RuntimeError("Checkpoint is not a dict or state_dict.")
            # common wrappers
            for k in ("state_dict", "model_state_dict", "policy_state_dict", "policy", "model", "net"):
                if k in obj and isinstance(obj[k], dict):
                    return obj[k]
            # maybe it's already the raw state_dict
            return obj

        def _load_into_policy(policy_module: torch.nn.Module, ckpt_path: str):
            sd = torch.load(ckpt_path, map_location=self.device)
            sd = _extract_state_dict(sd)
            sd = _strip_prefix(sd, "module.")
            policy_module.load_state_dict(sd, strict=False)
            policy_module.to(self.device)
            policy_module.eval()

        def patched_ppo_init(*args, **kwargs):
            """
            Call the original ppo_init, find the policy nn.Module it builds,
            load our state_dict into it, and return value unchanged in shape.
            """
            result = orig_ppo_init(*args, **kwargs)

            # Case A: ppo_init returns an nn.Module (the policy itself)
            if isinstance(result, torch.nn.Module):
                _load_into_policy(result, self.policy_pth)
                return result

            # Case B: returns a tuple; try to find a Module in it
            if isinstance(result, tuple):
                lst = list(result)
                loaded = False
                for i, obj in enumerate(lst):
                    if isinstance(obj, torch.nn.Module):
                        _load_into_policy(obj, self.policy_pth)
                        lst[i] = obj
                        loaded = True
                        break
                    if isinstance(obj, dict) and "policy" in obj and isinstance(obj["policy"], torch.nn.Module):
                        _load_into_policy(obj["policy"], self.policy_pth)
                        loaded = True
                        break
                if not loaded and hasattr(result, "policy") and isinstance(result.policy, torch.nn.Module):
                    _load_into_policy(result.policy, self.policy_pth)
                    loaded = True
                if not loaded:
                    raise RuntimeError("Patched ppo_init could not locate a policy nn.Module in its return.")
                return tuple(lst)

            # Case C: returns a dict with a policy
            if isinstance(result, dict) and "policy" in result and isinstance(result["policy"], torch.nn.Module):
                _load_into_policy(result["policy"], self.policy_pth)
                return result

            # Case D: object with .policy attribute
            if hasattr(result, "policy") and isinstance(result.policy, torch.nn.Module):
                _load_into_policy(result.policy, self.policy_pth)
                return result

            raise RuntimeError("Unsupported ppo_init() return type for loading weights.")

        # Monkey-patch the module function so ppo_agent_generator uses it
        ra.ppo_init = patched_ppo_init
        self._ppo_patched = True
