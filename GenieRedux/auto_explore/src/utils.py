"""
utils.py: grab bag of small utilities used across the project.

What it covers
--------------
- Simple data structures: dotdict, FileStructure (paths layout helper).
- Config & I/O helpers: dump_hydra(), EpisodeDirManager (save best/recent episodes).
- Training utilities: configure_optimizer() (AdamW param groups), init_weights(),
  extract_state_dict(), set_seed(), compute_lambda_returns() (TD(λ) targets),
  LossWithIntermediateLosses (tracks total + per-term losses).
- Misc: remove_dir() safety delete, RandomHeuristic (epsilon-random actions),
  make_video() to write RGB frames to an .mp4.
"""

from collections import OrderedDict
import os
import cv2
from pathlib import Path
import random
import shutil

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn

from .episode import Episode
from tools.logger import getLogger
log = getLogger(__name__)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class FileStructure:
    """
    Centralizes all project-relative paths (checkpoints, media, config, data).

    Parameters
    ----------
    root_dpath : str
        Root directory under which subfolders are organized.
    **kwargs
        Optional keys like 'data_test_dpath' to override defaults.

    Attributes (selected)
    ---------------------
    hydra_dpath, hydra_config_fpath
    checkpoints_dpath, last_checkpoint_fpath
    media_dpath, episodes_dpath, reconstructions_dpath, imagination_dpath
    config_dpath
    data_train_dpath, data_test_dpath
    """
    def __init__(self, root_dpath=".", **kwargs) -> None:
        root_dpath = os.path.abspath(root_dpath)
        self.hydra_subdpath = ".hydra"
        self.hydra_config_fname = "config.yaml"

        self.hydra_dpath = os.path.join(root_dpath, self.hydra_subdpath)
        self.hydra_config_fpath = os.path.join(self.hydra_dpath, self.hydra_config_fname)

        self.checkpoints_subdpath = "checkpoints"
        self.checkpoints_dpath = os.path.join(root_dpath, self.checkpoints_subdpath)
        self.last_checkpoint_fpath = os.path.join(self.checkpoints_dpath, "last.pt")

        self.media_subdpath = "media"
        self.media_dpath = os.path.join(root_dpath, self.media_subdpath)
        self.episodes_dpath = os.path.join(self.media_dpath, "episodes")
        self.reconstructions_dpath = os.path.join(self.media_dpath, "reconstructions")
        self.episodes_test_dpath = os.path.join(self.episodes_dpath, "test")
        self.imagination_dpath = os.path.join(self.episodes_dpath, "imagination")

        self.config_subdpath = "config"
        self.config_dpath = os.path.join(root_dpath, self.config_subdpath)

        if "data_test_dpath" in kwargs:
            self.data_test_dpath = os.path.realpath(kwargs["data_test_dpath"])
        else:
            log.w("data_test_dpath not provided. Using default value.")
            self.data_test_dpath = os.path.join(root_dpath, "data_test")

        self.data_train_dpath = os.path.join(root_dpath, "data")

    def create(self):
        # Create the minimal required directory structure (currently Hydra folder).
        os.makedirs(self.hydra_dpath, exist_ok=True)


def dump_hydra(cfg: DictConfig, out_fpath: str):
    # Serialize a Hydra/OmegaConf config to YAML on disk.
    with open(out_fpath, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

def configure_optimizer(model, learning_rate, weight_decay, *blacklist_module_names):
    """
    Build an AdamW optimizer with decoupled weight decay only on selected weights.

    Strategy (à la minGPT):
    - Apply weight decay to Linear/Conv1d 'weight' parameters.
    - Do NOT decay biases, LayerNorm, Embedding weights, or any name starting with
      provided blacklist prefixes.

    Parameters
    ----------
    model : nn.Module
    learning_rate : float
    weight_decay : float
    *blacklist_module_names : str
        Module name prefixes to exempt from weight decay.

    Returns
    -------
    torch.optim.Optimizer
        Configured AdamW optimizer with two parameter groups.
    """
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                no_decay.add(fpn)
            elif 'bias' in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def init_weights(module):
    """
    Initialize common module types:
    - Linear/Embedding: N(0, 0.02), bias zeros.
    - LayerNorm: weight ones, bias zeros.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def extract_state_dict(state_dict, module_name):
    """
    Extract a submodule state_dict by stripping the given prefix.

    Example
    -------
    extract_state_dict(sd, 'actor_critic') turns
    'actor_critic.layer.weight' -> 'layer.weight'.
    """    
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def set_seed(seed):
    """
    Seed numpy, Python, and PyTorch RNGs (CPU + CUDA) for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def remove_dir(path, should_ask=False):
    """
    Recursively delete a directory, with optional interactive confirmation.

    Parameters
    ----------
    path : Path
        Directory to remove.
    should_ask : bool
        If True, prompt the user before deletion.
    """    
    assert path.is_dir()
    if (not should_ask) or input(f"Remove directory : {path} ? [Y/n] ").lower() != 'n':
        shutil.rmtree(path)


def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
    """
    Compute TD(λ) target (a.k.a. λ-returns) for bootstrapped value learning.

    Shapes
    ------
    rewards, values, ends : (B, T) or (B, T, 1)
        'ends' is 1 when episode terminates at that step, else 0.
    gamma : float
        Discount factor in [0, 1].
    lambda_ : float
        Trace-decay parameter in [0, 1].

    Returns
    -------
    torch.Tensor
        λ-returns with same shape as values.

    Notes
    -----
    - Uses values[:, -1] as bootstrap at the final step.
    - Masks bootstrapping past terminal steps using 'ends'.
    """    
    assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
    assert rewards.shape == ends.shape == values.shape, f"{rewards.shape}, {values.shape}, {ends.shape}"  # (B, T, 1)
    t = rewards.size(1)
    lambda_returns = torch.empty_like(values)
    lambda_returns[:, -1] = values[:, -1]
    lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

    last = values[:, -1]
    for i in list(range(t - 1))[::-1]:
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


class LossWithIntermediateLosses:
    """
    Convenience wrapper to hold a summed loss and its named components.

    Attributes
    ----------
    loss_total : torch.Tensor
        Sum of all provided loss terms.
    intermediate_losses : Dict[str, float]
        Scalar (item()) snapshot of each term, useful for logging.

    Supports division (`/`) to scale both the total and component terms (e.g., for
    gradient accumulation averaging).
    """
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):   
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


class EpisodeDirManager:
    """
    Manages saving episode files to a directory with a rolling cap.

    - Keeps at most `max_num_episodes` recent episodes (evicts oldest).
    - Also tracks and persists the best-return episode as 'best_episode_*.pt'.
    """
    def __init__(self, episode_dir: Path, max_num_episodes: int) -> None:
        self.episode_dir = episode_dir
        self.episode_dir.mkdir(parents=False, exist_ok=True)
        self.max_num_episodes = max_num_episodes
        self.best_return = float('-inf')

    def save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        """Public save: only writes if max_num_episodes > 0."""
        if self.max_num_episodes is not None and self.max_num_episodes > 0:
            self._save(episode, episode_id, epoch)

    def _save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        """
        Save an episode as 'episode_{id}_epoch_{epoch}.pt'.
        - If the directory is full, remove the oldest episode file.
        - If this episode has the highest return so far, also overwrite the
          single 'best_episode_*.pt'.
        """        
        ep_paths = [p for p in self.episode_dir.iterdir() if p.stem.startswith('episode_')]
        assert len(ep_paths) <= self.max_num_episodes
        if len(ep_paths) == self.max_num_episodes:
            to_remove = min(ep_paths, key=lambda ep_path: int(ep_path.stem.split('_')[1]))
            to_remove.unlink()
        episode.save(self.episode_dir / f'episode_{episode_id}_epoch_{epoch}.pt')

        ep_return = episode.compute_metrics().episode_return
        if ep_return > self.best_return:
            self.best_return = ep_return
            path_best_ep = [p for p in self.episode_dir.iterdir() if p.stem.startswith('best_')]
            assert len(path_best_ep) in (0, 1)
            if len(path_best_ep) == 1:
                path_best_ep[0].unlink()
            episode.save(self.episode_dir / f'best_episode_{episode_id}_epoch_{epoch}.pt')


class RandomHeuristic:
    """
    Simple exploration policy that samples a uniform random action for each env
    in the batch.
    """
    def __init__(self, num_actions):
        """
        Parameters
        ----------
        num_actions : int
            Size of the discrete action space [0, num_actions).
        """
        self.num_actions = num_actions

    def act(self, obs):
        """
        Produce random discrete actions.

        Parameters
        ----------
        obs : torch.Tensor
            4D batch of observations (asserts ndim == 4). Only the batch size is used.

        Returns
        -------
        torch.LongTensor
            Shape (N,), random integers in [0, num_actions).
        """
        assert obs.ndim == 4  # (N, H, W, C)
        n = obs.size(0)
        return torch.randint(low=0, high=self.num_actions, size=(n,))


def make_video(fname, fps, frames):
    """
    Write an HxWx3 RGB uint8 frame sequence to an MP4 file.

    Parameters
    ----------
    fname : Path or str
        Output filename ('.mp4' recommended).
    fps : int or float
        Frames per second.
    frames : np.ndarray
        Array of shape (T, H, W, 3) in RGB order, dtype uint8.

    Notes
    -----
    OpenCV expects BGR, so frames are channel-reversed during writing.
    """
    assert frames.ndim == 4 # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3

    video = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()
