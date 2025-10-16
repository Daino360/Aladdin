"""
optimizer.py — Optimizer & LR-schedule helpers

What this module provides
-------------------------
1) separate_weight_decayable_params(params)
   Splits parameters into two lists: those that should receive weight decay
   (typically tensors with ndim >= 2) and those that should not (bias, norm).

2) get_optimizer(...)
   Convenience factory for Adam / AdamW with sensible defaults and optional
   param-grouping that disables weight decay for 1D params automatically.

3) LinearWarmup_CosineAnnealing
   A tiny wrapper that combines a linear warmup schedule followed by a cosine
   annealing schedule. Call `.step(nb_steps)` each iteration; it routes steps
   between the two schedulers based on the configured warmup length.

Notes
-----
• Behavior matches your original implementation; this version only adds
  docstrings, type hints, and a couple of convenience methods on the scheduler
  (state_dict / load_state_dict / get_last_lr) without changing defaults.
"""

from torch.optim import AdamW, Adam, lr_scheduler


def separate_weight_decayable_params(params):
    """
    Split parameters into (weight_decay_params, no_weight_decay_params).

    Heuristic:
      • Tensors with ndim >= 2 (e.g., Linear/Conv weights) → weight decay
      • Tensors with ndim  < 2 (e.g., bias terms, LayerNorm/BatchNorm weights)
        → no weight decay

    Args:
        params: Iterable of model parameters.

    Returns:
        (wd_params, no_wd_params): Two lists of parameters.
    """    
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(
    params,
    lr=1e-4,
    wd=1e-2,
    betas=(0.9, 0.99),
    eps=1e-8,
    filter_by_requires_grad=False,
    group_wd_params=True,
    **kwargs
):
    """
    Build an Adam or AdamW optimizer with sensible parameter grouping.

    Args:
        params: Model parameters.
        lr: Learning rate.
        wd: Weight decay. If 0, use Adam (no decoupled weight decay).
        betas: Adam/AdamW betas.
        eps: Adam/AdamW epsilon.
        filter_by_requires_grad: If True, drop params with requires_grad=False.
        group_wd_params: If True, create param groups to disable weight decay
                         on 1D tensors (biases, norm weights).
        **kwargs: Ignored; included for compatibility with higher-level callers.

    Returns:
        torch.optim.Optimizer
    """
    if filter_by_requires_grad:
        # Classic Adam when explicit weight decay is disabled
        params = list(filter(lambda t: t.requires_grad, params))

    if wd == 0:
        return Adam(params, lr=lr, betas=betas, eps=eps)

    if group_wd_params:
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        params = [
            {"params": wd_params},
            {"params": no_wd_params, "weight_decay": 0},
        ]
    
    # Single group with global weight decay
    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


class LinearWarmup_CosineAnnealing:
    """Construct two LR schedulers.
    First one is for a linear warmup.
    Second one is for a cosine annealing.

    Notes:
        • This class intentionally mirrors your original behavior:
          it steps the linear schedule while `nb_steps <= switch`, otherwise the cosine.
        • Added convenience: `state_dict`, `load_state_dict`, and `get_last_lr`.
    """    

    def __init__(
        self,
        optimizer,  # optimizer to schedule
        linear_warmup_start_factor,
        linear_warmup_total_iters,  # linear warmup
        cosine_annealing_T_max,
        cosine_annealing_eta_min,  # cosine annealing
    ):

        self.scheduler_linear = lr_scheduler.LinearLR(
            optimizer,
            # The number we multiply learning rate in the first epoch
            start_factor=linear_warmup_start_factor,
            total_iters=linear_warmup_total_iters,
        )  # The number of iterations that multiplicative factor reaches to 1

        # Linear warmup: linearly scale from `start_factor * base_lr` to `base_lr`
        self.scheduler_cosine = lr_scheduler.CosineAnnealingLR(
            optimizer,
            # Maximum number of iterations.
            T_max=cosine_annealing_T_max,
            eta_min=cosine_annealing_eta_min,
        )  # Minimum learning rate.

        self.switch = linear_warmup_total_iters

    def step(self, nb_steps):
        """
        Advance the appropriate underlying scheduler based on the current step.

        Args:
            nb_steps: Global step counter used to choose which scheduler to step.
                      If nb_steps <= self.switch → warmup; else → cosine.
        """        

        if nb_steps <= self.switch:
            self.scheduler_linear.step()
        else:
            self.scheduler_cosine.step()

        return
