#!/usr/bin/env python3
"""
NPZ → world model inference (single or many episodes) with one-time model load.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import csv
import sys
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Repo utilities
from models import construct_model
from data.data import video_tensor_to_gif, video_tensor_to_pil_images

# SSIM metric
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

# Hydra config loader
from omegaconf import OmegaConf

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from dataclasses import dataclass

# --------------------------- Common keys ---------------------------

FRAME_KEYS = ["frames", "videos", "input_frames", "obs", "images", "x"]
ACT_KEYS = ["actions", "acts", "a", "action"]


# --------------------------- Small utilities ---------------------------

def _to_chw(img_hwc: np.ndarray) -> torch.Tensor:
    """Convert a single frame H×W×C (uint8 or float) → CHW float in [0,1]."""
    if img_hwc.dtype not in (np.float32, np.float64):
        x = torch.from_numpy(img_hwc.astype(np.uint8)).permute(2, 0, 1).float() / 255.0
    else:
        x = torch.from_numpy(img_hwc).permute(2, 0, 1).float()
        if x.max() > 1.0:
            x = x / 255.0
    return x


def _resize_chw(x_chw: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Bilinear resize of a CHW tensor to (H, W)."""
    return F.interpolate(x_chw.unsqueeze(0), size=size_hw, mode="bilinear", align_corners=False).squeeze(0)


def psnr_frame(gt_chw: torch.Tensor, pr_chw: torch.Tensor) -> float:
    """PSNR in dB for two CHW frames in [0,1]."""
    mse = F.mse_loss(gt_chw, pr_chw)
    if mse.item() == 0:
        return 99.0
    return float(20.0 * torch.log10(torch.tensor(1.0, device=gt_chw.device) / torch.sqrt(mse)).item())


def ssim_frame_batched(gt: torch.Tensor, pr: torch.Tensor, metric: SSIM) -> float:
    """
    Compute mean SSIM across time using a single torchmetrics object.
    gt/pr: shape 1 × C × T × H × W
    """
    T = gt.shape[2]
    vals = []
    with torch.no_grad():
        for t in range(T):
            vals.append(float(metric(pr[:, :, t], gt[:, :, t]).item()))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------- Config / Model ---------------------------

def load_cfg(yaml_path: str):
    if OmegaConf is None:
        raise RuntimeError("omegaconf is required. Install with: pip install omegaconf")
    return OmegaConf.load(yaml_path)


def _maybe_extract(sd: dict, keys: List[str]) -> Optional[dict]:
    if not isinstance(sd, dict):
        return None
    for k in keys:
        if k in sd and isinstance(sd[k], dict):
            return sd[k]
    return None


def build_model(cfg, ckpt_path: str, device: torch.device, tokenizer_ckpt: Optional[str] = None):
    """
    Construct a Genie/GenieRedux model and load weights.
    Optionally load a separate tokenizer checkpoint (if your project stores it apart).
    """
    model = construct_model(cfg)

    # main ckpt
    sd = torch.load(ckpt_path, map_location="cpu")
    # Accept raw state_dict, or training blob with "model"/"state_dict"
    cand = _maybe_extract(sd, ["model", "state_dict"])
    if cand is not None:
        sd = cand
    model.load_state_dict(sd, strict=False)

    # optional tokenizer ckpt (robust best-effort)
    if tokenizer_ckpt:
        tsd = torch.load(tokenizer_ckpt, map_location="cpu")
        cand_tok = _maybe_extract(tsd, ["tokenizer", "model", "state_dict"])
        if cand_tok is None:
            cand_tok = tsd
        try:
            model.tokenizer.load_state_dict(cand_tok, strict=False)
            print("[Tokenizer] Loaded separate tokenizer checkpoint.")
        except Exception as e:
            print(f"[Tokenizer][WARN] Could not load tokenizer ckpt strictly: {e}")

    model.eval().to(device)
    return model


# --------------------------- NPZ parsing ---------------------------

def find_frames_actions(npz: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Find and normalize frames/actions from an NPZ dict."""
    notes: List[str] = []

    # --- frames ---
    frames = None
    for k in FRAME_KEYS:
        if k in npz:
            frames = npz[k]
            notes.append(f"frames: '{k}' {frames.shape} {frames.dtype}")
            break
    if frames is None:
        for k, v in npz.items():
            if isinstance(v, np.ndarray) and v.ndim >= 3 and 3 in v.shape:
                frames = v
                notes.append(f"frames fallback: '{k}' {v.shape}")
                break
    if frames is None:
        raise ValueError("Could not find frames array in NPZ.")

    # Normalize frames to T×H×W×C uint8
    f = frames
    if f.ndim == 5 and f.shape[0] in (1, 3):
        f = np.transpose(f, (1, 2, 3, 0))  # C T H W -> T H W C
        notes.append("frames converted CxTxHxW -> TxHxWxC")
    elif f.ndim == 4:
        if f.shape[-1] in (1, 3):
            pass
        elif f.shape[0] in (1, 3):         # C T H W -> T H W C
            f = np.transpose(f, (1, 2, 3, 0))
            notes.append("frames converted CxTxHxW -> TxHxWxC")
        elif f.shape[1] in (1, 3):         # T C H W -> T H W C
            f = np.transpose(f, (0, 2, 3, 1))
            notes.append("frames converted TxCxHxW -> TxHxWxC")
    else:
        raise ValueError(f"Unsupported frames shape: {f.shape}")

    if f.dtype != np.uint8:
        if f.max() <= 1.0:
            f = (np.clip(f, 0, 1) * 255).astype(np.uint8)
            notes.append("frames scaled [0,1]→uint8")
        else:
            f = np.clip(f, 0, 255).astype(np.uint8)
            notes.append("frames clipped→uint8")

    # --- actions ---
    actions = None
    for k in ACT_KEYS:
        if k in npz:
            actions = npz[k]
            notes.append(f"actions: '{k}' {actions.shape} {actions.dtype}")
            break
    if actions is None:
        for k, v in npz.items():
            if isinstance(v, np.ndarray) and v.ndim in (1, 2):
                actions = v
                notes.append(f"actions fallback: '{k}' {v.shape}")
                break
    if actions is None:
        raise ValueError("Could not find actions array in NPZ.")

    # Normalize actions to indices (T-1,)
    a = actions
    if a.ndim == 2 and a.shape[-1] > 1:  # one-hot
        a = a.argmax(axis=-1).astype(np.int64)
        notes.append("actions one-hot→indices")
    elif a.ndim == 1:
        a = a.astype(np.int64)
    elif a.ndim == 2 and a.shape[0] == 1:
        a = a[0].astype(np.int64)
    else:
        raise ValueError(f"Unsupported actions shape: {actions.shape}")

    # Align lengths (ensure len(actions) == T-1)
    if a.shape[0] != f.shape[0] - 1:
        min_len = min(a.shape[0], f.shape[0] - 1)
        notes.append(f"actions length {a.shape[0]} != T-1 {f.shape[0]-1}; clipping to {min_len}")
        a = a[:min_len]
        f = f[: min_len + 1]

    return f, a, notes


# --------------------------- Tensors for the model ---------------------------

def tensors_from_episode(
    frames_T_HWC_uint8: np.ndarray, image_hw: Tuple[int, int], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn NPZ frames into model tensors."""
    T, H, W, C = frames_T_HWC_uint8.shape
    Ht, Wt = image_hw
    frames_chw = []
    for t in range(T):
        x = _to_chw(frames_T_HWC_uint8[t])
        if (H, W) != (Ht, Wt):
            x = _resize_chw(x, (Ht, Wt))
        frames_chw.append(x)
    frames_chw = torch.stack(frames_chw, dim=1)  # C × T × H × W
    prime = frames_chw[:, :1]  # C × 1 × H × W
    gt = frames_chw[:, 1:]     # C × (T-1) × H × W
    return prime.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)


# --------------------------- Inference (chunked) ---------------------------

def chunked_sample(
    model,
    prime_frames: torch.Tensor,
    actions_idx: np.ndarray,
    inference_steps: int,
    device: torch.device,
    sample_temperature: float = 1.0,
    mask_schedule: str = "cosine",
) -> torch.Tensor:
    """Generate predictions in chunks respecting MaskGIT max_seq_len."""
    actions_t = torch.from_numpy(actions_idx.astype(np.int64)).unsqueeze(0).to(device)  # 1 × T_out

    tokens_per_frame = getattr(model.tokenizer, "image_num_tokens", None)
    max_seq_len = getattr(model.dynamics.maskgit, "max_seq_len", None)
    if tokens_per_frame is None or max_seq_len is None:
        raise RuntimeError("Model is missing tokenizer.image_num_tokens or dynamics.maskgit.max_seq_len.")
    max_frames_per_call = max(1, max_seq_len // tokens_per_frame)
    print(f"[Inference] max_frames_per_call={max_frames_per_call} (max_seq_len={max_seq_len}, tokens_per_frame={tokens_per_frame})")

    preds_chunks: List[torch.Tensor] = []
    remaining = int(actions_t.shape[1])
    start = 0
    prime = prime_frames  # 1 × C × Fp × H × W

    while remaining > 0:
        cur = min(max_frames_per_call, remaining)
        a_slice = actions_t[:, start : start + cur]  # 1 × cur
        with torch.no_grad():
            cur_pred = model.sample(
                prime_frames=prime,
                actions=a_slice,
                num_frames=cur,
                inference_steps=inference_steps,
                sample_temperature=sample_temperature,
                mask_schedule=mask_schedule,
                return_recons_only=True,
            )  # 1 × C × cur × H × W
        preds_chunks.append(cur_pred)
        prime = cur_pred[:, :, -1:].detach()  # next prime is the last predicted frame
        start += cur
        remaining -= cur

    return torch.cat(preds_chunks, dim=2)


# --------------------------- Save visuals ---------------------------

def save_like_evaluate(
    out_root: Path,
    dataset_name: str,
    model_name: str,
    base_name: str,
    first_frames: torch.Tensor,
    gt_video: torch.Tensor,
    preds: torch.Tensor,
) -> Path:
    """Save GIF and PNG in evaluate.py style and return the output directory."""
    out_dir = out_root / f"{dataset_name}/{model_name}/samples_npz_{base_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build strips C × T × H × W (prepend primes)
    gt_strip = torch.cat([first_frames[0], gt_video[0]], dim=1)
    pred_strip = torch.cat([first_frames[0], preds[0]], dim=1)

    # GIF
    gif_tensor = torch.cat([gt_strip, pred_strip], dim=2)  # C × T × (H*2) × W
    gif_path = out_dir / f"{base_name}.gif"
    video_tensor_to_gif(gif_tensor.cpu(), str(gif_path))

    # PNG
    gt_img = video_tensor_to_pil_images(gt_strip.cpu(), only_first_image=False)
    pred_img = video_tensor_to_pil_images(pred_strip.cpu(), only_first_image=False)
    combined = Image.new("RGB", (gt_img.width, gt_img.height + pred_img.height))
    combined.paste(gt_img, (0, 0))
    combined.paste(pred_img, (0, gt_img.height))
    combined.save(out_dir / f"{base_name}.png")

    return out_dir


# --------------------------- Episode runner (reuses a preloaded model) ---------------------------
@dataclass
class EpisodeState:
    name: str
    prime_for_save: torch.Tensor         # 1×C×Fp'×H×W (kept for saving rows)
    prime_live: torch.Tensor             # 1×C×Fp'×H×W (updated to last pred)
    gt: torch.Tensor                     # 1×C×T_out×H×W
    actions: np.ndarray                  # (T_out,)
    cursor: int = 0                      # how many frames already predicted
    pred_chunks: list = None             # list[Tensor 1×C×cur×H×W]

    def __post_init__(self):
        if self.pred_chunks is None:
            self.pred_chunks = []

def _prep_episode_state(npz_path: str, model, image_size, device,
                        num_first_frames: int, prime_index: int) -> EpisodeState:
    npz = np.load(npz_path, allow_pickle=True)
    frames_T_HWC, actions_idx, _notes = find_frames_actions(npz)

    # Action fixups
    actions_idx = actions_idx.astype(np.int64)
    act_dim = getattr(model.dynamics.maskgit, "action_dim", None)
    if act_dim == 7 and actions_idx.size > 0 and actions_idx.min() == 1 and actions_idx.max() in (6, 7):
        actions_idx = actions_idx - 1

    # Make tensors and slice by prime_index window
    prime, gt = tensors_from_episode(frames_T_HWC, image_size, device)
    all_frames = torch.cat([prime, gt], dim=2)  # 1×C×T×H×W
    T = int(all_frames.shape[2])
    if T < 2:
        raise ValueError(f"{npz_path}: episode too short (T={T})")

    P = max(0, min(int(prime_index), T - 2))
    Fp = max(1, int(num_first_frames))
    start = max(0, P - Fp + 1)

    prime_win = all_frames[:, :, start:P+1]   # 1×C×Fp'×H×W
    gt_after  = all_frames[:, :, P+1:]        # 1×C×T_out×H×W
    T_out = int(gt_after.shape[2])
    actions_idx = actions_idx[P : P + T_out]

    return EpisodeState(
        name=Path(npz_path).stem,
        prime_for_save=prime_win.clone(),  # keep original for saving rows
        prime_live=prime_win,              # will mutate during rollout
        gt=gt_after,
        actions=actions_idx,
        cursor=0,
        pred_chunks=[]
    )

def run_directory_batched(model, image_size, ep_paths, device,
                          inference_steps, num_first_frames, sample_temperature,
                          mask_schedule, out_root, dataset_name, model_name,
                          prime_index, batch_size, skip_existing=False, save_media: bool = True):

    # figure global per-call frame budget
    tokens_per_frame = getattr(model.tokenizer, "image_num_tokens", None)
    max_seq_len = getattr(model.dynamics.maskgit, "max_seq_len", None)
    if tokens_per_frame is None or max_seq_len is None:
        raise RuntimeError("Model missing tokenizer.image_num_tokens or dynamics.maskgit.max_seq_len.")
    max_frames_per_call = max(1, max_seq_len // tokens_per_frame)
    print(f"[Batched] max_frames_per_call={max_frames_per_call}")

    pending = list(ep_paths)  # paths left to load
    active: list[EpisodeState] = []
    finished_rows = []

    # optional progress
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(ep_paths), desc="Episodes")
    except Exception:
        pbar = None

    def finalize_episode(st: EpisodeState):
        preds = torch.cat(st.pred_chunks, dim=2) if st.pred_chunks else st.gt.new_zeros((1, st.gt.shape[1], 0, st.gt.shape[3], st.gt.shape[4]))

        # Metrics
        psnrs = [psnr_frame(st.gt[0, :, t], preds[0, :, t]) for t in range(st.gt.shape[2])]
        psnr_mean = float(np.mean(psnrs)) if psnrs else float("nan")

        ssim_metric = SSIM(data_range=1.0).to(device)
        ssim_vals = [float(ssim_metric(preds[:, :, t], st.gt[:, :, t]).item()) for t in range(st.gt.shape[2])]
        ssim_mean = float(np.mean(ssim_vals)) if ssim_vals else float("nan")

        out_dir_str = ""
        if save_media:
            out_dir = save_like_evaluate(
                out_root, dataset_name, model_name, base_name=st.name,
                first_frames=st.prime_for_save, gt_video=st.gt, preds=preds
            )
            out_dir_str = str(out_dir)

        finished_rows.append({
            "episode": f"{st.name}.npz",
            "psnr_mean": psnr_mean,
            "ssim_mean": ssim_mean,
        })

        del st


    # main loop
    while active or pending:
        # top up the active batch
        while len(active) < batch_size and pending:
            path = pending.pop(0)
            base = Path(path).stem
            out_dir = out_root / f"{dataset_name}/{model_name}/samples_npz_{base}"
            if skip_existing and (out_dir / f"{base}.png").exists() and (out_dir / f"{base}.gif").exists():
                if pbar: pbar.update(1)
                finished_rows.append({
                    "episode": f"{base}.npz",
                    "saved_dir": str(out_dir),
                    "psnr_mean": float("nan"),
                    "ssim_mean": float("nan"),
                    "n_pred_frames": 0,
                    "notes": "skipped_existing",
                })
                continue
            try:
                st = _prep_episode_state(path, model, image_size, device, num_first_frames, prime_index)
                active.append(st)
            except Exception as e:
                print(f"[Load ERROR] {path}: {e}")
                if pbar: pbar.update(1)
                finished_rows.append({
                    "episode": f"{base}.npz",
                    "saved_dir": "",
                    "psnr_mean": float("nan"),
                    "ssim_mean": float("nan"),
                    "n_pred_frames": 0,
                    "notes": f"error:{type(e).__name__}",
                })

        if not active:
            break

        # step size = min remaining across active, capped by max_frames_per_call
        remain = [s.gt.shape[2] - s.cursor for s in active]
        cur = int(min(max_frames_per_call, max(1, min(remain))))

        # build batch
        prime_batch = torch.cat([s.prime_live for s in active], dim=0)            # B×C×Fp'×H×W
        acts_batch = np.stack([s.actions[s.cursor:s.cursor+cur] for s in active], axis=0)  # B×cur
        acts_batch = torch.from_numpy(acts_batch.astype(np.int64)).to(device, non_blocking=True)

        with torch.no_grad():
            preds = model.sample(
                prime_frames=prime_batch,
                actions=acts_batch,
                num_frames=cur,
                inference_steps=inference_steps,
                sample_temperature=sample_temperature,
                mask_schedule=mask_schedule,
                return_recons_only=True,
            )  # B×C×cur×H×W

        # split and update each episode
        for b, s in enumerate(active):
            cur_pred = preds[b:b+1]                               # 1×C×cur×H×W
            s.pred_chunks.append(cur_pred.detach())
            s.prime_live = cur_pred[:, :, -1:].detach()           # last pred becomes next prime
            s.cursor += cur

        # finalize any finished episodes
        still_active = []
        for s in active:
            if s.cursor >= s.gt.shape[2]:
                finalize_episode(s)
                if pbar: pbar.update(1)
            else:
                still_active.append(s)
        active = still_active

        # small cleanup
        del preds
        torch.cuda.empty_cache()

    if pbar: pbar.close()
    return finished_rows



def run_episode_with_model(
    model,
    image_size: Tuple[int, int],
    npz_path: str,
    device: torch.device,
    inference_steps: int,
    num_first_frames: int,
    sample_temperature: float,
    mask_schedule: str,
    out_root: Path,
    dataset_name: str,
    model_name: str,
    prime_index: int,
    skip_existing: bool = False,
    save_media: bool = True,
    actions_random: bool = False,
    seed: int | None = None,
    action_const: int | None = None,
) -> Tuple[Optional[Path], float, float, int, List[str]]:
    """
    Run a single NPZ episode with a preloaded model.
    Returns (out_dir, psnr_mean, ssim_mean, n_frames_pred, notes)
    On failure, returns (None, nan, nan, 0, notes/err).
    """
    try:
        base = Path(npz_path).stem

        # Fast path: skip if outputs exist
        out_dir = out_root / f"{dataset_name}/{model_name}/samples_npz_{base}"
        if skip_existing and (out_dir / f"{base}.png").exists() and (out_dir / f"{base}.gif").exists():
            print(f"[Skip] Existing outputs for {base}")
            return out_dir, float("nan"), float("nan"), 0, ["skipped_existing"]

        # Load NPZ
        npz = np.load(npz_path, allow_pickle=True)
        frames_T_HWC, actions_idx, notes = find_frames_actions(npz)

        # Action sanity + off-by-one fix
        actions_idx = actions_idx.astype(np.int64)
        act_dim = getattr(model.dynamics.maskgit, "action_dim", None)
        print(f"[actions] model.action_dim={act_dim}, npz range=({actions_idx.min()}..{actions_idx.max()})")
        if act_dim == 7 and actions_idx.size > 0:
            if actions_idx.min() == 1 and actions_idx.max() in (6, 7):
                actions_idx = actions_idx - 1
                print("[actions] shifted 1-based→0-based (1..7→0..6)")
                notes.append("shifted_actions_1based_to_0based")

        if act_dim is not None and actions_idx.size > 0 and actions_idx.max() >= act_dim:
            msg = (f"[WARN] NPZ actions up to {actions_idx.max()} but model.action_dim={act_dim}. "
                   "Likely require PPO→model action mapping (e.g., 15→7).")
            print(msg)
            notes.append("warn_action_dim_mismatch")

        # Build tensors and select prime by index (generalized)
        prime, gt = tensors_from_episode(frames_T_HWC, image_size, device)
        all_frames = torch.cat([prime, gt], dim=2)  # 1×C×T×H×W
        total_T = int(all_frames.shape[2])
        if total_T < 2:
            raise ValueError(f"Episode too short (T={total_T}); need ≥2 frames.")

        orig_P = int(prime_index)
        P = max(0, min(orig_P, total_T - 2))
        if P != orig_P:
            print(f"[prime][WARN] prime_index={orig_P} clamped to {P} (T={total_T}).")
            notes.append(f"prime_index_clamped_{orig_P}_to_{P}")

        Fp = max(1, int(num_first_frames))
        start = max(0, P - Fp + 1)          # inclusive
        prime = all_frames[:, :, start:P+1] # 1×C×Fp'×H×W
        gt    = all_frames[:, :, P+1:]      # 1×C×T_out×H×W
        new_T_out = int(gt.shape[2])
        if new_T_out <= 0:
            raise ValueError(f"prime_index={P} leaves no future frames to predict.")
        
        if actions_random:
            hi = getattr(model.dynamics.maskgit, "action_dim", None) or 7  # fallback if not set
            rng = np.random.default_rng(seed)  # seed may be None → non-deterministic
            actions_idx = rng.integers(0, hi, size=new_T_out, dtype=np.int64)
            print(f"[override] Using RANDOM actions in [0,{hi})"
                f"{'' if seed is None else f' with seed={seed}'}")


        actions_idx = actions_idx[P : P + new_T_out]
        print(f"[prime] using frames [{start}..{P}] (Fp={int(prime.shape[2])}), predicting {new_T_out} frames.")

        # actions_idx currently sliced to length new_T_out
        act_dim = getattr(model.dynamics.maskgit, "action_dim", None)

        if actions_random:
            hi = act_dim or 7
            rng = np.random.default_rng(seed)
            actions_idx = rng.integers(0, hi, size=new_T_out, dtype=np.int64)
            print(f"[override] Using RANDOM actions in [0,{hi})"
                f"{'' if seed is None else f' with seed={seed}'}")

        elif action_const is not None:
            hi = act_dim or 7
            k = int(action_const)
            if k < 0 or k >= hi:
                print(f"[override][WARN] --action_const {k} out of range [0,{hi-1}] → clamping.")
                k = max(0, min(k, hi - 1))
            actions_idx = np.full((new_T_out,), k, dtype=np.int64)
            print(f"[override] Using CONSTANT action {k} for {new_T_out} steps.")



        # Inference
        preds = chunked_sample(
            model, prime, actions_idx, inference_steps, device,
            sample_temperature=sample_temperature, mask_schedule=mask_schedule,
        )

        # Metrics
        psnrs: List[float] = []
        ssim_metric = SSIM(data_range=1.0).to(device)
        with torch.no_grad():
            for t in range(gt.shape[2]):
                psnrs.append(psnr_frame(gt[0, :, t], preds[0, :, t]))
        psnr_mean = float(np.mean(psnrs)) if psnrs else float("nan")
        ssim_mean = ssim_frame_batched(gt, preds, ssim_metric)

        print(f"[Metrics] {base} → PSNR {psnr_mean:.3f} dB, SSIM {ssim_mean:.4f}")

        # Save outputs
        out_dir = None
        if save_media:
            out_dir = save_like_evaluate(
                out_root, dataset_name, model_name, base_name=base,
                first_frames=prime, gt_video=gt, preds=preds,
            )
        return out_dir, psnr_mean, ssim_mean, new_T_out, notes


    except Exception as e:
        print(f"[ERROR] {npz_path}: {e}")
        traceback.print_exc()
        return None, float("nan"), float("nan"), 0, [f"error:{type(e).__name__}"]


# --------------------------- Legacy one-off (kept for compatibility) ---------------------------

def run_episode(
    cfg_path: str,
    ckpt_path: str,
    npz_path: str,
    device: torch.device,
    inference_steps: int,
    num_first_frames: int,
    sample_temperature: float,
    mask_schedule: str,
    out_root: Path,
    dataset_name: str,
    model_name: str,
    prime_index: int,
) -> Path:
    """Backwards-compatible single-episode path (will load model once)."""
    cfg = load_cfg(cfg_path)
    model = build_model(cfg, ckpt_path, device)
    try:
        image_size = model.image_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
    except Exception:
        image_size = (64, 64)

    out_dir, *_ = run_episode_with_model(
        model=model,
        image_size=image_size,
        npz_path=npz_path,
        device=device,
        inference_steps=inference_steps,
        num_first_frames=num_first_frames,
        sample_temperature=sample_temperature,
        mask_schedule=mask_schedule,
        out_root=out_root,
        dataset_name=dataset_name,
        model_name=model_name,
        prime_index=prime_index,
        skip_existing=False,
    )
    if out_dir is None:
        raise RuntimeError(f"Episode failed: {npz_path}")
    return out_dir


# --------------------------- CLI ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="NPZ → world model inference (single or directory)")
    # Model/config
    ap.add_argument("--config", required=True, help="Hydra config .yaml for the model")
    ap.add_argument("--model_ckpt", required=True, help="Path to model checkpoint (.pt/.pth)")
    ap.add_argument("--tokenizer_ckpt", default=None, help="(Optional) separate tokenizer checkpoint")

    # Input: either --npz (single file) or --npz_dir (directory of files)
    ap.add_argument("--npz", default=None, help="Path to one NPZ episode file")
    ap.add_argument("--npz_dir", default=None, help="Directory containing NPZ episodes")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders of --npz_dir")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N episodes")
    ap.add_argument("--skip_existing", action="store_true", help="Skip episodes whose PNG+GIF already exist")

    # Inference opts
    ap.add_argument("--device", default="cuda", help='"cuda" or "cpu"')
    ap.add_argument("--inference_steps", type=int, default=64, help="MaskGIT inference steps")
    ap.add_argument("--num_first_frames", type=int, default=1, help="Number of prime frames (>1 = more context)")
    ap.add_argument("--sample_temperature", type=float, default=1.0, help="Sampling temperature")
    ap.add_argument("--mask_schedule", type=str, default="cosine", help="Mask schedule (e.g., cosine)")

    # Output labeling
    ap.add_argument("--save_root", type=str, default="npz_eval_outputs", help="Output root directory")
    ap.add_argument("--dataset_name", type=str, default="coinrun", help="Dataset label for output path")
    ap.add_argument("--model_name", type=str, default="GenieRedux_Guided_CoinRun_80mln_v1.0", help="Model label for output path")
    ap.add_argument("--metrics_only", action="store_true", help="Only compute metrics and write the CSV summary. Do NOT save GIF/PNG.")


    # Prime control
    ap.add_argument("--prime_index", type=int, default=0, help=(
        "0-based index of the last prime frame. If --num_first_frames > 1, a window ending at this index is used."
    ))

    #Batch size for directory mode
    ap.add_argument("--batch_size", type=int, default=8, help="Episodes processed in parallel on GPU")

    #Actions controls
    ap.add_argument("--action_const", type=int, default=None, help="Use a single action index for all predicted steps (overrides NPZ).")
    ap.add_argument("--actions_random", action="store_true", help="Use random actions uniformly in [0, A). Overrides NPZ/actions.")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed for --actions_random (omit for non-deterministic).")


    args = ap.parse_args()
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")

    save_media = not args.metrics_only

    # Validate inputs
    if (args.npz is None) == (args.npz_dir is None):
        print("Please provide exactly one of --npz (single file) OR --npz_dir (directory).")
        sys.exit(2)

    # Load cfg + model ONCE
    cfg = load_cfg(args.config)
    model = build_model(cfg, args.model_ckpt, device, tokenizer_ckpt=args.tokenizer_ckpt)
    try:
        image_size = model.image_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
    except Exception:
        image_size = (64, 64)

    out_root = Path(args.save_root)

    # SINGLE FILE MODE
    if args.npz is not None:
        out_dir, psnr_m, ssim_m, n_pred, notes = run_episode_with_model(
            model=model,
            image_size=image_size,
            npz_path=args.npz,
            device=device,
            inference_steps=args.inference_steps,
            num_first_frames=args.num_first_frames,
            sample_temperature=args.sample_temperature,
            mask_schedule=args.mask_schedule,
            out_root=out_root,
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            prime_index=args.prime_index,
            skip_existing=args.skip_existing and save_media,
            save_media=save_media,
            actions_random=args.actions_random,
            seed=args.seed,  
            action_const=args.action_const, 
        )
        if save_media and out_dir is not None:
            print(f"[Saved] GIF+PNG → {out_dir}")
        print(f"[Episode Summary] PSNR={psnr_m:.3f} dB, SSIM={ssim_m:.4f}, frames_pred={n_pred}")

        return

    # DIRECTORY MODE
    dir_path = Path(args.npz_dir)
    pattern = "**/*.npz" if args.recursive else "*.npz"
    all_eps = sorted(dir_path.glob(pattern))
    if args.limit is not None:
        all_eps = all_eps[: args.limit]
    if not all_eps:
        print("No .npz episodes found."); sys.exit(2)

    rows = run_directory_batched(
        model=model,
        image_size=image_size,
        ep_paths=[str(p) for p in all_eps],
        device=device,
        inference_steps=args.inference_steps,
        num_first_frames=args.num_first_frames,
        sample_temperature=args.sample_temperature,
        mask_schedule=args.mask_schedule,
        out_root=out_root,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        prime_index=args.prime_index,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing and save_media,  # skip_existing only makes sense if saving media
        save_media=save_media,                             # <— NEW
    )

    # write summary CSV (same as before)
    summary_csv = out_root / args.dataset_name / args.model_name /f"{args.model_name}_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv, numpy as np
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode","psnr_mean","ssim_mean"])
        writer.writeheader(); [writer.writerow(r) for r in rows]
    print(f"[Saved] Summary CSV → {summary_csv}")



if __name__ == "__main__":
    main()
