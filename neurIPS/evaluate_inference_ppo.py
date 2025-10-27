"""
==========================================================================
INFERENCE SCRIPT TO EVALUATE A PPO ROLLOUT NPZ USING A GENIE/GNIEREDEX WORLD MODEL
==========================================================================

What this script does
---------------------
1) Load a single PPO rollout episode from an .npz file (frames + actions).
2) Load a Genie/GenieRedux world model from a YAML Hydra config + checkpoint.
3) Use the **first frame** (or more, via --num_first_frames) and the **action sequence** --> TODO more frames in input
   to generate predicted frames (respecting MaskGIT's max sequence length by chunking).
4) Save a GIF and PNG panel like evaluate.py: top row GT, bottom row Predictions.
5) Print simple metrics (PSNR, SSIM).

Run example
-----------
python evaluate_inference_ppo.py \
  --config path/to/model.yaml \
  --model_ckpt path/to/weights.pt \
  --npz path/to/episode.npz \
  --device cuda \
  --inference_steps 16 \
  --num_first_frames 1 \
  --save_root npz_eval_outputs \
  --dataset_name coinrun \
  --model_name genie_model_name 

Outputs are saved under:
  npz_eval_outputs/<dataset_name>/<model_name>/samples_npz_<npz_stem>/<npz_stem>.gif/.png
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Repo utilities: to construct the model and save visuals
from models import construct_model
from data.data import video_tensor_to_gif, video_tensor_to_pil_images

# SSIM metric
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

# Hydra config loader
from omegaconf import OmegaConf


# Common keys used by various NPZ dumpers
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


def ssim_frame(gt_chw: torch.Tensor, pr_chw: torch.Tensor) -> Optional[float]:
    """SSIM for two CHW frames in [0,1]; returns None if torchmetrics is missing."""
    metric = SSIM(data_range=1.0).to(gt_chw.device)
    return float(metric(pr_chw.unsqueeze(0), gt_chw.unsqueeze(0)).item())


# --------------------------- Config / Model ---------------------------

def load_cfg(yaml_path: str):
    """Load a Hydra YAML config via OmegaConf (required)."""
    if OmegaConf is None:
        raise RuntimeError("omegaconf is required. Install with: pip install omegaconf")
    return OmegaConf.load(yaml_path)


def build_model(cfg, ckpt_path: str, device: torch.device):
    """Construct a Genie/GenieRedux model from cfg and load checkpoint weights.

    Returns a model on eval() and moved to the chosen device.
    """
    model = construct_model(cfg)
    sd = torch.load(ckpt_path, map_location="cpu")
    # allow either a raw state_dict or a training checkpoint with a 'model' field
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    model.eval().to(device)
    return model


# --------------------------- NPZ parsing ---------------------------

def find_frames_actions(npz: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Find and normalize frames/actions from an NPZ dict.

    Returns:
      frames: (T, H, W, C) uint8
      actions: (T-1,) int64 (indices; one-hot is converted to indices)
      notes: list of normalization steps applied
    """
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
    if f.ndim == 5 and f.shape[0] in (1, 3):  # C T H W (plus extra)?
        # treat as C×T×H×W×(maybe 1)
        C, T, H, W = f.shape[:4]
        f = np.transpose(f, (1, 2, 3, 0))
        notes.append("frames converted CxTxHxW -> TxHxWxC")
    elif f.ndim == 4:
        if f.shape[-1] in (1, 3):
            pass  # already T H W C
        elif f.shape[0] in (1, 3):  # C T H W
            C, T, H, W = f.shape
            f = np.transpose(f, (1, 2, 3, 0))
            notes.append("frames converted CxTxHxW -> TxHxWxC")
        elif f.shape[1] in (1, 3):  # T C H W
            T, C, H, W = f.shape
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
        notes.append(
            f"actions length {a.shape[0]} != T-1 {f.shape[0]-1}; clipping to {min_len}"
        )
        a = a[:min_len]
        f = f[: min_len + 1]

    return f, a, notes


# --------------------------- Tensors for the model ---------------------------

def tensors_from_episode(
    frames_T_HWC_uint8: np.ndarray, image_hw: Tuple[int, int], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn NPZ frames into model tensors.

    Args:
      frames_T_HWC_uint8: (T, H, W, C) uint8
      image_hw: (H, W) target size for the model
      device: torch device

    Returns:
      prime: 1 × C × Fp × H × W   (Fp=1 here; more primes handled by caller)
      gt:    1 × C × (T-1) × H × W
    """
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
    gt = frames_chw[:, 1:]  # C × (T-1) × H × W
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
    """Generate predictions in chunks respecting MaskGIT max_seq_len.

    Shapes:
      prime_frames: 1 × C × Fp × H × W
      actions_idx:  (T_out,) int64 (indices)
    Returns:
      preds: 1 × C × T_out × H × W
    """
    actions_t = torch.from_numpy(actions_idx.astype(np.int64)).unsqueeze(0).to(device)  # 1 × T_out

    # Determine the maximum number of frames we can request in a single call
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
    """Save GIF and PNG in evaluate.py style and return the output directory.

    first_frames, gt_video, preds are 1 × C × T × H × W (note: first_frames T=Fp)
    """
    out_dir = out_root / f"{dataset_name}/{model_name}/samples_npz_{base_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build strips C × T × H × W (prepend primes so we visualize the full timeline)
    gt_strip = torch.cat([first_frames[0], gt_video[0]], dim=1)
    pred_strip = torch.cat([first_frames[0], preds[0]], dim=1)

    # GIF: stack GT over Pred along height, keep T as animation dimension
    gif_tensor = torch.cat([gt_strip, pred_strip], dim=2)  # C × T × (H*2) × W
    gif_path = out_dir / f"{base_name}.gif"
    video_tensor_to_gif(gif_tensor.cpu(), str(gif_path))

    # PNG: two rows (GT, Pred)
    gt_img = video_tensor_to_pil_images(gt_strip.cpu(), only_first_image=False)
    pred_img = video_tensor_to_pil_images(pred_strip.cpu(), only_first_image=False)
    combined = Image.new("RGB", (gt_img.width, gt_img.height + pred_img.height))
    combined.paste(gt_img, (0, 0))
    combined.paste(pred_img, (0, gt_img.height))
    combined.save(out_dir / f"{base_name}.png")

    return out_dir


# --------------------------- Orchestration ---------------------------

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
) -> Path:
    """End-to-end run for a single NPZ episode; returns the output directory path."""
    # 1) Load model and its target image size
    cfg = load_cfg(cfg_path)
    model = build_model(cfg, ckpt_path, device)
    try:
        image_size = model.image_size  # could be int or (H, W)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
    except Exception:
        image_size = (64, 64)

    # 2) Load NPZ and normalize
    npz = np.load(npz_path, allow_pickle=True)
    frames_T_HWC, actions_idx, notes = find_frames_actions(npz)
    base = Path(npz_path).stem

    # 3) Action sanity + common off-by-one fix
    actions_idx = actions_idx.astype(np.int64)
    act_dim = getattr(model.dynamics.maskgit, "action_dim", None)
    print(
        f"[actions] model.action_dim={act_dim}, npz range=({actions_idx.min()}..{actions_idx.max()})"
    )
    if act_dim == 7 and actions_idx.size > 0:
        if actions_idx.min() == 1 and actions_idx.max() in (6, 7):
            actions_idx = actions_idx - 1
            print("[actions] shifted 1-based→0-based (1..7→0..6)")
    if act_dim is not None and actions_idx.max() >= act_dim:
        print(
            f"[WARN] NPZ actions go up to {actions_idx.max()} but model.action_dim={act_dim}. "
            "You likely need a PPO→model action mapping (e.g., 15→7)."
        )

    # 4) Build tensors and optionally use >1 prime frames
    prime, gt = tensors_from_episode(frames_T_HWC, image_size, device)
    Fp = max(1, int(num_first_frames))
    if Fp > 1:
        all_frames = torch.cat([prime, gt], dim=2)  # 1 × C × T × H × W
        if all_frames.shape[2] <= Fp:
            print(f"[WARN] num_first_frames={Fp} but T={all_frames.shape[2]} → falling back to 1")
            Fp = 1
        else:
            prime = all_frames[:, :, :Fp]
            gt = all_frames[:, :, Fp:]
            actions_idx = actions_idx[: gt.shape[2]]

    # 5) Inference (chunked)
    preds = chunked_sample(
        model,
        prime,
        actions_idx,
        inference_steps,
        device,
        sample_temperature=sample_temperature,
        mask_schedule=mask_schedule,
    )

    # 6) Metrics
    psnrs: List[float] = []
    ssims: List[Optional[float]] = []
    with torch.no_grad():
        for t in range(gt.shape[2]):
            psnrs.append(psnr_frame(gt[0, :, t], preds[0, :, t]))
            ssims.append(ssim_frame(gt[0, :, t], preds[0, :, t]))
    print(f"[Metrics] PSNR mean: {np.mean(psnrs):.3f} dB")
    if any(s is not None for s in ssims):
        ssim_vals = [s for s in ssims if s is not None]
        print(f"[Metrics] SSIM mean: {np.mean(ssim_vals):.4f}")
    else:
        print("[Metrics] SSIM not available (pip install torchmetrics)")

    # 7) Save outputs
    out_dir = save_like_evaluate(
        out_root,
        dataset_name,
        model_name,
        base_name=base,
        first_frames=prime,
        gt_video=gt,
        preds=preds,
    )
    return out_dir


# --------------------------- CLI ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="NPZ → world model inference (simplified)")
    ap.add_argument("--config", required=True, help="Hydra config .yaml for the model")
    ap.add_argument("--model_ckpt", required=True, help="Path to model checkpoint (.pt/.pth)")
    ap.add_argument("--npz", required=True, help="Path to NPZ episode file")
    ap.add_argument("--device", default="cuda", help='"cuda" or "cpu"')
    ap.add_argument("--inference_steps", type=int, default=16, help="MaskGIT inference steps")
    ap.add_argument("--num_first_frames", type=int, default=1, help="Number of prime frames (>1 = more context)")
    ap.add_argument("--sample_temperature", type=float, default=1.0, help="Sampling temperature")
    ap.add_argument("--mask_schedule", type=str, default="cosine", help="Mask schedule (e.g., cosine)")
    ap.add_argument("--save_root", type=str, default="npz_eval_outputs", help="Output root directory")
    ap.add_argument("--dataset_name", type=str, default="coinrun", help="Dataset label for output path")
    ap.add_argument("--model_name", type=str, default="genie_npz", help="Model label for output path")

    args = ap.parse_args()
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")

    out_dir = run_episode(
        cfg_path=args.config,
        ckpt_path=args.model_ckpt,
        npz_path=args.npz,
        device=device,
        inference_steps=args.inference_steps,
        num_first_frames=args.num_first_frames,
        sample_temperature=args.sample_temperature,
        mask_schedule=args.mask_schedule,
        out_root=Path(args.save_root),
        dataset_name=args.dataset_name,
        model_name=args.model_name,
    )
    print(f"[Saved] GIF+PNG → {out_dir}")


if __name__ == "__main__":
    main()
