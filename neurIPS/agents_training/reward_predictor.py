#!/usr/bin/env python3
"""
reward_predictor.py
===================

Purpose
-------
A lightweight RewardPredictor that estimates how much the agent moved to the right
between two consecutive frames. You can train it as:

1) **Classification (recommended)**: predict a bucket in {-2, -1, 0, +1, +2}
   (cross-entropy loss). Buckets are defined by thresholds on Δx = x_{t+1} - x_t.

2) **Regression**: predict a continuous Δx in [-1, +1] (or [−something, +something]),
   trained with MSE or L1; at runtime we bucketize Δx into {-2..+2}.

Labeling
--------
Labels can be generated from:
- Ground-truth x (if your NPZ has `x`, `x_pos`, `x_gt`, etc.)
- A robust **heuristic extractor** that estimates x from the frame.

The dataset samples frame pairs (t, t+1) from NPZ episodes and provides:
- Input: stacked frames [f_t, f_{t+1}] as a 6×H×W tensor
- Target: either Δx (float) or a 5-class label for buckets (-2..+2)

Typical use
-----------
- Train here (or from your own trainer).
- Load the predictor in `train_agent_wm.py` and use it for reward.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# --------------------------- Heuristic x extractor ---------------------------

@dataclass
class XExtractorConfig:
    fg_thresh: float = 0.12
    min_area: int = 12

XGT_KEYS   = ["x", "x_pos", "x_gt", "positions_x"]
FRAME_KEYS = ["frames", "videos", "input_frames", "obs", "images", "x"]

def extract_x_naive(frame_hwc_uint8: np.ndarray, cfg: XExtractorConfig) -> float:
    """
    Estimate x ∈ [0,1] from an RGB frame via foreground/background separation.
    Robust and dependency-free. Works well on simple side-scrollers.
    """
    img = frame_hwc_uint8.astype(np.float32) / 255.0
    H, W, _ = img.shape
    ds = img[::4, ::4]
    if ds.size == 0:
        return 0.0

    # modal background color in a coarse 32×32×32 histogram
    bins = (ds * 31).astype(np.int32)
    flat = bins.reshape(-1, 3)
    hvals = flat[:, 0] * 32 * 32 + flat[:, 1] * 32 + flat[:, 2]
    mode_hash = np.bincount(hvals).argmax()
    br = mode_hash // (32 * 32)
    bg = (mode_hash // 32) % 32
    bb = mode_hash % 32
    bg_color = np.array([br, bg, bb], dtype=np.float32) / 31.0

    # foreground mask
    dist = np.linalg.norm(img - bg_color[None, None, :], axis=-1)
    fg = dist > cfg.fg_thresh
    ys, xs = np.nonzero(fg)
    if xs.size == 0:
        return 0.0

    # vertical weighting (favor top pixels to reduce ground clutter)
    weights = 1.0 - (ys.astype(np.float32) / max(1, H - 1))
    x_cm = (xs.astype(np.float32) * weights).sum() / max(1e-6, weights.sum())
    return float(x_cm / max(1, W - 1))

# --------------------------- Bucketing utils ---------------------------

BUCKET_VALUES = np.array([-2, -1, 0, +1, +2], dtype=np.int64)

def bucketize_delta(delta: float, small: float, large: float) -> int:
    """
    Map Δx to {-2,-1,0,+1,+2} using thresholds.
      +2 if d ≥ +large
      +1 if +small ≤ d < +large
       0 if -small < d < +small
      -1 if -large < d ≤ -small
      -2 if d ≤ -large
    """
    if delta >= large: return +2
    if delta >= small: return +1
    if delta > -small: return 0
    if delta > -large: return -1
    return -2

def bucket_to_index(b: int) -> int:
    """Map bucket value in {-2,-1,0,+1,+2} → class index in {0..4}."""
    return int(b + 2)

def index_to_bucket(i: int) -> int:
    """Map class index {0..4} → bucket value {-2..+2}."""
    return int(i - 2)

# --------------------------- Dataset of (frame_t, frame_t+1) ---------------------------

def _to_chw(img_hwc: np.ndarray) -> torch.Tensor:
    """HWC [0..255|0..1] → CHW [0..1] float32."""
    if img_hwc.dtype not in (np.float32, np.float64):
        x = torch.from_numpy(img_hwc.astype(np.uint8)).permute(2, 0, 1).float() / 255.0
    else:
        x = torch.from_numpy(img_hwc).permute(2, 0, 1).float()
        if x.max() > 1.0:
            x = x / 255.0
    return x

def _resize_chw(x_chw: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize CHW to (H,W) via bilinear."""
    return F.interpolate(x_chw.unsqueeze(0), size=size_hw, mode="bilinear", align_corners=False).squeeze(0)

class NPZDeltaDataset(Dataset):
    """
    Build training pairs from NPZ episodes to predict Δx or a 5-way bucket.
    Yields:
      inputs:  (6,H,W)  = concat([f_t, f_{t+1}], dim=0)
      targets: float Δx  OR int class idx in {0..4} if classification=True
    """
    def __init__(
        self,
        glob_pattern: str,
        image_hw: Tuple[int, int],
        use_gt_x: bool = True,
        classification: bool = True,
        bucket_small: float = 0.01,
        bucket_large: float = 0.05,
        max_pairs_per_npz: int = 256,
        seed: int = 0,
    ):
        super().__init__()
        self.paths = sorted(glob.glob(glob_pattern))
        if not self.paths:
            raise RuntimeError(f"No NPZs matched: {glob_pattern}")
        self.Ht, self.Wt = image_hw
        self.use_gt_x = use_gt_x
        self.classification = classification
        self.bucket_small = float(bucket_small)
        self.bucket_large = float(bucket_large)

        self.items: List[Tuple[str, int]] = []  # (path, t)
        rng = random.Random(seed)
        for p in self.paths:
            try:
                with np.load(p, allow_pickle=True) as npz:
                    f = self._locate_frames(npz)
                    if f is None: continue
                    frames = self._normalize_frames(f)
                    T = frames.shape[0]
                    if T < 2: continue
                    idxs = list(range(T - 1))
                    rng.shuffle(idxs)
                    for t in idxs[:max_pairs_per_npz]:
                        self.items.append((p, t))
            except Exception:
                continue

    @staticmethod
    def _locate_frames(npz) -> Optional[np.ndarray]:
        for k in FRAME_KEYS:
            if k in npz: return npz[k]
        return None

    @staticmethod
    def _normalize_frames(f: np.ndarray) -> np.ndarray:
        # Return T×H×W×C uint8
        v = f
        if v.ndim == 5 and v.shape[0] in (1, 3):
            v = np.transpose(v, (1, 2, 3, 0))
        elif v.ndim == 4:
            if v.shape[-1] in (1, 3): pass
            elif v.shape[0] in (1, 3): v = np.transpose(v, (1, 2, 3, 0))
            elif v.shape[1] in (1, 3): v = np.transpose(v, (0, 2, 3, 1))
        else:
            raise ValueError(f"Unsupported frames shape: {f.shape}")
        if v.dtype != np.uint8:
            if v.max() <= 1.0: v = (np.clip(v, 0, 1) * 255).astype(np.uint8)
            else:              v = np.clip(v, 0, 255).astype(np.uint8)
        return v

    @staticmethod
    def _load_x_gt(npz) -> Optional[np.ndarray]:
        for k in XGT_KEYS:
            if k in npz:
                x = npz[k].astype(np.float32).reshape(-1)
                if x.max() > 1.0: x = np.clip(x, 0, 1)
                return x
        return None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        p, t = self.items[idx]
        with np.load(p, allow_pickle=True) as npz:
            frames = self._normalize_frames(self._locate_frames(npz))
            f_t   = frames[t]
            f_tp1 = frames[t + 1]
            x_t, x_tp1 = None, None

            if self.use_gt_x:
                xgt = self._load_x_gt(npz)
                if xgt is not None and (t + 1) < xgt.shape[0]:
                    x_t, x_tp1 = float(xgt[t]), float(xgt[t + 1])

            if x_t is None or x_tp1 is None:
                # fallback heuristic
                x_t   = extract_x_naive(f_t, XExtractorConfig())
                x_tp1 = extract_x_naive(f_tp1, XExtractorConfig())

        # inputs: concat frames along channel
        xt = _resize_chw(_to_chw(f_t),   (self.Ht, self.Wt))
        xp = _resize_chw(_to_chw(f_tp1), (self.Ht, self.Wt))
        x6 = torch.cat([xt, xp], dim=0)  # 6×H×W

        delta = float(x_tp1 - x_t)
        if self.classification:
            b = bucketize_delta(delta, self.bucket_small, self.bucket_large)
            y = torch.tensor(bucket_to_index(b), dtype=torch.long)
        else:
            y = torch.tensor(delta, dtype=torch.float32)

        return x6, y

# --------------------------- Reward predictor model ---------------------------

class RewardPredictor(nn.Module):
    """
    CNN that consumes a 6-channel tensor [f_t, f_{t+1}] and predicts either:
      • 5 logits (classification over {-2,-1,0,+1,+2}) or
      • 1 scalar (regression for Δx).
    """
    def __init__(self, H: int, W: int, mode: str = "cls"):
        super().__init__()
        assert mode in {"cls", "reg"}
        self.mode = mode
        C_in = 6  # stacked frames
        self.enc = nn.Sequential(
            nn.Conv2d(C_in, 32, 8, stride=4, padding=2), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  nn.ReLU(True),
        )
        with torch.no_grad():
            flat = self.enc(torch.zeros(1, C_in, H, W)).view(1, -1).shape[1]
        if mode == "cls":
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, 256), nn.ReLU(True),
                nn.Linear(256, 5),  # 5 classes
            )
        else:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, 256), nn.ReLU(True),
                nn.Linear(256, 1),
                nn.Tanh(),  # keep Δx roughly in [-1,1] (scale if needed)
            )

    def forward(self, x6: torch.Tensor):
        h = self.enc(x6)
        out = self.head(h)
        if self.mode == "reg":
            return out.squeeze(-1)
        return out  # logits

# --------------------------- Tiny trainer (optional) ---------------------------

@dataclass
class RPTrainCfg:
    mode: str = "cls"            # "cls" or "reg"
    loss: str = "ce"             # "ce" | "mse" | "l1"
    epochs: int = 5
    batch_size: int = 256
    lr: float = 3e-4
    val_frac: float = 0.1

@torch.no_grad()
def evaluate(loader: DataLoader, net: RewardPredictor, cfg: RPTrainCfg, device: torch.device) -> float:
    net.eval()
    meter, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        if cfg.mode == "cls":
            logits = net(xb)
            pred = logits.argmax(dim=-1).cpu()
            acc = (pred == yb).float().mean().item()
            meter += acc * yb.shape[0]
            n += yb.shape[0]
        else:
            yb = yb.to(device)
            pred = net(xb)
            if cfg.loss == "l1":  val = F.l1_loss(pred, yb, reduction="sum")
            elif cfg.loss == "mse": val = F.mse_loss(pred, yb, reduction="sum")
            else: val = F.mse_loss(pred, yb, reduction="sum")  # default
            meter += val.item()
            n += yb.numel()
    return meter / max(1, n)

def train_reward_predictor(
    glob_pattern: str,
    image_hw: Tuple[int, int],
    device: torch.device,
    out_path: str,
    use_gt_x: bool = True,
    classification: bool = True,
    loss: str = "ce",   # "ce"|"mse"|"l1"
    bucket_small: float = 0.01,
    bucket_large: float = 0.05,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 3e-4,
    max_pairs_per_npz: int = 256,
    seed: int = 0,
):
    ds = NPZDeltaDataset(
        glob_pattern=glob_pattern,
        image_hw=image_hw,
        use_gt_x=use_gt_x,
        classification=classification,
        bucket_small=bucket_small,
        bucket_large=bucket_large,
        max_pairs_per_npz=max_pairs_per_npz,
        seed=seed,
    )
    val_len = int(len(ds) * 0.1)
    train_len = len(ds) - val_len
    ds_tr, ds_va = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(seed))

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=2)

    net = RewardPredictor(image_hw[0], image_hw[1], mode=("cls" if classification else "reg")).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    if classification:
        criterion = nn.CrossEntropyLoss()
        cfg = RPTrainCfg(mode="cls", loss="ce", epochs=epochs, batch_size=batch_size, lr=lr)
    else:
        if loss == "l1":      criterion = nn.L1Loss()
        elif loss == "mse":   criterion = nn.MSELoss()
        else:                 criterion = nn.MSELoss()
        cfg = RPTrainCfg(mode="reg", loss=loss, epochs=epochs, batch_size=batch_size, lr=lr)

    best_metric = float("inf") if not classification else 0.0
    for ep in range(1, epochs + 1):
        net.train()
        total, n_batches = 0.0, 0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            if classification:
                yb = yb.to(device)
                logits = net(xb)
                loss_val = criterion(logits, yb)
            else:
                yb = yb.to(device)
                pred = net(xb)
                loss_val = criterion(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss_val.backward()
            opt.step()

            total += loss_val.item()
            n_batches += 1

        if classification:
            val_acc = evaluate(dl_va, net, cfg, device) if val_len > 0 else float("nan")
            print(f"[RewardPred] ep={ep}/{epochs} train_loss={total/max(1,n_batches):.4f} val_acc={val_acc:.4f}")
            is_best = val_acc > best_metric
            if is_best:
                best_metric = val_acc
                torch.save({"model": net.state_dict(), "image_hw": image_hw, "mode": "cls"}, out_path)
        else:
            val_mse = evaluate(dl_va, net, cfg, device) if val_len > 0 else float("nan")
            print(f"[RewardPred] ep={ep}/{epochs} train_loss={total/max(1,n_batches):.4f} val_mse={val_mse:.6f}")
            is_best = val_mse < best_metric
            if is_best:
                best_metric = val_mse
                torch.save({"model": net.state_dict(), "image_hw": image_hw, "mode": "reg"}, out_path)

    if best_metric in (float("inf"), 0.0):  # no val
        torch.save({"model": net.state_dict(), "image_hw": image_hw, "mode": ("cls" if classification else "reg")}, out_path)
        print(f"[RewardPred] saved → {out_path}")

def load_reward_predictor(path: str, device: torch.device, image_hw: Tuple[int, int]) -> RewardPredictor:
    ck = torch.load(path, map_location="cpu")
    mode = ck.get("mode", "cls")
    net = RewardPredictor(image_hw[0], image_hw[1], mode=mode).to(device)
    sd = ck["model"] if "model" in ck else ck
    net.load_state_dict(sd)
    net.eval()
    return net
