#!/usr/bin/env python3
"""
debug_extractor.py
====================

Debug the `extract_x_naive` heuristic:
- Print modal background color, foreground ratio, x_cm (pixels), x_norm ([0,1]).
- Save diagnostics:
  • original.png
  • dist_gray.png     (distance-to-bg grayscale)
  • mask_binary.png   (foreground=255, background=0)
  • semantic.png      (foreground=red, background=blue)
  • overlay.png       (original + red FG overlay + green vertical line at x_cm)
  • debug.json        (numbers for inspection)

Use with either:
  - A real image frame:      --image /path/to/frame.png
  - A synthetic demo image:  --demo

You can also pass TWO frames to see Δx between them:
  python debug_x_extractor.py --image frame_t.png --image2 frame_t1.png

Params:
  --fg_thresh: foreground threshold (L2 distance in [0,√3])
  --down:      downsample stride for modal background estimation
  --out:       output folder
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw

# --------------------------- Config ---------------------------

@dataclass
class XExtractorConfig:
    fg_thresh: float = 0.12   # Foreground if L2 distance from bg > fg_thresh
    down: int = 4             # Downsample step for modal background

# --------------------------- Core functions ---------------------------

def _modal_bg_color(img_float: np.ndarray, down: int) -> np.ndarray:
    """
    Estimate background color by taking the mode in a 32^3 RGB histogram
    over a downsampled grid.
    """
    ds = img_float[::down, ::down]                      # H'×W'×3
    bins = (np.clip(ds, 0, 1) * 31).astype(np.int32)    # 0..31 per channel
    flat = bins.reshape(-1, 3)
    hvals = flat[:, 0] * 32 * 32 + flat[:, 1] * 32 + flat[:, 2]
    mode_hash = np.bincount(hvals).argmax()
    r = mode_hash // (32 * 32)
    g = (mode_hash // 32) % 32
    b = mode_hash % 32
    return np.array([r, g, b], np.float32) / 31.0       # in [0,1]

def extract_x_naive(frame_hwc_uint8: np.ndarray,
                    cfg: XExtractorConfig = XExtractorConfig(),
                    return_debug: bool = False) -> float | Tuple[float, Dict[str, Any]]:
    """
    Compute x ∈ [0,1] as a vertically weighted horizontal center of mass
    of pixels considered 'foreground' (far from modal background color).
    """
    img = frame_hwc_uint8.astype(np.float32) / 255.0
    H, W, _ = img.shape
    if H == 0 or W == 0:
        out = 0.0
        return (out, {}) if return_debug else out

    # 1) modal bg
    bg = _modal_bg_color(img, cfg.down)

    # 2) distance + fg mask
    dist = np.linalg.norm(img - bg[None, None, :], axis=-1)  # H×W
    fg = dist > cfg.fg_thresh
    ys, xs = np.nonzero(fg)

    if xs.size == 0:
        dbg = {
            "bg_color": bg.tolist(),
            "fg_ratio": 0.0,
            "x_cm_pix": 0.0,
            "x_norm": 0.0,
        }
        return (0.0, dbg) if return_debug else 0.0

    # 3) vertical weights favor upper rows
    weights = 1.0 - (ys.astype(np.float32) / max(1, H - 1))
    x_cm = float((xs.astype(np.float32) * weights).sum() / max(1e-6, weights.sum()))
    x_norm = x_cm / max(1, W - 1)

    if return_debug:
        dbg = {
            "bg_color": bg.tolist(),
            "fg_ratio": float(xs.size) / float(H * W),
            "x_cm_pix": x_cm,
            "x_norm": x_norm,
            "H": H, "W": W,
            "fg_thresh": cfg.fg_thresh,
            "down": cfg.down,
        }
        return x_norm, dbg
    return x_norm

# --------------------------- Visualization helpers ---------------------------

def _to_uint8_gray(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, None)
    if x.max() > 0:
        x = x / x.max()
    return (x * 255.0).astype(np.uint8)

def save_diagnostics(frame_u8: np.ndarray,
                     bg_rgb01: np.ndarray,
                     dist: np.ndarray,
                     fg_mask: np.ndarray,
                     x_cm_pix: float,
                     out_dir: str) -> None:
    """
    Save a few useful images:
      - original.png
      - dist_gray.png
      - mask_binary.png
      - semantic.png (BG=blue, FG=red)
      - overlay.png  (original + red FG + green vertical line at x_cm)
    """
    os.makedirs(out_dir, exist_ok=True)
    H, W, _ = frame_u8.shape

    # Original
    Image.fromarray(frame_u8).save(os.path.join(out_dir, "original.png"))

    # Distance grayscale
    Image.fromarray(_to_uint8_gray(dist)).save(os.path.join(out_dir, "dist_gray.png"))

    # Binary mask
    mask_u8 = np.where(fg_mask, 255, 0).astype(np.uint8)
    Image.fromarray(mask_u8).save(os.path.join(out_dir, "mask_binary.png"))

    # Semantic (BG=blue, FG=red)
    sem = np.zeros_like(frame_u8)
    sem[~fg_mask] = np.array([0, 0, 255], np.uint8)     # BG → blue
    sem[ fg_mask] = np.array([255, 0, 0], np.uint8)     # FG → red
    Image.fromarray(sem).save(os.path.join(out_dir, "semantic.png"))

    # Overlay: original + red FG + green x_cm line
    overlay = frame_u8.copy().astype(np.float32)
    red = np.zeros_like(frame_u8)
    red[..., 0] = 255
    alpha = (fg_mask[..., None].astype(np.float32)) * 0.35
    overlay = (overlay * (1 - alpha) + red * alpha).astype(np.uint8)
    im = Image.fromarray(overlay)
    draw = ImageDraw.Draw(im)
    x_line = int(round(x_cm_pix))
    draw.line([(x_line, 0), (x_line, H - 1)], fill=(0, 255, 0), width=2)
    im.save(os.path.join(out_dir, "overlay.png"))

    # BG color swatch for reference
    bg_patch = np.ones((50, 50, 3), np.float32) * bg_rgb01[None, None, :]
    Image.fromarray((bg_patch * 255).astype(np.uint8)).save(os.path.join(out_dir, "bg_color.png"))

def run_once(img_u8: np.ndarray, cfg: XExtractorConfig, out_dir: str, label: str):
    x_norm, dbg = extract_x_naive(img_u8, cfg, return_debug=True)
    print(f"[{label}] H×W={dbg['H']}×{dbg['W']}")
    print(f"[{label}] bg_color ~ {dbg['bg_color']}")
    print(f"[{label}] fg_thresh={dbg['fg_thresh']}  down={dbg['down']}")
    print(f"[{label}] fg_ratio={dbg['fg_ratio']*100:.2f}%")
    print(f"[{label}] x_cm={dbg['x_cm_pix']:.2f} px  x_norm={dbg['x_norm']:.4f}")
    # Save debug JSON
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"debug_{label}.json"), "w") as f:
        json.dump(dbg, f, indent=2)

    # Recompute dist + mask for saving (to avoid passing internals out of extract)
    img = img_u8.astype(np.float32) / 255.0
    bg = np.array(dbg["bg_color"], np.float32)
    dist = np.linalg.norm(img - bg[None, None, :], axis=-1)
    fg = dist > cfg.fg_thresh
    save_diagnostics(img_u8, bg, dist, fg, dbg["x_cm_pix"], out_dir)
    return x_norm

# --------------------------- Synthetic demo ---------------------------

def make_synthetic(H=64, W=128) -> np.ndarray:
    """
    Create a synthetic frame with bluish background + a red-ish 10×10 blob near the right.
    """
    bg = np.array([0.2, 0.4, 0.7], np.float32)
    img = np.ones((H, W, 3), np.float32) * bg[None, None, :]
    # A 10×10 foreground square:
    y0, x0 = H//3, int(W*0.7)
    img[y0:y0+10, x0:x0+10] = np.array([0.90, 0.20, 0.10], np.float32)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

# --------------------------- CLI ---------------------------

def load_image_u8(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.array(im, dtype=np.uint8)

def main():
    ap = argparse.ArgumentParser(description="Debug extractor: modal-bg + foreground mask + x_cm")
    ap.add_argument("--image", type=str, default=None, help="Path to one RGB frame (png/jpg)")
    ap.add_argument("--image2", type=str, default=None, help="Optional second frame to compute Δx")
    ap.add_argument("--demo", action="store_true", help="Use synthetic demo image instead of a file")
    ap.add_argument("--fg_thresh", type=float, default=0.12, help="Foreground threshold (L2 distance)")
    ap.add_argument("--down", type=int, default=4, help="Downsample step for modal background")
    ap.add_argument("--out", type=str, default="extractor_debug", help="Output directory")
    args = ap.parse_args()

    cfg = XExtractorConfig(fg_thresh=args.fg_thresh, down=args.down)
    os.makedirs(args.out, exist_ok=True)

    # Load/generate image(s)
    if args.demo:
        img1 = make_synthetic()
        print("[info] Using synthetic demo frame.")
    else:
        if args.image is None:
            raise SystemExit("Provide --image PATH or use --demo")
        img1 = load_image_u8(args.image)

    # Run on frame 1
    x1 = run_once(img1, cfg, args.out, label="t")

    # Optional second frame to illustrate Δx
    if args.image2 is not None:
        img2 = load_image_u8(args.image2)
        x2 = run_once(img2, cfg, args.out, label="t+1")
        dx = x2 - x1
        print(f"[Δx] x(t)={x1:.4f}  x(t+1)={x2:.4f}  Δx={dx:+.4f}")
        with open(os.path.join(args.out, "delta.json"), "w") as f:
            json.dump({"x_t": x1, "x_t1": x2, "delta": dx}, f, indent=2)

    print(f"[done] Diagnostics saved to: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
