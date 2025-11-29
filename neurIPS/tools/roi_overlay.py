#!/usr/bin/env python3
"""
Visualize an ROI on a single image using the same weighting logic as agent_train,
but without drawing the textual "Y=" overlay. The script reads an input image,
computes the ROI heatmap, draws the ROI rectangle, and saves the blended result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def _downscale_to_longside(gray: np.ndarray, longside: int) -> np.ndarray:
    """Resize grayscale image so that max(H, W) == longside (keeps aspect)."""
    H, W = gray.shape
    if longside is None or max(H, W) == longside:
        return gray
    scale = float(longside) / float(max(H, W))
    new_size = (max(8, int(W * scale)), max(8, int(H * scale)))
    return cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)


def _roi_weight(H: int, W: int, *, bottom_bias: float, roi) -> np.ndarray:
    """Build weight map with bottom emphasis + rectangular ROI mask."""
    row = np.linspace(0, 1, H, dtype=np.float32)
    wv = 1.0 + (row ** 2) * (bottom_bias - 1.0)
    wv /= max(np.mean(wv), 1e-9)
    weight = np.repeat(wv[:, None], W, axis=1)

    y0, y1, x0, x1 = roi
    iy0, iy1, ix0, ix1 = int(y0 * H), int(y1 * H), int(x0 * W), int(x1 * W)
    mask = np.zeros_like(weight, dtype=np.float32)
    mask[iy0:iy1, ix0:ix1] = 1.0
    return weight * mask


def parse_roi(roi_str: str):
    parts = [float(v.strip()) for v in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be four comma-separated floats: y0,y1,x0,x1")
    y0, y1, x0, x1 = parts
    if not (0.0 <= y0 < y1 <= 1.0 and 0.0 <= x0 < x1 <= 1.0):
        raise ValueError("ROI values must satisfy 0<=y0<y1<=1 and 0<=x0<x1<=1")
    return y0, y1, x0, x1


def draw_roi_overlay(frame: np.ndarray, roi, bottom_bias: float, downscale: int, with_heatmap: bool = True):
    """Return the frame with ROI rectangle (and optional heat overlay) drawn."""
    H, W = frame.shape[:2]
    if with_heatmap:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        small = _downscale_to_longside(gray, downscale)
        Hf, Wf = small.shape
        weight = _roi_weight(Hf, Wf, bottom_bias=bottom_bias, roi=roi)
        wnorm = (weight - weight.min()) / (weight.max() - weight.min() + 1e-9)
        wimg = cv2.resize(wnorm, (W, H), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap((wimg * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = (0.6 * heat + 0.4 * frame).astype(np.uint8)
    else:
        overlay = frame.copy()

    y0, y1, x0, x1 = roi
    iy0, iy1, ix0, ix1 = int(y0 * H), int(y1 * H), int(x0 * W), int(x1 * W)
    cv2.rectangle(overlay, (ix0, iy0), (ix1, iy1), (0, 128, 255), 2)
    return overlay


def main():
    ap = argparse.ArgumentParser(description="Generate an ROI overlay image without text labels.")
    ap.add_argument("--image", required=True, help="Input image path (RGB).")
    ap.add_argument("--out", default="roi_overlay.png", help="Output path for the overlay image.")
    ap.add_argument("--roi", default="0.20,0.95,0.05,0.95", help="Normalized ROI y0,y1,x0,x1 in [0,1].")
    ap.add_argument("--bottom_bias", type=float, default=2.0, help="Emphasize lower rows when building the weight map.")
    ap.add_argument("--downscale", type=int, default=64, help="Flow/weight map longside resolution before upsampling.")
    ap.add_argument("--no-heatmap", action="store_true", help="Skip the heat overlay and draw only the ROI rectangle.")
    args = ap.parse_args()

    roi = parse_roi(args.roi)
    frame = imageio.imread(args.image)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.shape[-1] == 4:
        frame = frame[..., :3]  # drop alpha if present

    overlay = draw_roi_overlay(
        frame,
        roi=roi,
        bottom_bias=args.bottom_bias,
        downscale=args.downscale,
        with_heatmap=not args.no_heatmap,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, overlay)
    print(f"Saved ROI overlay to {out_path}")


if __name__ == "__main__":
    main()
