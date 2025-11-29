#!/usr/bin/env python3
"""
Quick optical flow visualizer that mirrors the surrogate reward math.

Usage:
  python tools/visualize_optical_flow.py \
    --frame1 path/to/frame0.png \
    --frame2 path/to/frame1.png \
    --out flow_debug.png

It uses the same Farneback flow residual + ROI weighting as the training
pipeline (see agent_train_simulator.py) and annotates the weighted dx/dy
estimate that drives the reward.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

def _downscale_to_longside(gray: np.ndarray, longside: int) -> np.ndarray:
    H, W = gray.shape
    if longside is None or max(H, W) == longside:
        return gray
    s = float(longside) / float(max(H, W))
    new_size = (max(8, int(W * s)), max(8, int(H * s)))
    return cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)


def _flow_residual(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    *,
    longside: int = 64,
    baseline: str = "sky",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Farneback residual flow with the same baselining used in training.
    Mirrors agent_train_simulator._flow_residual to stay reward-consistent.
    """
    p = _downscale_to_longside(prev_gray, longside)
    n = _downscale_to_longside(next_gray, longside)
    flow = cv2.calcOpticalFlowFarneback(p, n, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    fx, fy = flow[..., 0], flow[..., 1]
    if baseline == "global":
        fx -= np.median(fx)
        fy -= np.median(fy)
    elif baseline == "sky":
        h = fx.shape[0]
        top = slice(0, max(1, int(0.2 * h)))
        fx -= np.median(fx[top, :])
        fy -= np.median(fy[top, :])
    elif baseline == "none":
        pass
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    return fx, fy


def _roi_weight(
    H: int,
    W: int,
    *,
    bottom_bias: float,
    roi: tuple[float, float, float, float],
) -> np.ndarray:
    row = np.linspace(0, 1, H, dtype=np.float32)
    wv = 1.0 + (row**2) * (bottom_bias - 1.0)
    wv /= max(np.mean(wv), 1e-9)
    weight = np.repeat(wv[:, None], W, axis=1)
    y0, y1, x0, x1 = roi
    iy0, iy1 = int(y0 * H), int(y1 * H)
    ix0, ix1 = int(x0 * W), int(x1 * W)
    mask = np.zeros_like(weight, dtype=np.float32)
    mask[iy0:iy1, ix0:ix1] = 1.0
    return weight * mask


def _load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _flow_to_rgb(fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """Dense HSV -> RGB flow visualization."""
    mag, ang = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((*fx.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)  # direction
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _weighted_dx_dy(fx: np.ndarray, fy: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    denom = float(weight.sum() + 1e-9)
    dxf_px = float(((-fx) * weight).sum() / denom)
    dyf_px = float(((-fy) * weight).sum() / denom)
    H, W = fx.shape
    dx = dxf_px / max(W, 1)
    dy = dyf_px / max(H, 1)
    return dx, dy


def visualize_optical_flow(
    frame1_path: Path,
    frame2_path: Path,
    out_path: Path,
    roi: tuple[float, float, float, float],
    longside: int,
    bottom_bias: float,
    baseline: str,
) -> None:
    f1 = _load_rgb(frame1_path)
    f2 = _load_rgb(frame2_path)
    g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)

    fx, fy = _flow_residual(g1, g2, longside=longside, baseline=baseline)
    Hf, Wf = fx.shape
    weight = _roi_weight(Hf, Wf, bottom_bias=bottom_bias, roi=roi)
    dx, dy = _weighted_dx_dy(fx, fy, weight)

    flow_rgb = _flow_to_rgb(fx, fy)
    wnorm = (weight - weight.min()) / (weight.max() - weight.min() + 1e-9)
    weight_heat = cv2.applyColorMap((wnorm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    weight_heat = cv2.cvtColor(weight_heat, cv2.COLOR_BGR2RGB)
    weight_heat = cv2.resize(weight_heat, (flow_rgb.shape[1], flow_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    flow_overlay = (0.65 * flow_rgb + 0.35 * weight_heat).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axes:
        ax.axis("off")

    def _draw_roi(ax, img):
        y0, y1, x0, x1 = roi
        H, W = img.shape[:2]
        ax.imshow(img)
        ax.add_patch(
            plt.Rectangle(
                (x0 * W, y0 * H),
                (x1 - x0) * W,
                (y1 - y0) * H,
                edgecolor="orange",
                linewidth=2,
                facecolor="none",
            )
        )

    axes[0].set_title("Frame 1")
    _draw_roi(axes[0], f1)

    axes[1].set_title("Frame 2")
    _draw_roi(axes[1], f2)

    axes[2].set_title(f"Flow (dx={dx:.3f}, dy={dy:.3f})")
    axes[2].imshow(flow_overlay)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[flow] saved visualization to {out_path}")


def parse_roi(s: str) -> tuple[float, float, float, float]:
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("ROI must be y0,y1,x0,x1")
    return tuple(vals)  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Visualize optical flow between two frames.")
    parser.add_argument("--frame1", required=True, type=Path, help="path to first frame (RGB image)")
    parser.add_argument("--frame2", required=True, type=Path, help="path to second frame (RGB image)")
    parser.add_argument("--out", type=Path, default=Path("flow_debug.png"), help="output PNG path")
    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=(0.20, 0.95, 0.05, 0.95),
        help="normalized ROI y0,y1,x0,x1 (defaults to training ROI)",
    )
    parser.add_argument("--downscale", type=int, default=64, help="flow longside resolution")
    parser.add_argument(
        "--bottom-bias",
        type=float,
        default=2.0,
        help="emphasize lower rows for odometry weighting",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["sky", "global", "none"],
        default="sky",
        help="baseline subtraction used during training",
    )
    args = parser.parse_args()

    visualize_optical_flow(
        args.frame1,
        args.frame2,
        args.out,
        roi=args.roi,
        longside=args.downscale,
        bottom_bias=args.bottom_bias,
        baseline=args.baseline,
    )


if __name__ == "__main__":
    main()
