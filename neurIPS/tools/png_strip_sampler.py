from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


def infer_frame_dimensions(
    image: "Image.Image",
    frame_width: Optional[int],
    frames_in_strip: Optional[int],
) -> Tuple[int, int]:
    """
    Return (frame_width, frame_count) for a horizontal sprite strip.

    If explicit hints are not provided we assume square frames laid out horizontally,
    so the frame width equals the image height.
    """
    if frames_in_strip:
        if frames_in_strip <= 0:
            raise ValueError("--frames-in-strip must be positive.")
        if image.width % frames_in_strip != 0:
            raise ValueError(
                f"Image width {image.width} not divisible by --frames-in-strip {frames_in_strip}."
            )
        return image.width // frames_in_strip, frames_in_strip

    if frame_width:
        if frame_width <= 0:
            raise ValueError("--frame-width must be positive.")
        if image.width % frame_width != 0:
            raise ValueError(
                f"Image width {image.width} not divisible by --frame-width {frame_width}."
            )
        return frame_width, image.width // frame_width

    if image.width % image.height == 0:
        return image.height, image.width // image.height

    raise ValueError(
        "Unable to infer frame width automatically. "
        "Pass --frame-width or --frames-in-strip to specify it explicitly."
    )


def compute_sample_indices(frame_coin: int, out_frames: int) -> List[int]:
    """
    Compute the frame indices (inclusive) between frame 0 and frame_coin.

    We space frames evenly so the first index is 0 and the last is frame_coin.
    """
    if out_frames <= 0:
        raise ValueError("--out_frames must be positive.")
    if frame_coin < 0:
        raise ValueError("--frame_coin must be non-negative.")
    if out_frames > frame_coin + 1:
        raise ValueError(
            "Requested more output frames than are available before the coin frame."
        )

    if out_frames == 1:
        return [0]

    step = frame_coin / (out_frames - 1)
    indices = [round(i * step) for i in range(out_frames)]
    # Ensure the endpoints are exact.
    indices[0] = 0
    indices[-1] = frame_coin
    return indices


def slice_strip(image: "Image.Image", frame_width: int, indices: List[int]) -> "Image.Image":
    """Extract the requested frames and return them as a new strip."""
    frame_height = image.height
    frame_count = image.width // frame_width

    for idx in indices:
        if idx < 0 or idx >= frame_count:
            raise ValueError(
                f"Index {idx} is outside the source strip (0-{frame_count - 1})."
            )

    strip = Image.new(image.mode, (frame_width * len(indices), frame_height))
    for i, idx in enumerate(indices):
        left = idx * frame_width
        crop = image.crop((left, 0, left + frame_width, frame_height))
        strip.paste(crop, (i * frame_width, 0))
    return strip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a PNG sprite strip up to a target frame and write a new strip "
            "containing evenly spaced frames."
        )
    )
    parser.add_argument(
        "--png",
        required=True,
        type=Path,
        help="Path to the source PNG strip.",
    )
    parser.add_argument(
        "--frame_coin",
        required=True,
        type=int,
        help="Frame index (0-based by default) of the coin frame.",
    )
    parser.add_argument(
        "--out_frames",
        required=True,
        type=int,
        help="Number of frames to include in the output strip.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the output PNG. Defaults to <input>_sampled.png next to the source.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        dest="frame_width",
        default=None,
        help="Width of one frame if it cannot be inferred automatically.",
    )
    parser.add_argument(
        "--frames-in-strip",
        type=int,
        dest="frames_in_strip",
        default=None,
        help="Total frames in the source strip (used to infer frame width).",
    )
    parser.add_argument(
        "--one-based",
        action="store_true",
        help="Treat --frame_coin as 1-based instead of 0-based.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Image is None:
        sys.exit("Pillow is required for this script. Install it with `pip install pillow`.")

    png_path: Path = args.png
    if not png_path.exists():
        sys.exit(f"PNG not found: {png_path}")

    frame_coin = args.frame_coin - 1 if args.one_based else args.frame_coin
    out_frames = args.out_frames

    with Image.open(png_path) as image:
        frame_width, frame_count = infer_frame_dimensions(
            image, args.frame_width, args.frames_in_strip
        )
        if frame_coin >= frame_count:
            sys.exit(
                f"--frame_coin ({args.frame_coin}) exceeds available frames ({frame_count})."
            )

        indices = compute_sample_indices(frame_coin, out_frames)
        new_strip = slice_strip(image, frame_width, indices)

    output_path = args.output or png_path.with_name(
        f"{png_path.stem}_first_to_{args.frame_coin}_x{out_frames}.png"
    )
    new_strip.save(output_path)
    print(
        f"Wrote {len(indices)} frames (indices {indices}) to {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
