#!/usr/bin/env python3
"""
Normalize CoinRun goal rewards into comparable [0,1] scores by converting returns
to estimated distance progress.

For each CSV:
  1. Read the per-episode return `G`.
  2. Optionally subtract a success bonus (if a boolean success column is present).
  3. Convert the return into traveled distance using `progress = max(0, distance_sign * G_base) / reward_scale`.
  4. Estimate a maximum plausible distance `D_max` (user supplied or percentile across all rows).
  5. Emit normalized scores `s = progress / D_max` (clipped to [0,1] by default).

The resulting CSV contains the per-episode mean distance and normalized score, plus summary stats.

Example:
    python agents_training/goal_score_normalizer.py \
        --csv wm_eval_outputs/coinrun/TestPPO_SIMtoWM/TestPPO_SIMtoWM_ppo_wm_eval.csv \
        --csv agents_training/fromWM_simulator_eval_outputs/PPO_02_coinrun_20251113-161057/simulator_eval_summary.csv \
        --reward-column return --length-column len \
        --success-bonus 1.0 --success-column success \
        --d-max-percentile 99
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, List, Optional, Sequence, Tuple


DEFAULT_REWARD_FIELDS = ("return", "custom_return", "goal_reward")
DEFAULT_LENGTH_FIELDS = ("len", "steps", "length", "episode_len")


def detect_column(
    fieldnames: Sequence[str],
    preferred: Optional[str],
    fallbacks: Sequence[str],
    label: str,
) -> str:
    if preferred:
        if preferred not in fieldnames:
            raise ValueError(f"{label} column '{preferred}' is missing from CSV headers: {fieldnames}")
        return preferred
    for cand in fallbacks:
        if cand in fieldnames:
            return cand
    raise ValueError(f"Could not find a {label} column among {fallbacks}; CSV headers: {fieldnames}")


def parse_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def parse_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in ("1", "true", "t", "yes", "y", "success", "done")


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty list.")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    vals = sorted(values)
    pos = (q / 100.0) * (len(vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] + (vals[hi] - vals[lo]) * frac


@dataclass
class EpisodeRecord:
    episode: str
    base_return: float
    length: float
    progress_distance: float
    success: bool
    raw_reward: float


def read_csv_metrics(
    path: Path,
    *,
    reward_column: Optional[str],
    length_column: Optional[str],
    success_column: Optional[str],
    success_bonus: float,
    distance_sign: float,
    reward_scale: float,
    episode_column: str,
) -> Tuple[List[EpisodeRecord], str, str]:
    records: List[EpisodeRecord] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header row.")
        reward_col = detect_column(reader.fieldnames, reward_column, DEFAULT_REWARD_FIELDS, "reward")
        length_col = detect_column(reader.fieldnames, length_column, DEFAULT_LENGTH_FIELDS, "length")
        success_col = success_column if success_column and success_column in reader.fieldnames else None

        for row in reader:
            episode_name = str(row.get(episode_column, "")).strip()
            reward = parse_float(row.get(reward_col))
            length = parse_float(row.get(length_col))
            if reward is None or length is None or length <= 0:
                continue
            success = parse_bool(row.get(success_col)) if success_col else False
            base_return = reward - (success_bonus if success else 0.0)
            progress_distance = max(0.0, distance_sign * base_return) / max(reward_scale, 1e-9)
            records.append(
                EpisodeRecord(
                    episode=episode_name or f"row{len(records)}",
                    base_return=base_return,
                    length=length,
                    progress_distance=progress_distance,
                    success=success,
                    raw_reward=reward,
                )
            )
    return records, reward_col, length_col


def summarize_scores(scores: Iterable[float]) -> Tuple[float, float, float, float]:
    vals = list(scores)
    if not vals:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    return (
        mean(vals),
        pstdev(vals),
        min(vals),
        max(vals),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize goal rewards into [0,1] scores based on mean distance.")
    ap.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more CSV files with per-episode returns and lengths.",
    )
    ap.add_argument("--reward-column", default=None, help="Name of the reward/return column (auto-detect if omitted).")
    ap.add_argument("--length-column", default=None, help="Name of the episode length column (auto-detect if omitted).")
    ap.add_argument("--success-column", default=None, help="Boolean column that indicates success (optional).")
    ap.add_argument(
        "--success-bonus",
        type=float,
        default=0.0,
        help="Bonus to subtract from the return when success_column is true (default: 0).",
    )
    ap.add_argument(
        "--distance-sign",
        type=float,
        default=1.0,
        help="Multiplier applied before computing progress. Use 1 if higher returns mean better progress, "
        "-1 if returns are negative distances.",
    )
    ap.add_argument(
        "--reward-scale",
        type=float,
        default=5.0,
        help="Scale factor used when computing the flow reward (distance units per reward point).",
    )
    ap.add_argument(
        "--d-max",
        type=float,
        default=None,
        help="Optional fixed D_max (maximum distance). If omitted, computed via percentile over all progress distances.",
    )
    ap.add_argument(
        "--d-max-percentile",
        type=float,
        default=99.0,
        help="Percentile used to derive D_max when not provided explicitly (default: 99).",
    )
    ap.add_argument(
        "--clip-scores",
        action="store_true",
        default=True,
        help="Clamp normalized scores to [0,1] (default: enabled).",
    )
    ap.add_argument(
        "--no-clip-scores",
        action="store_false",
        dest="clip_scores",
        help="Disable clamping so scores may exceed [0,1].",
    )
    ap.add_argument(
        "--episode-column",
        default="episode",
        help="Column storing the episode identifier (used in the output).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where normalized CSVs are written (defaults to each input file's directory).",
    )
    ap.add_argument(
        "--success-threshold",
        type=float,
        default=0.9,
        help="Score threshold used to count an episode as 'coin taken' when no explicit success column is present.",
    )
    args = ap.parse_args()

    all_records: List[Tuple[Path, List[EpisodeRecord]]] = []
    all_progress: List[float] = []
    for csv_path in args.csv:
        recs, reward_col, length_col = read_csv_metrics(
            csv_path,
            reward_column=args.reward_column,
            length_column=args.length_column,
            success_column=args.success_column,
            success_bonus=args.success_bonus,
            distance_sign=args.distance_sign,
            reward_scale=args.reward_scale,
            episode_column=args.episode_column,
        )
        if not recs:
            print(f"[WARN] {csv_path}: no valid episodes (missing reward/length columns?).")
            continue
        all_records.append((csv_path, recs))
        all_progress.extend(r.progress_distance for r in recs)
        print(
            f"[INFO] {csv_path.name}: parsed {len(recs)} episodes "
            f"(reward column '{args.reward_column or reward_col}', length column '{args.length_column or length_col}')."
        )

    if not all_records:
        raise SystemExit("No CSVs with valid data were processed.")

    if args.d_max is not None:
        d_max = args.d_max
    else:
        positive_dists = [d for d in all_progress if d >= 0]
        target_vals = positive_dists if positive_dists else all_progress
        d_max = percentile(target_vals, args.d_max_percentile)
    if d_max <= 0:
        print(f"[WARN] Computed non-positive D_max={d_max}; using 1.0 instead.")
        d_max = 1.0
    print(f"[INFO] Using D_max = {d_max:.6f}")

    for csv_path, recs in all_records:
        out_dir = args.output_dir or csv_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{csv_path.stem}_goal_scores.csv"
        scores: List[float] = []
        inferred_success = 0
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "raw_reward",
                    "base_return",
                    "length",
                    "progress_distance",
                    "normalized_score",
                    "success",
                    "score_success",
                ],
            )
            writer.writeheader()
            for rec in recs:
                raw_score = rec.progress_distance / d_max
                score = max(0.0, min(1.0, raw_score)) if args.clip_scores else raw_score
                scores.append(score)
                score_success = score >= args.success_threshold
                inferred_success += int(score_success)
                writer.writerow(
                    {
                        "episode": rec.episode,
                        "raw_reward": rec.raw_reward,
                        "base_return": rec.base_return,
                        "length": rec.length,
                        "progress_distance": rec.progress_distance,
                        "normalized_score": score,
                        "success": int(rec.success),
                        "score_success": int(score_success),
                    }
                )
        avg, std, mn, mx = summarize_scores(scores)
        success_rate = inferred_success / len(scores) if scores else float("nan")
        print(
            f"[DONE] {csv_path.name}: wrote {len(scores)} rows → {out_path} | "
            f"mean={avg:.3f} std={std:.3f} min={mn:.3f} max={mx:.3f} | "
            f"score>={args.success_threshold:.2f}: {success_rate*100:.1f}%"
        )


if __name__ == "__main__":
    main()
