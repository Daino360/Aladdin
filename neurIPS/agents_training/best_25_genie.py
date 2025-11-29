#!/usr/bin/env python3
"""
Filter evaluation CSVs: keep the best quartile of rows whose target metric lies
inside a specified range, then list rows whose metric exceeds a threshold.
Optionally emit a second CSV containing rows that match a fixed value (e.g.,
native_return == 10).
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List, Dict, Any, Sequence, Set


def read_rows(path: Path, numeric_cols: Set[str]) -> tuple[List[Dict[str, Any]], List[str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        rows: List[Dict[str, Any]] = []
        for row in reader:
            for col in numeric_cols:
                if col in row and row[col] not in (None, ""):
                    try:
                        row[col] = float(row[col])
                    except ValueError:
                        pass
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows, list(reader.fieldnames)


def write_rows(path: Path,
               header: Sequence[str],
               top_rows: List[Dict[str, Any]],
               gt_rows: List[Dict[str, Any]],
               gt_cols: Sequence[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in top_rows:
            writer.writerow([row.get(col, "") for col in header])

        writer.writerow([])  # separator

        writer.writerow(gt_cols)
        for row in gt_rows:
            writer.writerow([row.get(col, "") for col in gt_cols])

        writer.writerow([])  # trailing blank line


def write_coin_csv(path: Path,
                   header: Sequence[str],
                   rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(col, "") for col in header])


def parse_args():
    ap = argparse.ArgumentParser(description="Quartile filter + optional coin-taken export.")
    ap.add_argument("--input", required=True, help="Source CSV path")
    ap.add_argument("--output", required=True, help="Destination CSV path")
    ap.add_argument("--value-column", default="return", help="Column used for range/threshold filtering")
    ap.add_argument("--range-min", type=float, default=0.0, help="Inclusive lower bound for range filter")
    ap.add_argument("--range-max", type=float, default=5.0, help="Inclusive upper bound for range filter")
    ap.add_argument("--gt-threshold", type=float, default=5.0, help="Rows strictly above this go to the second section")
    ap.add_argument("--gt-columns", nargs="*", default=None,
                    help="Columns to show in the >threshold section (default: episode + value-column)")

    ap.add_argument("--coin-output", help="Optional CSV for rows matching --coin-filter-value")
    ap.add_argument("--coin-filter-column", default="native_return", help="Column to match in coin output")
    ap.add_argument("--coin-filter-value", type=float, help="Value to match in --coin-filter-column")
    ap.add_argument("--coin-sort-column", default=None, help="Column used for sorting the coin output")
    ap.add_argument("--coin-sort-desc", action="store_true", help="Sort coin output descending (default ascending)")
    return ap.parse_args()


def get_float(row: Dict[str, Any], column: str) -> float:
    val = row.get(column)
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        raise ValueError(f"Row missing/invalid '{column}': {row}")


def main():
    args = parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    numeric_cols: Set[str] = {args.value_column}
    if args.coin_filter_column:
        numeric_cols.add(args.coin_filter_column)
    if args.coin_sort_column:
        numeric_cols.add(args.coin_sort_column)

    rows, header = read_rows(in_path, numeric_cols)

    if args.value_column not in header:
        raise ValueError(f"Column '{args.value_column}' not found in {in_path}")

    def metric(row):
        return get_float(row, args.value_column)

    in_range = [row for row in rows if args.range_min <= metric(row) <= args.range_max]
    if not in_range:
        raise ValueError(f"No rows with {args.value_column} in [{args.range_min}, {args.range_max}]")

    in_range.sort(key=metric, reverse=True)
    top_n = max(1, math.ceil(len(in_range) * 0.25))
    top_rows = in_range[:top_n]

    gt_rows = sorted(
        [row for row in rows if metric(row) > args.gt_threshold],
        key=metric,
        reverse=True,
    )

    if args.gt_columns:
        gt_cols = args.gt_columns
    else:
        cols = []
        if "episode" in header:
            cols.append("episode")
        elif header:
            cols.append(header[0])
        if args.value_column not in cols:
            cols.append(args.value_column)
        gt_cols = cols

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_rows(out_path, header, top_rows, gt_rows, gt_cols)
    print(f"[save] top-quartile rows: {len(top_rows)} | >{args.gt_threshold} rows: {len(gt_rows)} -> {out_path}")

    if args.coin_output and args.coin_filter_value is not None:
        coin_rows = []
        for row in rows:
            try:
                val = get_float(row, args.coin_filter_column)
            except ValueError:
                continue
            if math.isclose(val, args.coin_filter_value, rel_tol=1e-9, abs_tol=1e-9):
                coin_rows.append(row)

        if args.coin_sort_column:
            def sort_key(r):
                try:
                    return get_float(r, args.coin_sort_column)
                except ValueError:
                    return float("-inf") if args.coin_sort_desc else float("inf")

            coin_rows.sort(key=sort_key, reverse=args.coin_sort_desc)

        coin_path = Path(args.coin_output).expanduser().resolve()
        coin_path.parent.mkdir(parents=True, exist_ok=True)
        write_coin_csv(coin_path, header, coin_rows)
        print(f"[save] coin-filter rows: {len(coin_rows)} -> {coin_path}")


if __name__ == "__main__":
    main()
