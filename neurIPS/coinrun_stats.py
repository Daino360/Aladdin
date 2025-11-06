#!/usr/bin/env python3
import json, argparse, sys

def iter_records(path):
    """Yield episode dicts from either a JSON array file or JSON-lines file."""
    with open(path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":  # JSON array
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Top-level JSON must be a list or use JSON-lines format.")
            for rec in data:
                yield rec
        else:             # JSON-lines
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

def pick_steps(rec, use):
    """Return steps for one episode based on requested field."""
    if use == "actions_len":
        return rec.get("actions_len")
    if use == "T":
        t = rec.get("T")
        return (t - 1) if t is not None else None
    # auto: prefer actions_len, else T-1
    if "actions_len" in rec:
        return rec["actions_len"]
    t = rec.get("T")
    return (t - 1) if t is not None else None

def main():
    ap = argparse.ArgumentParser(description="Compute avg/min/max steps from CoinRun manifest.")
    ap.add_argument("manifest", help="Path to manifest.json (array) or JSON-lines file.")
    ap.add_argument("--use", choices=["auto", "actions_len", "T"], default="auto",
                    help="Which field to treat as steps: actions_len, T (as T-1), or auto (default).")
    ap.add_argument("--only-coin", action="store_true",
                    help="If present, include only episodes that indicate they reached the coin "
                         "(via reached_coin=True or done_reason=='coin'; if missing, episodes are kept).")
    args = ap.parse_args()

    steps = []
    checked = 0
    for rec in iter_records(args.manifest):
        checked += 1
        s = pick_steps(rec, args.use)
        if s is not None:
            steps.append(int(s))

    if not steps:
        print("No episodes with steps found (check --use/--only-coin or manifest fields).", file=sys.stderr)
        sys.exit(1)

    avg = sum(steps) / len(steps)
    print(f"Episodes counted: {len(steps)} / scanned: {checked}")
    print(f"Average steps to coin: {avg:.2f}")
    print(f"Min steps: {min(steps)}")
    print(f"Max steps: {max(steps)}")

if __name__ == "__main__":
    main()
