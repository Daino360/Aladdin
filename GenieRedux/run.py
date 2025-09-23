"""
run.py: unified command-line entrypoint for two stacks:
- AutoExplore (training/evaluation driven directly via Hydra-composed configs)
- GenieRedux (training/evaluation delegated to Hydra-enabled scripts, with optional
  multi-process launch through Accelerate)

It also exposes a special subcommand:
  `python run.py generate <hydra overrides>`
which forwards to data_generation/generate.py after validating that any path-like
overrides (_fpath/_dpath/_path) are absolute.

Key behaviors
-------------
- Parses the first positional args: <stack> (auto_explore|genie_redux) and
  <action> (train|eval).
- For AutoExplore: composes Hydra config in-process and calls run(cfg)
  from train_auto_explore or eval_auto_explore (lazy-imported to avoid side effects).
- For GenieRedux: reads num_processes from Hydra config and either:
  * launches the target script via accelerate.launch (multiprocess),
  * or runs the script directly (single process).
- Ensures unbuffered Python output for subprocesses so logs/progress are immediate.
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path

from pathlib import Path
from hydra import compose, initialize
from omegaconf import DictConfig

# Avoid importing stacks at module import time to prevent side effects


def main() -> int:
    """
    Parse CLI, route to the requested stack/action, and execute.

    Supported modes
    ---------------
    1) Data generation:
       `python run.py generate <hydra overrides>`
       - Validates that any *_fpath|*_dpath|*_path override is absolute
         (to avoid ambiguous CWD issues).
       - Delegates to data_generation/generate.py with the given overrides,
         setting CWD to the data_generation directory.

    2) AutoExplore:
       `python run.py auto_explore (train|eval) <hydra overrides>`
       - Composes the corresponding Hydra config (trainer/evaluate).
       - Lazily imports train_auto_explore or eval_auto_explore.
       - Calls their run(cfg) directly (no subprocess).

    3) GenieRedux:
       `python run.py genie_redux (train|eval) <hydra overrides>`
       - Composes the default config (configs/default.yaml) with overrides.
       - Reads train/eval.num_processes to decide:
         * If >1, launches the chosen script via accelerate.launch (bf16).
         * Otherwise, runs the script directly so its @hydra.main handles overrides.

    Returns
    -------
    int
        Process return code (0 on success, non-zero on failure).
    """    
    # Special-case: expose data generation via `run.py generate <hydra overrides>`
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        overrides = sys.argv[2:]
        gen_dir = (Path(__file__).parent / "data_generation").resolve()
        script_path = str(gen_dir / "generate.py")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Validate that any path-like overrides are absolute (ban relative paths)
        # Enforce for keys ending with _fpath, _dpath, or _path
        path_suffixes = ("_fpath", "_dpath", "_path")
        for ov in overrides:
            if "=" not in ov:
                continue
            key, val = ov.split("=", 1)
            key = key.strip()
            if not key.endswith(path_suffixes):
                continue
            v = val.strip().strip('"').strip("'")
            # Allow null-like values to pass through
            if v in ("", "null", "None"):
                continue
            v_expanded = os.path.expanduser(v)
            if not os.path.isabs(v_expanded):
                sys.stderr.write(
                    f"Error: override '{key}' must be an absolute path, got '{val}'.\n"
                )
                return 2
        # Delegate to the Hydra-enabled script so overrides are handled uniformly
        # Run with CWD set to data_generation so relative paths (configs/, annotations/, etc.) resolve correctly.
        return subprocess.call([sys.executable, "-u", script_path] + list(overrides or []), env=env, cwd=str(gen_dir))

    parser = argparse.ArgumentParser(description="Unified entrypoint for GenieRedux and AutoExplore stacks")
    parser.add_argument("stack", choices=["auto_explore", "genie_redux"], help="Target stack")
    parser.add_argument("action", choices=["train", "eval"], help="Action to perform")
    # Note: distributed launch settings now come from Hydra config
    # Everything after is passed as Hydra overrides to the selected stack
    # Parse known args and treat the rest as Hydra overrides
    args, overrides = parser.parse_known_args()

    # Compose Hydra configs and call run functions directly
    if args.stack == "auto_explore":
        cfg_name = "trainer" if args.action == "train" else "evaluate"
        with initialize(version_base=None, config_path="auto_explore/configs"):
            cfg: DictConfig = compose(config_name=cfg_name, overrides=overrides)
        # Lazy import to avoid side effects
        import train_auto_explore as ae_train
        import eval_auto_explore as ae_eval
        if args.action == "train":
            ae_train.run(cfg)
            return 0
        if args.action == "eval":
            ae_eval.run(cfg)
            return 0

    if args.stack == "genie_redux":
        # Always delegate to the Hydra-enabled scripts so overrides are handled uniformly.
        script = "train_genie_redux.py" if args.action == "train" else "eval_genie_redux.py"
        script_path = str((Path(__file__).parent / script).resolve())
        # Ensure unbuffered output so prints/progress bars are visible immediately
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Read num_processes from Hydra config (train.num_processes or eval.num_processes)
        with initialize(version_base=None, config_path="configs"):
            cfg: DictConfig = compose(config_name="default", overrides=overrides)
        num_processes = 1
        if args.action == "train":
            num_processes = int(getattr(cfg.train, "num_processes", 1))
        elif args.action == "eval":
            # Prefer eval override; fallback to train if not present
            num_processes = int(getattr(cfg.eval, "num_processes", getattr(cfg.train, "num_processes", 1)))

        if num_processes and num_processes > 1:
            launch_cmd = [
                sys.executable,
                "-u",
                "-m",
                "accelerate.commands.launch",
                f"--num_processes={num_processes}",
                "--mixed_precision=bf16",
                script_path,
            ] + list(overrides or [])
            return subprocess.call(launch_cmd, env=env)
        # Single-process: run the script directly so its @hydra.main parses overrides
        return subprocess.call([sys.executable, "-u", script_path] + list(overrides or []), env=env)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
