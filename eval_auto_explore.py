"""
eval_auto_explore.py: evaluation entrypoint for the AutoExplore agent.

What it does (briefly)
----------------------
- Loads a Hydra config for evaluation and resolves the checkpoint directory
  by model id (using CheckpointDirManager).
- Initializes Lightning Fabric (DDP, single device, bf16 mixed precision).
- Switches into the run directory, sets up a simple file structure, and dumps
  the resolved Hydra config for reproducibility.
- Instantiates `Trainer` in resume/eval mode, forces evaluation-style epoch
  semantics (start at epoch 1), enables per-epoch and final summaries,
  and calls `trainer.run()` to perform evaluation rollouts and logging.

Note: If running under SLURM, selected SLURM env vars are removed to avoid
conflicts with Lightning’s launcher.
"""

import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from tools.model_management import CheckpointDirManager
from auto_explore.src.utils import dump_hydra, FileStructure
from auto_explore.src.trainer import Trainer
import argparse

if "SLURM_NTASKS" in os.environ:
    # Remove SLURM env variables to avoid issues with Lightning
    del os.environ["SLURM_NTASKS"]
    del os.environ["SLURM_JOB_NAME"]
    
from lightning.fabric import Fabric
from tools.logger import getLogger
log = getLogger(__name__)

def run(cfg: DictConfig):
    """
    Orchestrate a single evaluation run using a Hydra config.

    Steps
    -----
    1) Normalize important paths in the config (root, world model root).
    2) Locate the checkpoint directory by resume_id with CheckpointDirManager.
    3) Initialize and launch Lightning Fabric (DDP, selected accelerator, bf16).
    4) Prepare run metadata: set W&B run name, mark resume=True, chdir to run dir.
    5) Create a FileStructure under the run dir and dump the resolved config there.
    6) Build a `Trainer` and adapt it for evaluation semantics:
       - start from epoch 1 (treat `common.epochs` as "how many eval epochs")
       - enable per-epoch + final evaluation summaries.
    7) Call `trainer.run()` to execute the evaluation loop.
    """
    cfg.common.root_dpath = os.path.abspath(cfg.common.root_dpath)
    cfg.world_model.root_dpath = os.path.abspath(cfg.world_model.root_dpath)

    model_id = cfg.common.resume_id
    cdm = CheckpointDirManager(cfg.common.root_dpath)
    dpath = cdm.get_dpath_by_id(model_id)
    
    fabric = Fabric(strategy="ddp", accelerator=cfg.common.device, devices=1, precision="bf16-mixed")
    fabric.launch()
    fabric.barrier()

    log.i("Evaluatin model id: ", model_id, dpath)

    fname = dpath.name
    cfg.wandb.name = fname + "_eval"
    cfg.common.resume=True
    dpath = Path(dpath)
    os.makedirs(dpath, exist_ok=True)
    os.chdir(dpath)

    log.i(f"Running experiment: {fname}")
    fs = FileStructure(dpath)
    fs.create()
    dump_hydra(cfg, fs.hydra_config_fpath)
    trainer = Trainer(cfg, fabric, fs)
    # In evaluation, interpret `common.epochs` as "how many eval epochs to run",
    # not as an absolute target epoch (avoid skipping to last trained epoch).
    try:
        trainer.start_epoch = 1
    except Exception:
        pass
    # Enable per-epoch and final evaluation summaries only for eval runs
    try:
        cfg.evaluation.print_summary = True
    except Exception:
        pass
    trainer.run()


@hydra.main(config_path="auto_explore/configs", config_name="evaluate")
def main(cfg: DictConfig):
    """
    Hydra entrypoint:
    - Parses the evaluation config from `auto_explore/configs/evaluate.yaml`.
    - Delegates to `run(cfg)` to perform the evaluation.
    """    
    run(cfg)


if __name__ == "__main__":
    main()
