from dataclasses import dataclass
import hydra
from omegaconf import MISSING, DictConfig, OmegaConf

import train
import evaluate


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    config = cfg.config

    if config.mode == "train":
        train.run(config)

    elif config.mode == "eval":
        evaluate.run(config)
        
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


if __name__ == "__main__":
    main()
