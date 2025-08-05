import argparse
import copy
import json
from generator.generator import EnvironmentDataGenerator


def run_env(config, connector_config):

    connector_class_name = connector_config["classname"]
    del connector_config["classname"]
    generator_config = connector_config["generator_config"]
    del connector_config["generator_config"]

    generator = EnvironmentDataGenerator(
        connector_class_name, connector_config, generator_config, config
    )
    generator.generate()
    print(f"Done with {generator.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data_gen.json")
    args = parser.parse_args()
    config = json.load(open(args.config, "r"))

    connector_configs = []
    if config["env"] == "coinrun":
        connector_config = config["connector_" + config["env"]]

        from generator.connector_coinrun import CoinRunConnector

        connector_config["classname"] = CoinRunConnector

        connector_configs.append(connector_config)
    else:
        raise ValueError(f"Unknown environment {config['env']}")

    run_env(config, connector_configs[0])

    print("Done")
