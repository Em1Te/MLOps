import os
import sys

sys.path.append(os.getcwd())

from src.stages.prepare_dataset import load_config
from src.model_scripts.train import train


if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    train(config)