import os
import sys

sys.path.append(os.getcwd())

import pandas as pd
from sklearn.model_selection import train_test_split

from src.loggers import get_logger
from src.stages.prepare_dataset import load_config


def data_split(config):
    logger = get_logger("DATA_SPLIT")

    data_frame = pd.read_csv(config["featurize"]["features_path"])

    train_dataset, test_dataset = train_test_split(
        data_frame,
        test_size=config["data_split"]["test_size"],
        random_state=config["data_split"]["random_state"],
        stratify=data_frame["Survived"]
    )

    logger.info("Save train and test sets")

    train_csv_path = config["data_split"]["trainset_path"]
    test_csv_path = config["data_split"]["testset_path"]

    train_dataset.to_csv(train_csv_path, index=False)
    test_dataset.to_csv(test_csv_path, index=False)

    logger.info(f"Train shape: {train_dataset.shape}")
    logger.info(f"Test shape: {test_dataset.shape}")


if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    data_split(config)