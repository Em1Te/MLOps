import os
import sys
import json
import pickle

sys.path.append(os.getcwd())

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from src.loggers import get_logger
from src.stages.prepare_dataset import load_config


def test_model(config):
    logger = get_logger("TEST_MODEL")

    model_path = config["test"]["model_path"]
    test_path = config["test"]["testset_path"]
    metrics_path = config["test"]["metrics_path"]
    report_path = config["test"]["report_path"]

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    df_test = pd.read_csv(test_path)

    target_column = "Survived"

    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4)
    }

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    logger.info("Model evaluation finished")
    logger.info(metrics)


if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    test_model(config)