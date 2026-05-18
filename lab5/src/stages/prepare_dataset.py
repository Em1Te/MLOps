import os
import sys
import yaml
import pandas as pd

sys.path.append(os.getcwd())

from src.loggers import get_logger


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as conf_file:
        config = yaml.safe_load(conf_file)

    return config


def clear_data(path2data):
    logger = get_logger("CLEAR_DATA")
    logger.info("Start data cleaning")

    df = pd.read_csv(path2data)

    useful_columns = [
        "Survived",
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]

    df = df[useful_columns].copy()

    df = df.drop_duplicates()

    df = df.dropna(subset=["Survived"])

    df["Survived"] = df["Survived"].astype(int)
    df["Pclass"] = df["Pclass"].astype(int)

    df["Sex"] = df["Sex"].astype(str).str.strip().str.lower()
    df["Embarked"] = df["Embarked"].astype(str).str.strip().str.upper()

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df["Embarked"] = df["Embarked"].replace("NAN", pd.NA)

    most_common_embarked = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(most_common_embarked)

    df = df[df["Age"].between(0, 100)]
    df = df[df["Fare"] >= 0]

    df = df.reset_index(drop=True)

    logger.info(f"Data cleaning finished. Shape: {df.shape}")

    return df


def extract_title(name):
    if "," not in str(name) or "." not in str(name):
        return "Unknown"

    title = str(name).split(",")[1].split(".")[0].strip()

    common_titles = ["Mr", "Mrs", "Miss", "Master"]

    if title in common_titles:
        return title

    return "Rare"


def age_group(age):
    if age < 13:
        return "child"
    elif age < 18:
        return "teen"
    elif age < 60:
        return "adult"
    else:
        return "senior"


def featurize(dframe, config):
    logger = get_logger("FEATURIZE")
    logger.info("Start feature generation")

    df = dframe.copy()

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["Title"] = df["Name"].apply(extract_title)
    df["AgeGroup"] = df["Age"].apply(age_group)

    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    df = df.drop(columns=["Name"])

    cleaned_path = config["prepare"]["cleaned_path"]
    features_path = config["featurize"]["features_path"]

    df.to_csv(features_path, index=False)

    dframe.to_csv(cleaned_path, index=False)

    logger.info(f"Cleaned data saved to {cleaned_path}")
    logger.info(f"Final dataset saved to {features_path}")
    logger.info(f"Final shape: {df.shape}")


if __name__ == "__main__":
    config = load_config("./src/config.yaml")

    df_prep = clear_data(config["data_load"]["dataset_csv"])

    featurize(df_prep, config)