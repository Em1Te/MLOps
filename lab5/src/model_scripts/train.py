import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.loggers import get_logger


def train(config):
    logger = get_logger("TRAIN")

    train_path = config["data_split"]["trainset_path"]
    test_path = config["data_split"]["testset_path"]

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    target_column = "Survived"

    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]

    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]

    numeric_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone",
        "FarePerPerson"
    ]

    categorical_features = [
        "Sex",
        "Embarked",
        "Title",
        "AgeGroup"
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = LogisticRegression(
        max_iter=config["train"]["max_iter"],
        random_state=config["data_split"]["random_state"]
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    params = {
        "model__C": config["train"]["C"]
    }

    clf = GridSearchCV(
        pipe,
        params,
        cv=config["train"]["cv"],
        n_jobs=-1
    )

    logger.info("Start model training")

    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Best params: {clf.best_params_}")
    logger.info(f"Test accuracy: {accuracy}")

    model_path = config["train"]["model_path"]

    with open(model_path, "wb") as file:
        pickle.dump(best_model, file)

    logger.info(f"Model saved to {model_path}")