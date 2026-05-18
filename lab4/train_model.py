import sys
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = "df_clear.csv"
TARGET_COLUMN = "Survived"
EXPERIMENT_NAME = "titanic classification"


def print_log(message):
    print(message, file=sys.stderr)


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, zero_division=0)
    recall = recall_score(actual, pred, zero_division=0)
    f1 = f1_score(actual, pred, zero_division=0)
    return accuracy, precision, recall, f1


def get_models():
    models = {
        "logistic_regression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42))
            ]),
            "params": {
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["liblinear"]
            }
        },
        "sgd_classifier": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SGDClassifier(random_state=42))
            ]),
            "params": {
                "model__loss": ["hinge", "log_loss"],
                "model__alpha": [0.0001, 0.001, 0.01],
                "model__penalty": ["l2", "elasticnet"]
            }
        },
        "random_forest": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(random_state=42))
            ]),
            "params": {
                "model__n_estimators": [50, 100],
                "model__max_depth": [3, 5, None],
                "model__min_samples_split": [2, 5]
            }
        }
    }

    return models


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    best_f1 = -1
    best_model_name = None
    best_model_path = None
    best_model_object = None

    models = get_models()

    for model_name, model_config in models.items():
        print_log(f"Start training: {model_name}")

        grid = GridSearchCV(
            estimator=model_config["pipeline"],
            param_grid=model_config["params"],
            cv=3,
            scoring="f1",
            n_jobs=2
        )

        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        y_pred = best_estimator.predict(X_val)

        accuracy, precision, recall, f1 = eval_metrics(y_val, y_pred)

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(grid.best_params_)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            signature = infer_signature(X_train, best_estimator.predict(X_train))

            mlflow.sklearn.log_model(
                sk_model=best_estimator,
                artifact_path="model",
                signature=signature
            )

            model_path = mlflow.get_artifact_uri("model").replace("file://", "")

        print_log(f"Model: {model_name}")
        print_log(f"Best params: {grid.best_params_}")
        print_log(f"accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
        print_log(f"MLflow model path: {model_path}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name
            best_model_path = model_path
            best_model_object = best_estimator

    joblib.dump(best_model_object, "best_titanic_model.pkl")

    print_log(f"Best model: {best_model_name}")
    print_log(f"Best f1: {best_f1:.4f}")
    print_log("Saved pickle model: best_titanic_model.pkl")


    print(best_model_path)
