import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature

np.random.seed(42)

RED_URL = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-red.csv"
WHITE_URL = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-white.csv"

# Настраиваем хранилище
mlflow.set_tracking_uri("sqlite:///mlruns.db")
EXPERIMENT_NAME = "Wine_Quality_XGBoost"
mlflow.set_experiment(EXPERIMENT_NAME)

# Загрузка данных, объединение и очистка
def load_wine_data():
    red = pd.read_csv(RED_URL, sep=";")
    red["wine_type"] = "red"

    white = pd.read_csv(WHITE_URL, sep=";")
    white["wine_type"] = "white"

    df = pd.concat([red, white], ignore_index=True).drop_duplicates()

    rename_map = {
        "fixed acidity": "fixed_acidity",
        "volatile acidity": "volatile_acidity",
        "citric acid": "citric_acid",
        "residual sugar": "residual_sugar",
        "free sulfur dioxide": "free_sulfur_dioxide",
        "total sulfur dioxide": "total_sulfur_dioxide",
        "pH": "ph",
    }
    df = df.rename(columns=rename_map)
    return df

# Метрики
def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


df = load_wine_data()

X = df.drop(columns=["quality"])
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

try:
    ohe = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse=False)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", ohe, ["wine_type"]),
    ],
    remainder="passthrough"
)

# Конфигурации экспериментов
experiments = [
    {
        "learning_rate": 0.03,
        "n_estimators": 1200,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
    },
    {
        "learning_rate": 0.05,
        "n_estimators": 900,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2,
        "reg_alpha": 0.0,
        "reg_lambda": 1.5,
        "gamma": 0.0,
    },
    {
        "learning_rate": 0.02,
        "n_estimators": 1800,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.0,
    },
    {
        "learning_rate": 0.07,
        "n_estimators": 700,
        "max_depth": 3,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
        "gamma": 0.0,
    },
]

for i, params in enumerate(experiments, start=1):
    with mlflow.start_run(run_name=f"XGB_Run_{i}"):

        model = Pipeline(steps=[
            ("preprocess", preprocess),
            ("regressor", XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                tree_method="hist",
                n_jobs=-1,
                **params
            ))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse, mae, r2 = eval_metrics(y_test, y_pred)

        mlflow.log_param("model", "XGBRegressor")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("tree_method", "hist")
        for k, v in params.items():
            mlflow.log_param(k, v)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        input_example = X_train.iloc[:5]
        signature = infer_signature(input_example, model.predict(input_example))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            signature=signature
        )

        print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])

best = runs.sort_values(
    by=["metrics.rmse", "metrics.r2"],
    ascending=[True, False]
).iloc[0]

print("\nЛучший запуск:")
cols_to_show = [
    "run_id",
    "metrics.rmse",
    "metrics.mae",
    "metrics.r2",
    "params.learning_rate",
    "params.n_estimators",
    "params.max_depth",
    "params.subsample",
    "params.colsample_bytree",
    "params.min_child_weight",
    "params.reg_lambda",
]
print(best[cols_to_show])

best_run_id = best["run_id"]
model_uri = f"runs:/{best_run_id}/model"

loaded_model = mlflow.sklearn.load_model(model_uri)

sample = X_test.iloc[[0]]
pred = loaded_model.predict(sample)