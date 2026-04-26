import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
import joblib
from dir_settings import WORK_DIR, CLEAR_DATA_DIR, MODEL_DIR


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_model(**kwargs):
    df = pd.read_csv(CLEAR_DATA_DIR)
    print(f"Загружены данные: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")

    target = 'Price_USD'
    cat_columns = ['Airline', 'Origin', 'Destination', 'Class']
    num_columns = ['Distance_km', 'Days_Before_Departure']

    existing_cat = [col for col in cat_columns if col in df.columns]
    existing_num = [col for col in num_columns if col in df.columns]

    print(f"Категориальные признаки: {existing_cat}")
    print(f"Числовые признаки: {existing_num}")

    X = df[existing_cat + existing_num]
    y = df[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), existing_num),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), existing_cat)
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    param_grid = {
        'regressor__n_estimators': [50, 100, 150],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.05, 0.1, 0.2],
        'regressor__min_samples_split': [2, 5]
    }

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])

    mlflow_tracking_uri = "file://" + WORK_DIR + "/mlruns"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("airline_price_prediction")

    with mlflow.start_run():
        print("Поиск лучших гиперпараметров")
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"\nЛучшие параметры: {best_params}")

        y_pred = best_model.predict(X_val)
        rmse, mae, r2 = eval_metrics(y_val, y_pred)

        print(f"\nРезультаты на валидации:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R2: {r2:.4f}")

        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name.replace('regressor__', ''), param_value)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "gradient_boosting_model", signature=signature)

        with open(MODEL_DIR, "wb") as file:
            joblib.dump(best_model, file)

if __name__ == "__main__":
    train_model()