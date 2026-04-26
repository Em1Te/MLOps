import pandas as pd
import numpy as np
import os
import glob
import re
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
from airflow import DAG
from airflow.operators.python import PythonOperator
import kagglehub

from dir_settings import WORK_DIR, RAW_DATA_DIR, CLEAR_DATA_DIR, MODEL_DIR
from train_model import train_model

os.makedirs(WORK_DIR, exist_ok=True)


def download_data(**kwargs):
    dataset_path = kagglehub.dataset_download("syedaeman2212/airline-ticket-prices-dataset")
    csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("CSV не найден")
    csv_file = csv_files[0]
    df = pd.read_csv(csv_file)
    df.to_csv(RAW_DATA_DIR, index=False)
    return True


def clear_data(**kwargs):
    df = pd.read_csv(RAW_DATA_DIR)

    target = 'Price_USD'
    cat_columns = ['Airline', 'Origin', 'Destination', 'Class']
    numeric_columns = ['Distance_km', 'Days_Before_Departure']

    initial_len = len(df)
    df = df.drop_duplicates()

    if 'Ticket_ID' in df.columns:
        df = df.drop(columns=['Ticket_ID'])
        print("Удалена колонка Ticket_ID")

    existing_cat = [col for col in cat_columns if col in df.columns]
    if existing_cat:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[existing_cat] = encoder.fit_transform(df[existing_cat])

    important_cols = existing_cat + numeric_columns + [target]
    important_cols = [col for col in important_cols if col in df.columns]
    initial_len = len(df)
    df = df.dropna(subset=important_cols)

    df.to_csv(CLEAR_DATA_DIR, index=False)
    print(f"Очищенные данные сохранены в {CLEAR_DATA_DIR}")
    return True


default_args = {
    'owner': 'your_name',
    'start_date': datetime(2026, 3, 22),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id="airline_price_pipeline",
    default_args=default_args,
    schedule=timedelta(days=1),
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(
    task_id="download_data",
    python_callable=download_data,
    dag=dag
)

clear_task = PythonOperator(
    task_id="clear_data",
    python_callable=clear_data,
    dag=dag
)

train_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag
)

download_task >> clear_task >> train_task