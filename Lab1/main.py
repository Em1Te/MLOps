from airflow_pipe import download_data, clear_data
from train_model import train_model

# Скачать данные
download_data()

# Очистить данные
clear_data()

# Обучить модель
train_model()