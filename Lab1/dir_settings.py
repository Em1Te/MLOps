import os

WORK_DIR = os.path.expanduser('~/airflow/data')

RAW_DATA_DIR = os.path.join(WORK_DIR, 'raw.csv')
CLEAR_DATA_DIR = os.path.join(WORK_DIR, 'clear.csv')
MODEL_DIR = os.path.join(WORK_DIR, 'model.pkl')