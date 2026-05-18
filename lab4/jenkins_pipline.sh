# Лабораторная №3. CI/CD Jenkins
# Проект: Titanic ML pipeline
#
# Предполагаемая структура репозитория:
# MLOPS/lab3/
# ├── download.py
# ├── train_model.py
# ├── requirements.txt
# └── jenkins_pipline.sh


# ============================================================
# №1. download
# ============================================================

python3 -m venv ./my_env
. ./my_env/bin/activate

cd ./MLOPS/lab3

python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip setuptools wheel
pip3 install -r requirements.txt

python3 download.py


# ============================================================
# №2. train_model
# ============================================================

echo "Start train model"

cd /var/lib/jenkins/workspace/download/
. ./my_env/bin/activate

cd ./MLOPS/lab3

python3 train_model.py > best_model.txt


# ============================================================
# №3. deploy
# ============================================================

cd /var/lib/jenkins/workspace/download/
. ./my_env/bin/activate

cd ./MLOPS/lab3

export BUILD_ID=dontKillMe
export JENKINS_NODE_COOKIE=dontKillMe

path_model=$(cat best_model.txt)

# Остановить старый сервис на 5003, если он уже был запущен.
fuser -k 5003/tcp || true

mlflow models serve -m $path_model -p 5003 --host 0.0.0.0 --no-conda &


# ============================================================
# №4. healthy
# ============================================================

curl http://127.0.0.1:5003/invocations \
-H "Content-Type: application/json" \
--data '{
  "dataframe_split": {
    "columns": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone"],
    "data": [[3, 1, 22.0, 1, 0, 7.25, 2, 2, 0]]
  }
}'


# ============================================================
# Pipeline для объединения freestyle jobs
# ============================================================

pipeline {
    agent any

    stages {
        stage('Download') {
            steps {
                build job: 'download'
            }
        }

        stage('Train') {
            steps {
                build job: 'train_model'
            }
        }

        stage('Deploy') {
            steps {
                build job: 'deploy'
            }
        }

        stage('Status') {
            steps {
                build job: 'healthy'
            }
        }
    }
}