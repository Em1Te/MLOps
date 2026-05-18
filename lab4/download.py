import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
RAW_DATA_PATH = "titanic_raw.csv"
CLEAR_DATA_PATH = "df_clear.csv"


def download_data():
    df = pd.read_csv(DATA_URL)
    df.to_csv(RAW_DATA_PATH, index=False)

    print("Dataset downloaded")
    print("Raw dataset shape:", df.shape)

    return df


def clear_data(path2df):
    df = pd.read_csv(path2df)

    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    cat_columns = ["Sex", "Embarked"]

    encoder = OrdinalEncoder()
    df[cat_columns] = encoder.fit_transform(df[cat_columns])

    # Удаляем дубликаты
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df.to_csv(CLEAR_DATA_PATH, index=False)

    print("Clear dataset saved")
    print("Clear dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    return True


if __name__ == "__main__":
    download_data()
    clear_data(RAW_DATA_PATH)