import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
RAW_DATA_PATH = "titanic_raw.csv"
CLEAR_DATA_PATH = "df_clear.csv"


def download_data():
    df = pd.read_csv(DATA_URL)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Downloaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clear_data(path2df):
    df = pd.read_csv(path2df)

    drop_columns = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=drop_columns)

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    cat_columns = ["Sex", "Embarked"]
    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df.to_csv(CLEAR_DATA_PATH, index=False
    return True


if __name__ == "__main__":
    download_data()
    clear_data(RAW_DATA_PATH)