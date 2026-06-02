
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(input_file, output_file):

    df = pd.read_csv(input_file)

    df["Age"] = df["Age"].fillna(
        df["Age"].median()
    )

    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

    df.drop(
        columns=[
            "PassengerId",
            "Name",
            "Ticket",
            "Cabin"
        ],
        inplace=True
    )

    encoder = LabelEncoder()

    df["Sex"] = encoder.fit_transform(
        df["Sex"]
    )

    df["Embarked"] = encoder.fit_transform(
        df["Embarked"]
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("Preprocessing selesai")


if __name__ == "__main__":

    preprocess(
        "../dataset_raw/train.csv",
        "titanic_preprocessed.csv"
    )
