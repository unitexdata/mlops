import pandas as pd 
from pathlib import Path
from sklearn.model_selection import train_test_split

path = Path("data/raw/phishing_email.csv")
PROCESSED_PATH = Path("data/processed")

def data_ingestion(path=path):
    df = pd.read_csv(path)
    return df

def save_data(df, test_size=0.2, random_state=42):
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )

    train_path = PROCESSED_PATH / "train.csv"
    test_path = PROCESSED_PATH / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train data to {train_path}")
    print(f"Saved test data to {test_path}")

if __name__ == "__main__":
    data = data_ingestion()
    print(data.head())
    save_data(data)
