import pandas as pd
from pathlib import Path
import re

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
PROCESSED_PATH = Path("data/processed")

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def preprocess(df):
    df["clean_text"] = df["text_combined"].apply(clean_text)
    return df

def save_data(train_df, test_df):
    train_df.to_csv(PROCESSED_PATH / "train_clean.csv", index=False)
    test_df.to_csv(PROCESSED_PATH / "test_clean.csv", index=False)
    print("Saved cleaned train and test data")

if __name__ == "__main__":
    train_df, test_df = load_data()
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    save_data(train_df, test_df)
