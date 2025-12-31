import pandas as pd
from pathlib import Path

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

def validate_data(df):
    print("Checking data...")

    # Check empty dataframe
    if df.empty:
        raise ValueError("Dataframe is empty")

    # Check missing values
    if df.isnull().sum().sum() > 0:
        print("Warning: Missing values found")

    # Check required columns
    required_columns = ["text_combined", "label"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    print("Data validation passed")

if __name__ == "__main__":
    train_df, test_df = load_data()
    validate_data(train_df)
    validate_data(test_df)
