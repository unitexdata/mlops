import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

TRAIN_PATH = Path("data/processed/train_clean.csv")
TEST_PATH = Path("data/processed/test_clean.csv")
VECTORIZER_PATH = Path("models/vectorizer.pkl")

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

def transform(train_df, test_df):
    vectorizer = TfidfVectorizer(max_features=5000)

    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    y_train = train_df["label"]
    y_test = test_df["label"]

    return X_train, X_test, y_train, y_test, vectorizer

def save_vectorizer(vectorizer):
    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("Saved vectorizer")

if __name__ == "__main__":
    train_df, test_df = load_data()
    X_train, X_test, y_train, y_test, vectorizer = transform(train_df, test_df)
    save_vectorizer(vectorizer)
    print("Transformation done")
