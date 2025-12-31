import joblib
from pathlib import Path
from sklearn.metrics import classification_report
from data_transformation import load_data, transform
from sklearn.metrics import accuracy_score
import mlflow

MODEL_PATH = Path("models/phishing_model.pkl")

def load_model():
    return joblib.load(MODEL_PATH)

if __name__ == "__main__":
    train_df, test_df = load_data()
    X_train, X_test, y_train, y_test, vectorizer = transform(train_df, test_df)
    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    mlflow.set_experiment("phishing_detection")

    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)

        print("Model Evaluation Report:")
        print(classification_report(y_test, preds))

