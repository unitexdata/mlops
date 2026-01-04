import joblib
import yaml
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from data_transformation import load_data, transform
import mlflow

MODEL_PATH = Path("models/phishing_model.pkl")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_config()
    model_cfg = cfg["model"]

    train_df, test_df = load_data()
    X_train, X_test, y_train, y_test, vectorizer = transform(train_df, test_df)

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.set_experiment("phishing_detection")

    with mlflow.start_run():
        mlflow.log_param("model_type", model_cfg["name"])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print("\nModel Evaluation Metrics")
        # print("-------------------------")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}\n")

        print("Detailed Classification Report:")
        print(classification_report(y_test, preds))
