import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from data_transformation import load_data, transform
import mlflow
import mlflow.sklearn

MODEL_PATH = Path("models/phishing_model.pkl")
MODEL_NAME = "phishing_detector"

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# def save_model(model):
#     MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
#     joblib.dump(model, MODEL_PATH)
#     print("Saved model")

if __name__ == "__main__":
    train_df, test_df = load_data()
    X_train, X_test, y_train, y_test, vectorizer = transform(train_df, test_df)
    mlflow.set_experiment("phishing_detection")

    with mlflow.start_run():
        model = train_model(X_train, y_train)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print("Model trained, logged, and registered in Model Registry.")

    # save_model(model)
