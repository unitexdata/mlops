import joblib
import yaml
from pathlib import Path
from data_transformation import load_data, transform
from model_factory import get_model
import mlflow
import mlflow.sklearn

MODEL_PATH = Path("models/phishing_model.pkl")
MODEL_NAME = "phishing_detector"

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def train_model(X_train, y_train, model_cfg):
    model = get_model(model_cfg)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    cfg = load_config()
    model_cfg = cfg["model"]

    train_df, test_df = load_data()
    X_train, X_test, y_train, y_test, vectorizer = transform(train_df, test_df)

    mlflow.set_experiment("phishing_detection")

    with mlflow.start_run():
        model = train_model(X_train, y_train, model_cfg)

        mlflow.log_param("model_type", model_cfg["name"])

        for k, v in model_cfg.get(model_cfg["name"], {}).items():
            mlflow.log_param(k, v)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print(f"Model `{model_cfg['name']}` trained, logged, and saved.")
