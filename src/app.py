from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_URI = os.getenv("MODEL_URI", "models:/phishing_detector/Production")

app = FastAPI()

print(f"Loading model from: {MODEL_URI}")
model = mlflow.pyfunc.load_model(MODEL_URI)

class EmailRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model")
def model_info():
    return {"model_uri": MODEL_URI}

@app.post("/predict")
def predict(request: EmailRequest):
    preds = model.predict([request.text])
    return {"prediction": int(preds[0])}
