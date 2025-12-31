Project Overview

This project is an end-to-end MLOps implementation of a phishing email detection system.
It demonstrates how to convert a traditional machine learning project into a reproducible, versioned, and production-ready pipeline using modern MLOps tools.

The system trains a machine learning model to classify whether an email is phishing or legitimate, and integrates:

Data versioning with DVC

Experiment tracking with MLflow

Pipeline orchestration with DVC stages

Containerized inference with FastAPI + Docker

CI/CD automation with GitHub Actions

Cloud artifact storage using AWS S3

🎯 Objectives

Build a phishing email classification model.
Ensure reproducibility and traceability of data, code, and experiments.
Enable automated retraining and deployment.
Follow professional MLOps practices used in industry.

🧱 Tech Stack
Layer	Tool
Programming	Python 3.10
ML	Scikit-learn
Data	Pandas
Experiment Tracking	MLflow
Data Versioning	DVC
Pipelines	DVC Pipeline
API	FastAPI
Containerization	Docker
CI/CD	GitHub Actions
Cloud	AWS S3
Project Structure

mini-project/
├── data/
│   ├── raw/phishing_email.csv
│   └── processed/
│       ├── train.csv
│       ├── test.csv
│       ├── train_clean.csv
│       └── test_clean.csv
├── models/
│   ├── phishing_model.pkl
│   └── vectorizer.pkl
├── src/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_preprocessing.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   └── app.py
├── dvc.yaml
├── dvc.lock
├── mlflow.db
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
├── .env
└── README.md


🔄 MLOps Workflow

Data versioning with DVC

Preprocessing pipeline (train/test split)

Model training with TF-IDF + Logistic Regression

Experiment tracking with MLflow

Pipeline orchestration with DVC

Inference API with FastAPI

Containerization with Docker

CI/CD automation using GitHub Actions

⚙️ Setup Instructions
1️⃣ Clone repository
git clone <repo-url>
cd mini-project

2️⃣ Create virtual environment
python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

📦 Data Versioning with DVC

Add dataset:

dvc add data/raw/phishing_email.csv
git add data/raw/phishing_email.csv.dvc .gitignore
git commit -m "Track dataset with DVC"


Run pipeline:

dvc repro

📊 Experiment Tracking

Start MLflow UI:

mlflow ui


Access: http://localhost:5000

MLflow tracks:

Parameters

Metrics

Model artifacts

🧪 Training Pipeline
dvc repro


This runs:

preprocess stage → creates train/test splits

train stage → trains and logs model

🚀 Inference API

Start API:

uvicorn src.inference:app --reload


Test prediction:

curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"email_text":"Your account has been suspended"}'

🐳 Docker

Build image:

docker build -t phishing-mlops .


Run container:

docker run -p 8000:8000 phishing-mlops

🔁 CI/CD Pipeline

CI/CD is implemented using GitHub Actions.

On every push to main:

Pulls data using DVC

Runs training pipeline

Builds Docker image

Pushes image to registry

Workflow file: .github/workflows/mlops.yml

☁️ Cloud Integration

Artifacts and data are stored in AWS S3 using:

dvc remote add -d s3remote s3://<bucket-name>
dvc push

🎤 Interview Summary

“This project demonstrates converting a traditional ML model into a production-ready MLOps pipeline using Git for versioning, DVC for data and pipeline tracking, MLflow for experiments, Docker for deployment, and GitHub Actions for CI/CD automation.”

✅ Outcomes

Reproducible ML training

Traceable experiments

Automated pipelines

Cloud-ready deployment

Professional MLOps implementation

📜 License

MIT License
If you want, I can:
Add monitoring & drift detection
Deploy to AWS ECS/EKS
Add model registry approval flow