# 📌 MLOps Drift Detection & Auto-Retraining System

A production-style machine learning monitoring pipeline that automatically detects data drift and triggers model retraining when input distributions shift.

This project demonstrates end-to-end MLOps practices including model serving, drift monitoring, experiment tracking, and multi-container orchestration.

---

## 🚀 Architecture Overview

The system consists of 4 orchestrated services:

| Service | Role |
|---|---|
| FastAPI | Serves real-time ML predictions |
| PostgreSQL | Stores reference and production data |
| Evidently AI | Detects data drift statistically |
| MLflow | Tracks experiments and model versions |

**Workflow:**
1. Reference dataset stored in PostgreSQL
2. New production data inserted into `new_data` table
3. Drift detection compares distributions using KS test
4. If drift detected → retraining pipeline runs automatically
5. New model logged and versioned in MLflow
6. API serves updated model for predictions

---

## 🧠 Key Features

- Statistical drift detection using Kolmogorov–Smirnov test
- Automated HTML drift report generation via Evidently AI
- Experiment tracking and model versioning with MLflow
- REST API for real-time predictions with probability scores
- Fully containerised multi-service setup via Docker Compose
- Schema validation and null-check safeguards before drift runs
- Graceful 503 handling when model is not yet loaded
- One-command deployment: `docker compose up --build`

---

## 🛠️ Tech Stack

`Python` `FastAPI` `Evidently AI` `MLflow` `PostgreSQL` `scikit-learn` `Docker` `Pydantic` `SQLAlchemy` `Uvicorn`

---

## 📂 Project Structure
```
ml-retraining-drift-system/
│
├── api/
│   ├── main.py            # FastAPI app + predict endpoint
│   ├── Dockerfile
│   └── requirements.txt
│
├── scripts/
│   ├── drift_check.py     # Evidently drift detection
│   ├── train.py           # Initial model training
│   ├── retrain.py         # Auto-retraining on drift
│   ├── utils.py           # DB loaders, model helpers
│   ├── schema.py          # Feature definitions
│   └── config.py          # Paths and env config
│
├── db/
│   └── init.sql           # Table definitions
│
├── docker-compose.yml
└── .env
```

---

## ⚙️ How to Run Locally

### Prerequisites
- Docker Desktop installed and running
- Git

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/fatimaammara180-ML/ml-retraining-drift-system.git
cd ml-retraining-drift-system
```

**2. Configure environment**
```bash
cp .env.example .env
```

**3. Start all services**
```bash
docker compose up --build
```

### Services

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check + model loaded status |
| GET | `/health` | Returns 503 if model not yet loaded |
| POST | `/predict` | Run inference and return probability |

**Example Request:**
```json
POST /predict
{
  "feature1": 1.5,
  "feature2": 45.0,
  "feature3": 22.3
}
```

**Example Response:**
```json
{
  "prediction": 1,
  "label": "default",
  "default_probability": 0.7842
}
```

---

## 📊 How Drift Detection Works

1. Reference data loaded from `reference_data` table in PostgreSQL
2. New production data loaded from `new_data` table
3. Null and row-count validation runs before Evidently is called
4. Evidently runs the Kolmogorov-Smirnov test per feature column
5. Dataset drift flagged if drifted share exceeds **0.5 threshold**
6. Interactive HTML report saved to `/reports/drift_report.html`

**To view the drift report:**
```bash
docker cp worker:/app/reports/drift_report.html ./reports/drift_report.html
```
Then open `reports/drift_report.html` in your browser.

---

## 🔁 Retraining Logic

When drift is detected:
- Load latest reference data from PostgreSQL
- Preprocess and validate feature columns
- Train new scikit-learn model on updated data
- Log metrics and model artifact to MLflow
- Save updated model to `/models/latest_model.pkl`
- API automatically serves the retrained model

---

## 📌 Lessons Learned

- Monitoring is as critical as model accuracy in production systems
- Schema consistency and null validation prevent silent production failures
- Container orchestration makes ML systems reproducible across environments
- Real-world ML pipelines require validation and safeguards at every step

---

## 🧩 Future Improvements

- Add automated scheduler (Airflow / Prefect / APScheduler)
- Add Prometheus + Grafana for real-time monitoring dashboards
- Add CI/CD pipeline with GitHub Actions
- Add model performance drift tracking alongside data drift
- Deploy to cloud infrastructure (Render / AWS ECS / GCP Cloud Run)

---

## 👩‍💻 Author

**Fatima Ammara** — AI/ML Student focused on MLOps & Production ML Systems

[GitHub](https://github.com/fatimaammara180-ML)
