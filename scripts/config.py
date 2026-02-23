
import os
DATABASE_URL = os.getenv("DATABASE_URL")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REPORTS_DIR = "/app/reports"
MODELS_DIR = "/app/models"