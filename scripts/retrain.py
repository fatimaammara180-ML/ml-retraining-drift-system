import os
import shutil
from datetime import datetime

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from scripts.train import build_pipeline
from scripts.config import (
    REFERENCE_DATA_PATH, CURRENT_DATA_PATH,
    LATEST_MODEL_PATH, MODELS_DIR,
    MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODEL_NAME
)
from scripts.utils import load_csv, save_joblib, load_joblib, ensure_dir
from scripts.schema import TARGET_COL, ALL_FEATURES

MIN_IMPROVEMENT = 0.000  # require >= old accuracy (you can set 0.01 etc.)

def main():
    ref = load_csv(REFERENCE_DATA_PATH)
    cur = load_csv(CURRENT_DATA_PATH)

    data = pd.concat([ref, cur], ignore_index=True)

    missing = [c for c in ALL_FEATURES + [TARGET_COL] if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns for retraining: {missing}")

    X = data[ALL_FEATURES].copy()
    y = data[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    new_pipe = build_pipeline()
    new_pipe.fit(X_train, y_train)
    new_acc = accuracy_score(y_test, new_pipe.predict(X_test))
    print(f"New model accuracy: {new_acc:.4f}")

    # Compare with old model if exists
    old_acc = None
    if os.path.exists(LATEST_MODEL_PATH):
        old_pipe = load_joblib(LATEST_MODEL_PATH)
        old_acc = accuracy_score(y_test, old_pipe.predict(X_test))
        print(f"Old model accuracy (same test split): {old_acc:.4f}")

    if old_acc is not None and (new_acc < old_acc + MIN_IMPROVEMENT):
        print("Not deploying new model (no improvement). Logging run only.")
    else:
        # Archive old model
        ensure_dir(os.path.join(MODELS_DIR, "archive"))
        if os.path.exists(LATEST_MODEL_PATH):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived = os.path.join(MODELS_DIR, "archive", f"latest_model_{ts}.pkl")
            shutil.move(LATEST_MODEL_PATH, archived)
            print(f"Archived old model to: {archived}")

        save_joblib(new_pipe, LATEST_MODEL_PATH)
        print(f"Deployed new model to: {LATEST_MODEL_PATH}")

    # MLflow logging
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="retrain"):
        mlflow.log_metric("new_accuracy", new_acc)
        if old_acc is not None:
            mlflow.log_metric("old_accuracy", old_acc)
        mlflow.sklearn.log_model(
            sk_model=new_pipe,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

if __name__ == "__main__":
    main()