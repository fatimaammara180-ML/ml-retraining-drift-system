import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from scripts.config import (
    REFERENCE_DATA_PATH, LATEST_MODEL_PATH,
    MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODEL_NAME
)
from scripts.utils import load_csv, save_joblib
from scripts.schema import TARGET_COL, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES

def build_pipeline():
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])

def main():
    df = load_csv(REFERENCE_DATA_PATH)

    missing = [c for c in ALL_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in reference data: {missing}")

    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()

    # Split BEFORE fitting to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Initial model accuracy: {acc:.4f}")

    # Save locally
    save_joblib(pipe, LATEST_MODEL_PATH)

    # MLflow logging
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="initial_train"):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

if __name__ == "__main__":
    main()