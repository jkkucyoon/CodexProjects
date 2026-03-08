from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from preprocess import get_train_test_data


def build_models(random_state: int = 42):
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def train_and_select_best(data_path: str, model_dir: str):
    X_train, X_test, y_train, y_test, preprocessor = get_train_test_data(data_path)
    models = build_models()

    rows = []
    trained_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        probs = pipeline.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)

        rows.append({"model": name, "roc_auc": auc, "f1": f1})
        trained_pipelines[name] = pipeline

    results = pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False)
    best_name = results.iloc[0]["model"]
    best_pipeline = trained_pipelines[best_name]

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.csv"

    joblib.dump(best_pipeline, model_path)
    results.to_csv(metrics_path, index=False)

    print("Model comparison:")
    print(results.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train supplier/company risk model")
    parser.add_argument(
        "--data-path",
        default="data/company_bankruptcy.csv",
        help="Path to Kaggle dataset CSV",
    )
    parser.add_argument("--model-dir", default="model", help="Directory to save model artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_select_best(args.data_path, args.model_dir)
