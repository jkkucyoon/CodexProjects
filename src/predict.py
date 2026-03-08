from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


RISK_THRESHOLD = 0.5


def predict_single(model_path: str, input_json_path: str) -> dict:
    model = joblib.load(model_path)

    raw = json.loads(Path(input_json_path).read_text())
    features = pd.DataFrame([raw])

    proba = float(model.predict_proba(features)[0, 1])
    label = "high" if proba >= RISK_THRESHOLD else "low"

    return {"risk_prediction": label, "probability": round(proba, 4)}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict company risk from JSON features")
    parser.add_argument("--model-path", default="model/model.joblib")
    parser.add_argument("--input-json", required=True, help="Path to input JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict_single(args.model_path, args.input_json)
    print(json.dumps(result, indent=2))
