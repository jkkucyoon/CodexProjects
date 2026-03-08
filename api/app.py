from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("model/model.joblib")
RISK_THRESHOLD = 0.5

app = FastAPI(title="Supplier Risk Prediction API", version="1.0.0")


class PredictionResponse(BaseModel):
    risk_prediction: str
    probability: float


@app.on_event("startup")
def load_model() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "model/model.joblib was not found. Train the model first using src/train.py."
        )
    app.state.model = joblib.load(MODEL_PATH)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: Dict[str, Any]):
    try:
        frame = pd.DataFrame([features])
        proba = float(app.state.model.predict_proba(frame)[0, 1])
        risk = "high" if proba >= RISK_THRESHOLD else "low"
        return PredictionResponse(risk_prediction=risk, probability=round(proba, 4))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
