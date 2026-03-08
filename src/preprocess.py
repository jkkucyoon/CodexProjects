from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COL = "Bankrupt?"


@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    feature_names: List[str]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load Kaggle company bankruptcy dataset from disk."""
    path = Path(csv_path)

    if not path.exists():
        fallback_candidates = [
            Path("data/company_bankruptcy.csv"),
            Path("data/data.csv"),
            Path("data/Company Bankruptcy Prediction.csv"),
        ]

        resolved = next((candidate for candidate in fallback_candidates if candidate.exists()), None)

        if resolved is None:
            csv_files = sorted(Path("data").glob("*.csv")) if Path("data").exists() else []
            if len(csv_files) == 1:
                resolved = csv_files[0]

        if resolved is None:
            raise FileNotFoundError(
                "Dataset not found. Checked requested path "
                f"'{path}' and common alternatives inside 'data/'.\n"
                "Download it from Kaggle and place the CSV as either:\n"
                "- data/company_bankruptcy.csv\n"
                "- data/data.csv"
            )

        path = resolved
        print(f"[INFO] Dataset not found at requested path. Using: {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from Kaggle and place the CSV there."
        )

    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataset.")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and sanity checks."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Replace inf values from ratio features.
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    cleaned[numeric_cols] = cleaned[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return cleaned


def split_features_target(df: pd.DataFrame) -> DatasetBundle:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return DatasetBundle(X=X, y=y, feature_names=X.columns.tolist())


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = feature_frame.select_dtypes(include=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_cols)],
        remainder="drop",
    )
    return preprocessor


def get_train_test_data(
    csv_path: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    from sklearn.model_selection import train_test_split

    df = load_dataset(csv_path)
    df = clean_dataset(df)
    bundle = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.2,
        random_state=42,
        stratify=bundle.y,
    )

    preprocessor = build_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor
