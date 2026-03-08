from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from zipfile import ZipFile

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


def _extract_csv_from_zip(zip_path: Path, target_dir: Path) -> Path | None:
    """Extract first CSV from a ZIP archive into target_dir and return its path."""
    try:
        with ZipFile(zip_path) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_members:
                return None

            member = csv_members[0]
            archive.extract(member, path=target_dir)
            return target_dir / member
    except Exception:
        return None


def _resolve_dataset_path(requested_path: Path) -> Path:
    """Resolve dataset path from common Kaggle layouts and filenames."""
    if requested_path.exists():
        return requested_path

    data_dir = Path("data")
    fallback_candidates = [
        data_dir / "company_bankruptcy.csv",
        data_dir / "data.csv",
        data_dir / "Company Bankruptcy Prediction.csv",
        data_dir / "company-bankruptcy-prediction" / "data.csv",
        data_dir / "Company Bankruptcy Prediction" / "data.csv",
    ]

    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate

    if data_dir.exists():
        recursive_csv_files = sorted(data_dir.glob("**/*.csv"))
        if len(recursive_csv_files) == 1:
            return recursive_csv_files[0]

        if len(recursive_csv_files) > 1:
            preferred = [p for p in recursive_csv_files if p.name.lower() in {"company_bankruptcy.csv", "data.csv"}]
            if preferred:
                return preferred[0]

        zip_files = sorted(data_dir.glob("**/*.zip"))
        for zip_path in zip_files:
            extracted = _extract_csv_from_zip(zip_path, data_dir)
            if extracted is not None and extracted.exists():
                print(f"[INFO] Extracted dataset from ZIP: {zip_path} -> {extracted}")
                return extracted

    inspected_data_files = sorted([str(p) for p in data_dir.glob("**/*")]) if data_dir.exists() else []
    inspected_preview = "\n".join(inspected_data_files[:20])
    if len(inspected_data_files) > 20:
        inspected_preview += "\n..."

    raise FileNotFoundError(
        "Dataset not found. Checked requested path "
        f"'{requested_path}' and common alternatives inside 'data/'.\n"
        "Accepted locations include:\n"
        "- data/company_bankruptcy.csv\n"
        "- data/data.csv\n"
        "- data/<subfolder>/data.csv\n\n"
        "Current files under data/ (first 20):\n"
        f"{inspected_preview if inspected_preview else '[data directory missing or empty]'}"
    )


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load Kaggle company bankruptcy dataset from disk."""
    requested_path = Path(csv_path)
    resolved_path = _resolve_dataset_path(requested_path)

    if resolved_path != requested_path:
        print(f"[INFO] Dataset not found at requested path. Using: {resolved_path}")

    df = pd.read_csv(resolved_path)
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COL}' in dataset loaded from '{resolved_path}'."
        )
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
