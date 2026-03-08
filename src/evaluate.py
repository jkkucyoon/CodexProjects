from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score

from preprocess import get_train_test_data


def evaluate(model_path: str, data_path: str, output_dir: str = "model") -> None:
    model = joblib.load(model_path)
    _, X_test, _, y_test, _ = get_train_test_data(data_path)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds, output_dict=True)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(report).transpose()
    report_path = out / "classification_report.csv"
    report_df.to_csv(report_path, index=True)

    cm_plot_path = out / "confusion_matrix.png"
    disp = ConfusionMatrixDisplay.from_predictions(y_test, preds)
    disp.figure_.savefig(cm_plot_path, bbox_inches="tight")
    plt.close(disp.figure_)

    print(f"ROC-AUC: {auc:.4f}")
    print(f"Saved report to: {report_path}")
    print(f"Saved confusion matrix plot to: {cm_plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved risk model")
    parser.add_argument("--model-path", default="model/model.joblib")
    parser.add_argument("--data-path", default="data/company_bankruptcy.csv")
    parser.add_argument("--output-dir", default="model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model_path, args.data_path, args.output_dir)
