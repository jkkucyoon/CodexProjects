# Supplier / Company Risk Prediction (Freelance-Style ML Project)

This project simulates a realistic freelance machine learning engagement for a supply-chain intelligence company (similar to Z2Data). The objective is to predict whether a company is at **high risk** of failure using financial indicators.

## 1) What this project does

- Loads a real-world Kaggle dataset about company bankruptcy risk.
- Cleans and prepares the data for modeling.
- Compares multiple models (Logistic Regression and Random Forest).
- Selects the best model using ROC-AUC and F1 score.
- Saves the best model as a reusable artifact (`model/model.joblib`).
- Provides:
  - CLI inference script (`src/predict.py`)
  - FastAPI prediction endpoint (`POST /predict`)

## 2) Dataset used

**Dataset:** Company Bankruptcy Prediction (Kaggle)
- Kaggle page: https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction
- Expected local path in this project: `data/company_bankruptcy.csv`
- Target column: `Bankrupt?` (1 = bankrupt/high risk, 0 = non-bankrupt/lower risk)
- The loader auto-detects common Kaggle layouts: `data/data.csv`, nested paths like `data/<subfolder>/data.csv`, and can extract CSVs from a ZIP found in `data/`.

### Download via Kaggle CLI

```bash
kaggle datasets download -d fedesoriano/company-bankruptcy-prediction -p data --unzip
mv data/data.csv data/company_bankruptcy.csv
```

## 3) ML pipeline

1. **Data loading** from CSV.
2. **Data cleaning**
   - Drop duplicate rows
   - Replace `inf/-inf` with missing values
3. **EDA** in `notebooks/exploration.ipynb`
   - Target balance check
   - Basic inspection of feature distributions
4. **Feature engineering / preprocessing**
   - Numeric median imputation
   - Standard scaling
5. **Model training**
   - Logistic Regression
   - Random Forest
   - (XGBoost can be added later if needed)
6. **Evaluation**
   - ROC-AUC
   - F1 score
   - Classification report
   - Confusion matrix
7. **Best model selection** by ROC-AUC.
8. **Save model** with `joblib`.
9. **Inference script** for one-record JSON prediction.
10. **Prediction API** with FastAPI.

## 4) Why these models were chosen

- **Logistic Regression**
  - Strong baseline for binary risk classification.
  - Easy to explain to non-technical stakeholders.
- **Random Forest**
  - Handles non-linear patterns well.
  - Robust with tabular financial features.
  - Usually strong performance with limited feature tuning.
- **XGBoost (optional)**
  - Often high-performing for tabular data but adds setup complexity.

## 5) Evaluation metrics used

- **ROC-AUC**: Ranking quality across decision thresholds.
- **F1 score**: Balances precision and recall for risk detection.
- **Classification report**: Per-class precision, recall, F1.
- **Confusion matrix**: Practical error breakdown.

## 6) How to run training

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --data-path data/company_bankruptcy.csv --model-dir model
```

Outputs:
- `model/model.joblib`
- `model/metrics.csv`

## 7) How to run prediction (CLI)

Create an input JSON (`sample_input.json`) with feature names from the training dataset:

```json
{
  "ROA(C) before interest and depreciation before interest": 0.37,
  "Operating Gross Margin": 0.61,
  "Realized Sales Gross Margin": 0.60,
  "Debt ratio %": 0.45
}
```

Run prediction:

```bash
python src/predict.py --model-path model/model.joblib --input-json sample_input.json
```

Example output:

```json
{
  "risk_prediction": "high",
  "probability": 0.82
}
```

## 8) How to run API

```bash
uvicorn api.app:app --reload
```

### Endpoint

`POST /predict`

Example body:

```json
{
  "feature1": 0.35,
  "feature2": 0.72
}
```

Example response:

```json
{
  "risk_prediction": "high",
  "probability": 0.82
}
```

> In actual use, request keys should match the model training feature names.

## Project structure

```text
project/
│
├── data/
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── api/
│   └── app.py
│
├── model/
│   └── model.joblib
│
├── requirements.txt
└── README.md
```

## How to explain this project in a job interview

- **Why these algorithms were used**
  - I started with Logistic Regression as a transparent baseline and compared it with Random Forest to capture non-linear behavior common in financial risk data.

- **Why preprocessing was needed**
  - Financial ratio datasets frequently include missing and extreme values. Median imputation and scaling improved model stability and made the baseline model more reliable.

- **How the model could be improved**
  - Add class imbalance strategies (SMOTE, threshold optimization).
  - Add feature selection and feature importance analysis.
  - Add XGBoost/LightGBM comparison and hyperparameter tuning.
  - Add cross-validation and model monitoring.

- **How this system could be deployed for a real company**
  - Package training/inference in Docker.
  - Deploy FastAPI behind an API gateway.
  - Log predictions and drift metrics.
  - Schedule periodic retraining from refreshed supplier/company data.

## Notes

- This repository is intentionally simple and delivery-focused, matching typical freelance scope.
- You can extend the same pipeline to supplier-level risk scoring once additional supplier-specific features are available.
