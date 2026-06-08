<<<<<<< HEAD

# CropSense — Crop Yield Prediction Dashboard

CropSense is an interactive Streamlit application that demonstrates a full machine learning workflow for crop yield prediction. It includes data ingestion, automated preprocessing, visualization, model training (with hyperparameter tuning and cross-validation), an interactive prediction UI, and a local SQLite-backed prediction history.

## Highlights

- Interactive web dashboard built with `Streamlit` for easy experimentation and demoing.
- Supports CSV upload or a built-in synthetic crop dataset generator for quick testing.
- End-to-end ML pipeline: data cleaning, outlier detection (IQR / IsolationForest), encoding, scaling, PCA visualization, feature selection, and training.
- Model training with `LinearRegression`, `RandomForestRegressor`, `SVR`; optional `GridSearchCV` hyperparameter tuning and K‑fold cross-validation.
- Evaluation metrics: R², MAE, MSE and visual diagnostics (correlation heatmaps, PCA, actual vs predicted plots).
- Prediction form to enter feature values and save predictions to a local SQLite database (`ml_results.db`).

## Tech Stack

- Python 3
- Streamlit
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- SQLite (`sqlite3`)

## Quickstart

1. Create a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run pipeline.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`).

## Usage Overview

- Section 1 — Upload Crop Dataset: Upload a CSV or use the sample dataset generator.
- Section 2 — Data Analysis: View summary statistics, missing values, and correlation heatmaps.
- Section 3 — Data Preprocessing: Automatic cleaning, choose target/features, outlier handling, PCA preview.
- Section 4 — Feature Selection: Correlation, variance threshold, and feature-importance via Random Forest.
- Section 5 — Visualization: Charts for rainfall, temperature, area vs yield, crop-wise averages, actual vs predicted.
- Section 6 — Model Training: Train models, optionally run `GridSearchCV`, and view metrics and cross-validation scores.
- Section 7 — Crop Yield Prediction: Enter feature values, predict, and persist the result to SQLite.
- Section 8 — Save Results: Inspect and download saved prediction history as CSV.

## Files of Interest

- `pipeline.py` — main Streamlit application and the full ML workflow.
- `requirements.txt` — Python dependencies for the project.
- `ml_results.db` — created at runtime to store prediction history (SQLite).

## Contribution

Feel free to open issues or submit pull requests. Suggested improvements:

- Add unit tests and CI workflow.
- Improve model selection and add more algorithms.
- Add Docker support for reproducible deployment.

## License

MIT License — see LICENSE (or add one) for details.

---

If you want, I can also polish the README into a one-page GitHub description or prepare LinkedIn/GitHub-ready bullets describing this project for your resume.
