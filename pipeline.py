import json
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


APP_TITLE = "Crop Yield Prediction Dashboard"
DB_PATH = Path(__file__).with_name("ml_results.db")


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #070b14;
                --panel: #101826;
                --panel-2: #0d1420;
                --accent: #7dd3fc;
                --accent-2: #34d399;
                --text: #e5eef9;
                --muted: #94a3b8;
                --border: rgba(148, 163, 184, 0.18);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(125, 211, 252, 0.12), transparent 24%),
                    radial-gradient(circle at top right, rgba(52, 211, 153, 0.10), transparent 22%),
                    linear-gradient(180deg, #050815 0%, #070b14 100%);
                color: var(--text);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%);
                border-right: 1px solid var(--border);
            }

            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
            }

            h1, h2, h3, h4, p, label, span, div {
                color: var(--text);
            }

            .hero {
                padding: 1.2rem 1.4rem;
                border: 1px solid var(--border);
                border-radius: 18px;
                background: linear-gradient(135deg, rgba(16, 24, 38, 0.98), rgba(10, 16, 28, 0.92));
                box-shadow: 0 20px 45px rgba(0, 0, 0, 0.25);
                margin-bottom: 1rem;
            }

            .hero h1 {
                margin: 0;
                font-size: 2rem;
                letter-spacing: 0.2px;
            }

            .hero p {
                margin: 0.35rem 0 0 0;
                color: var(--muted);
            }

            .card {
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1rem 1.1rem;
                background: rgba(16, 24, 38, 0.92);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.16);
            }

            .card h4 {
                margin: 0 0 0.35rem 0;
            }

            .subtle {
                color: var(--muted);
                font-size: 0.93rem;
            }

            [data-testid="stMetric"] {
                background: rgba(16, 24, 38, 0.90);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 0.85rem 0.95rem;
                box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12);
            }

            [data-testid="stMetricLabel"] {
                color: var(--muted);
            }

            .section-chip {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                border: 1px solid rgba(125, 211, 252, 0.35);
                color: var(--accent);
                font-size: 0.82rem;
                margin-bottom: 0.65rem;
            }

            div[data-testid="stDataFrame"] {
                border-radius: 14px;
                overflow: hidden;
                border: 1px solid var(--border);
            }

            .stButton button {
                border-radius: 12px;
                border: 1px solid rgba(125, 211, 252, 0.38);
                background: linear-gradient(135deg, rgba(125, 211, 252, 0.18), rgba(52, 211, 153, 0.16));
                color: var(--text);
            }

            .stDownloadButton button {
                border-radius: 12px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def generate_sample_crop_dataset(rows: int = 420, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean", "Potato"]
    seasons = ["Kharif", "Rabi", "Zaid"]
    states = ["Punjab", "Haryana", "Maharashtra", "Karnataka", "Tamil Nadu", "Uttar Pradesh"]
    district_map = {
        "Punjab": ["Ludhiana", "Amritsar", "Patiala"],
        "Haryana": ["Karnal", "Hisar", "Panipat"],
        "Maharashtra": ["Pune", "Nashik", "Nagpur"],
        "Karnataka": ["Mysuru", "Dharwad", "Belagavi"],
        "Tamil Nadu": ["Coimbatore", "Madurai", "Thanjavur"],
        "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra"],
    }

    crop_base = {
        "Rice": 3.25,
        "Wheat": 2.95,
        "Maize": 4.75,
        "Cotton": 1.85,
        "Sugarcane": 68.0,
        "Soybean": 2.45,
        "Potato": 21.0,
    }
    optimal_rainfall = {
        "Rice": 1250,
        "Wheat": 520,
        "Maize": 850,
        "Cotton": 700,
        "Sugarcane": 1450,
        "Soybean": 650,
        "Potato": 400,
    }
    optimal_temp = {
        "Rice": 27,
        "Wheat": 20,
        "Maize": 24,
        "Cotton": 28,
        "Sugarcane": 30,
        "Soybean": 26,
        "Potato": 18,
    }
    season_factor = {"Kharif": 0.12, "Rabi": 0.06, "Zaid": 0.08}
    state_factor = {
        "Punjab": 0.16,
        "Haryana": 0.12,
        "Maharashtra": 0.09,
        "Karnataka": 0.10,
        "Tamil Nadu": 0.11,
        "Uttar Pradesh": 0.13,
    }

    crop = rng.choice(crops, rows)
    season = rng.choice(seasons, rows)
    state = rng.choice(states, rows)
    district = np.array([rng.choice(district_map[s]) for s in state])
    area = rng.uniform(0.8, 45.0, rows).round(2)
    rainfall = rng.uniform(220, 1800, rows).round(1)
    temperature = rng.uniform(16, 38, rows).round(1)

    yield_values = []
    production_values = []
    for i in range(rows):
        c = crop[i]
        rain_score = np.exp(-((rainfall[i] - optimal_rainfall[c]) ** 2) / (2 * 260 ** 2))
        temp_score = np.exp(-((temperature[i] - optimal_temp[c]) ** 2) / (2 * 4.0 ** 2))
        area_effect = 0.025 * np.log1p(area[i])
        base = crop_base[c]
        yld = (
            base
            + (base * 0.85 * rain_score)
            + (base * 0.35 * temp_score)
            + area_effect
            + season_factor[season[i]]
            + state_factor[state[i]]
            + rng.normal(0, base * 0.08)
        )
        yld = max(0.25, float(yld))
        yield_values.append(round(yld, 2))
        production_values.append(round(yld * area[i], 2))

    df = pd.DataFrame(
        {
            "Crop": crop,
            "Area": area,
            "Rainfall": rainfall,
            "Temperature": temperature,
            "Season": season,
            "State": state,
            "District": district,
            "Yield": yield_values,
            "Production": production_values,
        }
    )

    for col in ["Area", "Rainfall", "Temperature", "Yield", "Production"]:
        mask = rng.random(rows) < 0.03
        df.loc[mask, col] = np.nan

    return df


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                dataset_name TEXT,
                model_name TEXT,
                target_name TEXT,
                inputs_json TEXT NOT NULL,
                predicted_value REAL NOT NULL,
                r2_score REAL,
                mae REAL,
                mse REAL,
                cv_score REAL
            )
            """
        )
        conn.commit()


def save_prediction_record(
    *,
    dataset_name: str,
    model_name: str,
    target_name: str,
    inputs: dict,
    predicted_value: float,
    r2_score_value: float | None,
    mae_value: float | None,
    mse_value: float | None,
    cv_score_value: float | None,
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO prediction_history (
                timestamp, dataset_name, model_name, target_name,
                inputs_json, predicted_value, r2_score, mae, mse, cv_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                dataset_name,
                model_name,
                target_name,
                json.dumps(inputs, ensure_ascii=True),
                float(predicted_value),
                None if r2_score_value is None else float(r2_score_value),
                None if mae_value is None else float(mae_value),
                None if mse_value is None else float(mse_value),
                None if cv_score_value is None else float(cv_score_value),
            ),
        )
        conn.commit()


def fetch_prediction_history() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM prediction_history ORDER BY id DESC",
            conn,
        )


def clear_prediction_history() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM prediction_history")
        conn.commit()


def init_session_state() -> None:
    defaults = {
        "df_raw": None,
        "df_clean": None,
        "dataset_name": None,
        "target_col": None,
        "feature_cols": None,
        "train_data_ready": False,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "model": None,
        "model_name": None,
        "metrics": None,
        "cv_scores": None,
        "y_pred": None,
        "last_prediction": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned.replace(
        to_replace=r"^\s*$",
        value=np.nan,
        regex=True,
        inplace=True,
    )

    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            cleaned[col] = cleaned[col].astype(str).str.strip()
            cleaned[col] = cleaned[col].replace(
                {
                    "nan": np.nan,
                    "None": np.nan,
                    "null": np.nan,
                    "NULL": np.nan,
                    "NaN": np.nan,
                    "N/A": np.nan,
                    "": np.nan,
                }
            )

            numeric_series = pd.to_numeric(cleaned[col], errors="coerce")
            if numeric_series.notna().mean() >= 0.85:
                cleaned[col] = numeric_series

    return cleaned


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    numeric_cols = cleaned.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = cleaned.select_dtypes(exclude=np.number).columns.tolist()

    for col in numeric_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in categorical_cols:
        if cleaned[col].isna().any():
            mode_series = cleaned[col].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)

    return cleaned


def infer_target_column(df: pd.DataFrame) -> str | None:
    preferred = [
        "Yield",
        "Production",
        "Yield_kg_ha",
        "Yield per hectare",
        "Crop_Yield",
        "Output",
    ]
    lower_map = {col.lower(): col for col in df.columns}
    for name in preferred:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        return numeric_cols[-1]
    return None


def recommended_features(df: pd.DataFrame, target_col: str | None) -> list[str]:
    columns = [col for col in df.columns if col != target_col]
    if not target_col:
        return columns

    target_lower = target_col.lower()
    leakage_names = []
    if "yield" in target_lower:
        leakage_names = ["production", "output"]
    elif "production" in target_lower:
        leakage_names = ["yield", "output"]

    recommended = [
        col
        for col in columns
        if not any(leak in col.lower() for leak in leakage_names)
    ]
    return recommended or columns


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        try:
            one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", one_hot),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_features, categorical_features


def apply_sidebar() -> str:
    st.sidebar.markdown("### Crop Yield Workflow")
    section = st.sidebar.radio(
        "Navigate the app",
        [
            "1. Upload Crop Dataset",
            "2. Data Analysis",
            "3. Data Preprocessing",
            "4. Feature Selection",
            "5. Visualization",
            "6. Model Training",
            "7. Crop Yield Prediction",
            "8. Save Results",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    loaded = st.session_state.get("df_raw")
    metrics = st.session_state.get("metrics") or {}

    st.sidebar.markdown(
        f"""
        <div class="card">
            <h4>Session Status</h4>
            <div class="subtle">Dataset: {"Loaded" if loaded is not None else "Not loaded"}</div>
            <div class="subtle">Target: {st.session_state.get("target_col") or "Not selected"}</div>
            <div class="subtle">Model: {st.session_state.get("model_name") or "Not trained"}</div>
            <div class="subtle">R2: {metrics.get("r2", "N/A") if metrics else "N/A"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history_count = 0
    try:
        history_count = len(fetch_prediction_history())
    except Exception:
        history_count = 0

    st.sidebar.caption(f"Saved predictions: {history_count}")
    return section


def render_title() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_TITLE}</h1>
            <p>
                A modern Streamlit dashboard for crop yield prediction with data analysis,
                preprocessing, visualization, training, prediction, and SQLite history.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_upload() -> None:
    st.markdown("### Upload Crop Dataset")
    st.write("Upload a CSV file or switch to the built-in sample crop dataset.")

    use_sample = st.checkbox("Use sample crop dataset instead", value=False)

    if use_sample:
        sample_df = generate_sample_crop_dataset()
        st.session_state["df_raw"] = sample_df.copy()
        st.session_state["dataset_name"] = "sample_crop_dataset"
        st.session_state["df_clean"] = None
        st.session_state["target_col"] = None
        st.session_state["feature_cols"] = None
        st.session_state["train_data_ready"] = False
        st.success(f"Sample crop dataset loaded: {sample_df.shape[0]} rows x {sample_df.shape[1]} columns")
        st.dataframe(sample_df.head(10))
    else:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read the CSV file: {exc}")
                return

            st.session_state["df_raw"] = df.copy()
            st.session_state["dataset_name"] = uploaded.name
            st.session_state["df_clean"] = None
            st.session_state["target_col"] = None
            st.session_state["feature_cols"] = None
            st.session_state["train_data_ready"] = False
            st.success(f"File loaded: {df.shape[0]} rows x {df.shape[1]} columns")
            st.dataframe(df.head(10))
        else:
            st.info("Upload a CSV file or enable the sample dataset toggle.")

    df = st.session_state.get("df_raw")
    if df is not None:
        with st.expander("Dataset quick summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing cells", int(df.isna().sum().sum()))
            st.write("Column names:")
            st.code(", ".join(df.columns.astype(str).tolist()))


def section_analysis() -> None:
    st.markdown("### Data Analysis")
    df = st.session_state.get("df_raw")
    if df is None:
        st.warning("Please upload a crop dataset first.")
        return

    preview_cols = st.columns(4)
    preview_cols[0].metric("Rows", df.shape[0])
    preview_cols[1].metric("Columns", df.shape[1])
    preview_cols[2].metric("Missing Values", int(df.isna().sum().sum()))
    preview_cols[3].metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))

    st.markdown("#### Preview")
    st.dataframe(df.head(10))

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.markdown("#### Summary Statistics")
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty:
            st.info("No numeric columns available for summary statistics.")
        else:
            st.dataframe(numeric_df.describe().T)

    with col2:
        st.markdown("#### Missing Values")
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values found in the uploaded data.")
        else:
            st.dataframe(missing.rename("Missing Count"))

    st.markdown("#### Correlation Heatmap")
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(min(12, 1.2 * numeric_df.shape[1]), 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Between Numeric Columns")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("At least two numeric columns are required for a correlation heatmap.")

    st.markdown("#### Basic Insights")
    insights = []
    if "Rainfall" in df.columns and pd.api.types.is_numeric_dtype(df["Rainfall"]):
        insights.append(f"Average rainfall: {df['Rainfall'].mean():.2f}")
    if "Temperature" in df.columns and pd.api.types.is_numeric_dtype(df["Temperature"]):
        insights.append(f"Average temperature: {df['Temperature'].mean():.2f}")
    if "Area" in df.columns and pd.api.types.is_numeric_dtype(df["Area"]):
        insights.append(f"Average cultivated area: {df['Area'].mean():.2f}")

    output_col = None
    for candidate in ["Yield", "Production"]:
        if candidate in df.columns and pd.api.types.is_numeric_dtype(df[candidate]):
            output_col = candidate
            break
    if output_col:
        insights.append(f"Average {output_col.lower()}: {df[output_col].mean():.2f}")

    if "Crop" in df.columns:
        top_crop = df["Crop"].astype(str).mode(dropna=True)
        if not top_crop.empty:
            insights.append(f"Most common crop: {top_crop.iloc[0]}")

    if insights:
        for item in insights:
            st.markdown(f"- {item}")
    else:
        st.write("No basic insights could be derived from the current dataset.")


def section_preprocessing() -> None:
    st.markdown("### Data Preprocessing")
    df = st.session_state.get("df_raw")
    if df is None:
        st.warning("Please upload a crop dataset first.")
        return

    normalized = normalize_dataframe(df)
    cleaned = clean_missing_values(normalized)
    st.session_state["df_clean"] = cleaned.copy()

    inferred_target = infer_target_column(cleaned)
    target_col = st.selectbox(
        "Select the target column to predict",
        options=cleaned.columns.tolist(),
        index=cleaned.columns.tolist().index(inferred_target) if inferred_target in cleaned.columns else 0,
        help="For this project, choose the numeric output column such as Yield or Production.",
    )

    default_features = recommended_features(cleaned, target_col)
    feature_cols = st.multiselect(
        "Select the input features",
        options=[col for col in cleaned.columns if col != target_col],
        default=default_features,
        help="Choose the columns that should be used to predict crop yield.",
    )

    test_size = st.slider("Test set size", min_value=10, max_value=35, value=20, step=5)
    split_seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)

    st.markdown("#### Outlier Detection")
    outlier_method = st.radio("Outlier Detection Method", ["None", "IQR", "Isolation Forest"], horizontal=True)
    remove_outliers = st.checkbox("Remove Outliers", value=False)
    
    st.markdown("#### PCA Visualization")
    apply_pca = st.checkbox("Apply PCA Visualization (2D)")

    st.markdown("#### Cleaned Preview")
    st.dataframe(cleaned.head(10))

    st.markdown("#### Missing Values After Cleaning")
    remaining_missing = cleaned.isna().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    if remaining_missing.empty:
        st.success("All missing values have been handled.")
    else:
        st.dataframe(remaining_missing.rename("Missing Count"))

    if st.button("Prepare Training Data"):
        if not feature_cols:
            st.error("Please select at least one feature column.")
            return

        target_series = pd.to_numeric(cleaned[target_col], errors="coerce")
        valid_mask = target_series.notna()
        if valid_mask.sum() < 10:
            st.error("The target column does not contain enough numeric values for regression.")
            return

        working_df = cleaned.loc[valid_mask, feature_cols + [target_col]].copy()
        working_df[target_col] = target_series.loc[valid_mask]

        st.session_state["target_col"] = target_col
        st.session_state["feature_cols"] = feature_cols

        outlier_indices = pd.Series(False, index=working_df.index)
        num_cols = working_df[feature_cols].select_dtypes(include=np.number).columns
        if outlier_method == "IQR" and len(num_cols) > 0:
            Q1 = working_df[num_cols].quantile(0.25)
            Q3 = working_df[num_cols].quantile(0.75)
            IQR = Q3 - Q1
            outlier_condition = ((working_df[num_cols] < (Q1 - 1.5 * IQR)) | (working_df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            outlier_indices = outlier_condition
        elif outlier_method == "Isolation Forest" and len(num_cols) > 0:
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(working_df[num_cols].fillna(working_df[num_cols].median()))
            outlier_indices = pd.Series(preds == -1, index=working_df.index)
        
        if outlier_method != "None":
            st.info(f"Detected {outlier_indices.sum()} outliers using {outlier_method}.")
            if remove_outliers:
                working_df = working_df[~outlier_indices]
                st.success(f"Removed outliers. Remaining rows: {len(working_df)}")

        X = working_df[feature_cols].copy()
        y = working_df[target_col].copy()

        if apply_pca and len(num_cols) >= 2:
            st.markdown("#### PCA Visualization (2D)")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X[num_cols].fillna(X[num_cols].median()))
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7)
            plt.colorbar(scatter, label=target_col)
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            st.pyplot(fig)
            plt.close(fig)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size / 100.0,
                random_state=int(split_seed),
            )
        except ValueError as exc:
            st.error(f"Could not create a train/test split: {exc}")
            return

        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["train_data_ready"] = True
        st.session_state["model"] = None
        st.session_state["model_name"] = None
        st.session_state["metrics"] = None
        st.session_state["cv_scores"] = None
        st.session_state["y_pred"] = None

        st.success(
            f"Training data prepared with {X_train.shape[0]} training rows and {X_test.shape[0]} testing rows."
        )
        st.write(f"Target column: **{target_col}**")
        st.write(f"Feature columns: {', '.join(feature_cols)}")
        st.dataframe(X_train.head(8))


def section_feature_selection() -> None:
    st.markdown("### Feature Selection")
    if not st.session_state.get("train_data_ready"):
        st.warning("Please complete data preprocessing first.")
        return

    X_train = st.session_state["X_train"]
    y_train = st.session_state["y_train"]
    feature_cols = st.session_state["feature_cols"]

    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.info("No numeric features available for selection routines.")
        return

    sel_method = st.selectbox("Feature Selection Method", ["None", "Correlation-based", "Variance Threshold", "Feature Importance"])

    if sel_method == "Correlation-based":
        corr = X_train[num_cols].corr()
        st.write("Correlation matrix of numeric features:")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)
        st.info("Features with low correlation to target or high correlation with each other can be manually removed in the preprocessing step.")

    elif sel_method == "Variance Threshold":
        threshold = st.slider("Variance Threshold", 0.0, 1.0, 0.01)
        selector = VarianceThreshold(threshold=threshold)
        try:
            selector.fit(X_train[num_cols].fillna(X_train[num_cols].median()))
            kept = [num_cols[i] for i, mask in enumerate(selector.get_support()) if mask]
            dropped = [c for c in num_cols if c not in kept]
            st.success(f"Features kept: {kept}")
            if dropped:
                st.warning(f"Features dropped (low variance): {dropped}")
        except Exception as e:
            st.error(f"Variance threshold failed: {e}")

    elif sel_method == "Feature Importance":
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        X_imputed = X_train[num_cols].fillna(X_train[num_cols].median())
        rf.fit(X_imputed, y_train)
        importances = pd.Series(rf.feature_importances_, index=num_cols).sort_values(ascending=False)
        st.bar_chart(importances)
        st.success(f"Top feature: {importances.index[0]} ({importances.iloc[0]:.4f})")


def section_visualization() -> None:
    st.markdown("### Visualization")
    df = st.session_state.get("df_clean")
    if df is None:
        df = st.session_state.get("df_raw")
    if df is None:
        st.warning("Please upload and preprocess a crop dataset first.")
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns are available for charts.")
        return

    left, right = st.columns(2)
    with left:
        if "Rainfall" in df.columns and pd.api.types.is_numeric_dtype(df["Rainfall"]):
            st.markdown("#### Rainfall Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df["Rainfall"], bins=24, kde=True, color="#38bdf8", ax=ax)
            ax.set_xlabel("Rainfall")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close(fig)

        target_col = "Yield" if "Yield" in df.columns else ("Production" if "Production" in df.columns else None)
        if "Area" in df.columns and target_col:
            st.markdown(f"#### Area vs {target_col}")
            fig, ax = plt.subplots()
            if "Crop" in df.columns:
                sns.scatterplot(data=df, x="Area", y=target_col, hue="Crop", alpha=0.75, ax=ax)
            else:
                sns.scatterplot(data=df, x="Area", y=target_col, alpha=0.75, ax=ax)
            ax.set_title(f"Area vs {target_col}")
            st.pyplot(fig)
            plt.close(fig)

    with right:
        if "Temperature" in df.columns and pd.api.types.is_numeric_dtype(df["Temperature"]):
            st.markdown("#### Temperature Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df["Temperature"], bins=24, kde=True, color="#34d399", ax=ax)
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close(fig)

        if "Crop" in df.columns and target_col and pd.api.types.is_numeric_dtype(df[target_col]):
            st.markdown(f"#### Crop-wise {target_col}")
            grouped = df.groupby("Crop")[target_col].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            grouped.plot(kind="bar", color="#7dd3fc", ax=ax)
            ax.set_ylabel(f"Average {target_col}")
            ax.set_xlabel("Crop")
            ax.set_title(f"Crop-wise Average {target_col}")
            plt.xticks(rotation=25, ha="right")
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("#### Correlation Heatmap")
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(min(12, 1.2 * len(numeric_cols)), 6))
        sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=True, linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Actual vs Predicted Values")
    y_test = st.session_state.get("y_test")
    y_pred = st.session_state.get("y_pred")
    if y_test is not None and y_pred is not None:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.75, color="#f97316", edgecolors="white")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "w--", linewidth=1.2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Train a model first to see the actual vs predicted plot.")


def section_training() -> None:
    st.markdown("### Model Training")
    if not st.session_state.get("train_data_ready"):
        st.warning("Please complete preprocessing first.")
        return

    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test = st.session_state["y_test"]

    model_name = st.selectbox(
        "Choose a regression model",
        ["Random Forest Regressor", "Linear Regression", "Support Vector Machine"],
        index=0,
    )

    k_folds = st.number_input("K-Fold Cross Validation", min_value=2, max_value=10, value=5, step=1)
    use_grid_search = st.checkbox("Enable Hyperparameter Tuning (GridSearchCV)")

    if st.button("Train Model"):
        preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

        if model_name == "Random Forest Regressor":
            base_estimator = RandomForestRegressor(random_state=42)
            param_grid = {"regressor__n_estimators": [50, 100], "regressor__max_depth": [None, 10]}
        elif model_name == "Support Vector Machine":
            base_estimator = SVR()
            param_grid = {"regressor__C": [1, 10], "regressor__epsilon": [0.1, 0.2]}
        else:
            base_estimator = LinearRegression()
            param_grid = {}

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", base_estimator)])

        if use_grid_search and param_grid:
            st.info("Running GridSearchCV... This may take a moment.")
            search = GridSearchCV(pipeline, param_grid, cv=k_folds, scoring="r2", n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            st.success(f"Best parameters selected: {search.best_params_}")
        else:
            model = pipeline
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        n_splits = min(k_folds, len(X_train))
        cv_score = None
        cv_scores = None
        if n_splits >= 2:
            try:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
                cv_score = float(np.mean(cv_scores))
            except Exception as exc:
                st.warning(f"Cross-validation could not be completed: {exc}")

        st.session_state["model"] = model
        st.session_state["model_name"] = model_name
        st.session_state["metrics"] = {"r2": float(r2), "mae": float(mae), "mse": float(mse), "cv": cv_score}
        st.session_state["cv_scores"] = cv_scores
        st.session_state["y_pred"] = y_pred

        metric_cols = st.columns(4)
        metric_cols[0].metric("Model Accuracy (R2 Score)", f"{r2:.4f}")
        metric_cols[1].metric("MAE", f"{mae:.4f}")
        metric_cols[2].metric("MSE", f"{mse:.4f}")
        metric_cols[3].metric("K-Fold CV Score", "N/A" if cv_score is None else f"{cv_score:.4f}")

        st.markdown("#### Actual vs Predicted")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.8, color="#22c55e", edgecolors="white")
        lower = min(float(np.min(y_test)), float(np.min(y_pred)))
        upper = max(float(np.max(y_test)), float(np.max(y_pred)))
        ax.plot([lower, upper], [lower, upper], "r--", linewidth=1.2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted on Test Set")
        st.pyplot(fig)
        plt.close(fig)

        if cv_scores is not None:
            st.markdown("#### Cross-Validation Scores")
            st.write([round(float(score), 4) for score in cv_scores])
            st.success(f"Average {n_splits}-fold CV score: {cv_score:.4f}")


def render_prediction_metrics() -> None:
    metrics = st.session_state.get("metrics") or {}
    cols = st.columns(4)
    cols[0].metric("Model Accuracy (R2 Score)", "N/A" if metrics.get("r2") is None else f"{metrics['r2']:.4f}")
    cols[1].metric("MAE", "N/A" if metrics.get("mae") is None else f"{metrics['mae']:.4f}")
    cols[2].metric("MSE", "N/A" if metrics.get("mse") is None else f"{metrics['mse']:.4f}")
    cols[3].metric("K-Fold CV Score", "N/A" if metrics.get("cv") is None else f"{metrics['cv']:.4f}")


def section_prediction() -> None:
    st.markdown("### Crop Yield Prediction")
    model = st.session_state.get("model")
    feature_cols = st.session_state.get("feature_cols")
    cleaned_df = st.session_state.get("df_clean")
    if cleaned_df is None:
        cleaned_df = st.session_state.get("df_raw")

    if model is None or not feature_cols:
        st.warning("Please train a regression model first.")
        return

    render_prediction_metrics()
    st.markdown("#### Prediction Form")
    st.write("Enter only the input features used by the model. The output column is not required.")

    with st.form("prediction_form"):
        input_values: dict[str, object] = {}
        columns = st.columns(2)
        for idx, feature in enumerate(feature_cols):
            widget_col = columns[idx % 2]

            if feature.lower() == "year":
                input_values[feature] = widget_col.number_input(
                    "Enter Year" if feature.lower() == "year" else feature,
                    min_value=2000,
                    max_value=2035,
                    value=2020,
                    step=1
                )
            elif pd.api.types.is_numeric_dtype(cleaned_df[feature]):
                min_val = float(cleaned_df[feature].min())
                max_val = float(cleaned_df[feature].max())
                if np.isclose(min_val, max_val):
                    max_val = min_val + 1.0
                default_val = float(cleaned_df[feature].median())
                input_values[feature] = widget_col.number_input(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                )
            else:
                options = cleaned_df[feature].dropna().astype(str).unique().tolist()
                if not options:
                    options = ["Unknown"]
                input_values[feature] = widget_col.selectbox(feature, options)

        submitted = st.form_submit_button("Predict Yield")

    if submitted:
        input_df = pd.DataFrame([input_values], columns=feature_cols)
        try:
            prediction = float(model.predict(input_df)[0])
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        st.session_state["last_prediction"] = prediction
        target_col = st.session_state.get("target_col") or "Yield"
        title = "Predicted Yield" if "yield" in target_col.lower() else f"Predicted {target_col}"

        st.success("Prediction completed successfully.")
        st.metric(title, f"{prediction:.4f}")

        metrics = st.session_state.get("metrics") or {}
        save_prediction_record(
            dataset_name=st.session_state.get("dataset_name") or "unknown",
            model_name=st.session_state.get("model_name") or "unknown",
            target_name=target_col,
            inputs=input_values,
            predicted_value=prediction,
            r2_score_value=metrics.get("r2"),
            mae_value=metrics.get("mae"),
            mse_value=metrics.get("mse"),
            cv_score_value=metrics.get("cv"),
        )
        st.info("The prediction has been saved to SQLite history.")

        st.markdown("#### Model Performance")
        render_prediction_metrics()


def section_save_results() -> None:
    st.markdown("### Save Results")
    st.write("Prediction history is stored locally in SQLite and can be reviewed here.")

    history = fetch_prediction_history()
    if history.empty:
        st.info("No saved predictions yet. Make a prediction first.")
        return

    display_df = history.copy()
    expanded_inputs = pd.json_normalize(display_df["inputs_json"].apply(json.loads))
    display_df = pd.concat([display_df.drop(columns=["inputs_json"]), expanded_inputs], axis=1)
    display_df["timestamp"] = pd.to_datetime(display_df["timestamp"])

    summary_cols = st.columns(4)
    summary_cols[0].metric("Saved Predictions", len(display_df))
    summary_cols[1].metric("Best R2", f"{display_df['r2_score'].dropna().max():.4f}" if display_df["r2_score"].notna().any() else "N/A")
    summary_cols[2].metric("Average MAE", f"{display_df['mae'].dropna().mean():.4f}" if display_df["mae"].notna().any() else "N/A")
    summary_cols[3].metric("Average CV", f"{display_df['cv_score'].dropna().mean():.4f}" if display_df["cv_score"].notna().any() else "N/A")

    st.dataframe(display_df)

    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prediction History",
        data=csv_data,
        file_name="crop_yield_prediction_history.csv",
        mime="text/csv",
    )

    if st.button("Clear Prediction History"):
        clear_prediction_history()
        st.success("Prediction history cleared.")
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_db()
    init_session_state()
    render_title()
    section = apply_sidebar()

    if section == "1. Upload Crop Dataset":
        section_upload()
    elif section == "2. Data Analysis":
        section_analysis()
    elif section == "3. Data Preprocessing":
        section_preprocessing()
    elif section == "4. Feature Selection":
        section_feature_selection()
    elif section == "5. Visualization":
        section_visualization()
    elif section == "6. Model Training":
        section_training()
    elif section == "7. Crop Yield Prediction":
        section_prediction()
    elif section == "8. Save Results":
        section_save_results()


if __name__ == "__main__":
    main()
