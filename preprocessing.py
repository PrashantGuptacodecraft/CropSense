# preprocessing.py
# Section 3 – Preprocessing
# Steps: drop columns → handle nulls → encode categoricals → scale → pick target/features.

import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def show():
    st.header("Preprocessing")

    df = st.session_state.get("df")
    if df is None:
        st.warning("Please upload a dataset first (Section 1).")
        return

    df_proc = df.copy()

    # ── Step 1: Drop columns ──────────────────────────────────────────────────
    st.subheader("Step 1 – Drop Unnecessary Columns")
    drop_cols = st.multiselect("Select columns to remove", df.columns.tolist())
    if drop_cols:
        df_proc.drop(columns=drop_cols, inplace=True)
        st.write(f"Dropped: {drop_cols}")

    # ── Step 2: Handle missing values ─────────────────────────────────────────
    st.subheader("Step 2 – Handle Missing Values")
    strategy = st.radio(
        "Strategy for numeric columns",
        ["Fill with Mean", "Fill with Median", "Drop rows with missing values"],
    )

    num_cols = df_proc.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_proc.select_dtypes(include="object").columns.tolist()

    if strategy == "Fill with Mean":
        df_proc[num_cols] = df_proc[num_cols].fillna(df_proc[num_cols].mean())
    elif strategy == "Fill with Median":
        df_proc[num_cols] = df_proc[num_cols].fillna(df_proc[num_cols].median())
    else:
        df_proc.dropna(inplace=True)

    # Fill remaining categorical nulls with the most common value
    for col in cat_cols:
        if df_proc[col].isnull().any():
            df_proc[col].fillna(df_proc[col].mode()[0], inplace=True)

    # ── Step 3: Encode categorical columns ────────────────────────────────────
    st.subheader("Step 3 – Encode Categorical Columns")
    encode_cols = st.multiselect(
        "Select columns to label-encode",
        cat_cols,
        default=cat_cols,
    )

    encoders = {}
    for col in encode_cols:
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col].astype(str))
        encoders[col] = le
    st.session_state["encoders"] = encoders

    # ── Step 4: Feature scaling ───────────────────────────────────────────────
    st.subheader("Step 4 – Feature Scaling")
    apply_scale = st.checkbox("Apply StandardScaler to numeric features")

    # ── Step 5: Target and feature selection ──────────────────────────────────
    st.subheader("Step 5 – Select Target & Feature Columns")
    target = st.selectbox("Target column (what to predict)", df_proc.columns.tolist())
    features = st.multiselect(
        "Feature columns (inputs)",
        [c for c in df_proc.columns if c != target],
        default=[c for c in df_proc.columns if c != target],
    )

    if st.button("Apply Preprocessing"):
        if not features:
            st.error("Please select at least one feature column.")
            return

        if apply_scale:
            scaler = StandardScaler()
            df_proc[features] = scaler.fit_transform(df_proc[features])
            st.session_state["scaler"] = scaler
        else:
            st.session_state["scaler"] = None

        # Save processed data and selections to session state
        st.session_state["df_processed"] = df_proc
        st.session_state["target_col"]   = target
        st.session_state["feature_cols"] = features

        st.success("Preprocessing complete!")
        st.write(f"Target: **{target}** | Features: {features}")
        st.dataframe(df_proc[features + [target]].head(8))
