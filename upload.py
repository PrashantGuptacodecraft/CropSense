# upload.py
# Section 1 – Upload Dataset
# Lets the user upload a CSV or pick a built-in sample dataset.

import streamlit as st
import pandas as pd
import numpy as np


def show():
    st.header("Upload Dataset")
    st.write("Upload a CSV file to get started, or pick one of the sample datasets below.")

    use_sample = st.checkbox("Use a sample dataset instead")

    if use_sample:
        sample_choice = st.selectbox(
            "Choose sample dataset",
            ["Iris (Classification)", "Housing Price (Regression)"],
        )

        if sample_choice == "Iris (Classification)":
            from sklearn.datasets import load_iris
            iris = load_iris(as_frame=True)
            df = iris.frame
            # Map numeric labels to readable class names
            df["target"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
            st.session_state["dataset_name"] = "iris"

        else:
            # Simple synthetic housing price dataset
            np.random.seed(42)
            n = 200
            df = pd.DataFrame({
                "rooms":       np.random.randint(2, 8, n),
                "area_sqft":   np.random.randint(500, 3000, n),
                "age_years":   np.random.randint(1, 50, n),
                "distance_km": np.round(np.random.uniform(1, 30, n), 1),
            })
            df["price"] = (
                df["rooms"] * 15000
                + df["area_sqft"] * 120
                - df["age_years"] * 500
                - df["distance_km"] * 2000
                + np.random.normal(0, 20000, n)
            ).round(2)
            st.session_state["dataset_name"] = "housing"

        st.session_state["df"] = df
        st.success(f"Loaded '{sample_choice}' — {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10))

    else:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state["df"] = df
            st.session_state["dataset_name"] = uploaded.name
            st.success(f"File loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(10))
        else:
            st.info("Please upload a CSV file or tick the checkbox above to use a sample.")
