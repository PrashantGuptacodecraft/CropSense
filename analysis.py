# analysis.py
# Section 2 – Data Analysis
# Shows basic info about the loaded dataset: shape, types, stats, missing values.

import streamlit as st
import pandas as pd


def show():
    st.header("Data Analysis")

    df = st.session_state.get("df")
    if df is None:
        st.warning("Please upload a dataset first (Section 1).")
        return

    # Top-level summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    # Preview
    st.subheader("First 10 rows")
    st.dataframe(df.head(10))

    # Column info table
    st.subheader("Column Info")
    info_df = pd.DataFrame({
        "Column":     df.columns,
        "Type":       df.dtypes.values,
        "Non-Null":   df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
    })
    st.dataframe(info_df, use_container_width=True)

    # Descriptive statistics for numeric columns
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    # Missing value details
    st.subheader("Missing Values per Column")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values found in the dataset.")
    else:
        st.dataframe(missing.rename("Missing Count"))
