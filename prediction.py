# prediction.py
# Section 6 – Prediction
# User enters feature values manually and gets a prediction from the trained model.

import streamlit as st
import numpy as np
import pandas as pd


def show():
    st.header("Make a Prediction")

    model    = st.session_state.get("model")
    features = st.session_state.get("feature_cols")
    df_proc  = st.session_state.get("df_processed")

    if model is None:
        st.warning("Please train a model first (Section 5).")
        return

    st.write("Enter values for each feature below:")

    input_vals = {}
    n_cols = min(3, len(features))
    cols = st.columns(n_cols)

    for i, feat in enumerate(features):
        widget = cols[i % n_cols]
        col_mean = float(df_proc[feat].mean())
        col_min  = float(df_proc[feat].min())
        col_max  = float(df_proc[feat].max())

        # Set a reasonable input range around the actual data range
        input_min = col_min * 2 if col_min < 0 else col_min - abs(col_min)
        input_max = col_max * 2 if col_max > 0 else col_max + abs(col_max)

        input_vals[feat] = widget.number_input(
            feat,
            min_value=input_min,
            max_value=input_max,
            value=col_mean,
        )

    if st.button("Predict"):
        # Build input array in the same column order as training
        input_array = np.array([[input_vals[f] for f in features]])
        prediction  = model.predict(input_array)[0]
        st.success(f"**Prediction: {prediction}**")

        # For classifiers that support probabilities, show a breakdown
        task = st.session_state.get("task_type")
        if task == "Classification" and hasattr(model, "predict_proba"):
            proba   = model.predict_proba(input_array)[0]
            classes = model.classes_
            prob_df = pd.DataFrame({"Class": classes, "Probability": proba})
            st.subheader("Class Probabilities")
            st.dataframe(prob_df)
