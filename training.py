# training.py
# Section 5 – Model Training
# Trains a classification or regression model and shows evaluation metrics.

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score,
)

from database import save_result


def show():
    st.header("Model Training")

    df_proc  = st.session_state.get("df_processed")
    target   = st.session_state.get("target_col")
    features = st.session_state.get("feature_cols")

    if df_proc is None or target is None:
        st.warning("Please complete preprocessing first (Section 3).")
        return

    X = df_proc[features].values
    y = df_proc[target].values

    # Auto-suggest task type based on number of unique target values
    unique_count = len(np.unique(y))
    auto_task = "Classification" if unique_count <= 20 else "Regression"
    task_type = st.radio(
        "Task type",
        ["Classification", "Regression"],
        index=0 if auto_task == "Classification" else 1,
    )
    st.session_state["task_type"] = task_type

    # Model selection based on task type
    if task_type == "Classification":
        model_name = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
    else:
        model_name = st.selectbox("Model", ["Linear Regression", "Decision Tree", "Random Forest"])

    test_size = st.slider("Test set size (%)", 10, 40, 20)

    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=42
        )

        # Map model name to sklearn object
        clf_map = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Decision Tree":       DecisionTreeClassifier(random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        }
        reg_map = {
            "Linear Regression": LinearRegression(),
            "Decision Tree":     DecisionTreeRegressor(random_state=42),
            "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
        }

        model = clf_map[model_name] if task_type == "Classification" else reg_map[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Save trained model to session state for use in prediction
        st.session_state["model"] = model

        st.subheader("Results")

        if task_type == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc:.4f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Confusion matrix heatmap
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            plt.close()

            save_result(model_name, task_type, "Accuracy", acc,
                        st.session_state.get("dataset_name", "unknown"))
            st.info("Result saved to database.")

        else:
            mse  = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2   = r2_score(y_test, y_pred)

            m1, m2, m3 = st.columns(3)
            m1.metric("MSE",      f"{mse:.2f}")
            m2.metric("RMSE",     f"{rmse:.2f}")
            m3.metric("R² Score", f"{r2:.4f}")

            # Actual vs Predicted scatter
            fig2, ax2 = plt.subplots()
            ax2.scatter(y_test, y_pred, alpha=0.6, color="teal")
            ax2.plot([y_test.min(), y_test.max()],
                     [y_test.min(), y_test.max()], "r--")
            ax2.set_xlabel("Actual")
            ax2.set_ylabel("Predicted")
            ax2.set_title("Actual vs Predicted")
            st.pyplot(fig2)
            plt.close()

            save_result(model_name, task_type, "R2_Score", r2,
                        st.session_state.get("dataset_name", "unknown"))
            st.info("Result saved to database.")
