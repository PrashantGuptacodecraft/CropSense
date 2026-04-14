# app.py
# Main entry point for the ML Project Streamlit app.
# Each section is handled by its own module file.
#
# Run with:  streamlit run app.py

import streamlit as st
from database import init_db

# Import each section module
import upload
import analysis
import preprocessing
import visualization
import training
import prediction
import storage

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ML Project", layout="wide")
st.title("🎓 Machine Learning Project")
st.caption("A simple end-to-end ML pipeline | College Assignment")

# ── Initialize SQLite database on first run ───────────────────────────────────
init_db()

# ── Initialize session state keys ────────────────────────────────────────────
for key in [
    "df", "df_processed", "target_col", "feature_cols",
    "model", "task_type", "scaler", "encoders", "dataset_name",
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar navigation ────────────────────────────────────────────────────────
section = st.sidebar.radio(
    "Navigation",
    [
        "1. Upload Dataset",
        "2. Data Analysis",
        "3. Preprocessing",
        "4. Visualization",
        "5. Model Training",
        "6. Prediction",
        "7. Database Storage",
    ],
)

# ── Route to the selected section ────────────────────────────────────────────
if section == "1. Upload Dataset":
    upload.show()

elif section == "2. Data Analysis":
    analysis.show()

elif section == "3. Preprocessing":
    preprocessing.show()

elif section == "4. Visualization":
    visualization.show()

elif section == "5. Model Training":
    training.show()

elif section == "6. Prediction":
    prediction.show()

elif section == "7. Database Storage":
    storage.show()