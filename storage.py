# storage.py
# Section 7 – Database Storage
# View, compare, export, and clear saved model results from SQLite.

import streamlit as st
import matplotlib.pyplot as plt

from database import fetch_results, clear_results


def show():
    st.header("Database Storage")
    st.write("All model results are saved locally in `ml_results.db` (SQLite).")

    results_df = fetch_results()

    if results_df.empty:
        st.info("No results yet. Train a model first (Section 5).")
        return

    # Show stored records
    st.subheader(f"Stored Results ({len(results_df)} records)")
    st.dataframe(results_df, use_container_width=True)

    # Bar chart comparing average metric per model
    st.subheader("Average Metric Value per Model")
    grouped = results_df.groupby("model_name")["metric_value"].mean()
    colors  = plt.cm.Set2.colors

    fig, ax = plt.subplots()
    for i, (name, val) in enumerate(grouped.items()):
        ax.bar(name, val, color=colors[i % len(colors)])
    ax.set_ylabel("Average Metric Value")
    ax.set_title("Model Comparison")
    plt.xticks(rotation=15)
    st.pyplot(fig)
    plt.close()

    # CSV export
    st.subheader("Export")
    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name="ml_results.csv",
        mime="text/csv",
    )

    # Clear all records
    st.subheader("Manage Database")
    if st.button("Clear All Results"):
        clear_results()
        st.success("All records deleted.")
        st.rerun()
