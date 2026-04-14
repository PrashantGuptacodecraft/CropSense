# visualization.py
# Section 4 – Visualization
# Four chart types: histogram, scatter plot, correlation heatmap, box plot.

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def show():
    st.header("Visualization")

    df = st.session_state.get("df")
    if df is None:
        st.warning("Please upload a dataset first (Section 1).")
        return

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found in the dataset.")
        return

    # ── Histogram ─────────────────────────────────────────────────────────────
    st.subheader("Distribution of a Column")
    col_hist = st.selectbox("Column", num_cols, key="hist")
    fig, ax = plt.subplots()
    ax.hist(df[col_hist].dropna(), bins=20, color="steelblue", edgecolor="white")
    ax.set_xlabel(col_hist)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {col_hist}")
    st.pyplot(fig)
    plt.close()

    # ── Scatter plot ──────────────────────────────────────────────────────────
    st.subheader("Scatter Plot")
    c1, c2 = st.columns(2)
    x_col = c1.selectbox("X axis", num_cols, key="sc_x")
    y_col = c2.selectbox("Y axis", num_cols, index=min(1, len(num_cols) - 1), key="sc_y")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df[x_col], df[y_col], alpha=0.6, color="coral",
                edgecolors="white", linewidths=0.4)
    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)
    ax2.set_title(f"{x_col} vs {y_col}")
    st.pyplot(fig2)
    plt.close()

    # ── Correlation heatmap ───────────────────────────────────────────────────
    if len(num_cols) >= 2:
        st.subheader("Correlation Heatmap")
        fig3, ax3 = plt.subplots(
            figsize=(max(6, len(num_cols)), max(4, len(num_cols) - 1))
        )
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax3, linewidths=0.5, square=True)
        ax3.set_title("Feature Correlation Matrix")
        st.pyplot(fig3)
        plt.close()

    # ── Box plot ──────────────────────────────────────────────────────────────
    st.subheader("Box Plot (outlier detection)")
    col_box = st.selectbox("Column for box plot", num_cols, key="box")
    fig4, ax4 = plt.subplots()
    ax4.boxplot(df[col_box].dropna(), patch_artist=True,
                boxprops=dict(facecolor="lightblue"))
    ax4.set_ylabel(col_box)
    ax4.set_title(f"Box Plot – {col_box}")
    st.pyplot(fig4)
    plt.close()
