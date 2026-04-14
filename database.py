# database.py
# Handles all SQLite operations for storing and retrieving model results.

import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "ml_results.db"


def init_db():
    """Create the results table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT,
            model_name   TEXT,
            task_type    TEXT,
            metric_name  TEXT,
            metric_value REAL,
            dataset      TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_result(model_name, task_type, metric_name, metric_value, dataset):
    """Insert one row of model evaluation results into SQLite."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO results (timestamp, model_name, task_type, metric_name, metric_value, dataset) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            task_type,
            metric_name,
            metric_value,
            dataset,
        ),
    )
    conn.commit()
    conn.close()


def fetch_results():
    """Return all stored results as a pandas DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM results ORDER BY id DESC", conn)
    conn.close()
    return df


def clear_results():
    """Delete all rows from the results table."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM results")
    conn.commit()
    conn.close()
