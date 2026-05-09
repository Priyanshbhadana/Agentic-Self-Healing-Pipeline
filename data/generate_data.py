"""
Mock Dataset Generator
Produces intentionally broken datasets to simulate real pipeline failures.
"""

import pandas as pd
import numpy as np
import json
import os

# ─────────────────────────────────────────────
#  Scenario 1 — Missing Values (Data Quality)
# ─────────────────────────────────────────────
def generate_missing_values_dataset():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "user_id":    range(1, n + 1),
        "age":        [np.nan if i % 7 == 0 else np.random.randint(18, 65) for i in range(n)],
        "salary":     [np.nan if i % 5 == 0 else np.random.randint(30000, 120000) for i in range(n)],
        "email":      [None if i % 10 == 0 else f"user{i}@example.com" for i in range(n)],
        "score":      [np.nan if i % 8 == 0 else round(np.random.uniform(0, 100), 2) for i in range(n)],
        "country":    ["IN", "US", "UK", "DE", "FR"] * 20,
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="3D").astype(str),
    })
    return df


# ─────────────────────────────────────────────
#  Scenario 2 — Schema Mismatch
# ─────────────────────────────────────────────
def generate_schema_mismatch_dataset():
    """Adds unexpected columns + renames expected ones."""
    np.random.seed(99)
    n = 80
    df = pd.DataFrame({
        "uid":           range(1, n + 1),          # was: user_id
        "usr_age":       np.random.randint(18, 70, n),  # was: age
        "annual_salary": np.random.randint(30000, 150000, n),  # was: salary
        "email_address": [f"u{i}@corp.com" for i in range(n)],  # was: email
        "risk_score":    np.random.uniform(0, 1, n).round(3),   # was: score
        "geo":           ["IN"] * n,                             # was: country
        "registered_on": pd.date_range("2024-01-01", periods=n, freq="2D").astype(str),  # was: signup_date
        "extra_flag":    np.random.choice([True, False], n),     # UNEXPECTED column
        "legacy_id":     [f"L-{i:04d}" for i in range(n)],      # UNEXPECTED column
    })
    return df


# ─────────────────────────────────────────────
#  Scenario 3 — Data Anomalies (outliers, bad types)
# ─────────────────────────────────────────────
def generate_anomaly_dataset():
    np.random.seed(7)
    n = 60
    df = pd.DataFrame({
        "user_id": range(1, n + 1),
        "age":     list(np.random.randint(18, 65, n - 5)) + [999, -1, 0, 200, 150],  # outliers
        "salary":  list(np.random.randint(30000, 120000, n - 3)) + [99999999, -5000, 0],
        "score":   list(np.random.uniform(0, 100, n - 2)) + [9999.9, -50.0],
        "country": ["IN", "US", "INVALID_CC", "XX", "UK"] * 12,
        "email":   [f"user{i}@example.com" for i in range(n)],
        "signup_date": pd.date_range("2022-01-01", periods=n, freq="5D").astype(str),
    })
    return df


# ─────────────────────────────────────────────
#  Expected Schema definition
# ─────────────────────────────────────────────
EXPECTED_SCHEMA = {
    "user_id":    "int64",
    "age":        "float64",
    "salary":     "float64",
    "email":      "object",
    "score":      "float64",
    "country":    "object",
    "signup_date": "object",
}

def save_all():
    os.makedirs("data", exist_ok=True)
    generate_missing_values_dataset().to_csv("data/scenario_missing.csv", index=False)
    generate_schema_mismatch_dataset().to_csv("data/scenario_schema.csv", index=False)
    generate_anomaly_dataset().to_csv("data/scenario_anomaly.csv", index=False)
    with open("data/expected_schema.json", "w") as f:
        json.dump(EXPECTED_SCHEMA, f, indent=2)
    print("✅ All datasets generated.")

if __name__ == "__main__":
    save_all()