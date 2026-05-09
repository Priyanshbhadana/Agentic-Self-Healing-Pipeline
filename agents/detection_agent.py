"""
agents/detection_agent.py
─────────────────────────
Detection Agent
  Reads the raw dataset, runs statistical + structural checks,
  and populates issues_detected in the shared state.

  No LLM call here — this is fast, deterministic scanning.
  The LLM reasoning kicks in at Classification and Decision stages.
"""

import pandas as pd
import numpy as np
import json
import uuid
import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import PipelineLogger, PipelineState

logger = PipelineLogger("DetectionAgent")


# ── Helpers ──────────────────────────────────────────────────

def _load_expected_schema(data_path: str = None) -> dict:
    """
    Load expected schema.
    - For built-in scenarios: use data/expected_schema.json
    - For custom uploads: INFER schema from the data itself
      (we treat the uploaded CSV's own dtypes as the expected schema,
       so schema checks are skipped and we focus on nulls + anomalies)
    """
    # Try the standard schema file first
    candidates = [
        "data/expected_schema.json",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "expected_schema.json"),
    ]
    for schema_path in candidates:
        if os.path.exists(schema_path):
            with open(schema_path) as f:
                return json.load(f)

    # Fallback: infer schema from the CSV itself
    if data_path and os.path.exists(data_path):
        try:
            df_sample = pd.read_csv(data_path, nrows=5)
            return {col: str(df_sample[col].dtype) for col in df_sample.columns}
        except Exception:
            pass
    return {}


def _check_missing_values(df: pd.DataFrame) -> list:
    """Detect columns with null / NaN values."""
    issues = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            pct = round(null_count / len(df) * 100, 2)
            issues.append({
                "issue_id":   str(uuid.uuid4())[:8],
                "type":       "MISSING_VALUES",
                "column":     col,
                "null_count": null_count,
                "null_pct":   pct,
                "severity":   "HIGH" if pct > 20 else "MEDIUM",
                "detail":     f"Column '{col}' has {null_count} missing values ({pct}%)",
            })
    return issues


def _check_schema(df: pd.DataFrame, expected: dict) -> list:
    """Detect missing expected columns and unexpected extra columns."""
    issues = []
    actual_cols = set(df.columns)
    expected_cols = set(expected.keys())

    # Missing expected columns
    missing = expected_cols - actual_cols
    for col in missing:
        issues.append({
            "issue_id": str(uuid.uuid4())[:8],
            "type":     "SCHEMA_MISSING_COLUMN",
            "column":   col,
            "severity": "HIGH",
            "detail":   f"Expected column '{col}' is absent from the dataset",
        })

    # Unexpected extra columns
    extra = actual_cols - expected_cols
    for col in extra:
        issues.append({
            "issue_id": str(uuid.uuid4())[:8],
            "type":     "SCHEMA_EXTRA_COLUMN",
            "column":   col,
            "severity": "MEDIUM",
            "detail":   f"Unexpected column '{col}' found — not in schema contract",
        })

    return issues


def _check_dtype_mismatches(df: pd.DataFrame, expected: dict) -> list:
    """Check if actual dtypes match the expected schema dtypes."""
    issues = []
    for col, exp_dtype in expected.items():
        if col not in df.columns:
            continue  # already caught above
        actual_dtype = str(df[col].dtype)
        if actual_dtype != exp_dtype:
            issues.append({
                "issue_id":      str(uuid.uuid4())[:8],
                "type":          "DTYPE_MISMATCH",
                "column":        col,
                "expected_dtype": exp_dtype,
                "actual_dtype":  actual_dtype,
                "severity":      "MEDIUM",
                "detail":        f"Column '{col}': expected {exp_dtype}, got {actual_dtype}",
            })
    return issues


def _check_anomalies(df: pd.DataFrame) -> list:
    """Detect statistical outliers using IQR for numeric columns."""
    issues = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) > 0:
            issues.append({
                "issue_id":      str(uuid.uuid4())[:8],
                "type":          "DATA_ANOMALY",
                "column":        col,
                "outlier_count": int(len(outliers)),
                "outlier_values": outliers.tolist()[:5],
                "bounds":        {"lower": round(lower, 2), "upper": round(upper, 2)},
                "severity":      "HIGH" if len(outliers) > 5 else "LOW",
                "detail":        f"Column '{col}' has {len(outliers)} outlier(s) outside IQR bounds",
            })
    return issues


def _check_row_count(df: pd.DataFrame) -> list:
    """Flag suspiciously small datasets."""
    issues = []
    if len(df) == 0:
        issues.append({
            "issue_id": str(uuid.uuid4())[:8],
            "type":     "EMPTY_DATASET",
            "severity": "CRITICAL",
            "detail":   "Dataset is completely empty — pipeline produced 0 rows",
        })
    elif len(df) < 10:
        issues.append({
            "issue_id": str(uuid.uuid4())[:8],
            "type":     "LOW_ROW_COUNT",
            "severity": "MEDIUM",
            "detail":   f"Dataset has only {len(df)} rows — possible upstream truncation",
        })
    return issues


# ── Main agent node ──────────────────────────────────────────

def detection_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Detection Agent
    Reads raw CSV, runs all checks, returns updated state.
    """
    logger.info(f"Starting detection on: {state['raw_data_path']}")

    try:
        df = pd.read_csv(state["raw_data_path"])
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {**state, "issues_detected": [], "detection_summary": f"LOAD ERROR: {e}",
                "logs": [f"[DetectionAgent] Load failed: {e}"]}

    logger.info(f"Loaded dataset — {len(df)} rows × {len(df.columns)} columns")

    expected_schema = _load_expected_schema(state["raw_data_path"])
    all_issues = []

    # For custom/uploaded CSVs skip built-in schema checks — only run
    # null + anomaly detection so we don't flood with false positives.
    BUILTIN_SCENARIOS = {"missing_values", "schema_mismatch", "data_anomaly"}
    is_custom = state.get("scenario_name", "") not in BUILTIN_SCENARIOS

    all_issues += _check_row_count(df)
    all_issues += _check_missing_values(df)
    if not is_custom:
        all_issues += _check_schema(df, expected_schema)
        all_issues += _check_dtype_mismatches(df, expected_schema)
    all_issues += _check_anomalies(df)

    if all_issues:
        for issue in all_issues:
            logger.warn(f"Issue detected → [{issue['type']}] {issue['detail']}")
    else:
        logger.success("No issues detected — dataset looks healthy!")

    summary = (
        f"Found {len(all_issues)} issue(s) in '{state['scenario_name']}'. "
        f"Types: {list(set(i['type'] for i in all_issues))}"
    ) if all_issues else "Dataset passed all checks."

    logger.info(f"Detection summary: {summary}")

    return {
        **state,
        "issues_detected": all_issues,
        "detection_summary": summary,
        "logs": [f"[DetectionAgent] {summary}"],
    }