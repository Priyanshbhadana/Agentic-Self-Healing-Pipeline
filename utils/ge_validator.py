"""
utils/ge_validator.py
─────────────────────
Great Expectations (GE) Integration
Runs data quality validation BEFORE and AFTER healing.

Expectation suites:
  pre_healing_suite  — validates raw incoming data
  post_healing_suite — validates healed output meets quality bar

Results are stored in MockDB and surfaced in the Streamlit UI.
"""

import pandas as pd
import numpy as np
import json, os, datetime
from typing import Any

# ── Pure-Python GE validation (no file-system context needed) ─

class ExpectationResult:
    """Lightweight result object mirroring GE's ValidationResult."""
    def __init__(self, expectation: str, column: str,
                 passed: bool, observed: Any, expected: Any, detail: str = ""):
        self.expectation = expectation
        self.column      = column
        self.passed      = passed
        self.observed    = observed
        self.expected    = expected
        self.detail      = detail

    def to_dict(self) -> dict:
        return {
            "expectation": self.expectation,
            "column":      self.column,
            "passed":      self.passed,
            "observed":    str(self.observed),
            "expected":    str(self.expected),
            "detail":      self.detail,
        }


class GEValidator:
    """
    Lightweight Great Expectations–style validator.
    Runs a suite of expectations against a DataFrame and returns
    structured pass/fail results — identical API to real GE.
    """

    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.results: list[ExpectationResult] = []

    # ── Core expectations ─────────────────────────────────────

    def expect_column_values_to_not_be_null(
            self, df: pd.DataFrame, column: str,
            mostly: float = 1.0) -> ExpectationResult:
        """Column should have at most (1-mostly)*100% nulls."""
        if column not in df.columns:
            r = ExpectationResult(
                "expect_column_values_to_not_be_null", column,
                False, "column missing", f"≤{int((1-mostly)*100)}% null",
                f"Column '{column}' does not exist"
            )
        else:
            null_pct = df[column].isna().mean()
            passed = null_pct <= (1 - mostly)
            r = ExpectationResult(
                "expect_column_values_to_not_be_null", column,
                passed,
                f"{round(null_pct*100,1)}% null",
                f"≤{round((1-mostly)*100,1)}% null",
                "" if passed else f"Found {round(null_pct*100,1)}% nulls — exceeds threshold"
            )
        self.results.append(r)
        return r

    def expect_column_values_to_be_between(
            self, df: pd.DataFrame, column: str,
            min_val: float, max_val: float) -> ExpectationResult:
        """All values in column should be within [min_val, max_val]."""
        if column not in df.columns:
            r = ExpectationResult(
                "expect_column_values_to_be_between", column,
                False, "column missing", f"[{min_val}, {max_val}]",
                f"Column '{column}' does not exist"
            )
        else:
            series  = df[column].dropna()
            bad     = series[(series < min_val) | (series > max_val)]
            passed  = len(bad) == 0
            r = ExpectationResult(
                "expect_column_values_to_be_between", column,
                passed,
                f"min={round(float(series.min()),2)}, max={round(float(series.max()),2)}" if len(series) else "empty",
                f"[{min_val}, {max_val}]",
                "" if passed else f"{len(bad)} values outside range"
            )
        self.results.append(r)
        return r

    def expect_column_values_to_be_of_type(
            self, df: pd.DataFrame, column: str,
            expected_type: str) -> ExpectationResult:
        """Column dtype should match expected_type."""
        if column not in df.columns:
            r = ExpectationResult(
                "expect_column_values_to_be_of_type", column,
                False, "column missing", expected_type,
                f"Column '{column}' does not exist"
            )
        else:
            actual = str(df[column].dtype)
            passed = expected_type.lower() in actual.lower()
            r = ExpectationResult(
                "expect_column_values_to_be_of_type", column,
                passed, actual, expected_type,
                "" if passed else f"Expected {expected_type}, got {actual}"
            )
        self.results.append(r)
        return r

    def expect_table_row_count_to_be_between(
            self, df: pd.DataFrame,
            min_rows: int, max_rows: int = 10_000_000) -> ExpectationResult:
        """Row count should be in [min_rows, max_rows]."""
        n      = len(df)
        passed = min_rows <= n <= max_rows
        r = ExpectationResult(
            "expect_table_row_count_to_be_between", "__table__",
            passed, n, f"[{min_rows}, {max_rows}]",
            "" if passed else f"Row count {n} outside [{min_rows}, {max_rows}]"
        )
        self.results.append(r)
        return r

    def expect_column_to_exist(
            self, df: pd.DataFrame, column: str) -> ExpectationResult:
        """Column must exist in the DataFrame."""
        passed = column in df.columns
        r = ExpectationResult(
            "expect_column_to_exist", column,
            passed, "present" if passed else "missing", "present",
            "" if passed else f"Required column '{column}' is absent"
        )
        self.results.append(r)
        return r

    def expect_column_values_to_be_unique(
            self, df: pd.DataFrame, column: str) -> ExpectationResult:
        """Column should have no duplicate values."""
        if column not in df.columns:
            r = ExpectationResult(
                "expect_column_values_to_be_unique", column,
                False, "column missing", "unique",
                f"Column '{column}' does not exist"
            )
        else:
            dupes  = df[column].duplicated().sum()
            passed = dupes == 0
            r = ExpectationResult(
                "expect_column_values_to_be_unique", column,
                passed, f"{dupes} duplicates", "0 duplicates",
                "" if passed else f"{dupes} duplicate values found"
            )
        self.results.append(r)
        return r

    # ── Summary ───────────────────────────────────────────────

    def summary(self) -> dict:
        total   = len(self.results)
        passed  = sum(1 for r in self.results if r.passed)
        failed  = total - passed
        score   = round(passed / max(total, 1) * 100, 1)
        return {
            "suite":         self.suite_name,
            "total":         total,
            "passed":        passed,
            "failed":        failed,
            "success_pct":   score,
            "evaluated_at":  datetime.datetime.now().isoformat(),
            "results":       [r.to_dict() for r in self.results],
        }

    def passed_all(self) -> bool:
        return all(r.passed for r in self.results)


# ── Suite runners ─────────────────────────────────────────────

def run_pre_healing_suite(df: pd.DataFrame, scenario: str) -> dict:
    """
    Validate RAW data before healing.
    Expectations are deliberately strict — we EXPECT failures here.
    """
    v = GEValidator(f"pre_healing_{scenario}")

    # Table-level
    v.expect_table_row_count_to_be_between(df, min_rows=1)

    # Column existence (built-in schema columns)
    expected_cols = ["user_id","age","salary","email","score","country","signup_date"]
    for col in expected_cols:
        v.expect_column_to_exist(df, col)

    # Null checks — strict (should fail on raw dirty data)
    for col in df.columns:
        v.expect_column_values_to_not_be_null(df, col, mostly=1.0)

    # Range checks on numeric columns
    if "age"    in df.columns: v.expect_column_values_to_be_between(df, "age",    0,  120)
    if "salary" in df.columns: v.expect_column_values_to_be_between(df, "salary", 0,  1_000_000)
    if "score"  in df.columns: v.expect_column_values_to_be_between(df, "score",  0,  100)

    # Type checks
    for col in ["age","salary","score"]:
        if col in df.columns:
            v.expect_column_values_to_be_of_type(df, col, "float")

    return v.summary()


def run_post_healing_suite(df: pd.DataFrame, scenario: str) -> dict:
    """
    Validate HEALED data after fixing.
    Expectations are relaxed — we check the data is now acceptable.
    """
    v = GEValidator(f"post_healing_{scenario}")

    # Table-level
    v.expect_table_row_count_to_be_between(df, min_rows=1)

    # Null checks — relaxed (allow up to 5% nulls after healing)
    for col in df.columns:
        v.expect_column_values_to_not_be_null(df, col, mostly=0.95)

    # Range checks — same bounds
    if "age"    in df.columns: v.expect_column_values_to_be_between(df, "age",    0,  120)
    if "salary" in df.columns: v.expect_column_values_to_be_between(df, "salary", 0,  1_000_000)
    if "score"  in df.columns: v.expect_column_values_to_be_between(df, "score",  0,  100)

    # user_id should be unique
    if "user_id" in df.columns:
        v.expect_column_values_to_be_unique(df, "user_id")

    return v.summary()


def run_custom_suite(df: pd.DataFrame, scenario: str) -> dict:
    """
    For uploaded CSVs with unknown schema.
    Runs generic quality checks that apply to any dataset.
    """
    v = GEValidator(f"custom_{scenario}")

    v.expect_table_row_count_to_be_between(df, min_rows=1)

    # Null check on every column (lenient — 80% non-null)
    for col in df.columns:
        v.expect_column_values_to_not_be_null(df, col, mostly=0.8)

    # Range check on all numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        if len(series) > 0:
            q1, q3 = series.quantile(0.01), series.quantile(0.99)
            v.expect_column_values_to_be_between(
                df, col,
                float(q1) * 3 if q1 < 0 else float(q1) / 3,
                float(q3) * 3,
            )

    return v.summary()