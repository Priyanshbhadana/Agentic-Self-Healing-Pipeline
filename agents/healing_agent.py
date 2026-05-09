"""
agents/healing_agent.py
───────────────────────
Self-Healing Agent — Fixed DuckDB writes
Uses db_* write functions (open/write/close) instead of get_db() singleton.
"""

import pandas as pd
import numpy as np
import json as _json
import os, sys, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── GE + MockDB integration ───────────────────────────────────
try:
    from utils.ge_validator import (
        run_pre_healing_suite, run_post_healing_suite, run_custom_suite
    )
    from utils.mock_db import (
        db_snapshot_dataframe, db_insert_quality_score,
        _ensure_schema_exists
    )
    _ensure_schema_exists()
    INTEGRATIONS_AVAILABLE = True
except Exception as _ie:
    INTEGRATIONS_AVAILABLE = False
    print(f"[HealingAgent] Integrations unavailable: {_ie}")

from utils.helpers import PipelineLogger, PipelineState

logger = PipelineLogger("HealingAgent")


# ── Individual fix executors ──────────────────────────────────

def _fill_mean(df, col, params):
    if col not in df.columns: return df, f"SKIP: '{col}' not found"
    mean_val = df[col].mean()
    filled = int(df[col].isna().sum())
    df[col] = df[col].fillna(mean_val)
    return df, f"Filled {filled} nulls in '{col}' with mean={round(mean_val,4)}"

def _fill_median(df, col, params):
    if col not in df.columns: return df, f"SKIP: '{col}' not found"
    if df[col].isna().sum() == 0: return df, f"No nulls in '{col}' — skipped"
    if not pd.api.types.is_numeric_dtype(df[col]):
        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "UNKNOWN"
        filled = int(df[col].isna().sum())
        df[col] = df[col].fillna(mode_val)
        return df, f"Non-numeric '{col}' — filled {filled} nulls with mode='{mode_val}'"
    median_val = df[col].median()
    filled = int(df[col].isna().sum())
    df[col] = df[col].fillna(median_val)
    return df, f"Filled {filled} nulls in '{col}' with median={round(median_val,4)}"

def _fill_mode(df, col, params):
    if col not in df.columns: return df, f"SKIP: '{col}' not found"
    if df[col].isna().sum() == 0: return df, f"No nulls in '{col}' — skipped"
    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "UNKNOWN"
    filled = int(df[col].isna().sum())
    df[col] = df[col].fillna(mode_val)
    return df, f"Filled {filled} nulls in '{col}' with mode='{mode_val}'"

def _fill_constant(df, col, params):
    if col not in df.columns: return df, f"SKIP: '{col}' not found"
    value = params.get("value", 0)
    filled = int(df[col].isna().sum())
    df[col] = df[col].fillna(value)
    return df, f"Filled {filled} nulls in '{col}' with constant='{value}'"

def _drop_rows(df, col, params):
    if col not in df.columns: return df, f"SKIP: '{col}' not found"
    before = len(df)
    df = df.dropna(subset=[col])
    return df, f"Dropped {before - len(df)} rows where '{col}' was null"

def _drop_column(df, col, params):
    if col not in df.columns: return df, f"SKIP: '{col}' not found"
    df = df.drop(columns=[col])
    return df, f"Dropped extra column '{col}'"

def _rename_column(df, col, params):
    old = params.get("old", col); new = params.get("new", col)
    if old not in df.columns: return df, f"SKIP: source column '{old}' not found"
    df = df.rename(columns={old: new})
    return df, f"Renamed column '{old}' → '{new}'"

def _clip_outliers(df, col, params):
    if col not in df.columns: return df, f"SKIP: '{col}' not found"
    lower = params.get("lower"); upper = params.get("upper")
    if lower is None or upper is None:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 3*IQR, Q3 + 3*IQR
    n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df, f"Clipped {n_out} outliers in '{col}' to [{round(lower,2)}, {round(upper,2)}]"

def _cast_dtype(df, col, params):
    if col not in df.columns: return df, f"SKIP: '{col}' not found"
    target = params.get("dtype","float64")
    try:
        df[col] = df[col].astype(target)
        return df, f"Cast '{col}' to {target}"
    except Exception as e:
        return df, f"CAST FAILED for '{col}': {e}"

def _retry_pipeline(df, col, params):
    logger.warn("RETRY_PIPELINE triggered — simulating upstream re-run (mock).")
    return df, "Retry signal sent to upstream pipeline (simulated)"

def _skip_batch(df, col, params):
    return df, f"Batch skipped for '{col}' — flagged for manual review"

ACTION_MAP = {
    "FILL_MEAN": _fill_mean, "FILL_MEDIAN": _fill_median,
    "FILL_MODE": _fill_mode, "FILL_CONSTANT": _fill_constant,
    "DROP_ROWS": _drop_rows, "DROP_COLUMN": _drop_column,
    "RENAME_COLUMN": _rename_column, "CLIP_OUTLIERS": _clip_outliers,
    "CAST_DTYPE": _cast_dtype, "RETRY_PIPELINE": _retry_pipeline,
    "SKIP_BATCH": _skip_batch,
}


# ── Main agent node ───────────────────────────────────────────

def healing_agent(state: PipelineState) -> PipelineState:
    fix_plan = state.get("fix_plan", [])
    issues   = state.get("issues_detected", [])

    if not fix_plan:
        logger.info("No fixes to apply.")
        return {
            **state,
            "healed_data_path":  state["raw_data_path"],
            "removed_data_path": "",
            "quality_report":    {},
            "ge_pre_results":    {},
            "ge_post_results":   {},
            "fixes_applied":     [],
            "logs": ["[HealingAgent] No fixes required."],
        }

    try:
        df = pd.read_csv(state["raw_data_path"])
    except Exception as e:
        logger.error(f"Cannot load data: {e}")
        return {**state, "fixes_applied": [], "final_status": "FAILED",
                "logs": [f"[HealingAgent] Load error: {e}"]}

    df_original = df.copy()
    scenario    = state.get("scenario_name", "unknown")
    run_id      = state.get("run_id", "")
    is_builtin  = scenario in ("missing_values", "schema_mismatch", "data_anomaly")

    # ── GE Pre-healing ────────────────────────────────────────
    ge_pre_results = {}
    if INTEGRATIONS_AVAILABLE:
        try:
            ge_pre_results = (run_pre_healing_suite if is_builtin else run_custom_suite)(
                df_original, scenario
            )
            logger.warn(
                f"GE pre-healing: {ge_pre_results.get('passed',0)}/"
                f"{ge_pre_results.get('total',0)} passed "
                f"({ge_pre_results.get('success_pct',0)}%)"
            )
        except Exception as e:
            logger.warn(f"GE pre-healing failed: {e}")

    # ── MockDB: snapshot pre-healing ──────────────────────────
    if INTEGRATIONS_AVAILABLE and run_id:
        try:
            db_snapshot_dataframe(run_id, "pre_healing", df_original)
        except Exception as e:
            logger.warn(f"MockDB pre-snapshot failed (non-fatal): {e}")

    # ── Apply fixes ───────────────────────────────────────────
    issue_map    = {i["issue_id"]: i for i in issues}
    fixes_applied = []

    for fix in fix_plan:
        iid    = fix["issue_id"]
        action = fix.get("action", "SKIP_BATCH")
        params = fix.get("params") or {}
        col    = issue_map.get(iid, {}).get("column", "")
        executor = ACTION_MAP.get(action, _skip_batch)

        logger.info(f"Applying [{action}] on column='{col}' | issue_id={iid}")
        try:
            df, result_msg = executor(df, col, params)
            logger.success(f"✓ {result_msg}")
            fixes_applied.append({"issue_id": iid, "action": action,
                                   "result": result_msg, "status": "SUCCESS"})
        except Exception as e:
            err = f"Fix [{action}] on '{col}' raised: {e}"
            logger.error(err)
            fixes_applied.append({"issue_id": iid, "action": action,
                                   "result": err, "status": "FAILED"})

    # ── Guard: restore original if all rows dropped ───────────
    if len(df) == 0:
        logger.warn("All rows removed — restoring original to avoid empty output.")
        df = pd.read_csv(state["raw_data_path"])

    # ── GE Post-healing ───────────────────────────────────────
    ge_post_results = {}
    if INTEGRATIONS_AVAILABLE:
        try:
            ge_post_results = (run_post_healing_suite if is_builtin else run_custom_suite)(
                df, scenario
            )
            logger.success(
                f"GE post-healing: {ge_post_results.get('passed',0)}/"
                f"{ge_post_results.get('total',0)} passed "
                f"({ge_post_results.get('success_pct',0)}%)"
            )
        except Exception as e:
            logger.warn(f"GE post-healing failed: {e}")

    # ── Build removed-rows CSV ────────────────────────────────
    os.makedirs("data", exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in scenario)

    try:
        removed_idx = set(df_original.index) - set(df.index)
        df_removed  = df_original.loc[list(removed_idx)].copy() if removed_idx else pd.DataFrame(columns=df_original.columns)
        if not df_removed.empty:
            df_removed["_removal_reason"] = "Dropped during healing"
    except Exception:
        df_removed = pd.DataFrame()

    healed_path  = f"data/healed_{safe_name}.csv"
    removed_path = f"data/removed_{safe_name}.csv"
    report_path  = f"data/report_{safe_name}.json"

    df.to_csv(healed_path, index=False)
    df_removed.to_csv(removed_path, index=False)
    logger.success(f"Healed  → {healed_path} ({len(df)} rows × {len(df.columns)} cols)")
    logger.info(   f"Removed → {removed_path} ({len(df_removed)} rows)")

    # ── MockDB: snapshot post-healing + quality score ─────────
    if INTEGRATIONS_AVAILABLE and run_id:
        try:
            db_snapshot_dataframe(run_id, "post_healing", df)
        except Exception as e:
            logger.warn(f"MockDB post-snapshot failed (non-fatal): {e}")
        try:
            db_insert_quality_score(run_id, scenario, df_original, df)
            logger.info(f"MockDB quality score written for run {run_id}")
        except Exception as e:
            logger.warn(f"MockDB quality score failed (non-fatal): {e}")

    # ── Quality report JSON ───────────────────────────────────
    quality_report = {
        "run_id":          run_id,
        "scenario":        scenario,
        "generated_at":    datetime.datetime.now().isoformat(),
        "original_shape":  {"rows": len(df_original), "cols": len(df_original.columns)},
        "healed_shape":    {"rows": len(df),           "cols": len(df.columns)},
        "removed_rows":    len(df_removed),
        "null_before":     int(df_original.isna().sum().sum()),
        "null_after":      int(df.isna().sum().sum()),
        "fixes_applied":   fixes_applied,
        "ge_pre_validation":  ge_pre_results,
        "ge_post_validation": ge_post_results,
        "column_profiles": {
            col: {
                "dtype":      str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "unique":     int(df[col].nunique()),
                "sample":     df[col].dropna().head(3).tolist(),
            } for col in df.columns
        },
    }
    with open(report_path, "w") as f:
        _json.dump(quality_report, f, indent=2, default=str)
    logger.info(f"Quality report → {report_path}")

    success_count = sum(1 for f in fixes_applied if f["status"] == "SUCCESS")
    total         = len(fixes_applied)

    return {
        **state,
        "healed_data_path":  healed_path,
        "removed_data_path": removed_path,
        "quality_report":    quality_report,
        "ge_pre_results":    ge_pre_results,
        "ge_post_results":   ge_post_results,
        "fixes_applied":     fixes_applied,
        "final_status":      "SUCCESS" if success_count == total else "PARTIAL",
        "logs": [f"[HealingAgent] Applied {success_count}/{total} fixes. "
                 f"Healed={healed_path} Removed={removed_path}"],
    }