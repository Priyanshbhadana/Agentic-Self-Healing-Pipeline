"""
utils/sqlite_export.py
──────────────────────
SQLite Export Utility
Saves healed pipeline data + metadata into a permanent SQLite database.

Tables saved:
  healed_data        — the fully cleaned dataset
  removed_rows       — rows that were dropped during healing
  pipeline_run_meta  — run metadata (issues, fixes, quality score)
  issue_log          — all detected issues
  fix_log            — all applied fixes
  ge_results         — Great Expectations validation results
"""

import sqlite3
import pandas as pd
import json
import os
import datetime

SQLITE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "pipeline_results.sqlite"
)


def export_to_sqlite(
    result: dict,
    df_healed: pd.DataFrame,
    df_removed: pd.DataFrame,
    db_path: str = SQLITE_PATH,
) -> dict:
    """
    Export a full pipeline run result to SQLite.
    Returns a summary dict with table names and row counts.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    summary = {}

    scenario  = result.get("scenario_name", "unknown")
    run_id    = result.get("run_id", "unknown")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize table name: only alphanumeric + underscore
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in scenario)

    try:
        # ── 1. Healed data table ──────────────────────────────
        if df_healed is not None and not df_healed.empty:
            table_name = f"healed_{safe_name}"
            df_export = df_healed.copy()
            df_export["_run_id"]    = run_id
            df_export["_exported_at"] = timestamp
            df_export.to_sql(table_name, conn, if_exists="replace", index=False)
            summary["healed_table"] = {
                "name": table_name,
                "rows": len(df_healed),
                "cols": len(df_healed.columns),
            }

        # ── 2. Removed rows table ─────────────────────────────
        if df_removed is not None and not df_removed.empty:
            removed_table = f"removed_{safe_name}"
            df_rem_export = df_removed.copy()
            df_rem_export["_run_id"]      = run_id
            df_rem_export["_exported_at"] = timestamp
            df_rem_export.to_sql(removed_table, conn, if_exists="replace", index=False)
            summary["removed_table"] = {
                "name": removed_table,
                "rows": len(df_removed),
            }
        else:
            summary["removed_table"] = {"name": "—", "rows": 0}

        # ── 3. Pipeline run metadata ──────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_run_meta (
                run_id          TEXT,
                scenario_name   TEXT,
                exported_at     TEXT,
                final_status    TEXT,
                total_issues    INTEGER,
                total_fixes     INTEGER,
                null_before     INTEGER,
                null_after      INTEGER,
                rows_original   INTEGER,
                rows_healed     INTEGER,
                rows_removed    INTEGER,
                quality_score   REAL,
                ge_pre_pct      REAL,
                ge_post_pct     REAL
            )
        """)
        qr = result.get("quality_report", {})
        ge_pre  = result.get("ge_pre_results",  {})
        ge_post = result.get("ge_post_results", {})

        conn.execute("""
            INSERT INTO pipeline_run_meta VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, [
            run_id, scenario, timestamp,
            result.get("final_status", "UNKNOWN"),
            len(result.get("issues_detected", [])),
            len(result.get("fixes_applied",   [])),
            qr.get("null_before", 0),
            qr.get("null_after",  0),
            qr.get("original_shape", {}).get("rows", 0),
            qr.get("healed_shape",   {}).get("rows", 0),
            len(df_removed) if df_removed is not None else 0,
            0,  # quality_score computed below
            ge_pre.get("success_pct",  0),
            ge_post.get("success_pct", 0),
        ])
        conn.commit()
        summary["meta_table"] = "pipeline_run_meta"

        # ── 4. Issue log ──────────────────────────────────────
        issues      = result.get("issues_detected", [])
        classifs    = result.get("classifications", [])
        class_map   = {c["issue_id"]: c for c in classifs}

        if issues:
            issue_rows = []
            for iss in issues:
                iid = iss.get("issue_id", "")
                cls = class_map.get(iid, {})
                issue_rows.append({
                    "run_id":      run_id,
                    "scenario":    scenario,
                    "issue_id":    iid,
                    "issue_type":  iss.get("type", ""),
                    "column_name": iss.get("column", ""),
                    "severity":    iss.get("severity", ""),
                    "detail":      iss.get("detail", ""),
                    "category":    cls.get("category", ""),
                    "confidence":  cls.get("confidence", 0),
                    "exported_at": timestamp,
                })
            df_issues = pd.DataFrame(issue_rows)
            df_issues.to_sql("issue_log", conn, if_exists="append", index=False)
            summary["issue_log"] = {"name": "issue_log", "rows": len(df_issues)}

        # ── 5. Fix log ────────────────────────────────────────
        fixes    = result.get("fixes_applied", [])
        fix_plan = result.get("fix_plan", [])
        plan_map = {f["issue_id"]: f for f in fix_plan}

        if fixes:
            fix_rows = []
            for fix in fixes:
                iid  = fix.get("issue_id", "")
                plan = plan_map.get(iid, {})
                fix_rows.append({
                    "run_id":      run_id,
                    "scenario":    scenario,
                    "issue_id":    iid,
                    "action":      fix.get("action", ""),
                    "params":      json.dumps(plan.get("params", {})),
                    "status":      fix.get("status", ""),
                    "result":      fix.get("result", ""),
                    "rationale":   plan.get("rationale", ""),
                    "confidence":  plan.get("confidence", 0),
                    "exported_at": timestamp,
                })
            df_fixes = pd.DataFrame(fix_rows)
            df_fixes.to_sql("fix_log", conn, if_exists="append", index=False)
            summary["fix_log"] = {"name": "fix_log", "rows": len(df_fixes)}

        # ── 6. GE results ─────────────────────────────────────
        ge_rows = []
        for stage, ge_result in [("pre_healing", ge_pre), ("post_healing", ge_post)]:
            for r in ge_result.get("results", []):
                ge_rows.append({
                    "run_id":        run_id,
                    "scenario":      scenario,
                    "stage":         stage,
                    "expectation":   r.get("expectation", ""),
                    "column_name":   r.get("column", ""),
                    "passed":        1 if r.get("passed") else 0,
                    "observed":      str(r.get("observed", "")),
                    "expected":      str(r.get("expected", "")),
                    "detail":        r.get("detail", ""),
                    "exported_at":   timestamp,
                })
        if ge_rows:
            df_ge = pd.DataFrame(ge_rows)
            df_ge.to_sql("ge_results", conn, if_exists="append", index=False)
            summary["ge_results"] = {"name": "ge_results", "rows": len(df_ge)}

        summary["db_path"]   = db_path
        summary["run_id"]    = run_id
        summary["timestamp"] = timestamp
        summary["success"]   = True

    except Exception as e:
        summary["success"] = False
        summary["error"]   = str(e)
    finally:
        conn.close()

    return summary


def get_sqlite_tables(db_path: str = SQLITE_PATH) -> list[str]:
    """Return list of all tables in the SQLite DB."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


def query_sqlite(sql: str, db_path: str = SQLITE_PATH) -> pd.DataFrame:
    """Run a SELECT query against the SQLite DB."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df


def get_sqlite_db_size(db_path: str = SQLITE_PATH) -> str:
    """Return human-readable size of the SQLite file."""
    if not os.path.exists(db_path):
        return "0 B"
    size = os.path.getsize(db_path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"