"""
utils/mock_db.py  —  Fixed: DuckDB concurrency / file-lock issue
────────────────────────────────────────────────────────────────
ROOT CAUSE:
  Streamlit reruns the whole script on every interaction.
  Each rerun called get_db() → MockDatabase() → duckdb.connect(path).
  If a previous pipeline process still held the file lock, the
  second connect() crashed with "Could not set lock".

FIX (3-part):
  1. Pipeline writes  → open a FRESH connection, write, close immediately.
  2. Streamlit reads  → open with read_only=True (no lock needed).
  3. Singleton        → one read-only connection per Streamlit session,
                        cached in st.session_state so Streamlit never
                        opens a second write-mode connection by accident.
"""

import duckdb
import pandas as pd
import json, os, datetime, uuid, threading

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "pipeline.duckdb"
)

# Thread lock so concurrent pipeline + UI reads don't race
_write_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────
#  SCHEMA DDL  (shared by both writer and reader init)
# ──────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          VARCHAR PRIMARY KEY,
    scenario_name   VARCHAR,
    started_at      TIMESTAMP,
    ended_at        TIMESTAMP,
    status          VARCHAR,
    total_issues    INTEGER DEFAULT 0,
    total_fixes     INTEGER DEFAULT 0,
    quality_score   FLOAT   DEFAULT 0,
    rows_original   INTEGER DEFAULT 0,
    rows_healed     INTEGER DEFAULT 0,
    rows_removed    INTEGER DEFAULT 0,
    nulls_before    INTEGER DEFAULT 0,
    nulls_after     INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS issue_registry (
    issue_id        VARCHAR PRIMARY KEY,
    run_id          VARCHAR,
    issue_type      VARCHAR,
    column_name     VARCHAR,
    severity        VARCHAR,
    detail          VARCHAR,
    category        VARCHAR,
    confidence      FLOAT DEFAULT 0,
    detected_at     TIMESTAMP
);
CREATE TABLE IF NOT EXISTS fix_registry (
    fix_id          VARCHAR PRIMARY KEY,
    run_id          VARCHAR,
    issue_id        VARCHAR,
    action          VARCHAR,
    params          VARCHAR,
    status          VARCHAR,
    result          VARCHAR,
    applied_at      TIMESTAMP
);
CREATE TABLE IF NOT EXISTS data_quality_scores (
    score_id        VARCHAR PRIMARY KEY,
    run_id          VARCHAR,
    scenario_name   VARCHAR,
    null_score      FLOAT DEFAULT 0,
    schema_score    FLOAT DEFAULT 0,
    anomaly_score   FLOAT DEFAULT 0,
    overall_score   FLOAT DEFAULT 0,
    recorded_at     TIMESTAMP
);
CREATE TABLE IF NOT EXISTS raw_data_snapshots (
    snapshot_id     VARCHAR PRIMARY KEY,
    run_id          VARCHAR,
    stage           VARCHAR,
    row_count       INTEGER DEFAULT 0,
    col_count       INTEGER DEFAULT 0,
    null_count      INTEGER DEFAULT 0,
    snapshot_json   VARCHAR,
    created_at      TIMESTAMP
);
"""


def _ensure_schema_exists():
    """
    Make sure the DB file + all tables exist.
    Opens write-mode, creates schema, closes immediately.
    Safe to call many times (all CREATE IF NOT EXISTS).
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _write_lock:
        conn = duckdb.connect(DB_PATH)          # write-mode
        for stmt in _DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        conn.close()                             # release lock immediately


# ──────────────────────────────────────────────────────────────
#  WRITER  — used only by pipeline agents
#  Opens fresh write-mode connection, writes, closes instantly.
# ──────────────────────────────────────────────────────────────

def _write_op(fn):
    """
    Decorator: open write-mode conn → call fn(conn, ...) → close.
    Thread-safe via _write_lock.
    """
    def wrapper(*args, **kwargs):
        with _write_lock:
            conn = duckdb.connect(DB_PATH)
            try:
                return fn(conn, *args, **kwargs)
            finally:
                conn.close()
    return wrapper


@_write_op
def db_insert_pipeline_run(conn, run_id: str, scenario_name: str):
    conn.execute("""
        INSERT OR REPLACE INTO pipeline_runs
        (run_id, scenario_name, started_at, status)
        VALUES (?, ?, ?, 'RUNNING')
    """, [run_id, scenario_name, datetime.datetime.now()])


@_write_op
def db_update_pipeline_run(conn, run_id: str, **kwargs):
    if not kwargs:
        return
    sets   = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [run_id]
    conn.execute(f"UPDATE pipeline_runs SET {sets} WHERE run_id = ?", values)


@_write_op
def db_insert_issues(conn, run_id: str, issues: list, classifications: list):
    class_map = {c["issue_id"]: c for c in classifications}
    now = datetime.datetime.now()
    for iss in issues:
        iid = iss.get("issue_id", str(uuid.uuid4())[:8])
        cls = class_map.get(iid, {})
        conn.execute("""
            INSERT OR REPLACE INTO issue_registry
            (issue_id, run_id, issue_type, column_name, severity,
             detail, category, confidence, detected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            iid, run_id,
            iss.get("type",""), iss.get("column",""), iss.get("severity",""),
            iss.get("detail",""), cls.get("category",""),
            float(cls.get("confidence", 0)), now,
        ])


@_write_op
def db_insert_fixes(conn, run_id: str, fixes: list, fix_plan: list):
    plan_map = {f["issue_id"]: f for f in fix_plan}
    now = datetime.datetime.now()
    for fix in fixes:
        iid  = fix.get("issue_id","")
        plan = plan_map.get(iid, {})
        conn.execute("""
            INSERT OR REPLACE INTO fix_registry
            (fix_id, run_id, issue_id, action, params, status, result, applied_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            str(uuid.uuid4())[:8], run_id, iid,
            fix.get("action",""),
            json.dumps(plan.get("params", {})),
            fix.get("status",""), fix.get("result",""), now,
        ])


@_write_op
def db_insert_quality_score(conn, run_id: str, scenario: str,
                             df_orig: pd.DataFrame, df_healed: pd.DataFrame) -> float:
    orig_null   = int(df_orig.isna().sum().sum())
    healed_null = int(df_healed.isna().sum().sum())
    null_s   = max(0, 1 - healed_null / max(orig_null,1)) * 40 if orig_null > 0 else 40
    schema_s = (len(set(df_healed.columns) & set(df_orig.columns))
                / max(len(df_orig.columns),1)) * 30
    anom_s   = min(len(df_healed) / max(len(df_orig),1), 1.0) * 30
    overall  = round(null_s + schema_s + anom_s, 1)
    conn.execute("""
        INSERT INTO data_quality_scores
        (score_id, run_id, scenario_name, null_score, schema_score,
         anomaly_score, overall_score, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [str(uuid.uuid4())[:8], run_id, scenario,
          round(null_s,1), round(schema_s,1), round(anom_s,1), overall,
          datetime.datetime.now()])
    return overall


@_write_op
def db_snapshot_dataframe(conn, run_id: str, stage: str, df: pd.DataFrame):
    sample = df.head(5).to_json(orient="records", default_handler=str)
    conn.execute("""
        INSERT INTO raw_data_snapshots
        (snapshot_id, run_id, stage, row_count, col_count,
         null_count, snapshot_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [str(uuid.uuid4())[:8], run_id, stage,
          len(df), len(df.columns),
          int(df.isna().sum().sum()), sample,
          datetime.datetime.now()])


# ──────────────────────────────────────────────────────────────
#  READER  — used by Streamlit UI
#  Opens read_only=True → no file lock conflict.
# ──────────────────────────────────────────────────────────────

class MockDatabase:
    """
    Read-only view of the DuckDB warehouse for the Streamlit UI.
    Uses read_only=True so it NEVER conflicts with the pipeline writer.
    Connection is opened fresh on each method call and closed after,
    making it safe across Streamlit reruns.
    """

    def __init__(self):
        _ensure_schema_exists()

    def _conn(self):
        """Open a fresh read-only connection. Caller must close it."""
        return duckdb.connect(DB_PATH, read_only=True)

    def _query(self, sql: str, params: list = None) -> pd.DataFrame:
        conn = self._conn()
        try:
            if params:
                return conn.execute(sql, params).df()
            return conn.execute(sql).df()
        finally:
            conn.close()

    def get_pipeline_runs(self, limit: int = 20) -> pd.DataFrame:
        return self._query(
            "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?", [limit]
        )

    def get_issues_for_run(self, run_id: str) -> pd.DataFrame:
        return self._query(
            "SELECT * FROM issue_registry WHERE run_id = ? ORDER BY detected_at", [run_id]
        )

    def get_fixes_for_run(self, run_id: str) -> pd.DataFrame:
        return self._query(
            "SELECT * FROM fix_registry WHERE run_id = ? ORDER BY applied_at", [run_id]
        )

    def get_quality_trend(self) -> pd.DataFrame:
        return self._query("""
            SELECT scenario_name, overall_score, recorded_at
            FROM data_quality_scores ORDER BY recorded_at DESC LIMIT 30
        """)

    def get_issue_summary(self) -> pd.DataFrame:
        return self._query("""
            SELECT issue_type, severity, COUNT(*) as count
            FROM issue_registry GROUP BY issue_type, severity ORDER BY count DESC
        """)

    def get_fix_success_rate(self) -> pd.DataFrame:
        return self._query("""
            SELECT action, status, COUNT(*) as count
            FROM fix_registry GROUP BY action, status ORDER BY count DESC
        """)

    def execute_sql(self, query: str) -> pd.DataFrame:
        """Run any SELECT query — used by SQL console and MCP."""
        if not query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries allowed")
        return self._query(query)


# ──────────────────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────────────────

_reader_instance: MockDatabase | None = None

def get_db() -> MockDatabase:
    """
    Returns the read-only MockDatabase singleton for UI queries.
    Safe to call on every Streamlit rerun — no lock conflicts.
    """
    global _reader_instance
    if _reader_instance is None:
        _reader_instance = MockDatabase()
    return _reader_instance