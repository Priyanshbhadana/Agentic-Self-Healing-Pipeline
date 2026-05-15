"""
mcp_server/pipeline_mcp_server.py
──────────────────────────────────
MCP (Model Context Protocol) Server
Exposes the Self-Healing Pipeline as callable tools that any
MCP-compatible client (Gemini/Claude Desktop, etc.) can invoke.

Tools exposed:
  run_pipeline_tool        — run a scenario end-to-end
  query_database_tool      — SQL query against MockDB
  get_run_history_tool     — list past pipeline runs
  get_issue_stats_tool     — aggregate issue statistics
  validate_data_tool       — run GE suite on a CSV
  get_quality_scores_tool  — fetch quality score trend

Run standalone:
  python mcp_server/pipeline_mcp_server.py

Or import and embed in app.py for in-process use.
"""

import sys, os, json, asyncio, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd


# ── Tool definitions ──────────────────────────────────────────

TOOLS = [
    {
        "name": "run_pipeline_tool",
        "description": (
            "Run the self-healing data pipeline on a scenario. "
            "Detects issues, classifies them, decides fixes, and heals the data automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario_name": {
                    "type": "string",
                    "description": "One of: missing_values, schema_mismatch, data_anomaly, or a custom CSV name",
                    "enum": ["missing_values", "schema_mismatch", "data_anomaly"],
                },
                "save_to_db": {
                    "type": "boolean",
                    "description": "Whether to save results to MockDB (default true)",
                    "default": True,
                },
            },
            "required": ["scenario_name"],
        },
    },
    {
        "name": "query_database_tool",
        "description": (
            "Execute a SQL query against the pipeline MockDB (DuckDB). "
            "Available tables: pipeline_runs, issue_registry, fix_registry, "
            "data_quality_scores, raw_data_snapshots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL SELECT query to execute",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default 20)",
                    "default": 20,
                },
            },
            "required": ["sql"],
        },
    },
    {
        "name": "get_run_history_tool",
        "description": "Get the history of all pipeline runs with their stats and quality scores.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max runs to return (default 10)",
                    "default": 10,
                },
            },
        },
    },
    {
        "name": "get_issue_stats_tool",
        "description": "Get aggregated statistics about all issues ever detected across all runs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Optional: filter to a specific run_id",
                },
            },
        },
    },
    {
        "name": "validate_data_tool",
        "description": (
            "Run Great Expectations validation suite on a CSV file. "
            "Returns pass/fail for each expectation with details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "csv_path": {
                    "type": "string",
                    "description": "Path to CSV file to validate",
                },
                "suite_type": {
                    "type": "string",
                    "description": "Which suite: pre_healing, post_healing, or custom",
                    "enum": ["pre_healing", "post_healing", "custom"],
                    "default": "pre_healing",
                },
                "scenario": {
                    "type": "string",
                    "description": "Scenario name for the suite",
                    "default": "default",
                },
            },
            "required": ["csv_path"],
        },
    },
    {
        "name": "get_quality_scores_tool",
        "description": "Get data quality score trend across all pipeline runs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario_filter": {
                    "type": "string",
                    "description": "Optional: filter to a specific scenario name",
                },
            },
        },
    },
    {
        "name": "run_b1_agent_tool",
        "description": (
            "Run the B1 Ingestion Quality Agent. "
            "Profiles data, generates quality rules via LLM, validates, and auto-heals violations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario_name": {
                    "type": "string",
                    "description": "Scenario name",
                    "enum": ["missing_values", "schema_mismatch", "data_anomaly"],
                },
            },
            "required": ["scenario_name"],
        },
    },
    {
        "name": "run_b2_agent_tool",
        "description": (
            "Run the B2 Lineage & Governance Agent. "
            "Parses SQL, extracts lineage, tags PII, enriches catalogue, generates GDPR compliance report."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario_name": {
                    "type": "string",
                    "description": "Scenario name",
                    "enum": ["missing_values", "schema_mismatch", "data_anomaly"],
                },
                "sql_query": {
                    "type": "string",
                    "description": "Optional SQL query for lineage extraction (auto-generated if blank)",
                    "default": "",
                },
            },
            "required": ["scenario_name"],
        },
    },
]


# ── Tool executors ────────────────────────────────────────────

def _run_pipeline_tool(scenario_name: str, save_to_db: bool = True) -> dict:
    from workflow.pipeline_graph import run_pipeline

    path_map = {
        "missing_values":  "data/scenario_missing.csv",
        "schema_mismatch": "data/scenario_schema.csv",
        "data_anomaly":    "data/scenario_anomaly.csv",
    }
    path = path_map.get(scenario_name, f"data/{scenario_name}.csv")

    result = run_pipeline(scenario_name, path)
    return {
        "run_id":       result.get("run_id"),
        "status":       result.get("final_status"),
        "issues_found": len(result.get("issues_detected", [])),
        "fixes_applied":len(result.get("fixes_applied", [])),
        "healed_path":  result.get("healed_data_path"),
        "summary":      result.get("detection_summary"),
    }


def _query_database_tool(sql: str, limit: int = 20) -> dict:
    from utils.mock_db import get_db
    db = get_db()
    # Safety: only allow SELECT
    if not sql.strip().upper().startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed"}
    # Inject LIMIT if not present
    if "LIMIT" not in sql.upper():
        sql = sql.rstrip(";") + f" LIMIT {limit}"
    try:
        df = db.execute_sql(sql)
        return {
            "rows":    len(df),
            "columns": list(df.columns),
            "data":    df.head(limit).to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": str(e)}


def _get_run_history_tool(limit: int = 10) -> dict:
    from utils.mock_db import get_db
    db = get_db()
    df = db.get_pipeline_runs(limit=limit)
    if df.empty:
        return {"runs": [], "total": 0}
    return {
        "runs":  df.to_dict(orient="records"),
        "total": len(df),
    }


def _get_issue_stats_tool(run_id: str = None) -> dict:
    from utils.mock_db import get_db
    db = get_db()
    if run_id:
        df = db.get_issues_for_run(run_id)
    else:
        df = db.get_issue_summary()
    if df.empty:
        return {"stats": [], "total": 0}
    return {
        "stats": df.to_dict(orient="records"),
        "total": len(df),
    }


def _validate_data_tool(csv_path: str, suite_type: str = "pre_healing",
                         scenario: str = "default") -> dict:
    from utils.ge_validator import (
        run_pre_healing_suite,
        run_post_healing_suite,
        run_custom_suite,
    )
    if not os.path.exists(csv_path):
        return {"error": f"File not found: {csv_path}"}
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": f"Cannot read CSV: {e}"}

    if suite_type == "pre_healing":
        result = run_pre_healing_suite(df, scenario)
    elif suite_type == "post_healing":
        result = run_post_healing_suite(df, scenario)
    else:
        result = run_custom_suite(df, scenario)

    return result


def _get_quality_scores_tool(scenario_filter: str = None) -> dict:
    from utils.mock_db import get_db
    db = get_db()
    df = db.get_quality_trend()
    if scenario_filter:
        df = df[df["scenario_name"] == scenario_filter]
    if df.empty:
        return {"scores": [], "total": 0}
    return {
        "scores": df.to_dict(orient="records"),
        "average": round(float(df["overall_score"].mean()), 1),
        "total":   len(df),
    }


def _run_b1_agent_tool(scenario_name: str) -> dict:
    from agents.b1_ingestion_quality_agent import run_b1_pipeline
    path_map = {
        "missing_values":  "data/scenario_missing.csv",
        "schema_mismatch": "data/scenario_schema.csv",
        "data_anomaly":    "data/scenario_anomaly.csv",
    }
    path = path_map.get(scenario_name, f"data/{scenario_name}.csv")
    result = run_b1_pipeline(scenario_name, path)
    return {
        "run_id":     result.get("run_id"),
        "status":     result.get("final_status"),
        "rules":      len(result.get("quality_rules", [])),
        "violations": len(result.get("violations", [])),
        "heals":      len(result.get("heals_applied", [])),
        "score":      result.get("validation_score", 0),
        "healed_path":result.get("healed_data_path"),
    }


def _run_b2_agent_tool(scenario_name: str, sql_query: str = "") -> dict:
    from agents.b2_lineage_governance_agent import run_b2_pipeline
    path_map = {
        "missing_values":  "data/scenario_missing.csv",
        "schema_mismatch": "data/scenario_schema.csv",
        "data_anomaly":    "data/scenario_anomaly.csv",
    }
    path = path_map.get(scenario_name, f"data/{scenario_name}.csv")
    result = run_b2_pipeline(scenario_name, path, sql_query=sql_query)
    gr = result.get("governance_report", {})
    return {
        "run_id":       result.get("run_id"),
        "status":       result.get("final_status"),
        "pii_columns":  len(result.get("pii_tags", [])),
        "lineage_nodes":len(result.get("lineage_graph", {}).get("nodes", [])),
        "catalogue":    len(result.get("data_catalogue", [])),
        "gdpr_score":   gr.get("gdpr_compliance", {}).get("score", 0),
        "masked_path":  result.get("masked_data_path"),
    }


# ── MCP Server class ──────────────────────────────────────────

class PipelineMCPServer:
    """
    In-process MCP server.
    Can be used directly from Python or called via the Streamlit UI.
    """

    def __init__(self):
        self.tools = {t["name"]: t for t in TOOLS}
        self.executor_map = {
            "run_pipeline_tool":      _run_pipeline_tool,
            "query_database_tool":    _query_database_tool,
            "get_run_history_tool":   _get_run_history_tool,
            "get_issue_stats_tool":   _get_issue_stats_tool,
            "validate_data_tool":     _validate_data_tool,
            "get_quality_scores_tool":_get_quality_scores_tool,
            "run_b1_agent_tool":      _run_b1_agent_tool,
            "run_b2_agent_tool":      _run_b2_agent_tool,
        }

    def list_tools(self) -> list:
        """Return all available tool definitions."""
        return TOOLS

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Execute a tool by name with given arguments.
        Returns a result dict.
        """
        if tool_name not in self.executor_map:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return self.executor_map[tool_name](**arguments)
        except Exception as e:
            return {
                "error":     str(e),
                "traceback": traceback.format_exc(),
            }

    def handle_request(self, request: dict) -> dict:
        """
        Handle a raw MCP request dict.
        Supports: tools/list, tools/call
        """
        method = request.get("method", "")

        if method == "tools/list":
            return {"tools": self.list_tools()}

        elif method == "tools/call":
            params = request.get("params", {})
            return self.call_tool(
                params.get("name", ""),
                params.get("arguments", {}),
            )
        else:
            return {"error": f"Unknown method: {method}"}


# ── Singleton ─────────────────────────────────────────────────
_server_instance = None

def get_mcp_server() -> PipelineMCPServer:
    global _server_instance
    if _server_instance is None:
        _server_instance = PipelineMCPServer()
    return _server_instance


# ── CLI mode ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Pipeline MCP Server — running in CLI mode")
    print("  Type JSON requests or 'quit' to exit")
    print("=" * 60)
    server = PipelineMCPServer()

    # Demo: list tools
    print("\nAvailable tools:")
    for t in server.list_tools():
        print(f"  • {t['name']}: {t['description'][:60]}...")

    print("\nSend JSON requests like:")
    print('  {"method":"tools/call","params":{"name":"get_run_history_tool","arguments":{}}}')
    print()

    while True:
        try:
            line = input("mcp> ").strip()
            if line.lower() in ("quit", "exit", "q"):
                break
            if not line:
                continue
            req    = json.loads(line)
            result = server.handle_request(req)
            print(json.dumps(result, indent=2, default=str))
        except json.JSONDecodeError:
            print("Invalid JSON")
        except KeyboardInterrupt:
            break