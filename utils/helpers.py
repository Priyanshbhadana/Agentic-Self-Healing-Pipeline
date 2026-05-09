"""
utils/helpers.py
────────────────
Shared utilities used across all agents:
  - Structured logger
  - Alert simulator
  - LangGraph state TypedDict
  - Confidence scorer helper
"""

import json
import datetime
from typing import TypedDict, Optional, Any
from typing_extensions import Annotated
import operator


# ──────────────────────────────────────────────────────────────
#  Structured Logger
# ──────────────────────────────────────────────────────────────
class PipelineLogger:
    """
    Emits timestamped, leveled log lines.
    Writes to console AND appends to a JSON log file.
    """
    LOG_FILE = "logs/pipeline_run.jsonl"

    def __init__(self, component: str):
        self.component = component
        import os; os.makedirs("logs", exist_ok=True)

    def _emit(self, level: str, message: str, data: dict = None):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = {
            "INFO":    "\033[94m",   # blue
            "WARN":    "\033[93m",   # yellow
            "SUCCESS": "\033[92m",   # green
            "ERROR":   "\033[91m",   # red
            "ALERT":   "\033[95m",   # magenta
        }.get(level, "\033[0m")
        reset = "\033[0m"

        log_line = f"[{ts}] [{level}] [{self.component}] {message}"
        print(f"{color}{log_line}{reset}")

        # Persist to JSONL
        record = {
            "timestamp": ts, "level": level,
            "component": self.component, "message": message,
        }
        if data:
            record["data"] = data
        with open(self.LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    def info(self, msg, data=None):    self._emit("INFO", msg, data)
    def warn(self, msg, data=None):    self._emit("WARN", msg, data)
    def success(self, msg, data=None): self._emit("SUCCESS", msg, data)
    def error(self, msg, data=None):   self._emit("ERROR", msg, data)
    def alert(self, msg, data=None):   self._emit("ALERT", msg, data)


# ──────────────────────────────────────────────────────────────
#  Alert Simulator  (mocks email / Slack / PagerDuty)
# ──────────────────────────────────────────────────────────────
def send_alert(subject: str, body: str, severity: str = "HIGH"):
    """Simulates sending an alert to an on-call channel."""
    logger = PipelineLogger("AlertSystem")
    border = "═" * 60
    logger.alert(f"\n{border}")
    logger.alert(f"  🚨 ALERT [{severity}]: {subject}")
    logger.alert(f"  📧 Mock Email → oncall@company.com")
    logger.alert(f"  💬 Mock Slack → #data-alerts")
    logger.alert(f"  📄 Body: {body}")
    logger.alert(f"{border}\n")


# ──────────────────────────────────────────────────────────────
#  LangGraph State Schema
# ──────────────────────────────────────────────────────────────
class PipelineState(TypedDict):
    """
    Shared state passed between every node in the LangGraph.
    Each agent reads from and writes to this dict.
    """
    # Input
    scenario_name: str           # e.g. "missing_values"
    raw_data_path: str           # path to the CSV

    # Detection outputs
    issues_detected: list        # list of issue dicts
    detection_summary: str       # human-readable summary

    # Classification outputs
    classifications: list        # [{issue, category, confidence}]
    primary_category: str        # dominant category for routing

    # Decision outputs
    fix_plan: list               # [{issue, action, params}]
    decision_rationale: str      # LLM reasoning text

    # Healing outputs
    healed_data_path: str        # path to fixed CSV
    removed_data_path: str       # path to removed rows CSV
    quality_report: dict         # data quality summary dict
    ge_pre_results: dict         # GE pre-healing validation
    ge_post_results: dict        # GE post-healing validation
    fixes_applied: list          # what was actually done

    # Logging / meta
    run_id: str
    start_time: str
    logs: Annotated[list, operator.add]   # accumulated across agents
    final_status: str            # SUCCESS / PARTIAL / FAILED