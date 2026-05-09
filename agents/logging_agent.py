"""
agents/logging_agent.py
───────────────────────
Logging & Alert Agent
  Final node in the LangGraph pipeline.
  - Assembles the full run report
  - Prints structured summary
  - Sends mock alerts for HIGH severity issues
  - Saves JSON run report to disk
"""

import json
import datetime
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import PipelineLogger, PipelineState, send_alert

logger = PipelineLogger("LoggingAgent")


def logging_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Logging & Alert Agent
    Produces the final run report and triggers alerts if needed.
    """
    run_id       = state.get("run_id", "unknown")
    scenario     = state.get("scenario_name", "unknown")
    issues       = state.get("issues_detected", [])
    classifs     = state.get("classifications", [])
    fix_plan     = state.get("fix_plan", [])
    fixes_applied= state.get("fixes_applied", [])
    final_status = state.get("final_status", "UNKNOWN")
    start_time   = state.get("start_time", "")
    healed_path  = state.get("healed_data_path", "N/A")
    all_logs     = state.get("logs", [])

    end_time = datetime.datetime.now().isoformat()

    # ── Print structured summary ─────────────────────────────
    border = "═" * 70
    print(f"\n\033[1m{border}\033[0m")
    print(f"\033[1m  SELF-HEALING PIPELINE — FINAL REPORT\033[0m")
    print(f"{border}")
    print(f"  Run ID   : {run_id}")
    print(f"  Scenario : {scenario}")
    print(f"  Started  : {start_time}")
    print(f"  Ended    : {end_time}")
    print(f"  Status   : {'✅ ' if final_status == 'SUCCESS' else '⚠️  '}{final_status}")
    print(f"{border}")

    print(f"\n  {'─'*30} ISSUES DETECTED ({len(issues)}) {'─'*20}")
    for issue in issues:
        sev_icon = "🔴" if issue.get("severity") in ("HIGH","CRITICAL") else "🟡"
        print(f"  {sev_icon} [{issue.get('type')}] {issue.get('detail','')}")

    print(f"\n  {'─'*30} CLASSIFICATIONS ({len(classifs)}) {'─'*18}")
    for c in classifs:
        conf = int(c.get("confidence", 0) * 100)
        print(f"  📋 [{c['issue_id']}] {c.get('category')} / {c.get('subcategory','')} (conf: {conf}%)")

    print(f"\n  {'─'*30} FIX PLAN ({len(fix_plan)}) {'─'*23}")
    for fix in fix_plan:
        print(f"  🔧 [{fix['issue_id']}] Action: {fix.get('action')} → {fix.get('rationale','')}")

    print(f"\n  {'─'*30} FIXES APPLIED ({len(fixes_applied)}) {'─'*20}")
    for fix in fixes_applied:
        icon = "✅" if fix["status"] == "SUCCESS" else "❌"
        print(f"  {icon} [{fix['issue_id']}] {fix['action']} — {fix['result']}")

    print(f"\n  📁 Healed Dataset: {healed_path}")
    print(f"{border}\n")

    # ── Trigger alerts for critical / high severity issues ───
    high_issues = [i for i in issues if i.get("severity") in ("HIGH", "CRITICAL")]
    if high_issues:
        alert_body = (
            f"Run {run_id} — Scenario '{scenario}' encountered {len(high_issues)} "
            f"high-severity issue(s). Final status: {final_status}. "
            f"Healed output: {healed_path}"
        )
        send_alert(
            subject=f"[Self-Healing Pipeline] {final_status} — {scenario}",
            body=alert_body,
            severity="HIGH" if final_status != "SUCCESS" else "INFO",
        )

    # ── Build & save JSON report ─────────────────────────────
    report = {
        "run_id":          run_id,
        "scenario":        scenario,
        "start_time":      start_time,
        "end_time":        end_time,
        "final_status":    final_status,
        "issues_detected": issues,
        "classifications": classifs,
        "fix_plan":        fix_plan,
        "fixes_applied":   fixes_applied,
        "healed_path":     healed_path,
        "agent_logs":      all_logs,
    }

    os.makedirs("logs", exist_ok=True)
    report_path = f"logs/report_{run_id}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.success(f"Run report saved → {report_path}")

    return {
        **state,
        "final_status": final_status,
        "logs": [f"[LoggingAgent] Report saved: {report_path}"],
    }