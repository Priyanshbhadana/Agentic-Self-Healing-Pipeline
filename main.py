"""
main.py
───────
Entry point for the Self-Healing Data Pipeline Agent.
Runs all 3 demo scenarios sequentially and prints results.
"""

import os
import sys
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY from .env
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Generate mock datasets first
from data.generate_data import save_all
save_all()

# Import the pipeline runner
from workflow.pipeline_graph import run_pipeline

# ──────────────────────────────────────────────
#  Define Scenarios to Run
# ──────────────────────────────────────────────
SCENARIOS = [
    {
        "name": "missing_values",
        "path": "data/scenario_missing.csv",
        "description": "Dataset with missing values in age, salary, email, score columns"
    },
    {
        "name": "schema_mismatch",
        "path": "data/scenario_schema.csv",
        "description": "Dataset with renamed columns + unexpected extra columns"
    },
    {
        "name": "data_anomaly",
        "path": "data/scenario_anomaly.csv",
        "description": "Dataset with statistical outliers in age, salary, score"
    },
]

# ──────────────────────────────────────────────
#  Run All Scenarios
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("   SELF-HEALING DATA PIPELINE AGENT — STARTING")
    print("   Powered by LangGraph + Claude (Anthropic)")
    print("█" * 70)

    results = []

    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n{'='*70}")
        print(f"  SCENARIO {i}/3 : {scenario['name'].upper()}")
        print(f"  Info        : {scenario['description']}")
        print(f"{'='*70}")

        try:
            final_state = run_pipeline(
                scenario_name=scenario["name"],
                data_path=scenario["path"]
            )
            results.append({
                "scenario": scenario["name"],
                "status":   final_state.get("final_status", "UNKNOWN"),
                "issues":   len(final_state.get("issues_detected", [])),
                "fixes":    len(final_state.get("fixes_applied", [])),
                "output":   final_state.get("healed_data_path", "N/A"),
            })
        except Exception as e:
            print(f"\n❌ Scenario '{scenario['name']}' crashed: {e}")
            import traceback; traceback.print_exc()
            results.append({
                "scenario": scenario["name"],
                "status":   "CRASHED",
                "issues":   0,
                "fixes":    0,
                "output":   "N/A",
            })

    # ── Final Summary Table ──────────────────────────────────
    print("\n" + "█" * 70)
    print("   ALL SCENARIOS COMPLETE — SUMMARY")
    print("█" * 70)
    print(f"  {'Scenario':<20} {'Status':<12} {'Issues':<10} {'Fixes':<10} {'Output'}")
    print(f"  {'─'*20} {'─'*12} {'─'*10} {'─'*10} {'─'*30}")
    for r in results:
        icon = "✅" if r["status"] == "SUCCESS" else "⚠️ "
        print(f"  {icon} {r['scenario']:<18} {r['status']:<12} {r['issues']:<10} {r['fixes']:<10} {r['output']}")

    print("\n  📁 Check the following folders for outputs:")
    print("     • data/     → healed CSV files")
    print("     • logs/     → JSON run reports + pipeline_run.jsonl")
    print("█" * 70 + "\n")