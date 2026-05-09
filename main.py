"""
main.py
───────
Entry point for the Self-Healing Data Pipeline — Full Agentic DE Automation.
Runs all 3 agents sequentially:
  B3: Self-Healing Pipeline (Detection → Classification → Decision → Healing → Logging)
  B1: Ingestion Quality    (Profile → Rules → Validate → Heal → Report)
  B2: Lineage & Governance (SQL → Lineage → PII → Catalogue → GDPR Report)
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY from .env
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Generate mock datasets first
from data.generate_data import save_all
save_all()

# Import all pipeline runners
from workflow.pipeline_graph import run_pipeline
from agents.b1_ingestion_quality_agent import run_b1_pipeline
from agents.b2_lineage_governance_agent import run_b2_pipeline

# ──────────────────────────────────────────────
#  Define Scenarios
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
#  Run All Agents
# ──────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()
    print("\n" + "█" * 70)
    print("   AGENTIC DE AUTOMATION — FULL PIPELINE")
    print("   B3: Self-Healing | B1: Ingestion Quality | B2: Lineage & Governance")
    print("   Powered by LangGraph + Claude (Anthropic)")
    print("█" * 70)

    b3_results = []
    b1_results = []
    b2_results = []

    # ── PHASE 1: B3 Self-Healing Pipeline ─────────────────────
    print(f"\n{'━'*70}")
    print(f"  PHASE 1/3: B3 SELF-HEALING PIPELINE (3 scenarios)")
    print(f"{'━'*70}")

    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n{'='*70}")
        print(f"  B3 SCENARIO {i}/3 : {scenario['name'].upper()}")
        print(f"  Info        : {scenario['description']}")
        print(f"{'='*70}")

        try:
            final_state = run_pipeline(
                scenario_name=scenario["name"],
                data_path=scenario["path"]
            )
            b3_results.append({
                "agent": "B3",
                "scenario": scenario["name"],
                "status":   final_state.get("final_status", "UNKNOWN"),
                "issues":   len(final_state.get("issues_detected", [])),
                "fixes":    len(final_state.get("fixes_applied", [])),
                "output":   final_state.get("healed_data_path", "N/A"),
            })
        except Exception as e:
            print(f"\n❌ B3 '{scenario['name']}' crashed: {e}")
            import traceback; traceback.print_exc()
            b3_results.append({
                "agent": "B3", "scenario": scenario["name"],
                "status": "CRASHED", "issues": 0, "fixes": 0, "output": "N/A",
            })

    # ── PHASE 2: B1 Ingestion Quality ─────────────────────────
    print(f"\n{'━'*70}")
    print(f"  PHASE 2/3: B1 INGESTION QUALITY AGENT")
    print(f"{'━'*70}")

    b1_scenario = SCENARIOS[0]  # Run B1 on missing_values
    try:
        b1_state = run_b1_pipeline(b1_scenario["name"], b1_scenario["path"])
        b1_results.append({
            "agent": "B1",
            "scenario": b1_scenario["name"],
            "status":   b1_state.get("final_status", "UNKNOWN"),
            "rules":    len(b1_state.get("quality_rules", [])),
            "violations": len(b1_state.get("violations", [])),
            "heals":    len(b1_state.get("heals_applied", [])),
            "score":    b1_state.get("validation_score", 0),
            "output":   b1_state.get("healed_data_path", "N/A"),
        })
    except Exception as e:
        print(f"\n❌ B1 crashed: {e}")
        import traceback; traceback.print_exc()
        b1_results.append({
            "agent": "B1", "scenario": b1_scenario["name"],
            "status": "CRASHED", "rules": 0, "violations": 0,
            "heals": 0, "score": 0, "output": "N/A",
        })

    # ── PHASE 3: B2 Lineage & Governance ──────────────────────
    print(f"\n{'━'*70}")
    print(f"  PHASE 3/3: B2 LINEAGE & GOVERNANCE AGENT")
    print(f"{'━'*70}")

    b2_scenario = SCENARIOS[0]  # Run B2 on missing_values
    try:
        b2_state = run_b2_pipeline(b2_scenario["name"], b2_scenario["path"])
        gr = b2_state.get("governance_report", {})
        b2_results.append({
            "agent": "B2",
            "scenario": b2_scenario["name"],
            "status":   b2_state.get("final_status", "UNKNOWN"),
            "pii_cols": len(b2_state.get("pii_tags", [])),
            "lineage_nodes": len(b2_state.get("lineage_graph", {}).get("nodes", [])),
            "catalogue": len(b2_state.get("data_catalogue", [])),
            "gdpr_score": gr.get("gdpr_compliance", {}).get("score", 0),
            "output":   b2_state.get("masked_data_path", "N/A"),
        })
    except Exception as e:
        print(f"\n❌ B2 crashed: {e}")
        import traceback; traceback.print_exc()
        b2_results.append({
            "agent": "B2", "scenario": b2_scenario["name"],
            "status": "CRASHED", "pii_cols": 0, "lineage_nodes": 0,
            "catalogue": 0, "gdpr_score": 0, "output": "N/A",
        })

    # ── Final Summary ─────────────────────────────────────────
    elapsed = round(time.time() - t_start, 1)
    print("\n" + "█" * 70)
    print(f"   ALL 3 AGENTS COMPLETE — SUMMARY (total: {elapsed}s)")
    print("█" * 70)

    print(f"\n  ─── B3: Self-Healing Pipeline ───")
    print(f"  {'Scenario':<20} {'Status':<12} {'Issues':<10} {'Fixes':<10} {'Output'}")
    for r in b3_results:
        icon = "✅" if r["status"] == "SUCCESS" else "⚠️ "
        print(f"  {icon} {r['scenario']:<18} {r['status']:<12} {r['issues']:<10} {r['fixes']:<10} {r['output']}")

    print(f"\n  ─── B1: Ingestion Quality ───")
    for r in b1_results:
        icon = "✅" if r["status"] == "SUCCESS" else "⚠️ "
        print(f"  {icon} {r['scenario']:<18} Rules={r['rules']} Violations={r['violations']} Heals={r['heals']} Score={r['score']}%")

    print(f"\n  ─── B2: Lineage & Governance ───")
    for r in b2_results:
        icon = "✅" if r["status"] == "SUCCESS" else "⚠️ "
        print(f"  {icon} {r['scenario']:<18} PII={r['pii_cols']} Lineage={r['lineage_nodes']} Catalogue={r['catalogue']} GDPR={r['gdpr_score']}%")

    print(f"\n  📁 Output directories:")
    print("     • data/     → healed CSVs, masked CSVs, lineage JSON")
    print("     • logs/     → B1/B2/B3 reports + pipeline_run.jsonl")
    print("█" * 70 + "\n")