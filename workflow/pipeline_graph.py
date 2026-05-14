"""
workflow/pipeline_graph.py  —  B3 Self-Healing Pipeline (with Memory)
"""

import uuid, datetime, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from utils.helpers import PipelineState, numpy_safe
from agents.detection_agent      import detection_agent
from agents.classification_agent import classification_agent
from agents.decision_agent       import decision_agent
from agents.healing_agent        import healing_agent
from agents.logging_agent        import logging_agent


def route_after_detection(state: PipelineState) -> str:
    return "logging" if not state.get("issues_detected") else "classification"


_b3_memory = MemorySaver()


def b3_memory_recall(state: PipelineState) -> PipelineState:
    """Recall past pipeline runs to inform current healing decisions."""
    from utils.helpers import PipelineLogger
    logger = PipelineLogger("B3-Memory")
    logger.info("[B3] 🧠 Memory Recall: checking for past healing data...")
    past_context = {}

    try:
        pipeline = build_pipeline()
        scenario = state["scenario_name"]
        recall_config = {"configurable": {"thread_id": f"b3_{scenario}"}}
        past_state = pipeline.get_state(recall_config)

        if past_state and past_state.values:
            pv = past_state.values
            past_context = {
                "had_prior_run": True,
                "prior_run_id": pv.get("run_id", ""),
                "prior_issues": len(pv.get("issues_detected", [])),
                "prior_fixes": len(pv.get("fixes_applied", [])),
                "prior_status": pv.get("final_status", ""),
            }
            logger.success(
                f"[B3] 🧠 Recalled prior run {past_context['prior_run_id']} | "
                f"issues={past_context['prior_issues']} | "
                f"fixes={past_context['prior_fixes']} | "
                f"status={past_context['prior_status']}"
            )
        else:
            past_context = {"had_prior_run": False}
            logger.info("[B3] 🧠 No prior run found — first run for this scenario")
    except Exception as e:
        past_context = {"had_prior_run": False}
        logger.warn(f"[B3] 🧠 Memory recall failed (non-fatal): {e}")

    return {
        **state,
        "logs": [f"[B3-Memory] Recall: {past_context}"],
    }


def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)
    graph.add_node("memory_recall",   numpy_safe(b3_memory_recall))
    graph.add_node("detection",       numpy_safe(detection_agent))
    graph.add_node("classification",  numpy_safe(classification_agent))
    graph.add_node("decision",        numpy_safe(decision_agent))
    graph.add_node("healing",         numpy_safe(healing_agent))
    graph.add_node("logging",         numpy_safe(logging_agent))
    graph.set_entry_point("memory_recall")
    graph.add_edge("memory_recall", "detection")
    graph.add_conditional_edges(
        "detection", route_after_detection,
        {"classification": "classification", "logging": "logging"}
    )
    graph.add_edge("classification", "decision")
    graph.add_edge("decision",       "healing")
    graph.add_edge("healing",        "logging")
    graph.add_edge("logging",        END)
    return graph.compile(checkpointer=_b3_memory)


def run_pipeline(scenario_name: str, data_path: str) -> dict:
    run_id = str(uuid.uuid4())[:8].upper()
    thread_id = f"b3_{scenario_name}"  # shared per scenario
    start  = datetime.datetime.now().isoformat()

    print(f"\n{'▓'*70}")
    print(f"  🚀 PIPELINE START | scenario={scenario_name} | run_id={run_id}")
    print(f"  🧠 Memory thread: {thread_id} (shared across runs)")
    print(f"{'▓'*70}\n")

    initial_state: PipelineState = {
        "run_id":             run_id,
        "start_time":         start,
        "scenario_name":      scenario_name,
        "raw_data_path":      data_path,
        "issues_detected":    [],
        "detection_summary":  "",
        "classifications":    [],
        "primary_category":   "",
        "fix_plan":           [],
        "decision_rationale": "",
        "healed_data_path":   "",
        "removed_data_path":  "",
        "quality_report":     {},
        "ge_pre_results":     {},
        "ge_post_results":    {},
        "fixes_applied":      [],
        "logs":               [],
        "final_status":       "RUNNING",
    }

    # ── Register run in MockDB (write → close) ────────────────
    try:
        from utils.mock_db import db_insert_pipeline_run, _ensure_schema_exists
        _ensure_schema_exists()
        db_insert_pipeline_run(run_id, scenario_name)
    except Exception:
        pass

    # ── Run LangGraph pipeline (with memory) ───────────────────
    config = {"configurable": {"thread_id": thread_id}}
    pipeline    = build_pipeline()
    final_state = pipeline.invoke(initial_state, config=config)

    # ── Update MockDB with final stats (write → close) ────────
    try:
        import pandas as pd
        from utils.mock_db import (
            db_update_pipeline_run, db_insert_issues, db_insert_fixes
        )
        issues   = final_state.get("issues_detected", [])
        classifs = final_state.get("classifications",  [])
        fix_plan = final_state.get("fix_plan",         [])
        fixes    = final_state.get("fixes_applied",    [])

        db_insert_issues(run_id, issues, classifs)
        db_insert_fixes(run_id, fixes, fix_plan)

        healed_p = final_state.get("healed_data_path","")
        if os.path.exists(healed_p) and os.path.exists(data_path):
            df_orig   = pd.read_csv(data_path)
            df_healed = pd.read_csv(healed_p)
            null_b    = int(df_orig.isna().sum().sum())
            null_a    = int(df_healed.isna().sum().sum())
            rows_h    = len(df_healed)
            rows_o    = len(df_orig)
        else:
            null_b = null_a = rows_h = rows_o = 0

        db_update_pipeline_run(
            run_id,
            ended_at      = datetime.datetime.now(),
            status        = final_state.get("final_status","UNKNOWN"),
            total_issues  = len(issues),
            total_fixes   = len(fixes),
            rows_original = rows_o,
            rows_healed   = rows_h,
            nulls_before  = null_b,
            nulls_after   = null_a,
        )
    except Exception:
        pass   # non-fatal — pipeline result is still valid

    return final_state