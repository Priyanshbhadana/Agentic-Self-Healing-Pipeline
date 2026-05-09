"""
agents/decision_agent.py
────────────────────────
Decision Agent
  Uses LLM reasoning to decide the BEST fix for each classified issue.
  Outputs a structured fix_plan with action + parameters.

  Possible actions:
    - FILL_MEAN / FILL_MEDIAN / FILL_MODE  (for missing numeric values)
    - FILL_CONSTANT                         (for missing categorical)
    - DROP_ROWS                             (severe nulls > 50%)
    - DROP_COLUMN                           (irrelevant extra columns)
    - RENAME_COLUMN                         (schema mismatch column names)
    - CLIP_OUTLIERS                         (IQR-based capping)
    - RETRY_PIPELINE                        (system failure)
    - SKIP_BATCH                            (unrecoverable data)
    - CAST_DTYPE                            (dtype mismatch)
"""

import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import PipelineLogger, PipelineState

logger = PipelineLogger("DecisionAgent")


def _call_llm(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _build_decision_prompt(issues: list, classifications: list) -> str:
    # Merge issues + classifications for context
    issue_map = {i["issue_id"]: i for i in issues}
    enriched = []
    for c in classifications:
        iid = c["issue_id"]
        enriched.append({**issue_map.get(iid, {}), **c})

    enriched_json = json.dumps(enriched, indent=2)

    return f"""You are an expert Data Engineering Architect designing automated fixes for a broken data pipeline.

For each issue below, decide the best remediation action.

Available actions:
- FILL_MEAN       → fill numeric nulls with column mean
- FILL_MEDIAN     → fill numeric nulls with column median  
- FILL_MODE       → fill categorical nulls with most frequent value
- FILL_CONSTANT   → fill with a specific constant (specify value in params)
- DROP_ROWS       → remove rows where this column is null (use if null_pct > 40)
- DROP_COLUMN     → remove an unexpected/extra column entirely
- RENAME_COLUMN   → rename column to match schema (specify old→new in params)
- CLIP_OUTLIERS   → cap values at IQR bounds
- CAST_DTYPE      → cast column to expected dtype
- RETRY_PIPELINE  → trigger upstream retry (for SYSTEM_FAILURE)
- SKIP_BATCH      → mark batch as skipped (last resort)

For each issue, output a JSON object with:
- issue_id: (same as input)
- action: one of the actions above
- params: dict of any action-specific parameters (e.g. {{"value": 0}} or {{"old": "uid", "new": "user_id"}})
- confidence: float 0.0–1.0 (certainty this is the right fix)
- rationale: one sentence explaining the decision

IMPORTANT RULES:
1. If null_pct > 40 for a numeric column → prefer DROP_ROWS
2. If null_pct <= 40 for numeric → prefer FILL_MEDIAN (robust to skew)
3. Extra schema columns → always DROP_COLUMN
4. Missing schema columns that look like renames → RENAME_COLUMN
5. Outliers → CLIP_OUTLIERS with the bounds provided
6. SYSTEM_FAILURE → RETRY_PIPELINE first

Return ONLY a valid JSON array. No markdown, no preamble.

Issues + Classifications:
{enriched_json}
"""


def decision_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Decision Agent
    Takes classifications → outputs fix_plan with LLM reasoning.
    """
    issues = state.get("issues_detected", [])
    classifications = state.get("classifications", [])

    if not classifications:
        logger.info("No classifications to decide on.")
        return {
            **state,
            "fix_plan": [],
            "decision_rationale": "No issues to fix.",
            "logs": ["[DecisionAgent] No fix plan needed."],
        }

    logger.info(f"Building fix plan for {len(classifications)} classified issue(s)...")

    prompt = _build_decision_prompt(issues, classifications)

    try:
        raw = _call_llm(prompt)
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:-1])
        fix_plan = json.loads(clean)
    except Exception as e:
        logger.error(f"LLM decision failed: {e}. Using fallback rules.")
        fix_plan = _fallback_decide(issues, classifications)

    # Log the plan
    rationale_parts = []
    for fix in fix_plan:
        logger.info(
            f"[{fix['issue_id']}] Action: {fix['action']} | "
            f"Confidence: {int(fix.get('confidence',0)*100)}% | "
            f"Reason: {fix.get('rationale','')}"
        )
        rationale_parts.append(f"{fix['issue_id']}: {fix['action']}")

    rationale = "Fix plan → " + " | ".join(rationale_parts)

    return {
        **state,
        "fix_plan": fix_plan,
        "decision_rationale": rationale,
        "logs": [f"[DecisionAgent] {rationale}"],
    }


def _fallback_decide(issues: list, classifications: list) -> list:
    """Deterministic fallback decision rules."""
    issue_map = {i["issue_id"]: i for i in issues}
    plan = []
    for c in classifications:
        iid = c["issue_id"]
        issue = issue_map.get(iid, {})
        action, params = "SKIP_BATCH", {}

        if c["category"] == "DATA_QUALITY_ISSUE":
            null_pct = issue.get("null_pct", 0)
            col = issue.get("column", "")

            # ✅ NEW: Check the actual column type from the issue detail
            # String/object columns → use FILL_MODE
            # Numeric columns → use FILL_MEDIAN or DROP_ROWS
            is_string_col = col in ["email", "country", "signup_date", "name", "address"]

            if null_pct > 40:
                action = "DROP_ROWS"
            elif is_string_col:
                action = "FILL_MODE"       # ✅ correct for string columns
            else:
                action = "FILL_MEDIAN"     # ✅ correct for numeric columns

        elif c["category"] == "SCHEMA_ISSUE":
            if issue.get("type") == "SCHEMA_EXTRA_COLUMN":
                action = "DROP_COLUMN"
            elif issue.get("type") == "SCHEMA_MISSING_COLUMN":
                action = "RENAME_COLUMN"
                params = {"old": issue.get("column"), "new": issue.get("column")}
            elif issue.get("type") == "DTYPE_MISMATCH":
                action = "CAST_DTYPE"
                params = {"dtype": issue.get("expected_dtype")}

        elif c["category"] == "DATA_ANOMALY":
            action = "CLIP_OUTLIERS"
            bounds = issue.get("bounds", {})
            params = {"lower": bounds.get("lower"), "upper": bounds.get("upper")}

        elif c["category"] == "SYSTEM_FAILURE":
            action = "RETRY_PIPELINE"

        plan.append({
            "issue_id":   iid,
            "action":     action,
            "params":     params,
            "confidence": 0.70,
            "rationale":  f"Fallback rule for {c['category']}",
        })
    return plan