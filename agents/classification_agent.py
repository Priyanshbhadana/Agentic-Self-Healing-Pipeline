"""
agents/classification_agent.py
───────────────────────────────
Classification Agent
  Uses an LLM (claude-sonnet-4) to classify each detected issue
  into a structured category with a confidence score.

  Categories:
    - DATA_QUALITY_ISSUE   (nulls, outliers, bad values)
    - SCHEMA_ISSUE         (missing/extra columns, dtype mismatch)
    - SYSTEM_FAILURE       (empty dataset, load error, truncation)
    - DATA_ANOMALY         (statistical outliers)
"""

import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import PipelineLogger, PipelineState

logger = PipelineLogger("ClassificationAgent")

# ── LLM caller ───────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """
    Calls the Anthropic API directly (no LangChain wrapper needed here).
    Returns the raw text response.
    """
    from google import genai
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


# ── Classification prompt ─────────────────────────────────────

def _build_classification_prompt(issues: list) -> str:
    issues_json = json.dumps(issues, indent=2)
    return f"""You are a senior Data Engineering expert. Classify each data pipeline issue below.

For each issue, output a JSON object with:
- issue_id: (same as input)
- category: one of [DATA_QUALITY_ISSUE, SCHEMA_ISSUE, SYSTEM_FAILURE, DATA_ANOMALY]
- subcategory: a more specific label (e.g. "Null Values", "Missing Column", "Statistical Outlier")
- confidence: float 0.0–1.0 (your certainty about this classification)
- reasoning: one concise sentence explaining why

Return ONLY a valid JSON array. No markdown, no explanation outside the array.

Issues to classify:
{issues_json}
"""


# ── Main agent node ──────────────────────────────────────────

def classification_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Classification Agent
    Takes issues_detected → produces classifications with confidence scores.
    """
    issues = state.get("issues_detected", [])

    if not issues:
        logger.info("No issues to classify.")
        return {
            **state,
            "classifications": [],
            "primary_category": "NONE",
            "logs": ["[ClassificationAgent] No issues to classify."],
        }

    logger.info(f"Classifying {len(issues)} detected issue(s) via LLM...")

    prompt = _build_classification_prompt(issues)

    try:
        raw_response = _call_llm(prompt)
        # Strip markdown fences if present
        clean = raw_response.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:-1])
        classifications = json.loads(clean)
    except Exception as e:
        logger.error(f"LLM classification failed: {e}. Using fallback rule-based classification.")
        # Fallback: rule-based classification
        classifications = _fallback_classify(issues)

    # Log each classification
    for c in classifications:
        conf_pct = int(c.get("confidence", 0) * 100)
        logger.info(
            f"[{c['issue_id']}] → {c['category']} / {c.get('subcategory','')} "
            f"| Confidence: {conf_pct}%"
        )

    # Determine the dominant category (most frequent)
    categories = [c["category"] for c in classifications]
    primary = max(set(categories), key=categories.count) if categories else "NONE"
    logger.info(f"Primary failure category: {primary}")

    return {
        **state,
        "classifications": classifications,
        "primary_category": primary,
        "logs": [f"[ClassificationAgent] Classified {len(classifications)} issues. Primary: {primary}"],
    }


def _fallback_classify(issues: list) -> list:
    """Rule-based fallback if LLM is unavailable."""
    CATEGORY_MAP = {
        "MISSING_VALUES":       ("DATA_QUALITY_ISSUE", "Null Values",       0.90),
        "SCHEMA_MISSING_COLUMN":("SCHEMA_ISSUE",       "Missing Column",    0.95),
        "SCHEMA_EXTRA_COLUMN":  ("SCHEMA_ISSUE",       "Extra Column",      0.90),
        "DTYPE_MISMATCH":       ("SCHEMA_ISSUE",       "Type Mismatch",     0.85),
        "DATA_ANOMALY":         ("DATA_ANOMALY",        "Statistical Outlier",0.80),
        "EMPTY_DATASET":        ("SYSTEM_FAILURE",     "Empty Dataset",     0.99),
        "LOW_ROW_COUNT":        ("SYSTEM_FAILURE",     "Row Truncation",    0.75),
    }
    results = []
    for issue in issues:
        cat, sub, conf = CATEGORY_MAP.get(issue["type"], ("DATA_QUALITY_ISSUE", "Unknown", 0.5))
        results.append({
            "issue_id":    issue["issue_id"],
            "category":    cat,
            "subcategory": sub,
            "confidence":  conf,
            "reasoning":   f"Rule-based fallback for issue type {issue['type']}",
        })
    return results