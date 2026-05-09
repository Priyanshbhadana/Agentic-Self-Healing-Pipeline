"""
agents/b1_ingestion_quality_agent.py
─────────────────────────────────────
B1: Ingestion Quality Agent
Flow: profile → generate_rules → validate → self_heal

Nodes:
  1. profiler_node       — statistical profiling of incoming data
  2. rule_generator_node — LLM generates quality rules from profile
  3. validator_node      — validates data against generated rules
  4. healer_node         — auto-heals violations found
  5. b1_report_node      — final structured report
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import datetime
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import PipelineLogger

logger = PipelineLogger("B1-IngestionQuality")

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END


# ── B1 State ─────────────────────────────────────────────────
class B1State(TypedDict):
    run_id:          str
    scenario_name:   str
    raw_data_path:   str
    start_time:      str

    # Profiler outputs
    profile:         dict      # statistical profile per column
    profile_summary: str       # human-readable summary

    # Rule generator outputs
    quality_rules:   list      # [{rule_id, column, rule_type, params, description}]
    rules_rationale: str       # LLM reasoning

    # Validator outputs
    violations:      list      # [{rule_id, column, violation_type, rows_affected, detail}]
    validation_score: float    # 0–100

    # Healer outputs
    healed_data_path: str
    heals_applied:   list

    # Final
    b1_report:       dict
    final_status:    str
    logs:            Annotated[list, operator.add]


# ──────────────────────────────────────────────────────────────
#  NODE 1: PROFILER
# ──────────────────────────────────────────────────────────────
def profiler_node(state: B1State) -> B1State:
    """Statistical profiling of every column."""
    logger.info(f"[B1] Profiling dataset: {state['raw_data_path']}")
    try:
        df = pd.read_csv(state["raw_data_path"])
    except Exception as e:
        return {**state, "profile": {}, "profile_summary": f"Load error: {e}",
                "logs": [f"[B1-Profiler] Error: {e}"]}

    profile = {}
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        col_profile = {
            "dtype":       str(s.dtype),
            "count":       int(s.count()),
            "null_count":  int(s.isna().sum()),
            "null_pct":    round(s.isna().mean() * 100, 2),
            "unique":      int(s.nunique()),
            "unique_pct":  round(s.nunique() / max(len(df), 1) * 100, 2),
            "sample":      s.dropna().head(3).tolist(),
        }
        if is_num:
            non_null = s.dropna()
            col_profile.update({
                "min":    round(float(non_null.min()), 4) if len(non_null) else None,
                "max":    round(float(non_null.max()), 4) if len(non_null) else None,
                "mean":   round(float(non_null.mean()), 4) if len(non_null) else None,
                "median": round(float(non_null.median()), 4) if len(non_null) else None,
                "std":    round(float(non_null.std()), 4) if len(non_null) else None,
                "q1":     round(float(non_null.quantile(0.25)), 4) if len(non_null) else None,
                "q3":     round(float(non_null.quantile(0.75)), 4) if len(non_null) else None,
                "skewness": round(float(non_null.skew()), 4) if len(non_null) > 2 else None,
                "outlier_pct": _iqr_outlier_pct(s),
            })
        else:
            top_vals = s.value_counts().head(5).to_dict()
            col_profile.update({
                "top_values": {str(k): int(v) for k, v in top_vals.items()},
                "avg_length": round(s.dropna().astype(str).str.len().mean(), 1) if len(s.dropna()) else None,
                "contains_pii": _detect_pii_type(col, s),
            })
        profile[col] = col_profile

    # Table-level stats
    profile["__table__"] = {
        "rows": len(df), "cols": len(df.columns),
        "total_nulls": int(df.isna().sum().sum()),
        "null_pct": round(df.isna().sum().sum() / max(df.size, 1) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
    }

    summary = (
        f"Dataset: {len(df)} rows × {len(df.columns)} cols | "
        f"Nulls: {profile['__table__']['total_nulls']} ({profile['__table__']['null_pct']}%) | "
        f"Duplicates: {profile['__table__']['duplicate_rows']}"
    )
    logger.info(f"[B1] Profile complete: {summary}")

    return {
        **state,
        "profile": profile,
        "profile_summary": summary,
        "logs": [f"[B1-Profiler] {summary}"],
    }


def _iqr_outlier_pct(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 4: return 0.0
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)).sum()
    return round(outliers / len(s) * 100, 2)


def _detect_pii_type(col_name: str, series: pd.Series) -> str:
    """Heuristic PII detection by column name and content."""
    import re
    col_lower = col_name.lower()
    pii_keywords = {
        "email": ["email","e_mail","mail"],
        "phone": ["phone","mobile","cell","tel","contact"],
        "name":  ["name","firstname","lastname","fullname","first_name","last_name"],
        "ssn":   ["ssn","social","tax_id"],
        "dob":   ["dob","birth","birthday","born"],
        "address":["address","street","zip","postal"],
        "ip":    ["ip","ipv4","ipv6"],
        "credit_card":["card","credit","cc_num"],
    }
    for pii_type, keywords in pii_keywords.items():
        if any(kw in col_lower for kw in keywords):
            return pii_type.upper()

    # Content-based detection on sample
    sample = " ".join(str(v) for v in series.dropna().head(20).tolist())
    if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', sample): return "EMAIL"
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', sample): return "SSN"
    if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', sample): return "IP_ADDRESS"
    return "NONE"


# ──────────────────────────────────────────────────────────────
#  NODE 2: RULE GENERATOR (LLM)
# ──────────────────────────────────────────────────────────────
def rule_generator_node(state: B1State) -> B1State:
    """LLM generates data quality rules from the statistical profile."""
    logger.info("[B1] Generating quality rules via LLM...")

    profile = state.get("profile", {})
    table   = profile.get("__table__", {})
    cols    = {k: v for k, v in profile.items() if k != "__table__"}

    prompt = f"""You are a senior Data Quality Engineer. Based on the dataset profile below,
generate a comprehensive set of data quality rules for ingestion validation.

Dataset: {state['scenario_name']}
Rows: {table.get('rows',0)} | Cols: {table.get('cols',0)} | Null%: {table.get('null_pct',0)}

Column profiles (JSON):
{json.dumps(cols, indent=2, default=str)}

For each relevant column, generate rules. Available rule types:
- NOT_NULL           → column must not have nulls (use if null_pct > 0)
- RANGE_CHECK        → numeric value must be within [min, max]
- UNIQUE_CHECK       → column values must be unique
- DTYPE_CHECK        → column must be of expected dtype
- PATTERN_MATCH      → string must match a regex pattern
- OUTLIER_CHECK      → flag statistical outliers (IQR)
- CARDINALITY_CHECK  → number of unique values should be within range
- FRESHNESS_CHECK    → date column values should be recent (if date column)
- PII_DETECTED       → column contains PII and should be masked

Return a JSON array of rules. Each rule must have:
- rule_id: "R001", "R002", etc.
- column: column name (or "__table__" for table-level)
- rule_type: one of the above
- params: dict of rule-specific parameters e.g. {{"min": 0, "max": 120}}
- severity: "HIGH", "MEDIUM", or "LOW"
- description: one sentence explaining the rule

Return ONLY valid JSON array. No markdown, no explanation."""

    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"): raw = "\n".join(raw.split("\n")[1:-1])
        quality_rules = json.loads(raw)
        rationale = f"LLM generated {len(quality_rules)} rules from profile analysis."
    except Exception as e:
        logger.warn(f"[B1] LLM rule generation failed: {e} — using fallback rules")
        quality_rules = _fallback_rules(cols)
        rationale = f"Fallback rule generation ({len(quality_rules)} rules)"

    for rule in quality_rules:
        logger.info(f"[B1] Rule [{rule.get('rule_id')}] {rule.get('rule_type')} on '{rule.get('column')}': {rule.get('description','')[:60]}")

    return {
        **state,
        "quality_rules":   quality_rules,
        "rules_rationale": rationale,
        "logs": [f"[B1-RuleGen] {rationale}"],
    }


def _fallback_rules(cols: dict) -> list:
    rules = []
    rule_id = 1
    for col, profile in cols.items():
        if profile.get("null_pct", 0) > 0:
            rules.append({"rule_id": f"R{rule_id:03d}", "column": col,
                "rule_type": "NOT_NULL", "params": {},
                "severity": "HIGH" if profile["null_pct"] > 20 else "MEDIUM",
                "description": f"Column '{col}' should not contain nulls"})
            rule_id += 1
        if profile.get("dtype") in ("float64","int64"):
            mn, mx = profile.get("min",0), profile.get("max",9999)
            rules.append({"rule_id": f"R{rule_id:03d}", "column": col,
                "rule_type": "RANGE_CHECK", "params": {"min": mn, "max": mx},
                "severity": "MEDIUM",
                "description": f"Column '{col}' values must be between {mn} and {mx}"})
            rule_id += 1
        if profile.get("contains_pii") not in ("NONE", None):
            rules.append({"rule_id": f"R{rule_id:03d}", "column": col,
                "rule_type": "PII_DETECTED", "params": {"pii_type": profile["contains_pii"]},
                "severity": "HIGH",
                "description": f"Column '{col}' contains PII ({profile['contains_pii']}) — must be masked"})
            rule_id += 1
    return rules


# ──────────────────────────────────────────────────────────────
#  NODE 3: VALIDATOR
# ──────────────────────────────────────────────────────────────
def validator_node(state: B1State) -> B1State:
    """Validates data against every generated quality rule."""
    logger.info("[B1] Validating data against quality rules...")
    try:
        df = pd.read_csv(state["raw_data_path"])
    except Exception as e:
        return {**state, "violations": [], "validation_score": 0,
                "logs": [f"[B1-Validator] Load error: {e}"]}

    rules     = state.get("quality_rules", [])
    violations = []

    for rule in rules:
        col       = rule.get("column","")
        rule_type = rule.get("rule_type","")
        params    = rule.get("params", {})
        rule_id   = rule.get("rule_id","")

        try:
            if rule_type == "NOT_NULL":
                if col in df.columns:
                    n = int(df[col].isna().sum())
                    if n > 0:
                        violations.append({
                            "rule_id": rule_id, "column": col,
                            "violation_type": "NULL_VALUES",
                            "rows_affected": n,
                            "detail": f"{n} null values found in '{col}'",
                            "severity": rule.get("severity","MEDIUM"),
                        })

            elif rule_type == "RANGE_CHECK":
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    mn = params.get("min"); mx = params.get("max")
                    if mn is not None and mx is not None:
                        bad = df[col].dropna()
                        bad = bad[(bad < mn) | (bad > mx)]
                        if len(bad) > 0:
                            violations.append({
                                "rule_id": rule_id, "column": col,
                                "violation_type": "OUT_OF_RANGE",
                                "rows_affected": len(bad),
                                "detail": f"{len(bad)} values outside [{mn},{mx}]",
                                "severity": rule.get("severity","MEDIUM"),
                            })

            elif rule_type == "UNIQUE_CHECK":
                if col in df.columns:
                    dupes = int(df[col].duplicated().sum())
                    if dupes > 0:
                        violations.append({
                            "rule_id": rule_id, "column": col,
                            "violation_type": "DUPLICATE_VALUES",
                            "rows_affected": dupes,
                            "detail": f"{dupes} duplicate values in '{col}'",
                            "severity": rule.get("severity","MEDIUM"),
                        })

            elif rule_type == "OUTLIER_CHECK":
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    s = df[col].dropna()
                    if len(s) >= 4:
                        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
                        IQR = Q3 - Q1
                        n_out = int(((s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)).sum())
                        if n_out > 0:
                            violations.append({
                                "rule_id": rule_id, "column": col,
                                "violation_type": "STATISTICAL_OUTLIER",
                                "rows_affected": n_out,
                                "detail": f"{n_out} IQR outliers in '{col}'",
                                "severity": rule.get("severity","LOW"),
                            })

            elif rule_type == "PII_DETECTED":
                if col in df.columns:
                    pii_type = params.get("pii_type","PII")
                    violations.append({
                        "rule_id": rule_id, "column": col,
                        "violation_type": "PII_EXPOSURE",
                        "rows_affected": int(df[col].count()),
                        "detail": f"Column '{col}' contains unmasked {pii_type} data",
                        "severity": "HIGH",
                    })

            elif rule_type == "DTYPE_CHECK":
                expected = params.get("dtype","")
                if col in df.columns and expected and expected not in str(df[col].dtype):
                    violations.append({
                        "rule_id": rule_id, "column": col,
                        "violation_type": "WRONG_DTYPE",
                        "rows_affected": len(df),
                        "detail": f"'{col}' is {df[col].dtype}, expected {expected}",
                        "severity": rule.get("severity","MEDIUM"),
                    })

        except Exception as e:
            logger.warn(f"[B1] Rule {rule_id} validation error: {e}")

    passed = len(rules) - len(violations)
    score  = round(passed / max(len(rules), 1) * 100, 1)

    for v in violations:
        logger.warn(f"[B1] Violation [{v['rule_id']}] {v['violation_type']} in '{v['column']}': {v['detail']}")

    logger.info(f"[B1] Validation: {passed}/{len(rules)} rules passed — score: {score}%")

    return {
        **state,
        "violations":       violations,
        "validation_score": score,
        "logs": [f"[B1-Validator] Score: {score}% | {len(violations)} violations"],
    }


# ──────────────────────────────────────────────────────────────
#  NODE 4: HEALER
# ──────────────────────────────────────────────────────────────
def healer_node(state: B1State) -> B1State:
    """Auto-heals violations found by the validator."""
    logger.info("[B1] Auto-healing violations...")
    try:
        df = pd.read_csv(state["raw_data_path"])
    except Exception as e:
        return {**state, "healed_data_path": "", "heals_applied": [],
                "logs": [f"[B1-Healer] Load error: {e}"]}

    violations  = state.get("violations", [])
    heals       = []

    for v in violations:
        col   = v.get("column","")
        vtype = v.get("violation_type","")

        try:
            if vtype == "NULL_VALUES" and col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    median = df[col].median()
                    n = int(df[col].isna().sum())
                    df[col] = df[col].fillna(median)
                    heals.append({"column": col, "action": "FILL_MEDIAN",
                                  "result": f"Filled {n} nulls with median={round(median,4)}", "status": "SUCCESS"})
                else:
                    mode = df[col].mode()[0] if len(df[col].mode()) > 0 else "UNKNOWN"
                    n = int(df[col].isna().sum())
                    df[col] = df[col].fillna(mode)
                    heals.append({"column": col, "action": "FILL_MODE",
                                  "result": f"Filled {n} nulls with mode='{mode}'", "status": "SUCCESS"})

            elif vtype == "OUT_OF_RANGE" and col in df.columns:
                params = next((r.get("params",{}) for r in state.get("quality_rules",[])
                               if r.get("rule_id") == v.get("rule_id")), {})
                mn, mx = params.get("min"), params.get("max")
                if mn is not None and mx is not None:
                    n = int(((df[col] < mn) | (df[col] > mx)).sum())
                    df[col] = df[col].clip(lower=mn, upper=mx)
                    heals.append({"column": col, "action": "CLIP_RANGE",
                                  "result": f"Clipped {n} values to [{mn},{mx}]", "status": "SUCCESS"})

            elif vtype == "STATISTICAL_OUTLIER" and col in df.columns:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                n = int(((df[col] < lb) | (df[col] > ub)).sum())
                df[col] = df[col].clip(lower=lb, upper=ub)
                heals.append({"column": col, "action": "CLIP_IQR",
                              "result": f"Clipped {n} outliers to IQR bounds", "status": "SUCCESS"})

            elif vtype == "PII_EXPOSURE" and col in df.columns:
                df, mask_result = _mask_pii_column(df, col, v)
                heals.append({"column": col, "action": "MASK_PII",
                              "result": mask_result, "status": "SUCCESS"})

            elif vtype == "DUPLICATE_VALUES":
                before = len(df)
                df = df.drop_duplicates()
                heals.append({"column": col, "action": "REMOVE_DUPLICATES",
                              "result": f"Removed {before - len(df)} duplicate rows", "status": "SUCCESS"})

        except Exception as e:
            heals.append({"column": col, "action": vtype, "result": str(e), "status": "FAILED"})

    os.makedirs("data", exist_ok=True)
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in state["scenario_name"])
    healed_path = f"data/b1_healed_{safe}.csv"
    df.to_csv(healed_path, index=False)

    for h in heals:
        icon = "✓" if h["status"] == "SUCCESS" else "✗"
        logger.success(f"[B1] {icon} [{h['action']}] {h['column']}: {h['result']}")

    logger.success(f"[B1] Healed data saved → {healed_path}")

    return {
        **state,
        "healed_data_path": healed_path,
        "heals_applied":    heals,
        "logs": [f"[B1-Healer] {len(heals)} heals applied → {healed_path}"],
    }


def _mask_pii_column(df: pd.DataFrame, col: str, violation: dict):
    """Mask PII values based on detected PII type. Returns (df, message) tuple."""
    import re, hashlib
    pii_type = violation.get("detail","").split("unmasked ")[-1].split(" data")[0]

    if "EMAIL" in pii_type:
        def mask_email(v):
            if pd.isna(v): return v
            parts = str(v).split("@")
            if len(parts) == 2:
                return f"{str(parts[0])[:2]}***@{parts[1]}"
            return "***@***.***"
        df[col] = df[col].apply(mask_email)
        return df, f"Masked {len(df[col].dropna())} email addresses"

    elif "PHONE" in pii_type:
        df[col] = df[col].apply(lambda v: "***-***-" + str(v)[-4:] if not pd.isna(v) else v)
        return df, f"Masked {len(df[col].dropna())} phone numbers"

    elif "NAME" in pii_type:
        df[col] = df[col].apply(lambda v: str(v)[0] + "***" if not pd.isna(v) and len(str(v)) > 0 else v)
        return df, f"Masked {len(df[col].dropna())} names"

    elif "SSN" in pii_type:
        df[col] = df[col].apply(lambda v: "***-**-" + str(v)[-4:] if not pd.isna(v) else v)
        return df, f"Masked {len(df[col].dropna())} SSNs"

    else:
        # Generic hash-based masking
        df[col] = df[col].apply(lambda v: "MASKED_" + hashlib.md5(str(v).encode()).hexdigest()[:8] if not pd.isna(v) else v)
        return df, f"Hash-masked {len(df[col].dropna())} PII values"


# ──────────────────────────────────────────────────────────────
#  NODE 5: REPORT
# ──────────────────────────────────────────────────────────────
def b1_report_node(state: B1State) -> B1State:
    logger.info("[B1] Generating final ingestion quality report...")
    heals   = state.get("heals_applied", [])
    ok      = sum(1 for h in heals if h["status"] == "SUCCESS")
    status  = "SUCCESS" if ok == len(heals) else "PARTIAL" if ok > 0 else "FAILED"

    report = {
        "run_id":           state.get("run_id"),
        "scenario":         state.get("scenario_name"),
        "generated_at":     datetime.datetime.now().isoformat(),
        "profile_summary":  state.get("profile_summary"),
        "rules_generated":  len(state.get("quality_rules", [])),
        "violations_found": len(state.get("violations", [])),
        "validation_score": state.get("validation_score", 0),
        "heals_applied":    len(heals),
        "heals_ok":         ok,
        "final_status":     status,
        "healed_path":      state.get("healed_data_path",""),
        "quality_rules":    state.get("quality_rules",[]),
        "violations":       state.get("violations",[]),
        "heals":            heals,
    }

    os.makedirs("logs", exist_ok=True)
    rpath = f"logs/b1_report_{state.get('run_id','')}.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.success(f"[B1] Report saved → {rpath} | Status: {status} | Score: {state.get('validation_score',0)}%")

    return {
        **state,
        "b1_report":    report,
        "final_status": status,
        "logs": [f"[B1-Report] Score={state.get('validation_score',0)}% | Status={status}"],
    }


# ──────────────────────────────────────────────────────────────
#  BUILD + RUN
# ──────────────────────────────────────────────────────────────
def build_b1_pipeline():
    graph = StateGraph(B1State)
    graph.add_node("profiler",      profiler_node)
    graph.add_node("rule_generator",rule_generator_node)
    graph.add_node("validator",     validator_node)
    graph.add_node("healer",        healer_node)
    graph.add_node("b1_report",     b1_report_node)
    graph.set_entry_point("profiler")
    graph.add_edge("profiler",      "rule_generator")
    graph.add_edge("rule_generator","validator")
    graph.add_edge("validator",     "healer")
    graph.add_edge("healer",        "b1_report")
    graph.add_edge("b1_report",     END)
    return graph.compile()


def run_b1_pipeline(scenario_name: str, data_path: str) -> dict:
    run_id = str(uuid.uuid4())[:8].upper()
    print(f"\n{'━'*60}")
    print(f"  🔍 B1 INGESTION QUALITY AGENT | {scenario_name} | {run_id}")
    print(f"{'━'*60}\n")

    initial: B1State = {
        "run_id": run_id, "scenario_name": scenario_name,
        "raw_data_path": data_path, "start_time": datetime.datetime.now().isoformat(),
        "profile": {}, "profile_summary": "",
        "quality_rules": [], "rules_rationale": "",
        "violations": [], "validation_score": 0.0,
        "healed_data_path": "", "heals_applied": [],
        "b1_report": {}, "final_status": "RUNNING", "logs": [],
    }
    pipeline = build_b1_pipeline()
    return pipeline.invoke(initial)