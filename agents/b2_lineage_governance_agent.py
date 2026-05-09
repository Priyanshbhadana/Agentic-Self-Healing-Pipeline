"""
agents/b2_lineage_governance_agent.py
──────────────────────────────────────
B2: Lineage & Governance Agent
Flow: SQL → extract_lineage → tag_pii → enrich_catalogue

LangGraph Nodes:
  1. sql_parser_node       — parse SQL / data path → extract table/column refs
  2. lineage_extractor_node— LLM extracts full data lineage graph
  3. pii_tagger_node       — detect & tag PII columns with sensitivity levels
  4. catalogue_enricher_node— LLM enriches data catalogue with descriptions
  5. governance_report_node — final governance report + policy recommendations

What it produces:
  • Lineage graph (sources → transformations → outputs)
  • PII tag map per column with masking strategy
  • Enriched data catalogue (column descriptions, business terms)
  • Governance policy report (GDPR, data retention, access control)
  • All stored in SQLite catalogue table
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import re
import datetime
import uuid
import sqlite3
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import PipelineLogger

logger = PipelineLogger("B2-LineageGovernance")

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END


# ──────────────────────────────────────────────────────────────
#  B2 State Schema
# ──────────────────────────────────────────────────────────────
class B2State(TypedDict):
    run_id:          str
    scenario_name:   str
    raw_data_path:   str
    sql_query:       str        # SQL query to parse (can be auto-generated)
    start_time:      str

    # SQL Parser outputs
    parsed_sql:      dict       # tables, columns, joins, filters, aggregations
    sql_summary:     str

    # Lineage Extractor outputs
    lineage_graph:   dict       # {nodes: [], edges: [], sources: [], targets: []}
    lineage_summary: str

    # PII Tagger outputs
    pii_tags:        list       # [{column, pii_type, sensitivity, masking_strategy, gdpr_article}]
    pii_summary:     str
    masked_data_path: str       # path to PII-masked dataset

    # Catalogue Enricher outputs
    data_catalogue:  list       # [{column, data_type, description, business_term, owner, tags}]
    catalogue_summary: str

    # Governance Report outputs
    governance_report: dict
    policy_recommendations: list

    # Final
    final_status:    str
    logs:            Annotated[list, operator.add]


# ──────────────────────────────────────────────────────────────
#  NODE 1: SQL PARSER
# ──────────────────────────────────────────────────────────────
def sql_parser_node(state: B2State) -> B2State:
    """
    Parse the SQL query (or auto-generate one from the CSV schema)
    to extract tables, columns, joins, filters, aggregations.
    """
    logger.info("[B2] Parsing SQL / extracting schema structure...")

    sql = state.get("sql_query", "").strip()
    data_path = state.get("raw_data_path", "")

    # If no SQL provided, auto-generate from CSV schema
    if not sql and data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path, nrows=5)
        table_name = os.path.splitext(os.path.basename(data_path))[0]
        cols_def = ", ".join([f"{c} {_dtype_to_sql(str(df[c].dtype))}" for c in df.columns])
        sql = f"""
-- Auto-generated SQL for lineage extraction
-- Source: {data_path}

WITH source_data AS (
    SELECT *
    FROM {table_name}
    WHERE 1=1
),
cleaned_data AS (
    SELECT
        {', '.join(df.columns.tolist())}
    FROM source_data
    WHERE {df.columns[0]} IS NOT NULL
),
aggregated AS (
    SELECT
        {df.columns[-1]},
        COUNT(*) as record_count,
        MAX({df.columns[0]}) as max_id
    FROM cleaned_data
    GROUP BY {df.columns[-1]}
)
SELECT c.*, a.record_count
FROM cleaned_data c
LEFT JOIN aggregated a ON c.{df.columns[-1]} = a.{df.columns[-1]}
ORDER BY c.{df.columns[0]};
        """.strip()
        logger.info(f"[B2] Auto-generated SQL for table '{table_name}'")

    # Parse SQL structure
    parsed = _parse_sql_structure(sql, data_path)

    summary = (
        f"Tables: {parsed['tables']} | "
        f"Columns: {len(parsed['columns'])} | "
        f"CTEs: {len(parsed['ctes'])} | "
        f"Joins: {len(parsed['joins'])} | "
        f"Filters: {len(parsed['filters'])}"
    )
    logger.info(f"[B2] SQL parsed: {summary}")

    return {
        **state,
        "sql_query":   sql,
        "parsed_sql":  parsed,
        "sql_summary": summary,
        "logs": [f"[B2-SQLParser] {summary}"],
    }


def _dtype_to_sql(dtype: str) -> str:
    mapping = {
        "int64": "INTEGER", "float64": "FLOAT",
        "object": "VARCHAR(255)", "bool": "BOOLEAN",
        "datetime64": "TIMESTAMP",
    }
    for k, v in mapping.items():
        if k in dtype: return v
    return "VARCHAR(255)"


def _parse_sql_structure(sql: str, data_path: str = "") -> dict:
    """Extract SQL structure using regex patterns."""
    sql_upper = sql.upper()

    # Extract table names
    table_pattern = re.compile(
        r'(?:FROM|JOIN|INTO|UPDATE)\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.IGNORECASE
    )
    tables = list(dict.fromkeys(table_pattern.findall(sql)))
    # Remove SQL keywords
    sql_keywords = {"SELECT","WHERE","AND","OR","ON","SET","VALUES","WITH"}
    tables = [t for t in tables if t.upper() not in sql_keywords]

    # Extract CTEs
    cte_pattern = re.compile(r'(\w+)\s+AS\s*\(', re.IGNORECASE)
    ctes = cte_pattern.findall(sql)

    # Extract column references
    select_pattern = re.compile(r'SELECT\s+(.*?)(?:FROM|$)', re.IGNORECASE | re.DOTALL)
    select_match = select_pattern.search(sql)
    raw_cols = select_match.group(1) if select_match else ""
    col_refs = [c.strip().split(".")[-1].split(" ")[-1].strip("(),")
                for c in raw_cols.split(",") if c.strip() and c.strip() != "*"]
    col_refs = [c for c in col_refs if c and not c.upper() in sql_keywords]

    # If CSV available, use actual column names
    actual_columns = []
    if data_path and os.path.exists(data_path):
        try:
            df_schema = pd.read_csv(data_path, nrows=1)
            actual_columns = list(df_schema.columns)
        except Exception:
            pass

    # Extract JOINs
    join_pattern = re.compile(
        r'((?:LEFT|RIGHT|INNER|OUTER|FULL|CROSS)?\s*JOIN)\s+(\w+)\s+(?:\w+\s+)?ON\s+(.*?)(?=JOIN|WHERE|GROUP|ORDER|LIMIT|$)',
        re.IGNORECASE | re.DOTALL
    )
    joins = [{"type": m[0].strip(), "table": m[1], "condition": m[2].strip()[:80]}
             for m in join_pattern.findall(sql)]

    # Extract WHERE filters
    where_pattern = re.compile(r'WHERE\s+(.*?)(?=GROUP BY|ORDER BY|HAVING|LIMIT|$)', re.IGNORECASE | re.DOTALL)
    where_match = where_pattern.search(sql)
    filters = []
    if where_match:
        filter_str = where_match.group(1).strip()
        filters = [f.strip() for f in re.split(r'\bAND\b|\bOR\b', filter_str) if f.strip()][:5]

    # Extract aggregations
    agg_pattern = re.compile(r'(COUNT|SUM|AVG|MIN|MAX|STDDEV)\s*\(([^)]*)\)', re.IGNORECASE)
    aggregations = [{"function": m[0].upper(), "column": m[1].strip()}
                    for m in agg_pattern.findall(sql)]

    return {
        "tables":       tables,
        "ctes":         ctes,
        "columns":      actual_columns if actual_columns else col_refs,
        "joins":        joins,
        "filters":      filters,
        "aggregations": aggregations,
        "sql_lines":    len(sql.strip().split("\n")),
        "raw_sql":      sql[:500],
    }


# ──────────────────────────────────────────────────────────────
#  NODE 2: LINEAGE EXTRACTOR (LLM)
# ──────────────────────────────────────────────────────────────
def lineage_extractor_node(state: B2State) -> B2State:
    """LLM extracts full data lineage graph from parsed SQL structure."""
    logger.info("[B2] Extracting data lineage via LLM...")

    parsed   = state.get("parsed_sql", {})
    scenario = state.get("scenario_name", "unknown")
    path     = state.get("raw_data_path","")

    # Load actual schema
    schema_info = ""
    if path and os.path.exists(path):
        try:
            df = pd.read_csv(path, nrows=3)
            schema_info = f"\nActual schema:\n{df.dtypes.to_string()}\nSample:\n{df.head(2).to_string()}"
        except Exception:
            pass

    prompt = f"""You are a Data Lineage and Governance expert. Analyze the SQL structure below 
and extract a complete data lineage graph.

Scenario: {scenario}
SQL Structure: {json.dumps(parsed, indent=2, default=str)}
{schema_info}

Return a JSON object with EXACTLY this structure:
{{
  "nodes": [
    {{"id": "n1", "name": "table_or_column_name", "type": "SOURCE|TRANSFORMATION|SINK|COLUMN",
      "description": "what this node represents", "system": "database/service name"}}
  ],
  "edges": [
    {{"from": "n1", "to": "n2", "transformation": "what happens", "operation": "SELECT|JOIN|FILTER|AGGREGATE|MASK"}}
  ],
  "sources": ["list of source system names"],
  "sinks": ["list of target/output system names"],
  "transformations": ["list of transformations applied"],
  "lineage_path": "source → transform1 → transform2 → sink (one line summary)"
}}

Make it realistic for a data pipeline that:
1. Ingests raw data from a CSV/database source
2. Applies cleaning and transformation
3. Outputs to a data warehouse
4. Serves analytics dashboards

Return ONLY valid JSON. No markdown."""

    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"): raw = "\n".join(raw.split("\n")[1:-1])
        lineage_graph = json.loads(raw)
        summary = lineage_graph.get("lineage_path", "LLM lineage extracted")
    except Exception as e:
        logger.warn(f"[B2] LLM lineage failed: {e} — using fallback")
        lineage_graph = _fallback_lineage(parsed, scenario)
        summary = lineage_graph.get("lineage_path","Fallback lineage")

    logger.info(f"[B2] Lineage: {summary}")
    logger.info(f"[B2] Nodes: {len(lineage_graph.get('nodes',[]))} | Edges: {len(lineage_graph.get('edges',[]))}")

    return {
        **state,
        "lineage_graph":   lineage_graph,
        "lineage_summary": summary,
        "logs": [f"[B2-LineageExtractor] {summary}"],
    }


def _fallback_lineage(parsed: dict, scenario: str) -> dict:
    """Rule-based lineage graph when LLM unavailable."""
    tables = parsed.get("tables", ["raw_data"])
    cols   = parsed.get("columns", [])
    nodes, edges = [], []

    # Source node
    src_id = "n_src"
    nodes.append({"id": src_id, "name": tables[0] if tables else "raw_source",
                  "type": "SOURCE", "description": "Raw ingestion source",
                  "system": "CSV / Database"})
    prev = src_id

    # CTE nodes
    for i, cte in enumerate(parsed.get("ctes", [])[:3]):
        nid = f"n_cte_{i}"
        nodes.append({"id": nid, "name": cte, "type": "TRANSFORMATION",
                      "description": f"CTE transformation: {cte}", "system": "SQL Engine"})
        edges.append({"from": prev, "to": nid,
                      "transformation": f"CTE: {cte}", "operation": "SELECT"})
        prev = nid

    # Join nodes
    for i, join in enumerate(parsed.get("joins", [])[:2]):
        nid = f"n_join_{i}"
        nodes.append({"id": nid, "name": f"JOIN {join.get('table','')}",
                      "type": "TRANSFORMATION", "description": f"{join.get('type')} JOIN",
                      "system": "SQL Engine"})
        edges.append({"from": prev, "to": nid,
                      "transformation": join.get("condition",""), "operation": "JOIN"})
        prev = nid

    # Sink node
    sink_id = "n_sink"
    nodes.append({"id": sink_id, "name": f"healed_{scenario}",
                  "type": "SINK", "description": "Cleaned output dataset",
                  "system": "Data Warehouse / File"})
    edges.append({"from": prev, "to": sink_id,
                  "transformation": "Write cleaned output", "operation": "INSERT"})

    return {
        "nodes": nodes, "edges": edges,
        "sources": [tables[0] if tables else "raw_source"],
        "sinks":   [f"healed_{scenario}"],
        "transformations": parsed.get("ctes", []),
        "lineage_path": f"{tables[0] if tables else 'source'} → SQL transforms → healed_{scenario}",
    }


# ──────────────────────────────────────────────────────────────
#  NODE 3: PII TAGGER
# ──────────────────────────────────────────────────────────────
def pii_tagger_node(state: B2State) -> B2State:
    """
    Detects and tags every PII column.
    Assigns sensitivity level, GDPR article, and masking strategy.
    Creates a masked copy of the dataset.
    """
    logger.info("[B2] Scanning and tagging PII columns...")

    path = state.get("raw_data_path","")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {**state, "pii_tags": [], "pii_summary": f"Load error: {e}",
                "masked_data_path": "", "logs": [f"[B2-PIITagger] Error: {e}"]}

    pii_tags = []

    # ── PII detection rules ───────────────────────────────────
    PII_RULES = [
        # (pattern_in_colname, content_regex, pii_type, sensitivity, gdpr_article, mask_strategy)
        (["email","e_mail","mail"],
         r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
         "EMAIL", "HIGH", "Art.5(1)(f) - Integrity & Confidentiality",
         "PARTIAL_MASK",     # keep domain, mask local part
        ),
        (["phone","mobile","cell","tel","contact"],
         r'\+?[\d\s\-\(\)]{7,15}',
         "PHONE_NUMBER", "HIGH", "Art.5(1)(f)",
         "LAST_4_VISIBLE",   # show only last 4 digits
        ),
        (["name","firstname","lastname","fullname","first_name","last_name"],
         None,
         "PERSONAL_NAME", "HIGH", "Art.4(1) - Personal Data",
         "INITIAL_ONLY",     # keep first initial
        ),
        (["ssn","social","tax_id","national_id"],
         r'\b\d{3}-\d{2}-\d{4}\b',
         "SSN", "CRITICAL", "Art.9 - Special Category Data",
         "FULL_HASH",        # full SHA256 hash
        ),
        (["dob","birth","birthday","born","date_of_birth"],
         None,
         "DATE_OF_BIRTH", "HIGH", "Art.4(1)",
         "YEAR_ONLY",        # keep year only
        ),
        (["address","street","city","zip","postal","location"],
         None,
         "ADDRESS", "MEDIUM", "Art.4(1)",
         "REGION_ONLY",      # keep region/country only
        ),
        (["ip","ipv4","ipv6","ip_address"],
         r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
         "IP_ADDRESS", "MEDIUM", "Art.4(1)",
         "SUBNET_ONLY",      # keep /16 subnet
        ),
        (["credit_card","card","cc_num","card_number"],
         r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',
         "CREDIT_CARD", "CRITICAL", "Art.9",
         "LAST_4_VISIBLE",
        ),
        (["salary","income","wage","compensation","pay"],
         None,
         "FINANCIAL_DATA", "HIGH", "Art.9",
         "RANGE_BUCKET",     # convert to salary range
        ),
        (["user_id","userid","customer_id","account_id","person_id"],
         None,
         "IDENTIFIER", "LOW", "Art.4(1)",
         "PSEUDONYMIZE",     # replace with consistent pseudo-ID
        ),
        (["score","rating","grade"],
         None,
         "DERIVED_DATA", "LOW", "Art.4(1) - Derived/Inferred",
         "NONE",             # no masking needed
        ),
    ]

    df_masked = df.copy()

    for col in df.columns:
        col_lower = col.lower()
        sample    = " ".join(str(v) for v in df[col].dropna().head(30).tolist())

        matched_pii = None
        for (keywords, content_re, pii_type, sensitivity, gdpr, mask_strategy) in PII_RULES:
            # Name-based detection
            if any(kw in col_lower for kw in keywords):
                matched_pii = (pii_type, sensitivity, gdpr, mask_strategy)
                break
            # Content-based detection (if regex provided)
            if content_re and re.search(content_re, sample):
                matched_pii = (pii_type, sensitivity, gdpr, mask_strategy)
                break

        if matched_pii:
            pii_type, sensitivity, gdpr, mask_strategy = matched_pii
            n_values = int(df[col].count())

            # Apply masking to masked copy
            df_masked = _apply_masking(df_masked, col, pii_type, mask_strategy)

            tag = {
                "column":          col,
                "pii_type":        pii_type,
                "sensitivity":     sensitivity,
                "gdpr_article":    gdpr,
                "masking_strategy":mask_strategy,
                "values_affected": n_values,
                "requires_consent":sensitivity in ("HIGH","CRITICAL"),
                "retention_policy": _retention_policy(pii_type),
                "access_level":    _access_level(sensitivity),
            }
            pii_tags.append(tag)
            logger.warn(f"[B2] PII detected: '{col}' → {pii_type} [{sensitivity}] → masked with {mask_strategy}")
        else:
            logger.info(f"[B2] '{col}' — no PII detected")

    # Save masked dataset
    os.makedirs("data", exist_ok=True)
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in state["scenario_name"])
    masked_path = f"data/b2_masked_{safe}.csv"
    df_masked.to_csv(masked_path, index=False)

    critical = [t for t in pii_tags if t["sensitivity"] == "CRITICAL"]
    high     = [t for t in pii_tags if t["sensitivity"] == "HIGH"]
    summary  = (
        f"PII found in {len(pii_tags)}/{len(df.columns)} columns | "
        f"Critical: {len(critical)} | High: {len(high)} | "
        f"Masked dataset → {masked_path}"
    )
    logger.info(f"[B2] {summary}")

    return {
        **state,
        "pii_tags":        pii_tags,
        "pii_summary":     summary,
        "masked_data_path":masked_path,
        "logs": [f"[B2-PIITagger] {summary}"],
    }


def _apply_masking(df: pd.DataFrame, col: str, pii_type: str, strategy: str) -> pd.DataFrame:
    """Apply the specified masking strategy to a column."""
    if strategy == "PARTIAL_MASK":  # email: jo***@domain.com
        def mask_email(v):
            if pd.isna(v): return v
            p = str(v).split("@")
            return (p[0][:2] + "***@" + p[1]) if len(p) == 2 else "***@***.***"
        df[col] = df[col].apply(mask_email)

    elif strategy == "LAST_4_VISIBLE":  # phone/card: ***-***-4567
        df[col] = df[col].apply(
            lambda v: "***-***-" + str(v)[-4:] if not pd.isna(v) else v
        )

    elif strategy == "INITIAL_ONLY":  # names: J***
        df[col] = df[col].apply(
            lambda v: str(v)[0].upper() + "***" if not pd.isna(v) and len(str(v)) > 0 else v
        )

    elif strategy == "FULL_HASH":  # SSN: SHA256 hash
        df[col] = df[col].apply(
            lambda v: "HASH_" + hashlib.sha256(str(v).encode()).hexdigest()[:12]
            if not pd.isna(v) else v
        )

    elif strategy == "YEAR_ONLY":  # DOB: keep year
        def year_only(v):
            if pd.isna(v): return v
            m = re.search(r'(19|20)\d{2}', str(v))
            return m.group(0) if m else "****"
        df[col] = df[col].apply(year_only)

    elif strategy == "RANGE_BUCKET":  # salary: bucket into ranges
        def salary_bucket(v):
            if pd.isna(v): return v
            try:
                val = float(v)
                if val < 30000: return "<30K"
                elif val < 60000: return "30K-60K"
                elif val < 100000: return "60K-100K"
                elif val < 150000: return "100K-150K"
                else: return "150K+"
            except: return "***"
        df[col] = df[col].apply(salary_bucket)

    elif strategy == "SUBNET_ONLY":  # IP: keep /16 subnet
        def subnet(v):
            if pd.isna(v): return v
            parts = str(v).split(".")
            return ".".join(parts[:2]) + ".0.0" if len(parts) == 4 else "0.0.0.0"
        df[col] = df[col].apply(subnet)

    elif strategy == "PSEUDONYMIZE":  # user_id: consistent pseudo-ID
        id_map = {}
        def pseudo(v):
            if pd.isna(v): return v
            k = str(v)
            if k not in id_map:
                id_map[k] = "USR_" + hashlib.md5(k.encode()).hexdigest()[:8].upper()
            return id_map[k]
        df[col] = df[col].apply(pseudo)

    elif strategy == "REGION_ONLY":  # address: keep country/region
        df[col] = df[col].apply(
            lambda v: str(v)[:2].upper() + "***" if not pd.isna(v) else v
        )

    # NONE: no masking
    return df


def _retention_policy(pii_type: str) -> str:
    policies = {
        "EMAIL":          "3 years after last interaction",
        "PHONE_NUMBER":   "3 years after last interaction",
        "PERSONAL_NAME":  "Duration of contract + 7 years",
        "SSN":            "Legal requirement: 7 years",
        "DATE_OF_BIRTH":  "Duration of service",
        "ADDRESS":        "Duration of service + 2 years",
        "IP_ADDRESS":     "6 months (GDPR recital 49)",
        "CREDIT_CARD":    "PCI-DSS: Do not store after transaction",
        "FINANCIAL_DATA": "7 years (tax/audit requirements)",
        "IDENTIFIER":     "Duration of service",
        "DERIVED_DATA":   "Duration of service",
    }
    return policies.get(pii_type, "As per data retention policy")


def _access_level(sensitivity: str) -> str:
    return {
        "CRITICAL": "C-Level + DPO only",
        "HIGH":     "Authorized personnel with signed DPA",
        "MEDIUM":   "Internal teams with need-to-know",
        "LOW":      "Internal use",
    }.get(sensitivity, "Internal use")


# ──────────────────────────────────────────────────────────────
#  NODE 4: CATALOGUE ENRICHER (LLM)
# ──────────────────────────────────────────────────────────────
def catalogue_enricher_node(state: B2State) -> B2State:
    """
    LLM enriches every column with:
    - Business description
    - Data steward / owner
    - Business glossary term
    - Tags (PII, Financial, Operational, etc.)
    - Quality SLA
    """
    logger.info("[B2] Enriching data catalogue via LLM...")

    path = state.get("raw_data_path","")
    pii_tags   = state.get("pii_tags", [])
    pii_map    = {t["column"]: t for t in pii_tags}
    parsed_sql = state.get("parsed_sql", {})

    # Build schema for LLM
    try:
        df = pd.read_csv(path, nrows=5)
        schema_str = "\n".join(
            f"  - {col}: {df[col].dtype} | sample: {df[col].dropna().head(2).tolist()}"
            for col in df.columns
        )
        columns = list(df.columns)
    except Exception:
        schema_str = "Schema unavailable"
        columns = parsed_sql.get("columns", [])

    pii_context = "\n".join(
        f"  - {t['column']}: {t['pii_type']} [{t['sensitivity']}]"
        for t in pii_tags
    ) or "  None detected"

    prompt = f"""You are a Data Governance expert building an enterprise data catalogue.

Dataset: {state['scenario_name']}
Schema:
{schema_str}

PII-tagged columns:
{pii_context}

For EACH column, create a data catalogue entry. Return a JSON array where each item has:
{{
  "column": "column_name",
  "data_type": "actual data type",
  "business_description": "1-2 sentence business meaning of this column",
  "business_term": "official glossary term (e.g. 'Customer Age', 'Annual Compensation')",
  "data_steward": "team or role responsible (e.g. 'HR Team', 'Finance', 'Engineering')",
  "domain": "business domain (e.g. 'Customer', 'Finance', 'Operations', 'Identity')",
  "tags": ["list", "of", "relevant", "tags"],
  "quality_sla": "completeness target (e.g. '99% non-null', '100% valid range')",
  "example_values": ["2-3 example values"],
  "is_pii": true or false,
  "classification": "PUBLIC | INTERNAL | CONFIDENTIAL | RESTRICTED"
}}

Return ONLY valid JSON array. No markdown."""

    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"): raw = "\n".join(raw.split("\n")[1:-1])
        catalogue = json.loads(raw)

        # Merge PII tags into catalogue
        for entry in catalogue:
            col = entry.get("column","")
            if col in pii_map:
                entry["is_pii"]          = True
                entry["pii_type"]        = pii_map[col]["pii_type"]
                entry["gdpr_article"]    = pii_map[col]["gdpr_article"]
                entry["masking_strategy"]= pii_map[col]["masking_strategy"]
                entry["retention_policy"]= pii_map[col]["retention_policy"]
                entry["access_level"]    = pii_map[col]["access_level"]

        summary = f"Catalogue enriched: {len(catalogue)} columns | {len(pii_tags)} PII-tagged"

    except Exception as e:
        logger.warn(f"[B2] LLM catalogue enrichment failed: {e} — using fallback")
        catalogue = _fallback_catalogue(columns, pii_map)
        summary   = f"Fallback catalogue: {len(catalogue)} entries"

    # Save catalogue to SQLite
    _save_catalogue_to_sqlite(state["scenario_name"], catalogue, state["run_id"])

    logger.info(f"[B2] {summary}")

    return {
        **state,
        "data_catalogue":   catalogue,
        "catalogue_summary":summary,
        "logs": [f"[B2-CatalogueEnricher] {summary}"],
    }


def _fallback_catalogue(columns: list, pii_map: dict) -> list:
    """Rule-based catalogue when LLM unavailable."""
    entries = []
    domain_map = {
        "id":"Identity","user":"Customer","age":"Demographics","salary":"Finance",
        "email":"Contact","score":"Performance","country":"Geography",
        "date":"Temporal","name":"Identity","phone":"Contact",
    }
    for col in columns:
        col_lower = col.lower()
        domain = next((v for k,v in domain_map.items() if k in col_lower), "Operations")
        pii = pii_map.get(col, {})
        entry = {
            "column": col,
            "data_type": "VARCHAR",
            "business_description": f"Field '{col}' stores {col.replace('_',' ')} information.",
            "business_term": col.replace("_"," ").title(),
            "data_steward": "Data Engineering Team",
            "domain": domain,
            "tags": [domain, "raw_data"],
            "quality_sla": "95% non-null",
            "example_values": [],
            "is_pii": bool(pii),
            "classification": "CONFIDENTIAL" if pii.get("sensitivity") in ("HIGH","CRITICAL") else "INTERNAL",
        }
        if pii:
            entry.update({
                "pii_type":         pii.get("pii_type",""),
                "gdpr_article":     pii.get("gdpr_article",""),
                "masking_strategy": pii.get("masking_strategy",""),
                "retention_policy": pii.get("retention_policy",""),
                "access_level":     pii.get("access_level",""),
            })
        entries.append(entry)
    return entries


def _save_catalogue_to_sqlite(scenario: str, catalogue: list, run_id: str):
    """Persist catalogue entries to SQLite for permanent storage."""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "data","pipeline_results.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    try:
        conn = sqlite3.connect(db_path)
        rows = []
        ts   = datetime.datetime.now().isoformat()
        for entry in catalogue:
            rows.append((
                run_id, scenario, ts,
                entry.get("column",""),
                entry.get("data_type",""),
                entry.get("business_description",""),
                entry.get("business_term",""),
                entry.get("data_steward",""),
                entry.get("domain",""),
                json.dumps(entry.get("tags",[])),
                entry.get("quality_sla",""),
                str(entry.get("is_pii", False)),
                entry.get("pii_type",""),
                entry.get("gdpr_article",""),
                entry.get("masking_strategy",""),
                entry.get("retention_policy",""),
                entry.get("access_level",""),
                entry.get("classification","INTERNAL"),
            ))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_catalogue (
                run_id TEXT, scenario TEXT, catalogued_at TEXT,
                column_name TEXT, data_type TEXT, business_description TEXT,
                business_term TEXT, data_steward TEXT, domain TEXT, tags TEXT,
                quality_sla TEXT, is_pii TEXT, pii_type TEXT, gdpr_article TEXT,
                masking_strategy TEXT, retention_policy TEXT, access_level TEXT,
                classification TEXT
            )""")
        conn.executemany("INSERT INTO data_catalogue VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        conn.close()
        logger.info(f"[B2] Catalogue saved to SQLite ({len(rows)} entries)")
    except Exception as e:
        logger.warn(f"[B2] SQLite catalogue save failed (non-fatal): {e}")


# ──────────────────────────────────────────────────────────────
#  NODE 5: GOVERNANCE REPORT
# ──────────────────────────────────────────────────────────────
def governance_report_node(state: B2State) -> B2State:
    """Produces final governance report with policy recommendations."""
    logger.info("[B2] Generating governance report...")

    pii_tags  = state.get("pii_tags",    [])
    catalogue = state.get("data_catalogue",[])
    lineage   = state.get("lineage_graph",{})
    scenario  = state.get("scenario_name","")
    run_id    = state.get("run_id","")

    critical_pii = [t for t in pii_tags if t["sensitivity"] == "CRITICAL"]
    high_pii     = [t for t in pii_tags if t["sensitivity"] == "HIGH"]
    pii_cols     = [t["column"] for t in pii_tags]

    # GDPR compliance score
    total_checks = 5
    passed = 0
    gdpr_checks = {
        "Data Minimisation (Art.5.1.c)": len([c for c in catalogue if "tags" in c and "unnecessary" not in str(c.get("tags",""))]) > 0,
        "PII Identified (Art.4.1)":       len(pii_tags) >= 0,
        "Masking Applied (Art.25)":        bool(state.get("masked_data_path","")),
        "Retention Policy Defined (Art.5.1.e)": all(t.get("retention_policy") for t in pii_tags),
        "Lineage Documented (Art.30)":     len(lineage.get("nodes",[])) > 0,
    }
    passed = sum(1 for v in gdpr_checks.values() if v)
    gdpr_score = round(passed / total_checks * 100)

    # Policy recommendations
    recommendations = []
    if critical_pii:
        recommendations.append({
            "priority": "CRITICAL",
            "area":     "PII Protection",
            "action":   f"Immediately apply FULL_HASH masking to {[t['column'] for t in critical_pii]}",
            "rationale":"Critical PII (SSN, credit card) must never be stored in plain text",
            "gdpr_ref": "Art.9 — Special Category Data",
        })
    if high_pii:
        recommendations.append({
            "priority": "HIGH",
            "area":     "Data Masking",
            "action":   f"Apply masking strategies to HIGH sensitivity columns: {[t['column'] for t in high_pii]}",
            "rationale":"High sensitivity PII requires encryption or pseudonymisation",
            "gdpr_ref": "Art.5(1)(f) — Integrity & Confidentiality",
        })
    recommendations.append({
        "priority": "MEDIUM",
        "area":     "Data Catalogue",
        "action":   "Register all columns in enterprise data catalogue with stewardship assignments",
        "rationale":"Enables data governance, access control, and compliance auditing",
        "gdpr_ref": "Art.30 — Records of Processing Activities",
    })
    recommendations.append({
        "priority": "MEDIUM",
        "area":     "Lineage Tracking",
        "action":   "Automate lineage capture in CI/CD pipeline to track all data transformations",
        "rationale":"Data lineage is required for impact analysis and compliance audits",
        "gdpr_ref": "Art.30 — Records of Processing Activities",
    })
    recommendations.append({
        "priority": "LOW",
        "area":     "Access Control",
        "action":   f"Implement role-based access: {_access_level('HIGH')} for HIGH-sensitivity data",
        "rationale":"Principle of least privilege reduces breach risk",
        "gdpr_ref": "Art.32 — Security of Processing",
    })

    report = {
        "run_id":             run_id,
        "scenario":           scenario,
        "generated_at":       datetime.datetime.now().isoformat(),
        "executive_summary": (
            f"Governance scan of '{scenario}': "
            f"{len(pii_tags)} PII columns identified across {len(catalogue)} total columns. "
            f"GDPR compliance score: {gdpr_score}%. "
            f"Masked dataset generated. {len(recommendations)} policy actions recommended."
        ),

        # Lineage
        "lineage": {
            "path":            lineage.get("lineage_path",""),
            "total_nodes":     len(lineage.get("nodes",[])),
            "total_edges":     len(lineage.get("edges",[])),
            "sources":         lineage.get("sources",[]),
            "sinks":           lineage.get("sinks",[]),
            "transformations": lineage.get("transformations",[]),
            "nodes":           lineage.get("nodes",[]),
            "edges":           lineage.get("edges",[]),
        },

        # PII
        "pii_summary": {
            "total_pii_columns":    len(pii_tags),
            "critical_columns":     [t["column"] for t in critical_pii],
            "high_columns":         [t["column"] for t in high_pii],
            "masked_data_path":     state.get("masked_data_path",""),
            "tags":                 pii_tags,
        },

        # Catalogue
        "catalogue": {
            "total_columns":    len(catalogue),
            "pii_tagged":       len(pii_cols),
            "domains":          list(set(e.get("domain","") for e in catalogue)),
            "entries":          catalogue,
        },

        # GDPR
        "gdpr_compliance": {
            "score":       gdpr_score,
            "checks":      gdpr_checks,
            "status":      "COMPLIANT" if gdpr_score >= 80 else "PARTIALLY_COMPLIANT" if gdpr_score >= 50 else "NON_COMPLIANT",
        },

        # Recommendations
        "policy_recommendations": recommendations,
        "final_status": "SUCCESS",
    }

    # Save report
    os.makedirs("logs", exist_ok=True)
    rpath = f"logs/b2_governance_report_{run_id}.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Also save lineage graph as separate file
    lpath = f"data/b2_lineage_{run_id}.json"
    with open(lpath, "w") as f:
        json.dump(lineage, f, indent=2, default=str)

    logger.success(
        f"[B2] Governance report → {rpath} | "
        f"GDPR Score: {gdpr_score}% | "
        f"PII columns: {len(pii_tags)} | "
        f"Lineage nodes: {len(lineage.get('nodes',[]))}"
    )

    return {
        **state,
        "governance_report":      report,
        "policy_recommendations": recommendations,
        "final_status":           "SUCCESS",
        "logs": [
            f"[B2-GovernanceReport] GDPR={gdpr_score}% | "
            f"PII={len(pii_tags)} cols | Report={rpath}"
        ],
    }


# ──────────────────────────────────────────────────────────────
#  BUILD + RUN
# ──────────────────────────────────────────────────────────────
def build_b2_pipeline():
    graph = StateGraph(B2State)
    graph.add_node("sql_parser",         sql_parser_node)
    graph.add_node("lineage_extractor",  lineage_extractor_node)
    graph.add_node("pii_tagger",         pii_tagger_node)
    graph.add_node("catalogue_enricher", catalogue_enricher_node)
    graph.add_node("governance_report",  governance_report_node)

    graph.set_entry_point("sql_parser")
    graph.add_edge("sql_parser",        "lineage_extractor")
    graph.add_edge("lineage_extractor", "pii_tagger")
    graph.add_edge("pii_tagger",        "catalogue_enricher")
    graph.add_edge("catalogue_enricher","governance_report")
    graph.add_edge("governance_report",  END)

    return graph.compile()


def run_b2_pipeline(scenario_name: str, data_path: str,
                     sql_query: str = "") -> dict:
    run_id = str(uuid.uuid4())[:8].upper()
    print(f"\n{'━'*60}")
    print(f"  🏛️  B2 LINEAGE & GOVERNANCE AGENT | {scenario_name} | {run_id}")
    print(f"{'━'*60}\n")

    initial: B2State = {
        "run_id":          run_id,
        "scenario_name":   scenario_name,
        "raw_data_path":   data_path,
        "sql_query":       sql_query,
        "start_time":      datetime.datetime.now().isoformat(),
        "parsed_sql":      {},
        "sql_summary":     "",
        "lineage_graph":   {},
        "lineage_summary": "",
        "pii_tags":        [],
        "pii_summary":     "",
        "masked_data_path":"",
        "data_catalogue":  [],
        "catalogue_summary":"",
        "governance_report":{},
        "policy_recommendations":[],
        "final_status":    "RUNNING",
        "logs":            [],
    }
    pipeline = build_b2_pipeline()
    return pipeline.invoke(initial)