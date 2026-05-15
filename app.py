"""
app.py  —  Self-Healing Data Pipeline Agent  (v3 — MCP + MockDB + GE)
KEY FIX: All get_db() calls are LAZY (inside tab callbacks only).
         No module-level DB connection → no DuckDB lock conflict.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os, sys, re, time, datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="Self-Healing Pipeline Agent",
    page_icon="🔧", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(160deg,#080c18 0%,#0c1324 30%,#101b30 60%,#0e1628 100%) !important; }
[data-testid="stMetric"] {
    background: linear-gradient(145deg,rgba(22,30,48,0.9),rgba(16,22,38,0.7));
    border: 1px solid rgba(99,130,255,0.1); border-radius: 14px;
    padding: 16px 20px; box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(59,130,246,0.15); }
[data-testid="stMetricLabel"] { color:#7b8aa3 !important; font-size:10px !important; font-weight:700 !important; text-transform:uppercase; letter-spacing:1px !important; }
[data-testid="stMetricValue"] { color:#e2e8f0 !important; font-size:24px !important; font-weight:700 !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,rgba(10,16,30,0.97),rgba(8,12,24,0.99)) !important; border-right: 1px solid rgba(99,130,255,0.08) !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: linear-gradient(135deg,#3b82f6,#8b5cf6) !important; border:none !important; border-radius:12px !important; font-weight:600 !important; box-shadow:0 4px 20px rgba(59,130,246,0.3) !important; transition: all 0.2s ease !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover { box-shadow:0 6px 28px rgba(59,130,246,0.5) !important; transform: translateY(-1px) !important; }
.stTabs [data-baseweb="tab-list"] { gap:3px; background:rgba(12,16,28,0.7); border-radius:14px; padding:5px; border:1px solid rgba(99,130,255,0.06); }
.stTabs [data-baseweb="tab"] { border-radius:11px !important; padding:10px 20px !important; font-weight:500 !important; font-size:13px !important; color:#6b7b94 !important; transition: all 0.2s ease !important; }
.stTabs [data-baseweb="tab"]:hover { color:#93c5fd !important; background:rgba(59,130,246,0.08) !important; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,rgba(59,130,246,0.18),rgba(139,92,246,0.12)) !important; color:#93c5fd !important; font-weight:600 !important; box-shadow: 0 2px 12px rgba(59,130,246,0.15) !important; }
.pipe-box { display:inline-block; padding:8px 16px; border-radius:10px; font-size:12px; font-weight:600; text-align:center; margin:0 2px; box-shadow:0 2px 12px rgba(0,0,0,0.2); transition: transform 0.2s ease; }
.pipe-box:hover { transform: translateY(-2px); }
.agent-row { display:flex; align-items:center; justify-content:center; gap:0; flex-wrap:wrap; margin:10px 0 18px; padding:14px; background:rgba(12,16,28,0.6); border-radius:14px; border:1px solid rgba(99,130,255,0.06); }
.arrow-sep { color:#3b4a63; font-size:18px; padding:0 3px; }
.log-block { background:linear-gradient(135deg,#080c18,#0a0f1c); border-radius:12px; padding:16px; font-family:'JetBrains Mono','Fira Code',monospace; font-size:11px; line-height:2; max-height:300px; overflow-y:auto; border:1px solid rgba(99,130,255,0.08); }
.log-info{color:#60a5fa;} .log-warn{color:#fbbf24;} .log-ok{color:#34d399;} .log-err{color:#f87171;} .log-alert{color:#c084fc;}
.ge-pass { background:linear-gradient(135deg,rgba(2,44,34,0.6),rgba(6,78,59,0.3)); border:1px solid rgba(16,185,129,0.25); border-radius:10px; padding:8px 14px; margin:4px 0; font-size:12px; color:#34d399; }
.ge-fail { background:linear-gradient(135deg,rgba(76,5,25,0.6),rgba(127,29,29,0.3)); border:1px solid rgba(239,68,68,0.25); border-radius:10px; padding:8px 14px; margin:4px 0; font-size:12px; color:#f87171; }
.mcp-card { background:linear-gradient(145deg,rgba(22,30,48,0.9),rgba(16,22,38,0.6)); border:1px solid rgba(99,130,255,0.08); border-radius:14px; padding:16px 20px; margin-bottom:10px; transition: transform 0.2s ease, border-color 0.2s ease; }
.mcp-card:hover { transform: translateY(-2px); border-color: rgba(99,130,255,0.2); }
.mcp-tool-name { font-size:14px; font-weight:600; color:#93c5fd; }
.mcp-tool-desc { font-size:12px; color:#7b8aa3; margin-top:4px; line-height:1.6; }
[data-testid="stExpander"] { background:rgba(12,16,28,0.5) !important; border:1px solid rgba(99,130,255,0.06) !important; border-radius:14px !important; }
hr { border:none !important; height:1px !important; background:linear-gradient(90deg,transparent,rgba(99,130,255,0.15),transparent) !important; margin:18px 0 !important; }
.stDownloadButton > button { background:rgba(22,30,48,0.7) !important; border:1px solid rgba(99,130,255,0.12) !important; border-radius:10px !important; color:#93c5fd !important; transition: all 0.2s ease !important; }
.stDownloadButton > button:hover { border-color: rgba(99,130,255,0.3) !important; background:rgba(30,40,60,0.8) !important; }
::-webkit-scrollbar { width:5px; } ::-webkit-scrollbar-track { background:rgba(12,16,28,0.3); } ::-webkit-scrollbar-thumb { background:rgba(99,130,255,0.15); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:rgba(99,130,255,0.3); }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
for k, v in [("all_results",[]),("ran",False),("uploaded_df",None),
              ("uploaded_path",None),("upload_name",None),("run_history",[]),
              ("sqlite_exports",{})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_builtin_datasets():
    os.makedirs(os.path.join(ROOT,"data"), exist_ok=True)
    os.makedirs(os.path.join(ROOT,"logs"), exist_ok=True)
    from data.generate_data import (generate_missing_values_dataset,
        generate_schema_mismatch_dataset, generate_anomaly_dataset, save_all)
    save_all()
    return {
        "missing_values":  generate_missing_values_dataset(),
        "schema_mismatch": generate_schema_mismatch_dataset(),
        "data_anomaly":    generate_anomaly_dataset(),
    }

def run_single_scenario(scenario_name, data_path):
    import io as _io, contextlib
    from workflow.pipeline_graph import run_pipeline
    buf  = _io.StringIO()
    ansi = re.compile(r'\x1b\[[0-9;]*m')
    with contextlib.redirect_stdout(buf):
        result = run_pipeline(scenario_name, data_path)
    logs = [ansi.sub('',l).strip() for l in buf.getvalue().splitlines() if l.strip()]
    return result, logs

def log_html(line):
    safe = line.replace("<","&lt;").replace(">","&gt;")
    if   "[SUCCESS]" in line: css = "log-ok"
    elif "[WARN]"    in line: css = "log-warn"
    elif "[ERROR]"   in line: css = "log-err"
    elif "[ALERT]"   in line: css = "log-alert"
    else:                      css = "log-info"
    return f'<div class="{css}">{safe}</div>'

def safe_read_csv(path):
    try:
        if not path or not os.path.exists(path): return None
        if os.path.getsize(path) == 0: return None
        df = pd.read_csv(path)
        return df if not df.empty and len(df.columns) > 0 else None
    except: return None

def quality_score(df_orig, df_healed, n_issues, n_fixes):
    if df_orig is None or df_healed is None: return 0
    on = df_orig.isna().sum().sum(); hn = df_healed.isna().sum().sum()
    s  = (max(0,1-hn/max(on,1))*40) if on > 0 else 40
    s += (n_fixes/max(n_issues,1))*30
    s += min(len(df_healed)/max(len(df_orig),1),1.0)*20
    s += (len(set(df_healed.columns)&set(df_orig.columns))/max(len(df_orig.columns),1))*10
    return int(min(s,100))

def score_color(s):
    return "#34d399" if s>=85 else "#fbbf24" if s>=60 else "#f87171"

def render_gauge(score):
    c = score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text":"Quality Score","font":{"size":13,"color":"#8b95a8"}},
        number={"suffix":"/100","font":{"size":28,"color":c}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#4b5563","tickfont":{"color":"#8b95a8","size":10}},
               "bar":{"color":c,"thickness":0.3},"bgcolor":"#1e2535","bordercolor":"#2d3748",
               "steps":[{"range":[0,60],"color":"#4c0519"},{"range":[60,85],"color":"#451a03"},{"range":[85,100],"color":"#022c22"}]}
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#8b95a8",height=200,margin=dict(l=10,r=10,t=40,b=5))
    return fig

datasets = load_builtin_datasets()

# ── Import integrations (lazy — no module-level DB connection) ─
DB_AVAILABLE = False
try:
    from utils.mock_db import get_db, _ensure_schema_exists
    from utils.ge_validator import run_pre_healing_suite, run_post_healing_suite, run_custom_suite
    from mcp_server.pipeline_mcp_server import get_mcp_server, TOOLS
    _ensure_schema_exists()   # creates tables if missing — write+close, safe
    DB_AVAILABLE = True
except Exception as _e:
    pass  # shown in sidebar


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style='text-align:center;padding:8px 0 4px'>
  <div style='font-size:22px;font-weight:800;background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent'>🔧 Pipeline Agent</div>
  <div style='font-size:10px;color:#6b7280;font-weight:500;letter-spacing:1.5px;text-transform:uppercase;margin-top:2px'>v4 · B1+B2+B3 · LangGraph · Gemini · MCP</div>
</div>""", unsafe_allow_html=True)

    _dot = '🟢' if DB_AVAILABLE else '🔴'
    st.markdown(f"""
<div style='display:flex;justify-content:center;gap:12px;margin:12px 0;padding:10px;background:rgba(15,20,35,0.5);border-radius:12px;border:1px solid rgba(99,130,255,0.08)'>
  <span style='font-size:12px;font-weight:500'>{_dot} <span style='color:#8b95a8'>MockDB</span></span>
  <span style='color:#2d3748'>│</span>
  <span style='font-size:12px;font-weight:500'>{_dot} <span style='color:#8b95a8'>GE</span></span>
  <span style='color:#2d3748'>│</span>
  <span style='font-size:12px;font-weight:500'>{_dot} <span style='color:#8b95a8'>MCP</span></span>
</div>""", unsafe_allow_html=True)

    st.divider()
    api_key = st.text_input("🔑 Gemini API Key", type="password",
                             placeholder="AIzaSy...",
                             value=os.environ.get("GEMINI_API_KEY",""))
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    st.divider()
    st.markdown("""
<div style='font-size:12px;font-weight:600;color:#93c5fd;margin-bottom:8px'>📁 Upload Dataset</div>
""", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed", accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) > 1:
            st.warning("⚠️ Multiple files uploaded. Only the first file will be processed.")
        uploaded_file = uploaded_files[0]
        
        # Only re-process if file changed
        if st.session_state.upload_name != uploaded_file.name:
            try:
                uploaded_file.seek(0)  # ensure read from start
                df_up = pd.read_csv(uploaded_file)
                if df_up.empty or len(df_up.columns) == 0:
                    st.error("❌ File is empty or has no columns.")
                else:
                    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
                    p = os.path.join(ROOT, "data", "custom_upload.csv")
                    df_up.to_csv(p, index=False)
                    st.session_state.uploaded_df   = df_up
                    st.session_state.uploaded_path = p
                    st.session_state.upload_name   = uploaded_file.name
                    st.session_state.ran = False
                    st.session_state.all_results = []
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Cannot read CSV: {e}")
    else:
        # User cleared the uploader — reset state
        if st.session_state.upload_name is not None:
            st.session_state.uploaded_df   = None
            st.session_state.uploaded_path = None
            st.session_state.upload_name   = None

    if st.session_state.uploaded_df is not None:
        df_info = st.session_state.uploaded_df
        st.success(f"✅ {st.session_state.upload_name}")
        st.caption(f"{len(df_info)} rows · {len(df_info.columns)} cols · {int(df_info.isna().sum().sum())} nulls")
    else:
        st.info("⬆️ Upload a CSV file to get started.")

    # scenario_choice is always "custom" — agents use their own selectors
    scenario_choice = "custom"

    st.divider()
    run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)
    st.divider()
    st.markdown("""
<div style='background:rgba(15,20,35,0.5);border-radius:12px;border:1px solid rgba(99,130,255,0.08);padding:14px;font-size:12px;line-height:2.2'>
  <div style='color:#9ca3af;font-weight:700;font-size:10px;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:4px'>Tech Stack</div>
  <div style='color:#60a5fa'>🔗 LangGraph 15-node orchestration</div>
  <div style='color:#a78bfa'>🧠 Gemini LLM reasoning</div>
  <div style='color:#fbbf24'>🗄️ DuckDB MockDB warehouse</div>
  <div style='color:#34d399'>✅ Great Expectations GE</div>
  <div style='color:#c084fc'>🔌 MCP tool server (8 tools)</div>
  <div style='margin-top:8px;font-size:11px;color:#6b7280'>
    <b style='color:#93c5fd'>B1</b> Ingestion Quality ·
    <b style='color:#c4b5fd'>B2</b> Lineage & Governance ·
    <b style='color:#6ee7b7'>B3</b> Self-Healing
  </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding:6px 0 2px'>
  <h1 style='font-size:26px;font-weight:800;margin-bottom:0;letter-spacing:-0.5px'>
    <span style='background:linear-gradient(135deg,#60a5fa,#a78bfa,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent'>Self-Healing Data Pipeline</span>
    <span style='font-size:10px;background:linear-gradient(135deg,rgba(30,37,53,0.8),rgba(46,16,101,0.4));color:#93c5fd;padding:3px 10px;border-radius:16px;margin-left:8px;font-weight:600;vertical-align:middle;border:1px solid rgba(99,130,255,0.1)'>v4 · B1+B2+B3</span>
  </h1>
  <p style='color:#4b5c75;font-size:12px;margin-top:4px'>
    <span style='color:#60a5fa'>LangGraph</span> ·
    <span style='color:#a78bfa'>Gemini LLM</span> ·
    <span style='color:#fbbf24'>DuckDB</span> ·
    <span style='color:#34d399'>Great Expectations</span> ·
    <span style='color:#c084fc'>MCP Server</span>
  </p>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class='agent-row'>
  <span class='pipe-box' style='background:linear-gradient(135deg,#1e3a5f,#1a365d);color:#93c5fd;border:1px solid rgba(59,130,246,0.15)'>🔍 Detection</span><span class='arrow-sep'>→</span>
  <span class='pipe-box' style='background:linear-gradient(135deg,#2e1065,#4c1d95);color:#c4b5fd;border:1px solid rgba(139,92,246,0.15)'>🏷️ Classification</span><span class='arrow-sep'>→</span>
  <span class='pipe-box' style='background:linear-gradient(135deg,#451a03,#78350f);color:#fcd34d;border:1px solid rgba(245,158,11,0.15)'>🧠 Decision</span><span class='arrow-sep'>→</span>
  <span class='pipe-box' style='background:linear-gradient(135deg,#022c22,#064e3b);color:#6ee7b7;border:1px solid rgba(16,185,129,0.15)'>🔨 Healing+GE</span><span class='arrow-sep'>→</span>
  <span class='pipe-box' style='background:linear-gradient(135deg,#14532d,#166534);color:#86efac;border:1px solid rgba(34,197,94,0.15)'>📋 Logging</span><span class='arrow-sep'>→</span>
  <span class='pipe-box' style='background:linear-gradient(135deg,#2d1b4e,#581c87);color:#d8b4fe;border:1px solid rgba(168,85,247,0.15)'>🗄️ MockDB</span>
</div>""", unsafe_allow_html=True)
st.divider()


# ═══════════════════════════════════════════════════════════════
#  TABS (7 consolidated tabs)
# ═══════════════════════════════════════════════════════════════
(tab_preview, tab_run,
 tab_b1, tab_b2, tab_b3,
 tab_results, tab_infra) = st.tabs([
    "📊 Data Preview", "▶️ Run Pipeline",
    "🔬 B1 · Data Quality", "🏛️ B2 · Governance", "🔧 B3 · Self-Healing",
    "📈 Results & History", "⚙️ Infrastructure",
])


# ─── TAB 1: DATA PREVIEW ──────────────────────────────────────
with tab_preview:
    st.markdown("### Dataset Preview")
    if st.session_state.uploaded_df is not None:
        df_p = st.session_state.uploaded_df
        st.caption(f"📁 Showing uploaded file: **{st.session_state.upload_name}**")
    else:
        st.info("⬆️ Upload a CSV in the sidebar to preview your data.")
        df_p = None
    if df_p is not None:
        nt = int(df_p.isna().sum().sum())
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows",len(df_p)); c2.metric("Columns",len(df_p.columns))
        c3.metric("Missing cells",nt)
        c4.metric("Missing %",f"{round(nt/max(df_p.size,1)*100,1)}%",
                  delta="needs fix" if nt>0 else "✓ clean",delta_color="inverse" if nt>0 else "normal")
        st.dataframe(df_p.head(20).style.highlight_null(color="#4c0519").format(precision=2),
                     use_container_width=True, height=280)
        if nt>0:
            fig_h = px.imshow(df_p.isna().astype(int).T,color_continuous_scale=["#1e2535","#ef4444"],height=150)
            fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#8b95a8",
                                margin=dict(l=0,r=0,t=10,b=0),coloraxis_showscale=False)
            fig_h.update_xaxes(showticklabels=False)
            st.plotly_chart(fig_h, use_container_width=True)


# ─── TAB 2: RUN PIPELINE ──────────────────────────────────────
with tab_run:
    st.markdown("### Run the Self-Healing Agent")
    if not api_key:
        st.warning("⚠️ No API key — fallback rule-based engine will run.")


    if run_btn:
        if st.session_state.uploaded_df is None:
            st.error("⬆️ Upload a CSV first in the sidebar.")
            scenarios_to_run = []
        else:
            scenarios_to_run = [(st.session_state.upload_name or "custom",
                                 st.session_state.uploaded_path)]

        all_results = []
        for s_name, s_path in scenarios_to_run:
            st.markdown(f"---\n#### ▶ `{s_name}`")
            if not os.path.exists(s_path):
                st.error(f"File not found: {s_path}"); continue
            t0 = time.time()
            with st.status(f"Running **{s_name}**...", expanded=True) as status:
                st.write("🔍 Detection Agent scanning...")
                st.write("🏷️ Classification Agent (LLM)...")
                st.write("🧠 Decision Agent (LLM)...")
                st.write("🔨 Healing Agent + GE validation...")
                st.write("🗄️ Writing to MockDB (write→close)...")
                st.write("📋 Logging Agent saving report...")
                try:
                    result, log_lines = run_single_scenario(s_name, s_path)
                    dur = round(time.time()-t0,1)
                    status.update(label=f"✅ `{s_name}` done in {dur}s",state="complete",expanded=False)
                except Exception as e:
                    import traceback
                    st.error(f"Error: {e}"); st.code(traceback.format_exc())
                    status.update(label=f"❌ `{s_name}` failed",state="error")
                    result = {"final_status":"FAILED","scenario_name":s_name,
                              "issues_detected":[],"fixes_applied":[],
                              "healed_data_path":"","ge_pre_results":{},"ge_post_results":{}}
                    log_lines=[f"ERROR: {e}"]; dur=0

            n_i=len(result.get("issues_detected",[])); n_f=len(result.get("fixes_applied",[]))
            df_o=safe_read_csv(s_path); df_h=safe_read_csv(result.get("healed_data_path",""))
            qs=quality_score(df_o,df_h,n_i,n_f)
            c1,c2,c3,c4,c5=st.columns(5)
            c1.metric("Status",result.get("final_status","?")); c2.metric("Duration",f"{dur}s")
            c3.metric("Issues",n_i); c4.metric("Fixes",n_f); c5.metric("Quality",f"{qs}/100")
            pre=result.get("ge_pre_results",{}); post=result.get("ge_post_results",{})
            if pre or post:
                gc1,gc2=st.columns(2)
                gc1.markdown(f"**GE Pre:** {pre.get('passed',0)}/{pre.get('total',0)} passed ({pre.get('success_pct',0)}%)")
                gc2.markdown(f"**GE Post:** {post.get('passed',0)}/{post.get('total',0)} passed ({post.get('success_pct',0)}%)")
            if log_lines:
                with st.expander("📋 Logs"):
                    st.markdown(f'<div class="log-block">{"".join(log_html(l) for l in log_lines)}</div>',
                                unsafe_allow_html=True)
            all_results.append(result)
            st.session_state.run_history.append({
                "time":datetime.datetime.now().strftime("%H:%M:%S"),
                "scenario":s_name,"status":result.get("final_status","?"),
                "issues":n_i,"fixes":n_f,"score":qs,"duration":f"{dur}s",
            })
        st.session_state.all_results=all_results; st.session_state.ran=True
        st.divider()
        st.success("🎉 Done! Check Results, GE, MockDB, MCP, and Output tabs.")
    else:
        st.info("👈 Configure in sidebar, then click **🚀 Run Pipeline**.")


# ─── TAB 6: RESULTS & HISTORY (merged) ─────────────────────────
with tab_results:
    if not st.session_state.ran and not st.session_state.run_history:
        st.info("Run the pipeline first to see results here.")
    else:
        res_sub1, res_sub2, res_sub3, res_sub4 = st.tabs(["📈 Results","✅ Great Expectations","🕐 History","📦 Outputs"])

        # ── Results sub-tab ──────────────────────────────────
        with res_sub1:
            if not st.session_state.ran:
                st.info("Run the pipeline first.")
            else:
                all_results=st.session_state.all_results
                total_i=sum(len(r.get("issues_detected",[])) for r in all_results)
                total_f=sum(len(r.get("fixes_applied",[])) for r in all_results)
                total_ok=sum(sum(1 for f in r.get("fixes_applied",[]) if f.get("status")=="SUCCESS") for r in all_results)
                m1,m2,m3,m4=st.columns(4)
                m1.metric("Scenarios",len(all_results)); m2.metric("Issues",total_i)
                m3.metric("Fixes",total_f); m4.metric("Success rate",f"{int(total_ok/max(total_f,1)*100)}%")
                for result in all_results:
                    sname=result.get("scenario_name","?"); status=result.get("final_status","?")
                    issues=result.get("issues_detected",[]); fixes=result.get("fixes_applied",[])
                    healed=result.get("healed_data_path",""); raw_p=result.get("raw_data_path","")
                    df_o=safe_read_csv(raw_p); df_h=safe_read_csv(healed)
                    qs=quality_score(df_o,df_h,len(issues),len(fixes))
                    with st.expander(f"{'✅' if status=='SUCCESS' else '⚠️'} **{sname}** — score {qs}/100 · {status}",expanded=True):
                        g_col,s_col=st.columns([1,3])
                        with g_col: st.plotly_chart(render_gauge(qs),use_container_width=True)
                        with s_col:
                            if df_o is not None and df_h is not None:
                                r1,r2,r3,r4=st.columns(4)
                                r1.metric("Orig rows",len(df_o)); r2.metric("Healed rows",len(df_h),delta=str(len(df_h)-len(df_o)))
                                r3.metric("Nulls",f"{int(df_o.isna().sum().sum())}→{int(df_h.isna().sum().sum())}")
                                r4.metric("Cols",f"{len(df_o.columns)}→{len(df_h.columns)}")
                        t1,t2,t3=st.tabs(["Issues","Fix Plan","Before vs After"])
                        with t1:
                            if issues:
                                df_i=pd.DataFrame([{"Type":i.get("type",""),"Column":i.get("column","—"),"Severity":i.get("severity",""),"Detail":i.get("detail","")} for i in issues])
                                sc=df_i["Severity"].value_counts()
                                l,r=st.columns(2)
                                with l:
                                    fig=px.pie(values=sc.values,names=sc.index,title="Severity",hole=.5,height=200,color=sc.index,color_discrete_map={"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#3b82f6"})
                                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#8b95a8",margin=dict(l=0,r=0,t=30,b=0))
                                    fig.update_traces(textinfo="percent+label",textfont_size=10)
                                    st.plotly_chart(fig,use_container_width=True)
                                with r: st.dataframe(df_i,use_container_width=True,hide_index=True)
                        with t2:
                            fp=result.get("fix_plan",[])
                            if fp:
                                df_p=pd.DataFrame([{"Action":f.get("action",""),"Column":next((i.get("column","") for i in issues if i.get("issue_id")==f.get("issue_id")),""),"Confidence":f.get("confidence",0),"Rationale":f.get("rationale","")} for f in fp])
                                st.dataframe(df_p,use_container_width=True,hide_index=True,column_config={"Confidence":st.column_config.ProgressColumn("Confidence",min_value=0,max_value=1,format="%.0%")})
                        with t3:
                            if df_o is not None and df_h is not None:
                                oc,hc=st.columns(2)
                                with oc:
                                    st.caption("🔴 Original")
                                    st.dataframe(df_o.head(8).style.highlight_null(color="#4c0519").format(precision=2),use_container_width=True,height=200)
                                with hc:
                                    st.caption("✅ Healed")
                                    st.dataframe(df_h.head(8).style.format(precision=2),use_container_width=True,height=200)

        # ── GE sub-tab ───────────────────────────────────────
        with res_sub2:
            st.markdown("### ✅ Great Expectations Validation")
            if not st.session_state.ran:
                st.info("Run the pipeline first, or validate manually below.")
                st.markdown("#### Manual GE Validation")
                ge_path  = st.text_input("CSV path", value="data/scenario_missing.csv")
                ge_suite = st.selectbox("Suite", ["pre_healing","post_healing","custom"])
                if st.button("▶ Run GE Validation") and DB_AVAILABLE:
                    if os.path.exists(ge_path):
                        df_ge = pd.read_csv(ge_path)
                        fns = {"pre_healing":run_pre_healing_suite,"post_healing":run_post_healing_suite,"custom":run_custom_suite}
                        with st.spinner("Running expectations..."):
                            res = fns[ge_suite](df_ge,"manual")
                        st.metric("Passed",f"{res['passed']}/{res['total']} ({res['success_pct']}%)")
                        for r in res["results"]:
                            cls="ge-pass" if r["passed"] else "ge-fail"
                            icon="✓" if r["passed"] else "✗"
                            st.markdown(f"<div class='{cls}'>{icon} <b>{r['expectation']}</b> · <code>{r['column']}</code> · observed: {r['observed']}"+(f" — {r['detail']}" if r.get('detail') else "")+"</div>",unsafe_allow_html=True)
                    else:
                        st.error("File not found.")
            else:
                for result in st.session_state.all_results:
                    sname=result.get("scenario_name","?")
                    pre=result.get("ge_pre_results",{}); post=result.get("ge_post_results",{})
                    if not pre and not post:
                        st.info(f"No GE results for `{sname}`."); continue
                    with st.expander(f"**{sname}** — GE Results",expanded=True):
                        pc1,pc2=st.columns(2)
                        def render_ge(res,label,col):
                            with col:
                                if not res: st.info(f"No {label} results"); return
                                pct=res.get("success_pct",0); color="#34d399" if pct>=80 else "#fbbf24" if pct>=50 else "#f87171"
                                st.markdown(f"<div style='background:rgba(30,37,53,0.8);border:1px solid rgba(99,130,255,0.12);border-radius:14px;padding:16px;margin-bottom:12px'><div style='font-size:13px;font-weight:600;color:#e2e8f0'>{label}</div><div style='font-size:26px;font-weight:700;color:{color};margin:6px 0'>{res.get('passed',0)}/{res.get('total',0)} passed</div><div style='font-size:12px;color:#8b95a8'>{pct}% success rate</div></div>",unsafe_allow_html=True)
                                for r in res.get("results",[]):
                                    cls="ge-pass" if r["passed"] else "ge-fail"
                                    icon="✓" if r["passed"] else "✗"
                                    st.markdown(f"<div class='{cls}'>{icon} <b>{r['expectation']}</b> · <code>{r['column']}</code> · {r['observed']}"+(f" ({r['detail']})" if r.get('detail') else "")+"</div>",unsafe_allow_html=True)
                        render_ge(pre,"🔴 Pre-Healing",pc1); render_ge(post,"✅ Post-Healing",pc2)
                        if pre and post:
                            fig_ge=go.Figure()
                            fig_ge.add_bar(x=["Pre","Post"],y=[pre.get("success_pct",0),post.get("success_pct",0)],marker_color=["#ef4444","#34d399"],text=[f"{pre.get('success_pct',0)}%",f"{post.get('success_pct',0)}%"],textposition="auto")
                            fig_ge.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#8b95a8",height=200,margin=dict(l=0,r=0,t=10,b=0),yaxis=dict(range=[0,105],gridcolor="#1e2535"),xaxis=dict(showgrid=False),showlegend=False)
                            st.plotly_chart(fig_ge,use_container_width=True)

        # ── History sub-tab ──────────────────────────────────
        with res_sub3:
            st.markdown("### 🕐 Run History")
            if not st.session_state.run_history:
                st.info("No runs yet.")
            else:
                h=st.session_state.run_history
                h1,h2,h3=st.columns(3)
                h1.metric("Total runs",len(h))
                h2.metric("Successful",sum(1 for r in h if r["status"]=="SUCCESS"))
                h3.metric("Avg score",f"{round(sum(r.get('score',0) for r in h)/max(len(h),1))}/100")
                df_hh=pd.DataFrame(h); df_hh.index=range(1,len(df_hh)+1)
                st.dataframe(df_hh,use_container_width=True,column_config={"score":st.column_config.ProgressColumn("Quality",min_value=0,max_value=100,format="%d/100")})
                if len(h)>1:
                    fig_t=px.line(df_hh,x=df_hh.index,y="score",color="scenario",markers=True,title="Quality score trend",height=260,color_discrete_sequence=["#3b82f6","#8b5cf6","#10b981"])
                    fig_t.add_hline(y=85,line_dash="dash",line_color="#34d399",annotation_text="Target")
                    fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#8b95a8",margin=dict(l=0,r=0,t=40,b=0),yaxis=dict(range=[0,105],gridcolor="#1e2535"),xaxis=dict(showgrid=False))
                    st.plotly_chart(fig_t,use_container_width=True)
                if st.button("🗑️ Clear history"): st.session_state.run_history=[]; st.rerun()

        # ── Outputs sub-tab ──────────────────────────────────
        with res_sub4:
            st.markdown("### 📦 Output Files")
            if not st.session_state.ran:
                st.info("Run the pipeline to generate output files.")
            else:
                try:
                    from utils.sqlite_export import (
                        export_to_sqlite, get_sqlite_tables,
                        query_sqlite, get_sqlite_db_size, SQLITE_PATH
                    )
                    SQLITE_OK = True
                except Exception:
                    SQLITE_OK = False

                if SQLITE_OK:
                    db_size  = get_sqlite_db_size()
                    db_tables= get_sqlite_tables()
                    n_tables = len(db_tables)
                    st.markdown(f"""
<div style='background:linear-gradient(135deg,rgba(45,27,78,0.6),rgba(88,28,135,0.4));
border:1px solid rgba(168,85,247,0.25);border-radius:14px;padding:16px 20px;
margin-bottom:20px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px'>
  <div>
    <div style='font-size:15px;font-weight:600;color:#d8b4fe'>🗄️ SQLite Database</div>
    <div style='font-size:12px;color:#9ca3af;margin-top:3px'>Permanent storage · <code style="color:#c084fc">data/pipeline_results.sqlite</code></div>
  </div>
  <div style='display:flex;gap:16px'>
    <div style='text-align:center'><div style='font-size:20px;font-weight:600;color:#d8b4fe'>{n_tables}</div><div style='font-size:11px;color:#9ca3af'>tables</div></div>
    <div style='text-align:center'><div style='font-size:20px;font-weight:600;color:#d8b4fe'>{db_size}</div><div style='font-size:11px;color:#9ca3af'>size</div></div>
  </div>
</div>""", unsafe_allow_html=True)

                for result in st.session_state.all_results:
                    sname=result.get("scenario_name","?")
                    df_h=safe_read_csv(result.get("healed_data_path",""))
                    df_r=safe_read_csv(result.get("removed_data_path",""))
                    qr=result.get("quality_report",{})
                    already_exported = st.session_state.sqlite_exports.get(sname, False)
                    st.markdown(f"#### 📁 `{sname}`")
                    c1,c2,c3=st.columns(3)
                    with c1:
                        st.markdown("<div style=\'background:linear-gradient(135deg,rgba(2,44,34,0.7),rgba(6,78,59,0.4));border:1px solid rgba(16,185,129,0.25);border-radius:14px;padding:18px;text-align:center\'><div style=\'font-size:28px\'>✅</div><div style=\'color:#34d399;font-weight:600;font-size:14px\'>Clean CSV</div></div>",unsafe_allow_html=True)
                        if df_h is not None:
                            st.caption(f"{len(df_h)} rows · {len(df_h.columns)} cols")
                            st.download_button("⬇️ Download clean CSV",data=df_h.to_csv(index=False).encode(),file_name=f"clean_{sname}.csv",mime="text/csv",use_container_width=True,key=f"dl_h_{sname}")
                    with c2:
                        cnt=len(df_r) if df_r is not None else 0
                        st.markdown(f"<div style=\'background:linear-gradient(135deg,rgba(76,5,25,0.7),rgba(127,29,29,0.4));border:1px solid rgba(239,68,68,0.25);border-radius:14px;padding:18px;text-align:center\'><div style=\'font-size:28px\'>🗑️</div><div style=\'color:#fca5a5;font-weight:600;font-size:14px\'>Removed ({cnt})</div></div>",unsafe_allow_html=True)
                        if df_r is not None and cnt>0:
                            st.download_button(f"⬇️ Download removed ({cnt})",data=df_r.to_csv(index=False).encode(),file_name=f"removed_{sname}.csv",mime="text/csv",use_container_width=True,key=f"dl_r_{sname}")
                        else: st.success("No rows removed")
                    with c3:
                        st.markdown("<div style=\'background:linear-gradient(135deg,rgba(30,37,53,0.8),rgba(20,28,45,0.5));border:1px solid rgba(59,130,246,0.25);border-radius:14px;padding:18px;text-align:center\'><div style=\'font-size:28px\'>📋</div><div style=\'color:#93c5fd;font-weight:600;font-size:14px\'>Quality Report</div></div>",unsafe_allow_html=True)
                        if qr:
                            st.download_button("⬇️ Download JSON",data=json.dumps(qr,indent=2,default=str).encode(),file_name=f"report_{sname}.json",mime="application/json",use_container_width=True,key=f"dl_q_{sname}")
                    st.divider()


# ─── TAB 7: INFRASTRUCTURE (merged MockDB + MCP + Architecture) ──
with tab_infra:
    infra_sub1, infra_sub2, infra_sub3, infra_sub4 = st.tabs(["🗄️ MockDB","🔌 MCP Server","🏗️ Architecture","📜 Run Log"])

    with infra_sub1:
        st.markdown("### 🗄️ MockDB — DuckDB Data Warehouse")
        st.markdown("All runs persisted to DuckDB (simulates Snowflake). Uses **read-only connections** in the UI — no lock conflicts.")

        if not DB_AVAILABLE:
            st.error("MockDB not available.")
        else:
            # ✅ KEY FIX: get_db() called HERE (inside tab), not at module level
            db_tab1,db_tab2,db_tab3,db_tab4,db_tab5=st.tabs([
                "📋 Pipeline Runs","🔍 Issues","🔧 Fixes","📊 Quality Scores","💻 SQL Console"
            ])
            with db_tab1:
                st.markdown("#### All pipeline runs")
                try:
                    df_runs=get_db().get_pipeline_runs(limit=50)
                    if df_runs.empty: st.info("No runs yet.")
                    else:
                        r1,r2,r3=st.columns(3)
                        r1.metric("Total runs",len(df_runs))
                        r2.metric("Successful",len(df_runs[df_runs["status"]=="SUCCESS"]) if "status" in df_runs else 0)
                        r3.metric("Avg quality",f"{round(float(df_runs['quality_score'].mean()),1)}/100" if "quality_score" in df_runs else "—")
                        st.dataframe(df_runs,use_container_width=True,hide_index=True)
                except Exception as e: st.error(f"DB error: {e}")

            with db_tab2:
                st.markdown("#### Issue registry")
                try:
                    df_iss=get_db().get_issue_summary()
                    if df_iss.empty: st.info("No issues recorded yet.")
                    else:
                        fig_i=px.bar(df_iss,x="issue_type",y="count",color="severity",title="Issues by type",height=280,color_discrete_map={"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#3b82f6"})
                        fig_i.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#8b95a8",margin=dict(l=0,r=0,t=40,b=50),xaxis=dict(showgrid=False,tickangle=-20),yaxis=dict(gridcolor="#1e2535"))
                        st.plotly_chart(fig_i,use_container_width=True)
                        st.dataframe(df_iss,use_container_width=True,hide_index=True)
                except Exception as e: st.error(f"DB error: {e}")

            with db_tab3:
                st.markdown("#### Fix registry")
                try:
                    df_fix=get_db().get_fix_success_rate()
                    if df_fix.empty: st.info("No fixes recorded yet.")
                    else:
                        fig_f=px.bar(df_fix,x="action",y="count",color="status",title="Fix actions by status",height=260,barmode="group",color_discrete_map={"SUCCESS":"#34d399","FAILED":"#ef4444"})
                        fig_f.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#8b95a8",margin=dict(l=0,r=0,t=40,b=50),xaxis=dict(showgrid=False,tickangle=-20),yaxis=dict(gridcolor="#1e2535"))
                        st.plotly_chart(fig_f,use_container_width=True)
                        st.dataframe(df_fix,use_container_width=True,hide_index=True)
                except Exception as e: st.error(f"DB error: {e}")

            with db_tab4:
                st.markdown("#### Data quality trend")
                try:
                    df_q=get_db().get_quality_trend()
                    if df_q.empty: st.info("No quality scores yet.")
                    else:
                        fig_q=px.line(df_q,x="recorded_at",y="overall_score",color="scenario_name",markers=True,title="Quality score over time",height=280,color_discrete_sequence=["#3b82f6","#8b5cf6","#10b981"])
                        fig_q.add_hline(y=85,line_dash="dash",line_color="#34d399",annotation_text="Target (85)")
                        fig_q.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#8b95a8",margin=dict(l=0,r=0,t=40,b=0),yaxis=dict(range=[0,105],gridcolor="#1e2535"),xaxis=dict(showgrid=False))
                        st.plotly_chart(fig_q,use_container_width=True)
                        st.dataframe(df_q,use_container_width=True,hide_index=True)
                except Exception as e: st.error(f"DB error: {e}")

            with db_tab5:
                st.markdown("#### 💻 SQL Console")
                st.markdown("""
    <div style='background:rgba(15,20,35,0.6);border:1px solid rgba(99,130,255,0.1);border-radius:10px;padding:12px 16px;margin-bottom:12px;font-size:12px'>
    <span style='color:#9ca3af;font-weight:600'>Available tables:</span>
    <code style='background:#0d1117;color:#34d399;padding:2px 7px;border-radius:4px;margin:2px'>pipeline_runs</code>
    <code style='background:#0d1117;color:#34d399;padding:2px 7px;border-radius:4px;margin:2px'>issue_registry</code>
    <code style='background:#0d1117;color:#34d399;padding:2px 7px;border-radius:4px;margin:2px'>fix_registry</code>
    <code style='background:#0d1117;color:#34d399;padding:2px 7px;border-radius:4px;margin:2px'>data_quality_scores</code>
    <code style='background:#0d1117;color:#34d399;padding:2px 7px;border-radius:4px;margin:2px'>raw_data_snapshots</code>
    </div>""", unsafe_allow_html=True)
                # Pre-built query selector
                quick_selected = st.selectbox("Quick queries", [
                    "Custom query...",
                    "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 10",
                    "SELECT issue_type, severity, COUNT(*) as cnt FROM issue_registry GROUP BY issue_type, severity ORDER BY cnt DESC",
                    "SELECT action, SUM(CASE WHEN status='SUCCESS' THEN 1 ELSE 0 END)*100.0/COUNT(*) as success_pct FROM fix_registry GROUP BY action",
                    "SELECT scenario_name, ROUND(AVG(overall_score),1) as avg_score, COUNT(*) as runs FROM data_quality_scores GROUP BY scenario_name",
                    "SELECT run_id, stage, row_count, null_count, created_at FROM raw_data_snapshots ORDER BY created_at DESC LIMIT 10",
                    "SELECT r.run_id, r.scenario_name, r.status, COUNT(i.issue_id) as issues FROM pipeline_runs r LEFT JOIN issue_registry i ON r.run_id=i.run_id GROUP BY r.run_id, r.scenario_name, r.status",
                ], key="quick_sel")
                default_q = "" if quick_selected == "Custom query..." else quick_selected
                sql_input = st.text_area("SQL Query", value=default_q or "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 10", height=110, key="sql_input_area")
                exec_col, dl_col = st.columns([1,1])
                with exec_col:
                    exec_btn = st.button("▶ Execute Query", type="primary", key="sql_exec", use_container_width=True)
                if exec_btn:
                    try:
                        df_sql = get_db().execute_sql(sql_input)
                        st.success(f"✅ {len(df_sql)} rows returned")
                        st.dataframe(df_sql, use_container_width=True, hide_index=True)
                        st.download_button(
                            "⬇️ Download result as CSV",
                            data=df_sql.to_csv(index=False).encode(),
                            file_name="query_result.csv", mime="text/csv",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Query failed: {e}")
                        st.caption("Tip: Check table names and column names above.")



    with infra_sub2:
        st.markdown("### 🔌 MCP Tool Server")
        st.markdown("Exposes pipeline capabilities as callable tools for any MCP-compatible client.")

        if not DB_AVAILABLE:
            st.error("MCP server not available.")
        else:
            mcp=get_mcp_server()
            mcp_t1,mcp_t2,mcp_t3=st.tabs(["🧰 Available Tools","🖥️ Interactive Console","📖 Integration Guide"])

            with mcp_t1:
                st.markdown("#### Available MCP Tools")
                for tool in TOOLS:
                    st.markdown(
                        f"<div class='mcp-card'><div class='mcp-tool-name'>🔧 {tool['name']}</div>"
                        f"<div class='mcp-tool-desc'>{tool['description']}</div>"
                        f"<div style='margin-top:8px;font-size:11px;color:#4b5563'>Params: "
                        +", ".join(f"<code style='background:#0d1117;padding:1px 5px;border-radius:3px;color:#a5d6a7'>{k}</code>"
                                   for k in tool["input_schema"].get("properties",{}).keys())
                        +"</div></div>",unsafe_allow_html=True)

            with mcp_t2:
                st.markdown("#### MCP Interactive Console")
                st.caption("Simulates how an external agent calls the MCP server.")
                tool_names=[t["name"] for t in TOOLS]
                selected_tool=st.selectbox("Select tool",tool_names)
                tool_def=next(t for t in TOOLS if t["name"]==selected_tool)
                props=tool_def["input_schema"].get("properties",{}); required=tool_def["input_schema"].get("required",[])
                args={}
                if props:
                    st.markdown("**Arguments:**")
                    for param,schema in props.items():
                        rl=" *(required)*" if param in required else " *(optional)*"
                        if schema.get("type")=="boolean":
                            args[param]=st.checkbox(f"{param}{rl}",value=schema.get("default",True))
                        elif schema.get("enum"):
                            args[param]=st.selectbox(f"{param}{rl}",schema["enum"],index=0)
                        elif schema.get("type")=="integer":
                            args[param]=st.number_input(f"{param}{rl}",value=int(schema.get("default",10)),step=1)
                        else:
                            dv={"run_pipeline_tool":{"scenario_name":"missing_values"},"query_database_tool":{"sql":"SELECT * FROM pipeline_runs LIMIT 5"},"validate_data_tool":{"csv_path":"data/scenario_missing.csv"}}.get(selected_tool,{}).get(param,schema.get("default",""))
                            args[param]=st.text_input(f"{param}{rl}",value=str(dv))
                if st.button("📡 Call Tool",type="primary"):
                    with st.spinner(f"Calling `{selected_tool}`..."):
                        res_mcp=mcp.call_tool(selected_tool,args)
                    if "error" in res_mcp: st.error(f"Error: {res_mcp['error']}")
                    else: st.success("✅ Success"); st.json(res_mcp)
                st.markdown("#### Raw JSON request")
                raw_req=st.text_area("JSON",value=json.dumps({"method":"tools/call","params":{"name":selected_tool,"arguments":args}},indent=2),height=140)
                if st.button("📨 Send Raw Request"):
                    try:
                        resp=mcp.handle_request(json.loads(raw_req)); st.json(resp)
                    except json.JSONDecodeError: st.error("Invalid JSON")

            with mcp_t3:
                st.markdown("#### Integration Guide")
                st.code("""
    from mcp_server.pipeline_mcp_server import get_mcp_server

    mcp = get_mcp_server()

    # Run pipeline
    result = mcp.call_tool("run_pipeline_tool", {"scenario_name": "missing_values"})

    # Query DB
    data = mcp.call_tool("query_database_tool", {"sql": "SELECT * FROM pipeline_runs LIMIT 5"})

    # GE Validation
    val = mcp.call_tool("validate_data_tool", {"csv_path": "data/scenario_missing.csv", "suite_type": "pre_healing"})
    print(f"{val['passed']}/{val['total']} passed")

    # Quality scores
    scores = mcp.call_tool("get_quality_scores_tool", {})
    print(f"Average: {scores['average']}")
    """, language="python")



    with infra_sub3:
        st.markdown("""
    <h2 style='font-size:22px;font-weight:700;margin-bottom:4px'>
      🏗️ System Architecture
    </h2>
    <p style='color:#6b7280;font-size:13px'>
      Multi-agent agentic DE automation with LangGraph, MCP, DuckDB, and Great Expectations
    </p>
    """, unsafe_allow_html=True)

        # Overall system architecture
        st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(15,20,35,0.8),rgba(10,14,26,0.6));
    border:1px solid rgba(99,130,255,0.15);border-radius:20px;padding:28px;margin-bottom:24px'>
      <div style='text-align:center;font-size:16px;font-weight:700;color:#e2e8f0;margin-bottom:20px'>
        🏗️ Agentic DE Automation — System Architecture
      </div>
      <div style='display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin-bottom:20px'>
        <div style='background:linear-gradient(135deg,#1e3a5f,#1a365d);border:1px solid rgba(59,130,246,0.3);border-radius:14px;padding:16px 20px;text-align:center;min-width:180px'>
          <div style='font-size:24px'>🔬</div>
          <div style='color:#93c5fd;font-weight:600;font-size:13px'>B1 — Ingestion Quality</div>
          <div style='color:#6b7280;font-size:11px;margin-top:4px'>5 LangGraph nodes</div>
          <div style='color:#4b5563;font-size:10px;margin-top:2px'>Profile → Rules → Validate → Heal → Report</div>
        </div>
        <div style='background:linear-gradient(135deg,#2e1065,#4c1d95);border:1px solid rgba(139,92,246,0.3);border-radius:14px;padding:16px 20px;text-align:center;min-width:180px'>
          <div style='font-size:24px'>🏛️</div>
          <div style='color:#c4b5fd;font-weight:600;font-size:13px'>B2 — Lineage & Governance</div>
          <div style='color:#6b7280;font-size:11px;margin-top:4px'>5 LangGraph nodes</div>
          <div style='color:#4b5563;font-size:10px;margin-top:2px'>SQL → Lineage → PII → Catalogue → GDPR</div>
        </div>
        <div style='background:linear-gradient(135deg,#022c22,#064e3b);border:1px solid rgba(16,185,129,0.3);border-radius:14px;padding:16px 20px;text-align:center;min-width:180px'>
          <div style='font-size:24px'>🔧</div>
          <div style='color:#6ee7b7;font-weight:600;font-size:13px'>B3 — Self-Healing Pipeline</div>
          <div style='color:#6b7280;font-size:11px;margin-top:4px'>5 LangGraph nodes</div>
          <div style='color:#4b5563;font-size:10px;margin-top:2px'>Detect → Classify → Decide → Heal → Log</div>
        </div>
      </div>
      <div style='text-align:center;color:#4b5563;font-size:18px;margin:8px 0'>⬇ ⬇ ⬇</div>
      <div style='display:flex;justify-content:center;gap:12px;flex-wrap:wrap'>
        <div style='background:rgba(30,37,53,0.8);border:1px solid rgba(99,130,255,0.15);border-radius:10px;padding:12px 18px;text-align:center'>
          <div style='font-size:13px;font-weight:600;color:#fbbf24'>🗄️ DuckDB MockDB</div>
          <div style='font-size:10px;color:#6b7280'>Simulates Snowflake DW</div>
        </div>
        <div style='background:rgba(30,37,53,0.8);border:1px solid rgba(99,130,255,0.15);border-radius:10px;padding:12px 18px;text-align:center'>
          <div style='font-size:13px;font-weight:600;color:#34d399'>✅ Great Expectations</div>
          <div style='font-size:10px;color:#6b7280'>Pre/Post validation suites</div>
        </div>
        <div style='background:rgba(30,37,53,0.8);border:1px solid rgba(99,130,255,0.15);border-radius:10px;padding:12px 18px;text-align:center'>
          <div style='font-size:13px;font-weight:600;color:#c084fc'>🔌 MCP Server</div>
          <div style='font-size:10px;color:#6b7280'>6 callable tools</div>
        </div>
        <div style='background:rgba(30,37,53,0.8);border:1px solid rgba(99,130,255,0.15);border-radius:10px;padding:12px 18px;text-align:center'>
          <div style='font-size:13px;font-weight:600;color:#60a5fa'>🧠 Gemini LLM</div>
          <div style='font-size:10px;color:#6b7280'>Classification + Decision</div>
        </div>
        <div style='background:rgba(30,37,53,0.8);border:1px solid rgba(99,130,255,0.15);border-radius:10px;padding:12px 18px;text-align:center'>
          <div style='font-size:13px;font-weight:600;color:#fb7185'>🗃️ SQLite</div>
          <div style='font-size:10px;color:#6b7280'>Permanent export storage</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

        # Agent detail cards
        st.markdown("#### Agent Node Details")
        arch_t1, arch_t2, arch_t3 = st.tabs(["🔬 B1 Nodes","🏛️ B2 Nodes","🔧 B3 Nodes"])
        with arch_t1:
            b1_nodes = [
                ("1. Profiler","profiler_node","Stats per column: dtype, nulls, IQR outliers, PII heuristics","📊","#3b82f6"),
                ("2. Rule Generator","rule_generator_node","Gemini LLM generates NOT_NULL, RANGE, OUTLIER, PII rules from profile","📋","#8b5cf6"),
                ("3. Validator","validator_node","Checks every row against generated rules, flags violations","✅","#f59e0b"),
                ("4. Healer","healer_node","Auto-fills nulls, clips ranges, masks PII, removes duplicates","🔨","#10b981"),
                ("5. Report","b1_report_node","Generates validation score and JSON report","📄","#ef4444"),
            ]
            for name,func,desc,icon,color in b1_nodes:
                st.markdown(f"<div style='background:linear-gradient(135deg,rgba(30,37,53,0.8),rgba(20,28,45,0.5));border-left:3px solid {color};border-radius:0 12px 12px 0;padding:12px 18px;margin-bottom:8px'><div style='font-size:14px;font-weight:600;color:#e2e8f0'>{icon} {name}</div><div style='font-size:11px;color:#6b7280;font-family:monospace'>{func}</div><div style='font-size:12px;color:#8b95a8;margin-top:4px'>{desc}</div></div>", unsafe_allow_html=True)
        with arch_t2:
            b2_nodes = [
                ("1. SQL Parser","sql_parser_node","Parses SQL or auto-generates from CSV schema; extracts tables, CTEs, joins","🔍","#3b82f6"),
                ("2. Lineage Extractor","lineage_extractor_node","Gemini LLM builds full lineage graph: sources → transforms → sinks","🕸️","#8b5cf6"),
                ("3. PII Tagger","pii_tagger_node","Detects 11 PII types, assigns GDPR articles, applies 9 masking strategies","🔴","#ef4444"),
                ("4. Catalogue Enricher","catalogue_enricher_node","Gemini LLM writes business descriptions, stewardship, quality SLAs","📚","#10b981"),
                ("5. Governance Report","governance_report_node","GDPR compliance scoring, policy recommendations, full audit trail","📋","#f59e0b"),
            ]
            for name,func,desc,icon,color in b2_nodes:
                st.markdown(f"<div style='background:linear-gradient(135deg,rgba(30,37,53,0.8),rgba(20,28,45,0.5));border-left:3px solid {color};border-radius:0 12px 12px 0;padding:12px 18px;margin-bottom:8px'><div style='font-size:14px;font-weight:600;color:#e2e8f0'>{icon} {name}</div><div style='font-size:11px;color:#6b7280;font-family:monospace'>{func}</div><div style='font-size:12px;color:#8b95a8;margin-top:4px'>{desc}</div></div>", unsafe_allow_html=True)
        with arch_t3:
            b3_nodes = [
                ("1. Detection","detection_agent","Scans for nulls, schema mismatches, dtype errors, IQR outliers","🔍","#3b82f6"),
                ("2. Classification","classification_agent","Gemini LLM classifies issues into DATA_QUALITY / SCHEMA / ANOMALY / SYSTEM_FAILURE","🏷️","#8b5cf6"),
                ("3. Decision","decision_agent","Gemini LLM decides optimal fix: FILL_MEDIAN, CLIP_OUTLIERS, DROP_COLUMN, etc.","🧠","#f59e0b"),
                ("4. Healing","healing_agent","Executes 11 fix actions, runs GE pre/post, writes MockDB snapshots","🔨","#10b981"),
                ("5. Logging","logging_agent","Structured summary, JSON report, mock alerts for HIGH severity","📋","#ef4444"),
            ]
            for name,func,desc,icon,color in b3_nodes:
                st.markdown(f"<div style='background:linear-gradient(135deg,rgba(30,37,53,0.8),rgba(20,28,45,0.5));border-left:3px solid {color};border-radius:0 12px 12px 0;padding:12px 18px;margin-bottom:8px'><div style='font-size:14px;font-weight:600;color:#e2e8f0'>{icon} {name}</div><div style='font-size:11px;color:#6b7280;font-family:monospace'>{func}</div><div style='font-size:12px;color:#8b95a8;margin-top:4px'>{desc}</div></div>", unsafe_allow_html=True)

        # Tech stack summary
        st.markdown("#### Tech Stack")
        tc1,tc2,tc3 = st.columns(3)
        with tc1:
            st.markdown("""**🔗 Orchestration**
    - LangGraph (15 total nodes)
    - 3 separate StateGraph pipelines
    - Conditional routing (B3)
    - TypedDict state schemas""")
        with tc2:
            st.markdown("""**🧠 LLM Integration**
    - Gemini gemini-2.5-flash (Google)
    - Classification + Decision + Rules
    - Lineage + Catalogue enrichment
    - Rule-based fallback if no API key""")
        with tc3:
            st.markdown("""**🗄️ Data Layer**
    - DuckDB (MockDB — Snowflake sim)
    - SQLite (permanent export)
    - Great Expectations validation
    - MCP Server (6 tools)""")



    with infra_sub4:
        st.markdown("""
    <h2 style='font-size:22px;font-weight:700;margin-bottom:4px'>
      📜 Autonomous Run Log
    </h2>
    <p style='color:#6b7280;font-size:13px'>
      Full JSONL log of all agent actions — timestamped, color-coded, filterable
    </p>
    """, unsafe_allow_html=True)

        log_file = os.path.join(ROOT, "logs", "pipeline_run.jsonl")
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            try:
                with open(log_file, "r") as f:
                    log_entries = [json.loads(line) for line in f if line.strip()]
            except Exception:
                log_entries = []

            if log_entries:
                # Filters
                fc1,fc2,fc3 = st.columns(3)
                with fc1:
                    all_components = sorted(set(e.get("component","?") for e in log_entries))
                    sel_comp = st.multiselect("Filter by component", all_components, default=all_components, key="log_comp_filter")
                with fc2:
                    all_levels = sorted(set(e.get("level","?") for e in log_entries))
                    sel_levels = st.multiselect("Filter by level", all_levels, default=all_levels, key="log_level_filter")
                with fc3:
                    max_entries = st.slider("Max entries", 10, min(500,len(log_entries)), min(100,len(log_entries)), key="log_max")

                filtered = [e for e in log_entries
                            if e.get("component","?") in sel_comp and e.get("level","?") in sel_levels]
                filtered = filtered[-max_entries:]

                # Stats
                s1,s2,s3,s4 = st.columns(4)
                s1.metric("Total entries", len(log_entries))
                s2.metric("Shown", len(filtered))
                s3.metric("Components", len(all_components))
                s4.metric("Errors", sum(1 for e in log_entries if e.get("level")=="ERROR"))

                # Render log
                log_html_parts = []
                level_css = {"INFO":"log-info","WARN":"log-warn","SUCCESS":"log-ok","ERROR":"log-err","ALERT":"log-alert"}
                for entry in reversed(filtered):
                    ts = entry.get("timestamp","")
                    lvl = entry.get("level","INFO")
                    comp = entry.get("component","?")
                    msg = entry.get("message","").replace("<","&lt;").replace(">","&gt;")
                    css = level_css.get(lvl, "log-info")
                    log_html_parts.append(
                        f'<div class="{css}"><span style="color:#4b5563;font-size:10px">{ts}</span> '
                        f'<span style="font-weight:600">[{lvl}]</span> '
                        f'<span style="color:#6b7280">[{comp}]</span> {msg}</div>'
                    )
                st.markdown(f'<div class="log-block">{"".join(log_html_parts)}</div>', unsafe_allow_html=True)

                # Download
                st.download_button("⬇️ Download full log (JSONL)",
                    data=open(log_file,"rb").read(),
                    file_name="pipeline_run.jsonl", mime="application/jsonl",
                    use_container_width=True)
            else:
                st.info("Log file exists but is empty. Run a pipeline to generate logs.")
        else:
            st.info("No log file found. Run any pipeline (B1, B2, or B3) to generate autonomous run logs.")
# ─── TAB 9: B1 INGESTION QUALITY ──────────────────────────────
with tab_b1:
    st.markdown("""
<h2 style='font-size:22px;font-weight:700;margin-bottom:4px'>
  🔬 B1 · Ingestion Quality Agent
</h2>
<p style='color:#6b7280;font-size:13px'>
  <b style='color:#60a5fa'>Profile</b> → 
  <b style='color:#a78bfa'>Generate Rules (LLM)</b> → 
  <b style='color:#fbbf24'>Validate</b> → 
  <b style='color:#34d399'>Self-Heal</b>
</p>
""", unsafe_allow_html=True)

    # Init B1 session state
    if "b1_results" not in st.session_state: st.session_state.b1_results = {}
    if "b1_ran" not in st.session_state: st.session_state.b1_ran = False

    # Controls
    b1c1, b1c2 = st.columns([2,1])
    with b1c1:
        if st.session_state.uploaded_df is not None:
            st.markdown(f"**📁 Dataset:** `{st.session_state.upload_name}` ({len(st.session_state.uploaded_df)} rows)")
            b1_scenario = "custom_upload"
        else:
            st.warning("⬆️ Upload a CSV in the sidebar first.")
            b1_scenario = None
    with b1c2:
        b1_run_btn = st.button("🔬 Run B1 Agent", type="primary",
                               use_container_width=True, key="b1_run",
                               disabled=b1_scenario is None)

    if b1_run_btn and b1_scenario:
        data_path = st.session_state.get("uploaded_path", "")
        with st.status(f"🔬 Running B1 on **{b1_scenario}**...", expanded=True) as b1_status:
            st.write("📊 Step 1/4 — Profiling dataset (statistics per column)...")
            st.write("📋 Step 2/4 — Generating quality rules via Gemini LLM...")
            st.write("✅ Step 3/4 — Validating data against all rules...")
            st.write("🔨 Step 4/4 — Auto-healing violations found...")
            try:
                import io as _io, contextlib, re as _re
                from agents.b1_ingestion_quality_agent import run_b1_pipeline
                buf = _io.StringIO()
                ansi = _re.compile(r"\x1b\[[0-9;]*m")
                with contextlib.redirect_stdout(buf):
                    b1_result = run_b1_pipeline(b1_scenario, data_path)
                b1_logs = [ansi.sub("",l).strip() for l in buf.getvalue().splitlines() if l.strip()]
                b1_status.update(label=f"✅ B1 complete | Score: {b1_result.get('validation_score',0)}%",
                                 state="complete", expanded=False)
                st.session_state.b1_results[b1_scenario] = b1_result
                st.session_state.b1_ran = True
            except Exception as e:
                import traceback
                st.error(f"B1 error: {e}"); st.code(traceback.format_exc())
                b1_status.update(label="❌ B1 failed", state="error")
                b1_result = {}; b1_logs = []

        if b1_result:
            # Metrics
            score = b1_result.get("validation_score",0)
            sc_col = "#34d399" if score>=80 else "#fbbf24" if score>=60 else "#f87171"
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Status",     b1_result.get("final_status","?"))
            m2.metric("Rules gen.", len(b1_result.get("quality_rules",[])))
            m3.metric("Violations", len(b1_result.get("violations",[])))
            m4.metric("Heals",      len(b1_result.get("heals_applied",[])))
            m5.metric("Score",      f"{score}%")

            # Quality score gauge
            import plotly.graph_objects as go
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                gauge={"axis":{"range":[0,100]},"bar":{"color":sc_col},
                       "steps":[{"range":[0,60],"color":"rgba(239,68,68,0.15)"},
                                {"range":[60,80],"color":"rgba(251,191,36,0.15)"},
                                {"range":[80,100],"color":"rgba(52,211,153,0.15)"}],
                       "threshold":{"line":{"color":"#34d399","width":3},"thickness":0.8,"value":85}},
                title={"text":"Validation Score","font":{"size":14,"color":"#8b95a8"}}))
            fig_gauge.update_layout(height=200,margin=dict(l=20,r=20,t=40,b=10),
                                    paper_bgcolor="rgba(0,0,0,0)",font_color="#8b95a8")
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Sub-tabs
            b1t1,b1t2,b1t3,b1t4,b1t5 = st.tabs([
                "📊 Profile","📋 Quality Rules","⚠️ Violations",
                "🔨 Heals Applied","🔄 Before / After"
            ])

            with b1t1:
                st.markdown("#### Statistical Profile")
                profile = b1_result.get("profile",{})
                table_meta = profile.get("__table__",{})
                if table_meta:
                    pm1,pm2,pm3,pm4 = st.columns(4)
                    pm1.metric("Rows",       table_meta.get("rows",0))
                    pm2.metric("Columns",    table_meta.get("cols",0))
                    pm3.metric("Total nulls",table_meta.get("total_nulls",0))
                    pm4.metric("Duplicates", table_meta.get("duplicate_rows",0))

                profile_rows = []
                for col, p in profile.items():
                    if col == "__table__": continue
                    profile_rows.append({
                        "Column":     col,
                        "Dtype":      p.get("dtype",""),
                        "Nulls":      f"{p.get('null_count',0)} ({p.get('null_pct',0)}%)",
                        "Unique":     f"{p.get('unique',0)} ({p.get('unique_pct',0)}%)",
                        "Min":        p.get("min","—"),
                        "Max":        p.get("max","—"),
                        "Mean":       p.get("mean","—"),
                        "Outlier%":   p.get("outlier_pct","—"),
                        "PII":        p.get("contains_pii","NONE"),
                    })
                if profile_rows:
                    df_prof = pd.DataFrame(profile_rows)
                    st.dataframe(df_prof, use_container_width=True, hide_index=True, height=260)

            with b1t2:
                st.markdown("#### LLM-Generated Quality Rules")
                rules = b1_result.get("quality_rules",[])
                if rules:
                    st.info(f"Gemini generated {len(rules)} rules from the statistical profile.")
                    df_rules = pd.DataFrame([{
                        "ID":          r.get("rule_id",""),
                        "Column":      r.get("column",""),
                        "Rule Type":   r.get("rule_type",""),
                        "Severity":    r.get("severity",""),
                        "Description": r.get("description",""),
                        "Params":      str(r.get("params",{})),
                    } for r in rules])
                    st.dataframe(df_rules, use_container_width=True, hide_index=True)

                    import plotly.express as px
                    rt_counts = df_rules["Rule Type"].value_counts().reset_index()
                    rt_counts.columns = ["Rule Type","Count"]
                    fig_rt = px.bar(rt_counts, x="Count", y="Rule Type", orientation="h",
                                   title="Rules by type", height=240,
                                   color="Rule Type",
                                   color_discrete_sequence=["#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444"])
                    fig_rt.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                        font_color="#8b95a8",margin=dict(l=0,r=0,t=40,b=0),
                                        showlegend=False,xaxis=dict(gridcolor="#1e2535"),yaxis=dict(showgrid=False))
                    st.plotly_chart(fig_rt, use_container_width=True)

            with b1t3:
                violations = b1_result.get("violations",[])
                if not violations:
                    st.success("✅ No violations found — data passed all rules!")
                else:
                    # Severity breakdown chart
                    import plotly.express as px
                    sev_data = pd.DataFrame(violations)
                    if "severity" in sev_data.columns:
                        sev_counts = sev_data["severity"].value_counts().reset_index()
                        sev_counts.columns = ["Severity","Count"]
                        fig_sev = px.pie(sev_counts, values="Count", names="Severity", hole=0.5, height=200,
                                        color="Severity", color_discrete_map={"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#3b82f6"})
                        fig_sev.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#8b95a8",
                                             margin=dict(l=0,r=0,t=20,b=0))
                        fig_sev.update_traces(textinfo="percent+label",textfont_size=10)
                        st.plotly_chart(fig_sev, use_container_width=True)

                    df_viol = pd.DataFrame([{
                        "Rule ID":       v.get("rule_id",""),
                        "Column":        v.get("column",""),
                        "Violation":     v.get("violation_type",""),
                        "Rows Affected": v.get("rows_affected",0),
                        "Severity":      v.get("severity",""),
                        "Detail":        v.get("detail",""),
                    } for v in violations])
                    st.dataframe(df_viol, use_container_width=True, hide_index=True)

            with b1t4:
                heals = b1_result.get("heals_applied",[])
                if not heals:
                    st.info("No heals applied.")
                else:
                    # Success/fail breakdown
                    ok = sum(1 for h in heals if h.get("status")=="SUCCESS")
                    fail = len(heals) - ok
                    hc1,hc2,hc3 = st.columns(3)
                    hc1.metric("Total Heals", len(heals))
                    hc2.metric("✅ Succeeded", ok)
                    hc3.metric("❌ Failed", fail)

                    df_heals = pd.DataFrame([{
                        "Column": h.get("column",""),
                        "Action": h.get("action",""),
                        "Result": h.get("result",""),
                        "Status": h.get("status",""),
                    } for h in heals])
                    st.dataframe(df_heals, use_container_width=True, hide_index=True)
                    healed_path = b1_result.get("healed_data_path","")
                    if healed_path and os.path.exists(healed_path):
                        df_h = pd.read_csv(healed_path)
                        st.success(f"✅ B1 Healed data: `{healed_path}` — {len(df_h)} rows")
                        st.dataframe(df_h.head(10), use_container_width=True, height=220, hide_index=True)
                        st.download_button("⬇️ Download B1 healed CSV",
                            data=df_h.to_csv(index=False).encode(),
                            file_name=f"b1_healed_{b1_scenario}.csv", mime="text/csv")

            # ── Before / After tab ───────────────────────────
            with b1t5:
                st.markdown("#### 🔄 Before / After Comparison")
                data_path = st.session_state.get("uploaded_path", "")
                healed_path = b1_result.get("healed_data_path","")
                if data_path and os.path.exists(data_path) and healed_path and os.path.exists(healed_path):
                    df_before = pd.read_csv(data_path)
                    df_after  = pd.read_csv(healed_path)

                    # Stats comparison
                    bc1,bc2 = st.columns(2)
                    with bc1:
                        st.markdown("##### 🔴 Before Healing")
                        bm1,bm2,bm3 = st.columns(3)
                        bm1.metric("Rows", len(df_before))
                        bm2.metric("Nulls", int(df_before.isna().sum().sum()))
                        bm3.metric("Columns", len(df_before.columns))
                        st.dataframe(df_before.head(10), use_container_width=True, height=220, hide_index=True)
                    with bc2:
                        st.markdown("##### ✅ After Healing")
                        am1,am2,am3 = st.columns(3)
                        am1.metric("Rows", len(df_after))
                        am2.metric("Nulls", int(df_after.isna().sum().sum()),
                                   delta=f"{int(df_after.isna().sum().sum()) - int(df_before.isna().sum().sum())}",
                                   delta_color="inverse")
                        am3.metric("Columns", len(df_after.columns))
                        st.dataframe(df_after.head(10), use_container_width=True, height=220, hide_index=True)

                    # Null heatmap comparison
                    if df_before.isna().sum().sum() > 0:
                        st.markdown("##### Null Distribution — Before vs After")
                        import plotly.express as px
                        nc1,nc2 = st.columns(2)
                        with nc1:
                            fig_nb = px.imshow(df_before.isna().astype(int).T,
                                             color_continuous_scale=["#1e2535","#ef4444"],height=140,
                                             title="Before")
                            fig_nb.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#8b95a8",
                                                margin=dict(l=0,r=0,t=30,b=0),coloraxis_showscale=False)
                            fig_nb.update_xaxes(showticklabels=False)
                            st.plotly_chart(fig_nb, use_container_width=True)
                        with nc2:
                            fig_na = px.imshow(df_after.isna().astype(int).T,
                                             color_continuous_scale=["#1e2535","#34d399"],height=140,
                                             title="After")
                            fig_na.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#8b95a8",
                                                margin=dict(l=0,r=0,t=30,b=0),coloraxis_showscale=False)
                            fig_na.update_xaxes(showticklabels=False)
                            st.plotly_chart(fig_na, use_container_width=True)
                else:
                    st.info("Run B1 to see before/after comparison.")

            # Logs
            if b1_logs:
                with st.expander("📋 B1 Logs"):
                    st.markdown(
                        f'<div class="log-block">{"".join(log_html(l) for l in b1_logs)}</div>',
                        unsafe_allow_html=True)
    else:
        st.info("👈 Select a dataset and click **🔬 Run B1 Agent**.")
        st.markdown("""
#### B1 Pipeline Steps:
| Step | Agent Node | What it does |
|------|-----------|--------------|
| 1 | **Profiler** | Statistical analysis of every column (min, max, nulls, outliers, PII) |
| 2 | **Rule Generator** | Gemini LLM generates quality rules from the profile |
| 3 | **Validator** | Checks data against every rule — flags violations |
| 4 | **Self-Healer** | Auto-fixes violations (fill nulls, clip outliers, mask PII) |
| 5 | **Reporter** | Generates final quality report with score |
""")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("""**🔴 PII Detection & Masking**
- EMAIL → `us***@domain.com`
- PHONE → `***-***-7890`
- NAME → `J***`
- SSN → `***-**-6789`
- Generic → Hash masking
- GDPR Art.4(1), Art.25""")
        with col_b:
            st.markdown("""**📋 Quality Rules**
- NOT_NULL — null checks
- RANGE_CHECK — bounds
- OUTLIER — IQR-based
- PII_DETECTED — mask
- DUPLICATE — dedup
- TYPE_CHECK — dtypes""")
        with col_c:
            st.markdown("""**🔨 Auto-Healing**
- FILL_MEDIAN — numeric
- FILL_MODE — categorical
- CLIP_RANGE — bounds
- CLIP_IQR — outliers
- MASK_PII — protection
- REMOVE_DUPLICATES""")


# ─── TAB 10: B2 LINEAGE & GOVERNANCE ─────────────────────────
with tab_b2:
    st.markdown("""
<h2 style='font-size:22px;font-weight:700;margin-bottom:4px'>
  🏛️ B2 · Lineage &amp; Governance Agent
</h2>
<p style='color:#6b7280;font-size:13px'>
  <b style='color:#60a5fa'>SQL Parse</b> → 
  <b style='color:#a78bfa'>Extract Lineage (LLM)</b> → 
  <b style='color:#ef4444'>Tag PII</b> → 
  <b style='color:#34d399'>Enrich Catalogue (LLM)</b> → 
  <b style='color:#fbbf24'>Governance Report</b>
</p>
""", unsafe_allow_html=True)

    if "b2_results" not in st.session_state: st.session_state.b2_results = {}
    if "b2_ran" not in st.session_state: st.session_state.b2_ran = False

    b2c1, b2c2 = st.columns([2,1])
    with b2c1:
        if st.session_state.uploaded_df is not None:
            st.markdown(f"**📁 Dataset:** `{st.session_state.upload_name}` ({len(st.session_state.uploaded_df)} rows)")
            b2_scenario = "custom_upload"
        else:
            st.warning("⬆️ Upload a CSV in the sidebar first.")
            b2_scenario = None
    with b2c2:
        b2_run_btn = st.button("🏛️ Run B2 Agent", type="primary",
                               use_container_width=True, key="b2_run",
                               disabled=b2_scenario is None)

    # Optional custom SQL
    with st.expander("⚙️ Custom SQL (optional — auto-generated if left blank)"):
        custom_sql = st.text_area(
            "SQL Query for lineage extraction",
            placeholder="SELECT u.user_id, u.email, o.order_id\nFROM users u\nJOIN orders o ON u.user_id = o.user_id\nWHERE u.country = \'IN\'",
            height=120, key="b2_sql_input"
        )


    if b2_run_btn and b2_scenario:
        data_path = st.session_state.get("uploaded_path", "")
        with st.status(f"🏛️ Running B2 on **{b2_scenario}**...", expanded=True) as b2_status:
            st.write("🔍 Step 1/5 — Parsing SQL / extracting schema structure...")
            st.write("🕸️ Step 2/5 — Extracting data lineage graph via LLM...")
            st.write("🔴 Step 3/5 — Scanning and tagging PII columns...")
            st.write("📚 Step 4/5 — Enriching data catalogue via LLM...")
            st.write("📋 Step 5/5 — Generating governance report + GDPR compliance...")
            try:
                import io as _io, contextlib, re as _re
                from agents.b2_lineage_governance_agent import run_b2_pipeline
                buf = _io.StringIO()
                ansi = _re.compile(r"\x1b\[[0-9;]*m")
                with contextlib.redirect_stdout(buf):
                    b2_result = run_b2_pipeline(
                        b2_scenario, data_path,
                        sql_query=custom_sql or ""
                    )
                b2_logs = [ansi.sub("",l).strip() for l in buf.getvalue().splitlines() if l.strip()]
                gdpr_score = b2_result.get("governance_report",{}).get("gdpr_compliance",{}).get("score",0)
                b2_status.update(
                    label=f"✅ B2 complete | GDPR: {gdpr_score}% | PII: {len(b2_result.get('pii_tags',[]))} cols",
                    state="complete", expanded=False
                )
                st.session_state.b2_results[b2_scenario] = b2_result
                st.session_state.b2_ran = True
            except Exception as e:
                import traceback
                st.error(f"B2 error: {e}"); st.code(traceback.format_exc())
                b2_status.update(label="❌ B2 failed", state="error")
                b2_result = {}; b2_logs = []

        if b2_result:
            gr       = b2_result.get("governance_report",{})
            pii_tags = b2_result.get("pii_tags",[])
            lineage  = b2_result.get("lineage_graph",{})
            catalogue= b2_result.get("data_catalogue",[])
            gdpr     = gr.get("gdpr_compliance",{})

            # Top metrics
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("GDPR Score",     f"{gdpr.get('score',0)}%")
            m2.metric("GDPR Status",    gdpr.get("status","?"))
            m3.metric("PII Columns",    len(pii_tags))
            m4.metric("Lineage Nodes",  len(lineage.get("nodes",[])))
            m5.metric("Catalogue Cols", len(catalogue))

            # Sub-tabs
            b2t1,b2t2,b2t3,b2t4,b2t5 = st.tabs([
                "🕸️ Data Lineage","🔴 PII Tags","📚 Data Catalogue",
                "📋 GDPR Compliance","🔒 Policy Recommendations"
            ])

            # ── Lineage ───────────────────────────────────────
            with b2t1:
                st.markdown("#### Data Lineage Graph")
                st.info(f"**Lineage path:** {lineage.get('lineage_path','N/A')}")

                nodes = lineage.get("nodes",[])
                edges = lineage.get("edges",[])
                lc1,lc2 = st.columns(2)
                with lc1:
                    st.markdown("**Sources**")
                    for src in lineage.get("sources",[]):
                        st.markdown(f"<span style='background:#1e3a5f;color:#93c5fd;padding:3px 10px;border-radius:6px;font-size:12px'>📥 {src}</span>", unsafe_allow_html=True)
                    st.markdown("**Sinks**")
                    for sink in lineage.get("sinks",[]):
                        st.markdown(f"<span style='background:#022c22;color:#6ee7b7;padding:3px 10px;border-radius:6px;font-size:12px'>📤 {sink}</span>", unsafe_allow_html=True)
                with lc2:
                    st.markdown("**Transformations**")
                    for t in lineage.get("transformations",[])[:5]:
                        if t: st.markdown(f"<span style='background:#451a03;color:#fcd34d;padding:3px 10px;border-radius:6px;font-size:12px'>⚙️ {t}</span>", unsafe_allow_html=True)

                if nodes:
                    st.markdown("**All Nodes**")
                    df_nodes = pd.DataFrame([{
                        "ID": n.get("id",""), "Name": n.get("name",""),
                        "Type": n.get("type",""), "System": n.get("system",""),
                        "Description": n.get("description","")[:60],
                    } for n in nodes])
                    st.dataframe(df_nodes, use_container_width=True, hide_index=True)

                if edges:
                    st.markdown("**Lineage Edges (transformations)**")
                    df_edges = pd.DataFrame([{
                        "From": e.get("from",""), "To": e.get("to",""),
                        "Operation": e.get("operation",""),
                        "Transformation": str(e.get("transformation",""))[:60],
                    } for e in edges])
                    st.dataframe(df_edges, use_container_width=True, hide_index=True)

                # Download lineage JSON
                st.download_button("⬇️ Download lineage graph JSON",
                    data=json.dumps(lineage, indent=2, default=str).encode(),
                    file_name=f"lineage_{b2_scenario}.json", mime="application/json")

            # ── PII Tags ──────────────────────────────────────
            with b2t2:
                st.markdown("#### PII Column Tags & Masking")
                if not pii_tags:
                    st.success("✅ No PII detected in this dataset.")
                else:
                    # Sensitivity breakdown
                    import plotly.express as px
                    sens_counts = pd.Series([t["sensitivity"] for t in pii_tags]).value_counts()
                    pc1,pc2 = st.columns([1,2])
                    with pc1:
                        fig_pii = px.pie(values=sens_counts.values, names=sens_counts.index,
                                        title="PII by sensitivity", hole=0.55, height=220,
                                        color=sens_counts.index,
                                        color_discrete_map={"CRITICAL":"#7c3aed","HIGH":"#ef4444",
                                                            "MEDIUM":"#f59e0b","LOW":"#3b82f6"})
                        fig_pii.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#8b95a8",
                                             margin=dict(l=0,r=0,t=30,b=0))
                        fig_pii.update_traces(textinfo="percent+label",textfont_size=10)
                        st.plotly_chart(fig_pii, use_container_width=True)
                    with pc2:
                        df_pii = pd.DataFrame([{
                            "Column":           t.get("column",""),
                            "PII Type":         t.get("pii_type",""),
                            "Sensitivity":      t.get("sensitivity",""),
                            "Masking Strategy": t.get("masking_strategy",""),
                            "GDPR Article":     t.get("gdpr_article",""),
                            "Retention":        t.get("retention_policy",""),
                            "Access Level":     t.get("access_level",""),
                        } for t in pii_tags])
                        st.dataframe(df_pii, use_container_width=True, hide_index=True, height=220)

                # Show masked dataset
                masked_path = b2_result.get("masked_data_path","")
                if masked_path and os.path.exists(masked_path):
                    df_masked = pd.read_csv(masked_path)
                    st.markdown("#### Masked Dataset Preview")
                    st.caption("PII values have been masked using the strategies above.")
                    orig_path = st.session_state.get("uploaded_path", "")
                    oc,mc = st.columns(2)
                    with oc:
                        st.caption("🔴 Original (with PII)")
                        if orig_path and os.path.exists(orig_path):
                            df_orig = pd.read_csv(orig_path)
                            st.dataframe(df_orig.head(8), use_container_width=True, height=220, hide_index=True)
                    with mc:
                        st.caption("✅ Masked (PII protected)")
                        st.dataframe(df_masked.head(8), use_container_width=True, height=220, hide_index=True)
                    st.download_button("⬇️ Download masked dataset",
                        data=df_masked.to_csv(index=False).encode(),
                        file_name=f"b2_masked_{b2_scenario}.csv", mime="text/csv")

            # ── Data Catalogue ────────────────────────────────
            with b2t3:
                st.markdown("#### Enterprise Data Catalogue")
                st.caption("Enriched by Gemini LLM — business descriptions, stewardship, GDPR tags")
                if catalogue:
                    df_cat = pd.DataFrame([{
                        "Column":       e.get("column",""),
                        "Business Term":e.get("business_term",""),
                        "Domain":       e.get("domain",""),
                        "Data Steward": e.get("data_steward",""),
                        "Description":  e.get("business_description","")[:70],
                        "Is PII":       "🔴 YES" if e.get("is_pii") else "✅ No",
                        "Classification":e.get("classification",""),
                        "Quality SLA":  e.get("quality_sla",""),
                    } for e in catalogue])
                    st.dataframe(df_cat, use_container_width=True, hide_index=True, height=300)

                    # Domain breakdown
                    domain_counts = df_cat["Domain"].value_counts().reset_index()
                    domain_counts.columns = ["Domain","Count"]
                    import plotly.express as px
                    fig_dom = px.pie(domain_counts, values="Count", names="Domain",
                                   title="Columns by domain", hole=0.5, height=220,
                                   color_discrete_sequence=["#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444"])
                    fig_dom.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#8b95a8",
                                         margin=dict(l=0,r=0,t=30,b=0))
                    fig_dom.update_traces(textinfo="percent+label",textfont_size=10)
                    st.plotly_chart(fig_dom, use_container_width=True)

                    st.download_button("⬇️ Download data catalogue CSV",
                        data=df_cat.to_csv(index=False).encode(),
                        file_name=f"b2_catalogue_{b2_scenario}.csv", mime="text/csv")

            # ── GDPR Compliance ───────────────────────────────
            with b2t4:
                st.markdown("#### GDPR Compliance Report")
                gdpr_score = gdpr.get("score",0)
                gdpr_status= gdpr.get("status","UNKNOWN")
                sc_color   = "#34d399" if gdpr_score>=80 else "#fbbf24" if gdpr_score>=50 else "#f87171"

                st.markdown(f"""
<div style='background:linear-gradient(135deg,rgba(30,37,53,0.8),rgba(20,28,45,0.5));
border:1px solid rgba(99,130,255,0.15);border-radius:16px;padding:20px;margin-bottom:16px'>
  <div style='display:flex;align-items:center;gap:20px'>
    <div style='text-align:center'>
      <div style='font-size:52px;font-weight:700;color:{sc_color}'>{gdpr_score}%</div>
      <div style='font-size:13px;color:#8b95a8'>GDPR Compliance Score</div>
    </div>
    <div>
      <div style='font-size:16px;font-weight:600;color:#e2e8f0'>{gdpr_status}</div>
      <div style='font-size:12px;color:#8b95a8;margin-top:4px'>{gr.get("executive_summary","")[:150]}...</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

                # GDPR checks
                checks = gdpr.get("checks",{})
                for check_name, passed in checks.items():
                    icon = "✅" if passed else "❌"
                    color = "#34d399" if passed else "#f87171"
                    st.markdown(
                        f"<div style='padding:8px 12px;margin:4px 0;border-radius:8px;"
                        f"background:rgba(30,37,53,0.5);border:1px solid rgba(99,130,255,0.08);"
                        f"font-size:13px;color:{color}'>"
                        f"{icon} {check_name}</div>",
                        unsafe_allow_html=True
                    )

                # Download full report
                st.download_button("⬇️ Download GDPR governance report",
                    data=json.dumps(gr, indent=2, default=str).encode(),
                    file_name=f"b2_governance_{b2_scenario}.json",
                    mime="application/json")

            # ── Policy Recommendations ────────────────────────
            with b2t5:
                st.markdown("#### Policy Recommendations")
                recommendations = b2_result.get("policy_recommendations",[])
                for rec in recommendations:
                    priority = rec.get("priority","MEDIUM")
                    bg_map = {"CRITICAL":"rgba(76,5,25,0.7)","HIGH":"rgba(69,26,3,0.7)",
                              "MEDIUM":"rgba(30,37,53,0.7)","LOW":"rgba(15,20,35,0.5)"}
                    border_map = {"CRITICAL":"#7c3aed","HIGH":"#ef4444","MEDIUM":"#3b82f6","LOW":"#6b7280"}
                    st.markdown(f"""
<div style='background:{bg_map.get(priority,"rgba(30,37,53,0.7)")};
border:1px solid {border_map.get(priority,"#3b82f6")};border-radius:12px;
padding:14px 18px;margin-bottom:10px'>
  <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>
    <span style='background:{border_map.get(priority)};color:white;padding:2px 10px;
    border-radius:20px;font-size:11px;font-weight:600'>{priority}</span>
    <span style='font-size:14px;font-weight:600;color:#e2e8f0'>{rec.get("area","")}</span>
  </div>
  <div style='font-size:13px;color:#e2e8f0;margin-bottom:4px'><b>Action:</b> {rec.get("action","")}</div>
  <div style='font-size:12px;color:#8b95a8'><b>Rationale:</b> {rec.get("rationale","")}</div>
  <div style='font-size:11px;color:#6b7280;margin-top:4px'>📜 {rec.get("gdpr_ref","")}</div>
</div>""", unsafe_allow_html=True)

            # B2 Logs
            if b2_logs:
                with st.expander("📋 B2 Logs"):
                    st.markdown(
                        f'<div class="log-block">{"".join(log_html(l) for l in b2_logs)}</div>',
                        unsafe_allow_html=True)
    else:
        st.info("👈 Select a dataset and click **🏛️ Run B2 Agent**.")
        st.markdown("""
#### B2 Pipeline Steps:
| Step | Agent Node | What it does |
|------|-----------|--------------|
| 1 | **SQL Parser** | Parse SQL or auto-generate from CSV schema → extract tables, columns, joins |
| 2 | **Lineage Extractor** | Gemini LLM builds full lineage graph (sources → transforms → sinks) |
| 3 | **PII Tagger** | Detects 11 PII types, assigns GDPR articles, applies 9 masking strategies |
| 4 | **Catalogue Enricher** | Gemini LLM writes business descriptions, stewardship, quality SLAs |
| 5 | **Governance Report** | GDPR compliance score, policy recommendations, full audit trail |
""")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("""**🔴 PII Types Detected**
- EMAIL → Partial mask  
- PHONE → Last 4 visible
- NAME → Initial only
- SSN → Full hash
- DOB → Year only
- ADDRESS → Region only
- IP_ADDRESS → /16 subnet
- CREDIT_CARD → Last 4
- SALARY → Range bucket
- USER_ID → Pseudonymize
- DERIVED_DATA → None""")
        with col_b:
            st.markdown("""**📋 GDPR Checks**
- Data Minimisation (Art.5.1.c)
- PII Identified (Art.4.1)
- Masking Applied (Art.25)
- Retention Policy (Art.5.1.e)
- Lineage Documented (Art.30)
- Access Control (Art.32)""")
        with col_c:
            st.markdown("""**📦 Outputs Generated**
- `b2_masked_*.csv` — PII-masked dataset
- `b2_governance_*.json` — Full GDPR report
- `lineage_*.json` — Lineage graph
- `b2_catalogue_*.csv` — Data catalogue
- SQLite `data_catalogue` table""")


# ─── TAB 11: B3 SELF-HEALING PIPELINE AGENT ───────────────────
with tab_b3:
    st.markdown("""
<h2 style='font-size:22px;font-weight:700;margin-bottom:4px'>
  🔧 B3 · Self-Healing Pipeline Agent
</h2>
<p style='color:#6b7280;font-size:13px'>
  <b style='color:#60a5fa'>Detect Failure</b> →
  <b style='color:#a78bfa'>Classify (LLM)</b> →
  <b style='color:#fbbf24'>Decide Fix (LLM)</b> →
  <b style='color:#34d399'>Auto-Heal + GE</b> →
  <b style='color:#c084fc'>Alert & Log</b>
</p>
""", unsafe_allow_html=True)

    if "b3_results" not in st.session_state: st.session_state.b3_results = {}
    if "b3_ran" not in st.session_state: st.session_state.b3_ran = False

    b3c1, b3c2 = st.columns([2,1])
    with b3c1:
        if st.session_state.uploaded_df is not None:
            st.markdown(f"**📁 Dataset:** `{st.session_state.upload_name}` ({len(st.session_state.uploaded_df)} rows)")
            b3_scenario = "custom_upload"
        else:
            st.warning("⬆️ Upload a CSV in the sidebar first.")
            b3_scenario = None
    with b3c2:
        b3_run_btn = st.button("🔧 Run B3 Agent", type="primary",
                               use_container_width=True, key="b3_run",
                               disabled=b3_scenario is None)

    if b3_run_btn and b3_scenario:
        data_path = st.session_state.get("uploaded_path", "")
        t0 = time.time()
        with st.status(f"🔧 Running B3 on **{b3_scenario}**...", expanded=True) as b3_status:
            st.write("🔍 Node 1/5 — Detection Agent scanning dataset...")
            st.write("🏷️ Node 2/5 — Classification Agent (LLM reasoning)...")
            st.write("🧠 Node 3/5 — Decision Agent (LLM fix planning)...")
            st.write("🔨 Node 4/5 — Healing Agent + GE validation...")
            st.write("📋 Node 5/5 — Logging & Alert Agent...")
            try:
                b3_result, b3_logs = run_single_scenario(b3_scenario, data_path)
                dur = round(time.time()-t0,1)
                b3_status.update(label=f"✅ B3 complete in {dur}s | {b3_result.get('final_status','?')}",
                                 state="complete", expanded=False)
                st.session_state.b3_results[b3_scenario] = b3_result
                st.session_state.b3_ran = True
            except Exception as e:
                import traceback
                st.error(f"B3 error: {e}"); st.code(traceback.format_exc())
                b3_status.update(label="❌ B3 failed", state="error")
                b3_result = {}; b3_logs = []; dur = 0

        if b3_result:
            issues = b3_result.get("issues_detected",[])
            fixes  = b3_result.get("fixes_applied",[])
            n_i, n_f = len(issues), len(fixes)
            df_o = safe_read_csv(data_path)
            df_h = safe_read_csv(b3_result.get("healed_data_path",""))
            qs = quality_score(df_o, df_h, n_i, n_f)

            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Status", b3_result.get("final_status","?"))
            m2.metric("Duration", f"{dur}s")
            m3.metric("Issues", n_i)
            m4.metric("Fixes", n_f)
            m5.metric("Quality", f"{qs}/100")

            # LangGraph node breakdown
            st.markdown("#### 🔗 LangGraph Node Execution")
            node_data = [
                ("🔍 Detection", "DetectionAgent", len(issues), f"{len(issues)} issues found"),
                ("🏷️ Classification", "ClassificationAgent", len(b3_result.get("classifications",[])), b3_result.get("primary_category","—")),
                ("🧠 Decision", "DecisionAgent", len(b3_result.get("fix_plan",[])), b3_result.get("decision_rationale","—")[:80]),
                ("🔨 Healing", "HealingAgent", n_f, f"{sum(1 for f in fixes if f.get('status')=='SUCCESS')}/{n_f} succeeded"),
                ("📋 Logging", "LoggingAgent", 1, b3_result.get("final_status","?")),
            ]
            for icon_name, agent, count, detail in node_data:
                st.markdown(f"""<div style='background:linear-gradient(135deg,rgba(30,37,53,0.8),rgba(20,28,45,0.5));
                border:1px solid rgba(99,130,255,0.12);border-radius:12px;padding:12px 18px;margin-bottom:8px;
                display:flex;align-items:center;justify-content:space-between'>
                <div><span style='font-size:14px;font-weight:600;color:#e2e8f0'>{icon_name}</span>
                <span style='font-size:12px;color:#8b95a8;margin-left:8px'>{agent}</span></div>
                <div style='text-align:right'><span style='font-size:13px;color:#93c5fd;font-weight:500'>{count}</span>
                <div style='font-size:11px;color:#6b7280;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{detail}</div></div>
                </div>""", unsafe_allow_html=True)

            # Sub-tabs
            b3t1,b3t2,b3t3,b3t4 = st.tabs(["⚠️ Issues","🔧 Fix Plan","📊 Before vs After","✅ GE Validation"])
            with b3t1:
                if issues:
                    df_i = pd.DataFrame([{"Type":i.get("type",""),"Column":i.get("column","—"),
                        "Severity":i.get("severity",""),"Detail":i.get("detail","")} for i in issues])
                    st.dataframe(df_i, use_container_width=True, hide_index=True)
                else:
                    st.success("No issues detected!")
            with b3t2:
                fp = b3_result.get("fix_plan",[])
                if fp:
                    df_fp = pd.DataFrame([{"Action":f.get("action",""),"Confidence":f.get("confidence",0),
                        "Rationale":f.get("rationale","")} for f in fp])
                    st.dataframe(df_fp, use_container_width=True, hide_index=True,
                                column_config={"Confidence":st.column_config.ProgressColumn("Confidence",min_value=0,max_value=1,format="%.0%")})
                else:
                    st.info("No fix plan generated.")
            with b3t3:
                if df_o is not None and df_h is not None:
                    oc,hc = st.columns(2)
                    with oc:
                        st.caption("🔴 Original"); st.dataframe(df_o.head(8).style.highlight_null(color="#4c0519").format(precision=2),use_container_width=True,height=200)
                    with hc:
                        st.caption("✅ Healed"); st.dataframe(df_h.head(8).style.format(precision=2),use_container_width=True,height=200)
            with b3t4:
                pre = b3_result.get("ge_pre_results",{}); post = b3_result.get("ge_post_results",{})
                if pre or post:
                    gc1,gc2 = st.columns(2)
                    gc1.metric("GE Pre", f"{pre.get('passed',0)}/{pre.get('total',0)} ({pre.get('success_pct',0)}%)")
                    gc2.metric("GE Post", f"{post.get('passed',0)}/{post.get('total',0)} ({post.get('success_pct',0)}%)")
                else:
                    st.info("No GE results available.")

            # Logs
            if b3_logs:
                with st.expander("📋 B3 Autonomous Run Log"):
                    st.markdown(f'<div class="log-block">{"".join(log_html(l) for l in b3_logs)}</div>',
                                unsafe_allow_html=True)
    else:
        st.info("👈 Select a dataset and click **🔧 Run B3 Agent**.")
        st.markdown("""
#### B3 Pipeline — LangGraph 5-Node Graph:
| Node | Agent | What it does |
|------|-------|--------------| 
| 1 | **Detection Agent** | Statistical + structural scanning (nulls, schema, outliers) |
| 2 | **Classification Agent** | LLM classifies each issue (DATA_QUALITY, SCHEMA, ANOMALY, SYSTEM_FAILURE) |
| 3 | **Decision Agent** | LLM decides optimal fix action per issue with confidence scores |
| 4 | **Healing Agent** | Executes fixes + runs GE pre/post validation + MockDB snapshots |
| 5 | **Logging Agent** | Structured report + alerts for HIGH severity issues |
""")


