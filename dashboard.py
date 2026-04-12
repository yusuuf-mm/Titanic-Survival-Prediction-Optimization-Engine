import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="Titanic · Decision Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── Root tokens ─────────────────────────── */
:root {
  --bg:        #06090f;
  --surface:   #0d1421;
  --surface2:  #111c2e;
  --border:    rgba(255,255,255,0.07);
  --accent:    #00d4aa;
  --accent2:   #0066ff;
  --danger:    #ff4560;
  --warn:      #f5a623;
  --text:      #e8edf5;
  --muted:     #6b7a99;
  --font-head: 'Bebas Neue', sans-serif;
  --font-body: 'DM Sans', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

/* ── App shell ───────────────────────────── */
.stApp {
  background: var(--bg);
  font-family: var(--font-body);
  color: var(--text);
}
.block-container {
  padding: 2rem 2.5rem 4rem;
  max-width: 1400px;
}

/* ── Sidebar ─────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
  font-family: var(--font-body) !important;
}

/* ── Hero banner ─────────────────────────── */
.hero {
  position: relative;
  padding: 3.5rem 3rem 2.5rem;
  margin: -2rem -2.5rem 2.5rem;
  background: linear-gradient(135deg, #060e1f 0%, #0a1628 40%, #062030 100%);
  border-bottom: 1px solid var(--border);
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse 70% 60% at 80% 50%, rgba(0,212,170,0.06) 0%, transparent 70%),
              radial-gradient(ellipse 50% 80% at 10% 90%, rgba(0,102,255,0.07) 0%, transparent 60%);
  pointer-events: none;
}
.hero-grid {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,212,170,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,170,0.04) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}
.hero-label {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  letter-spacing: 0.22em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}
.hero h1 {
  font-family: var(--font-head) !important;
  font-size: clamp(2.8rem, 5vw, 4.8rem) !important;
  letter-spacing: 0.06em;
  line-height: 1;
  color: #fff !important;
  margin: 0 0 0.6rem !important;
  padding: 0 !important;
}
.hero h1 span { color: var(--accent); }
.hero-sub {
  font-size: 0.95rem;
  color: var(--muted);
  font-weight: 300;
  max-width: 560px;
  line-height: 1.6;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 20px;
  font-family: var(--font-mono);
  font-size: 0.62rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  border: 1px solid;
}
.badge-online  { color: var(--accent);  border-color: rgba(0,212,170,0.35); background: rgba(0,212,170,0.08); }
.badge-offline { color: var(--danger);  border-color: rgba(255,69,96,0.35);  background: rgba(255,69,96,0.08); }
.badge-dot { width: 6px; height: 6px; border-radius: 50%; }
.dot-on  { background: var(--accent); box-shadow: 0 0 6px var(--accent); }
.dot-off { background: var(--danger); }

/* ── KPI cards ───────────────────────────── */
.kpi-row { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
.kpi-card {
  flex: 1;
  min-width: 140px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.1rem 1.3rem;
  position: relative;
  overflow: hidden;
  transition: border-color .2s;
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--accent-line, var(--accent));
}
.kpi-card:hover { border-color: rgba(255,255,255,0.15); }
.kpi-label {
  font-family: var(--font-mono);
  font-size: 0.58rem;
  letter-spacing: 0.18em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}
.kpi-value {
  font-family: var(--font-head);
  font-size: 2.2rem;
  letter-spacing: 0.04em;
  line-height: 1;
  color: #fff;
}
.kpi-sub {
  font-size: 0.72rem;
  color: var(--muted);
  margin-top: 0.2rem;
}

/* ── Tabs ────────────────────────────────── */
[data-testid="stTabs"] button {
  font-family: var(--font-body) !important;
  font-weight: 500 !important;
  font-size: 0.85rem !important;
  color: var(--muted) !important;
  border-radius: 6px 6px 0 0 !important;
  padding: 0.6rem 1.2rem !important;
  transition: color .2s !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--accent) !important;
  background: rgba(0,212,170,0.06) !important;
  border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid var(--border) !important;
  gap: 4px;
}

/* ── Section titles ──────────────────────── */
.section-title {
  font-family: var(--font-head);
  font-size: 1.8rem;
  letter-spacing: 0.06em;
  color: #fff;
  margin: 0 0 0.3rem;
}
.section-sub {
  font-size: 0.82rem;
  color: var(--muted);
  margin-bottom: 1.6rem;
  line-height: 1.6;
}

/* ── Prediction result cards ─────────────── */
.result-card {
  padding: 1.6rem 2rem;
  border-radius: 12px;
  margin-top: 1.2rem;
  border: 1px solid;
  position: relative;
  overflow: hidden;
}
.result-survived {
  background: rgba(0,212,170,0.07);
  border-color: rgba(0,212,170,0.3);
}
.result-perished {
  background: rgba(255,69,96,0.07);
  border-color: rgba(255,69,96,0.3);
}
.result-emoji { font-size: 2.5rem; margin-bottom: 0.5rem; }
.result-verdict {
  font-family: var(--font-head);
  font-size: 2rem;
  letter-spacing: 0.06em;
  color: #fff;
  margin-bottom: 0.3rem;
}
.result-prob { font-family: var(--font-mono); font-size: 0.9rem; color: var(--muted); }
.prob-highlight { color: var(--accent); font-size: 1.1rem; }

/* ── Prob bar ────────────────────────────── */
.prob-bar-wrap { margin-top: 1rem; }
.prob-bar-label {
  font-family: var(--font-mono);
  font-size: 0.62rem;
  letter-spacing: 0.1em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 5px;
}
.prob-bar-track {
  background: rgba(255,255,255,0.07);
  border-radius: 4px;
  height: 8px;
  overflow: hidden;
}
.prob-bar-fill {
  height: 100%;
  border-radius: 4px;
  background: linear-gradient(90deg, var(--fill-start), var(--fill-end));
  width: var(--fill-w);
  transition: width 0.8s cubic-bezier(.4,0,.2,1);
}

/* ── Form elements ───────────────────────── */
.stSelectbox label, .stSlider label, .stNumberInput label {
  font-family: var(--font-mono) !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.1em !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
}
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] input {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
}
[data-testid="stSelectbox"] > div:focus-within,
[data-testid="stNumberInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(0,212,170,0.15) !important;
}

/* ── Buttons ─────────────────────────────── */
.stButton button {
  font-family: var(--font-body) !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.05em !important;
  background: linear-gradient(135deg, #00d4aa, #00a87f) !important;
  color: #06090f !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.6rem 1.8rem !important;
  cursor: pointer !important;
  transition: opacity .2s, transform .15s !important;
}
.stButton button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
.stButton button:active { transform: translateY(0) !important; }

/* ── Expander ────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
  font-family: var(--font-mono) !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.1em !important;
  color: var(--muted) !important;
}

/* ── Dataframe ───────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  overflow: hidden !important;
}

/* ── Info / warning / success banners ────── */
.stAlert {
  border-radius: 10px !important;
  border: 1px solid var(--border) !important;
  font-family: var(--font-body) !important;
}

/* ── Divider ─────────────────────────────── */
.hl { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }

/* ── Manifest header ─────────────────────── */
.manifest-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 1.5rem 0 0.8rem;
}
.manifest-header-title {
  font-family: var(--font-head);
  font-size: 1.4rem;
  letter-spacing: 0.06em;
  color: #fff;
}
.manifest-count {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--accent);
  background: rgba(0,212,170,0.1);
  border: 1px solid rgba(0,212,170,0.25);
  border-radius: 20px;
  padding: 2px 10px;
}

/* ── Sidebar ─────────────────────────────── */
.sidebar-section {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.9rem 1rem;
  margin-bottom: 1rem;
}
.sidebar-label {
  font-family: var(--font-mono);
  font-size: 0.58rem;
  letter-spacing: 0.2em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}
.sidebar-value {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  color: var(--text);
}

/* ── Status pill ─────────────────────────── */
.status-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 0.5rem;
}

/* ── 3D tab placeholder ──────────────────── */
.vis-prompt {
  min-height: 400px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  color: var(--muted);
  background: var(--surface);
  border: 1px dashed var(--border);
  border-radius: 14px;
}
.vis-prompt-icon { font-size: 3rem; opacity: 0.5; }
.vis-prompt-text {
  font-family: var(--font-mono);
  font-size: 0.78rem;
  letter-spacing: 0.1em;
  text-align: center;
  color: var(--muted);
  max-width: 340px;
  line-height: 1.7;
}

/* ── Constraint card ─────────────────────── */
.constraint-grid {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  margin: 1rem 0;
}
.constraint-card {
  flex: 1;
  min-width: 180px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.1rem;
}
.constraint-title {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  letter-spacing: 0.18em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}

/* ── Scrollbar ───────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS & STATE
# ─────────────────────────────────────────────
DEFAULT_API_URL = "http://localhost:8000"

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Bebas Neue",sans-serif; font-size:1.4rem;
                letter-spacing:.1em; color:#fff; margin-bottom:1.5rem;
                padding-bottom:.8rem; border-bottom:1px solid rgba(255,255,255,.07);'>
      ⚙ SYSTEM CONFIG
    </div>
    """, unsafe_allow_html=True)

    api_url = st.text_input("API Endpoint", DEFAULT_API_URL,
                            help="URL of your FastAPI backend")

    try:
        health_response = requests.get(f"{api_url}/health", timeout=2)
        api_health = health_response.status_code == 200
    except Exception:
        api_health = False

    status_html = """
    <div class='sidebar-section'>
      <div class='sidebar-label'>Backend Status</div>
      <div class='status-row'>
        <span class='sidebar-value'>FastAPI Server</span>
        <span class='hero-badge {cls}'><span class='badge-dot {dot}'></span>{label}</span>
      </div>
    </div>
    """.format(
        cls="badge-online" if api_health else "badge-offline",
        dot="dot-on" if api_health else "dot-off",
        label="ONLINE" if api_health else "OFFLINE"
    )
    st.markdown(status_html, unsafe_allow_html=True)

    if api_health:
        try:
            model_ver = requests.get(f"{api_url}/model/version", timeout=2).json()
            active = model_ver.get("active_version", "—")
        except Exception:
            active = "—"
        st.markdown(f"""
        <div class='sidebar-section'>
          <div class='sidebar-label'>Active Model</div>
          <div class='sidebar-value'>{active}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='sidebar-section' style='border-color:rgba(255,69,96,.25);'>
          <div class='sidebar-label' style='color:rgba(255,69,96,.7);'>Required Action</div>
          <div class='sidebar-value' style='font-size:.75rem; color:#6b7a99; line-height:1.6;'>
            Run <code style='background:rgba(255,255,255,.06);
            padding:1px 5px; border-radius:3px; font-size:.7rem;'>
            uvicorn predict:app --reload</code> to bring the API online.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace; font-size:.62rem;
                color:rgba(107,122,153,.6); line-height:1.8;'>
      TITANIC DECISION INTELLIGENCE<br>
      Operations Research × ML<br>
      Portfolio Project · v2.0
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-grid"></div>
  <div class="hero-label">🚢 &nbsp; Portfolio · Decision Intelligence System</div>
  <h1>TITANIC <span>RESCUE</span><br>INTELLIGENCE</h1>
  <p class="hero-sub">
    Machine learning predictions fused with Operations Research optimization —
    allocating survival under hard constraints.
  </p>
  <br>
  <span class='hero-badge {"badge-online" if api_health else "badge-offline"}'>
    <span class='badge-dot {"dot-on" if api_health else "dot-off"}'></span>
    {"API Online" if api_health else "API Offline"}
  </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  Single Prediction",
    "🧮  Lifeboat Optimizer",
    "🛳  3D Rescue Map",
    "📊  Model Insights"
])

# ══════════════════════════════════════════════
#  TAB 1 — SINGLE PREDICTION
# ══════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class='section-title'>Passenger Survival Predictor</div>
    <div class='section-sub'>
      Submit a passenger profile to the ML endpoint and receive a survival prediction
      with probability score from the trained classifier.
    </div>
    """, unsafe_allow_html=True)

    col_form, col_gap, col_result = st.columns([2, 0.2, 1.8])

    with col_form:
        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                pclass   = st.selectbox("Passenger Class", [1, 2, 3], index=2)
                sex      = st.selectbox("Sex", ["male", "female"])
                age      = st.slider("Age", 0.0, 100.0, 25.0, step=0.5)
                fare     = st.number_input("Fare (£)", 0.0, 500.0, 15.0, step=0.5)
            with c2:
                sibsp    = st.number_input("Siblings / Spouses", 0, 10, 0)
                parch    = st.number_input("Parents / Children", 0, 10, 0)
                embarked = st.selectbox("Embarked Port", ["C — Cherbourg", "Q — Queenstown", "S — Southampton"], index=2)

            port_map = {"C — Cherbourg": "C", "Q — Queenstown": "Q", "S — Southampton": "S"}
            predict_btn = st.button("▶  Run Prediction", use_container_width=True)

    with col_result:
        st.markdown("<br>", unsafe_allow_html=True)
        if predict_btn:
            if not api_health:
                st.error("Backend API is offline. Start the server to enable predictions.")
            else:
                payload = {
                    "pclass": pclass, "sex": sex, "age": age,
                    "sibsp": sibsp, "parch": parch,
                    "fare": fare, "embarked": port_map[embarked]
                }
                with st.spinner("Calling ML endpoint…"):
                    try:
                        response = requests.post(f"{api_url}/predict", json=payload, timeout=5)
                        if response.status_code == 200:
                            result   = response.json()
                            survived = result['survived']
                            prob     = result['survival_probability']
                            message  = result['message']

                            bar_color_start = "#00d4aa" if survived else "#ff4560"
                            bar_color_end   = "#00a87f" if survived else "#cc2a40"
                            card_cls        = "result-survived" if survived else "result-perished"
                            emoji           = "🟢" if survived else "🔴"
                            verdict         = "SURVIVED" if survived else "PERISHED"

                            st.markdown(f"""
                            <div class='result-card {card_cls}'>
                              <div class='result-emoji'>{emoji}</div>
                              <div class='result-verdict'>{verdict}</div>
                              <div class='result-prob'>
                                Survival probability:
                                <span class='prob-highlight'>{prob:.1%}</span>
                              </div>
                              <div class='prob-bar-wrap'>
                                <div class='prob-bar-label'>Confidence Score</div>
                                <div class='prob-bar-track'>
                                  <div class='prob-bar-fill' style='
                                    --fill-w:{prob*100:.1f}%;
                                    --fill-start:{bar_color_start};
                                    --fill-end:{bar_color_end};
                                  '></div>
                                </div>
                              </div>
                              <div style='margin-top:.8rem; font-size:.75rem;
                                          color:rgba(107,122,153,.9); font-family:"JetBrains Mono",monospace;'>
                                {message}
                              </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"API error {response.status_code}: {response.text}")
                    except requests.RequestException as e:
                        st.error(f"Request failed: {e}")
        else:
            st.markdown("""
            <div style='height:280px; display:flex; flex-direction:column;
                        align-items:center; justify-content:center;
                        background:var(--surface); border:1px dashed rgba(255,255,255,.07);
                        border-radius:12px; gap:.8rem;'>
              <div style='font-size:2.5rem; opacity:.3;'>🔮</div>
              <div style='font-family:"JetBrains Mono",monospace; font-size:.7rem;
                          letter-spacing:.12em; color:rgba(107,122,153,.7); text-align:center;
                          text-transform:uppercase;'>
                Configure a passenger<br>profile and run prediction
              </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — LIFEBOAT OPTIMIZER
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class='section-title'>Lifeboat Resource Allocator</div>
    <div class='section-sub'>
      Linear Programming (LP) optimization engine maximizes expected survivors
      under hard capacity and ethical priority constraints.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙  Optimization Constraints", expanded=True):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown("<div class='constraint-title'>Capacity</div>", unsafe_allow_html=True)
            capacity = st.number_input("Lifeboat Seats", min_value=1, max_value=500, value=50)
        with cc2:
            st.markdown("<div class='constraint-title'>Ethical Priorities</div>", unsafe_allow_html=True)
            priority_children = st.checkbox("Children (<18) ≥ 30%", value=True)
            priority_women    = st.checkbox("Women ≥ 50%", value=True)
        with cc3:
            st.markdown("<div class='constraint-title'>Family Rule</div>", unsafe_allow_html=True)
            max_family = st.number_input("Max Seats per Family", min_value=1, max_value=20, value=3)

    run_opt = st.button("▶  Generate Crowd & Optimize", use_container_width=False)

    if run_opt:
        if not api_health:
            st.error("Backend API is offline. Optimization requires the FastAPI server.")
        else:
            with st.spinner("Generating synthetic crowd and solving LP…"):
                np.random.seed(int(time.time()))
                n_crowd = int(capacity * 2.5)
                crowd = [{
                    "pclass":   int(np.random.choice([1, 2, 3], p=[0.2, 0.3, 0.5])),
                    "sex":      str(np.random.choice(["male", "female"], p=[0.65, 0.35])),
                    "age":      float(max(1.0, np.random.normal(29.0, 14.0))),
                    "sibsp":    int(np.random.choice([0,1,2,3,4,5,8], p=[0.68,0.23,0.03,0.02,0.02,0.01,0.01])),
                    "parch":    int(np.random.choice([0,1,2,3,4,5,6], p=[0.76,0.13,0.09,0.005,0.005,0.005,0.005])),
                    "fare":     float(abs(np.random.normal(32.0, 20.0))),
                    "embarked": str(np.random.choice(["C","Q","S"], p=[0.2,0.1,0.7]))
                } for _ in range(n_crowd)]

                payload = {
                    "passengers": crowd,
                    "capacity": capacity,
                    "priority_children": priority_children,
                    "priority_women": priority_women,
                    "max_family_members": max_family
                }
                res = requests.post(f"{api_url}/optimize-allocation", json=payload, timeout=30)

            if res.status_code == 200:
                data = res.json()
                selected_df = pd.DataFrame(data['selected_passengers'])
                crowd_df    = pd.DataFrame(crowd)

                # KPI row
                st.markdown(f"""
                <div class='kpi-row'>
                  <div class='kpi-card' style='--accent-line:var(--accent);'>
                    <div class='kpi-label'>Expected Survivors</div>
                    <div class='kpi-value'>{data['objective_value']:.2f}</div>
                    <div class='kpi-sub'>LP objective value</div>
                  </div>
                  <div class='kpi-card' style='--accent-line:#0066ff;'>
                    <div class='kpi-label'>Seat Utilization</div>
                    <div class='kpi-value'>{data['utilization']:.1f}<span style='font-size:1.2rem;'>%</span></div>
                    <div class='kpi-sub'>of {data['capacity']} total seats</div>
                  </div>
                  <div class='kpi-card' style='--accent-line:#f5a623;'>
                    <div class='kpi-label'>Seats Allocated</div>
                    <div class='kpi-value'>{data['selected_count']}</div>
                    <div class='kpi-sub'>from crowd of {n_crowd}</div>
                  </div>
                  <div class='kpi-card' style='--accent-line:#a855f7;'>
                    <div class='kpi-label'>Optimizer Status</div>
                    <div class='kpi-value' style='font-size:1.2rem; padding-top:.4rem;'>{data['status'].upper()}</div>
                    <div class='kpi-sub'>LP solver result</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if data['selected_count'] > 0:
                    st.session_state['crowd_data']    = crowd
                    st.session_state['selected_data'] = data['selected_passengers']

                    selected_df['is_child'] = selected_df['age'].apply(
                        lambda x: "Child (<18)" if x < 18 else "Adult"
                    )

                    st.markdown("<hr class='hl'>", unsafe_allow_html=True)
                    st.markdown("<div class='section-title' style='font-size:1.4rem;'>Ethics Audit</div>", unsafe_allow_html=True)

                    plot_c1, plot_c2 = st.columns(2)

                    PLOT_BG   = "rgba(0,0,0,0)"
                    PAPER_BG  = "rgba(0,0,0,0)"
                    FONT_CLR  = "#6b7a99"
                    TITLE_CLR = "#e8edf5"

                    def style_fig(fig, title):
                        fig.update_layout(
                            title=dict(text=title, font=dict(family="Bebas Neue", size=18, color=TITLE_CLR)),
                            paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                            font=dict(color=FONT_CLR, family="DM Sans"),
                            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_CLR)),
                            margin=dict(l=10, r=10, t=50, b=10)
                        )
                        return fig

                    with plot_c1:
                        fig_gender = px.pie(
                            selected_df, names="sex",
                            color="sex",
                            color_discrete_map={'female': '#00d4aa', 'male': '#0066ff'},
                            hole=0.45
                        )
                        fig_gender = style_fig(fig_gender, "Gender Distribution — Allocated")
                        fig_gender.update_traces(textfont_color="#e8edf5")
                        st.plotly_chart(fig_gender, use_container_width=True)

                    with plot_c2:
                        fig_age = px.pie(
                            selected_df, names="is_child",
                            color="is_child",
                            color_discrete_map={'Child (<18)': '#f5a623', 'Adult': '#334466'},
                            hole=0.45
                        )
                        fig_age = style_fig(fig_age, "Age Demographics — Allocated")
                        fig_age.update_traces(textfont_color="#e8edf5")
                        st.plotly_chart(fig_age, use_container_width=True)

                    # Class comparison
                    crowd_class          = crowd_df['pclass'].value_counts().reset_index()
                    crowd_class.columns  = ['Class', 'Count']
                    crowd_class['Group'] = 'Original Crowd'

                    sel_class            = selected_df['pclass'].value_counts().reset_index()
                    sel_class.columns    = ['Class', 'Count']
                    sel_class['Group']   = 'Allocated'

                    combined = pd.concat([crowd_class, sel_class])
                    fig_class = px.bar(
                        combined, x='Class', y='Count', color='Group', barmode='group',
                        color_discrete_map={'Original Crowd': '#1e2f50', 'Allocated': '#00d4aa'}
                    )
                    fig_class.update_layout(
                        bargap=0.3, bargroupgap=0.05,
                        xaxis=dict(title="Passenger Class", color=FONT_CLR,
                                   gridcolor="rgba(255,255,255,.04)"),
                        yaxis=dict(title="Count", color=FONT_CLR,
                                   gridcolor="rgba(255,255,255,.04)"),
                    )
                    fig_class = style_fig(fig_class, "Socioeconomic Class Representation — Crowd vs Allocated")
                    st.plotly_chart(fig_class, use_container_width=True)

                    # Manifest
                    st.markdown(f"""
                    <div class='manifest-header'>
                      <span class='manifest-header-title'>Allocation Manifest</span>
                      <span class='manifest-count'>{data['selected_count']} passengers</span>
                    </div>
                    """, unsafe_allow_html=True)
                    display_cols = ['pclass', 'sex', 'age', 'fare', 'sibsp', 'parch', 'survival_prob']
                    avail_cols   = [c for c in display_cols if c in selected_df.columns]
                    st.dataframe(
                        selected_df[avail_cols].style.highlight_max(
                            subset=['survival_prob'] if 'survival_prob' in avail_cols else [],
                            color='#0a2e20'
                        ),
                        use_container_width=True, height=320
                    )
            else:
                st.error(f"Optimization failed ({res.status_code}): {res.text}")

# ══════════════════════════════════════════════
#  TAB 3 — 3D RESCUE MAP
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class='section-title'>3D Rescue Blueprint</div>
    <div class='section-sub'>
      LP output mapped onto a synthetic 3D ship hull. Extraction routes connect
      allocated survivors to their nearest lifeboat station.
    </div>
    """, unsafe_allow_html=True)

    if 'crowd_data' in st.session_state and 'selected_data' in st.session_state:
        crowd_df    = pd.DataFrame(st.session_state['crowd_data'])
        selected_df = pd.DataFrame(st.session_state['selected_data'])
        np.random.seed(42)

        crowd_df['X'] = np.random.uniform(5, 195, len(crowd_df))

        def hull_width(x):
            return 30.0 * np.sqrt(max(0.0, 1 - ((x - 100) / 100.0) ** 2))

        crowd_df['Y'] = crowd_df['X'].apply(
            lambda x: np.random.uniform(-hull_width(x), hull_width(x))
        )
        crowd_df['Z'] = crowd_df['pclass'].map({1: 3, 2: 2, 3: 1})
        crowd_df['Z'] += np.random.uniform(-0.1, 0.1, len(crowd_df))

        viz_df           = crowd_df.copy()
        viz_df['Status'] = "Perished"
        saved_indices    = []

        for pclass_val in [1, 2, 3]:
            for sex_val in ["male", "female"]:
                n_target   = len(selected_df[(selected_df['pclass'] == pclass_val) & (selected_df['sex'] == sex_val)])
                candidates = viz_df[(viz_df['pclass'] == pclass_val) & (viz_df['sex'] == sex_val)].index.tolist()
                if n_target > 0 and candidates:
                    chosen = np.random.choice(candidates, min(n_target, len(candidates)), replace=False)
                    saved_indices.extend(chosen)

        viz_df.loc[saved_indices, 'Status'] = "Rescued"

        fig = go.Figure()

        perished = viz_df[viz_df['Status'] == "Perished"]
        fig.add_trace(go.Scatter3d(
            x=perished['X'], y=perished['Y'], z=perished['Z'],
            mode='markers',
            marker=dict(size=3.5, color='#8B0000', opacity=0.25, symbol='circle'),
            name='Perished',
            hoverinfo='text',
            text=perished.apply(
                lambda r: f"Class {r['pclass']} · {r['sex']} · Age {r['age']:.0f}", axis=1
            )
        ))

        rescued = viz_df[viz_df['Status'] == "Rescued"]
        fig.add_trace(go.Scatter3d(
            x=rescued['X'], y=rescued['Y'], z=rescued['Z'],
            mode='markers',
            marker=dict(
                size=6, color='#00d4aa', opacity=1.0, symbol='diamond',
                line=dict(color='rgba(255,255,255,0.6)', width=1)
            ),
            name='Allocated (Rescued)',
            hoverinfo='text',
            text=rescued.apply(
                lambda r: f"Class {r['pclass']} · {r['sex']} · Age {r['age']:.0f} · ✅ ALLOCATED", axis=1
            )
        ))

        lifeboats = pd.DataFrame([
            {'X': 50,  'Y': -36, 'Z': 3.5, 'Name': 'LB-1 Port'},
            {'X': 150, 'Y': -36, 'Z': 3.5, 'Name': 'LB-2 Port'},
            {'X': 50,  'Y': 36,  'Z': 3.5, 'Name': 'LB-3 Stbd'},
            {'X': 150, 'Y': 36,  'Z': 3.5, 'Name': 'LB-4 Stbd'},
        ])
        fig.add_trace(go.Scatter3d(
            x=lifeboats['X'], y=lifeboats['Y'], z=lifeboats['Z'],
            mode='markers+text',
            marker=dict(size=11, color='#f5a623', symbol='square',
                        line=dict(color='white', width=1)),
            text=lifeboats['Name'], textposition="top center",
            name='Lifeboat Stations',
            textfont=dict(color='#f5a623', size=10)
        ))

        line_x, line_y, line_z = [], [], []
        for _, r in rescued.iterrows():
            dist     = lifeboats.apply(lambda lb: (lb['X']-r['X'])**2 + (lb['Y']-r['Y'])**2, axis=1)
            nearest  = lifeboats.loc[dist.idxmin()]
            line_x.extend([r['X'], nearest['X'], None])
            line_y.extend([r['Y'], nearest['Y'], None])
            line_z.extend([r['Z'], nearest['Z'], None])

        fig.add_trace(go.Scatter3d(
            x=line_x, y=line_y, z=line_z,
            mode='lines',
            line=dict(color='rgba(0,212,170,0.35)', width=1.5),
            name='Extraction Routes',
            hoverinfo='none'
        ))

        SCENE_BG = '#06090f'
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Stern → Bow',         showgrid=False, backgroundcolor=SCENE_BG, showbackground=True, color='#3a4a66'),
                yaxis=dict(title='Port ↔ Starboard',    showgrid=False, backgroundcolor=SCENE_BG, showbackground=True, color='#3a4a66'),
                zaxis=dict(title='Deck Level',           showgrid=True,  backgroundcolor=SCENE_BG, showbackground=True, color='#3a4a66',
                           gridcolor='rgba(255,255,255,.04)'),
                bgcolor=SCENE_BG
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            paper_bgcolor='#06090f',
            font=dict(color='#6b7a99', family='DM Sans'),
            legend=dict(
                x=0.02, y=0.95, bgcolor='rgba(13,20,33,0.85)',
                bordercolor='rgba(255,255,255,.07)', borderwidth=1,
                font=dict(size=11, color='#e8edf5')
            ),
            title=dict(
                text='Decision Intelligence · Extraction Blueprint',
                font=dict(family='Bebas Neue', size=20, color='#e8edf5')
            )
        )

        st.plotly_chart(fig, use_container_width=True, height=680)

        st.markdown("""
        <div style='background:var(--surface); border:1px solid rgba(0,212,170,.15);
                    border-radius:10px; padding:1rem 1.3rem; margin-top:.5rem;
                    font-size:.8rem; color:var(--muted); line-height:1.7;'>
          <span style='color:var(--accent); font-weight:600;'>Visual Key —</span>
          <strong style='color:#e8edf5;'>Green diamonds</strong> = LP-allocated survivors · 
          <strong style='color:#8B0000;'>Dark red circles</strong> = Perished passengers · 
          <strong style='color:#f5a623;'>Orange squares</strong> = Lifeboat stations · 
          <strong style='color:rgba(0,212,170,.7);'>Teal lines</strong> = Computed extraction routes.
          Deck Z-axis reflects passenger class (3 = upper/first class).
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='vis-prompt'>
          <div class='vis-prompt-icon'>🛳</div>
          <div class='vis-prompt-text'>
            Run the Lifeboat Optimizer first.<br>
            The 3D blueprint is generated from<br>
            LP allocation output.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 4 — MODEL INSIGHTS
# ══════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class='section-title'>Model Insights</div>
    <div class='section-sub'>
      Statistical breakdowns of Titanic survival dynamics — the patterns
      the ML model learned from.
    </div>
    """, unsafe_allow_html=True)

    PLOT_BG  = "rgba(0,0,0,0)"
    FONT_CLR = "#6b7a99"
    TITLE_CLR= "#e8edf5"

    def style_insight(fig, title):
        fig.update_layout(
            title=dict(text=title, font=dict(family="Bebas Neue", size=17, color=TITLE_CLR)),
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_CLR, family="DM Sans"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_CLR)),
            xaxis=dict(gridcolor="rgba(255,255,255,.04)", color=FONT_CLR),
            yaxis=dict(gridcolor="rgba(255,255,255,.04)", color=FONT_CLR),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        return fig

    # ── Feature Importance (illustrative) ──
    features = ['Sex (Female)', 'Passenger Class', 'Age', 'Fare',
                'Sibsp', 'Parch', 'Embarked Port']
    importance = [0.38, 0.27, 0.14, 0.09, 0.05, 0.04, 0.03]
    colors = ['#00d4aa' if v == max(importance) else '#1e3a5f' for v in importance]

    fig_imp = go.Figure(go.Bar(
        x=importance, y=features, orientation='h',
        marker_color=colors,
        text=[f"{v:.0%}" for v in importance],
        textfont=dict(color='#e8edf5', size=11),
        textposition='outside'
    ))
    fig_imp = style_insight(fig_imp, "Feature Importance — Trained Classifier")
    fig_imp.update_layout(
        xaxis=dict(range=[0, 0.48], tickformat='.0%', gridcolor="rgba(255,255,255,.04)", color=FONT_CLR),
        height=320
    )

    # ── Survival by Class & Sex ──
    survival_data = {
        'Class':    [1,1,2,2,3,3],
        'Sex':      ['female','male','female','male','female','male'],
        'SurvRate': [0.97,    0.37,  0.92,    0.16,  0.50,    0.14]
    }
    df_surv = pd.DataFrame(survival_data)
    fig_cls = px.bar(
        df_surv, x='Class', y='SurvRate', color='Sex', barmode='group',
        color_discrete_map={'female': '#00d4aa', 'male': '#0066ff'},
        text=[f"{v:.0%}" for v in df_surv['SurvRate']]
    )
    fig_cls.update_traces(textfont_color='#e8edf5', textposition='outside')
    fig_cls = style_insight(fig_cls, "Survival Rate by Class & Sex")
    fig_cls.update_layout(
        yaxis=dict(tickformat='.0%', range=[0,1.1], gridcolor="rgba(255,255,255,.04)", color=FONT_CLR),
        xaxis_title="Passenger Class", yaxis_title="Survival Rate", height=320
    )

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(fig_imp, use_container_width=True)
    with r1c2:
        st.plotly_chart(fig_cls, use_container_width=True)

    # ── Age survival distribution ──
    np.random.seed(7)
    ages_surv   = np.concatenate([np.random.normal(28, 12, 340), np.random.normal(6, 3, 60)])
    ages_perish = np.concatenate([np.random.normal(32, 15, 540), np.random.normal(22, 8, 80)])
    ages_surv   = np.clip(ages_surv, 0, 80)
    ages_perish = np.clip(ages_perish, 0, 80)

    fig_age_dist = go.Figure()
    fig_age_dist.add_trace(go.Histogram(
        x=ages_surv, name='Survived',
        marker_color='rgba(0,212,170,0.7)', nbinsx=30,
        xbins=dict(start=0, end=80, size=2.5)
    ))
    fig_age_dist.add_trace(go.Histogram(
        x=ages_perish, name='Perished',
        marker_color='rgba(255,69,96,0.5)', nbinsx=30,
        xbins=dict(start=0, end=80, size=2.5)
    ))
    fig_age_dist.update_layout(barmode='overlay')
    fig_age_dist = style_insight(fig_age_dist, "Age Distribution — Survived vs Perished")
    fig_age_dist.update_layout(
        xaxis_title="Age", yaxis_title="Count", height=300,
        legend=dict(x=0.75, y=0.95)
    )
    st.plotly_chart(fig_age_dist, use_container_width=True)

    # ── Fare vs probability scatter (illustrative) ──
    np.random.seed(12)
    n = 200
    fares  = np.abs(np.random.exponential(35, n))
    probs  = np.clip(0.12 + 0.45*(fares/300) + np.random.normal(0, 0.12, n), 0, 1)
    pclass_s = np.random.choice([1,2,3], n, p=[0.25,0.3,0.45])
    fig_fare = px.scatter(
        x=fares, y=probs, color=pclass_s.astype(str),
        color_discrete_map={'1':'#00d4aa','2':'#0066ff','3':'#6b7a99'},
        labels={'x':'Fare (£)','y':'Survival Probability','color':'Class'},
        opacity=0.7
    )
    fig_fare.update_traces(marker=dict(size=5))
    fig_fare = style_insight(fig_fare, "Fare vs Survival Probability — By Class")
    fig_fare.update_layout(
        yaxis=dict(tickformat='.0%', range=[0,1.05]),
        height=300
    )
    st.plotly_chart(fig_fare, use_container_width=True)

    st.markdown("""
    <div style='background:var(--surface); border:1px solid rgba(255,255,255,.06);
                border-radius:10px; padding:1rem 1.3rem; margin-top:.5rem;
                font-size:.78rem; color:var(--muted); line-height:1.8;'>
      <strong style='color:#e8edf5; font-family:"Bebas Neue",sans-serif; font-size:.95rem;
                     letter-spacing:.06em;'>
        INSIGHT NOTE
      </strong><br>
      Feature importances and survival rate charts above are illustrative values aligned
      with published Titanic analysis literature. Age distribution and fare scatter use
      synthetic samples drawn from historical distributions. Connect to a live model endpoint
      to load real SHAP values and model-specific metrics.
    </div>
    """, unsafe_allow_html=True)