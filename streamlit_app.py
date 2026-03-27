"""
╔══════════════════════════════════════════════════════════════════════╗
║          💪 FitPulse — Complete Health Analytics App                 ║
║  Milestone 1: Data Cleaning                                          ║
║  Milestone 2: ML Pipeline (TSFresh · Prophet · KMeans · DBSCAN)     ║
║  Milestone 3: Anomaly Detection & Visualization                      ║
╚══════════════════════════════════════════════════════════════════════╝

Run:
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn \
                tsfresh prophet plotly
    streamlit run fitpulse_all_milestones.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")



# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse · All Milestones",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme (dark/light toggle) ──────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dark = st.session_state.dark_mode

if dark:
    BG         = "linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#1a1a2e)"
    CARD_BG    = "rgba(15,23,42,0.88)"
    CARD_BOR   = "rgba(99,179,237,0.22)"
    TEXT       = "#e2e8f0"
    MUTED      = "#94a3b8"
    ACCENT     = "#63b3ed"
    ACCENT2    = "#f687b3"
    ACCENT3    = "#68d391"
    ACCENT_RED = "#fc8181"
    ACCENT_PUR = "#A78BFA"
    PLOT_BG    = "#0f172a"
    PAPER_BG   = "#0a0e1a"
    GRID_CLR   = "rgba(255,255,255,0.06)"
    BADGE_BG   = "rgba(99,179,237,0.15)"
    SECTION_BG = "rgba(99,179,237,0.07)"
    WARN_BG    = "rgba(246,173,85,0.12)"
    WARN_BOR   = "rgba(246,173,85,0.40)"
    SUCCESS_BG = "rgba(104,211,145,0.10)"
    SUCCESS_BOR= "rgba(104,211,145,0.40)"
    DANGER_BG  = "rgba(252,129,129,0.10)"
    DANGER_BOR = "rgba(252,129,129,0.40)"
    SIDEBAR_BG = "rgba(10,14,26,0.97)"
else:
    BG         = "linear-gradient(135deg,#f0f4ff 0%,#fafbff 50%,#f5f0ff 100%)"
    CARD_BG    = "rgba(255,255,255,0.92)"
    CARD_BOR   = "rgba(66,153,225,0.25)"
    TEXT       = "#1a202c"
    MUTED      = "#4a5568"
    ACCENT     = "#3182ce"
    ACCENT2    = "#d53f8c"
    ACCENT3    = "#38a169"
    ACCENT_RED = "#e53e3e"
    ACCENT_PUR = "#7C3AED"
    PLOT_BG    = "#ffffff"
    PAPER_BG   = "#f8faff"
    GRID_CLR   = "rgba(0,0,0,0.06)"
    BADGE_BG   = "rgba(49,130,206,0.10)"
    SECTION_BG = "rgba(49,130,206,0.05)"
    WARN_BG    = "rgba(221,107,32,0.08)"
    WARN_BOR   = "rgba(221,107,32,0.35)"
    SUCCESS_BG = "rgba(56,161,105,0.08)"
    SUCCESS_BOR= "rgba(56,161,105,0.35)"
    DANGER_BG  = "rgba(229,62,62,0.08)"
    DANGER_BOR = "rgba(229,62,62,0.35)"
    SIDEBAR_BG = "rgba(240,244,255,0.97)"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font_color=TEXT,
    font_family="Inter, sans-serif",
    xaxis=dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False,
               linecolor=CARD_BOR, tickfont_color=MUTED),
    yaxis=dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False,
               linecolor=CARD_BOR, tickfont_color=MUTED),
    legend=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, borderwidth=1, font_color=TEXT),
    margin=dict(l=50, r=30, t=60, b=50),
    hoverlabel=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, font_color=TEXT),
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"], .main {{
    background: {BG} !important;
    font-family: 'Inter', 'Poppins', sans-serif;
    color: {TEXT} !important;
}}
@keyframes gradientMove {{
    0%   {{ background-position: 0% 50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
    border-right: 1px solid {CARD_BOR};
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
.block-container {{ padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1400px; }}
p, div, span, label {{ color: {TEXT}; }}

/* ── Hero ── */
.fp-hero {{
    text-align: center; padding: 36px 40px; border-radius: 22px;
    background: rgba(255,255,255,0.05); backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.12); margin-bottom: 28px;
}}
.fp-hero-title {{ font-size: 46px; font-weight: 800; color: {TEXT}; letter-spacing: -1px; font-family: 'Syne', sans-serif; }}
.fp-hero-sub   {{ font-size: 17px; color: {MUTED}; margin-top: 6px; }}

</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  SESSION STATE — unified across all milestones
# ════════════════════════════════════════════════════════════════════════════════
_all_keys = {
    # M1
    "original_df": None, "df": None, "preprocessing": None,
    "_cleaner_filename": None,
    # M2/M3 shared data
    "data_loaded": False,
    "daily": None, "hourly_s": None, "hourly_i": None,
    "sleep": None, "hr": None, "hr_minute": None, "master": None,
    # M2
    "tsfresh_done": False, "prophet_done": False, "cluster_done": False,
    "features": None, "features_norm": None,
    "prophet_hr_fcst": None, "prophet_hr_df": None,
    "prophet_steps_fcst": None, "prophet_steps_df": None,
    "prophet_sleep_fcst": None, "prophet_sleep_df": None,
    "prophet_method": None,
    "cluster_df": None, "kmeans_labels": None, "dbscan_labels": None,
    "pca_df": None, "tsne_df": None, "profile": None,
    "inertias": None, "k_range": None, "pca_var": None,
    "n_clusters": None, "n_noise": None,
    # M3
    "anomaly_done": False, "simulation_done": False,
    "anom_hr": None, "anom_steps": None, "anom_sleep": None,
    "sim_results": None,
}
for _k, _v in _all_keys.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ════════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════
def sec(icon, title, badge=None):
    badge_html = f'<span class="sec-badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="sec-header">
      <div class="sec-icon">{icon}</div>
      <p class="sec-title">{title}</p>
      {badge_html}
    </div>""", unsafe_allow_html=True)

def step_pill(n, label):
    st.markdown(f'<div class="step-pill">◆ Step {n} &nbsp;·&nbsp; {label}</div>', unsafe_allow_html=True)

def screenshot_badge(ref):
    st.markdown(f'<div class="screenshot-badge">📸 Screenshot · {ref}</div>', unsafe_allow_html=True)

def anom_tag(label):
    st.markdown(f'<div class="anom-tag">🚨 {label}</div>', unsafe_allow_html=True)

def ui_success(msg): st.markdown(f'<div class="alert-success">✅ {msg}</div>', unsafe_allow_html=True)
def ui_warn(msg):    st.markdown(f'<div class="alert-warn">⚠️ {msg}</div>', unsafe_allow_html=True)
def ui_info(msg):    st.markdown(f'<div class="alert-info">ℹ️ {msg}</div>', unsafe_allow_html=True)
def ui_danger(msg):  st.markdown(f'<div class="alert-danger">🚨 {msg}</div>', unsafe_allow_html=True)

def metrics_html(*items, red_indices=None):
    red_indices = red_indices or []
    html = '<div class="metric-grid">'
    for i, (val, label) in enumerate(items):
        val_class = "metric-val metric-val-red" if i in red_indices else "metric-val"
        html += (f'<div class="metric-card">'
                 f'<div class="{val_class}">{val}</div>'
                 f'<div class="metric-label">{label}</div></div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def apply_plotly_theme(fig, title=""):
    fig.update_layout(**PLOTLY_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font_color=TEXT, font_size=14,
                                     font_family="Syne, sans-serif"))
    return fig

def dropdown_header(icon, title, desc):
    st.markdown(f"""
    <div class="dropdown-header">
      <div class="dropdown-header-icon">{icon}</div>
      <div>
        <div class="dropdown-header-title">{title}</div>
        <div class="dropdown-header-desc">{desc}</div>
      </div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  ANOMALY DETECTION FUNCTIONS  (M3)
# ════════════════════════════════════════════════════════════════════════════════
def detect_hr_anomalies(master, hr_high=100, hr_low=50, residual_sigma=2.0):
    df = master[["Id","Date","AvgHR"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    hr_daily = df.groupby("Date")["AvgHR"].mean().reset_index()
    hr_daily.columns = ["Date","AvgHR"]
    hr_daily = hr_daily.sort_values("Date")
    hr_daily["thresh_high"]  = hr_daily["AvgHR"] > hr_high
    hr_daily["thresh_low"]   = hr_daily["AvgHR"] < hr_low
    hr_daily["rolling_med"]  = hr_daily["AvgHR"].rolling(3, center=True, min_periods=1).median()
    hr_daily["residual"]     = hr_daily["AvgHR"] - hr_daily["rolling_med"]
    resid_std                = hr_daily["residual"].std()
    hr_daily["resid_anomaly"]= hr_daily["residual"].abs() > (residual_sigma * resid_std)
    hr_daily["is_anomaly"]   = hr_daily["thresh_high"] | hr_daily["thresh_low"] | hr_daily["resid_anomaly"]
    def reason(row):
        r = []
        if row["thresh_high"]:   r.append(f"HR>{hr_high}")
        if row["thresh_low"]:    r.append(f"HR<{hr_low}")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    hr_daily["reason"] = hr_daily.apply(reason, axis=1)
    return hr_daily

def detect_steps_anomalies(master, steps_low=500, steps_high=25000, residual_sigma=2.0):
    df = master[["Date","TotalSteps"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    sd = df.groupby("Date")["TotalSteps"].mean().reset_index().sort_values("Date")
    sd["thresh_low"]    = sd["TotalSteps"] < steps_low
    sd["thresh_high"]   = sd["TotalSteps"] > steps_high
    sd["rolling_med"]   = sd["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    sd["residual"]      = sd["TotalSteps"] - sd["rolling_med"]
    sd["resid_anomaly"] = sd["residual"].abs() > (sd["residual"].std() * residual_sigma)
    sd["is_anomaly"]    = sd["thresh_low"] | sd["thresh_high"] | sd["resid_anomaly"]
    def reason(row):
        r = []
        if row["thresh_low"]:    r.append(f"Steps<{steps_low}")
        if row["thresh_high"]:   r.append(f"Steps>{steps_high}")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    sd["reason"] = sd.apply(reason, axis=1)
    return sd

def detect_sleep_anomalies(master, sleep_low=60, sleep_high=600, residual_sigma=2.0):
    df = master[["Date","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    sl = df.groupby("Date")["TotalSleepMinutes"].mean().reset_index().sort_values("Date")
    sl["thresh_low"]    = (sl["TotalSleepMinutes"] > 0) & (sl["TotalSleepMinutes"] < sleep_low)
    sl["thresh_high"]   = sl["TotalSleepMinutes"] > sleep_high
    sl["no_data"]       = sl["TotalSleepMinutes"] == 0
    sl["rolling_med"]   = sl["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
    sl["residual"]      = sl["TotalSleepMinutes"] - sl["rolling_med"]
    sl["resid_anomaly"] = sl["residual"].abs() > (sl["residual"].std() * residual_sigma)
    sl["is_anomaly"]    = sl["thresh_low"] | sl["thresh_high"] | sl["resid_anomaly"]
    def reason(row):
        r = []
        if row["no_data"]:       r.append("No device worn")
        if row["thresh_low"]:    r.append(f"Sleep<{sleep_low}min")
        if row["thresh_high"]:   r.append(f"Sleep>{sleep_high}min")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    sl["reason"] = sl.apply(reason, axis=1)
    return sl

def simulate_accuracy(master, n_inject=10):
    np.random.seed(42)
    df = master[["Date","AvgHR","TotalSteps","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df_daily = df.groupby("Date").mean().reset_index().sort_values("Date")
    results = {}
    # HR
    hr_sim = df_daily[["Date","AvgHR"]].copy().reset_index(drop=True)
    idx = np.random.choice(len(hr_sim), n_inject, replace=False)
    hr_sim.loc[idx, "AvgHR"] = np.random.choice(
        [115,120,125,35,40,45,118,130,38,42], n_inject, replace=True)
    hr_sim["rolling_med"] = hr_sim["AvgHR"].rolling(3,center=True,min_periods=1).median()
    hr_sim["residual"]    = hr_sim["AvgHR"] - hr_sim["rolling_med"]
    hr_sim["detected"]    = ((hr_sim["AvgHR"]>100)|(hr_sim["AvgHR"]<50)|
                             (hr_sim["residual"].abs()>2*hr_sim["residual"].std()))
    tp = int(hr_sim["detected"].iloc[idx].sum())
    results["Heart Rate"] = {"injected":n_inject,"detected":int(tp),
                              "accuracy":round(tp/n_inject*100,1)}
    # Steps
    st_sim = df_daily[["Date","TotalSteps"]].copy().reset_index(drop=True)
    idx2 = np.random.choice(len(st_sim), n_inject, replace=False)
    st_sim.loc[idx2,"TotalSteps"] = np.random.choice(
        [50,100,150,30000,35000,28000,80,200,31000,29000], n_inject, replace=True)
    st_sim["rolling_med"] = st_sim["TotalSteps"].rolling(3,center=True,min_periods=1).median()
    st_sim["residual"]    = st_sim["TotalSteps"] - st_sim["rolling_med"]
    st_sim["detected"]    = ((st_sim["TotalSteps"]<500)|(st_sim["TotalSteps"]>25000)|
                             (st_sim["residual"].abs()>2*st_sim["residual"].std()))
    tp2 = int(st_sim["detected"].iloc[idx2].sum())
    results["Steps"] = {"injected":n_inject,"detected":int(tp2),
                         "accuracy":round(tp2/n_inject*100,1)}
    # Sleep
    sl_sim = df_daily[["Date","TotalSleepMinutes"]].copy().reset_index(drop=True)
    idx3 = np.random.choice(len(sl_sim), n_inject, replace=False)
    sl_sim.loc[idx3,"TotalSleepMinutes"] = np.random.choice(
        [10,20,30,700,750,800,15,25,710,720], n_inject, replace=True)
    sl_sim["rolling_med"] = sl_sim["TotalSleepMinutes"].rolling(3,center=True,min_periods=1).median()
    sl_sim["residual"]    = sl_sim["TotalSleepMinutes"] - sl_sim["rolling_med"]
    sl_sim["detected"]    = (((sl_sim["TotalSleepMinutes"]>0)&(sl_sim["TotalSleepMinutes"]<60))|
                              (sl_sim["TotalSleepMinutes"]>600)|
                              (sl_sim["residual"].abs()>2*sl_sim["residual"].std()))
    tp3 = int(sl_sim["detected"].iloc[idx3].sum())
    results["Sleep"] = {"injected":n_inject,"detected":int(tp3),
                         "accuracy":round(tp3/n_inject*100,1)}
    results["Overall"] = round(np.mean([results[k]["accuracy"] for k in ["Heart Rate","Steps","Sleep"]]),1)
    return results


# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:20px 0 14px">
      <div style="font-size:2.6rem">💪</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;color:{ACCENT_PUR}">FitPulse</div>
      <div style="font-size:0.7rem;color:{MUTED};margin-top:2px;font-family:'JetBrains Mono',monospace">AI HEALTH ANALYTICS</div>
    </div>
    """, unsafe_allow_html=True)

    new_dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if new_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark
        st.rerun()

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:1rem 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.72rem;color:{MUTED};font-family:JetBrains Mono,monospace;letter-spacing:.1em">ML CONTROLS (M2)</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    kmeans_k   = st.slider("KMeans Clusters (K)",  2, 8, 3)
    dbscan_eps = st.slider("DBSCAN EPS",            1.0, 5.0, 2.2, 0.1)
    dbscan_min = st.slider("DBSCAN min_samples",    1, 10, 2)

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:1rem 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.72rem;color:{MUTED};font-family:JetBrains Mono,monospace;letter-spacing:.1em">ANOMALY THRESHOLDS (M3)</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    hr_high  = st.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180)
    hr_low   = st.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70)
    st_low   = st.number_input("Steps Low",        value=500, min_value=0,   max_value=2000)
    sl_low   = st.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120)
    sl_high  = st.number_input("Sleep High (min)", value=600, min_value=300, max_value=900)
    sigma    = st.slider("Residual σ threshold",   1.0, 4.0, 2.0, 0.5, key="sigma_slider")

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:1rem 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.72rem;color:{MUTED};font-family:JetBrains Mono,monospace;letter-spacing:.1em">PIPELINE STATUS</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    _dot = lambda ok: f'<span style="color:{ACCENT3}">●</span>' if ok else f'<span style="color:{MUTED}">○</span>'
    st.markdown(f"""
    <div style='font-size:.82rem;line-height:2.4;color:{TEXT}'>
        {_dot(st.session_state.original_df is not None)} 🧹 &nbsp;M1 · Data Cleaning<br>
        {_dot(st.session_state.data_loaded)}  🔧 &nbsp;M2 · Load &amp; Parse<br>
        {_dot(st.session_state.tsfresh_done)} 🧪 &nbsp;M2 · TSFresh Features<br>
        {_dot(st.session_state.prophet_done)} 📈 &nbsp;M2 · Prophet Forecast<br>
        {_dot(st.session_state.cluster_done)} 🤖 &nbsp;M2 · Clustering<br>
        {_dot(st.session_state.anomaly_done)} 🚨 &nbsp;M3 · Anomaly Detection<br>
        {_dot(st.session_state.simulation_done)} 🎯 &nbsp;M3 · Accuracy Simulation
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:1rem 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.7rem;color:{MUTED};text-align:center;font-family:JetBrains Mono,monospace">Real Fitbit Dataset<br>30 users · Mar–Apr 2016<br>Minute-level HR data</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  HERO
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="fp-hero">
  <div class="fp-hero-title">💪 FitPulse</div>
  <div class="fp-hero-sub">AI-Powered Health Analytics · All 3 Milestones in One App</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="ms-tab-bar">
  <div class="ms-tab">
    <div class="ms-tab-icon">🧹</div>
    <div class="ms-tab-title">Milestone 1</div>
    <div class="ms-tab-desc">Data Cleaning · Null checks · CSV preview</div>
  </div>
  <div class="ms-tab">
    <div class="ms-tab-icon">🧬</div>
    <div class="ms-tab-title">Milestone 2</div>
    <div class="ms-tab-desc">TSFresh · Prophet · KMeans · DBSCAN · PCA</div>
  </div>
  <div class="ms-tab">
    <div class="ms-tab-icon">🚨</div>
    <div class="ms-tab-title">Milestone 3</div>
    <div class="ms-tab-desc">Anomaly Detection · Plotly Charts · Accuracy Sim</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# ██  MILESTONE 1 · DATA CLEANING
# ════════════════════════════════════════════════════════════════════════════════
dropdown_header("🧹", "Milestone 1 · Data Cleaning",
                "Upload any CSV · Auto-clean nulls · Preview data · Before & after null check")

with st.expander("▼  Open Milestone 1 — Data Cleaning", expanded=False):
    sec("📂", "Upload Fitness CSV Data")

    uploaded_file = st.file_uploader("Upload a fitness device CSV", type=["csv"], key="m1_upload")

    if uploaded_file is not None:
        if (st.session_state.original_df is None or
                st.session_state.get("_cleaner_filename") != uploaded_file.name):
            st.session_state.original_df = pd.read_csv(uploaded_file)
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state["_cleaner_filename"] = uploaded_file.name
            st.session_state["preprocessing"] = None

        df_clean = st.session_state.df
        orig_df  = st.session_state.original_df

        ui_success("File uploaded successfully!")

        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Total Rows",    df_clean.shape[0])
        c2.metric("📂 Total Columns", df_clean.shape[1])
        c3.metric("⚠️ Null Values",   int(df_clean.isnull().sum().sum()))
        st.divider()

        b1, b2, b3 = st.columns(3)

        if b1.button("🧹 Clean Data", key="m1_btn_clean"):
            with st.spinner("Cleaning data…"):
                df_work = orig_df.copy()
                steps   = []
                for col in df_work.columns:
                    if "date" in col.lower():
                        df_work[col] = pd.to_datetime(df_work[col], errors="coerce")
                        df_work[col] = df_work[col].ffill().bfill()
                        steps.append(f"✔ '{col}' → datetime · missing filled via ffill/bfill")
                num_cols = df_work.select_dtypes(include=["number"]).columns
                df_work[num_cols] = df_work[num_cols].ffill().bfill()
                steps.append("✔ Numeric columns filled via Forward / Backward Fill")
                obj_cols = df_work.select_dtypes(include=["object"]).columns
                df_work[obj_cols] = df_work[obj_cols].fillna("No Workout")
                steps.append("✔ Categorical columns filled with 'No Workout'")
                st.session_state.df = df_work
                st.session_state["preprocessing"] = steps
            ui_success("Data cleaned successfully!")

        if b2.button("👀 Show Data", key="m1_btn_show"):
            st.subheader("📋 Data Preview")
            st.dataframe(st.session_state.df, use_container_width=True)

        if b3.button("🔍 Check Null Values", key="m1_btn_nulls"):
            st.subheader("📊 Before Cleaning")
            null_before = orig_df.isnull().sum()
            st.dataframe(pd.DataFrame({"Null Count": null_before,
                                       "Null %": (null_before/len(orig_df)*100).round(2)}))
            st.bar_chart(null_before)
            st.divider()
            st.subheader("📊 After Cleaning")
            null_after = st.session_state.df.isnull().sum()
            st.dataframe(pd.DataFrame({"Null Count": null_after,
                                       "Null %": (null_after/len(st.session_state.df)*100).round(2)}))
            st.bar_chart(null_after)
            if null_after.sum() == 0:
                ui_success("All null values removed!")
            else:
                ui_warn("Some null values still remain.")

        if st.session_state.get("preprocessing"):
            st.divider()
            st.subheader("⚙️ Pre-Processing Steps Applied")
            log_html = "<div class='fp-log'>" + "<br>".join(st.session_state["preprocessing"]) + "</div>"
            st.markdown(log_html, unsafe_allow_html=True)
    else:
        ui_info("Upload any fitness CSV file above to start Milestone 1 cleaning.")

st.markdown("<br>", unsafe_allow_html=True)
st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# ██  MILESTONE 2 · ML PIPELINE
# ════════════════════════════════════════════════════════════════════════════════
dropdown_header("🧬", "Milestone 2 · ML Anomaly Detection Pipeline",
                "TSFresh Features · Prophet Forecasting · KMeans & DBSCAN · PCA · t-SNE")

with st.expander("▼  Open Milestone 2 — ML Pipeline", expanded=False):

    # ── Progress bar ────────────────────────────────────────────────────────────
    _done_count = sum([bool(st.session_state.data_loaded),
                       bool(st.session_state.tsfresh_done),
                       bool(st.session_state.prophet_done),
                       bool(st.session_state.cluster_done)])
    _pct = int(_done_count / 4 * 100)
    _badges = "".join(
        f'<span class="step-pill" style="background:{SUCCESS_BG};color:{ACCENT3}">✓ {s}</span>'
        if ok else f'<span class="step-pill">{s}</span>'
        for s, ok in [("📂 Data Load", bool(st.session_state.data_loaded)),
                      ("🧪 TSFresh",   bool(st.session_state.tsfresh_done)),
                      ("📈 Prophet",   bool(st.session_state.prophet_done)),
                      ("🤖 Clustering",bool(st.session_state.cluster_done))]
    )
    st.markdown(f"""
    <div class="fp-card">
      <div style="display:flex;justify-content:space-between;font-size:.78rem;color:{MUTED};margin-bottom:6px">
        <span>M2 PIPELINE PROGRESS</span>
        <span style="color:{ACCENT_PUR};font-weight:700">{_pct}%</span>
      </div>
      <div class="fp-progress"><div class="fp-progress-fill" style="width:{_pct}%"></div></div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">{_badges}</div>
      {"<div style='margin-top:10px;font-size:.82rem;color:"+ACCENT3+";font-weight:700'>🎉 M2 Pipeline 100% complete!</div>" if _pct==100 else ""}
    </div>""", unsafe_allow_html=True)

    # ── Section 2A: File Upload ─────────────────────────────────────────────────
    sec("📂", "Upload Fitbit Dataset Files (5 CSVs)")
    ui_info("<b>All at once:</b> Ctrl+click / ⌘+click to select multiple files.")

    REQUIRED = {
        "dailyActivity":     {"emoji":"🏃","label":"Daily Activity",    "hint":"dailyActivity_merged.csv"},
        "hourlySteps":       {"emoji":"👣","label":"Hourly Steps",       "hint":"hourlySteps_merged.csv"},
        "hourlyIntensities": {"emoji":"⚡","label":"Hourly Intensities", "hint":"hourlyIntensities_merged.csv"},
        "minuteSleep":       {"emoji":"💤","label":"Minute Sleep",        "hint":"minuteSleep_merged.csv"},
        "heartrate":         {"emoji":"❤️","label":"Heart Rate",          "hint":"heartrate_seconds_merged.csv"},
    }
    file_map = {}

    bulk_files = st.file_uploader("📦 Upload all files at once", type="csv",
                                   accept_multiple_files=True, key="m2_bulk")
    if bulk_files:
        for f in bulk_files:
            n = f.name.lower()
            if   "dailyactivity"     in n: file_map["dailyActivity"]     = f
            elif "hourlysteps"       in n: file_map["hourlySteps"]       = f
            elif "hourlyintensities" in n: file_map["hourlyIntensities"] = f
            elif "minutesleep"       in n: file_map["minuteSleep"]       = f
            elif "heartrate"         in n: file_map["heartrate"]         = f

    st.markdown(f'<div style="text-align:center;color:{MUTED};font-size:.78rem;margin:8px 0">— or upload individually —</div>', unsafe_allow_html=True)
    solo_cols = st.columns(5)
    for i, (key, meta) in enumerate(REQUIRED.items()):
        with solo_cols[i]:
            st.markdown(f'<div style="font-size:.76rem;font-weight:600;color:{TEXT}">{meta["emoji"]} {meta["label"]}</div>'
                        f'<div style="font-size:.64rem;color:{MUTED};margin-bottom:4px">{meta["hint"]}</div>',
                        unsafe_allow_html=True)
            f = st.file_uploader(meta["label"], type="csv",
                                 key=f"m2_solo_{key}", label_visibility="collapsed")
            if f is not None:
                file_map[key] = f

    n_found = len(file_map)
    sc = st.columns(5)
    for i, (key, meta) in enumerate(REQUIRED.items()):
        ok = key in file_map
        with sc[i]:
            border = SUCCESS_BOR if ok else WARN_BOR
            bg     = SUCCESS_BG  if ok else WARN_BG
            st.markdown(f'<div style="background:{bg};border:1px solid {border};border-radius:10px;padding:10px 14px;text-align:center">'
                        f'<div>{meta["emoji"]} {"✅" if ok else "⬜"}</div>'
                        f'<div style="font-size:.72rem;color:{TEXT};margin-top:4px">{meta["label"]}</div></div>',
                        unsafe_allow_html=True)

    metrics_html((n_found,"Detected"),(5-n_found,"Missing"),("✓ Ready" if n_found==5 else "⚠ Incomplete","Status"))
    if n_found == 5:
        ui_success("All 5 files detected — ready to run pipeline!")
    elif n_found > 0:
        missing = [m["label"] for k,m in REQUIRED.items() if k not in file_map]
        ui_warn(f"Still missing: {', '.join(missing)}")
    else:
        ui_info("Upload the 5 Fitbit CSV files above.")

    # ── Section 2B: Load & Parse ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sec("🔧", "Load & Parse the Data")
    step_pill(1, "Time normalization · Master DataFrame assembly")

    if st.button("🔧 Load & Parse the Data", disabled=(n_found < 5), key="m2_btn_load"):
        with st.spinner("Loading and parsing all datasets…"):
            try:
                daily    = pd.read_csv(file_map["dailyActivity"])
                hourly_s = pd.read_csv(file_map["hourlySteps"])
                hourly_i = pd.read_csv(file_map["hourlyIntensities"])
                sleep    = pd.read_csv(file_map["minuteSleep"])
                hr       = pd.read_csv(file_map["heartrate"])

                for _df in [daily, hourly_s, hourly_i, sleep, hr]:
                    _df.columns = _df.columns.str.strip()

                daily["ActivityDate"]    = pd.to_datetime(daily["ActivityDate"])
                hourly_s["ActivityHour"] = pd.to_datetime(hourly_s["ActivityHour"])
                hourly_i["ActivityHour"] = pd.to_datetime(hourly_i["ActivityHour"])

                sleep_time_col = next(
                    (c for c in sleep.columns if "date" in c.lower() or "time" in c.lower()),
                    sleep.columns[1])
                sleep[sleep_time_col] = pd.to_datetime(sleep[sleep_time_col])
                sleep_val_col = next(
                    (c for c in sleep.columns if "value" in c.lower() or "stage" in c.lower()), None)
                if sleep_val_col is None:
                    num_s = sleep.select_dtypes(include="number").columns.tolist()
                    sleep_val_col = next((c for c in num_s if c != "Id"), num_s[0])

                hr_time_col = next(
                    (c for c in hr.columns if "time" in c.lower() or "date" in c.lower()),
                    hr.columns[1])
                hr_val_col = next((c for c in hr.columns if "value" in c.lower()), None)
                if hr_val_col is None:
                    num_h = hr.select_dtypes(include="number").columns.tolist()
                    hr_val_col = next((c for c in num_h if c != "Id"), num_h[0])
                hr[hr_time_col] = pd.to_datetime(hr[hr_time_col])
                hr = hr.rename(columns={hr_time_col:"Time", hr_val_col:"Value"})

                hr_minute = (hr.groupby(["Id", pd.Grouper(key="Time", freq="1min")])["Value"]
                             .mean().reset_index())
                hr_minute.columns = ["Id","Time","HeartRate"]
                hr_minute = hr_minute.dropna()
                hr_minute["Date"] = hr_minute["Time"].dt.date

                hr_daily = (hr_minute.groupby(["Id","Date"])["HeartRate"]
                            .agg(["mean","max","min","std"]).reset_index()
                            .rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"}))

                sleep["Date"] = sleep[sleep_time_col].dt.date
                sleep_agg = sleep.groupby(["Id","Date"])[sleep_val_col].count().reset_index()
                sleep_agg.columns = ["Id","Date","TotalSleepMinutes"]

                daily["Date"] = daily["ActivityDate"].dt.date
                master = daily.copy()
                master = master.rename(columns={"ActivityDate": "ActivityDateOrig"})
                keep_cols = [c for c in ["Id","Date","TotalSteps","Calories","VeryActiveMinutes",
                                         "FairlyActiveMinutes","LightlyActiveMinutes","SedentaryMinutes"]
                             if c in master.columns]
                master = master[keep_cols].copy()
                master = master.merge(hr_daily,   on=["Id","Date"], how="left")
                master = master.merge(sleep_agg,  on=["Id","Date"], how="left")
                master["Date"] = pd.to_datetime(master["Date"])
                master["TotalSleepMinutes"] = master["TotalSleepMinutes"].fillna(0)
                for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
                    if col in master.columns:
                        master[col] = master.groupby("Id")[col].transform(
                            lambda x: x.fillna(x.median()))

                for k, v in [("daily",daily),("hourly_s",hourly_s),("hourly_i",hourly_i),
                              ("sleep",sleep),("hr",hr),("hr_minute",hr_minute),("master",master)]:
                    st.session_state[k] = v
                st.session_state.data_loaded = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error during loading: {e}")

    if st.session_state.data_loaded:
        master    = st.session_state.master
        hr_minute = st.session_state.hr_minute
        daily     = st.session_state.daily

        ui_success(f"Master DataFrame ready — {master.shape[0]} rows · {master['Id'].nunique()} users")

        with st.expander("◆ Null Value Check"):
            frames = {"dailyActivity":st.session_state.daily,"hourlySteps":st.session_state.hourly_s,
                      "hourlyIntensities":st.session_state.hourly_i,"minuteSleep":st.session_state.sleep,
                      "heartrate":st.session_state.hr}
            ncols = st.columns(5)
            for i, (nm, df_) in enumerate(frames.items()):
                nulls = df_.isnull().sum().sum()
                with ncols[i]:
                    col_c = ACCENT3 if nulls==0 else ACCENT_RED
                    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{col_c};font-size:1.4rem">{int(nulls)}</div>'
                                f'<div class="metric-label">{nm}</div>'
                                f'<div style="font-size:.68rem;color:{MUTED}">{len(df_):,} rows</div></div>',
                                unsafe_allow_html=True)

        with st.expander("◆ Cleaned Dataset Preview"):
            show = [c for c in ["Id","Date","TotalSteps","Calories","AvgHR",
                                 "TotalSleepMinutes","VeryActiveMinutes","SedentaryMinutes"]
                    if c in master.columns]
            st.dataframe(master[show].head(20), use_container_width=True)

    # ── Section 2C: TSFresh ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sec("🧪", "TSFresh Feature Extraction")
    step_pill(2, "Statistical features from minute-level HR · MinimalFCParameters")

    if st.button("🧪 Run TSFresh Feature Extraction",
                 disabled=not st.session_state.data_loaded, key="m2_btn_tsfresh"):
        try:
            from tsfresh import extract_features
            from tsfresh.feature_extraction import MinimalFCParameters
            from sklearn.preprocessing import MinMaxScaler

            hr_minute = st.session_state.hr_minute
            with st.spinner("Extracting TSFresh features (~30 s)…"):
                ts_input = hr_minute[["Id","Time","HeartRate"]].copy()
                ts_input["time_idx"] = ts_input.groupby("Id").cumcount()
                ts_df = ts_input[["Id","time_idx","HeartRate"]].rename(
                    columns={"Id":"id","time_idx":"time","HeartRate":"value"})
                features = extract_features(
                    ts_df, column_id="id", column_sort="time", column_value="value",
                    default_fc_parameters=MinimalFCParameters(), disable_progressbar=True)
                features = features.dropna(axis=1, how="all").dropna(axis=0)
                scaler = MinMaxScaler()
                features_norm = pd.DataFrame(
                    scaler.fit_transform(features),
                    index=features.index, columns=features.columns)
                st.session_state.features      = features
                st.session_state.features_norm = features_norm
                st.session_state.tsfresh_done  = True
                st.rerun()
        except ImportError:
            st.error("❌ tsfresh not installed. Run: `pip install tsfresh`")
        except Exception as e:
            st.error(f"❌ TSFresh error: {e}")

    if st.session_state.tsfresh_done and st.session_state.features is not None:
        features      = st.session_state.features
        features_norm = st.session_state.features_norm
        ui_success(f"TSFresh — {len(features)} users × {features.shape[1]} features")

        with st.expander("◆ Feature Matrix Heatmap", expanded=True):
            fig, ax = plt.subplots(figsize=(14, max(4, len(features)*0.9)))
            fig.patch.set_facecolor("#0F0D1A"); ax.set_facecolor("#0F0D1A")
            sc = features_norm.copy()
            sc.columns = [c.split("__")[-1][:20] for c in sc.columns]
            sc.index   = [str(i)[-6:] for i in sc.index]
            sns.heatmap(sc, ax=ax, cmap="magma", annot=True, fmt=".2f",
                        linewidths=0.4, linecolor="#2D2B45", cbar_kws={"shrink":0.7})
            ax.set_title("TSFresh Feature Matrix (Normalized 0–1)", color="#E2E8F0", fontsize=11)
            ax.tick_params(colors="#94A3B8", labelsize=7)
            plt.xticks(rotation=45, ha="right", fontsize=7)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with st.expander("◆ Raw Feature Matrix"):
            st.dataframe(features.round(4), use_container_width=True)

    # ── Section 2D: Prophet ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sec("📈", "Prophet Trend Forecasting")
    step_pill(3, "Additive models · Weekly seasonality · 80% CI · 30-day forecasts")

    def _fit_prophet_safe(df_input, periods=30):
        import logging
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        try:
            from prophet import Prophet
            m = Prophet(weekly_seasonality=True, daily_seasonality=False,
                        interval_width=0.80, uncertainty_samples=0)
            m.fit(df_input)
            future = m.make_future_dataframe(periods=periods)
            fcst   = m.predict(future)
            if (fcst["yhat_upper"] == fcst["yhat"]).all():
                sigma_ = df_input["y"].std() * 0.5
                fcst["yhat_upper"] = fcst["yhat"] + 1.28 * sigma_
                fcst["yhat_lower"] = fcst["yhat"] - 1.28 * sigma_
            return fcst, "prophet"
        except Exception:
            pass
        # Fallback: linear + weekly seasonality
        df = df_input.copy().reset_index(drop=True)
        df["t"] = (df["ds"] - df["ds"].min()).dt.days.astype(float)
        X_seas = np.column_stack([np.ones(len(df)), df["t"],
                                  np.sin(2*np.pi*df["t"]/7),
                                  np.cos(2*np.pi*df["t"]/7)])
        try:
            sc = np.linalg.lstsq(X_seas, df["y"].values, rcond=None)[0]
        except Exception:
            sc = np.array([df["y"].mean(), 0., 0., 0.])
        sigma_ = df["y"].std()
        future_dates = pd.date_range(start=df["ds"].max()+pd.Timedelta(days=1), periods=periods)
        all_dates = pd.concat([df["ds"], pd.Series(future_dates)]).reset_index(drop=True)
        t_all  = (all_dates - df["ds"].min()).dt.days.astype(float)
        X_all  = np.column_stack([np.ones(len(t_all)), t_all,
                                  np.sin(2*np.pi*t_all/7), np.cos(2*np.pi*t_all/7)])
        yhat = X_all @ sc
        fcst = pd.DataFrame({"ds":all_dates,"yhat":yhat,
                             "yhat_upper":yhat+1.28*sigma_,"yhat_lower":yhat-1.28*sigma_})
        return fcst, "fallback"

    def _prophet_fig(df_hist, fcst, title, color, ylabel):
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#0F0D1A"); ax.set_facecolor("#13111F")
        cutoff = df_hist["ds"].max()
        ax.fill_between(fcst["ds"], fcst["yhat_lower"], fcst["yhat_upper"],
                        color=color, alpha=0.18, label="80% CI")
        ax.plot(fcst["ds"], fcst["yhat"], color=color, lw=1.5, label="Forecast")
        ax.scatter(df_hist["ds"], df_hist["y"], color="white", s=14, zorder=5, alpha=0.7, label="Actual")
        ax.axvline(cutoff, color="#64748B", linestyle="--", lw=1, alpha=0.7)
        ax.set_title(title, color="#E2E8F0", fontsize=11)
        ax.set_xlabel("Date", color="#94A3B8", fontsize=8)
        ax.set_ylabel(ylabel, color="#94A3B8", fontsize=8)
        ax.tick_params(colors="#94A3B8", labelsize=7)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom","left"]: ax.spines[sp].set_color("#2D2B45")
        ax.legend(fontsize=7, facecolor="#1E1B2E", labelcolor="#CBD5E1")
        ax.xaxis.set_tick_params(rotation=30)
        plt.tight_layout(); return fig

    if st.button("📈 Run Prophet Trend Forecasting",
                 disabled=not st.session_state.data_loaded, key="m2_btn_prophet"):
        master    = st.session_state.master
        hr_minute = st.session_state.hr_minute
        with st.spinner("Fitting forecast models…"):
            try:
                hr_df = hr_minute.groupby("Date")["HeartRate"].mean().reset_index()
                hr_df.columns = ["ds","y"]
                hr_df["ds"] = pd.to_datetime(hr_df["ds"])
                hr_df = hr_df.dropna().sort_values("ds")
                fcst_hr, meth = _fit_prophet_safe(hr_df)

                steps_df = master.groupby("Date")["TotalSteps"].mean().reset_index()
                steps_df.columns = ["ds","y"]
                steps_df["ds"] = pd.to_datetime(steps_df["ds"])
                steps_df = steps_df.dropna().sort_values("ds")
                fcst_steps, _ = _fit_prophet_safe(steps_df)

                fcst_sleep = None; sleep_df_p = None
                if "TotalSleepMinutes" in master.columns:
                    sleep_df_p = master.groupby("Date")["TotalSleepMinutes"].mean().reset_index()
                    sleep_df_p.columns = ["ds","y"]
                    sleep_df_p["ds"] = pd.to_datetime(sleep_df_p["ds"])
                    sleep_df_p = sleep_df_p.dropna().sort_values("ds")
                    if len(sleep_df_p) >= 3:
                        fcst_sleep, _ = _fit_prophet_safe(sleep_df_p)

                for k, v in [("prophet_hr_fcst",fcst_hr),("prophet_hr_df",hr_df),
                              ("prophet_steps_fcst",fcst_steps),("prophet_steps_df",steps_df),
                              ("prophet_sleep_fcst",fcst_sleep),("prophet_sleep_df",sleep_df_p),
                              ("prophet_method",meth)]:
                    st.session_state[k] = v
                st.session_state.prophet_done = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ Forecasting error: {e}")

    if st.session_state.prophet_done:
        _engine = "Prophet (Stan)" if st.session_state.prophet_method=="prophet" else "Linear Trend + Weekly Seasonality (fallback)"
        ui_success(f"3 forecast models ready · engine: {_engine}")

        with st.expander("◆ Heart Rate Forecast (30-day)", expanded=True):
            fig = _prophet_fig(st.session_state.prophet_hr_df, st.session_state.prophet_hr_fcst,
                               "Prophet — Heart Rate (30-day · 80% CI)", "#F472B6", "Heart Rate (bpm)")
            st.pyplot(fig, use_container_width=True); plt.close()

        with st.expander("◆ Steps & Sleep Forecast (30-day)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                fig2 = _prophet_fig(st.session_state.prophet_steps_df,
                                    st.session_state.prophet_steps_fcst,
                                    "Prophet — Daily Steps", "#34D399", "Steps")
                st.pyplot(fig2, use_container_width=True); plt.close()
            with c2:
                if st.session_state.prophet_sleep_fcst is not None:
                    fig3 = _prophet_fig(st.session_state.prophet_sleep_df,
                                        st.session_state.prophet_sleep_fcst,
                                        "Prophet — Sleep Minutes", "#A78BFA", "Sleep (min)")
                    st.pyplot(fig3, use_container_width=True); plt.close()

        with st.expander("◆ Correlation Heatmap", expanded=False):
            master = st.session_state.master
            corr_cols = [c for c in ["TotalSteps","Calories","VeryActiveMinutes",
                                     "SedentaryMinutes","AvgHR","TotalSleepMinutes"]
                         if c in master.columns]
            corr_mat = master[corr_cols].corr()
            fig_c, ax_c = plt.subplots(figsize=(8,6))
            fig_c.patch.set_facecolor("#0F0D1A"); ax_c.set_facecolor("#0F0D1A")
            mask = np.triu(np.ones_like(corr_mat, dtype=bool))
            sns.heatmap(corr_mat, ax=ax_c, mask=mask, cmap="coolwarm",
                        annot=True, fmt=".2f", vmin=-1, vmax=1,
                        linewidths=0.5, linecolor="#2D2B45", cbar_kws={"shrink":0.8})
            ax_c.set_title("Feature Correlation Matrix", color="#E2E8F0", fontsize=11)
            ax_c.tick_params(colors="#94A3B8", labelsize=8)
            plt.xticks(rotation=30, ha="right"); plt.tight_layout()
            st.pyplot(fig_c, use_container_width=True); plt.close()

    # ── Section 2E: Clustering ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sec("🤖", f"Clustering — KMeans + DBSCAN + PCA + t-SNE")
    step_pill(4, f"Activity-based user segmentation · K={kmeans_k} · eps={dbscan_eps:.1f}")

    if st.button("🤖 Run Clustering (KMeans + DBSCAN + PCA + t-SNE)",
                 disabled=not st.session_state.data_loaded, key="m2_btn_cluster"):
        try:
            from sklearn.cluster import KMeans, DBSCAN
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            from sklearn.preprocessing import StandardScaler

            master = st.session_state.master
            with st.spinner("Running clustering…"):
                feat_cols = [c for c in ["TotalSteps","Calories","VeryActiveMinutes",
                                         "SedentaryMinutes","TotalSleepMinutes"]
                             if c in master.columns]
                cluster_features = master.groupby("Id")[feat_cols].mean().reset_index()
                X_raw = cluster_features[feat_cols].fillna(0).values
                X = StandardScaler().fit_transform(X_raw)

                km = KMeans(n_clusters=kmeans_k, random_state=42, n_init=10)
                km_labels = km.fit_predict(X)
                cluster_features["KMeans_Cluster"] = km_labels

                db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min)
                db_labels = db.fit_predict(X)
                cluster_features["DBSCAN_Cluster"] = db_labels

                inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_
                            for k in range(1, min(10, len(X)+1))]
                k_range = list(range(1, min(10, len(X)+1)))

                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                pca_df = pd.DataFrame(X_pca, columns=["PC1","PC2"])
                pca_df["KMeans"] = km_labels; pca_df["DBSCAN"] = db_labels

                perp = min(30, max(2, len(X)-1))
                try:
                    import sklearn as _sk
                    _ver = tuple(int(x) for x in _sk.__version__.split(".")[:2])
                    _kw  = {"max_iter":1000} if _ver >= (1,5) else {"n_iter":1000}
                except Exception:
                    _kw = {"max_iter":1000}
                X_tsne = TSNE(n_components=2, random_state=42, perplexity=perp, **_kw).fit_transform(X)
                tsne_df = pd.DataFrame(X_tsne, columns=["tSNE1","tSNE2"])
                tsne_df["KMeans"] = km_labels; tsne_df["DBSCAN"] = db_labels

                profile    = cluster_features.groupby("KMeans_Cluster")[feat_cols].mean().round(1)
                n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
                n_noise    = list(db_labels).count(-1)

                for k, v in [("cluster_df",cluster_features),("kmeans_labels",km_labels),
                              ("dbscan_labels",db_labels),("pca_df",pca_df),("tsne_df",tsne_df),
                              ("profile",profile),("inertias",inertias),("k_range",k_range),
                              ("pca_var",pca.explained_variance_ratio_),
                              ("n_clusters",n_clusters),("n_noise",n_noise)]:
                    st.session_state[k] = v
                st.session_state.cluster_done = True
                st.rerun()
        except Exception as e:
            st.error(f"❌ Clustering error: {e}")

    if st.session_state.cluster_done and st.session_state.pca_df is not None:
        cluster_df = st.session_state.cluster_df
        km_labels  = st.session_state.kmeans_labels
        db_labels  = st.session_state.dbscan_labels
        pca_df     = st.session_state.pca_df
        tsne_df    = st.session_state.tsne_df
        profile    = st.session_state.profile
        inertias   = st.session_state.inertias
        k_range    = st.session_state.k_range
        pca_var    = st.session_state.pca_var
        n_clusters = st.session_state.n_clusters
        n_noise    = st.session_state.n_noise

        ui_success(f"Clustering done · {len(cluster_df)} users · KMeans K={kmeans_k} · DBSCAN {n_clusters} clusters · {n_noise} noise")
        metrics_html((len(cluster_df),"Users"),(kmeans_k,"KMeans K"),
                     (f"{pca_var[0]*100:.1f}%","PC1 Var"),(f"{pca_var[1]*100:.1f}%","PC2 Var"),(n_clusters,"DBSCAN Clusters"))

        PAL_KM = ["#A78BFA","#34D399","#F472B6","#FBBF24","#60A5FA","#FB923C","#4ADE80","#E879F9"]
        PAL_DB = ["#94A3B8","#A78BFA","#34D399","#F472B6","#FBBF24"]

        def _scatter(ax, xs, ys, labels, palette, title, xlabel, ylabel):
            for lbl in sorted(set(labels)):
                mask  = np.array(labels) == lbl
                color = "#555" if lbl == -1 else palette[lbl % len(palette)]
                name  = "Noise" if lbl == -1 else f"Cluster {lbl}"
                ax.scatter(xs[mask], ys[mask], c=color, label=name,
                           s=70, alpha=0.85, edgecolors="white", linewidths=0.4)
            ax.set_title(title, color="#E2E8F0", fontsize=10)
            ax.set_xlabel(xlabel, color="#94A3B8", fontsize=8)
            ax.set_ylabel(ylabel, color="#94A3B8", fontsize=8)
            ax.tick_params(colors="#94A3B8", labelsize=7)
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            for sp in ["bottom","left"]: ax.spines[sp].set_color("#2D2B45")
            ax.legend(fontsize=7, facecolor="#1E1B2E", labelcolor="#CBD5E1")

        with st.expander("◆ KMeans Elbow Curve", expanded=True):
            fig_e, ax_e = plt.subplots(figsize=(8,4))
            fig_e.patch.set_facecolor("#0F0D1A"); ax_e.set_facecolor("#13111F")
            ax_e.plot(k_range, inertias, "o-", color="#A78BFA", lw=2, markersize=7)
            ax_e.axvline(kmeans_k, color="#34D399", linestyle="--", lw=1.5, label=f"K={kmeans_k}")
            ax_e.set_title("KMeans Elbow Curve", color="#E2E8F0", fontsize=11)
            ax_e.set_xlabel("K", color="#94A3B8"); ax_e.set_ylabel("Inertia", color="#94A3B8")
            ax_e.tick_params(colors="#94A3B8")
            for sp in ["top","right"]: ax_e.spines[sp].set_visible(False)
            for sp in ["bottom","left"]: ax_e.spines[sp].set_color("#2D2B45")
            ax_e.legend(fontsize=8, facecolor="#1E1B2E", labelcolor="#CBD5E1")
            plt.tight_layout(); st.pyplot(fig_e, use_container_width=True); plt.close()

        with st.expander("◆ PCA Projection — KMeans & DBSCAN", expanded=True):
            fig_pca, axes = plt.subplots(1, 2, figsize=(13,5))
            fig_pca.patch.set_facecolor("#0F0D1A")
            for ax in axes: ax.set_facecolor("#13111F")
            _scatter(axes[0], pca_df["PC1"].values, pca_df["PC2"].values, list(km_labels), PAL_KM,
                     f"KMeans (K={kmeans_k}) — PCA", f"PC1 ({pca_var[0]*100:.1f}%)", f"PC2 ({pca_var[1]*100:.1f}%)")
            _scatter(axes[1], pca_df["PC1"].values, pca_df["PC2"].values, list(db_labels), PAL_DB,
                     f"DBSCAN (eps={dbscan_eps:.1f}) — PCA", f"PC1 ({pca_var[0]*100:.1f}%)", f"PC2 ({pca_var[1]*100:.1f}%)")
            plt.tight_layout(pad=2); st.pyplot(fig_pca, use_container_width=True); plt.close()

        with st.expander("◆ t-SNE Projection", expanded=True):
            fig_ts, axes2 = plt.subplots(1, 2, figsize=(13,5))
            fig_ts.patch.set_facecolor("#0F0D1A")
            for ax in axes2: ax.set_facecolor("#13111F")
            _scatter(axes2[0], tsne_df["tSNE1"].values, tsne_df["tSNE2"].values, list(km_labels), PAL_KM,
                     f"t-SNE · KMeans (K={kmeans_k})", "tSNE-1", "tSNE-2")
            _scatter(axes2[1], tsne_df["tSNE1"].values, tsne_df["tSNE2"].values, list(db_labels), PAL_DB,
                     f"t-SNE · DBSCAN (eps={dbscan_eps:.1f})", "tSNE-1", "tSNE-2")
            plt.tight_layout(pad=2); st.pyplot(fig_ts, use_container_width=True); plt.close()

        with st.expander("◆ Cluster Profiles & Personas", expanded=True):
            feat_cols_p = [c for c in ["TotalSteps","Calories","VeryActiveMinutes",
                                       "SedentaryMinutes","TotalSleepMinutes"]
                           if c in profile.columns]
            st.markdown("**Mean feature values per KMeans cluster**")
            st.dataframe(profile, use_container_width=True)
            # Persona cards
            st.markdown("<br>**Cluster Personas**", unsafe_allow_html=True)
            pcols = st.columns(kmeans_k)
            css_cycle = ["fp-cluster-0","fp-cluster-1","fp-cluster-2"]
            for i in range(kmeans_k):
                with pcols[i % len(pcols)]:
                    if i in profile.index:
                        row   = profile.loc[i]
                        steps = row.get("TotalSteps", 0)
                        sed   = row.get("SedentaryMinutes", 0)
                        active= row.get("VeryActiveMinutes", 0)
                        users_in = cluster_df[cluster_df["KMeans_Cluster"]==i]
                        if steps > 10000:
                            persona, color, em = "HIGHLY ACTIVE","#F472B6","🏃"
                        elif steps > 5000:
                            persona, color, em = "MODERATELY ACTIVE","#34D399","🚶"
                        else:
                            persona, color, em = "SEDENTARY","#A78BFA","🛋️"
                        css_cls  = css_cycle[i % len(css_cycle)]
                        user_ids = ", ".join([str(x)[-4:] for x in users_in["Id"].tolist()[:4]])
                        if len(users_in)>4: user_ids += f" +{len(users_in)-4}"
                        st.markdown(f"""
                        <div class="fp-cluster {css_cls}">
                          <div style="font-size:1.8rem">{em}</div>
                          <div style="font-weight:700;color:{color};font-size:.9rem">Cluster {i}</div>
                          <div style="font-size:.75rem;color:{color};font-weight:600;letter-spacing:.05em">{persona}</div>
                          <div style="font-size:.75rem;color:#CBD5E1;margin-top:8px;line-height:1.8">
                            Steps: <b>{steps:,.0f}</b>/day<br>
                            Sedentary: <b>{sed:.0f}</b> min<br>
                            Very Active: <b>{active:.0f}</b> min<br>
                            Users ({len(users_in)}): <span style="color:#64748B">…{user_ids}</span>
                          </div>
                        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# ██  MILESTONE 3 · ANOMALY DETECTION & VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════════
dropdown_header("🚨", "Milestone 3 · Anomaly Detection & Visualization",
                "Threshold Violations · Prophet Residuals · DBSCAN Outliers · Plotly Charts · Accuracy Simulation")

with st.expander("▼  Open Milestone 3 — Anomaly Detection", expanded=False):

    # ── M3 data loading (re-use M2 data or load fresh) ─────────────────────────
    sec("📂", "Data Loading", "Step 1")
    ui_info("Milestone 3 uses the same 5 Fitbit CSV files. If you already loaded them in Milestone 2, click the button below to proceed. Otherwise upload the files here.")

    if st.session_state.data_loaded:
        ui_success(f"Master DataFrame already loaded from M2 — {st.session_state.master.shape[0]} rows · {st.session_state.master['Id'].nunique()} users. Ready for anomaly detection!")
    else:
        ui_warn("Files not loaded yet. Please complete the Load & Parse step in Milestone 2 first (or upload files below).")

        m3_files = st.file_uploader(
            "📁 Drop all 5 Fitbit CSV files here (M3 standalone upload)",
            type="csv", accept_multiple_files=True, key="m3_uploader")

        REQUIRED_FILES_M3 = {
            "dailyActivity_merged.csv":     {"key_cols":["ActivityDate","TotalSteps","Calories"],    "label":"Daily Activity","icon":"🏃"},
            "hourlySteps_merged.csv":       {"key_cols":["ActivityHour","StepTotal"],               "label":"Hourly Steps",  "icon":"👣"},
            "hourlyIntensities_merged.csv": {"key_cols":["ActivityHour","TotalIntensity"],          "label":"Hourly Int.",   "icon":"⚡"},
            "minuteSleep_merged.csv":       {"key_cols":["date","value","logId"],                   "label":"Sleep",         "icon":"💤"},
            "heartrate_seconds_merged.csv": {"key_cols":["Time","Value"],                           "label":"Heart Rate",    "icon":"❤️"},
        }

        detected_m3 = {}
        if m3_files:
            raw_uploads = []
            for uf in m3_files:
                try:
                    df_tmp = pd.read_csv(uf)
                    raw_uploads.append((uf.name, df_tmp))
                except Exception:
                    pass
            for req_name, finfo in REQUIRED_FILES_M3.items():
                best_score, best_df = 0, None
                for uname, udf in raw_uploads:
                    s = sum(1 for col in finfo["key_cols"] if col in udf.columns)
                    if s > best_score:
                        best_score, best_df = s, udf
                if best_score >= 2:
                    detected_m3[req_name] = best_df

        n_m3 = len(detected_m3)
        metrics_html((n_m3,"Detected"),(5-n_m3,"Missing"),("✓ Ready" if n_m3==5 else "⚠ Incomplete","Status"))

        if st.button("⚡ Load & Build Master DataFrame (M3)", disabled=(n_m3 < 5), key="m3_btn_load"):
            with st.spinner("Parsing and building master…"):
                try:
                    daily    = detected_m3["dailyActivity_merged.csv"].copy()
                    hourly_s = detected_m3["hourlySteps_merged.csv"].copy()
                    hourly_i = detected_m3["hourlyIntensities_merged.csv"].copy()
                    sleep    = detected_m3["minuteSleep_merged.csv"].copy()
                    hr       = detected_m3["heartrate_seconds_merged.csv"].copy()

                    daily["ActivityDate"]    = pd.to_datetime(daily["ActivityDate"])
                    hourly_s["ActivityHour"] = pd.to_datetime(hourly_s["ActivityHour"])
                    hourly_i["ActivityHour"] = pd.to_datetime(hourly_i["ActivityHour"])
                    sleep["date"]            = pd.to_datetime(sleep["date"])
                    hr["Time"]               = pd.to_datetime(hr["Time"])

                    hr_minute = (hr.groupby(["Id", pd.Grouper(key="Time", freq="1min")])["Value"]
                                 .mean().reset_index())
                    hr_minute.columns = ["Id","Time","HeartRate"]
                    hr_minute = hr_minute.dropna()
                    hr_minute["Date"] = hr_minute["Time"].dt.date

                    hr_daily = (hr_minute.groupby(["Id","Date"])["HeartRate"]
                                .agg(["mean","max","min","std"]).reset_index()
                                .rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"}))

                    sleep["Date"] = sleep["date"].dt.date
                    sleep_daily = (sleep.groupby(["Id","Date"])
                                   .agg(TotalSleepMinutes=("value","count")).reset_index())

                    daily["Date"] = daily["ActivityDate"].dt.date
                    keep = [c for c in ["Id","Date","TotalSteps","Calories","VeryActiveMinutes",
                                        "FairlyActiveMinutes","LightlyActiveMinutes","SedentaryMinutes"]
                            if c in daily.columns]
                    master = daily[keep].copy()
                    master = master.merge(hr_daily,    on=["Id","Date"], how="left")
                    master = master.merge(sleep_daily, on=["Id","Date"], how="left")
                    master["Date"] = pd.to_datetime(master["Date"])
                    master["TotalSleepMinutes"] = master["TotalSleepMinutes"].fillna(0)
                    for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
                        if col in master.columns:
                            master[col] = master.groupby("Id")[col].transform(
                                lambda x: x.fillna(x.median()))

                    for k, v in [("daily",daily),("hourly_s",hourly_s),("hourly_i",hourly_i),
                                  ("sleep",sleep),("hr",hr),("hr_minute",hr_minute),("master",master)]:
                        st.session_state[k] = v
                    st.session_state.data_loaded = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Anomaly Detection ───────────────────────────────────────────────────────
    if st.session_state.data_loaded:
        master = st.session_state.master

        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
        sec("🚨", "Anomaly Detection — Three Methods", "Steps 2–4")

        st.markdown(f"""
        <div class="fp-card">
          <div class="fp-card-title">Detection Methods Applied</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;font-size:0.83rem">
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.9rem">
              <div style="color:{ACCENT_RED};font-weight:600;margin-bottom:0.4rem">① Threshold Violations</div>
              <div style="color:{MUTED}">Hard upper/lower limits on HR, Steps, Sleep.</div>
            </div>
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.9rem">
              <div style="color:{ACCENT2};font-weight:600;margin-bottom:0.4rem">② Residual-Based</div>
              <div style="color:{MUTED}">Rolling median baseline · flag days ±{sigma:.0f}σ deviation.</div>
            </div>
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.9rem">
              <div style="color:{ACCENT3};font-weight:600;margin-bottom:0.4rem">③ DBSCAN Outliers</div>
              <div style="color:{MUTED}">Users labelled −1 by DBSCAN are structural outliers.</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        if st.button("🔍 Run Anomaly Detection (All 3 Methods)", key="m3_btn_detect"):
            with st.spinner("Detecting anomalies…"):
                try:
                    anom_hr    = detect_hr_anomalies(master, hr_high, hr_low, sigma)
                    anom_steps = detect_steps_anomalies(master, st_low, 25000, sigma)
                    anom_sleep = detect_sleep_anomalies(master, sl_low, sl_high, sigma)
                    st.session_state.anom_hr    = anom_hr
                    st.session_state.anom_steps = anom_steps
                    st.session_state.anom_sleep = anom_sleep
                    st.session_state.anomaly_done = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Detection error: {e}")

        if st.session_state.anomaly_done:
            anom_hr    = st.session_state.anom_hr
            anom_steps = st.session_state.anom_steps
            anom_sleep = st.session_state.anom_sleep

            n_hr    = int(anom_hr["is_anomaly"].sum())
            n_steps = int(anom_steps["is_anomaly"].sum())
            n_sleep = int(anom_sleep["is_anomaly"].sum())
            n_total = n_hr + n_steps + n_sleep

            ui_danger(f"Total anomalies flagged: {n_total}  (HR: {n_hr} · Steps: {n_steps} · Sleep: {n_sleep})")
            metrics_html((n_hr,"HR Anomalies"),(n_steps,"Steps Anomalies"),
                         (n_sleep,"Sleep Anomalies"),(n_total,"Total Flags"), red_indices=[0,1,2,3])

            # ── CHART 1: Heart Rate ─────────────────────────────────────────────
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            sec("❤️", "Heart Rate — Anomaly Chart", "Step 2")
            anom_tag(f"{n_hr} anomalous days detected")
            screenshot_badge("Heart Rate Chart with Anomaly Highlights")
            step_pill(2, "Threshold + Residual Detection")
            ui_info(f"Red markers = anomaly days. Dashed lines = thresholds (HR>{hr_high} or HR<{hr_low}). Shaded band = ±{sigma:.0f}σ residual zone.")

            hr_anom   = anom_hr[anom_hr["is_anomaly"]]
            hr_normal = anom_hr[~anom_hr["is_anomaly"]]

            fig_hr = go.Figure()
            rolling_upper = anom_hr["rolling_med"] + sigma * anom_hr["residual"].std()
            rolling_lower = anom_hr["rolling_med"] - sigma * anom_hr["residual"].std()

            fig_hr.add_trace(go.Scatter(x=anom_hr["Date"], y=rolling_upper,
                                        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_hr.add_trace(go.Scatter(x=anom_hr["Date"], y=rolling_lower, mode="lines",
                                        fill="tonexty", fillcolor="rgba(99,179,237,0.1)",
                                        line=dict(width=0), name=f"±{sigma:.0f}σ Band"))
            fig_hr.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["AvgHR"],
                                        mode="lines+markers", name="Avg Heart Rate",
                                        line=dict(color=ACCENT, width=2.5), marker=dict(size=5, color=ACCENT),
                                        hovertemplate="<b>%{x}</b><br>HR: %{y:.1f} bpm<extra></extra>"))
            fig_hr.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["rolling_med"],
                                        mode="lines", name="Rolling Median",
                                        line=dict(color=ACCENT3, width=1.5, dash="dot")))
            if not hr_anom.empty:
                fig_hr.add_trace(go.Scatter(x=hr_anom["Date"], y=hr_anom["AvgHR"],
                                            mode="markers", name="🚨 Anomaly",
                                            marker=dict(color=ACCENT_RED, size=14, symbol="circle",
                                                        line=dict(color="white", width=2)),
                                            hovertemplate="<b>%{x}</b><br>HR: %{y:.1f} bpm<br><b>ANOMALY</b><extra>⚠️</extra>"))
                for _, row in hr_anom.iterrows():
                    fig_hr.add_annotation(x=row["Date"], y=row["AvgHR"],
                                          text=f"⚠️ {row['reason']}", showarrow=True,
                                          arrowhead=2, arrowcolor=ACCENT_RED, arrowsize=1.2,
                                          ax=0, ay=-45, font=dict(color=ACCENT_RED, size=9),
                                          bgcolor=CARD_BG, bordercolor=DANGER_BOR, borderwidth=1, borderpad=4)
            fig_hr.add_hline(y=hr_high, line_dash="dash", line_color=ACCENT_RED, line_width=1.5, opacity=0.7,
                             annotation_text=f"High ({hr_high} bpm)", annotation_font_color=ACCENT_RED,
                             annotation_position="top right")
            fig_hr.add_hline(y=hr_low, line_dash="dash", line_color=ACCENT2, line_width=1.5, opacity=0.7,
                             annotation_text=f"Low ({hr_low} bpm)", annotation_font_color=ACCENT2,
                             annotation_position="bottom right")
            apply_plotly_theme(fig_hr, "❤️ Heart Rate — Anomaly Detection (Real Fitbit Data)")
            fig_hr.update_layout(height=480, xaxis_title="Date", yaxis_title="Heart Rate (bpm)",
                                 xaxis=dict(gridcolor=GRID_CLR, tickfont_color=MUTED),
                                 yaxis=dict(gridcolor=GRID_CLR, tickfont_color=MUTED))
            st.plotly_chart(fig_hr, use_container_width=True)

            if not hr_anom.empty:
                with st.expander(f"📋 View {len(hr_anom)} HR Anomaly Records"):
                    st.dataframe(hr_anom[hr_anom["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]]
                                 .rename(columns={"rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"})
                                 .round(2), use_container_width=True)

            # ── CHART 2: Sleep ──────────────────────────────────────────────────
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            sec("💤", "Sleep Pattern — Anomaly Visualization", "Step 3")
            anom_tag(f"{n_sleep} anomalous sleep days detected")
            screenshot_badge("Sleep Pattern Visualization with Alerts")
            step_pill(3, "Threshold Detection on Sleep Minutes")
            ui_info(f"Orange = insufficient sleep (<{sl_low} min). Diamond markers = anomaly days. Green band = healthy sleep zone ({sl_low}–{sl_high} min).")

            sleep_anom   = anom_sleep[anom_sleep["is_anomaly"]]
            sleep_normal = anom_sleep[~anom_sleep["is_anomaly"]]

            fig_sleep = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       row_heights=[0.7, 0.3],
                                       subplot_titles=["Sleep Duration (minutes/night)",
                                                       "Deviation from Expected"],
                                       vertical_spacing=0.08)
            fig_sleep.add_hrect(y0=sl_low, y1=sl_high, fillcolor="rgba(104,211,145,0.08)",
                                 line_width=0, row=1, col=1)
            fig_sleep.add_annotation(x=1, y=sl_high, xref="paper", yref="y1",
                                     text="✅ Healthy Sleep Zone", showarrow=False,
                                     font=dict(color=ACCENT3, size=9),
                                     xanchor="right", yanchor="bottom")
            fig_sleep.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["TotalSleepMinutes"],
                                            mode="lines+markers", name="Sleep Minutes",
                                            line=dict(color="#b794f4", width=2.5),
                                            marker=dict(size=5, color="#b794f4"),
                                            hovertemplate="<b>%{x}</b><br>Sleep: %{y:.0f} min<extra></extra>"),
                                 row=1, col=1)
            fig_sleep.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["rolling_med"],
                                            mode="lines", name="Rolling Median",
                                            line=dict(color=ACCENT3, width=1.5, dash="dot")), row=1, col=1)
            if not sleep_anom.empty:
                fig_sleep.add_trace(go.Scatter(x=sleep_anom["Date"], y=sleep_anom["TotalSleepMinutes"],
                                                mode="markers", name="🚨 Sleep Anomaly",
                                                marker=dict(color=ACCENT_RED, size=14, symbol="diamond",
                                                            line=dict(color="white", width=2)),
                                                hovertemplate="<b>%{x}</b><br>Sleep: %{y:.0f} min<br><b>ANOMALY</b><extra>⚠️</extra>"),
                                     row=1, col=1)
                for _, row in sleep_anom.iterrows():
                    fig_sleep.add_annotation(x=row["Date"], y=row["TotalSleepMinutes"],
                                             text=f"⚠️ {row['reason']}", showarrow=True,
                                             arrowhead=2, arrowcolor=ACCENT_RED, arrowsize=1.2,
                                             ax=20, ay=-40, font=dict(color=ACCENT_RED, size=9),
                                             bgcolor=CARD_BG, bordercolor=DANGER_BOR,
                                             borderwidth=1, borderpad=3,
                                             xref="x", yref="y")
            fig_sleep.add_hline(y=sl_low, line_dash="dash", line_color=ACCENT_RED, line_width=1.5,
                                 opacity=0.7, row=1, col=1)
            fig_sleep.add_annotation(x=1, y=sl_low, xref="paper", yref="y1",
                                     text=f"Min ({sl_low} min)", showarrow=False,
                                     font=dict(color=ACCENT_RED, size=9),
                                     xanchor="right", yanchor="top")
            fig_sleep.add_hline(y=sl_high, line_dash="dash", line_color=ACCENT, line_width=1.5,
                                 opacity=0.7, row=1, col=1)
            fig_sleep.add_annotation(x=1, y=sl_high, xref="paper", yref="y1",
                                     text=f"Max ({sl_high} min)", showarrow=False,
                                     font=dict(color=ACCENT, size=9),
                                     xanchor="right", yanchor="bottom")
            colors_resid = [ACCENT_RED if v else ACCENT for v in anom_sleep["resid_anomaly"]]
            fig_sleep.add_trace(go.Bar(x=anom_sleep["Date"], y=anom_sleep["residual"],
                                        name="Residual", marker_color=colors_resid,
                                        hovertemplate="<b>%{x}</b><br>Residual: %{y:.0f} min<extra></extra>"),
                                 row=2, col=1)
            fig_sleep.add_hline(y=0, line_dash="solid", line_color=MUTED, line_width=1, row=2, col=1)
            apply_plotly_theme(fig_sleep)
            fig_sleep.update_layout(height=560, showlegend=True,
                                     paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT)
            fig_sleep.update_xaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
            fig_sleep.update_yaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
            st.plotly_chart(fig_sleep, use_container_width=True)

            if not sleep_anom.empty:
                with st.expander(f"📋 View {len(sleep_anom)} Sleep Anomaly Records"):
                    st.dataframe(sleep_anom[sleep_anom["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]]
                                 .rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected",
                                                  "residual":"Deviation","reason":"Anomaly Reason"})
                                 .round(2), use_container_width=True)

            # ── CHART 3: Steps ──────────────────────────────────────────────────
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            sec("🚶", "Step Count Trend — Alerts & Anomalies", "Step 4")
            anom_tag(f"{n_steps} anomalous step-count days detected")
            screenshot_badge("Step Count Trend with Alert Bands")
            step_pill(4, "Threshold + Residual Detection on Steps")
            ui_info(f"Red vertical bands = anomaly alert days. Dashed lines = step thresholds. Bar chart shows daily deviation from trend.")

            steps_anom   = anom_steps[anom_steps["is_anomaly"]]
            steps_normal = anom_steps[~anom_steps["is_anomaly"]]

            fig_steps = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       row_heights=[0.65, 0.35],
                                       subplot_titles=["Daily Steps (avg across users)",
                                                       "Residual Deviation from Trend"],
                                       vertical_spacing=0.08)
            for _, row in steps_anom.iterrows():
                d = str(row["Date"])
                d_next = str(pd.Timestamp(d) + pd.Timedelta(days=1))[:10]
                fig_steps.add_vrect(x0=d, x1=d_next,
                                     fillcolor="rgba(252,129,129,0.15)",
                                     line_color="rgba(252,129,129,0.5)",
                                     line_width=1.5)
            fig_steps.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["TotalSteps"],
                                            mode="lines+markers", name="Avg Daily Steps",
                                            line=dict(color=ACCENT3, width=2.5),
                                            marker=dict(size=5, color=ACCENT3),
                                            hovertemplate="<b>%{x}</b><br>Steps: %{y:,.0f}<extra></extra>"),
                                 row=1, col=1)
            fig_steps.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["rolling_med"],
                                            mode="lines", name="Trend (Rolling Median)",
                                            line=dict(color=ACCENT, width=2, dash="dash"),
                                            hovertemplate="<b>%{x}</b><br>Trend: %{y:,.0f}<extra></extra>"),
                                 row=1, col=1)
            if not steps_anom.empty:
                fig_steps.add_trace(go.Scatter(x=steps_anom["Date"], y=steps_anom["TotalSteps"],
                                                mode="markers", name="🚨 Steps Anomaly",
                                                marker=dict(color=ACCENT_RED, size=14, symbol="triangle-up",
                                                            line=dict(color="white", width=2)),
                                                hovertemplate="<b>%{x}</b><br>Steps: %{y:,.0f}<br><b>ALERT</b><extra>⚠️</extra>"),
                                     row=1, col=1)
            fig_steps.add_hline(y=st_low, line_dash="dash", line_color=ACCENT_RED, line_width=1.5, opacity=0.8,
                                 row=1, col=1)
            fig_steps.add_annotation(x=1, y=st_low, xref="paper", yref="y1",
                                     text=f"Low Alert ({st_low:,} steps)", showarrow=False,
                                     font=dict(color=ACCENT_RED, size=9),
                                     xanchor="right", yanchor="top")
            fig_steps.add_hline(y=25000, line_dash="dash", line_color=ACCENT2, line_width=1.5, opacity=0.7,
                                 row=1, col=1)
            fig_steps.add_annotation(x=1, y=25000, xref="paper", yref="y1",
                                     text="High Alert (25,000 steps)", showarrow=False,
                                     font=dict(color=ACCENT2, size=9),
                                     xanchor="right", yanchor="bottom")
            res_colors = [ACCENT_RED if v else ACCENT3 for v in anom_steps["resid_anomaly"]]
            fig_steps.add_trace(go.Bar(x=anom_steps["Date"], y=anom_steps["residual"],
                                        name="Residual", marker_color=res_colors,
                                        hovertemplate="<b>%{x}</b><br>Deviation: %{y:,.0f} steps<extra></extra>"),
                                 row=2, col=1)
            fig_steps.add_hline(y=0, line_dash="solid", line_color=MUTED, line_width=1, row=2, col=1)
            apply_plotly_theme(fig_steps)
            fig_steps.update_layout(height=560, showlegend=True,
                                     paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT)
            fig_steps.update_xaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
            fig_steps.update_yaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
            st.plotly_chart(fig_steps, use_container_width=True)

            if not steps_anom.empty:
                with st.expander(f"📋 View {len(steps_anom)} Steps Anomaly Records"):
                    st.dataframe(anom_steps[anom_steps["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]]
                                 .rename(columns={"TotalSteps":"Steps","rolling_med":"Expected",
                                                  "residual":"Deviation","reason":"Anomaly Reason"})
                                 .round(2), use_container_width=True)

            # ── CHART 4: DBSCAN Outlier Users ────────────────────────────────────
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            sec("🔍", "DBSCAN Outlier Users — Cluster-Based Anomalies", "Step 5")
            step_pill(5, "Structural Outlier Detection via DBSCAN")
            anom_tag("Outlier = users with atypical overall behaviour pattern")
            ui_info("Users labelled −1 by DBSCAN are structural outliers — their behaviour doesn't fit any group.")

            cluster_cols = [c for c in ["TotalSteps","Calories","VeryActiveMinutes",
                                        "FairlyActiveMinutes","LightlyActiveMinutes",
                                        "SedentaryMinutes","TotalSleepMinutes"]
                            if c in master.columns]
            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import DBSCAN
                from sklearn.decomposition import PCA

                cf = master.groupby("Id")[cluster_cols].mean().round(3).dropna()
                X_scaled = StandardScaler().fit_transform(cf)
                db = DBSCAN(eps=2.2, min_samples=2)
                db_labels_m3 = db.fit_predict(X_scaled)
                pca_m3 = PCA(n_components=2)
                X_pca_m3 = pca_m3.fit_transform(X_scaled)
                var_m3   = pca_m3.explained_variance_ratio_ * 100
                cf["DBSCAN"] = db_labels_m3
                outlier_users = cf[cf["DBSCAN"]==-1].index.tolist()
                n_out = len(outlier_users)
                n_clu = len(set(db_labels_m3)) - (1 if -1 in db_labels_m3 else 0)

                metrics_html((n_clu,"DBSCAN Clusters"),(n_out,"Outlier Users"),
                             (len(cf)-n_out,"Normal Users"), red_indices=[1])

                CLUSTER_COLORS = [ACCENT, ACCENT3, "#f6ad55", "#b794f4", ACCENT2]
                fig_db = go.Figure()
                for lbl in sorted(set(db_labels_m3)):
                    if lbl == -1: continue
                    mask = db_labels_m3 == lbl
                    fig_db.add_trace(go.Scatter(
                        x=X_pca_m3[mask,0], y=X_pca_m3[mask,1],
                        mode="markers+text", name=f"Cluster {lbl}",
                        marker=dict(size=14, color=CLUSTER_COLORS[lbl%len(CLUSTER_COLORS)],
                                    opacity=0.85, line=dict(color="white", width=1.5)),
                        text=[str(uid)[-4:] for uid in cf.index[mask]],
                        textposition="top center", textfont=dict(size=8, color=TEXT),
                        hovertemplate="<b>User ...%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"))
                if n_out > 0:
                    mask_out = db_labels_m3 == -1
                    fig_db.add_trace(go.Scatter(
                        x=X_pca_m3[mask_out,0], y=X_pca_m3[mask_out,1],
                        mode="markers+text", name="🚨 Outlier / Anomaly",
                        marker=dict(size=20, color=ACCENT_RED, symbol="x",
                                    line=dict(color="white", width=2.5)),
                        text=[str(uid)[-4:] for uid in cf.index[mask_out]],
                        textposition="top center", textfont=dict(size=9, color=ACCENT_RED),
                        hovertemplate="<b>⚠️ OUTLIER User ...%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra>ANOMALY</extra>"))
                    for _oi, uid in enumerate(cf.index[mask_out]):
                        xi, yi = X_pca_m3[mask_out][_oi]
                        fig_db.add_shape(type="circle",
                            x0=xi-0.3, y0=yi-0.3, x1=xi+0.3, y1=yi+0.3,
                            line=dict(color=ACCENT_RED, width=2, dash="dot"),
                            fillcolor="rgba(252,129,129,0.1)")

                apply_plotly_theme(fig_db, "🔍 DBSCAN Outlier Detection — PCA Projection (eps=2.2)")
                fig_db.update_layout(height=500,
                    xaxis_title=f"PC1 ({var_m3[0]:.1f}% variance)",
                    yaxis_title=f"PC2 ({var_m3[1]:.1f}% variance)",
                    xaxis=dict(gridcolor=GRID_CLR, tickfont_color=MUTED),
                    yaxis=dict(gridcolor=GRID_CLR, tickfont_color=MUTED))
                st.plotly_chart(fig_db, use_container_width=True)

                if outlier_users:
                    ui_danger(f"{n_out} outlier user(s) detected: {[str(u)[-6:] for u in outlier_users]}")
                    st.dataframe(cf[cf["DBSCAN"]==-1][cluster_cols].round(2), use_container_width=True)

            except Exception as e:
                ui_warn(f"DBSCAN clustering skipped: {e}")

            # ── Accuracy Simulation ─────────────────────────────────────────────
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            sec("🎯", "Simulated Detection Accuracy — 90%+ Target", "Step 6")
            step_pill(6, "Inject Known Anomalies → Measure Detection Rate")
            ui_info("10 known anomalies are injected into each signal. The detector is run and we measure how many it catches. Validates the 90%+ accuracy requirement.")

            if st.button("🎯 Run Accuracy Simulation (10 injected anomalies per signal)", key="m3_btn_sim"):
                with st.spinner("Simulating…"):
                    try:
                        sim = simulate_accuracy(master, n_inject=10)
                        st.session_state.sim_results     = sim
                        st.session_state.simulation_done = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Simulation error: {e}")

            if st.session_state.simulation_done and st.session_state.sim_results:
                sim     = st.session_state.sim_results
                overall = sim["Overall"]
                passed  = overall >= 90.0

                if passed:
                    ui_success(f"Overall accuracy: {overall}% — ✅ MEETS 90%+ REQUIREMENT")
                else:
                    ui_warn(f"Overall accuracy: {overall}% — below 90% target, adjust thresholds in sidebar")

                html = '<div class="metric-grid">'
                for signal in ["Heart Rate","Steps","Sleep"]:
                    r   = sim[signal]
                    acc = r["accuracy"]
                    col = ACCENT3 if acc >= 90 else ACCENT_RED
                    html += f"""
                    <div class="metric-card" style="border-color:{col}44">
                      <div style="font-size:1.8rem;font-weight:800;color:{col};font-family:'Syne',sans-serif">{acc}%</div>
                      <div style="font-size:0.8rem;color:{TEXT};font-weight:600;margin:0.3rem 0">{signal}</div>
                      <div style="font-size:0.72rem;color:{MUTED}">{r['detected']}/{r['injected']} detected</div>
                      <div style="font-size:0.7rem;color:{'#9ae6b4' if acc>=90 else ACCENT_RED}">{'✅ PASS' if acc>=90 else '⚠️ LOW'}</div>
                    </div>"""
                html += f"""
                    <div class="metric-card" style="border-color:{'#68d391' if passed else ACCENT_RED}88;background:{'rgba(104,211,145,0.1)' if passed else DANGER_BG}">
                      <div style="font-size:1.8rem;font-weight:800;color:{'#68d391' if passed else ACCENT_RED};font-family:'Syne',sans-serif">{overall}%</div>
                      <div style="font-size:0.8rem;color:{TEXT};font-weight:600;margin:0.3rem 0">Overall</div>
                      <div style="font-size:0.7rem;color:{'#9ae6b4' if passed else ACCENT_RED}">{'✅ 90%+ ACHIEVED' if passed else '⚠️ BELOW TARGET'}</div>
                    </div>"""
                html += '</div>'
                st.markdown(html, unsafe_allow_html=True)

                signals    = ["Heart Rate","Steps","Sleep"]
                accs       = [sim[s]["accuracy"] for s in signals]
                bar_colors = [ACCENT3 if a >= 90 else ACCENT_RED for a in accs]

                fig_acc = go.Figure()
                fig_acc.add_trace(go.Bar(x=signals, y=accs, marker_color=bar_colors,
                                         text=[f"{a}%" for a in accs], textposition="outside",
                                         textfont=dict(color=TEXT, size=14, family="Syne, sans-serif"),
                                         hovertemplate="<b>%{x}</b><br>Accuracy: %{y}%<extra></extra>",
                                         name="Detection Accuracy"))
                fig_acc.add_hline(y=90, line_dash="dash", line_color=ACCENT_RED, line_width=2,
                                   annotation_text="90% Target", annotation_font_color=ACCENT_RED,
                                   annotation_position="top right")
                apply_plotly_theme(fig_acc, "🎯 Simulated Anomaly Detection Accuracy")
                fig_acc.update_layout(height=380, yaxis_range=[0,115],
                                       yaxis_title="Detection Accuracy (%)", xaxis_title="Signal",
                                       xaxis=dict(gridcolor=GRID_CLR, tickfont_color=MUTED),
                                       yaxis=dict(gridcolor=GRID_CLR, tickfont_color=MUTED),
                                       showlegend=False)
                screenshot_badge("Accuracy Bar Chart (90%+ target line)")
                st.plotly_chart(fig_acc, use_container_width=True)

    else:
        st.markdown(f"""
        <div class="fp-card" style="text-align:center;padding:3rem">
          <div style="font-size:3rem;margin-bottom:1rem">🚨</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:{TEXT};margin-bottom:0.5rem">
            Load data in Milestone 2 first
          </div>
          <div style="color:{MUTED};font-size:0.88rem">
            Complete the <b>Load &amp; Parse</b> step in Milestone 2 above, then return here for anomaly detection.
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()


# ════════════════════════════════════════════════════════════════════════════════
#  FOOTER SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="fp-card">
  <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;color:{TEXT};margin-bottom:16px">✅ Complete Pipeline Summary</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
""", unsafe_allow_html=True)

summary = [
    ("🧹","M1 · Data Cleaning",   st.session_state.original_df is not None,
     "CSV upload · null fill · datetime parse · preview"),
    ("🔧","M2 · Load & Parse",    bool(st.session_state.data_loaded),
     "5 Fitbit CSVs · master DataFrame · time normalization"),
    ("🧪","M2 · TSFresh Features",bool(st.session_state.tsfresh_done),
     "MinimalFCParameters · normalized heatmap"),
    ("📈","M2 · Prophet Forecast",bool(st.session_state.prophet_done),
     "HR + Steps + Sleep · 30-day · 80% CI"),
    (f"🤖",f"M2 · KMeans (K={kmeans_k})", bool(st.session_state.cluster_done),
     "PCA · t-SNE · persona cards"),
    (f"🔍",f"M2 · DBSCAN (eps={dbscan_eps:.1f})", bool(st.session_state.cluster_done),
     "Density-based · noise detection"),
    ("🚨","M3 · Anomaly Detection", bool(st.session_state.anomaly_done),
     "Threshold + Residual + DBSCAN · Plotly charts"),
    ("🎯","M3 · Accuracy Simulation", bool(st.session_state.simulation_done),
     "10 injected anomalies per signal · 90%+ target"),
]
for icon, title, done, desc in summary:
    color = ACCENT3 if done else MUTED
    ic    = "✅" if done else "⭕"
    border_c = SUCCESS_BOR if done else CARD_BOR
    st.markdown(f"""
    <div style="background:{SECTION_BG};border-radius:10px;padding:12px 16px;border:1px solid {border_c}">
      <div style="font-size:.85rem;font-weight:600;color:{color}">{ic} {icon} {title}</div>
      <div style="font-size:.75rem;color:{MUTED};margin-top:4px">{desc}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;padding:28px 0 12px;font-size:.72rem;color:{MUTED};font-family:'JetBrains Mono',monospace">
  💪 FitPulse &nbsp;|&nbsp; Milestone 1 · 2 · 3 &nbsp;|&nbsp;
  Data Cleaning · TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE · Anomaly Detection &nbsp;|&nbsp;
  Real Fitbit Dataset · Mar–Apr 2016
</div>
""", unsafe_allow_html=True)