"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          💪 FitPulse — Complete Health Analytics App                        ║
║  Milestone 1: Data Cleaning                                                  ║
║  Milestone 2: ML Pipeline (TSFresh · Prophet · KMeans · DBSCAN)            ║
║  Milestone 3: Anomaly Detection & Visualization                              ║
║  Milestone 4: Insights Dashboard · PDF/CSV Export · Date/User Filters       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run:
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn \
                tsfresh prophet plotly fpdf2
    streamlit run fitpulse_complete.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings, io, base64, tempfile, os
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")


# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse · All Milestones",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark/Light Mode ────────────────────────────────────────────────────────────
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
    ACCENT_ORG = "#f6ad55"
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
    ACCENT_ORG = "#dd6b20"
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
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
    border-right: 1px solid {CARD_BOR};
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
.block-container {{ padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1500px; }}
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
    # M4
    "m4_pipeline_done": False,
    "m4_master": None, "m4_anom_hr": None, "m4_anom_steps": None, "m4_anom_sleep": None,
}
for _k, _v in _all_keys.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ════════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS  (shared across M1–M3)
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

def ptheme(fig, title="", h=400):
    """Plotly theme helper (M4 style)."""
    fig.update_layout(**PLOTLY_LAYOUT, height=h)
    fig.update_xaxes(gridcolor=GRID_CLR, zeroline=False, linecolor=CARD_BOR, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID_CLR, zeroline=False, linecolor=CARD_BOR, tickfont_color=MUTED)
    if title:
        fig.update_layout(title=dict(text=title, font_color=TEXT, font_size=13, font_family="Syne, sans-serif"))
    return fig


# ════════════════════════════════════════════════════════════════════════════════
#  ANOMALY DETECTION FUNCTIONS  (M3 — original)
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
#  M4 DETECTION FUNCTIONS  (cleaner versions for the Insights Dashboard)
# ════════════════════════════════════════════════════════════════════════════════
def detect_hr(master, hr_high=100, hr_low=50, sigma=2.0):
    df = master[["Id","Date","AvgHR"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["AvgHR"].mean().reset_index().sort_values("Date")
    d["rolling_med"] = d["AvgHR"].rolling(3,center=True,min_periods=1).median()
    d["residual"]    = d["AvgHR"] - d["rolling_med"]
    std              = d["residual"].std()
    d["thresh_high"] = d["AvgHR"] > hr_high
    d["thresh_low"]  = d["AvgHR"] < hr_low
    d["resid_anom"]  = d["residual"].abs() > sigma * std
    d["is_anomaly"]  = d["thresh_high"] | d["thresh_low"] | d["resid_anom"]
    def reason(r):
        parts = []
        if r.thresh_high: parts.append(f"HR>{hr_high}")
        if r.thresh_low:  parts.append(f"HR<{hr_low}")
        if r.resid_anom:  parts.append(f"+/-{sigma:.0f}sigma residual")
        return ", ".join(parts)
    d["reason"] = d.apply(reason, axis=1)
    return d

def detect_steps(master, st_low=500, st_high=25000, sigma=2.0):
    df = master[["Date","TotalSteps"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["TotalSteps"].mean().reset_index().sort_values("Date")
    d["rolling_med"] = d["TotalSteps"].rolling(3,center=True,min_periods=1).median()
    d["residual"]    = d["TotalSteps"] - d["rolling_med"]
    std              = d["residual"].std()
    d["thresh_low"]  = d["TotalSteps"] < st_low
    d["thresh_high"] = d["TotalSteps"] > st_high
    d["resid_anom"]  = d["residual"].abs() > sigma * std
    d["is_anomaly"]  = d["thresh_low"] | d["thresh_high"] | d["resid_anom"]
    def reason(r):
        parts = []
        if r.thresh_low:  parts.append(f"Steps<{int(st_low):,}")
        if r.thresh_high: parts.append(f"Steps>{int(st_high):,}")
        if r.resid_anom:  parts.append(f"+/-{sigma:.0f}sigma residual")
        return ", ".join(parts)
    d["reason"] = d.apply(reason, axis=1)
    return d

def detect_sleep(master, sl_low=60, sl_high=600, sigma=2.0):
    df = master[["Date","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["TotalSleepMinutes"].mean().reset_index().sort_values("Date")
    d["rolling_med"] = d["TotalSleepMinutes"].rolling(3,center=True,min_periods=1).median()
    d["residual"]    = d["TotalSleepMinutes"] - d["rolling_med"]
    std              = d["residual"].std()
    d["thresh_low"]  = (d["TotalSleepMinutes"]>0) & (d["TotalSleepMinutes"]<sl_low)
    d["thresh_high"] = d["TotalSleepMinutes"] > sl_high
    d["resid_anom"]  = d["residual"].abs() > sigma * std
    d["is_anomaly"]  = d["thresh_low"] | d["thresh_high"] | d["resid_anom"]
    def reason(r):
        parts = []
        if r.thresh_low:  parts.append(f"Sleep<{int(sl_low)}min")
        if r.thresh_high: parts.append(f"Sleep>{int(sl_high)}min")
        if r.resid_anom:  parts.append(f"+/-{sigma:.0f}sigma residual")
        return ", ".join(parts)
    d["reason"] = d.apply(reason, axis=1)
    return d


# ════════════════════════════════════════════════════════════════════════════════
#  M4 CHART BUILDERS
# ════════════════════════════════════════════════════════════════════════════════
def chart_hr(anom_hr, hr_high, hr_low, sigma, h=380):
    fig = go.Figure()
    upper = anom_hr["rolling_med"] + sigma * anom_hr["residual"].std()
    lower = anom_hr["rolling_med"] - sigma * anom_hr["residual"].std()
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=upper, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=lower, mode="lines",
                             fill="tonexty", fillcolor="rgba(99,179,237,0.1)",
                             line=dict(width=0), name=f"+/-{sigma:.0f}sigma Band"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["AvgHR"],
                             mode="lines+markers", name="Avg HR",
                             line=dict(color=ACCENT, width=2.5), marker=dict(size=5),
                             hovertemplate="<b>%{x|%d %b}</b><br>HR: %{y:.1f} bpm<extra></extra>"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=ACCENT3, width=1.5, dash="dot")))
    a = anom_hr[anom_hr["is_anomaly"]]
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["AvgHR"], mode="markers",
                                 name="🚨 Anomaly",
                                 marker=dict(color=ACCENT_RED, size=13, symbol="circle",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>HR: %{y:.1f}<br><b>ANOMALY</b><extra>⚠️</extra>"))
        for _, row in a.iterrows():
            fig.add_annotation(x=row["Date"], y=row["AvgHR"],
                               text="⚠️", showarrow=True, arrowhead=2,
                               arrowcolor=ACCENT_RED, ax=0, ay=-35,
                               font=dict(color=ACCENT_RED, size=11))
    fig.add_hline(y=hr_high, line_dash="dash", line_color=ACCENT_RED,
                  line_width=1.5, opacity=0.6,
                  annotation_text=f"High ({int(hr_high)} bpm)",
                  annotation_font_color=ACCENT_RED, annotation_position="top right")
    fig.add_hline(y=hr_low, line_dash="dash", line_color=ACCENT2,
                  line_width=1.5, opacity=0.6,
                  annotation_text=f"Low ({int(hr_low)} bpm)",
                  annotation_font_color=ACCENT2, annotation_position="bottom right")
    ptheme(fig, "❤️ Heart Rate - Anomaly Detection", h)
    fig.update_layout(xaxis_title="Date", yaxis_title="HR (bpm)")
    return fig

def chart_steps(anom_steps, st_low, h=380):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65,0.35], vertical_spacing=0.07,
                        subplot_titles=["Daily Steps (avg users)","Residual Deviation"])
    a = anom_steps[anom_steps["is_anomaly"]]
    for _, row in a.iterrows():
        fig.add_vrect(x0=str(row["Date"]), x1=str(row["Date"]),
                      fillcolor="rgba(252,129,129,0.12)",
                      line_color="rgba(252,129,129,0.4)", line_width=1.5,
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["TotalSteps"],
                             mode="lines+markers", name="Avg Steps",
                             line=dict(color=ACCENT3, width=2.5), marker=dict(size=5),
                             hovertemplate="<b>%{x|%d %b}</b><br>Steps: %{y:,.0f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=ACCENT, width=2, dash="dash")),
                  row=1, col=1)
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["TotalSteps"],
                                 mode="markers", name="🚨 Alert",
                                 marker=dict(color=ACCENT_RED, size=13, symbol="triangle-up",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>Steps: %{y:,.0f}<br><b>ALERT</b><extra>⚠️</extra>"),
                      row=1, col=1)
    fig.add_hline(y=int(st_low), line_dash="dash", line_color=ACCENT_RED,
                  line_width=1.5, opacity=0.7, row=1, col=1,
                  annotation_text=f"Low ({int(st_low):,})",
                  annotation_font_color=ACCENT_RED)
    res_colors = [ACCENT_RED if v else ACCENT3 for v in anom_steps["resid_anom"]]
    fig.add_trace(go.Bar(x=anom_steps["Date"], y=anom_steps["residual"],
                         name="Residual", marker_color=res_colors,
                         hovertemplate="<b>%{x|%d %b}</b><br>Δ: %{y:,.0f}<extra></extra>"),
                  row=2, col=1)
    fig.add_hline(y=0, line_color=MUTED, line_width=1, row=2, col=1)
    ptheme(fig, "🚶 Step Count - Trend & Alerts", h)
    fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT)
    fig.update_xaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
    return fig

def chart_sleep(anom_sleep, sl_low, sl_high, h=380):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65,0.35], vertical_spacing=0.07,
                        subplot_titles=["Sleep Duration (min/night)","Residual Deviation"])
    fig.add_hrect(y0=sl_low, y1=sl_high,
                  fillcolor="rgba(104,211,145,0.07)", line_width=0,
                  annotation_text="✅ Healthy Zone", annotation_position="top right",
                  annotation_font_color=ACCENT3, row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["TotalSleepMinutes"],
                             mode="lines+markers", name="Sleep (min)",
                             line=dict(color="#b794f4", width=2.5), marker=dict(size=5),
                             hovertemplate="<b>%{x|%d %b}</b><br>Sleep: %{y:.0f} min<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=ACCENT3, width=1.5, dash="dot")),
                  row=1, col=1)
    a = anom_sleep[anom_sleep["is_anomaly"]]
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["TotalSleepMinutes"],
                                 mode="markers", name="🚨 Anomaly",
                                 marker=dict(color=ACCENT_RED, size=13, symbol="diamond",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>Sleep: %{y:.0f}<br><b>ANOMALY</b><extra>⚠️</extra>"),
                      row=1, col=1)
    fig.add_hline(y=int(sl_low), line_dash="dash", line_color=ACCENT_RED,
                  line_width=1.5, opacity=0.7, row=1, col=1,
                  annotation_text=f"Min ({int(sl_low)} min)",
                  annotation_font_color=ACCENT_RED)
    fig.add_hline(y=int(sl_high), line_dash="dash", line_color=ACCENT,
                  line_width=1.5, opacity=0.6, row=1, col=1,
                  annotation_text=f"Max ({int(sl_high)} min)",
                  annotation_font_color=ACCENT)
    res_colors = [ACCENT_RED if v else "#b794f4" for v in anom_sleep["resid_anom"]]
    fig.add_trace(go.Bar(x=anom_sleep["Date"], y=anom_sleep["residual"],
                         name="Residual", marker_color=res_colors,
                         hovertemplate="<b>%{x|%d %b}</b><br>Δ: %{y:.0f} min<extra></extra>"),
                  row=2, col=1)
    fig.add_hline(y=0, line_color=MUTED, line_width=1, row=2, col=1)
    ptheme(fig, "💤 Sleep Pattern - Anomaly Visualization", h)
    fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT)
    fig.update_xaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
    return fig


# ════════════════════════════════════════════════════════════════════════════════
#  M4 PDF + CSV GENERATION
# ════════════════════════════════════════════════════════════════════════════════
def generate_pdf(master, anom_hr, anom_steps, anom_sleep,
                 hr_high, hr_low, st_low, sl_low, sl_high, sigma,
                 fig_hr, fig_steps, fig_sleep):
    """Generate a valid PDF and return raw bytes (not BytesIO)."""
    from fpdf import FPDF, XPos, YPos

    # ── stats ──────────────────────────────────────────────────────────────────
    n_hr    = int(anom_hr["is_anomaly"].sum())
    n_steps = int(anom_steps["is_anomaly"].sum())
    n_sleep = int(anom_sleep["is_anomaly"].sum())
    n_users = master["Id"].nunique()
    n_days  = master["Date"].nunique()
    date_range_str = (
        f"{pd.to_datetime(master['Date']).min().strftime('%d %b %Y')} - "
        f"{pd.to_datetime(master['Date']).max().strftime('%d %b %Y')}"
    )

    # ── PDF class ──────────────────────────────────────────────────────────────
    class PDF(FPDF):
        def header(self):
            self.set_fill_color(15, 23, 42)
            self.rect(0, 0, 210, 20, "F")
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(99, 179, 237)
            self.set_xy(0, 4)
            self.cell(210, 8, "FitPulse Anomaly Detection Report  -  Milestone 4", align="C")
            self.set_text_color(148, 163, 184)
            self.set_font("Helvetica", "", 7)
            self.set_xy(0, 13)
            self.cell(210, 5, f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}", align="C")
            self.set_xy(10, 24)

        def footer(self):
            self.set_y(-13)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(148, 163, 184)
            self.cell(0, 8, f"FitPulse ML Pipeline   |   Page {self.page_no()}", align="C")

        def section(self, title, color=(99, 179, 237)):
            self.ln(4)
            self.set_fill_color(*color)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 8, f"  {title}", fill=True,
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(30, 30, 40)
            self.ln(2)

        def kv(self, key, val):
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(80, 80, 100)
            self.cell(58, 6, f"{key}:", new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(20, 20, 30)
            self.cell(0, 6, str(val), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        def para(self, text, size=8.5):
            self.set_font("Helvetica", "", size)
            self.set_text_color(60, 60, 80)
            self.multi_cell(0, 5, text)
            self.ln(1)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Page 1: Summary & Methodology ─────────────────────────────────────────
    pdf.section("1. EXECUTIVE SUMMARY", (15, 23, 60))
    pdf.kv("Dataset",    "Real Fitbit Device Data - Kaggle (arashnic/fitbit)")
    pdf.kv("Users",      f"{n_users} participants")
    pdf.kv("Date Range", date_range_str)
    pdf.kv("Total Days", f"{n_days} days of observations")
    pdf.kv("Pipeline",   "Milestone 4 - Anomaly Detection Dashboard")

    pdf.section("2. ANOMALY SUMMARY", (180, 50, 50))
    pdf.kv("Heart Rate Anomalies", f"{n_hr} days flagged")
    pdf.kv("Steps Anomalies",      f"{n_steps} days flagged")
    pdf.kv("Sleep Anomalies",      f"{n_sleep} days flagged")
    pdf.kv("Total Flags",          f"{n_hr + n_steps + n_sleep} across all signals")

    pdf.section("3. DETECTION THRESHOLDS", (40, 100, 60))
    pdf.kv("HR High",        f"> {int(hr_high)} bpm")
    pdf.kv("HR Low",         f"< {int(hr_low)} bpm")
    pdf.kv("Steps Low",      f"< {int(st_low):,} steps/day")
    pdf.kv("Sleep Low",      f"< {int(sl_low)} min/night")
    pdf.kv("Sleep High",     f"> {int(sl_high)} min/night")
    pdf.kv("Residual Sigma", f"+/- {float(sigma):.1f} sigma from rolling median")

    pdf.section("4. METHODOLOGY", (60, 80, 140))
    pdf.para(
        "Three complementary anomaly detection methods were applied:\n\n"
        "1. THRESHOLD VIOLATIONS  -  Hard upper/lower bounds on each metric.\n\n"
        "2. RESIDUAL-BASED DETECTION  -  A 3-day rolling median is computed as "
        f"the baseline. Days deviating by more than +/-{float(sigma):.1f} SD are flagged.\n\n"
        "3. DBSCAN OUTLIER CLUSTERING  -  Users profiled on 7 activity features. "
        "DBSCAN (eps=2.2, min_samples=2) assigns label -1 to structural outliers."
    )

    # ── Page 2: Charts ─────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.section("5. ANOMALY CHARTS", (15, 23, 60))

    # ── Check kaleido availability once ───────────────────────────────────────
    _kaleido_ok = False
    try:
        import kaleido  # noqa
        _kaleido_ok = True
    except ImportError:
        pass

    def _plotly_to_png_bytes(fig):
        """Try kaleido first, then orca, then matplotlib fallback."""
        # 1. kaleido / orca (proper Plotly export)
        if _kaleido_ok:
            try:
                return fig.to_image(format="png", width=1000, height=420, scale=1.4)
            except Exception:
                pass
        try:
            return fig.to_image(format="png", width=1000, height=420, scale=1.4, engine="orca")
        except Exception:
            pass
        # 2. matplotlib fallback — draw a simple line chart from trace data
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            fig_mpl, ax = _plt.subplots(figsize=(10, 4.2), facecolor="#0f172a")
            ax.set_facecolor("#0f172a")
            plotted = False
            for trace in fig.data:
                xs = list(getattr(trace, "x", []) or [])
                ys = list(getattr(trace, "y", []) or [])
                if not xs or not ys:
                    continue
                name = getattr(trace, "name", "") or ""
                color = "#63b3ed"
                if "anomal" in name.lower() or "alert" in name.lower():
                    color = "#fc8181"
                elif "trend" in name.lower():
                    color = "#68d391"
                marker = getattr(trace, "mode", "") or ""
                lw = 1.5
                ms = 4
                if "markers" in marker and "lines" not in marker:
                    ax.scatter(xs, ys, color=color, s=ms*8, zorder=3, label=name)
                else:
                    ax.plot(xs, ys, color=color, linewidth=lw, label=name,
                            marker="o" if "markers" in marker else None,
                            markersize=ms)
                plotted = True
            ax.tick_params(colors="#94a3b8")
            ax.spines[:].set_color("#1e293b")
            ax.grid(color="#1e293b", linewidth=0.5)
            title = (fig.layout.title.text or "") if fig.layout.title else ""
            if title:
                ax.set_title(title, color="#e2e8f0", fontsize=10)
            if plotted:
                ax.legend(fontsize=7, facecolor="#0f172a", labelcolor="#e2e8f0",
                          edgecolor="#1e293b")
            buf_mpl = io.BytesIO()
            _plt.tight_layout()
            _plt.savefig(buf_mpl, format="png", dpi=110, bbox_inches="tight",
                         facecolor="#0f172a")
            _plt.close(fig_mpl)
            buf_mpl.seek(0)
            return buf_mpl.read()
        except Exception:
            return None

    def embed_fig(fig, label, w=190, h=82):
        """Embed a Plotly/Matplotlib figure as PNG in the PDF."""
        img_bytes = _plotly_to_png_bytes(fig)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(80, 80, 100)
        pdf.cell(0, 6, label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if img_bytes:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            try:
                tmp.write(img_bytes)
                tmp.flush()
                tmp.close()
                pdf.image(tmp.name, x=10, w=w, h=h)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
        else:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(160, 60, 60)
            pdf.cell(0, 6,
                     "  [Chart render failed — run: pip install kaleido]",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    embed_fig(fig_hr,    "Figure 1 - Heart Rate with Anomaly Highlights")
    embed_fig(fig_steps, "Figure 2 - Step Count Trend with Alert Bands")
    embed_fig(fig_sleep, "Figure 3 - Sleep Pattern Visualization")

    # ── Page 3: Anomaly tables ─────────────────────────────────────────────────
    pdf.add_page()

    def draw_table(df, cols, rename_map, max_rows=20):
        df2 = df[df["is_anomaly"]][cols].copy().rename(columns=rename_map)
        if df2.empty:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(100, 120, 140)
            pdf.cell(0, 6, "  No anomalies detected in selected range.",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            return
        col_w = 180 // len(df2.columns)
        # header row
        pdf.set_fill_color(20, 30, 70)
        pdf.set_text_color(160, 200, 255)
        pdf.set_font("Helvetica", "B", 7.5)
        for col in df2.columns:
            pdf.cell(col_w, 6, str(col)[:18], fill=True,
                     new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln()
        # data rows
        pdf.set_font("Helvetica", "", 7.5)
        for i, (_, row) in enumerate(df2.head(max_rows).iterrows()):
            if i % 2 == 0:
                pdf.set_fill_color(235, 240, 255)
            else:
                pdf.set_fill_color(245, 248, 255)
            pdf.set_text_color(30, 30, 50)
            for val in row:
                cell_text = (f"{val:.2f}" if isinstance(val, float)
                             else str(val)[:20])
                pdf.cell(col_w, 5.5, cell_text, fill=True,
                         new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.ln()
        if len(df2) > max_rows:
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(100, 120, 160)
            pdf.cell(0, 5, f"  ... and {len(df2)-max_rows} more records (see CSV export)",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    pdf.section("6. ANOMALY RECORDS - HEART RATE", (180, 50, 50))
    draw_table(anom_hr, ["Date","AvgHR","rolling_med","residual","reason"],
               {"AvgHR":"Avg HR","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

    pdf.section("7. ANOMALY RECORDS - STEPS", (40, 130, 80))
    draw_table(anom_steps, ["Date","TotalSteps","rolling_med","residual","reason"],
               {"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

    pdf.section("8. ANOMALY RECORDS - SLEEP", (100, 60, 160))
    draw_table(anom_sleep, ["Date","TotalSleepMinutes","rolling_med","residual","reason"],
               {"TotalSleepMinutes":"Sleep(min)","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

    # ── Page 4: User profiles & conclusion ────────────────────────────────────
    pdf.add_page()
    pdf.section("9. USER ACTIVITY PROFILES", (15, 23, 60))
    profile_cols   = ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
    available_cols = [c for c in profile_cols if c in master.columns]
    user_profile   = master.groupby("Id")[available_cols].mean().round(1)
    col_w2 = 180 // (len(available_cols) + 1)
    pdf.set_fill_color(20, 30, 70)
    pdf.set_text_color(160, 200, 255)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.cell(col_w2, 6, "User ID", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    for col in available_cols:
        pdf.cell(col_w2, 6, col[:13], fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln()
    pdf.set_font("Helvetica", "", 7.5)
    for i, (uid, row) in enumerate(user_profile.iterrows()):
        if i % 2 == 0:
            pdf.set_fill_color(235, 240, 255)
        else:
            pdf.set_fill_color(245, 248, 255)
        pdf.set_text_color(30, 30, 50)
        pdf.cell(col_w2, 5.5, f"...{str(uid)[-6:]}", fill=True,
                 new_x=XPos.RIGHT, new_y=YPos.TOP)
        for val in row:
            pdf.cell(col_w2, 5.5, f"{val:,.0f}", fill=True,
                     new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln()

    pdf.section("10. CONCLUSION", (40, 100, 60))
    pdf.para(
        f"The FitPulse Milestone 4 pipeline processed {n_users} users over {n_days} days "
        f"of real Fitbit device data. A total of {n_hr + n_steps + n_sleep} anomalous events "
        f"were identified across heart rate, step count, and sleep duration signals.\n\n"
        f"Heart Rate: {n_hr} anomalous days   |   Steps: {n_steps} anomalous days   "
        f"|   Sleep: {n_sleep} anomalous days"
    )

    # ── Output: return raw bytes (critical — do NOT return BytesIO) ────────────
    raw = pdf.output()                     # fpdf2 >= 2.5 returns bytearray
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return raw.encode("latin-1")           # older fpdf2 returned str

def generate_csv(anom_hr, anom_steps, anom_sleep):
    hr_out = anom_hr[anom_hr["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]].copy()
    hr_out["signal"] = "Heart Rate"
    hr_out = hr_out.rename(columns={"AvgHR":"value","rolling_med":"expected"})
    st_out = anom_steps[anom_steps["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]].copy()
    st_out["signal"] = "Steps"
    st_out = st_out.rename(columns={"TotalSteps":"value","rolling_med":"expected"})
    sl_out = anom_sleep[anom_sleep["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]].copy()
    sl_out["signal"] = "Sleep"
    sl_out = sl_out.rename(columns={"TotalSleepMinutes":"value","rolling_med":"expected"})
    combined = pd.concat([hr_out, st_out, sl_out], ignore_index=True)
    combined = combined[["signal","Date","value","expected","residual","reason"]].sort_values(["signal","Date"]).round(2)
    buf = io.StringIO()
    combined.to_csv(buf, index=False)
    return buf.getvalue().encode()

# File matching helper (M4)
REQUIRED_FILES_M4 = {
    "dailyActivity_merged.csv":     {"key_cols":["ActivityDate","TotalSteps","Calories"],      "label":"Daily Activity", "icon":"🏃"},
    "hourlySteps_merged.csv":       {"key_cols":["ActivityHour","StepTotal"],                  "label":"Hourly Steps",   "icon":"👣"},
    "hourlyIntensities_merged.csv": {"key_cols":["ActivityHour","TotalIntensity"],             "label":"Hourly Int.",    "icon":"⚡"},
    "minuteSleep_merged.csv":       {"key_cols":["date","value","logId"],                      "label":"Sleep",          "icon":"💤"},
    "heartrate_seconds_merged.csv": {"key_cols":["Time","Value"],                              "label":"Heart Rate",     "icon":"❤️"},
}
def score_match(df, req_info):
    return sum(1 for c in req_info["key_cols"] if c in df.columns)


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
    st.markdown(f'<div style="font-size:.72rem;color:{MUTED};font-family:JetBrains Mono,monospace;letter-spacing:.1em">ANOMALY THRESHOLDS (M3 & M4)</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    hr_high  = st.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180)
    hr_low   = st.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70)
    st_low   = st.number_input("Steps Low",        value=500, min_value=0,   max_value=2000)
    sl_low   = st.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120)
    sl_high  = st.number_input("Sleep High (min)", value=600, min_value=300, max_value=900)
    sigma    = st.slider("Residual σ threshold",   1.0, 4.0, 2.0, 0.5, key="sigma_slider")

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:1rem 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.72rem;color:{MUTED};font-family:JetBrains Mono,monospace;letter-spacing:.1em">M4 — DATA SOURCE</div>', unsafe_allow_html=True)

    # ── Reuse M2 data if already loaded, else show uploader ───────────────────
    _m2_ready = (st.session_state.data_loaded and
                 st.session_state.daily is not None and
                 st.session_state.hr    is not None and
                 st.session_state.sleep is not None)

    m4_detected   = {}
    m4_use_m2     = False  # flag: True → pull from session state, False → from uploader

    if _m2_ready:
        st.markdown(f'<div style="background:{SUCCESS_BG};border:1px solid {SUCCESS_BOR};border-radius:8px;padding:0.5rem 0.7rem;font-size:0.72rem;color:{ACCENT3};margin-bottom:0.5rem">✅ Using CSVs already loaded in Milestone 2 — no re-upload needed!</div>', unsafe_allow_html=True)
        m4_use_m2 = True
        # Populate m4_detected from session state so n_m4 check passes
        m4_detected = {k: True for k in REQUIRED_FILES_M4}  # sentinel — actual data pulled in run block
    else:
        st.markdown(f'<div style="font-size:.65rem;color:{MUTED};margin:4px 0 8px">Load data in Milestone 2 first, <i>or</i> drop all 5 CSV files here</div>', unsafe_allow_html=True)
        m4_uploaded = st.file_uploader("M4 Files", type="csv", accept_multiple_files=True,
                                       key="m4_uploader", label_visibility="collapsed")
        if m4_uploaded:
            raw_m4 = []
            for uf in m4_uploaded:
                try:
                    raw_m4.append((uf.name, pd.read_csv(uf)))
                except Exception:
                    pass
            for req_name, finfo in REQUIRED_FILES_M4.items():
                best_s, best_d = 0, None
                for uname, udf in raw_m4:
                    s = score_match(udf, finfo)
                    if s > best_s:
                        best_s, best_d = s, udf
                if best_s >= 2:
                    m4_detected[req_name] = best_d

    n_m4 = len(m4_detected)
    if not m4_use_m2:
        status_html = '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:3px;margin:0.5rem 0">'
        for req_name, finfo in REQUIRED_FILES_M4.items():
            found = req_name in m4_detected
            col_s = SUCCESS_BOR if found else WARN_BOR
            bg_s  = SUCCESS_BG  if found else WARN_BG
            ico   = "✅" if found else "❌"
            status_html += f'<div style="background:{bg_s};border:1px solid {col_s};border-radius:6px;padding:0.3rem;text-align:center;font-size:0.7rem">{ico}<br><span style="font-size:0.55rem;color:{MUTED}">{finfo["label"][:5]}</span></div>'
        status_html += "</div>"
        st.markdown(status_html, unsafe_allow_html=True)

    m4_run_clicked = st.button("⚡ Run M4 Pipeline", disabled=(n_m4 < 5), key="m4_run_btn")
    if n_m4 < 5:
        st.markdown(f'<div style="font-size:0.7rem;color:{MUTED};text-align:center">{n_m4}/5 files ready</div>', unsafe_allow_html=True)

    # Date/User filter — only shown after M4 pipeline is done
    m4_date_range  = None
    m4_selected_user = None
    if st.session_state.m4_pipeline_done and st.session_state.m4_master is not None:
        m4_master_tmp = st.session_state.m4_master
        all_dates_m4  = pd.to_datetime(m4_master_tmp["Date"])
        d_min_m4 = all_dates_m4.min().date()
        d_max_m4 = all_dates_m4.max().date()
        st.markdown(f'<hr style="border-color:{CARD_BOR};margin:0.8rem 0">', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.7rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin-bottom:0.4rem">DATE FILTER (M4)</div>', unsafe_allow_html=True)
        m4_date_range = st.date_input("Date range", value=(d_min_m4, d_max_m4),
                                      min_value=d_min_m4, max_value=d_max_m4,
                                      key="m4_daterange", label_visibility="collapsed")
        all_users_m4 = sorted(m4_master_tmp["Id"].unique())
        user_options_m4 = ["All Users"] + [f"...{str(u)[-6:]}" for u in all_users_m4]
        st.markdown(f'<div style="font-size:0.7rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin:0.6rem 0 0.4rem">USER FILTER (M4)</div>', unsafe_allow_html=True)
        sel_lbl = st.selectbox("User", user_options_m4, key="m4_user", label_visibility="collapsed")
        m4_selected_user = None if sel_lbl == "All Users" else all_users_m4[user_options_m4.index(sel_lbl) - 1]

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
        {_dot(st.session_state.simulation_done)} 🎯 &nbsp;M3 · Accuracy Simulation<br>
        {_dot(st.session_state.m4_pipeline_done)} 📊 &nbsp;M4 · Insights Dashboard
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:1rem 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.7rem;color:{MUTED};text-align:center;font-family:JetBrains Mono,monospace">Real Fitbit Dataset<br>30 users · Mar–Apr 2016<br>Minute-level HR data</div>', unsafe_allow_html=True)


# ── M4 Pipeline Run ────────────────────────────────────────────────────────────
if m4_run_clicked and n_m4 == 5:
    with st.spinner("⏳ M4: Building master DataFrame and detecting anomalies..."):
        try:
            # ── Path A: reuse already-parsed M2 session-state data ────────────
            if m4_use_m2 and st.session_state.master is not None:
                m4_master = st.session_state.master.copy()
                # Ensure Date column is plain date (not datetime)
                m4_master["Date"] = pd.to_datetime(m4_master["Date"]).dt.date

            # ── Path B: parse freshly uploaded files ──────────────────────────
            else:
                daily    = m4_detected["dailyActivity_merged.csv"].copy()
                hourly_s = m4_detected["hourlySteps_merged.csv"].copy()
                hourly_i = m4_detected["hourlyIntensities_merged.csv"].copy()
                sleep    = m4_detected["minuteSleep_merged.csv"].copy()
                hr       = m4_detected["heartrate_seconds_merged.csv"].copy()

                daily["ActivityDate"]    = pd.to_datetime(daily["ActivityDate"],    format="%m/%d/%Y", errors="coerce")
                hourly_s["ActivityHour"] = pd.to_datetime(hourly_s["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
                hourly_i["ActivityHour"] = pd.to_datetime(hourly_i["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
                sleep["date"]            = pd.to_datetime(sleep["date"],            format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
                hr["Time"]               = pd.to_datetime(hr["Time"],               format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

                hr_minute = (hr.set_index("Time").groupby("Id")["Value"]
                             .resample("1min").mean().reset_index())
                hr_minute.columns = ["Id","Time","HeartRate"]
                hr_minute = hr_minute.dropna()
                hr_minute["Date"] = hr_minute["Time"].dt.date

                hr_daily = (hr_minute.groupby(["Id","Date"])["HeartRate"]
                            .agg(["mean","max","min","std"]).reset_index()
                            .rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"}))
                sleep["Date"] = sleep["date"].dt.date
                sleep_daily = (sleep.groupby(["Id","Date"])
                               .agg(TotalSleepMinutes=("value","count"),
                                    DominantSleepStage=("value", lambda x: x.mode()[0])).reset_index())
                m4_master = daily.copy().rename(columns={"ActivityDate":"Date"})
                m4_master["Date"] = m4_master["Date"].dt.date
                m4_master = m4_master.merge(hr_daily,    on=["Id","Date"], how="left")
                m4_master = m4_master.merge(sleep_daily, on=["Id","Date"], how="left")
                m4_master["TotalSleepMinutes"]  = m4_master["TotalSleepMinutes"].fillna(0)
                m4_master["DominantSleepStage"] = m4_master.get("DominantSleepStage", pd.Series(0, index=m4_master.index)).fillna(0)
                for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
                    if col in m4_master.columns:
                        m4_master[col] = m4_master.groupby("Id")[col].transform(lambda x: x.fillna(x.median()))

            m4_anom_hr    = detect_hr(m4_master,    hr_high, hr_low,   sigma)
            m4_anom_steps = detect_steps(m4_master, st_low,  25000,    sigma)
            m4_anom_sleep = detect_sleep(m4_master, sl_low,  sl_high,  sigma)

            st.session_state.m4_master      = m4_master
            st.session_state.m4_anom_hr     = m4_anom_hr
            st.session_state.m4_anom_steps  = m4_anom_steps
            st.session_state.m4_anom_sleep  = m4_anom_sleep
            st.session_state.m4_pipeline_done = True
            st.rerun()
        except Exception as e:
            st.error(f"M4 Pipeline error: {e}")


# ════════════════════════════════════════════════════════════════════════════════
#  HERO
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="fp-hero">
  <div class="fp-hero-title">💪 FitPulse</div>
  <div class="fp-hero-sub">AI-Powered Health Analytics · All 4 Milestones in One App</div>
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
  <div class="ms-tab">
    <div class="ms-tab-icon">📊</div>
    <div class="ms-tab-title">Milestone 4</div>
    <div class="ms-tab-desc">Insights Dashboard · Filters · PDF & CSV Export</div>
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
        df = df_input.copy().reset_index(drop=True)
        df["t"] = (df["ds"] - df["ds"].min()).dt.days.astype(float)
        X_seas = np.column_stack([np.ones(len(df)), df["t"],
                                  np.sin(2*np.pi*df["t"]/7),
                                  np.cos(2*np.pi*df["t"]/7)])
        try:
            sc_arr = np.linalg.lstsq(X_seas, df["y"].values, rcond=None)[0]
        except Exception:
            sc_arr = np.array([df["y"].mean(), 0., 0., 0.])
        sigma_ = df["y"].std()
        future_dates = pd.date_range(start=df["ds"].max()+pd.Timedelta(days=1), periods=periods)
        all_dates = pd.concat([df["ds"], pd.Series(future_dates)]).reset_index(drop=True)
        t_all  = (all_dates - df["ds"].min()).dt.days.astype(float)
        X_all  = np.column_stack([np.ones(len(t_all)), t_all,
                                  np.sin(2*np.pi*t_all/7), np.cos(2*np.pi*t_all/7)])
        yhat = X_all @ sc_arr
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

                    daily["ActivityDate"] = pd.to_datetime(daily["ActivityDate"])
                    sleep["date"]         = pd.to_datetime(sleep["date"])
                    hr["Time"]            = pd.to_datetime(hr["Time"])

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
                    st.dataframe(hr_anom[["Date","AvgHR","rolling_med","residual","reason"]]
                                 .rename(columns={"rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"})
                                 .round(2), use_container_width=True)

            # ── CHART 2: Sleep ──────────────────────────────────────────────────
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            sec("💤", "Sleep Pattern — Anomaly Visualization", "Step 3")
            anom_tag(f"{n_sleep} anomalous sleep days detected")
            screenshot_badge("Sleep Pattern Visualization with Alerts")
            step_pill(3, "Threshold Detection on Sleep Minutes")
            ui_info(f"Diamond markers = anomaly days. Green band = healthy sleep zone ({sl_low}–{sl_high} min).")

            sleep_anom = anom_sleep[anom_sleep["is_anomaly"]]
            fig_sleep = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       row_heights=[0.7, 0.3],
                                       subplot_titles=["Sleep Duration (minutes/night)",
                                                       "Deviation from Expected"],
                                       vertical_spacing=0.08)
            fig_sleep.add_hrect(y0=sl_low, y1=sl_high, fillcolor="rgba(104,211,145,0.08)", line_width=0, row=1, col=1)
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
            fig_sleep.add_hline(y=sl_low, line_dash="dash", line_color=ACCENT_RED, line_width=1.5, opacity=0.7, row=1, col=1)
            fig_sleep.add_hline(y=sl_high, line_dash="dash", line_color=ACCENT, line_width=1.5, opacity=0.7, row=1, col=1)
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
                    st.dataframe(anom_sleep[anom_sleep["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]]
                                 .rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected",
                                                  "residual":"Deviation","reason":"Anomaly Reason"})
                                 .round(2), use_container_width=True)

            # ── CHART 3: Steps ──────────────────────────────────────────────────
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            sec("🚶", "Step Count Trend — Alerts & Anomalies", "Step 4")
            anom_tag(f"{n_steps} anomalous step-count days detected")
            screenshot_badge("Step Count Trend with Alert Bands")
            step_pill(4, "Threshold + Residual Detection on Steps")
            ui_info("Red vertical bands = anomaly alert days. Bar chart shows daily deviation from trend.")

            steps_anom = anom_steps[anom_steps["is_anomaly"]]
            fig_steps = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       row_heights=[0.65, 0.35],
                                       subplot_titles=["Daily Steps (avg across users)",
                                                       "Residual Deviation from Trend"],
                                       vertical_spacing=0.08)
            for _, row in steps_anom.iterrows():
                d = str(row["Date"])
                d_next = str(pd.Timestamp(d) + pd.Timedelta(days=1))[:10]
                fig_steps.add_vrect(x0=d, x1=d_next, fillcolor="rgba(252,129,129,0.15)",
                                     line_color="rgba(252,129,129,0.5)", line_width=1.5)
            fig_steps.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["TotalSteps"],
                                            mode="lines+markers", name="Avg Daily Steps",
                                            line=dict(color=ACCENT3, width=2.5),
                                            marker=dict(size=5, color=ACCENT3),
                                            hovertemplate="<b>%{x}</b><br>Steps: %{y:,.0f}<extra></extra>"),
                                 row=1, col=1)
            fig_steps.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["rolling_med"],
                                            mode="lines", name="Trend (Rolling Median)",
                                            line=dict(color=ACCENT, width=2, dash="dash")),
                                 row=1, col=1)
            if not steps_anom.empty:
                fig_steps.add_trace(go.Scatter(x=steps_anom["Date"], y=steps_anom["TotalSteps"],
                                                mode="markers", name="🚨 Steps Anomaly",
                                                marker=dict(color=ACCENT_RED, size=14, symbol="triangle-up",
                                                            line=dict(color="white", width=2)),
                                                hovertemplate="<b>%{x}</b><br>Steps: %{y:,.0f}<br><b>ALERT</b><extra>⚠️</extra>"),
                                     row=1, col=1)
            fig_steps.add_hline(y=st_low, line_dash="dash", line_color=ACCENT_RED, line_width=1.5, opacity=0.8, row=1, col=1)
            fig_steps.add_hline(y=25000, line_dash="dash", line_color=ACCENT2, line_width=1.5, opacity=0.7, row=1, col=1)
            res_colors_steps = [ACCENT_RED if v else ACCENT3 for v in anom_steps["resid_anomaly"]]
            fig_steps.add_trace(go.Bar(x=anom_steps["Date"], y=anom_steps["residual"],
                                        name="Residual", marker_color=res_colors_steps,
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
# ██  MILESTONE 4 · INSIGHTS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
dropdown_header("📊", "Milestone 4 · Insights Dashboard",
                "Upload · Detect · Filter by Date & User · PDF Report · CSV Export")

with st.expander("▼  Open Milestone 4 — Insights Dashboard", expanded=False):

    st.markdown(f"""
    <div class="m4-hero">
      <div class="hero-badge">MILESTONE 4 . INSIGHTS DASHBOARD</div>
      <h1 class="hero-title">📊 FitPulse Insights Dashboard</h1>
      <p class="hero-sub">Upload · Detect · Filter · Export PDF & CSV — Real Fitbit Device Data</p>
    </div>""", unsafe_allow_html=True)

    if st.session_state.data_loaded and st.session_state.master is not None:
        ui_success("Milestone 2 data detected — M4 will reuse it automatically.")
    else:
        ui_info("Load data in Milestone 2 first (recommended), or upload all 5 CSV files via the sidebar <b>M4 — Data Source</b> section.")

    # ── In-body Run Button (mirrors M1/M2/M3 pattern) ────────────────────────
    st.markdown(f'<div style="font-size:.72rem;color:{MUTED};font-family:JetBrains Mono,monospace;letter-spacing:.1em;margin-bottom:0.4rem">M4 — DATA SOURCE</div>', unsafe_allow_html=True)
    _m2_ready_body = (st.session_state.data_loaded and st.session_state.master is not None)
    _body_n_m4 = 5 if _m2_ready_body else len([k for k in REQUIRED_FILES_M4 if k in m4_detected]) if 'm4_detected' in dir() else 0
    m4_run_body = st.button(
        "⚡ Run M4 Pipeline",
        key="m4_run_body_btn",
        disabled=(not _m2_ready_body and _body_n_m4 < 5),
        help="Uses M2 data if loaded, otherwise upload 5 CSV files in the sidebar first"
    )
    if not _m2_ready_body and _body_n_m4 < 5:
        st.markdown(f'<div style="font-size:0.7rem;color:{MUTED}">{_body_n_m4}/5 files ready — upload remaining files in sidebar M4 section</div>', unsafe_allow_html=True)
    if m4_run_body:
        # Trigger the same pipeline as the sidebar button
        with st.spinner("⏳ M4: Building master DataFrame and detecting anomalies..."):
            try:
                if _m2_ready_body and st.session_state.master is not None:
                    _m4_master_tmp = st.session_state.master.copy()
                    _m4_master_tmp["Date"] = pd.to_datetime(_m4_master_tmp["Date"]).dt.date
                else:
                    _daily    = m4_detected["dailyActivity_merged.csv"].copy()
                    _hourly_s = m4_detected["hourlySteps_merged.csv"].copy()
                    _hourly_i = m4_detected["hourlyIntensities_merged.csv"].copy()
                    _sleep    = m4_detected["minuteSleep_merged.csv"].copy()
                    _hr       = m4_detected["heartrate_seconds_merged.csv"].copy()
                    _daily["ActivityDate"]    = pd.to_datetime(_daily["ActivityDate"],    format="%m/%d/%Y", errors="coerce")
                    _hourly_s["ActivityHour"] = pd.to_datetime(_hourly_s["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
                    _sleep["date"]            = pd.to_datetime(_sleep["date"],            format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
                    _hr["Time"]               = pd.to_datetime(_hr["Time"],               format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
                    _hr_minute = (_hr.set_index("Time").groupby("Id")["Value"].resample("1min").mean().reset_index())
                    _hr_minute.columns = ["Id","Time","HeartRate"]
                    _hr_minute = _hr_minute.dropna()
                    _hr_minute["Date"] = _hr_minute["Time"].dt.date
                    _hr_daily = (_hr_minute.groupby(["Id","Date"])["HeartRate"]
                                 .agg(["mean","max","min","std"]).reset_index()
                                 .rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"}))
                    _sleep["Date"] = _sleep["date"].dt.date
                    _sleep_daily = (_sleep.groupby(["Id","Date"])
                                    .agg(TotalSleepMinutes=("value","count"),
                                         DominantSleepStage=("value", lambda x: x.mode()[0])).reset_index())
                    _m4_master_tmp = _daily.copy().rename(columns={"ActivityDate":"Date"})
                    _m4_master_tmp["Date"] = _m4_master_tmp["Date"].dt.date
                    _m4_master_tmp = _m4_master_tmp.merge(_hr_daily,    on=["Id","Date"], how="left")
                    _m4_master_tmp = _m4_master_tmp.merge(_sleep_daily, on=["Id","Date"], how="left")
                    _m4_master_tmp["TotalSleepMinutes"] = _m4_master_tmp["TotalSleepMinutes"].fillna(0)
                    for _col in ["AvgHR","MaxHR","MinHR","StdHR"]:
                        if _col in _m4_master_tmp.columns:
                            _m4_master_tmp[_col] = _m4_master_tmp.groupby("Id")[_col].transform(lambda x: x.fillna(x.median()))
                st.session_state.m4_master      = _m4_master_tmp
                st.session_state.m4_anom_hr     = detect_hr(_m4_master_tmp,    hr_high, hr_low,  sigma)
                st.session_state.m4_anom_steps  = detect_steps(_m4_master_tmp, st_low,  25000,   sigma)
                st.session_state.m4_anom_sleep  = detect_sleep(_m4_master_tmp, sl_low,  sl_high, sigma)
                st.session_state.m4_pipeline_done = True
                st.rerun()
            except Exception as e:
                st.error(f"M4 Pipeline error: {e}")

    if not st.session_state.m4_pipeline_done:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:3rem">
          <div style="font-size:3rem;margin-bottom:1rem">📂</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:{TEXT};margin-bottom:0.5rem">
            Upload Files & Run Pipeline to Begin
          </div>
          <div style="color:{MUTED};font-size:0.88rem;margin-bottom:1.5rem">
            1 · Upload all 5 CSV files in the sidebar (M4 section)<br>
            2 · Adjust thresholds if needed<br>
            3 · Click <b>⚡ Run M4 Pipeline</b>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;max-width:600px;margin:0 auto;text-align:left">
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.8rem">
              <div style="color:{ACCENT};font-weight:600;font-size:0.85rem">📤 Upload</div>
              <div style="color:{MUTED};font-size:0.75rem;margin-top:0.2rem">All 5 Fitbit CSV files auto-detected</div>
            </div>
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.8rem">
              <div style="color:{ACCENT_RED};font-weight:600;font-size:0.85rem">🚨 Detect</div>
              <div style="color:{MUTED};font-size:0.75rem;margin-top:0.2rem">3 detection methods run automatically</div>
            </div>
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.8rem">
              <div style="color:{ACCENT3};font-weight:600;font-size:0.85rem">📥 Export</div>
              <div style="color:{MUTED};font-size:0.75rem;margin-top:0.2rem">Download PDF report + CSV data</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    else:
        m4_master     = st.session_state.m4_master
        m4_anom_hr    = st.session_state.m4_anom_hr
        m4_anom_steps = st.session_state.m4_anom_steps
        m4_anom_sleep = st.session_state.m4_anom_sleep

        # ── Apply date / user filters ─────────────────────────────────────────
        try:
            if m4_date_range and isinstance(m4_date_range, tuple) and len(m4_date_range) == 2:
                d_from, d_to = pd.Timestamp(m4_date_range[0]), pd.Timestamp(m4_date_range[1])
            else:
                all_d = pd.to_datetime(m4_master["Date"])
                d_from, d_to = all_d.min(), all_d.max()
        except Exception:
            all_d = pd.to_datetime(m4_master["Date"])
            d_from, d_to = all_d.min(), all_d.max()

        def filt(df, date_col="Date"):
            df2 = df.copy()
            df2[date_col] = pd.to_datetime(df2[date_col])
            return df2[(df2[date_col] >= d_from) & (df2[date_col] <= d_to)]

        anom_hr_f    = filt(m4_anom_hr)
        anom_steps_f = filt(m4_anom_steps)
        anom_sleep_f = filt(m4_anom_sleep)
        master_f     = filt(m4_master)
        if m4_selected_user:
            master_f = master_f[master_f["Id"] == m4_selected_user]

        # ── KPI strip ─────────────────────────────────────────────────────────
        n_hr_f    = int(anom_hr_f["is_anomaly"].sum())
        n_steps_f = int(anom_steps_f["is_anomaly"].sum())
        n_sleep_f = int(anom_sleep_f["is_anomaly"].sum())
        n_total_f = n_hr_f + n_steps_f + n_sleep_f
        n_users_f = master_f["Id"].nunique()
        n_days_f  = master_f["Date"].nunique()

        worst_hr_row = anom_hr_f[anom_hr_f["is_anomaly"]].copy()
        worst_hr_day = (worst_hr_row.iloc[worst_hr_row["residual"].abs().argmax()]["Date"].strftime("%d %b")
                        if not worst_hr_row.empty else "-")

        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi-card" style="border-color:{DANGER_BOR}">
            <div class="kpi-val" style="color:{ACCENT_RED}">{n_total_f}</div>
            <div class="kpi-label">Total Anomalies</div>
            <div class="kpi-sub">across all signals</div>
          </div>
          <div class="kpi-card" style="border-color:rgba(246,135,179,0.3)">
            <div class="kpi-val" style="color:{ACCENT2}">{n_hr_f}</div>
            <div class="kpi-label">HR Flags</div>
            <div class="kpi-sub">heart rate anomalies</div>
          </div>
          <div class="kpi-card" style="border-color:rgba(104,211,145,0.3)">
            <div class="kpi-val" style="color:{ACCENT3}">{n_steps_f}</div>
            <div class="kpi-label">Steps Alerts</div>
            <div class="kpi-sub">step count anomalies</div>
          </div>
          <div class="kpi-card" style="border-color:rgba(183,148,244,0.3)">
            <div class="kpi-val" style="color:#b794f4">{n_sleep_f}</div>
            <div class="kpi-label">Sleep Flags</div>
            <div class="kpi-sub">sleep anomalies</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-val" style="color:{ACCENT}">{n_users_f}</div>
            <div class="kpi-label">Users</div>
            <div class="kpi-sub">in selected range</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-val" style="color:{ACCENT_ORG}">{worst_hr_day}</div>
            <div class="kpi-label">Peak HR Anomaly</div>
            <div class="kpi-sub">highest deviation day</div>
          </div>
        </div>""", unsafe_allow_html=True)

        ui_success(f"Pipeline complete · {n_users_f} users · {n_days_f} days · {n_total_f} anomalies flagged")

        # ── Tabs ──────────────────────────────────────────────────────────────
        tab_overview, tab_hr, tab_steps, tab_sleep, tab_export = st.tabs([
            "📊 Overview", "❤️ Heart Rate", "🚶 Steps", "💤 Sleep", "📥 Export"
        ])

        # ── TAB 1: OVERVIEW ───────────────────────────────────────────────────
        with tab_overview:
            st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)
            sec("📅", "Combined Anomaly Timeline")

            all_anoms = []
            for df_, sig, col in [
                (anom_hr_f,    "Heart Rate", ACCENT2),
                (anom_steps_f, "Steps",      ACCENT3),
                (anom_sleep_f, "Sleep",      "#b794f4"),
            ]:
                a = df_[df_["is_anomaly"]].copy()
                a["signal"] = sig
                a["color"]  = col
                all_anoms.append(a[["Date","signal","color","reason"]])

            if all_anoms:
                combined_anom = pd.concat(all_anoms, ignore_index=True)
                combined_anom["Date"] = pd.to_datetime(combined_anom["Date"])
                combined_anom["y"]    = combined_anom["signal"]

                fig_timeline = go.Figure()
                for sig, col in [("Heart Rate", ACCENT2), ("Steps", ACCENT3), ("Sleep", "#b794f4")]:
                    sub = combined_anom[combined_anom["signal"] == sig]
                    if not sub.empty:
                        fig_timeline.add_trace(go.Scatter(
                            x=sub["Date"], y=sub["y"], mode="markers",
                            name=sig, marker=dict(color=col, size=14, symbol="diamond",
                                                  line=dict(color="white", width=2)),
                            hovertemplate=f"<b>{sig}</b><br>%{{x|%d %b %Y}}<br>%{{customdata}}<extra>⚠️ ANOMALY</extra>",
                            customdata=sub["reason"].values))
                ptheme(fig_timeline, "📅 Anomaly Event Timeline - All Signals", h=280)
                fig_timeline.update_layout(
                    xaxis_title="Date", yaxis_title="Signal", showlegend=True,
                    yaxis=dict(categoryorder="array",
                               categoryarray=["Sleep","Steps","Heart Rate"],
                               gridcolor=GRID_CLR, tickfont_color=MUTED))
                st.plotly_chart(fig_timeline, use_container_width=True)

            st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)
            sec("🗂️", "Recent Anomaly Log")
            if all_anoms:
                log = combined_anom.sort_values("Date", ascending=False).head(10)
                for _, row in log.iterrows():
                    st.markdown(f"""
                    <div class="anom-row">
                      <span style="font-size:0.9rem">🚨</span>
                      <span style="color:{row['color']};font-family:'JetBrains Mono',monospace;font-size:0.75rem;min-width:90px">{row['signal']}</span>
                      <span style="color:{MUTED};font-size:0.78rem;min-width:90px">{row['Date'].strftime('%d %b %Y')}</span>
                      <span style="color:{TEXT};font-size:0.78rem">{row['reason']}</span>
                    </div>""", unsafe_allow_html=True)

        # ── TAB 2: HEART RATE ─────────────────────────────────────────────────
        with tab_hr:
            sec("❤️", "Heart Rate - Deep Dive", f"{n_hr_f} anomalies")
            st.plotly_chart(chart_hr(anom_hr_f, hr_high, hr_low, sigma, h=420),
                            use_container_width=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="card">
                  <div class="card-title">HR Statistics</div>
                  <div style="font-size:0.83rem;line-height:2">
                    <div>Mean HR: <b style="color:{ACCENT}">{anom_hr_f['AvgHR'].mean():.1f} bpm</b></div>
                    <div>Max HR: <b style="color:{ACCENT_RED}">{anom_hr_f['AvgHR'].max():.1f} bpm</b></div>
                    <div>Min HR: <b style="color:{ACCENT2}">{anom_hr_f['AvgHR'].min():.1f} bpm</b></div>
                    <div>Anomaly days: <b style="color:{ACCENT_RED}">{n_hr_f}</b> of {len(anom_hr_f)} total</div>
                  </div>
                </div>""", unsafe_allow_html=True)
            with col_b:
                st.markdown(f'<div class="card"><div class="card-title">HR Anomaly Records</div>', unsafe_allow_html=True)
                hr_display = anom_hr_f[anom_hr_f["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]].round(2)
                if not hr_display.empty:
                    st.dataframe(hr_display.rename(columns={"AvgHR":"Avg HR","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                                 use_container_width=True, height=200)
                else:
                    ui_success("No HR anomalies in selected range")
                st.markdown('</div>', unsafe_allow_html=True)

        # ── TAB 3: STEPS ──────────────────────────────────────────────────────
        with tab_steps:
            sec("🚶", "Step Count - Deep Dive", f"{n_steps_f} alerts")
            st.plotly_chart(chart_steps(anom_steps_f, st_low, h=420),
                            use_container_width=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="card">
                  <div class="card-title">Steps Statistics</div>
                  <div style="font-size:0.83rem;line-height:2">
                    <div>Mean steps/day: <b style="color:{ACCENT3}">{anom_steps_f['TotalSteps'].mean():,.0f}</b></div>
                    <div>Max steps/day: <b style="color:{ACCENT}">{anom_steps_f['TotalSteps'].max():,.0f}</b></div>
                    <div>Min steps/day: <b style="color:{ACCENT_RED}">{anom_steps_f['TotalSteps'].min():,.0f}</b></div>
                    <div>Alert days: <b style="color:{ACCENT_RED}">{n_steps_f}</b> of {len(anom_steps_f)} total</div>
                  </div>
                </div>""", unsafe_allow_html=True)
            with col_b:
                st.markdown(f'<div class="card"><div class="card-title">Steps Alert Records</div>', unsafe_allow_html=True)
                st_display = anom_steps_f[anom_steps_f["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]].round(2)
                if not st_display.empty:
                    st.dataframe(st_display.rename(columns={"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                                 use_container_width=True, height=200)
                else:
                    ui_success("No step anomalies in selected range")
                st.markdown('</div>', unsafe_allow_html=True)

        # ── TAB 4: SLEEP ──────────────────────────────────────────────────────
        with tab_sleep:
            sec("💤", "Sleep Pattern - Deep Dive", f"{n_sleep_f} anomalies")
            st.plotly_chart(chart_sleep(anom_sleep_f, sl_low, sl_high, h=420),
                            use_container_width=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="card">
                  <div class="card-title">Sleep Statistics</div>
                  <div style="font-size:0.83rem;line-height:2">
                    <div>Mean sleep/night: <b style="color:#b794f4">{anom_sleep_f['TotalSleepMinutes'].mean():.0f} min</b></div>
                    <div>Max sleep/night: <b style="color:{ACCENT}">{anom_sleep_f['TotalSleepMinutes'].max():.0f} min</b></div>
                    <div>Min (non-zero): <b style="color:{ACCENT_RED}">{anom_sleep_f[anom_sleep_f['TotalSleepMinutes']>0]['TotalSleepMinutes'].min():.0f} min</b></div>
                    <div>Anomaly days: <b style="color:{ACCENT_RED}">{n_sleep_f}</b> of {len(anom_sleep_f)} total</div>
                  </div>
                </div>""", unsafe_allow_html=True)
            with col_b:
                st.markdown(f'<div class="card"><div class="card-title">Sleep Anomaly Records</div>', unsafe_allow_html=True)
                sl_display = anom_sleep_f[anom_sleep_f["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]].round(2)
                if not sl_display.empty:
                    st.dataframe(sl_display.rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                                 use_container_width=True, height=200)
                else:
                    ui_success("No sleep anomalies in selected range")
                st.markdown('</div>', unsafe_allow_html=True)

        # ── TAB 5: EXPORT ─────────────────────────────────────────────────────
        with tab_export:
            sec("📥", "Export - PDF Report & CSV Data", "Downloadable")

            st.markdown(f"""
            <div class="card">
              <div class="card-title">What's Included in the Exports</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;font-size:0.83rem">
                <div style="background:{SECTION_BG};border-radius:10px;padding:0.9rem">
                  <div style="color:{ACCENT};font-weight:600;margin-bottom:0.5rem">📄 PDF Report (4 pages)</div>
                  <div style="color:{MUTED};line-height:1.8">
                    ✅ Executive summary<br>✅ Anomaly counts per signal<br>
                    ✅ Thresholds used<br>✅ Methodology explanation<br>
                    ✅ All 3 charts embedded<br>✅ Full anomaly records tables<br>
                    ✅ User activity profiles
                  </div>
                </div>
                <div style="background:{SECTION_BG};border-radius:10px;padding:0.9rem">
                  <div style="color:{ACCENT3};font-weight:600;margin-bottom:0.5rem">📊 CSV Export</div>
                  <div style="color:{MUTED};line-height:1.8">
                    ✅ All anomaly records<br>✅ Signal type column<br>
                    ✅ Date of anomaly<br>✅ Actual vs expected value<br>
                    ✅ Residual deviation<br>✅ Anomaly reason text<br>
                    ✅ All signals combined
                  </div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)
            col_pdf, col_csv = st.columns(2)

            with col_pdf:
                sec("📄", "PDF Report")
                st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;margin-bottom:0.8rem">Full 4-page PDF with charts embedded, anomaly tables, and user profiles.</div>', unsafe_allow_html=True)
                if st.button("📄 Generate PDF Report", key="gen_pdf"):
                    with st.spinner("⏳ Generating PDF (embedding charts)..."):
                        try:
                            fig_hr_exp    = chart_hr(anom_hr_f,    hr_high, hr_low, sigma, h=420)
                            fig_steps_exp = chart_steps(anom_steps_f, st_low, h=420)
                            fig_sleep_exp = chart_sleep(anom_sleep_f, sl_low, sl_high, h=420)
                            pdf_buf = generate_pdf(
                                master_f, anom_hr_f, anom_steps_f, anom_sleep_f,
                                hr_high, hr_low, st_low, sl_low, sl_high, sigma,
                                fig_hr_exp, fig_steps_exp, fig_sleep_exp)
                            fname = f"FitPulse_Anomaly_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=pdf_buf, file_name=fname, mime="application/pdf", key="dl_pdf")
                            ui_success(f"PDF ready — {fname}")
                        except Exception as e:
                            st.error(f"PDF error: {e}")

            with col_csv:
                sec("📊", "CSV Export")
                st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;margin-bottom:0.8rem">All anomaly records from all three signals in a single CSV file.</div>', unsafe_allow_html=True)
                csv_data  = generate_csv(anom_hr_f, anom_steps_f, anom_sleep_f)
                fname_csv = f"FitPulse_Anomalies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                st.download_button(
                    label="⬇️ Download Anomaly CSV",
                    data=csv_data, file_name=fname_csv, mime="text/csv", key="dl_csv")
                with st.expander("👁️ Preview CSV data"):
                    preview_df = pd.concat([
                        anom_hr_f[anom_hr_f["is_anomaly"]].assign(signal="Heart Rate").rename(columns={"AvgHR":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                        anom_steps_f[anom_steps_f["is_anomaly"]].assign(signal="Steps").rename(columns={"TotalSteps":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                        anom_sleep_f[anom_sleep_f["is_anomaly"]].assign(signal="Sleep").rename(columns={"TotalSleepMinutes":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                    ], ignore_index=True).sort_values(["signal","Date"]).round(2)
                    st.dataframe(preview_df, use_container_width=True, height=280)

            st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)
            sec("📸", "Screenshots Required for Submission")
            st.markdown(f"""
            <div class="card">
              <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem;font-size:0.82rem">
                <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
                  <span style="color:{ACCENT2}">📸</span> <b>Screenshot 1</b> — Full dashboard UI (Overview tab)
                </div>
                <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
                  <span style="color:{ACCENT2}">📸</span> <b>Screenshot 2</b> — Downloadable report buttons (Export tab)
                </div>
                <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
                  <span style="color:{ACCENT2}">📸</span> <b>Screenshot 3</b> — KPI strip with anomaly counts
                </div>
                <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
                  <span style="color:{ACCENT2}">📸</span> <b>Screenshot 4</b> — HR / Steps / Sleep deep dive tabs
                </div>
                <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem;grid-column:1/-1">
                  <span style="color:{ACCENT2}">📸</span> <b>Screenshot 5</b> — Sidebar with filters + date range visible
                </div>
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
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">
""", unsafe_allow_html=True)

summary = [
    ("🧹","M1 · Data Cleaning",       st.session_state.original_df is not None,
     "CSV upload · null fill · datetime parse · preview"),
    ("🔧","M2 · Load & Parse",         bool(st.session_state.data_loaded),
     "5 Fitbit CSVs · master DataFrame · time normalization"),
    ("🧪","M2 · TSFresh Features",     bool(st.session_state.tsfresh_done),
     "MinimalFCParameters · normalized heatmap"),
    ("📈","M2 · Prophet Forecast",     bool(st.session_state.prophet_done),
     "HR + Steps + Sleep · 30-day · 80% CI"),
    ("🤖",f"M2 · KMeans (K={kmeans_k})", bool(st.session_state.cluster_done),
     "PCA · t-SNE · persona cards"),
    (f"🔍",f"M2 · DBSCAN (eps={dbscan_eps:.1f})", bool(st.session_state.cluster_done),
     "Density-based · noise detection"),
    ("🚨","M3 · Anomaly Detection",    bool(st.session_state.anomaly_done),
     "Threshold + Residual + DBSCAN · Plotly charts"),
    ("🎯","M3 · Accuracy Simulation",  bool(st.session_state.simulation_done),
     "10 injected anomalies per signal · 90%+ target"),
    ("📊","M4 · Insights Dashboard",   bool(st.session_state.m4_pipeline_done),
     "KPI strip · Timeline · Deep-dive tabs"),
    ("📥","M4 · PDF & CSV Export",     bool(st.session_state.m4_pipeline_done),
     "4-page PDF · anomaly CSV · embedded charts"),
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
  💪 FitPulse &nbsp;|&nbsp; Milestones 1 · 2 · 3 · 4 &nbsp;|&nbsp;
  Data Cleaning · TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE · Anomaly Detection · Insights Dashboard &nbsp;|&nbsp;
  Real Fitbit Dataset · Mar–Apr 2016
</div>
""", unsafe_allow_html=True)
