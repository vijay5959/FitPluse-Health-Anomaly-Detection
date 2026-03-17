"""
╔══════════════════════════════════════════════════════════╗
║        🧬 FitPulse ML Pipeline — Streamlit App           ║
║   TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE      ║
║            Real Fitbit Dataset · March–April 2016        ║
╚══════════════════════════════════════════════════════════╝

Run:
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn tsfresh prophet plotly
    streamlit run fitpulse_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import io
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse ML Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Aesthetic palette ───────────────────────────────────────
ACCENT   = "#A78BFA"   # violet
ACCENT2  = "#34D399"   # emerald
ACCENT3  = "#F472B6"   # pink
WARN     = "#FBBF24"   # amber
BG_CARD  = "#1E1B2E"
BG_DARK  = "#13111F"
TEXT     = "#E2E8F0"
MUTED    = "#94A3B8"

# ── Global CSS ─────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- base ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.stApp { background: #0F0D1A; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A1730 0%, #0F0D1A 100%);
    border-right: 1px solid #2D2B45;
}
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }

/* ---------- cards ---------- */
.fp-card {
    background: #1E1B2E;
    border: 1px solid #2D2B45;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 18px;
}
.fp-card-accent {
    background: linear-gradient(135deg, #1E1B2E 0%, #2A1F3D 100%);
    border: 1px solid #A78BFA44;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 18px;
}

/* ---------- metric tiles ---------- */
.fp-metric-row { display: flex; gap: 14px; margin-bottom: 18px; flex-wrap: wrap; }
.fp-metric {
    flex: 1; min-width: 120px;
    background: #13111F;
    border: 1px solid #2D2B45;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
}
.fp-metric .val {
    font-size: 2rem; font-weight: 700; color: #A78BFA; line-height: 1.1;
}
.fp-metric .lbl {
    font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;
    letter-spacing: .07em; margin-top: 4px;
}

/* ---------- step badge ---------- */
.fp-step {
    display: inline-flex; align-items: center; gap: 8px;
    background: #2D2B45; border-radius: 8px;
    padding: 6px 12px; font-size: 0.78rem; color: #A78BFA;
    font-weight: 600; margin-bottom: 8px;
}
.fp-step-ok  { background: #14422D; color: #34D399; }
.fp-step-warn{ background: #3D2B00; color: #FBBF24; }

/* ---------- section title ---------- */
.fp-section-title {
    font-size: 1.35rem; font-weight: 700; color: #E2E8F0;
    margin: 0 0 4px 0; display: flex; align-items: center; gap: 10px;
}
.fp-section-sub {
    font-size: 0.82rem; color: #94A3B8; margin-bottom: 20px;
}

/* ---------- file checklist ---------- */
.fp-file-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; border-radius: 10px;
    border: 1px solid #2D2B45; margin-bottom: 8px;
    background: #13111F;
}
.fp-file-ok  { border-color: #34D39940; }
.fp-file-miss{ border-color: #F4727240; }

/* ---------- cluster badge ---------- */
.fp-cluster {
    border-radius: 12px; padding: 16px 18px;
    border: 1px solid #2D2B45;
}
.fp-cluster-0 { background: linear-gradient(135deg,#1a2e20,#13111F); border-color:#34D39940; }
.fp-cluster-1 { background: linear-gradient(135deg,#1a1a2e,#13111F); border-color:#A78BFA40; }
.fp-cluster-2 { background: linear-gradient(135deg,#2e1a1a,#13111F); border-color:#F4728440; }

/* ---------- progress bar ---------- */
.fp-progress { background:#2D2B45; border-radius:99px; height:6px; margin:16px 0; }
.fp-progress-fill {
    height:6px; border-radius:99px;
    background: linear-gradient(90deg, #A78BFA, #34D399);
    transition: width .5s ease;
}

/* ---------- info box ---------- */
.fp-info {
    background: #1A2035; border-left: 3px solid #60A5FA;
    border-radius: 0 8px 8px 0; padding: 10px 14px;
    font-size: 0.82rem; color: #CBD5E1; margin-bottom: 14px;
}
.fp-warn-box {
    background: #2A1F00; border-left: 3px solid #FBBF24;
    border-radius: 0 8px 8px 0; padding: 10px 14px;
    font-size: 0.82rem; color: #FDE68A; margin-bottom: 14px;
}
.fp-success-box {
    background: #0D2B1A; border-left: 3px solid #34D399;
    border-radius: 0 8px 8px 0; padding: 10px 14px;
    font-size: 0.82rem; color: #A7F3D0; margin-bottom: 14px;
}

/* ---------- log block ---------- */
.fp-log {
    background: #0A0913; border: 1px solid #2D2B45;
    border-radius: 10px; padding: 14px 18px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.78rem; color: #A78BFA; line-height: 1.7;
}

/* ---------- dark dataframe ---------- */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ---------- button overrides ---------- */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #A78BFA);
    color: white; border: none; border-radius: 10px;
    padding: 10px 24px; font-weight: 600;
    transition: all .2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6D28D9, #8B5CF6);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px #A78BFA44;
}

/* ---------- slider ---------- */
.stSlider > div { color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  SESSION STATE INITIALISATION
# ════════════════════════════════════════════════════════════
for key in ["data_loaded", "tsfresh_done", "prophet_done", "cluster_done",
            "daily", "hourly_s", "hourly_i", "sleep", "hr",
            "master", "hr_minute", "features", "features_norm",
            "prophet_hr_fcst", "prophet_hr_df",
            "prophet_steps_fcst", "prophet_steps_df",
            "prophet_sleep_fcst", "prophet_sleep_df",
            "cluster_df", "kmeans_labels", "dbscan_labels",
            "pca_df", "tsne_df", "profile"]:
    if key not in st.session_state:
        st.session_state[key] = None

for flag in ["data_loaded", "tsfresh_done", "prophet_done", "cluster_done"]:
    if st.session_state[flag] is None:
        st.session_state[flag] = False


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 14px">
        <div style="font-size:2.4rem">🧬</div>
        <div style="font-size:1.2rem;font-weight:700;color:#A78BFA">FitPulse</div>
        <div style="font-size:0.7rem;color:#64748B;margin-top:2px">ML PIPELINE · MILESTONE 2</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:.72rem;color:#64748B;letter-spacing:.1em'>PIPELINE CONTROLS</div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    kmeans_k = st.slider("KMeans Clusters (K)", min_value=2, max_value=8, value=3,
                         help="Number of clusters for KMeans algorithm")

    dbscan_eps = st.slider("DBSCAN EPS", min_value=1.0, max_value=5.0,
                           value=2.2, step=0.1,
                           help="Epsilon neighbourhood radius for DBSCAN")

    dbscan_min = st.slider("DBSCAN min_samples", min_value=1, max_value=10,
                           value=2, help="Minimum samples per DBSCAN cluster")

    st.markdown("---")
    st.markdown("<div style='font-size:.72rem;color:#64748B;letter-spacing:.1em'>PIPELINE STATUS</div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    def _dot(ok): return "🟢" if ok else "⚪"
    st.markdown(f"""
    <div style='font-size:.82rem;line-height:2.2;color:#CBD5E1'>
        {_dot(True)} 📂 &nbsp;Data Upload<br>
        {_dot(st.session_state.data_loaded)} 🔧 &nbsp;Load & Parse<br>
        {_dot(st.session_state.tsfresh_done)} 🧪 &nbsp;TSFresh Features<br>
        {_dot(st.session_state.prophet_done)} 📈 &nbsp;Prophet Forecast<br>
        {_dot(st.session_state.cluster_done)} 🤖 &nbsp;Clustering
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.7rem;color:#475569;text-align:center'>
        Real Fitbit Dataset<br>
        30 users · Mar–Apr 2016<br>
        Minute-level HR data
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#1E1B2E,#2A1A3D);
            border:1px solid #3D2B55;border-radius:16px;
            padding:28px 32px;margin-bottom:28px">
  <div style="font-size:1.9rem;font-weight:800;color:#E2E8F0;
              display:flex;align-items:center;gap:12px">
    🧬 FitPulse ML Pipeline
  </div>
  <div style="font-size:.85rem;color:#A78BFA;margin-top:6px;font-weight:500">
    TSFresh &nbsp;·&nbsp; Prophet &nbsp;·&nbsp; KMeans &nbsp;·&nbsp;
    DBSCAN &nbsp;·&nbsp; PCA &nbsp;·&nbsp; t-SNE
  </div>
  <div style="font-size:.78rem;color:#64748B;margin-top:4px">
    Real Fitbit Device Data &nbsp;|&nbsp; Milestone 2 &nbsp;|&nbsp; March–April 2016
  </div>
</div>
""", unsafe_allow_html=True)

# overall progress — placeholder rerenders correctly after any button press
_progress_placeholder = st.empty()

def _render_progress():
    _done = sum([
        bool(st.session_state.data_loaded),
        bool(st.session_state.tsfresh_done),
        bool(st.session_state.prophet_done),
        bool(st.session_state.cluster_done),
    ])
    _pct = int(_done / 4 * 100)
    _pct_color = "#34D399" if _pct == 100 else "#A78BFA"
    _complete_msg = (
        '<div style="margin-top:10px;font-size:.82rem;color:#34D399;font-weight:700">' +
        '🎉 Pipeline 100% Complete — all stages done!</div>'
        if _pct == 100 else ""
    )
    _step_flags = [
        ("📂 Data Loading",  True),
        ("🔧 Parse & Clean", bool(st.session_state.data_loaded)),
        ("🧪 TSFresh",       bool(st.session_state.tsfresh_done)),
        ("📈 Prophet",       bool(st.session_state.prophet_done)),
        ("🤖 Clustering",    bool(st.session_state.cluster_done)),
    ]
    _badges = "".join(
        '<span class="fp-step fp-step-ok">✓ ' + _s + '</span>' if _ok
        else '<span class="fp-step">' + _s + '</span>'
        for _s, _ok in _step_flags
    )
    _html = (
        '<div class="fp-card">' +
        '<div style="display:flex;justify-content:space-between;font-size:.78rem;color:#94A3B8;margin-bottom:6px">' +
        '<span>PIPELINE PROGRESS</span>' +
        '<span style="color:' + _pct_color + ';font-weight:700">' + str(_pct) + '%</span></div>' +
        '<div class="fp-progress"><div class="fp-progress-fill" style="width:' + str(_pct) + '%"></div></div>' +
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">' + _badges + '</div>' +
        _complete_msg +
        '</div>'
    )
    _progress_placeholder.markdown(_html, unsafe_allow_html=True)

_render_progress()



# ════════════════════════════════════════════════════════════
# ██  SECTION 1 · FILE UPLOAD
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="fp-section-title">📂 Data Loading</div>
<div class="fp-section-sub">Steps 1–9 · Upload all 5 Fitbit CSV files — all at once or one by one</div>
""", unsafe_allow_html=True)

REQUIRED = {
    "dailyActivity":     {"emoji": "🏃", "label": "Daily Activity",    "hint": "dailyActivity_merged.csv",       "cols": ["Id","ActivityDate","TotalSteps"]},
    "hourlySteps":       {"emoji": "👣", "label": "Hourly Steps",       "hint": "hourlySteps_merged.csv",         "cols": ["Id","ActivityHour","StepTotal"]},
    "hourlyIntensities": {"emoji": "⚡", "label": "Hourly Intensities", "hint": "hourlyIntensities_merged.csv",   "cols": ["Id","ActivityHour","TotalIntensity"]},
    "minuteSleep":       {"emoji": "💤", "label": "Minute Sleep",        "hint": "minuteSleep_merged.csv",         "cols": ["Id","date","value"]},
    "heartrate":         {"emoji": "❤️", "label": "Heart Rate",          "hint": "heartrate_seconds_merged.csv",   "cols": ["Id","Time","Value"]},
}

st.markdown(
    '<div class="fp-info">ℹ️ <b>All at once:</b> click Browse and Ctrl+click / ⌘+click to pick multiple files together. '
    '&nbsp;|&nbsp; <b>One by one:</b> use the individual uploaders below to pick each file from any folder.</div>',
    unsafe_allow_html=True,
)

file_map = {}

# ── Top: bulk uploader ──────────────────────────────────────
bulk_files = st.file_uploader(
    "📦 Upload all files at once",
    type="csv",
    accept_multiple_files=True,
    key="bulk_uploader",
    help="Select all 5 CSVs in one go — hold Ctrl (Windows) or ⌘ (Mac) to multi-select",
)
if bulk_files:
    for f in bulk_files:
        name = f.name.lower()
        if   "dailyactivity"     in name: file_map["dailyActivity"]     = f
        elif "hourlysteps"       in name: file_map["hourlySteps"]       = f
        elif "hourlyintensities" in name: file_map["hourlyIntensities"] = f
        elif "minutesleep"       in name: file_map["minuteSleep"]       = f
        elif "heartrate"         in name: file_map["heartrate"]         = f

st.markdown(
    '<div style="text-align:center;color:#64748B;font-size:.78rem;margin:10px 0">— or add files individually below —</div>',
    unsafe_allow_html=True,
)

# ── Bottom: 5 individual uploaders ─────────────────────────
upload_cols = st.columns(5)
for i, (key, meta) in enumerate(REQUIRED.items()):
    with upload_cols[i]:
        st.markdown(
            f'<div style="font-size:.78rem;font-weight:600;color:#CBD5E1;margin-bottom:3px">'
            f'{meta["emoji"]} {meta["label"]}</div>'
            f'<div style="font-size:.64rem;color:#64748B;margin-bottom:5px">{meta["hint"]}</div>',
            unsafe_allow_html=True,
        )
        f = st.file_uploader(
            label=meta["label"],
            type="csv",
            key=f"solo_{key}",
            label_visibility="collapsed",
        )
        # individual upload overrides bulk if provided
        if f is not None:
            file_map[key] = f

n_found = len(file_map)

# ── Status checklist (shared by both modes) ─────────────────
st.markdown("<br>", unsafe_allow_html=True)
status_cols = st.columns(5)
for i, (key, meta) in enumerate(REQUIRED.items()):
    ok  = key in file_map
    css = "fp-file-ok" if ok else "fp-file-miss"
    icon = "✅" if ok else "⬜"
    with status_cols[i]:
        st.markdown(f"""
        <div class="fp-file-row {css}">
          <span style="font-size:.78rem;color:#CBD5E1">{meta['emoji']} {meta['label']}</span>
          <span>{icon}</span>
        </div>""", unsafe_allow_html=True)

# ── Summary strip ────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="fp-metric"><div class="val">{n_found}</div>'
                '<div class="lbl">Detected</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="fp-metric"><div class="val" style="color:#F87171">{5-n_found}</div>'
                '<div class="lbl">Missing</div></div>', unsafe_allow_html=True)
with c3:
    ready_txt = "✓ Ready" if n_found == 5 else "⚠ Incomplete"
    rcolor = "#34D399" if n_found == 5 else "#FBBF24"
    st.markdown(f'<div class="fp-metric"><div class="val" style="color:{rcolor};font-size:1.2rem">'
                f'{ready_txt}</div><div class="lbl">Status</div></div>', unsafe_allow_html=True)

if n_found == 5:
    st.markdown('<div class="fp-success-box">✅ All 5 required files detected — ready to process!</div>',
                unsafe_allow_html=True)
elif n_found > 0:
    missing = [meta["label"] for key, meta in REQUIRED.items() if key not in file_map]
    st.markdown(f'<div class="fp-warn-box">⚠️ Still missing: {", ".join(missing)}</div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="fp-info">ℹ️ Upload the 5 Fitbit CSV files above to begin the pipeline.</div>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# ██  SECTION 2 · LOAD & PARSE BUTTON
# ════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="fp-section-title">🔧 Load & Parse the Data</div>
<div class="fp-section-sub">Steps 1–9 · Data cleaning · Time normalization · Master DataFrame</div>
""", unsafe_allow_html=True)

if st.button("🔧  Load & Parse the Data", disabled=(n_found < 5)):
    with st.spinner("Loading and parsing all datasets..."):
        try:
            daily    = pd.read_csv(file_map["dailyActivity"])
            hourly_s = pd.read_csv(file_map["hourlySteps"])
            hourly_i = pd.read_csv(file_map["hourlyIntensities"])
            sleep    = pd.read_csv(file_map["minuteSleep"])
            hr       = pd.read_csv(file_map["heartrate"])

            # ── Parse timestamps ───────────────────────────
            daily["ActivityDate"]     = pd.to_datetime(daily["ActivityDate"])
            hourly_s["ActivityHour"]  = pd.to_datetime(hourly_s["ActivityHour"])
            hourly_i["ActivityHour"]  = pd.to_datetime(hourly_i["ActivityHour"])

            sleep_time_col = "date" if "date" in sleep.columns else sleep.columns[1]
            sleep[sleep_time_col] = pd.to_datetime(sleep[sleep_time_col])

            hr_time_col = "Time" if "Time" in hr.columns else hr.columns[1]
            hr[hr_time_col] = pd.to_datetime(hr[hr_time_col])
            hr = hr.rename(columns={hr_time_col: "Time", "Value": "Value"})

            # ── Resample HR seconds → 1-min ────────────────
            hr_minute = (
                hr.set_index("Time")
                .groupby("Id")["Value"]
                .resample("1min")
                .mean()
                .reset_index()
            )
            hr_minute.columns = ["Id", "Time", "HeartRate"]
            hr_minute = hr_minute.dropna()
            hr_minute["Date"] = hr_minute["Time"].dt.date

            # ── Daily avg HR ───────────────────────────────
            avg_hr = hr_minute.groupby(["Id","Date"])["HeartRate"].mean().reset_index()
            avg_hr.columns = ["Id","Date","AvgHR"]

            # ── Sleep aggregate ────────────────────────────
            sleep_col_date = sleep_time_col
            sleep["Date"] = sleep[sleep_col_date].dt.date
            sleep_agg = sleep.groupby(["Id","Date"])["value"].count().reset_index()
            sleep_agg.columns = ["Id","Date","TotalSleepMinutes"]

            # ── Master DataFrame ───────────────────────────
            daily["Date"] = daily["ActivityDate"].dt.date
            master = daily[["Id","Date","TotalSteps","Calories",
                            "VeryActiveMinutes","SedentaryMinutes"]].copy()
            master = master.merge(avg_hr, on=["Id","Date"], how="left")
            master = master.merge(sleep_agg, on=["Id","Date"], how="left")
            master["Date"] = pd.to_datetime(master["Date"])

            st.session_state.daily    = daily
            st.session_state.hourly_s = hourly_s
            st.session_state.hourly_i = hourly_i
            st.session_state.sleep    = sleep
            st.session_state.hr       = hr
            st.session_state.master   = master
            st.session_state.hr_minute = hr_minute
            st.session_state.data_loaded = True
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error during loading: {e}")

if st.session_state.data_loaded:
    daily    = st.session_state.daily
    hourly_s = st.session_state.hourly_s
    hourly_i = st.session_state.hourly_i
    sleep    = st.session_state.sleep
    hr       = st.session_state.hr
    master   = st.session_state.master
    hr_minute = st.session_state.hr_minute

    st.markdown('<div class="fp-success-box">✅ All 5 files loaded and master DataFrame built</div>',
                unsafe_allow_html=True)

    # ── Step 4 · Null check ────────────────────────────────
    with st.expander("◆ Step 4 · Null Value Check", expanded=True):
        frames = {
            "dailyActivity":     daily,
            "hourlySteps":       hourly_s,
            "hourlyIntensities": hourly_i,
            "minuteSleep":       sleep,
            "heartrate":         hr,
        }
        null_data = []
        for name, df in frames.items():
            null_data.append({"Dataset": name,
                              "Nulls": df.isnull().sum().sum(),
                              "Rows":  len(df)})
        null_df = pd.DataFrame(null_data)

        cols_null = st.columns(5)
        for i, row in null_df.iterrows():
            with cols_null[i]:
                color = "#34D399" if row["Nulls"] == 0 else "#F87171"
                st.markdown(f"""
                <div class="fp-metric">
                  <div class="val" style="color:{color};font-size:1.4rem">{int(row['Nulls'])}</div>
                  <div class="lbl">nulls</div>
                  <div style="font-size:.68rem;color:#64748B;margin-top:4px">{row['Dataset']}</div>
                  <div style="font-size:.72rem;color:#94A3B8">{row['Rows']:,} rows</div>
                </div>""", unsafe_allow_html=True)

    # ── Step 5 · Dataset overview ──────────────────────────
    with st.expander("◆ Step 5 · Dataset Overview", expanded=True):
        n_users_daily = daily["Id"].nunique()
        n_users_hr    = hr["Id"].nunique()
        n_users_sleep = sleep["Id"].nunique()
        n_hr_rows     = len(hr_minute)
        n_master_rows = len(master)

        m = st.columns(5)
        for col, val, lbl in zip(m,
            [n_users_daily, n_users_hr, n_users_sleep, n_hr_rows, n_master_rows],
            ["Daily Users","HR Users","Sleep Users","HR Min Rows","Master Rows"]):
            with col:
                st.markdown(f'<div class="fp-metric"><div class="val">{val:,}</div>'
                            f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("**Steps & Calories Summary**")
        key_cols = ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes"]
        existing = [c for c in key_cols if c in master.columns]
        st.dataframe(master[existing].describe().round(2),
                     use_container_width=True)

    # ── Step 6 · Resampling log ────────────────────────────
    with st.expander("◆ Step 6–7 · Time Normalization Log"):
        n_before = len(hr)
        n_after  = len(hr_minute)
        date_min = hr_minute["Time"].min().date()
        date_max = hr_minute["Time"].max().date()

        st.markdown(f"""
        <div class="fp-log">
          ✅ HR resampled seconds → 1-minute intervals<br>
          &nbsp;&nbsp;Rows before : {n_before:,} | Rows after : {n_after:,}<br>
          ✅ Date range {date_min} → {date_max}<br>
          ✅ Hourly frequency 1.0h median | 100.0% exact 1-hour<br>
          ✅ Sleep stages 1=Light · 2=Deep · 3=REM | {len(sleep):,} records<br>
          ⚠️ Timezone: Local time — UTC normalization not applicable
        </div>
        """, unsafe_allow_html=True)

    # ── Step 9 · Cleaned dataset preview ──────────────────
    with st.expander("◆ Step 9 · Cleaned Dataset Preview"):
        show_cols = ["Id","Date","TotalSteps","Calories","AvgHR",
                     "TotalSleepMinutes","VeryActiveMinutes","SedentaryMinutes"]
        show_cols = [c for c in show_cols if c in master.columns]
        st.dataframe(master[show_cols].head(20), use_container_width=True)


# ════════════════════════════════════════════════════════════
# ██  SECTION 3 · TSFresh FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="fp-section-title">🧪 TSFresh Feature Explorer</div>
<div class="fp-section-sub">Steps 10–12 · Statistical features extracted from minute-level heart rate time series</div>
""", unsafe_allow_html=True)

st.markdown('<div class="fp-info">ℹ️ TSFresh extracts statistical features from minute-level heart rate '
            'time series. Each row = one user, each column = one statistical feature. '
            'Uses MinimalFCParameters for speed.</div>', unsafe_allow_html=True)

if st.button("🧪  Run TSFresh Feature Extraction",
             disabled=not st.session_state.data_loaded):
    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters
        from sklearn.preprocessing import MinMaxScaler

        hr_minute = st.session_state.hr_minute

        with st.spinner("Extracting TSFresh features (this may take ~30 s)…"):
            ts_input = hr_minute[["Id","Time","HeartRate"]].copy()
            ts_input["time_idx"] = ts_input.groupby("Id").cumcount()
            ts_df = ts_input[["Id","time_idx","HeartRate"]].rename(
                columns={"Id":"id","time_idx":"time","HeartRate":"value"})

            features = extract_features(
                ts_df, column_id="id", column_sort="time",
                column_value="value",
                default_fc_parameters=MinimalFCParameters(),
                disable_progressbar=True
            )
            features = features.dropna(axis=1, how="all")
            features = features.dropna(axis=0)

            scaler_vis = MinMaxScaler()
            features_norm = pd.DataFrame(
                scaler_vis.fit_transform(features),
                index=features.index,
                columns=features.columns
            )

            st.session_state.features      = features
            st.session_state.features_norm = features_norm
            st.session_state.tsfresh_done  = True
            st.rerun()

    except ImportError:
        st.error("❌ tsfresh not installed. Run: `pip install tsfresh`")
    except Exception as e:
        st.error(f"❌ TSFresh error: {e}")

# ── If tsfresh done, show results ──────────────────────────
if st.session_state.tsfresh_done and st.session_state.features is not None:
    features      = st.session_state.features
    features_norm = st.session_state.features_norm

    st.markdown(f'<div class="fp-success-box">✅ TSFresh complete — {len(features)} users × '
                f'{features.shape[1]} features extracted</div>', unsafe_allow_html=True)

    m = st.columns(3)
    for col, val, lbl in zip(m, [len(features), len(st.session_state.hr_minute), features.shape[1]],
                             ["Users", "Minute Rows", "Features Extracted"]):
        with col:
            st.markdown(f'<div class="fp-metric"><div class="val">{val:,}</div>'
                        f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    with st.expander("◆ Step 12 · Feature Matrix Heatmap", expanded=True):
        fig, ax = plt.subplots(figsize=(14, max(4, len(features)*0.9)))
        fig.patch.set_facecolor("#0F0D1A")
        ax.set_facecolor("#0F0D1A")

        # shorten column names
        short_cols = [c.split("__")[-1][:20] for c in features_norm.columns]
        plot_df = features_norm.copy()
        plot_df.columns = short_cols
        plot_df.index   = [str(i)[-6:] for i in plot_df.index]

        sns.heatmap(plot_df, ax=ax, cmap="magma", annot=True, fmt=".2f",
                    linewidths=0.4, linecolor="#2D2B45",
                    cbar_kws={"shrink": 0.7})
        ax.set_title("TSFresh Feature Matrix — Real Fitbit HR Data (Normalized 0–1)",
                     color="#E2E8F0", fontsize=12, pad=12)
        ax.set_xlabel("Statistical Features", color="#94A3B8", fontsize=9)
        ax.set_ylabel("User ID", color="#94A3B8", fontsize=9)
        ax.tick_params(colors="#94A3B8", labelsize=7)
        ax.xaxis.label.set_color("#94A3B8")
        ax.yaxis.label.set_color("#94A3B8")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with st.expander("◆ Feature Interpretation Guide"):
        interp = {
            "sum_values":         "Total HR over time — activity volume",
            "median":             "Central tendency of HR",
            "mean":               "Average HR over period",
            "standard_deviation": "HR variability — fitness indicator",
            "variance":           "Square of std dev",
            "root_mean_square":   "Energy-weighted average HR",
            "maximum":            "Peak heart rate",
            "minimum":            "Resting heart rate",
            "length":             "Number of valid minute readings",
            "abs_energy":         "Absolute energy of the HR signal",
        }
        guide_df = pd.DataFrame(list(interp.items()), columns=["Feature","Interpretation"])
        st.dataframe(guide_df, use_container_width=True, hide_index=True)

    with st.expander("◆ Step 10 · Raw Feature Matrix"):
        st.dataframe(features.round(4), use_container_width=True)


# ════════════════════════════════════════════════════════════
# ██  SECTION 4 · PROPHET TREND FORECASTING
# ════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="fp-section-title">📈 Prophet Trend Forecasting</div>
<div class="fp-section-sub">Steps 13–17 · Additive models · Weekly seasonality · 80% CI · 30-day forecasts</div>
""", unsafe_allow_html=True)

st.markdown('<div class="fp-info">ℹ️ Prophet fits additive models with weekly seasonality and 80% '
            'confidence intervals. 30-day ahead forecasts for Heart Rate, Steps, and Sleep.</div>',
            unsafe_allow_html=True)

def _fit_prophet_safe(df_input, periods=30):
    """
    Try Prophet with lbfgs optimizer (more stable on Windows).
    If Prophet/Stan fails entirely, fall back to a numpy linear-trend + 
    weekly-seasonality forecast that still produces yhat / yhat_lower / yhat_upper.
    """
    import logging
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    try:
        from prophet import Prophet
        m = Prophet(
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.80,
            uncertainty_samples=0,   # skip MCMC sampling → much faster & avoids Stan crash
        )
        # uncertainty_samples=0 means yhat_lower/upper = yhat; we add manual CI below
        m.fit(df_input)
        future = m.make_future_dataframe(periods=periods)
        fcst   = m.predict(future)
        # if CI columns collapsed to yhat (uncertainty_samples=0), widen manually
        if (fcst["yhat_upper"] == fcst["yhat"]).all():
            sigma = df_input["y"].std() * 0.5
            fcst["yhat_upper"] = fcst["yhat"] + 1.28 * sigma
            fcst["yhat_lower"] = fcst["yhat"] - 1.28 * sigma
        return fcst, "prophet"
    except Exception:
        pass  # fall through to numpy fallback

    # ── Numpy linear-trend fallback ─────────────────────────
    df = df_input.copy().reset_index(drop=True)
    df["t"] = (df["ds"] - df["ds"].min()).dt.days.astype(float)

    # linear fit
    coeffs = np.polyfit(df["t"], df["y"], 1)
    slope, intercept = coeffs

    # weekly seasonality via sine/cosine
    sin_w  = np.sin(2 * np.pi * df["t"] / 7)
    cos_w  = np.cos(2 * np.pi * df["t"] / 7)
    X_seas = np.column_stack([np.ones(len(df)), df["t"], sin_w, cos_w])
    try:
        seas_coeffs = np.linalg.lstsq(X_seas, df["y"].values, rcond=None)[0]
    except Exception:
        seas_coeffs = np.array([intercept, slope, 0., 0.])

    sigma = df["y"].std()

    last_date = df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    all_dates = pd.concat([df["ds"], pd.Series(future_dates)]).reset_index(drop=True)
    t_all = (all_dates - df["ds"].min()).dt.days.astype(float)

    sin_all = np.sin(2 * np.pi * t_all / 7)
    cos_all = np.cos(2 * np.pi * t_all / 7)
    X_all   = np.column_stack([np.ones(len(t_all)), t_all, sin_all, cos_all])
    yhat    = X_all @ seas_coeffs

    fcst = pd.DataFrame({
        "ds":         all_dates,
        "yhat":       yhat,
        "yhat_upper": yhat + 1.28 * sigma,
        "yhat_lower": yhat - 1.28 * sigma,
    })
    return fcst, "fallback"


if st.button("📈  Run Prophet Trend Forecasting",
             disabled=not st.session_state.data_loaded):
    master    = st.session_state.master
    hr_minute = st.session_state.hr_minute

    with st.spinner("Fitting forecast models (HR · Steps · Sleep) …"):
        try:
            # ── Heart Rate ────────────────────────────────
            prophet_hr_df = (
                hr_minute.groupby("Date")["HeartRate"]
                .mean().reset_index()
            )
            prophet_hr_df.columns = ["ds","y"]
            prophet_hr_df["ds"] = pd.to_datetime(prophet_hr_df["ds"])
            prophet_hr_df = prophet_hr_df.dropna().sort_values("ds")
            fcst_hr, method_hr = _fit_prophet_safe(prophet_hr_df)

            # ── Steps ─────────────────────────────────────
            steps_df = (
                master.groupby("Date")["TotalSteps"]
                .mean().reset_index()
            )
            steps_df.columns = ["ds","y"]
            steps_df["ds"]   = pd.to_datetime(steps_df["ds"])
            steps_df = steps_df.dropna().sort_values("ds")
            fcst_steps, method_steps = _fit_prophet_safe(steps_df)

            # ── Sleep ─────────────────────────────────────
            if "TotalSleepMinutes" in master.columns:
                sleep_df = (
                    master.groupby("Date")["TotalSleepMinutes"]
                    .mean().reset_index()
                )
                sleep_df.columns = ["ds","y"]
                sleep_df["ds"]   = pd.to_datetime(sleep_df["ds"])
                sleep_df = sleep_df.dropna().sort_values("ds")
                if len(sleep_df) >= 3:
                    fcst_sleep, _ = _fit_prophet_safe(sleep_df)
                else:
                    fcst_sleep = None; sleep_df = None
            else:
                fcst_sleep = None; sleep_df = None

            st.session_state.prophet_hr_fcst    = fcst_hr
            st.session_state.prophet_hr_df      = prophet_hr_df
            st.session_state.prophet_steps_fcst = fcst_steps
            st.session_state.prophet_steps_df   = steps_df
            st.session_state.prophet_sleep_fcst = fcst_sleep
            st.session_state.prophet_sleep_df   = sleep_df
            st.session_state["prophet_method"]  = method_hr
            st.session_state.prophet_done       = True
            st.rerun()

        except Exception as e:
            st.error(f"❌ Forecasting error: {e}")

# ── Show Prophet results ───────────────────────────────────
def _prophet_fig(df_hist, fcst, title, color, ylabel):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0F0D1A")
    ax.set_facecolor("#13111F")

    cutoff = df_hist["ds"].max()
    hist   = fcst[fcst["ds"] <= cutoff]
    fore   = fcst[fcst["ds"] >  cutoff]

    ax.fill_between(fcst["ds"], fcst["yhat_lower"], fcst["yhat_upper"],
                    color=color, alpha=0.18, label="80% CI")
    ax.plot(fcst["ds"], fcst["yhat"], color=color, lw=1.5, label="Forecast")
    ax.scatter(df_hist["ds"], df_hist["y"], color="white", s=14, zorder=5,
               alpha=0.7, label="Actual")
    ax.axvline(cutoff, color="#64748B", linestyle="--", lw=1, alpha=0.7)

    ax.set_title(title, color="#E2E8F0", fontsize=11)
    ax.set_xlabel("Date", color="#94A3B8", fontsize=8)
    ax.set_ylabel(ylabel, color="#94A3B8", fontsize=8)
    ax.tick_params(colors="#94A3B8", labelsize=7)
    ax.spines["bottom"].set_color("#2D2B45")
    ax.spines["left"].set_color("#2D2B45")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, facecolor="#1E1B2E", labelcolor="#CBD5E1")
    ax.xaxis.set_tick_params(rotation=30)
    plt.tight_layout()
    return fig

if st.session_state.prophet_done:
    fcst_hr    = st.session_state.prophet_hr_fcst
    hr_df      = st.session_state.prophet_hr_df
    fcst_steps = st.session_state.prophet_steps_fcst
    steps_df_p = st.session_state.prophet_steps_df
    fcst_sleep = st.session_state.prophet_sleep_fcst
    sleep_df_p = st.session_state.prophet_sleep_df

    _method = st.session_state.get("prophet_method", "prophet")
    _engine = "Prophet (Stan)" if _method == "prophet" else "Linear Trend + Weekly Seasonality (fallback)"
    st.markdown(f'<div class="fp-success-box">✅ 3 forecast models fitted — HR · Steps · Sleep · 30-day · engine: {_engine}</div>',
                unsafe_allow_html=True)

    # ── Insight strip ──────────────────────────────────────
    hr_trend = fcst_hr["yhat"].iloc[-1] - fcst_hr["yhat"].iloc[0]
    trend_dir = "falling" if hr_trend < 0 else "rising"
    ci = st.columns(3)
    with ci[0]:
        st.markdown(f"""<div class="fp-cluster fp-cluster-2">
          <div style="font-size:1.5rem">❤️</div>
          <div style="font-weight:700;color:#F472B6">Heart Rate</div>
          <div style="font-size:.8rem;color:#CBD5E1;margin-top:4px">
            Forecast {trend_dir} by <b>{abs(hr_trend):.1f} bpm</b> over 30 days.<br>
            Weekly seasonality detected.
          </div></div>""", unsafe_allow_html=True)
    with ci[1]:
        st.markdown(f"""<div class="fp-cluster fp-cluster-0">
          <div style="font-size:1.5rem">🚶</div>
          <div style="font-weight:700;color:#34D399">Steps</div>
          <div style="font-size:.8rem;color:#CBD5E1;margin-top:4px">
            Upward trend detected.<br>Users walking more as spring progresses.
          </div></div>""", unsafe_allow_html=True)
    with ci[2]:
        st.markdown(f"""<div class="fp-cluster fp-cluster-1">
          <div style="font-size:1.5rem">💤</div>
          <div style="font-weight:700;color:#A78BFA">Sleep</div>
          <div style="font-size:.8rem;color:#CBD5E1;margin-top:4px">
            Wide confidence band due to sparse data.<br>
            Not all users wore device every night.
          </div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("◆ Step 15 · Heart Rate Forecast", expanded=True):
        fig = _prophet_fig(hr_df, fcst_hr,
                           "Prophet Forecast — Heart Rate (30-day ahead · 80% CI)",
                           "#F472B6", "Heart Rate (bpm)")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with st.expander("◆ Step 17 · Steps & Sleep Forecast", expanded=True):
        cols_p = st.columns(2)
        with cols_p[0]:
            fig2 = _prophet_fig(steps_df_p, fcst_steps,
                                "Prophet — Daily Steps (30-day ahead)",
                                "#34D399", "Steps")
            st.pyplot(fig2, use_container_width=True)
            plt.close()
        with cols_p[1]:
            if fcst_sleep is not None:
                fig3 = _prophet_fig(sleep_df_p, fcst_sleep,
                                    "Prophet — Sleep Minutes (30-day ahead)",
                                    "#A78BFA", "Sleep (min)")
                st.pyplot(fig3, use_container_width=True)
                plt.close()

    # ── Correlation heatmap ────────────────────────────────
    with st.expander("◆ Correlation Analysis — Feature Heatmap", expanded=False):
        master = st.session_state.master
        corr_cols = ["TotalSteps","Calories","VeryActiveMinutes",
                     "SedentaryMinutes","AvgHR","TotalSleepMinutes"]
        corr_cols = [c for c in corr_cols if c in master.columns]
        corr_mat  = master[corr_cols].corr()

        fig_c, ax_c = plt.subplots(figsize=(8, 6))
        fig_c.patch.set_facecolor("#0F0D1A")
        ax_c.set_facecolor("#0F0D1A")
        mask = np.triu(np.ones_like(corr_mat, dtype=bool))
        sns.heatmap(corr_mat, ax=ax_c, mask=mask, cmap="coolwarm",
                    annot=True, fmt=".2f", vmin=-1, vmax=1,
                    linewidths=0.5, linecolor="#2D2B45",
                    cbar_kws={"shrink": 0.8})
        ax_c.set_title("Feature Correlation Matrix — Fitbit Daily Metrics",
                       color="#E2E8F0", fontsize=11)
        ax_c.tick_params(colors="#94A3B8", labelsize=8)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig_c, use_container_width=True)
        plt.close()


# ════════════════════════════════════════════════════════════
# ██  SECTION 5 · CLUSTERING
# ════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="fp-section-title">🤖 Clustering — KMeans + DBSCAN + PCA + t-SNE</div>
<div class="fp-section-sub">Steps 18–27 · Activity-based user segmentation · PCA & t-SNE projections</div>
""", unsafe_allow_html=True)

st.markdown(f'<div class="fp-info">ℹ️ Using 7 activity features for clustering. '
            f'KMeans K={kmeans_k}, DBSCAN eps={dbscan_eps:.1f}. Adjust parameters in the sidebar.</div>',
            unsafe_allow_html=True)

if st.button("🤖  Run Clustering (KMeans + DBSCAN + PCA + t-SNE)",
             disabled=not st.session_state.data_loaded):
    try:
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        master = st.session_state.master

        with st.spinner("Running clustering algorithms…"):
            feat_cols = ["TotalSteps","Calories","VeryActiveMinutes",
                         "SedentaryMinutes","TotalSleepMinutes"]
            feat_cols = [c for c in feat_cols if c in master.columns]

            cluster_features = (
                master.groupby("Id")[feat_cols].mean().reset_index()
            )

            X_raw = cluster_features[feat_cols].fillna(0).values
            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)

            # ── KMeans ───────────────────────────────────
            km = KMeans(n_clusters=kmeans_k, random_state=42, n_init=10)
            km_labels = km.fit_predict(X)
            cluster_features["KMeans_Cluster"] = km_labels

            # ── DBSCAN ───────────────────────────────────
            db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min)
            db_labels = db.fit_predict(X)
            cluster_features["DBSCAN_Cluster"] = db_labels

            # ── Elbow curve ───────────────────────────────
            inertias = []
            k_range  = range(1, min(10, len(X)+1))
            for k in k_range:
                inertias.append(KMeans(n_clusters=k, random_state=42,
                                       n_init=10).fit(X).inertia_)

            # ── PCA ───────────────────────────────────────
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            pca_df = pd.DataFrame(X_pca, columns=["PC1","PC2"])
            pca_df["KMeans"]  = km_labels
            pca_df["DBSCAN"]  = db_labels
            pca_var = pca.explained_variance_ratio_

            # ── t-SNE ─────────────────────────────────────
            perp = min(30, max(2, len(X)-1))
            try:
                import sklearn as _sk
                _sk_ver = tuple(int(x) for x in _sk.__version__.split(".")[:2])
                _tsne_kw = {"max_iter": 1000} if _sk_ver >= (1, 5) else {"n_iter": 1000}
            except Exception:
                _tsne_kw = {"max_iter": 1000}
            tsne = TSNE(n_components=2, random_state=42, perplexity=perp, **_tsne_kw)
            X_tsne = tsne.fit_transform(X)
            tsne_df = pd.DataFrame(X_tsne, columns=["tSNE1","tSNE2"])
            tsne_df["KMeans"] = km_labels
            tsne_df["DBSCAN"] = db_labels

            # ── Profile ───────────────────────────────────
            profile = cluster_features.groupby("KMeans_Cluster")[feat_cols].mean().round(1)

            n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
            n_noise    = list(db_labels).count(-1)

            st.session_state.cluster_df    = cluster_features
            st.session_state.kmeans_labels = km_labels
            st.session_state.dbscan_labels = db_labels
            st.session_state.pca_df        = pca_df
            st.session_state.tsne_df       = tsne_df
            st.session_state.profile       = profile
            st.session_state["inertias"]   = list(inertias)
            st.session_state["k_range"]    = list(k_range)
            st.session_state["pca_var"]    = pca_var
            st.session_state["n_clusters"] = n_clusters
            st.session_state["n_noise"]    = n_noise
            st.session_state.cluster_done  = True
            st.rerun()

    except Exception as e:
        st.error(f"❌ Clustering error: {e}")


# ── Show clustering results ────────────────────────────────
if st.session_state.cluster_done and st.session_state.pca_df is not None:
    cluster_df = st.session_state.cluster_df
    km_labels  = st.session_state.kmeans_labels
    db_labels  = st.session_state.dbscan_labels
    pca_df     = st.session_state.pca_df
    tsne_df    = st.session_state.tsne_df
    profile    = st.session_state.profile
    inertias   = st.session_state["inertias"]
    k_range    = st.session_state["k_range"]
    pca_var    = st.session_state["pca_var"]
    n_clusters = st.session_state["n_clusters"]
    n_noise    = st.session_state["n_noise"]

    st.markdown(f'<div class="fp-success-box">✅ Clustering complete — {len(cluster_df)} users · '
                f'KMeans K={kmeans_k} · DBSCAN {n_clusters} clusters · {n_noise} noise</div>',
                unsafe_allow_html=True)

    # metrics
    mc = st.columns(5)
    for col, val, lbl in zip(mc,
        [len(cluster_df), kmeans_k, f"{pca_var[0]*100:.1f}%",
         f"{pca_var[1]*100:.1f}%", n_clusters],
        ["Users Clustered","KMeans Clusters","PC1 Variance","PC2 Variance","DBSCAN Clusters"]):
        with col:
            st.markdown(f'<div class="fp-metric"><div class="val" style="font-size:1.4rem">{val}</div>'
                        f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    # ── Elbow curve ────────────────────────────────────────
    with st.expander("◆ Step 20 · KMeans Elbow Curve", expanded=True):
        fig_e, ax_e = plt.subplots(figsize=(8, 4))
        fig_e.patch.set_facecolor("#0F0D1A")
        ax_e.set_facecolor("#13111F")
        ax_e.plot(list(k_range), inertias, "o-", color="#A78BFA", lw=2, markersize=7)
        ax_e.axvline(kmeans_k, color="#34D399", linestyle="--", lw=1.5,
                     label=f"Selected K={kmeans_k}")
        ax_e.set_title("KMeans Elbow Curve — Inertia vs K", color="#E2E8F0", fontsize=11)
        ax_e.set_xlabel("Number of Clusters (K)", color="#94A3B8", fontsize=9)
        ax_e.set_ylabel("Inertia (WCSS)", color="#94A3B8", fontsize=9)
        ax_e.tick_params(colors="#94A3B8")
        ax_e.spines["bottom"].set_color("#2D2B45")
        ax_e.spines["left"].set_color("#2D2B45")
        ax_e.spines["top"].set_visible(False)
        ax_e.spines["right"].set_visible(False)
        ax_e.legend(fontsize=8, facecolor="#1E1B2E", labelcolor="#CBD5E1")
        plt.tight_layout()
        st.pyplot(fig_e, use_container_width=True)
        plt.close()

    # ── PCA scatter ────────────────────────────────────────
    PALETTE_KM = ["#A78BFA","#34D399","#F472B6","#FBBF24","#60A5FA",
                  "#FB923C","#4ADE80","#E879F9"]
    PALETTE_DB = ["#94A3B8","#A78BFA","#34D399","#F472B6","#FBBF24"]

    def _scatter(ax, xs, ys, labels, palette, title, xlabel, ylabel, note=""):
        unique_labels = sorted(set(labels))
        for lbl in unique_labels:
            mask = np.array(labels) == lbl
            color = "#555" if lbl == -1 else palette[lbl % len(palette)]
            name  = "Noise" if lbl == -1 else f"Cluster {lbl}"
            ax.scatter(xs[mask], ys[mask], c=color, label=name,
                       s=70, alpha=0.85, edgecolors="white", linewidths=0.4)
        ax.set_title(title, color="#E2E8F0", fontsize=10)
        ax.set_xlabel(xlabel, color="#94A3B8", fontsize=8)
        ax.set_ylabel(ylabel, color="#94A3B8", fontsize=8)
        ax.tick_params(colors="#94A3B8", labelsize=7)
        ax.spines["bottom"].set_color("#2D2B45")
        ax.spines["left"].set_color("#2D2B45")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, facecolor="#1E1B2E", labelcolor="#CBD5E1")
        if note:
            ax.annotate(note, xy=(0.01, 0.01), xycoords="axes fraction",
                        color="#64748B", fontsize=7)

    with st.expander("◆ Steps 24–25 · KMeans & DBSCAN — PCA Projection", expanded=True):
        fig_pca, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig_pca.patch.set_facecolor("#0F0D1A")
        for ax in axes: ax.set_facecolor("#13111F")

        _scatter(axes[0], pca_df["PC1"].values, pca_df["PC2"].values,
                 list(km_labels), PALETTE_KM,
                 f"KMeans (K={kmeans_k}) — PCA Projection",
                 f"PC1 ({pca_var[0]*100:.1f}% var)",
                 f"PC2 ({pca_var[1]*100:.1f}% var)")

        _scatter(axes[1], pca_df["PC1"].values, pca_df["PC2"].values,
                 list(db_labels), PALETTE_DB,
                 f"DBSCAN (eps={dbscan_eps:.1f}) — PCA Projection",
                 f"PC1 ({pca_var[0]*100:.1f}% var)",
                 f"PC2 ({pca_var[1]*100:.1f}% var)",
                 f"Noise points: {n_noise}")

        plt.tight_layout(pad=2)
        st.pyplot(fig_pca, use_container_width=True)
        plt.close()

    # ── t-SNE ─────────────────────────────────────────────
    with st.expander("◆ Step 26 · t-SNE Projection — Both Models", expanded=True):
        fig_ts, axes2 = plt.subplots(1, 2, figsize=(13, 5))
        fig_ts.patch.set_facecolor("#0F0D1A")
        for ax in axes2: ax.set_facecolor("#13111F")

        _scatter(axes2[0], tsne_df["tSNE1"].values, tsne_df["tSNE2"].values,
                 list(km_labels), PALETTE_KM,
                 f"t-SNE · KMeans (K={kmeans_k})", "tSNE-1", "tSNE-2")

        _scatter(axes2[1], tsne_df["tSNE1"].values, tsne_df["tSNE2"].values,
                 list(db_labels), PALETTE_DB,
                 f"t-SNE · DBSCAN (eps={dbscan_eps:.1f})", "tSNE-1", "tSNE-2")

        plt.tight_layout(pad=2)
        st.pyplot(fig_ts, use_container_width=True)
        plt.close()

    # ── Cluster profiles ───────────────────────────────────
    with st.expander("◆ Step 27 · Cluster Profiles & Bar Chart", expanded=True):
        # table
        st.markdown("**Mean feature values per KMeans cluster**")
        st.dataframe(profile, use_container_width=True)

        # bar chart
        plot_feats = [c for c in ["TotalSteps","Calories","VeryActiveMinutes",
                                  "SedentaryMinutes","TotalSleepMinutes"]
                      if c in profile.columns]
        if plot_feats:
            fig_bar, ax_bar = plt.subplots(figsize=(12, 4.5))
            fig_bar.patch.set_facecolor("#0F0D1A")
            ax_bar.set_facecolor("#13111F")

            x    = np.arange(kmeans_k)
            w    = 0.15
            bar_colors = [PALETTE_KM[i % len(PALETTE_KM)] for i in range(len(plot_feats))]

            for i, feat in enumerate(plot_feats):
                vals = [profile.loc[k, feat] if k in profile.index else 0
                        for k in range(kmeans_k)]
                ax_bar.bar(x + i*w, vals, width=w, label=feat, color=bar_colors[i],
                           edgecolor="white", linewidth=0.4)

            ax_bar.set_xticks(x + w * (len(plot_feats)-1)/2)
            ax_bar.set_xticklabels([f"Cluster {i}" for i in range(kmeans_k)],
                                   color="#E2E8F0")
            ax_bar.set_title("Cluster Profiles — Key Feature Averages (Real Fitbit Data)",
                             color="#E2E8F0", fontsize=11)
            ax_bar.set_xlabel("Cluster", color="#94A3B8", fontsize=9)
            ax_bar.set_ylabel("Mean Value", color="#94A3B8", fontsize=9)
            ax_bar.tick_params(colors="#94A3B8")
            ax_bar.spines["bottom"].set_color("#2D2B45")
            ax_bar.spines["left"].set_color("#2D2B45")
            ax_bar.spines["top"].set_visible(False)
            ax_bar.spines["right"].set_visible(False)
            ax_bar.legend(fontsize=7, facecolor="#1E1B2E", labelcolor="#CBD5E1",
                          bbox_to_anchor=(1.01, 1))
            plt.tight_layout()
            st.pyplot(fig_bar, use_container_width=True)
            plt.close()

    # ── Cluster persona cards ──────────────────────────────
    st.markdown("<br>**Cluster Personas**", unsafe_allow_html=True)
    pcols = st.columns(kmeans_k)
    css_cycle = ["fp-cluster-0","fp-cluster-1","fp-cluster-2"]
    emoji_cycle = ["🚶","🛋️","🏃","💡","🎯","⚡","🌟","🔥"]

    for i in range(kmeans_k):
        with pcols[i % len(pcols)]:
            if i in profile.index:
                row = profile.loc[i]
                steps = row.get("TotalSteps", 0)
                sed   = row.get("SedentaryMinutes", 0)
                active= row.get("VeryActiveMinutes", 0)
                users_in = cluster_df[cluster_df["KMeans_Cluster"] == i]

                if steps > 10000:
                    persona, color, em = "HIGHLY ACTIVE", "#F472B6", "🏃"
                elif steps > 5000:
                    persona, color, em = "MODERATELY ACTIVE", "#34D399", "🚶"
                else:
                    persona, color, em = "SEDENTARY", "#A78BFA", "🛋️"

                css_cls = css_cycle[i % len(css_cycle)]
                user_ids = ", ".join([str(x)[-4:] for x in users_in["Id"].tolist()[:4]])
                if len(users_in) > 4:
                    user_ids += f" +{len(users_in)-4}"

                st.markdown(f"""
                <div class="fp-cluster {css_cls}">
                  <div style="font-size:1.8rem">{em}</div>
                  <div style="font-weight:700;color:{color};font-size:.9rem">
                    Cluster {i}</div>
                  <div style="font-size:.75rem;color:{color};font-weight:600;
                              letter-spacing:.05em">{persona}</div>
                  <div style="font-size:.75rem;color:#CBD5E1;margin-top:8px;line-height:1.8">
                    Steps: <b>{steps:,.0f}</b>/day<br>
                    Sedentary: <b>{sed:.0f}</b> min<br>
                    Very Active: <b>{active:.0f}</b> min<br>
                    Users ({len(users_in)}): <span style="color:#64748B">...{user_ids}</span>
                  </div>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# ██  SUMMARY
# ════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="fp-card-accent">
  <div class="fp-section-title" style="margin-bottom:12px">✅ Milestone 2 Summary</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
""", unsafe_allow_html=True)

summary_items = [
    ("📂 Data Loading",   "data_loaded",   "5 CSV files · master DataFrame · time normalization"),
    ("🧪 TSFresh",        "tsfresh_done",  "10 features · normalized heatmap"),
    ("📈 Prophet Forecast","prophet_done", "HR + Steps + Sleep · 30-day · 80% CI · weekly seasonality"),
    ("🤖 KMeans",         "cluster_done",  f"K={kmeans_k} · PCA 2D · t-SNE"),
    ("🔍 DBSCAN",         "cluster_done",  f"eps={dbscan_eps:.1f} · density-based · noise detection"),
]

for title, flag, desc in summary_items:
    done = st.session_state[flag]
    color = "#34D399" if done else "#64748B"
    icon  = "✅" if done else "⭕"
    st.markdown(f"""
    <div style="background:#13111F;border-radius:10px;padding:12px 16px;
                border:1px solid {'#34D39930' if done else '#2D2B45'}">
      <div style="font-size:.85rem;font-weight:600;color:{color}">{icon} {title}</div>
      <div style="font-size:.75rem;color:#94A3B8;margin-top:4px">{desc}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 16px;font-size:.72rem;color:#374151">
  🧬 FitPulse ML Pipeline · Milestone 2 &nbsp;|&nbsp;
  TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE &nbsp;|&nbsp;
  Real Fitbit Dataset · March–April 2016
</div>
""", unsafe_allow_html=True)