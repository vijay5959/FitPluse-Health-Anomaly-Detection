"""
╔══════════════════════════════════════════════════════════════╗
║          🧬 FitPulse — Unified Health Analytics App          ║
║   Data Cleaning  ·  ML Pipeline  ·  TSFresh · Prophet        ║
║       KMeans · DBSCAN · PCA · t-SNE · Anomaly Detection      ║
╚══════════════════════════════════════════════════════════════╝

Run:
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn tsfresh prophet plotly
    streamlit run fitpulse_merged.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1a1a2e);
    background-size: 400% 400%;
    animation: gradientMove 14s ease infinite;
}

@keyframes gradientMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Hero ── */
.hero-container {
    text-align: center;
    padding: 36px 40px;
    border-radius: 22px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 28px;
}
.hero-title {
    font-size: 46px;
    font-weight: 700;
    color: white;
    letter-spacing: -1px;
}
.hero-subtitle {
    font-size: 17px;
    color: #cbd5e1;
    margin-top: 6px;
}

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.07);
    padding: 20px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.12);
    text-align: center;
}

/* ── Cards ── */
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

/* ── Metric tiles ── */
.fp-metric-row { display: flex; gap: 14px; margin-bottom: 18px; flex-wrap: wrap; }
.fp-metric {
    flex: 1; min-width: 120px;
    background: #13111F;
    border: 1px solid #2D2B45;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
}
.fp-metric .val { font-size: 2rem; font-weight: 700; color: #A78BFA; line-height: 1.1; }
.fp-metric .lbl { font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: .07em; margin-top: 4px; }

/* ── Step badges ── */
.fp-step { display: inline-flex; align-items: center; gap: 8px; background: #2D2B45; border-radius: 8px; padding: 6px 12px; font-size: 0.78rem; color: #A78BFA; font-weight: 600; margin-bottom: 8px; }
.fp-step-ok   { background: #14422D; color: #34D399; }
.fp-step-warn { background: #3D2B00; color: #FBBF24; }

/* ── Section titles ── */
.fp-section-title { font-size: 1.35rem; font-weight: 700; color: #E2E8F0; margin: 0 0 4px 0; display: flex; align-items: center; gap: 10px; }
.fp-section-sub   { font-size: 0.82rem; color: #94A3B8; margin-bottom: 20px; }

/* ── File rows ── */
.fp-file-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; border-radius: 10px;
    border: 1px solid #2D2B45; margin-bottom: 8px;
    background: #13111F;
}
.fp-file-ok   { border-color: #34D39940; }
.fp-file-miss { border-color: #F4727240; }

/* ── Cluster cards ── */
.fp-cluster   { border-radius: 12px; padding: 16px 18px; border: 1px solid #2D2B45; }
.fp-cluster-0 { background: linear-gradient(135deg,#1a2e20,#13111F); border-color:#34D39940; }
.fp-cluster-1 { background: linear-gradient(135deg,#1a1a2e,#13111F); border-color:#A78BFA40; }
.fp-cluster-2 { background: linear-gradient(135deg,#2e1a1a,#13111F); border-color:#F4728440; }

/* ── Progress bar ── */
.fp-progress { background:#2D2B45; border-radius:99px; height:6px; margin:16px 0; }
.fp-progress-fill { height:6px; border-radius:99px; background: linear-gradient(90deg, #00c6ff, #A78BFA, #34D399); transition: width .5s ease; }

/* ── Info / warn / success boxes ── */
.fp-info        { background:#1A2035; border-left:3px solid #60A5FA; border-radius:0 8px 8px 0; padding:10px 14px; font-size:.82rem; color:#CBD5E1; margin-bottom:14px; }
.fp-warn-box    { background:#2A1F00; border-left:3px solid #FBBF24; border-radius:0 8px 8px 0; padding:10px 14px; font-size:.82rem; color:#FDE68A; margin-bottom:14px; }
.fp-success-box { background:#0D2B1A; border-left:3px solid #34D399;  border-radius:0 8px 8px 0; padding:10px 14px; font-size:.82rem; color:#A7F3D0; margin-bottom:14px; }

/* ── Log block ── */
.fp-log {
    background:#0A0913; border:1px solid #2D2B45; border-radius:10px;
    padding:14px 18px; font-family:'JetBrains Mono','Courier New',monospace;
    font-size:.78rem; color:#A78BFA; line-height:1.7;
}

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white; font-weight: 600; border-radius: 12px;
    height: 3em; width: 100%; border: none; transition: 0.3s ease;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #A78BFA, #7C3AED);
    transform: scale(1.03);
    box-shadow: 0 4px 15px #A78BFA44;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A1730 0%, #0F0D1A 100%);
    border-right: 1px solid #2D2B45;
}
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }

/* ── File uploader ── */
section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    padding: 20px; border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* ── Dropdown section header ── */
.dropdown-header {
    background: linear-gradient(135deg, #1E1B2E, #2A1A3D);
    border: 1px solid #A78BFA55;
    border-radius: 16px;
    padding: 18px 24px;
    margin-bottom: 12px;
    display: flex; align-items: center; gap: 14px;
}
.dropdown-header-icon { font-size: 2rem; }
.dropdown-header-text {}
.dropdown-header-title { font-size: 1.2rem; font-weight: 700; color: #E2E8F0; }
.dropdown-header-desc  { font-size: .78rem; color: #94A3B8; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════
_ss_keys = [
    "original_df", "df", "preprocessing",          # cleaning
    "data_loaded", "tsfresh_done", "prophet_done", "cluster_done",
    "daily", "hourly_s", "hourly_i", "sleep", "hr",
    "master", "hr_minute", "features", "features_norm",
    "prophet_hr_fcst", "prophet_hr_df",
    "prophet_steps_fcst", "prophet_steps_df",
    "prophet_sleep_fcst", "prophet_sleep_df",
    "cluster_df", "kmeans_labels", "dbscan_labels",
    "pca_df", "tsne_df", "profile",
    "inertias", "k_range", "pca_var", "n_clusters", "n_noise",
    "prophet_method",
]
for _k in _ss_keys:
    if _k not in st.session_state:
        st.session_state[_k] = None

for _flag in ["data_loaded", "tsfresh_done", "prophet_done", "cluster_done"]:
    if st.session_state[_flag] is None:
        st.session_state[_flag] = False


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 14px">
        <div style="font-size:2.6rem">💪</div>
        <div style="font-size:1.25rem;font-weight:700;color:#A78BFA">FitPulse</div>
        <div style="font-size:0.7rem;color:#64748B;margin-top:2px">AI HEALTH ANALYTICS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:.72rem;color:#64748B;letter-spacing:.1em'>ML CONTROLS</div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    kmeans_k    = st.slider("KMeans Clusters (K)",  min_value=2, max_value=8, value=3)
    dbscan_eps  = st.slider("DBSCAN EPS",            min_value=1.0, max_value=5.0, value=2.2, step=0.1)
    dbscan_min  = st.slider("DBSCAN min_samples",    min_value=1, max_value=10, value=2)

    st.markdown("---")
    st.markdown("<div style='font-size:.72rem;color:#64748B;letter-spacing:.1em'>PIPELINE STATUS</div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    def _dot(ok): return "🟢" if ok else "⚪"
    _clean_ok = st.session_state.df is not None and "preprocessing" in st.session_state and st.session_state["preprocessing"] is not None
    st.markdown(f"""
    <div style='font-size:.82rem;line-height:2.4;color:#CBD5E1'>
        {_dot(st.session_state.original_df is not None)} 🧹 &nbsp;Data Cleaning<br>
        {_dot(st.session_state.data_loaded)}  🔧 &nbsp;ML Load & Parse<br>
        {_dot(st.session_state.tsfresh_done)} 🧪 &nbsp;TSFresh Features<br>
        {_dot(st.session_state.prophet_done)} 📈 &nbsp;Prophet Forecast<br>
        {_dot(st.session_state.cluster_done)} 🤖 &nbsp;Clustering
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.7rem;color:#475569;text-align:center'>
        Real Fitbit Dataset<br>30 users · Mar–Apr 2016<br>Minute-level HR data
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  HERO
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-container">
    <div class="hero-title">💪 FitPulse</div>
    <div class="hero-subtitle">
        AI-Powered Health Anomaly Detection &amp; ML Pipeline from Fitness Devices
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<h4 style='text-align:center;color:#cbd5e1;font-weight:400;margin-bottom:28px'>"
    "Clean your fitness data · Forecast trends · Discover user clusters</h4>",
    unsafe_allow_html=True
)

st.divider()


# ════════════════════════════════════════════════════════════
# ██  DROPDOWN 1 · DATA CLEANING
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="dropdown-header">
    <div class="dropdown-header-icon">🧹</div>
    <div class="dropdown-header-text">
        <div class="dropdown-header-title">Data Cleaning</div>
        <div class="dropdown-header-desc">Upload a CSV · Auto-clean nulls · Preview data · Check null values before &amp; after</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("▼  Open Data Cleaning Panel", expanded=False):

    # ── File Upload ──────────────────────────────────────
    st.markdown('<div class="fp-section-title" style="font-size:1.05rem">📂 Upload Fitness CSV Data</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a fitness device CSV", type=["csv"], key="cleaner_upload")

    if uploaded_file is not None:
        # Store only on first upload (reset if new file)
        if (st.session_state.original_df is None or
                st.session_state.get("_cleaner_filename") != uploaded_file.name):
            st.session_state.original_df = pd.read_csv(uploaded_file)
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state["_cleaner_filename"] = uploaded_file.name
            # reset preprocessing log on new upload
            st.session_state["preprocessing"] = None

        df_clean = st.session_state.df
        orig_df  = st.session_state.original_df

        st.success("✅ File uploaded successfully!")

        # ── Summary Metrics ──────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Total Rows",      df_clean.shape[0])
        c2.metric("📂 Total Columns",   df_clean.shape[1])
        c3.metric("⚠️ Total Null Values", int(df_clean.isnull().sum().sum()))

        st.divider()

        # ── Action Buttons ───────────────────────────────
        b1, b2, b3 = st.columns(3)

        # CLEAN DATA
        if b1.button("🧹 Clean Data", key="btn_clean"):
            with st.spinner("Cleaning data… ⏳"):
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

                st.session_state.df             = df_work
                st.session_state["preprocessing"] = steps

            st.success("✅ Data cleaned successfully!")

        # SHOW DATA
        if b2.button("👀 Show Data", key="btn_show"):
            st.subheader("📋 Data Preview")
            st.dataframe(st.session_state.df, use_container_width=True)

        # CHECK NULL VALUES
        if b3.button("🔍 Check Null Values", key="btn_nulls"):
            cleaned = st.session_state.df

            st.subheader("📊 Null Values — Before Cleaning")
            null_before   = orig_df.isnull().sum()
            pct_before    = (null_before / len(orig_df) * 100).round(2)
            st.dataframe(pd.DataFrame({"Null Count": null_before,
                                       "Null %": pct_before}))
            st.bar_chart(null_before)

            st.divider()

            st.subheader("📊 Null Values — After Cleaning")
            null_after = cleaned.isnull().sum()
            pct_after  = (null_after / len(cleaned) * 100).round(2)
            st.dataframe(pd.DataFrame({"Null Count": null_after,
                                       "Null %": pct_after}))
            st.bar_chart(null_after)

            if null_after.sum() == 0:
                st.success("✅ All null values removed!")
            else:
                st.warning("⚠️ Some null values still remain.")

        # ── Preprocessing Log ────────────────────────────
        if st.session_state.get("preprocessing"):
            st.divider()
            st.subheader("⚙️ Pre-Processing Steps Applied")
            log_html = "<div class='fp-log'>" + "<br>".join(st.session_state["preprocessing"]) + "</div>"
            st.markdown(log_html, unsafe_allow_html=True)

    else:
        st.markdown('<div class="fp-info">ℹ️ Upload any fitness CSV file above to start cleaning.</div>',
                    unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)
st.divider()


# ════════════════════════════════════════════════════════════
# ██  DROPDOWN 2 · ML ANOMALY DETECTION PIPELINE
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="dropdown-header">
    <div class="dropdown-header-icon">🧬</div>
    <div class="dropdown-header-text">
        <div class="dropdown-header-title">ML Anomaly Detection Pipeline</div>
        <div class="dropdown-header-desc">TSFresh Features · Prophet Forecasting · KMeans &amp; DBSCAN Clustering · PCA · t-SNE</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("▼  Open ML Pipeline Panel", expanded=False):

    # ── Progress bar ────────────────────────────────────
    def _render_progress():
        _done = sum([
            bool(st.session_state.data_loaded),
            bool(st.session_state.tsfresh_done),
            bool(st.session_state.prophet_done),
            bool(st.session_state.cluster_done),
        ])
        _pct = int(_done / 4 * 100)
        _pct_color = "#34D399" if _pct == 100 else "#A78BFA"
        _complete = (
            '<div style="margin-top:10px;font-size:.82rem;color:#34D399;font-weight:700">'
            '🎉 Pipeline 100% complete!</div>' if _pct == 100 else ""
        )
        _step_flags = [
            ("📂 Data Load",  bool(st.session_state.data_loaded)),
            ("🧪 TSFresh",    bool(st.session_state.tsfresh_done)),
            ("📈 Prophet",    bool(st.session_state.prophet_done)),
            ("🤖 Clustering", bool(st.session_state.cluster_done)),
        ]
        _badges = "".join(
            f'<span class="fp-step fp-step-ok">✓ {s}</span>' if ok
            else f'<span class="fp-step">{s}</span>'
            for s, ok in _step_flags
        )
        st.markdown(f"""
        <div class="fp-card">
          <div style="display:flex;justify-content:space-between;font-size:.78rem;color:#94A3B8;margin-bottom:6px">
            <span>PIPELINE PROGRESS</span>
            <span style="color:{_pct_color};font-weight:700">{_pct}%</span>
          </div>
          <div class="fp-progress"><div class="fp-progress-fill" style="width:{_pct}%"></div></div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">{_badges}</div>
          {_complete}
        </div>""", unsafe_allow_html=True)

    _render_progress()

    # ════════════════════════════════════════════════════
    # SECTION 2A · FILE UPLOAD (5 CSVs)
    # ════════════════════════════════════════════════════
    st.markdown("""
    <div class="fp-section-title" style="font-size:1.05rem">📂 Upload Fitbit Dataset Files</div>
    <div class="fp-section-sub">Upload all 5 CSV files — all at once or individually</div>
    """, unsafe_allow_html=True)

    REQUIRED = {
        "dailyActivity":     {"emoji": "🏃", "label": "Daily Activity",    "hint": "dailyActivity_merged.csv"},
        "hourlySteps":       {"emoji": "👣", "label": "Hourly Steps",       "hint": "hourlySteps_merged.csv"},
        "hourlyIntensities": {"emoji": "⚡", "label": "Hourly Intensities", "hint": "hourlyIntensities_merged.csv"},
        "minuteSleep":       {"emoji": "💤", "label": "Minute Sleep",        "hint": "minuteSleep_merged.csv"},
        "heartrate":         {"emoji": "❤️", "label": "Heart Rate",          "hint": "heartrate_seconds_merged.csv"},
    }

    file_map = {}

    st.markdown('<div class="fp-info">ℹ️ <b>All at once:</b> Ctrl+click / ⌘+click to select multiple files.</div>',
                unsafe_allow_html=True)

    bulk_files = st.file_uploader(
        "📦 Upload all files at once", type="csv",
        accept_multiple_files=True, key="ml_bulk",
    )
    if bulk_files:
        for f in bulk_files:
            n = f.name.lower()
            if   "dailyactivity"     in n: file_map["dailyActivity"]     = f
            elif "hourlysteps"       in n: file_map["hourlySteps"]       = f
            elif "hourlyintensities" in n: file_map["hourlyIntensities"] = f
            elif "minutesleep"       in n: file_map["minuteSleep"]       = f
            elif "heartrate"         in n: file_map["heartrate"]         = f

    st.markdown('<div style="text-align:center;color:#64748B;font-size:.78rem;margin:8px 0">— or upload individually —</div>',
                unsafe_allow_html=True)

    solo_cols = st.columns(5)
    for i, (key, meta) in enumerate(REQUIRED.items()):
        with solo_cols[i]:
            st.markdown(f'<div style="font-size:.76rem;font-weight:600;color:#CBD5E1">{meta["emoji"]} {meta["label"]}</div>'
                        f'<div style="font-size:.64rem;color:#64748B;margin-bottom:4px">{meta["hint"]}</div>',
                        unsafe_allow_html=True)
            f = st.file_uploader(meta["label"], type="csv",
                                 key=f"ml_solo_{key}", label_visibility="collapsed")
            if f is not None:
                file_map[key] = f

    n_found = len(file_map)

    # Status checklist
    st.markdown("<br>", unsafe_allow_html=True)
    sc = st.columns(5)
    for i, (key, meta) in enumerate(REQUIRED.items()):
        ok  = key in file_map
        css = "fp-file-ok" if ok else "fp-file-miss"
        with sc[i]:
            st.markdown(f'<div class="fp-file-row {css}">'
                        f'<span style="font-size:.76rem;color:#CBD5E1">{meta["emoji"]} {meta["label"]}</span>'
                        f'<span>{"✅" if ok else "⬜"}</span></div>', unsafe_allow_html=True)

    mc = st.columns(3)
    with mc[0]:
        st.markdown(f'<div class="fp-metric"><div class="val">{n_found}</div><div class="lbl">Detected</div></div>',
                    unsafe_allow_html=True)
    with mc[1]:
        st.markdown(f'<div class="fp-metric"><div class="val" style="color:#F87171">{5-n_found}</div><div class="lbl">Missing</div></div>',
                    unsafe_allow_html=True)
    with mc[2]:
        rc = "#34D399" if n_found == 5 else "#FBBF24"
        rt = "✓ Ready" if n_found == 5 else "⚠ Incomplete"
        st.markdown(f'<div class="fp-metric"><div class="val" style="color:{rc};font-size:1.1rem">{rt}</div><div class="lbl">Status</div></div>',
                    unsafe_allow_html=True)

    if n_found == 5:
        st.markdown('<div class="fp-success-box">✅ All 5 files detected — ready to run pipeline!</div>',
                    unsafe_allow_html=True)
    elif n_found > 0:
        missing = [m["label"] for k, m in REQUIRED.items() if k not in file_map]
        st.markdown(f'<div class="fp-warn-box">⚠️ Still missing: {", ".join(missing)}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="fp-info">ℹ️ Upload the 5 Fitbit CSV files above.</div>',
                    unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # SECTION 2B · LOAD & PARSE
    # ════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="fp-section-title" style="font-size:1.05rem">🔧 Load &amp; Parse the Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-section-sub">Time normalization · Master DataFrame assembly · Null checks</div>', unsafe_allow_html=True)

    if st.button("🔧  Load & Parse the Data", disabled=(n_found < 5), key="btn_load_parse"):
        with st.spinner("Loading and parsing all datasets…"):
            try:
                daily    = pd.read_csv(file_map["dailyActivity"])
                hourly_s = pd.read_csv(file_map["hourlySteps"])
                hourly_i = pd.read_csv(file_map["hourlyIntensities"])
                sleep    = pd.read_csv(file_map["minuteSleep"])
                hr       = pd.read_csv(file_map["heartrate"])

                # Strip whitespace from all column names across all dataframes
                for _df in [daily, hourly_s, hourly_i, sleep, hr]:
                    _df.columns = _df.columns.str.strip()

                daily["ActivityDate"]    = pd.to_datetime(daily["ActivityDate"])
                hourly_s["ActivityHour"] = pd.to_datetime(hourly_s["ActivityHour"])
                hourly_i["ActivityHour"] = pd.to_datetime(hourly_i["ActivityHour"])

                # Sleep time column — detect robustly
                sleep_time_col = next(
                    (c for c in sleep.columns if "date" in c.lower() or "time" in c.lower()),
                    sleep.columns[1]
                )
                sleep[sleep_time_col] = pd.to_datetime(sleep[sleep_time_col])

                # Sleep value column — detect robustly
                sleep_val_col = next(
                    (c for c in sleep.columns if "value" in c.lower() or "stage" in c.lower()),
                    None
                )
                if sleep_val_col is None:
                    numeric_sleep = sleep.select_dtypes(include="number").columns.tolist()
                    sleep_val_col = next((c for c in numeric_sleep if c != "Id"), numeric_sleep[0])

                # ── Robust HR column detection ─────────────────
                # Strip whitespace from all column names first
                hr.columns = hr.columns.str.strip()

                # Find the timestamp column (contains 'time' or 'date' case-insensitive)
                hr_time_col = next(
                    (c for c in hr.columns if "time" in c.lower() or "date" in c.lower()),
                    hr.columns[1]
                )
                # Find the value column (contains 'value' case-insensitive, or is numeric non-Id)
                hr_val_col = next(
                    (c for c in hr.columns if "value" in c.lower()),
                    None
                )
                if hr_val_col is None:
                    # fallback: pick the first numeric column that isn't Id
                    numeric_hr_cols = hr.select_dtypes(include="number").columns.tolist()
                    hr_val_col = next((c for c in numeric_hr_cols if c != "Id"), numeric_hr_cols[0])

                hr[hr_time_col] = pd.to_datetime(hr[hr_time_col])
                hr = hr.rename(columns={hr_time_col: "Time", hr_val_col: "Value"})

                hr_minute = (
                    hr.groupby(["Id", pd.Grouper(key="Time", freq="1min")])["Value"]
                    .mean()
                    .reset_index()
                )
                hr_minute.columns = ["Id", "Time", "HeartRate"]
                hr_minute = hr_minute.dropna()
                hr_minute["Date"] = hr_minute["Time"].dt.date

                avg_hr = hr_minute.groupby(["Id","Date"])["HeartRate"].mean().reset_index()
                avg_hr.columns = ["Id","Date","AvgHR"]

                sleep["Date"] = sleep[sleep_time_col].dt.date
                sleep_agg = sleep.groupby(["Id","Date"])[sleep_val_col].count().reset_index()
                sleep_agg.columns = ["Id","Date","TotalSleepMinutes"]

                daily["Date"] = daily["ActivityDate"].dt.date
                master = daily[["Id","Date","TotalSteps","Calories",
                                "VeryActiveMinutes","SedentaryMinutes"]].copy()
                master = master.merge(avg_hr,   on=["Id","Date"], how="left")
                master = master.merge(sleep_agg, on=["Id","Date"], how="left")
                master["Date"] = pd.to_datetime(master["Date"])

                for k, v in [("daily", daily), ("hourly_s", hourly_s),
                              ("hourly_i", hourly_i), ("sleep", sleep),
                              ("hr", hr), ("master", master), ("hr_minute", hr_minute)]:
                    st.session_state[k] = v

                st.session_state.data_loaded = True
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during loading: {e}")

    if st.session_state.data_loaded:
        daily     = st.session_state.daily
        hourly_s  = st.session_state.hourly_s
        hourly_i  = st.session_state.hourly_i
        sleep     = st.session_state.sleep
        hr        = st.session_state.hr
        master    = st.session_state.master
        hr_minute = st.session_state.hr_minute

        st.markdown('<div class="fp-success-box">✅ All 5 files loaded · master DataFrame built</div>',
                    unsafe_allow_html=True)

        with st.expander("◆ Null Value Check"):
            frames = {"dailyActivity": daily, "hourlySteps": hourly_s,
                      "hourlyIntensities": hourly_i, "minuteSleep": sleep, "heartrate": hr}
            null_data = [{"Dataset": nm, "Nulls": df.isnull().sum().sum(), "Rows": len(df)}
                         for nm, df in frames.items()]
            null_df = pd.DataFrame(null_data)
            ncols = st.columns(5)
            for i, row in null_df.iterrows():
                with ncols[i]:
                    color = "#34D399" if row["Nulls"] == 0 else "#F87171"
                    st.markdown(f'<div class="fp-metric"><div class="val" style="color:{color};font-size:1.4rem">{int(row["Nulls"])}</div>'
                                f'<div class="lbl">nulls</div>'
                                f'<div style="font-size:.68rem;color:#64748B;margin-top:4px">{row["Dataset"]}</div>'
                                f'<div style="font-size:.72rem;color:#94A3B8">{row["Rows"]:,} rows</div></div>',
                                unsafe_allow_html=True)

        with st.expander("◆ Dataset Overview"):
            m_cols = st.columns(5)
            for col, val, lbl in zip(m_cols,
                [daily["Id"].nunique(), hr["Id"].nunique(), sleep["Id"].nunique(),
                 len(hr_minute), len(master)],
                ["Daily Users","HR Users","Sleep Users","HR Min Rows","Master Rows"]):
                with col:
                    st.markdown(f'<div class="fp-metric"><div class="val">{val:,}</div>'
                                f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
            key_cols = [c for c in ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes"] if c in master.columns]
            st.dataframe(master[key_cols].describe().round(2), use_container_width=True)

        with st.expander("◆ Cleaned Dataset Preview"):
            show = [c for c in ["Id","Date","TotalSteps","Calories","AvgHR",
                                 "TotalSleepMinutes","VeryActiveMinutes","SedentaryMinutes"]
                    if c in master.columns]
            st.dataframe(master[show].head(20), use_container_width=True)

    # ════════════════════════════════════════════════════
    # SECTION 2C · TSFresh
    # ════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="fp-section-title" style="font-size:1.05rem">🧪 TSFresh Feature Extraction</div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-section-sub">Statistical features from minute-level heart rate time series · MinimalFCParameters</div>', unsafe_allow_html=True)

    if st.button("🧪  Run TSFresh Feature Extraction",
                 disabled=not st.session_state.data_loaded, key="btn_tsfresh"):
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
                    ts_df, column_id="id", column_sort="time",
                    column_value="value",
                    default_fc_parameters=MinimalFCParameters(),
                    disable_progressbar=True
                )
                features = features.dropna(axis=1, how="all").dropna(axis=0)

                scaler = MinMaxScaler()
                features_norm = pd.DataFrame(
                    scaler.fit_transform(features),
                    index=features.index, columns=features.columns
                )

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

        st.markdown(f'<div class="fp-success-box">✅ TSFresh — {len(features)} users × {features.shape[1]} features</div>',
                    unsafe_allow_html=True)

        with st.expander("◆ Feature Matrix Heatmap", expanded=True):
            fig, ax = plt.subplots(figsize=(14, max(4, len(features)*0.9)))
            fig.patch.set_facecolor("#0F0D1A"); ax.set_facecolor("#0F0D1A")
            sc = features_norm.copy()
            sc.columns = [c.split("__")[-1][:20] for c in sc.columns]
            sc.index   = [str(i)[-6:] for i in sc.index]
            sns.heatmap(sc, ax=ax, cmap="magma", annot=True, fmt=".2f",
                        linewidths=0.4, linecolor="#2D2B45", cbar_kws={"shrink": 0.7})
            ax.set_title("TSFresh Feature Matrix (Normalized 0–1)", color="#E2E8F0", fontsize=11)
            ax.tick_params(colors="#94A3B8", labelsize=7)
            plt.xticks(rotation=45, ha="right", fontsize=7)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with st.expander("◆ Raw Feature Matrix"):
            st.dataframe(features.round(4), use_container_width=True)

    # ════════════════════════════════════════════════════
    # SECTION 2D · PROPHET
    # ════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="fp-section-title" style="font-size:1.05rem">📈 Prophet Trend Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-section-sub">Additive models · Weekly seasonality · 80% CI · 30-day forecasts for HR, Steps &amp; Sleep</div>', unsafe_allow_html=True)

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
                sigma = df_input["y"].std() * 0.5
                fcst["yhat_upper"] = fcst["yhat"] + 1.28 * sigma
                fcst["yhat_lower"] = fcst["yhat"] - 1.28 * sigma
            return fcst, "prophet"
        except Exception:
            pass

        df = df_input.copy().reset_index(drop=True)
        df["t"] = (df["ds"] - df["ds"].min()).dt.days.astype(float)
        sin_w  = np.sin(2 * np.pi * df["t"] / 7)
        cos_w  = np.cos(2 * np.pi * df["t"] / 7)
        X_seas = np.column_stack([np.ones(len(df)), df["t"], sin_w, cos_w])
        try:
            sc = np.linalg.lstsq(X_seas, df["y"].values, rcond=None)[0]
        except Exception:
            sc = np.array([df["y"].mean(), 0., 0., 0.])
        sigma = df["y"].std()
        last_date   = df["ds"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        all_dates = pd.concat([df["ds"], pd.Series(future_dates)]).reset_index(drop=True)
        t_all  = (all_dates - df["ds"].min()).dt.days.astype(float)
        sin_all = np.sin(2 * np.pi * t_all / 7)
        cos_all = np.cos(2 * np.pi * t_all / 7)
        yhat = np.column_stack([np.ones(len(t_all)), t_all, sin_all, cos_all]) @ sc
        fcst = pd.DataFrame({"ds": all_dates, "yhat": yhat,
                             "yhat_upper": yhat + 1.28*sigma,
                             "yhat_lower": yhat - 1.28*sigma})
        return fcst, "fallback"

    if st.button("📈  Run Prophet Trend Forecasting",
                 disabled=not st.session_state.data_loaded, key="btn_prophet"):
        master    = st.session_state.master
        hr_minute = st.session_state.hr_minute
        with st.spinner("Fitting forecast models…"):
            try:
                # HR
                hr_df = (hr_minute.groupby("Date")["HeartRate"].mean().reset_index())
                hr_df.columns = ["ds","y"]
                hr_df["ds"] = pd.to_datetime(hr_df["ds"])
                hr_df = hr_df.dropna().sort_values("ds")
                fcst_hr, meth = _fit_prophet_safe(hr_df)

                # Steps
                steps_df = master.groupby("Date")["TotalSteps"].mean().reset_index()
                steps_df.columns = ["ds","y"]
                steps_df["ds"] = pd.to_datetime(steps_df["ds"])
                steps_df = steps_df.dropna().sort_values("ds")
                fcst_steps, _ = _fit_prophet_safe(steps_df)

                # Sleep
                fcst_sleep = None; sleep_df_p = None
                if "TotalSleepMinutes" in master.columns:
                    sleep_df_p = master.groupby("Date")["TotalSleepMinutes"].mean().reset_index()
                    sleep_df_p.columns = ["ds","y"]
                    sleep_df_p["ds"] = pd.to_datetime(sleep_df_p["ds"])
                    sleep_df_p = sleep_df_p.dropna().sort_values("ds")
                    if len(sleep_df_p) >= 3:
                        fcst_sleep, _ = _fit_prophet_safe(sleep_df_p)

                for k, v in [("prophet_hr_fcst", fcst_hr), ("prophet_hr_df", hr_df),
                              ("prophet_steps_fcst", fcst_steps), ("prophet_steps_df", steps_df),
                              ("prophet_sleep_fcst", fcst_sleep), ("prophet_sleep_df", sleep_df_p),
                              ("prophet_method", meth)]:
                    st.session_state[k] = v
                st.session_state.prophet_done = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ Forecasting error: {e}")

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

    if st.session_state.prophet_done:
        fcst_hr    = st.session_state.prophet_hr_fcst
        hr_df_p    = st.session_state.prophet_hr_df
        fcst_steps = st.session_state.prophet_steps_fcst
        steps_df_p = st.session_state.prophet_steps_df
        fcst_sleep = st.session_state.prophet_sleep_fcst
        sleep_df_p = st.session_state.prophet_sleep_df
        _engine    = "Prophet (Stan)" if st.session_state.prophet_method == "prophet" else "Linear Trend + Weekly Seasonality (fallback)"
        st.markdown(f'<div class="fp-success-box">✅ 3 forecast models ready · engine: {_engine}</div>',
                    unsafe_allow_html=True)

        with st.expander("◆ Heart Rate Forecast (30-day)", expanded=True):
            fig = _prophet_fig(hr_df_p, fcst_hr,
                               "Prophet — Heart Rate (30-day ahead · 80% CI)",
                               "#F472B6", "Heart Rate (bpm)")
            st.pyplot(fig, use_container_width=True); plt.close()

        with st.expander("◆ Steps & Sleep Forecast (30-day)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                fig2 = _prophet_fig(steps_df_p, fcst_steps,
                                    "Prophet — Daily Steps", "#34D399", "Steps")
                st.pyplot(fig2, use_container_width=True); plt.close()
            with c2:
                if fcst_sleep is not None:
                    fig3 = _prophet_fig(sleep_df_p, fcst_sleep,
                                        "Prophet — Sleep Minutes", "#A78BFA", "Sleep (min)")
                    st.pyplot(fig3, use_container_width=True); plt.close()

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

    # ════════════════════════════════════════════════════
    # SECTION 2E · CLUSTERING
    # ════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="fp-section-title" style="font-size:1.05rem">🤖 Clustering — KMeans + DBSCAN + PCA + t-SNE</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="fp-section-sub">Activity-based user segmentation · KMeans K={kmeans_k} · DBSCAN eps={dbscan_eps:.1f}</div>',
                unsafe_allow_html=True)

    if st.button("🤖  Run Clustering (KMeans + DBSCAN + PCA + t-SNE)",
                 disabled=not st.session_state.data_loaded, key="btn_cluster"):
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
                    _kw  = {"max_iter": 1000} if _ver >= (1, 5) else {"n_iter": 1000}
                except Exception:
                    _kw = {"max_iter": 1000}
                X_tsne = TSNE(n_components=2, random_state=42, perplexity=perp, **_kw).fit_transform(X)
                tsne_df = pd.DataFrame(X_tsne, columns=["tSNE1","tSNE2"])
                tsne_df["KMeans"] = km_labels; tsne_df["DBSCAN"] = db_labels

                profile    = cluster_features.groupby("KMeans_Cluster")[feat_cols].mean().round(1)
                n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
                n_noise    = list(db_labels).count(-1)

                for k, v in [("cluster_df", cluster_features), ("kmeans_labels", km_labels),
                              ("dbscan_labels", db_labels), ("pca_df", pca_df),
                              ("tsne_df", tsne_df), ("profile", profile),
                              ("inertias", inertias), ("k_range", k_range),
                              ("pca_var", pca.explained_variance_ratio_),
                              ("n_clusters", n_clusters), ("n_noise", n_noise)]:
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

        st.markdown(f'<div class="fp-success-box">✅ Clustering done · {len(cluster_df)} users · KMeans K={kmeans_k} · DBSCAN {n_clusters} clusters · {n_noise} noise</div>',
                    unsafe_allow_html=True)

        mc = st.columns(5)
        for col, val, lbl in zip(mc,
            [len(cluster_df), kmeans_k, f"{pca_var[0]*100:.1f}%", f"{pca_var[1]*100:.1f}%", n_clusters],
            ["Users","KMeans K","PC1 Var","PC2 Var","DBSCAN Clusters"]):
            with col:
                st.markdown(f'<div class="fp-metric"><div class="val" style="font-size:1.3rem">{val}</div>'
                            f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

        PALETTE_KM = ["#A78BFA","#34D399","#F472B6","#FBBF24","#60A5FA","#FB923C","#4ADE80","#E879F9"]
        PALETTE_DB = ["#94A3B8","#A78BFA","#34D399","#F472B6","#FBBF24"]

        def _scatter(ax, xs, ys, labels, palette, title, xlabel, ylabel, note=""):
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
            if note:
                ax.annotate(note, xy=(0.01,0.01), xycoords="axes fraction",
                            color="#64748B", fontsize=7)

        with st.expander("◆ KMeans Elbow Curve", expanded=True):
            fig_e, ax_e = plt.subplots(figsize=(8, 4))
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
            fig_pca, axes = plt.subplots(1, 2, figsize=(13, 5))
            fig_pca.patch.set_facecolor("#0F0D1A")
            for ax in axes: ax.set_facecolor("#13111F")
            _scatter(axes[0], pca_df["PC1"].values, pca_df["PC2"].values, list(km_labels), PALETTE_KM,
                     f"KMeans (K={kmeans_k}) — PCA", f"PC1 ({pca_var[0]*100:.1f}%)", f"PC2 ({pca_var[1]*100:.1f}%)")
            _scatter(axes[1], pca_df["PC1"].values, pca_df["PC2"].values, list(db_labels), PALETTE_DB,
                     f"DBSCAN (eps={dbscan_eps:.1f}) — PCA", f"PC1 ({pca_var[0]*100:.1f}%)", f"PC2 ({pca_var[1]*100:.1f}%)",
                     f"Noise: {n_noise}")
            plt.tight_layout(pad=2); st.pyplot(fig_pca, use_container_width=True); plt.close()

        with st.expander("◆ t-SNE Projection — KMeans & DBSCAN", expanded=True):
            fig_ts, axes2 = plt.subplots(1, 2, figsize=(13, 5))
            fig_ts.patch.set_facecolor("#0F0D1A")
            for ax in axes2: ax.set_facecolor("#13111F")
            _scatter(axes2[0], tsne_df["tSNE1"].values, tsne_df["tSNE2"].values, list(km_labels), PALETTE_KM,
                     f"t-SNE · KMeans (K={kmeans_k})", "tSNE-1", "tSNE-2")
            _scatter(axes2[1], tsne_df["tSNE1"].values, tsne_df["tSNE2"].values, list(db_labels), PALETTE_DB,
                     f"t-SNE · DBSCAN (eps={dbscan_eps:.1f})", "tSNE-1", "tSNE-2")
            plt.tight_layout(pad=2); st.pyplot(fig_ts, use_container_width=True); plt.close()

        with st.expander("◆ Cluster Profiles & Personas", expanded=True):
            feat_cols_plot = [c for c in ["TotalSteps","Calories","VeryActiveMinutes",
                                          "SedentaryMinutes","TotalSleepMinutes"]
                              if c in profile.columns]
            st.markdown("**Mean feature values per KMeans cluster**")
            st.dataframe(profile, use_container_width=True)

            # bar chart
            if feat_cols_plot:
                fig_bar, ax_bar = plt.subplots(figsize=(12, 4.5))
                fig_bar.patch.set_facecolor("#0F0D1A"); ax_bar.set_facecolor("#13111F")
                x = np.arange(kmeans_k); w = 0.15
                bar_colors = [PALETTE_KM[i % len(PALETTE_KM)] for i in range(len(feat_cols_plot))]
                for i, feat in enumerate(feat_cols_plot):
                    vals = [profile.loc[k, feat] if k in profile.index else 0 for k in range(kmeans_k)]
                    ax_bar.bar(x + i*w, vals, width=w, label=feat, color=bar_colors[i],
                               edgecolor="white", linewidth=0.4)
                ax_bar.set_xticks(x + w*(len(feat_cols_plot)-1)/2)
                ax_bar.set_xticklabels([f"Cluster {i}" for i in range(kmeans_k)], color="#E2E8F0")
                ax_bar.set_title("Cluster Profiles", color="#E2E8F0", fontsize=11)
                ax_bar.set_xlabel("Cluster", color="#94A3B8"); ax_bar.set_ylabel("Mean Value", color="#94A3B8")
                ax_bar.tick_params(colors="#94A3B8")
                for sp in ["top","right"]: ax_bar.spines[sp].set_visible(False)
                for sp in ["bottom","left"]: ax_bar.spines[sp].set_color("#2D2B45")
                ax_bar.legend(fontsize=7, facecolor="#1E1B2E", labelcolor="#CBD5E1", bbox_to_anchor=(1.01,1))
                plt.tight_layout(); st.pyplot(fig_bar, use_container_width=True); plt.close()

            # Persona cards
            st.markdown("<br>**Cluster Personas**", unsafe_allow_html=True)
            pcols = st.columns(kmeans_k)
            css_cycle   = ["fp-cluster-0","fp-cluster-1","fp-cluster-2"]
            for i in range(kmeans_k):
                with pcols[i % len(pcols)]:
                    if i in profile.index:
                        row   = profile.loc[i]
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
                        css_cls  = css_cycle[i % len(css_cycle)]
                        user_ids = ", ".join([str(x)[-4:] for x in users_in["Id"].tolist()[:4]])
                        if len(users_in) > 4: user_ids += f" +{len(users_in)-4}"
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


# ════════════════════════════════════════════════════════════
# ██  FOOTER / SUMMARY
# ════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.divider()

st.markdown("""
<div class="fp-card-accent">
  <div class="fp-section-title" style="margin-bottom:14px">✅ Pipeline Summary</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
""", unsafe_allow_html=True)

_clean_done = st.session_state.original_df is not None
summary = [
    ("🧹 Data Cleaning",       _clean_done,                    "CSV upload · null fill · datetime parse · preview"),
    ("🔧 Load & Parse",        bool(st.session_state.data_loaded),  "5 Fitbit CSVs · master DataFrame · time normalization"),
    ("🧪 TSFresh Features",    bool(st.session_state.tsfresh_done), "10 features · normalized heatmap"),
    ("📈 Prophet Forecast",    bool(st.session_state.prophet_done), "HR + Steps + Sleep · 30-day · 80% CI"),
    (f"🤖 KMeans (K={kmeans_k})", bool(st.session_state.cluster_done), "PCA · t-SNE · persona cards"),
    (f"🔍 DBSCAN (eps={dbscan_eps:.1f})", bool(st.session_state.cluster_done), "Density-based · noise detection"),
]
for title, done, desc in summary:
    color = "#34D399" if done else "#64748B"
    icon  = "✅" if done else "⭕"
    st.markdown(f"""
    <div style="background:#13111F;border-radius:10px;padding:12px 16px;
                border:1px solid {'#34D39930' if done else '#2D2B45'}">
      <div style="font-size:.85rem;font-weight:600;color:{color}">{icon} {title}</div>
      <div style="font-size:.75rem;color:#94A3B8;margin-top:4px">{desc}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:28px 0 12px;font-size:.72rem;color:#374151">
  💪 FitPulse &nbsp;|&nbsp; Data Cleaning · ML Anomaly Detection Pipeline &nbsp;|&nbsp;
  TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE &nbsp;|&nbsp;
  Real Fitbit Dataset · Mar–Apr 2016
</div>
""", unsafe_allow_html=True)