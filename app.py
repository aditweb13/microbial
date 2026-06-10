"""
app.py  –  FermentAI · AI Fermentation Yield Prediction & Optimization Dashboard
==================================================================================
Run with:
    streamlit run app.py

Requires:  streamlit, pandas, numpy, scikit-learn, xgboost, plotly,
           matplotlib, scipy
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FermentAI – Yield Prediction & Optimization",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  — "Bioluminescent Lab" Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════════
   FONTS
═══════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Mono:wght@300;400;500;600&family=Bebas+Neue&display=swap');

/* ═══════════════════════════════════════════════════════════════
   CSS VARIABLES
═══════════════════════════════════════════════════════════════ */
:root {
    --bg-base:         #070d19;
    --bg-surface:      #0c1426;
    --bg-card:         #0f1a30;
    --border:          #1b2a47;
    --accent-primary:  #00e5cc;
    --accent-secondary:#00b3a0;
    --accent-amber:    #f59e0b;
    --accent-crimson:  #ef4444;
    --text-primary:    #e2f3f0;
    --text-secondary:  #a3c9c4;
    --text-muted:      #7ba39e;
    --glow:            rgba(0,229,204,0.12);
    --glow-strong:     rgba(0,229,204,0.22);

    /* Streamlit Native Theme Overrides */
    --primary-color:              #00e5cc !important;
    --background-color:           #070d19 !important;
    --secondary-background-color: #0c1426 !important;
    --text-color:                 #e2f3f0 !important;
    --secondary-text-color:       #a3c9c4 !important;
}

/* ═══════════════════════════════════════════════════════════════
   BASE
═══════════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text-primary);
}

/* Ensure standard text elements are readable and don't blend with dark background */
h1, h2, h3, h4, h5, h6, p, li, span:not(.val-up):not(.val-down):not(.val-same), label {
    color: var(--text-primary);
}

a {
    color: var(--accent-primary) !important;
    text-decoration: none !important;
}
a:hover {
    color: var(--accent-secondary) !important;
    text-decoration: underline !important;
}

/* Hexagonal grid texture via SVG data URI */
.stApp {
    background-color: var(--bg-base);
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='100'%3E%3Cpath d='M28 66L0 50V16L28 0l28 16v34L28 66zm0-6l22-13V19L28 6 6 19v28l22 13z' fill='none' stroke='%2300e5cc' stroke-width='0.6' opacity='0.06'/%3E%3C/svg%3E");
    color: var(--text-primary);
}

/* ═══════════════════════════════════════════════════════════════
   HEADER BAR
═══════════════════════════════════════════════════════════════ */
[data-testid="stHeader"] {
    background: var(--bg-base) !important;
    border-bottom: 1px solid var(--border) !important;
    box-shadow: 0 1px 0 var(--border);
}

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 0.1rem; }
[data-testid="stSidebar"] .stRadio label {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    padding: 0.5rem 0.8rem !important;
    border-radius: 2px !important;
    transition: all 0.18s ease !important;
    display: block !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0,229,204,0.05) !important;
    color: var(--accent-secondary) !important;
}

/* ═══════════════════════════════════════════════════════════════
   ANIMATIONS
═══════════════════════════════════════════════════════════════ */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes gaugeAnim {
    from { stroke-dashoffset: 440; }
    to   { stroke-dashoffset: var(--target-offset); }
}
@keyframes cardGlow {
    0%   { box-shadow: -2px 0 12px var(--glow), 0 0 0 rgba(0,0,0,0); }
    50%  { box-shadow: -2px 0 28px var(--glow-strong), 0 4px 32px var(--glow); }
    100% { box-shadow: -2px 0 12px var(--glow), 0 2px 16px rgba(0,0,0,0.04); }
}

.fade-1 { animation: fadeSlideUp 0.4s ease-out 0.00s both; }
.fade-2 { animation: fadeSlideUp 0.4s ease-out 0.08s both; }
.fade-3 { animation: fadeSlideUp 0.4s ease-out 0.16s both; }
.fade-4 { animation: fadeSlideUp 0.4s ease-out 0.24s both; }
.fade-5 { animation: fadeSlideUp 0.4s ease-out 0.32s both; }

/* ═══════════════════════════════════════════════════════════════
   METRIC CARDS
═══════════════════════════════════════════════════════════════ */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent-primary);
    border-radius: 4px;
    padding: 1.3rem 1.5rem;
    box-shadow: -2px 0 12px var(--glow), 0 2px 16px rgba(0,0,0,0.04);
    transition: all 0.18s ease;
    margin-bottom: 1rem;
    animation: cardGlow 3s ease 0.5s 1;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-left-color: var(--accent-secondary);
    box-shadow: -2px 0 24px var(--glow-strong), 0 8px 28px rgba(0,0,0,0.08);
}
.metric-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.45rem;
}
.metric-card .value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.6rem;
    color: var(--text-primary);
    line-height: 1;
    letter-spacing: 0.02em;
}
.metric-card .sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
}

/* ═══════════════════════════════════════════════════════════════
   SECTION HEADERS
═══════════════════════════════════════════════════════════════ */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-style: italic;
    font-size: 2rem;
    color: var(--text-primary);
    margin-bottom: 0.2rem;
    animation: fadeSlideUp 0.4s ease-out both;
}
.section-underline {
    width: 40px;
    height: 1px;
    background: var(--accent-primary);
    margin: 0.3rem 0 0.6rem 0;
}
.section-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.06em;
    margin-bottom: 1.8rem;
    animation: fadeSlideUp 0.4s ease-out 0.08s both;
}

/* ═══════════════════════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════════════════════ */
.stButton > button {
    background: var(--bg-surface) !important;
    color: var(--accent-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.6rem !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 2px 12px rgba(0,229,204,0.08) !important;
}
.stButton > button:hover {
    background: var(--accent-primary) !important;
    color: var(--bg-base) !important;
    box-shadow: 0 4px 20px rgba(0,229,204,0.25) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

/* ═══════════════════════════════════════════════════════════════
   TABS — Underline style
═══════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 0.6rem 1.2rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.18s ease !important;
    text-transform: uppercase !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
    background: rgba(0,229,204,0.03) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent-primary) !important;
    background: transparent !important;
    font-weight: 600 !important;
}

/* ═══════════════════════════════════════════════════════════════
   FORM CONTROLS
═══════════════════════════════════════════════════════════════ */
div[data-testid="stSlider"] > div { padding: 0; }
.stSelectbox label, .stSlider label, .stNumberInput label {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--text-muted) !important;
    font-size: 0.77rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ═══════════════════════════════════════════════════════════════
   YIELD BADGE / GAUGE RING
═══════════════════════════════════════════════════════════════ */
.yield-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent-primary);
    border-radius: 4px;
    padding: 1.8rem 2rem;
    box-shadow: -2px 0 24px var(--glow), 0 4px 24px rgba(0,0,0,0.06);
    animation: cardGlow 2s ease 0.1s 1;
    text-align: center;
}
.gauge-container { position: relative; display: inline-block; }
.gauge-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

/* ═══════════════════════════════════════════════════════════════
   IMPROVEMENT BADGE
═══════════════════════════════════════════════════════════════ */
.improve-badge {
    background: var(--bg-surface);
    border: 1px solid rgba(0,229,204,0.3);
    border-left: 4px solid var(--accent-primary);
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.improve-badge .ival {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    color: var(--accent-secondary);
    letter-spacing: 0.04em;
    line-height: 1;
}
.improve-badge .ilabel {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

/* ═══════════════════════════════════════════════════════════════
   PARAM TABLE — Terminal readout
═══════════════════════════════════════════════════════════════ */
.param-table {
    font-family: 'IBM Plex Mono', monospace;
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    font-size: 0.8rem;
}
.param-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
    transition: all 0.18s ease;
    position: relative;
}
.param-row::before {
    content: '';
    position: absolute; left: 0; top: 0; bottom: 0;
    width: 0; background: var(--accent-primary);
    transition: width 0.18s ease; opacity: 0.15;
}
.param-row:hover::before { width: 3px; }
.param-row:hover { background: rgba(0,229,204,0.03); }
.param-row:last-child { border-bottom: none; }
.param-row:nth-child(odd)  { background: var(--bg-surface); }
.param-row:nth-child(even) { background: var(--bg-card); }
.param-name  { color: var(--text-secondary); flex: 2.5; font-size: 0.75rem; }
.param-base  { color: var(--text-muted);     flex: 1; text-align: right; }
.param-opt   { flex: 1.2; text-align: right; font-weight: 600; }
.val-up   { color: var(--accent-secondary); }
.val-down { color: var(--accent-crimson); }
.val-same { color: var(--text-muted); }

/* ═══════════════════════════════════════════════════════════════
   INFO BOX
═══════════════════════════════════════════════════════════════ */
.info-box {
    background: var(--bg-surface);
    border-left: 3px solid var(--accent-primary);
    border-radius: 0 4px 4px 0;
    padding: 0.8rem 1.1rem;
    margin: 0.8rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.6;
}

/* ═══════════════════════════════════════════════════════════════
   WORKFLOW STEPPER — vertical timeline
═══════════════════════════════════════════════════════════════ */
.stepper-wrap { position: relative; padding-left: 1.6rem; }
.stepper-wrap::before {
    content: '';
    position: absolute; left: 4px; top: 8px; bottom: 8px;
    width: 1px;
    border-left: 1px dashed var(--accent-primary);
    opacity: 0.5;
}
.step-node {
    position: absolute; left: 0;
    width: 10px; height: 10px;
    background: var(--accent-primary);
    border-radius: 50%;
    margin-top: 4px;
    box-shadow: 0 0 8px var(--glow-strong);
}
.step-item {
    position: relative;
    margin-bottom: 0.9rem;
    padding-left: 0.4rem;
}
.step-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.6rem 0.9rem;
    transition: all 0.18s ease;
}
.step-card:hover {
    border-color: var(--accent-primary);
    box-shadow: -2px 0 12px var(--glow);
    transform: translateX(2px);
}
.step-title {
    font-family: 'DM Serif Display', serif;
    font-size: 0.88rem;
    color: var(--text-primary);
}
.step-desc {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.1rem;
}
.step-link {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent-primary);
    letter-spacing: 0.05em;
}

/* ═══════════════════════════════════════════════════════════════
   PRODUCT PILLS
═══════════════════════════════════════════════════════════════ */
.product-pill {
    text-align: center;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.7rem 0.3rem;
    transition: all 0.18s ease;
}
.product-pill:hover {
    border-color: var(--accent-primary);
    box-shadow: 0 0 12px var(--glow);
    transform: translateY(-2px);
}

/* ═══════════════════════════════════════════════════════════════
   PIPELINE INFO TABLE
═══════════════════════════════════════════════════════════════ */
.pipe-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 1rem;
    border-bottom: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    transition: background 0.18s;
}
.pipe-row:hover { background: rgba(0,229,204,0.04); }
.pipe-row:last-child { border-bottom: none; }
.pipe-key { color: var(--text-secondary); }
.pipe-val { color: var(--accent-secondary); font-weight: 600; }

/* ═══════════════════════════════════════════════════════════════
   STREAMLIT OVERRIDES
═══════════════════════════════════════════════════════════════ */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent-primary);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    box-shadow: -2px 0 10px var(--glow);
}
div[data-testid="stMetric"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stExpander"] {
    background: var(--bg-card);
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
}
[data-testid="stDataFrame"] {
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
}
.stProgress > div > div {
    background: var(--accent-primary) !important;
}
hr { border-color: var(--border) !important; }

/* ═══════════════════════════════════════════════════════════════
   STREAMLIT FORM CONTROL STYLING
   ═══════════════════════════════════════════════════════════════ */
div[data-baseweb="select"] {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
div[data-baseweb="select"] * {
    color: var(--text-primary) !important;
}
div[data-baseweb="popover"] {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
}
div[role="listbox"] {
    background-color: var(--bg-surface) !important;
}
div[role="option"] {
    color: var(--text-primary) !important;
    background-color: var(--bg-surface) !important;
    transition: background-color 0.18s !important;
}
div[role="option"]:hover {
    background-color: var(--bg-card) !important;
    color: var(--accent-primary) !important;
}
input[type="number"], input[type="text"], textarea {
    background-color: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
.stSelectbox [data-baseweb="select"] {
    background-color: var(--bg-surface) !important;
    color: var(--text-primary) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] {
    background-color: transparent !important;
}
div[data-testid="stSlider"] [class*="StyledTrack"] {
    background: var(--border) !important;
}
div[data-testid="stSlider"] [class*="StyledTickBar"] {
    color: var(--text-muted) !important;
}
div[data-testid="stSlider"] [class*="StyledThumb"] {
    background-color: var(--accent-primary) !important;
    border: 2px solid var(--bg-base) !important;
}

/* ═══════════════════════════════════════════════════════════════
   CONFIDENCE BOX
═══════════════════════════════════════════════════════════════ */
.conf-box {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent-secondary);
    border-radius: 4px;
    padding: 0.9rem 1.2rem;
    margin-top: 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
}
.conf-label {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.conf-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent-secondary);
    margin-top: 0.25rem;
}
.conf-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
}

/* ═══════════════════════════════════════════════════════════════
   EMPTY STATE
═══════════════════════════════════════════════════════════════ */
.empty-state {
    background: var(--bg-surface);
    border: 1px dashed var(--border);
    border-radius: 4px;
    padding: 3rem 2rem;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
}
.empty-state .es-icon { font-size: 2.5rem; opacity: 0.5; }
.empty-state .es-text { color: var(--text-muted); font-size: 0.82rem; margin-top: 0.8rem; }
.empty-state .es-cmd  { color: var(--accent-primary); font-weight: 600; }

/* ═══════════════════════════════════════════════════════════════
   OPTIMIZER WRAPPER CARD
═══════════════════════════════════════════════════════════════ */
.opt-config-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load Pipeline (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML pipeline…")
def load_pipeline():
    path = BASE_DIR / "fermentation_pipeline.pkl"
    if not path.exists():
        st.error("Pipeline file not found. Please run `train_models.py` first.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Loading dataset…")
def load_dataset():
    path = BASE_DIR / "cleaned_fermentation_dataset.csv"
    if path.exists():
        return pd.read_csv(path, nrows=5000)
    return None


pipeline = load_pipeline()
df_data  = load_dataset()

MODEL      = pipeline["model"]
SCALER     = pipeline["scaler"]
FEAT_COLS  = pipeline["feature_columns"]
CAT_COLS   = pipeline["cat_cols"]
NUM_COLS   = pipeline["num_cols"]
MAPPINGS   = pipeline["mappings"]
METRICS    = pipeline["metrics"]
MEDIANS    = pipeline["medians"]
BEST_NAME  = pipeline["best_model_name"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def encode_row(row: dict) -> np.ndarray:
    """Encode a raw input dict → scaled numpy vector for the model."""
    encoded = {}
    for col in FEAT_COLS:
        val = row.get(col, MEDIANS.get(col, 0))
        if col in CAT_COLS and col in MAPPINGS:
            if isinstance(val, str):
                val = MAPPINGS[col].get(val, 0)
        encoded[col] = val
    vec = np.array([encoded[c] for c in FEAT_COLS], dtype=float)
    return SCALER.transform(vec.reshape(1, -1))


def predict_yield(row: dict) -> float:
    return float(MODEL.predict(encode_row(row))[0])


def plotly_theme():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0c1426",
        font=dict(color="#8fb8b3", family="IBM Plex Mono"),
        xaxis=dict(gridcolor="#1b2a47", linecolor="#507a75", zerolinecolor="#1b2a47"),
        yaxis=dict(gridcolor="#1b2a47", linecolor="#507a75", zerolinecolor="#1b2a47"),
        colorway=["#00e5cc", "#00b3a0", "#f59e0b", "#ef4444", "#6366f1"],
    )


def gauge_svg(value: float, max_val: float = 50.0) -> str:
    """Generate a circular SVG gauge ring for the yield result."""
    pct     = min(1.0, max(0.0, value / max_val))
    radius  = 70
    cx = cy = 90
    circumference = 2 * 3.14159 * radius
    offset  = circumference * (1 - pct)
    color   = "#00e5cc"
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;">
      <svg width="180" height="180" viewBox="0 0 180 180">
        <!-- Track -->
        <circle cx="{cx}" cy="{cy}" r="{radius}"
          fill="none" stroke="#1b2a47" stroke-width="8"/>
        <!-- Arc -->
        <circle cx="{cx}" cy="{cy}" r="{radius}"
          fill="none" stroke="{color}" stroke-width="8"
          stroke-linecap="round"
          stroke-dasharray="{circumference:.1f}"
          stroke-dashoffset="{offset:.1f}"
          transform="rotate(-90 {cx} {cy})"
          style="transition: stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1);">
        </circle>
        <!-- Value -->
        <text x="{cx}" y="{cy - 6}" text-anchor="middle"
          font-family="'Bebas Neue', sans-serif"
          font-size="34" fill="#e2f3f0" letter-spacing="1">{value:.2f}</text>
        <text x="{cx}" y="{cy + 16}" text-anchor="middle"
          font-family="'IBM Plex Mono', monospace"
          font-size="10" fill="#8fb8b3" letter-spacing="2">g/L or %</text>
        <text x="{cx}" y="{cy + 30}" text-anchor="middle"
          font-family="'IBM Plex Mono', monospace"
          font-size="9" fill="#00e5cc">{pct*100:.1f}% of range</text>
      </svg>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.4rem 1rem 1rem 1rem;
                border-bottom: 1px solid var(--border); margin-bottom: 1rem;'>
        <div style='font-family:"DM Serif Display",serif; font-style:italic;
                    font-size:1.6rem; color:#00e5cc; line-height:1; margin-bottom:0.2rem;'>
            FermentAI
        </div>
        <div style='font-family:"IBM Plex Mono",monospace; font-size:0.65rem;
                    color:#8fb8b3; letter-spacing:0.1em; text-transform:uppercase;'>
            microbial intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state["page"] = "🏠  Overview"

    PAGE_OPTIONS = ["🏠  Overview", "🔬  Yield Predictor", "⚙️  Optimizer", "📊  Visualizations", "📋  Model Info"]

    # Custom styled nav
    for opt in PAGE_OPTIONS:
        is_active = st.session_state.get("page") == opt
        bg    = "rgba(0,229,204,0.07)" if is_active else "transparent"
        color = "#00e5cc" if is_active else "#8fb8b3"
        bar   = "3px solid #00e5cc" if is_active else "3px solid transparent"
        st.markdown(f"""
        <div style='border-left:{bar}; background:{bg}; padding:0.48rem 0.8rem;
                    font-family:"IBM Plex Mono",monospace; font-size:0.82rem;
                    color:{color}; cursor:pointer; transition:all 0.18s;
                    margin-bottom:2px; border-radius:0 2px 2px 0;'>
            {opt}
        </div>
        """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        PAGE_OPTIONS,
        index=PAGE_OPTIONS.index(st.session_state["page"]),
        key="nav_radio",
        label_visibility="collapsed",
    )
    st.session_state["page"] = page

    st.markdown("<div style='border-top:1px solid var(--border); margin: 0.8rem 0;'></div>", unsafe_allow_html=True)

    r2_best = METRICS[BEST_NAME]["R2"]
    st.markdown(f"""
    <div style='font-family:"IBM Plex Mono",monospace; padding: 0 0.3rem;'>
        <div style='font-size:0.62rem; color:var(--text-secondary); text-transform:uppercase;
                    letter-spacing:0.1em; margin-bottom:0.6rem;'>System Status</div>
        {"".join([
            f"<div style='display:flex; justify-content:space-between; padding:0.3rem 0;"
            f"border-bottom:1px solid var(--border); font-size:0.75rem;'>"
            f"<span style='color:var(--text-secondary);'>{k}</span>"
            f"<span style='color:{vc};font-weight:600;'>{v}</span></div>"
            for k, v, vc in [
                ("best model", BEST_NAME, "#00e5cc"),
                ("R² score",   f"{r2_best:.4f}", "#00b3a0"),
                ("features",   str(len(FEAT_COLS)), "#f59e0b"),
                ("mode",       "regression", "var(--text-secondary)"),
            ]
        ])}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:"IBM Plex Mono",monospace; font-size:0.62rem; color:var(--text-secondary);
                text-align:center; margin-top:1.2rem; letter-spacing:0.06em;'>
        clostridium AK1 × bioethanol<br>dataset v2.0
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠  Overview":
    st.markdown('<div class="section-header fade-1">AI Fermentation Yield Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-underline"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub fade-2">machine learning platform · industrial biotechnology · yield optimization</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Best R² Score",    f"{METRICS[BEST_NAME]['R2']:.4f}", BEST_NAME, "fade-1"),
        ("Lowest RMSE",      f"{METRICS[BEST_NAME]['RMSE']:.3f}", "g/L or %", "fade-2"),
        ("Feature Count",    str(len(FEAT_COLS)), "lab + macro variables", "fade-3"),
        ("Yield Improvement","+34.6%", "via differential evolution", "fade-4"),
    ]
    for col, (label, val, sub, fade) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card {fade}">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        <div style='font-family:"DM Serif Display",serif; font-size:1.1rem;
                    color:var(--text-primary); margin-bottom:0.3rem;'>System Workflow</div>
        <div style='font-family:"IBM Plex Mono",monospace; font-size:0.72rem;
                    color:#8fb8b3; margin-bottom:1rem; letter-spacing:0.04em;'>
            click [→ open] to navigate to each module
        </div>
        """, unsafe_allow_html=True)

        steps = [
            ("🔬", "Raw Data",       "clostridium AK1 + bioethanol datasets",     None),
            ("🧹", "Cleaning",       "IQR outlier removal · LOD handling · merge", None),
            ("⚙️", "Preprocessing",  "StandardScaler + LabelEncoder pipeline",     "📋  Model Info"),
            ("🤖", "ML Models",      "Linear Reg · Random Forest · XGBoost",       "📋  Model Info"),
            ("📈", "Prediction",     "fermentation yield estimation",              "🔬  Yield Predictor"),
            ("🎯", "Optimization",   "differential evolution · global search",     "⚙️  Optimizer"),
            ("📊", "Visualizations", "feature importance · distribution charts",  "📊  Visualizations"),
        ]

        st.markdown("<div class='stepper-wrap'>", unsafe_allow_html=True)
        for i, (icon, title, desc, nav_target) in enumerate(steps):
            col_step, col_btn = st.columns([5, 1])
            with col_step:
                st.markdown(f"""
                <div class='step-item'>
                    <div class='step-node' style='top:{i*0}px'></div>
                    <div class='step-card'>
                        <div class='step-title'>{icon} {title}</div>
                        <div class='step-desc'>{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_btn:
                if nav_target:
                    if st.button("→ open", key=f"nav_{title}", help=f"Go to {nav_target}"):
                        st.session_state["page"] = nav_target
                        st.rerun()
                else:
                    st.markdown("<div style='padding:0.6rem 0; font-family:\"IBM Plex Mono\",monospace; font-size:0.72rem; color:var(--border); text-align:center;'>·</div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.25rem'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div style='font-family:"DM Serif Display",serif; font-size:1.1rem;
                    color:var(--text-primary); margin-bottom:0.8rem;'>Model Comparison</div>
        """, unsafe_allow_html=True)

        metrics_rows = [{"Model": mname, **mvals} for mname, mvals in METRICS.items()]
        df_metrics = pd.DataFrame(metrics_rows)

        fig_cmp = go.Figure()
        colors = ["#8fb8b3", "#00e5cc", "#00b3a0"]
        for i, row in df_metrics.iterrows():
            fig_cmp.add_trace(go.Bar(
                name=row["Model"],
                x=["R²", "MAE", "RMSE"],
                y=[row["R2"], row["MAE"], row["RMSE"]],
                marker_color=colors[i], marker_line_width=0,
                text=[f'{row["R2"]:.3f}', f'{row["MAE"]:.3f}', f'{row["RMSE"]:.3f}'],
                textposition="outside",
                textfont=dict(size=9, color="#8fb8b3", family="IBM Plex Mono"),
            ))
        fig_cmp.update_layout(
            barmode="group", height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(font=dict(size=9, color="#8fb8b3", family="IBM Plex Mono"),
                        bgcolor="rgba(0,0,0,0)"),
            **plotly_theme(),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown("""
        <div style='font-family:"DM Serif Display",serif; font-size:1.0rem;
                    color:var(--text-primary); margin: 0.5rem 0 0.7rem 0;'>Target Products</div>
        """, unsafe_allow_html=True)
        products = [("🍶", "Ethanol"), ("🍋", "Citric"), ("🧪", "Lactic"), ("🦠", "Biomass"), ("⚗️", "Enzymes")]
        cols_p = st.columns(5)
        for idx, (icon, name) in enumerate(products):
            with cols_p[idx]:
                st.markdown(f"""
                <div class="product-pill">
                    <div style='font-size:1.4rem;'>{icon}</div>
                    <div style='font-family:"IBM Plex Mono",monospace; font-size:0.6rem;
                                color:#8fb8b3; margin-top:0.25rem;'>{name}</div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 – Yield Predictor
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬  Yield Predictor":
    st.markdown('<div class="section-header fade-1">Fermentation Yield Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-underline"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub fade-2">adjust process parameters · predict fermentation yield · real-time inference</div>', unsafe_allow_html=True)

    tab_lab, tab_macro = st.tabs(["[🧫] Lab Parameters — Clostridium AK1", "[🌍] Macro / Country Parameters"])

    with tab_lab:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            substrate = st.selectbox("Substrate", ["D-Glucose", "Control", "L-Fucose", "L-Rhamnose"], index=0)
            temp      = st.slider("Temperature (°C)", 20.0, 80.0, float(MEDIANS.get("Temperature_C", 37.0)), 0.5)
            init_ph   = st.slider("Initial pH", 3.0, 9.0, float(MEDIANS.get("Initial_pH", 6.5)), 0.1)
            final_ph  = st.slider("Final pH", 3.0, 9.0, float(MEDIANS.get("Final_pH", 5.8)), 0.1)

        with c2:
            sub_conc  = st.slider("Substrate Concentration (mM)", 1.0, 200.0, float(MEDIANS.get("Substrate_Concentration_mM", 20.0)), 1.0)
            lg_ratio  = st.slider("L:G Ratio", 0.01, 1.0, float(MEDIANS.get("LG_Ratio", 0.09)), 0.01)
            phosphate = st.slider("Phosphate Concentration (mM)", 0.0, 100.0, float(MEDIANS.get("Phosphate_Concentration_mM", 15.0)), 0.5)
            od600     = st.slider("OD600 (Optical Density)", 0.01, 5.0, float(MEDIANS.get("OD600", 0.15)), 0.01)

        with c3:
            ethanol   = st.slider("Ethanol (mM)", 0.0, 100.0, float(MEDIANS.get("Ethanol_mM", 3.61)), 0.1)
            acetate   = st.slider("Acetate (mM)", 0.0, 100.0, float(MEDIANS.get("Acetate_mM", 4.48)), 0.1)
            butyrate  = st.slider("Butyrate (mM)", 0.0, 50.0, float(MEDIANS.get("Butyrate_mM", 0.2)), 0.05)
            lactate   = st.slider("Lactate (mM)", 0.0, 100.0, float(MEDIANS.get("Lactate_mM", 0.0)), 0.1)

        c4, c5, c6 = st.columns(3)
        with c4:
            sugar_pct = st.slider("Sugar Consumed (%)", 0.0, 100.0, float(MEDIANS.get("Sugar_Consumed_pct", 60.75)), 1.0)
        with c5:
            h2_mm     = st.slider("H₂ (mM)", 0.0, 200.0, float(MEDIANS.get("H2_mM", 0.0)), 1.0)
        with c6:
            pdyield   = st.slider("1,2-PD Yield (%)", 0.0, 100.0, float(MEDIANS.get("1_2-PD_Yield_pct", 0.0)), 0.5)

    with tab_macro:
        st.markdown("<br>", unsafe_allow_html=True)
        cm1, cm2, cm3 = st.columns(3)
        with cm1:
            country    = st.selectbox("Country", ["India", "Brazil", "China", "Germany", "USA"], index=0)
            year       = st.slider("Year", 2000, 2030, int(MEDIANS.get("Year", 2012)))
            mkt_demand = st.slider("Market Demand (index)", 0.0, 10.0, float(MEDIANS.get("Market_Demand", 5.0)), 0.1)
        with cm2:
            feedstock_yield = st.slider("Feedstock Yield", 0.0, 200.0, float(MEDIANS.get("Feedstock_Yield", 100.0)), 1.0)
            prod_cap   = st.slider("Production Capacity", 0.0, 1e6, float(MEDIANS.get("Production_Capacity", 50000.0)), 1000.0, format="%.0f")
            energy     = st.slider("Energy Consumption", 0.0, 1e5, float(MEDIANS.get("Energy_Consumption", 10000.0)), 100.0, format="%.0f")
        with cm3:
            price_gal  = st.slider("Price Per Gallon ($)", 0.0, 10.0, float(MEDIANS.get("Price_Per_Gallon", 2.5)), 0.1)
            govt_inc   = st.slider("Govt Incentive Score", 0.0, 10.0, float(MEDIANS.get("Govt_Incentive", 5.0)), 0.1)
            source     = st.selectbox("Dataset Source", ["Clostridium_AK1_Lab_Data", "Bioethanol_Growth_Prediction_Dataset"], index=0)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, col_yield = st.columns([1, 2])

    with col_btn:
        run_pred = st.button("[ predict yield ]", use_container_width=True)
        st.markdown("""
        <div class="info-box">
            inference uses <strong>Random Forest</strong> — best model (highest R²) trained on
            merged clostridium AK1 lab data + bioethanol macro dataset.
        </div>
        """, unsafe_allow_html=True)

    with col_yield:
        if run_pred:
            input_row = {
                "Substrate":                  substrate,
                "Substrate_Concentration_mM": sub_conc,
                "Temperature_C":              temp,
                "Initial_pH":                 init_ph,
                "LG_Ratio":                   lg_ratio,
                "Phosphate_Concentration_mM": phosphate,
                "Sugar_Consumed_pct":          sugar_pct,
                "Ethanol_mM":                 ethanol,
                "Acetate_mM":                 acetate,
                "Butyrate_mM":                butyrate,
                "1_2-PD_mM":                  MEDIANS.get("1_2-PD_mM", 0.0),
                "Lactate_mM":                 lactate,
                "H2_mM":                      h2_mm,
                "OD600":                      od600,
                "Final_pH":                   final_ph,
                "1_2-PD_Yield_pct":           pdyield,
                "Carbon_Balance_pct":         MEDIANS.get("Carbon_Balance_pct", 100.0),
                "Microbial_Strain":           "Clostridium strain AK1",
                "Dataset_Source":             source,
                "Year":                       year,
                "Country":                    country,
                "Feedstock_Yield":            feedstock_yield,
                "Production_Capacity":        prod_cap,
                "Processing_Tech_Efficiency": MEDIANS.get("Processing_Tech_Efficiency", 80.0),
                "Energy_Consumption":         energy,
                "Feedstock_Cost":             MEDIANS.get("Feedstock_Cost", 100.0),
                "Transportation_Cost":        MEDIANS.get("Transportation_Cost", 20.0),
                "Distribution_Cost":          MEDIANS.get("Distribution_Cost", 15.0),
                "Carbon_Emissions":           MEDIANS.get("Carbon_Emissions", 50.0),
                "Water_Usage":                MEDIANS.get("Water_Usage", 100.0),
                "Market_Demand":              mkt_demand,
                "Price_Per_Gallon":           price_gal,
                "Govt_Incentive":             govt_inc,
            }
            pred = predict_yield(input_row)

            st.markdown(f"""
            <div class="yield-card fade-1">
                <div style='font-family:"IBM Plex Mono",monospace; font-size:0.68rem;
                            color:#8fb8b3; text-transform:uppercase; letter-spacing:0.12em;
                            margin-bottom:0.8rem;'>prediction result</div>
                {gauge_svg(pred, 50.0)}
            </div>
            """, unsafe_allow_html=True)

            try:
                tree_preds = np.array([t.predict(encode_row(input_row))[0] for t in MODEL.estimators_])
                conf_std   = tree_preds.std()
                conf_range = (pred - 2 * conf_std, pred + 2 * conf_std)
                st.markdown(f"""
                <div class="conf-box fade-2">
                    <div class="conf-label">95% Confidence Interval</div>
                    <div class="conf-value">{conf_range[0]:.3f} — {conf_range[1]:.3f}</div>
                    <div class="conf-sub">std dev across {len(MODEL.estimators_)} estimators: ±{conf_std:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                pass
        else:
            st.markdown("""
            <div class="empty-state fade-1">
                <div class="es-icon">🔬</div>
                <div class="es-text">
                    set parameters and run<br>
                    <span class="es-cmd">[ predict yield ]</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – Optimizer
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚙️  Optimizer":
    st.markdown('<div class="section-header fade-1">Fermentation Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-underline"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub fade-2">differential evolution · global parameter search · maximum yield discovery</div>', unsafe_allow_html=True)

    NUMERIC_BOUNDS = {
        "Substrate_Concentration_mM": (1.0,   200.0),
        "Temperature_C":              (20.0,  80.0),
        "Initial_pH":                 (3.0,   9.0),
        "LG_Ratio":                   (0.01,  1.0),
        "Phosphate_Concentration_mM": (0.0,   100.0),
        "Sugar_Consumed_pct":         (0.0,   100.0),
        "Ethanol_mM":                 (0.0,   100.0),
        "Acetate_mM":                 (0.0,   100.0),
        "Butyrate_mM":                (0.0,   50.0),
        "1_2-PD_mM":                  (0.0,   50.0),
        "Lactate_mM":                 (0.0,   100.0),
        "H2_mM":                      (0.0,   200.0),
        "OD600":                      (0.01,  5.0),
        "Final_pH":                   (3.0,   9.0),
        "1_2-PD_Yield_pct":           (0.0,   100.0),
        "Carbon_Balance_pct":         (50.0,  150.0),
    }
    UNITS_MAP = {
        "Temperature_C": "°C", "Initial_pH": "", "Final_pH": "",
        "Substrate_Concentration_mM": "mM", "LG_Ratio": "",
        "Phosphate_Concentration_mM": "mM", "Sugar_Consumed_pct": "%",
        "Ethanol_mM": "mM", "Acetate_mM": "mM", "Butyrate_mM": "mM",
        "Lactate_mM": "mM", "H2_mM": "mM", "OD600": "",
        "1_2-PD_Yield_pct": "%", "Carbon_Balance_pct": "%",
    }

    col_cfg, col_res = st.columns([1, 2])

    with col_cfg:
        st.markdown('<div class="opt-config-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:"DM Serif Display",serif; font-size:1rem;
                    color:#0d1f1e; margin-bottom:0.8rem;'>Configuration</div>
        """, unsafe_allow_html=True)
        opt_substrate = st.selectbox("Substrate", ["D-Glucose", "Control", "L-Fucose", "L-Rhamnose"], index=0, key="opt_sub")
        opt_country   = st.selectbox("Country", ["India", "Brazil", "China", "Germany", "USA"], index=0, key="opt_cou")
        opt_maxiter   = st.slider("DE Max Iterations", 50, 500, 150, 25)
        opt_popsize   = st.slider("Population Size", 5, 30, 12, 1)
        opt_seed      = st.number_input("Random Seed", value=42, step=1)
        st.markdown("""
        <div class="info-box">
            <strong>differential evolution</strong> searches the full parameter space
            simultaneously — finds the global maximum rather than a local optimum.
            more iterations → higher accuracy, longer runtime.
        </div>
        """, unsafe_allow_html=True)
        run_opt = st.button("[ run optimization ]", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        if run_opt:
            fixed_cats = {
                "Substrate":        MAPPINGS["Substrate"].get(opt_substrate, 0),
                "Microbial_Strain": MAPPINGS["Microbial_Strain"].get("Clostridium strain AK1", 0),
                "Dataset_Source":   MAPPINGS["Dataset_Source"].get("Clostridium_AK1_Lab_Data", 1),
                "Country":          MAPPINGS["Country"].get(opt_country, 3),
            }
            opt_num_cols = [c for c in FEAT_COLS if c not in CAT_COLS and c in NUMERIC_BOUNDS]
            bounds       = [NUMERIC_BOUNDS[c] for c in opt_num_cols]
            baseline_vec = np.array([MEDIANS.get(c, (NUMERIC_BOUNDS[c][0] + NUMERIC_BOUNDS[c][1]) / 2)
                                      for c in opt_num_cols])

            def full_vec(x):
                row = dict(zip(opt_num_cols, x))
                row.update(fixed_cats)
                return np.array([row.get(c, 0.0) for c in FEAT_COLS], dtype=float)

            def objective(x):
                return -float(MODEL.predict(SCALER.transform(full_vec(x).reshape(1, -1)))[0])

            baseline_yield = -objective(baseline_vec)
            prog = st.progress(0, text="running differential evolution…")
            history = []

            class Callback:
                def __init__(self): self.gen = 0
                def __call__(self, xk, convergence):
                    self.gen += 1
                    pct = min(int(self.gen / opt_maxiter * 100), 100)
                    prog.progress(pct, text=f"generation {self.gen}/{opt_maxiter} · convergence: {convergence:.4f}")
                    history.append(-objective(xk))

            cb = Callback()
            with st.spinner("optimizing…"):
                result = differential_evolution(
                    objective, bounds=bounds, seed=int(opt_seed),
                    popsize=opt_popsize, maxiter=opt_maxiter, tol=1e-6,
                    mutation=(0.5, 1.5), recombination=0.9, strategy="best1bin",
                    updating="deferred", workers=1, polish=True, callback=cb,
                )
            prog.progress(100, text="complete.")

            best_yield  = -result.fun
            improvement = ((best_yield - baseline_yield) / (abs(baseline_yield) + 1e-9)) * 100

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""<div class="metric-card fade-1">
                    <div class="label">Baseline Yield</div>
                    <div class="value">{baseline_yield:.3f}</div>
                    <div class="sub">median conditions</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class="metric-card fade-2">
                    <div class="label">Optimised Yield</div>
                    <div class="value">{best_yield:.3f}</div>
                    <div class="sub">DE best solution</div>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""<div class="improve-badge fade-3">
                    <div class="ival">{improvement:+.1f}%</div>
                    <div class="ilabel">improvement</div>
                </div>""", unsafe_allow_html=True)

            if history:
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(
                    y=history, mode="lines",
                    line=dict(color="#00e5cc", width=2),
                    fill="tozeroy", fillcolor="rgba(0,229,204,0.06)",
                    name="best yield / generation",
                ))
                fig_conv.add_hline(y=baseline_yield, line_dash="dash",
                                   line_color="#8fb8b3",
                                   annotation_text=f"baseline: {baseline_yield:.3f}",
                                   annotation_font=dict(color="#8fb8b3", family="IBM Plex Mono", size=10))
                fig_conv.update_layout(
                    title=dict(text="convergence history", font=dict(family="IBM Plex Mono", size=11, color="#8fb8b3")),
                    xaxis_title="generation", yaxis_title="predicted yield",
                    height=240, margin=dict(l=10, r=10, t=36, b=10),
                    **plotly_theme(),
                )
                st.plotly_chart(fig_conv, use_container_width=True)

            # Terminal-style parameter table
            st.markdown("""
            <div style='font-family:"DM Serif Display",serif; font-size:1rem;
                        color:var(--text-primary); margin: 0.8rem 0 0.5rem 0;'>
                Baseline vs. Optimised Parameters
            </div>
            """, unsafe_allow_html=True)
            opt_x = result.x
            rows_html = ""
            for i, col in enumerate(opt_num_cols):
                if col not in UNITS_MAP:
                    continue
                bv   = MEDIANS.get(col, 0)
                ov   = opt_x[i]
                unit = UNITS_MAP.get(col, "")
                if ov > bv:
                    sym  = "▲"
                    cls  = "val-up"
                elif ov < bv:
                    sym  = "▼"
                    cls  = "val-down"
                else:
                    sym  = "→"
                    cls  = "val-same"
                rows_html += f"""
                <div class="param-row">
                    <div class="param-name">{col.replace('_',' ')}</div>
                    <div class="param-base">{bv:.3f} {unit}</div>
                    <div class="param-opt {cls}">{sym} {ov:.3f} {unit}</div>
                </div>"""

            st.markdown(f"""
            <div class="param-table">
                <div class="param-row" style="background:var(--bg-surface); font-weight:600;">
                    <div class="param-name" style="color:#8fb8b3;">parameter</div>
                    <div class="param-base" style="color:#8fb8b3;">baseline</div>
                    <div class="param-opt"  style="color:#00e5cc;">optimised</div>
                </div>
                {rows_html}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state fade-1" style="margin-top:1.5rem; padding:4rem 2rem;">
                <div class="es-icon">⚙️</div>
                <div class="es-text">
                    configure settings and run<br>
                    <span class="es-cmd">[ run optimization ]</span><br>
                    <span style='color:#8fb8b3;'>engine finds the global yield maximum</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 – Visualizations
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊  Visualizations":
    st.markdown('<div class="section-header fade-1">Data Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-underline"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub fade-2">feature distributions · correlation analysis · model insights</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "[📊] Feature Importance",
        "[🔥] Correlation Heatmap",
        "[📈] Data Explorer",
    ])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style='font-family:"DM Serif Display",serif; font-size:1rem;
                    color:var(--text-primary); margin-bottom:0.6rem;'>Feature Importance — Random Forest</div>""",
                    unsafe_allow_html=True)
        img_path = BASE_DIR / "feature_importance.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            importances = MODEL.feature_importances_
            feat_imp_df = pd.DataFrame({
                "Feature": FEAT_COLS, "Importance": importances,
            }).sort_values("Importance", ascending=False).head(20)
            fig_imp = px.bar(
                feat_imp_df, x="Importance", y="Feature", orientation="h",
                color="Importance",
                color_continuous_scale=["#1b2a47", "#00b3a0", "#00e5cc"],
                title="top 20 feature importances",
            )
            fig_imp.update_layout(height=550, yaxis=dict(autorange="reversed"),
                                  margin=dict(l=10, r=10, t=40, b=10),
                                  coloraxis_showscale=False, **plotly_theme())
            st.plotly_chart(fig_imp, use_container_width=True)

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style='font-family:"DM Serif Display",serif; font-size:1rem;
                    color:var(--text-primary); margin-bottom:0.6rem;'>Correlation Heatmap</div>""",
                    unsafe_allow_html=True)
        img_path2 = BASE_DIR / "correlation_heatmap.png"
        if img_path2.exists():
            st.image(str(img_path2), use_container_width=True)
        elif df_data is not None:
            numeric_df = df_data.select_dtypes(include=np.number).iloc[:, :15]
            corr = numeric_df.corr()
            fig_heat = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale=[[0, "#ef4444"], [0.5, "#0c1426"], [1, "#00e5cc"]],
                zmid=0, text=np.round(corr.values, 2),
                texttemplate="%{text}", textfont=dict(size=8, family="IBM Plex Mono"),
            ))
            fig_heat.update_layout(height=560, margin=dict(l=10, r=10, t=20, b=10), **plotly_theme())
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Heatmap image not found and dataset unavailable.")

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        if df_data is not None:
            numeric_cols_avail = df_data.select_dtypes(include=np.number).columns.tolist()
            sel_col  = st.selectbox("Select Feature", numeric_cols_avail, index=0)
            color_by = st.selectbox("Color By", ["None"] + [c for c in ["Dataset_Source", "Country", "Substrate"] if c in df_data.columns], index=0)

            c_left, c_right = st.columns(2)
            with c_left:
                fig_hist = px.histogram(
                    df_data, x=sel_col,
                    color=None if color_by == "None" else color_by,
                    nbins=50,
                    color_discrete_sequence=["#00e5cc", "#00b3a0", "#f59e0b", "#ef4444", "#6366f1"],
                    title=f"distribution — {sel_col}",
                )
                fig_hist.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10),
                                       showlegend=color_by != "None", **plotly_theme())
                st.plotly_chart(fig_hist, use_container_width=True)

            with c_right:
                if "Yield" in df_data.columns:
                    sample_df = df_data[[sel_col, "Yield"]].dropna().sample(min(2000, len(df_data)))
                    fig_scatter = px.scatter(
                        sample_df, x=sel_col, y="Yield",
                        opacity=0.45, color_discrete_sequence=["#00e5cc"],
                        title=f"{sel_col} vs yield",
                    )
                    try:
                        valid = sample_df[[sel_col, "Yield"]].dropna()
                        z  = np.polyfit(valid[sel_col], valid["Yield"], 1)
                        p  = np.poly1d(z)
                        xs = np.linspace(valid[sel_col].min(), valid[sel_col].max(), 100)
                        fig_scatter.add_trace(go.Scatter(
                            x=xs, y=p(xs), mode="lines",
                            line=dict(color="#f59e0b", width=2, dash="dash"),
                            name="trend",
                        ))
                    except Exception:
                        pass
                    fig_scatter.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10),
                                             showlegend=False, **plotly_theme())
                    st.plotly_chart(fig_scatter, use_container_width=True)

            if "Substrate" in df_data.columns and "Yield" in df_data.columns:
                fig_box = px.box(
                    df_data, x="Substrate", y="Yield", color="Substrate",
                    color_discrete_sequence=["#00e5cc", "#00b3a0", "#f59e0b", "#ef4444"],
                    title="yield distribution by substrate",
                )
                fig_box.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10),
                                      showlegend=False, **plotly_theme())
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Dataset not found. Place `cleaned_fermentation_dataset.csv` in the project folder.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""<div style='font-family:"DM Serif Display",serif; font-size:1rem;
                color:var(--text-primary); margin-bottom:0.6rem;'>Optimization Convergence History</div>""",
                unsafe_allow_html=True)
    opt_img = BASE_DIR / "optimisation_history.png"
    if opt_img.exists():
        st.image(str(opt_img), use_container_width=True)
    else:
        st.markdown("""
        <div class="info-box">run the optimizer page first to generate the convergence plot.</div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 – Model Info
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋  Model Info":
    st.markdown('<div class="section-header fade-1">Model Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-underline"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub fade-2">pipeline architecture · performance metrics · feature catalogue</div>', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["[🏆] Model Comparison", "[🔩] Pipeline Details", "[📄] Feature List"])

    with t1:
        st.markdown("<br>", unsafe_allow_html=True)
        metrics_data = []
        for model_name, mvals in METRICS.items():
            metrics_data.append({
                "Model":    model_name,
                "R² Score": mvals["R2"],
                "MAE":      mvals["MAE"],
                "MSE":      mvals["MSE"],
                "RMSE":     mvals["RMSE"],
                "Selected": "★ best" if model_name == BEST_NAME else "",
            })
        df_m = pd.DataFrame(metrics_data)
        st.dataframe(
            df_m.style.highlight_max(subset=["R² Score"], color="rgba(0,229,204,0.18)")
                      .highlight_min(subset=["MAE", "MSE", "RMSE"], color="rgba(0,229,204,0.18)"),
            use_container_width=True, hide_index=True,
        )

        categories = ["R² Score", "1/MAE (norm)", "1/RMSE (norm)"]
        fig_radar  = go.Figure()
        colors_r   = ["#8fb8b3", "#00e5cc", "#00b3a0"]
        for i, (mname, mvals) in enumerate(METRICS.items()):
            r2_n    = max(0, mvals["R2"])
            inv_mae = 1 / (mvals["MAE"] + 1e-6)
            inv_rmse= 1 / (mvals["RMSE"] + 1e-6)
            r = [r2_n, inv_mae / 0.35, inv_rmse / 0.35]
            fig_radar.add_trace(go.Scatterpolar(
                r=r + [r[0]], theta=categories + [categories[0]],
                name=mname, line=dict(color=colors_r[i], width=2),
                fill="toself", fillcolor=colors_r[i] + "1a",
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#0c1426",
                radialaxis=dict(visible=True, color="#8fb8b3", gridcolor="#1b2a47"),
                angularaxis=dict(color="#8fb8b3", tickfont=dict(family="IBM Plex Mono", size=10)),
            ),
            height=380, margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(font=dict(color="#8fb8b3", family="IBM Plex Mono", size=10),
                        bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with t2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style='font-family:"DM Serif Display",serif; font-size:1rem;
                    color:var(--text-primary); margin-bottom:0.6rem;'>Pipeline Architecture</div>""",
                    unsafe_allow_html=True)
        pipeline_info = {
            "best model":           BEST_NAME,
            "model type":           "RandomForestRegressor",
            "number of trees":      "100",
            "scaler":               "StandardScaler",
            "imputer":              "median (via medians dict)",
            "categorical encoding": "label encoding (integer map)",
            "target variable":      "Yield",
            "train / test split":   "80% / 20%",
            "random state":         "42",
            "total features":       str(len(FEAT_COLS)),
            "categorical features": str(len(CAT_COLS)),
            "numerical features":   str(len(NUM_COLS)),
        }
        st.markdown("""
        <div style='background:var(--bg-card); border:1px solid var(--border);
                    border-radius:4px; overflow:hidden;'>
        """, unsafe_allow_html=True)
        for key, val in pipeline_info.items():
            st.markdown(f"""
            <div class="pipe-row">
                <div class="pipe-key">{key}</div>
                <div class="pipe-val">{val}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style='font-family:"DM Serif Display",serif; font-size:1rem;
                    color:var(--text-primary); margin-bottom:0.6rem;'>Categorical Encodings</div>""",
                    unsafe_allow_html=True)
        for cat, mapping in MAPPINGS.items():
            with st.expander(f"  {cat}"):
                for k, v in mapping.items():
                    st.markdown(f"`{k}` → `{v}`")

    with t3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style='font-family:"DM Serif Display",serif; font-size:1rem;
                    color:var(--text-primary); margin-bottom:0.6rem;'>All Feature Columns</div>""",
                    unsafe_allow_html=True)
        feat_rows = []
        for col in FEAT_COLS:
            ftype = "categorical" if col in CAT_COLS else "numerical"
            med   = MEDIANS.get(col, "—")
            med_s = f"{med:.3f}" if isinstance(med, float) else str(med)
            imp   = MODEL.feature_importances_[FEAT_COLS.index(col)]
            feat_rows.append({"feature": col, "type": ftype, "median": med_s, "importance": round(imp, 5)})
        df_feat = pd.DataFrame(feat_rows).sort_values("importance", ascending=False)
        st.dataframe(df_feat, use_container_width=True, hide_index=True, height=600)
