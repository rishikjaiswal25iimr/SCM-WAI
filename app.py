"""
app.py
======
AI-Driven Warehouse Slotting and Order-Picking Optimization
A Decision Support Framework for BigBasket — Streamlit Application

This file builds the complete interactive executive dashboard. All
analytical logic lives in core.py; this file focuses on layout, shared
state, visual design, and managerial storytelling.

IMPORTANT: This project uses a synthetic/proxy dataset constructed for
academic analysis. It does not represent confidential BigBasket
operational data or actual BigBasket warehouse performance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core as c

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="BigBasket Warehouse Slotting Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "bigbasket_warehouse_slotting_2000_order_lines.csv"

# ----------------------------------------------------------------------
# DESIGN SYSTEM — colours, typography, card / section styling
# ----------------------------------------------------------------------
PRIMARY_COLOR = "#1F4E8C"       # executive navy-blue
PRIMARY_DARK = "#12305C"
ACCENT_COLOR = "#F59E0B"        # amber accent
POSITIVE_COLOR = "#0F9D58"      # improvement / good
NEGATIVE_COLOR = "#D93025"      # warning / negative
NEUTRAL_COLOR = "#1F4E8C"
WARNING_COLOR = "#B45309"
PAGE_BG = "#F4F6FA"
CARD_BG = "#FFFFFF"

TONE_STYLES = {
    "positive": {"bg": "#EAF7EF", "accent": POSITIVE_COLOR, "text": "#0B6B3A"},
    "neutral":  {"bg": "#EAF1FB", "accent": NEUTRAL_COLOR,  "text": PRIMARY_DARK},
    "warning":  {"bg": "#FEF3E2", "accent": WARNING_COLOR,  "text": "#8A5A0A"},
    "negative": {"bg": "#FBEAEA", "accent": NEGATIVE_COLOR, "text": "#8F1E17"},
}

PLOTLY_SEQUENCE = ["#1F4E8C", "#F59E0B", "#0F9D58", "#7C3AED", "#D93025", "#0891B2"]

st.markdown(
    f"""
    <style>
    /* ---------- page background ---------- */
    .stApp {{
        background-color: {PAGE_BG};
    }}
    div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}

    /* ---------- executive header ---------- */
    .app-header {{
        background: linear-gradient(90deg, {PRIMARY_DARK} 0%, {PRIMARY_COLOR} 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 0.6rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 2px 10px rgba(18, 48, 92, 0.25);
    }}
    .app-header-title {{
        color: #FFFFFF;
        font-size: 1.65rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.2px;
    }}
    .app-header-subtitle {{
        color: #D7E4F7;
        font-size: 0.98rem;
        margin-top: 0.15rem;
    }}

    /* ---------- data note / disclaimer ---------- */
    .disclaimer-box {{
        background-color: rgba(31, 78, 140, 0.06);
        border-left: 4px solid {PRIMARY_COLOR};
        padding: 0.65rem 1rem;
        border-radius: 0.35rem;
        font-size: 0.83rem;
        margin-bottom: 1rem;
        color: #33415C;
    }}

    /* ---------- KPI cards ---------- */
    .kpi-card {{
        background: var(--kpi-bg, #EAF1FB);
        border-left: 5px solid var(--kpi-accent, #1F4E8C);
        border-radius: 0.55rem;
        padding: 0.85rem 1rem 0.75rem 1rem;
        box-shadow: 0 1px 4px rgba(16, 24, 40, 0.08);
        margin-bottom: 0.6rem;
        min-height: 96px;
    }}
    .kpi-label {{
        font-size: 0.78rem;
        font-weight: 600;
        color: #52607A;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.25rem;
    }}
    .kpi-value {{
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--kpi-text, #12305C);
        line-height: 1.25;
    }}
    .kpi-sub {{
        font-size: 0.78rem;
        color: #6B7A93;
        margin-top: 0.2rem;
    }}

    /* ---------- section headers ---------- */
    .section-card {{
        background: {CARD_BG};
        border-radius: 0.6rem;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 1px 4px rgba(16, 24, 40, 0.06);
        border: 1px solid #E4E9F2;
        margin-bottom: 1rem;
    }}
    .section-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {PRIMARY_DARK};
        margin-bottom: 0.15rem;
    }}
    .section-caption {{
        font-size: 0.85rem;
        color: #6B7A93;
        margin-bottom: 0.6rem;
    }}
    .insight-box {{
        background: #FFF9EC;
        border-left: 4px solid {ACCENT_COLOR};
        border-radius: 0.4rem;
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
        color: #4A3B0C;
    }}
    .shared-caption {{
        font-size: 0.76rem;
        color: {PRIMARY_COLOR};
        background: #EAF1FB;
        border-radius: 0.3rem;
        padding: 0.25rem 0.55rem;
        display: inline-block;
        margin-bottom: 0.4rem;
    }}

    /* ---------- sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {PRIMARY_DARK} 0%, {PRIMARY_COLOR} 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: #F1F5FB !important;
    }}
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label {{
        color: #E7EEF9 !important;
        font-weight: 500;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.18);
    }}
    section[data-testid="stSidebar"] .stButton>button {{
        background-color: {ACCENT_COLOR};
        color: #1B1200;
        font-weight: 700;
        border: none;
        border-radius: 0.4rem;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] {{
        background-color: {ACCENT_COLOR} !important;
    }}

    /* ---------- tabs ---------- */
    div[data-baseweb="tab-list"] {{
        background-color: #E4E9F2;
        border-radius: 0.5rem;
        padding: 0.3rem;
        gap: 0.2rem;
    }}
    button[data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 0.4rem;
        padding: 0.45rem 0.8rem;
        color: {PRIMARY_DARK};
        font-weight: 600;
        font-size: 0.86rem;
    }}
    button[data-baseweb="tab"]:hover {{
        background-color: #D3DEF0;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background-color: {PRIMARY_COLOR};
        color: #FFFFFF !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] p {{
        color: #FFFFFF !important;
    }}

    /* ---------- roadmap chips ---------- */
    .roadmap-step {{
        background: {CARD_BG};
        border: 1px solid #E4E9F2;
        border-top: 4px solid {PRIMARY_COLOR};
        border-radius: 0.5rem;
        padding: 0.8rem 0.9rem;
        min-height: 150px;
    }}
    .roadmap-step-title {{
        font-weight: 700;
        color: {PRIMARY_DARK};
        margin-bottom: 0.3rem;
    }}
    .roadmap-arrow {{
        text-align: center;
        color: {PRIMARY_COLOR};
        font-size: 1.4rem;
        padding-top: 2.2rem;
    }}

    /* ---------- flow diagram boxes (methodology) ---------- */
    .flow-box {{
        background: {CARD_BG};
        border: 1.5px solid {PRIMARY_COLOR};
        border-radius: 0.5rem;
        padding: 0.55rem 0.5rem;
        text-align: center;
        font-size: 0.82rem;
        font-weight: 600;
        color: {PRIMARY_DARK};
        margin-bottom: 0.15rem;
    }}
    .flow-arrow {{
        text-align: center;
        color: {PRIMARY_COLOR};
        font-size: 1.1rem;
        margin: 0.05rem 0 0.15rem 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def synthetic_disclaimer():
    st.markdown(
        f'<div class="disclaimer-box">ℹ️ <b>Synthetic dataset:</b> {c.SYNTHETIC_DATA_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str | None = None):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        + (f'<div class="section-caption">{caption}</div>' if caption else ""),
        unsafe_allow_html=True,
    )


def kpi_card(col, label, value, sublabel=None, tone="neutral"):
    style = TONE_STYLES.get(tone, TONE_STYLES["neutral"])
    sub_html = f'<div class="kpi-sub">{sublabel}</div>' if sublabel else ""
    col.markdown(
        f"""
        <div class="kpi-card" style="--kpi-bg:{style['bg']}; --kpi-accent:{style['accent']}; --kpi-text:{style['text']};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(items):
    """items: list of dicts with keys label, value, sublabel(optional), tone(optional)"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        kpi_card(col, **item)


def insight_box(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def shared_caption(text: str):
    st.markdown(f'<div class="shared-caption">🔗 {text}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# SHARED CROSS-TAB BUSINESS ASSUMPTIONS (single source of truth)
# ----------------------------------------------------------------------
# These are global financial/business assumptions that must be identical
# wherever they are used (Scenario Simulator, Cost-Benefit & Feasibility,
# Executive Control Tower, Recommendations). They live in st.session_state
# so that changing them in ANY tab immediately updates every dependent
# calculation everywhere else in the app. Analytical/model parameters
# (sidebar sliders for weights, ABC/XYZ thresholds, clustering, etc.) and
# tab-local filters remain independent, since they legitimately differ in
# scope from these facility-wide financial assumptions.
_SHARED_DEFAULTS = {
    "shared_labour_cost": 150.0,
    "shared_orders_per_day": 1000.0,
    "shared_operating_days": 330.0,
    "shared_reslot_cost": float(c.DEFAULT_COST_ASSUMPTIONS["reslotting_cost_inr"]),
    "shared_software_cost": float(c.DEFAULT_COST_ASSUMPTIONS["software_analytics_cost_inr"]),
    "shared_training_cost": float(c.DEFAULT_COST_ASSUMPTIONS["training_cost_inr"]),
    "shared_disruption_cost": float(c.DEFAULT_COST_ASSUMPTIONS["disruption_cost_inr"]),
    "shared_relabel_cost": float(c.DEFAULT_COST_ASSUMPTIONS["relabelling_cost_inr"]),
}
for _k, _v in _SHARED_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _sync_widget_from_shared(session_key, widget_key):
    st.session_state[widget_key] = st.session_state[session_key]


def _push_shared_from_widget(session_key, widget_key):
    st.session_state[session_key] = st.session_state[widget_key]


def shared_number_input(label, session_key, widget_key, **kwargs):
    """A number_input whose value is bound to a shared session_state key,
    so the same assumption stays consistent no matter which tab it is
    edited in."""
    _sync_widget_from_shared(session_key, widget_key)
    return st.number_input(
        label, key=widget_key,
        on_change=_push_shared_from_widget, args=(session_key, widget_key),
        **kwargs,
    )


def get_shared_assumption_dicts():
    cost_assumptions = {
        "reslotting_cost_inr": st.session_state["shared_reslot_cost"],
        "software_analytics_cost_inr": st.session_state["shared_software_cost"],
        "training_cost_inr": st.session_state["shared_training_cost"],
        "disruption_cost_inr": st.session_state["shared_disruption_cost"],
        "relabelling_cost_inr": st.session_state["shared_relabel_cost"],
    }
    benefit_assumptions = {
        "labour_cost_per_hour_inr": st.session_state["shared_labour_cost"],
        "orders_per_day": st.session_state["shared_orders_per_day"],
        "operating_days_per_year": st.session_state["shared_operating_days"],
    }
    return cost_assumptions, benefit_assumptions


# ============================================================================
# CACHED DATA PIPELINE
# ============================================================================

@st.cache_data(show_spinner=False)
def cached_load_and_prepare(file_bytes: bytes | None, path_str: str):
    """Load + validate + clean the dataset. Accepts either uploaded file
    bytes (fallback) or the default repository path."""
    if file_bytes is not None:
        import io
        raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raw = c.load_raw_data(path_str)
    report = c.validate_data(raw)
    cleaned, prep_log = c.prepare_data(raw)
    return raw, report, cleaned, prep_log


@st.cache_data(show_spinner=False)
def cached_sku_profile(cleaned: pd.DataFrame, weights: dict):
    sku = c.build_sku_profile(cleaned)
    sku = c.compute_slotting_priority_score(sku, weights)
    return sku


@st.cache_data(show_spinner=False)
def cached_abc_xyz(sku: pd.DataFrame, abc_th: dict, xyz_th: dict):
    return c.classify_abc_xyz(sku, abc_th, xyz_th)


@st.cache_data(show_spinner=False)
def cached_clustering(sku: pd.DataFrame, k_min: int, k_max: int, manual_k: int | None):
    return c.run_kmeans_clustering(sku, range(k_min, k_max + 1), manual_k)


@st.cache_data(show_spinner=False)
def cached_association(cleaned: pd.DataFrame, min_support: float, top_n: int):
    return c.association_analysis(cleaned, min_support, top_n)


@st.cache_data(show_spinner=False)
def cached_warehouse(cleaned: pd.DataFrame):
    kpis = c.warehouse_baseline_kpis(cleaned)
    zone_stats = c.zone_level_activity(cleaned)
    aisle_stats = c.aisle_level_activity(cleaned)
    return kpis, zone_stats, aisle_stats


@st.cache_data(show_spinner=False)
def cached_optimization(sku_rank: pd.DataFrame, cleaned: pd.DataFrame, travel_w: float, congestion_w: float, top_k: int):
    return c.run_slotting_optimization(sku_rank, cleaned, travel_w, congestion_w, top_k)


# ============================================================================
# SIDEBAR — GLOBAL CONTROLS
# ============================================================================
st.sidebar.markdown(
    '<div style="font-size:1.25rem;font-weight:800;margin-bottom:0.1rem;">📦 Control Panel</div>'
    '<div style="font-size:0.8rem;opacity:0.85;margin-bottom:0.6rem;">AI-Driven Warehouse Slotting — BigBasket</div>',
    unsafe_allow_html=True,
)

uploaded_file = None
if not DATA_PATH.exists():
    st.sidebar.warning(
        "Default dataset not found at `data/`. Upload the supplied CSV to continue."
    )
    uploaded_file = st.sidebar.file_uploader(
        "Upload bigbasket_warehouse_slotting_2000_order_lines.csv", type=["csv"]
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**⚖️ SKU Priority Score Weights**")
st.sidebar.caption("Analytical model parameters — shape the Slotting Priority Score used across the pipeline.")
w_velocity = st.sidebar.slider("Velocity", 0.0, 1.0, c.DEFAULT_SLOTTING_WEIGHTS["velocity"], 0.05)
w_value = st.sidebar.slider("Business value", 0.0, 1.0, c.DEFAULT_SLOTTING_WEIGHTS["value"], 0.05)
w_variability = st.sidebar.slider("Demand-stability preference", 0.0, 1.0, c.DEFAULT_SLOTTING_WEIGHTS["variability_penalty"], 0.05)
w_travel = st.sidebar.slider("Current travel burden", 0.0, 1.0, c.DEFAULT_SLOTTING_WEIGHTS["travel_burden"], 0.05)
w_criticality = st.sidebar.slider("Criticality", 0.0, 1.0, c.DEFAULT_SLOTTING_WEIGHTS["criticality"], 0.05)
w_affinity = st.sidebar.slider("Co-purchase affinity", 0.0, 1.0, c.DEFAULT_SLOTTING_WEIGHTS["affinity"], 0.05)
slotting_weights = {
    "velocity": w_velocity, "value": w_value, "variability_penalty": w_variability,
    "travel_burden": w_travel, "criticality": w_criticality, "affinity": w_affinity,
}

st.sidebar.markdown("---")
st.sidebar.markdown("**🔤 ABC / XYZ Thresholds**")
abc_a = st.sidebar.slider("A-class cumulative value cutoff (%)", 40, 90, 70)
abc_b = st.sidebar.slider("B-class cumulative value cutoff (%)", int(abc_a) + 1, 99, max(90, int(abc_a) + 5))
abc_thresholds = {"A": abc_a, "B": abc_b}
xyz_x = st.sidebar.slider("X-class max demand CV", 0.1, 1.0, 0.5, 0.05)
xyz_y = st.sidebar.slider("Y-class max demand CV", xyz_x + 0.05, 2.0, 1.0, 0.05)
xyz_thresholds = {"X": xyz_x, "Y": xyz_y}

st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 Clustering**")
k_min, k_max = st.sidebar.slider("K search range", 2, 10, (2, 7))
manual_k = st.sidebar.selectbox("Manual K override", ["Auto (best silhouette)"] + list(range(2, 11)))
manual_k_val = None if manual_k == "Auto (best silhouette)" else int(manual_k)

st.sidebar.markdown("---")
st.sidebar.markdown("**🔗 Association Analysis**")
min_support = st.sidebar.slider("Minimum support", 0.001, 0.05, 0.005, 0.001, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.markdown("**🎯 Optimization Objective**")
travel_weight = st.sidebar.slider("Weight: travel distance", 0.0, 1.0, 0.7, 0.05)
congestion_weight = round(1.0 - travel_weight, 2)
st.sidebar.caption(f"Weight: congestion risk = {congestion_weight}")

st.sidebar.markdown("---")
run_optimization_btn = st.sidebar.button("🚀 Run / Refresh Optimization", use_container_width=True)


# ============================================================================
# LOAD & PREPARE DATA
# ============================================================================
file_bytes = uploaded_file.read() if uploaded_file is not None else None
path_str = str(DATA_PATH)

if file_bytes is None and not DATA_PATH.exists():
    st.markdown(
        '<div class="app-header"><div class="app-header-title">📦 AI-Driven Warehouse Slotting '
        '&amp; Order-Picking Optimization</div>'
        '<div class="app-header-subtitle">A Decision Support Framework for BigBasket</div></div>',
        unsafe_allow_html=True,
    )
    synthetic_disclaimer()
    st.error(
        "No dataset available yet. Please place "
        "`bigbasket_warehouse_slotting_2000_order_lines.csv` at "
        "`data/bigbasket_warehouse_slotting_2000_order_lines.csv`, or upload it "
        "using the sidebar uploader."
    )
    st.stop()

try:
    raw_df, val_report, cleaned_df, prep_log = cached_load_and_prepare(file_bytes, path_str)
except Exception as e:
    st.markdown(
        '<div class="app-header"><div class="app-header-title">📦 AI-Driven Warehouse Slotting '
        '&amp; Order-Picking Optimization</div>'
        '<div class="app-header-subtitle">A Decision Support Framework for BigBasket</div></div>',
        unsafe_allow_html=True,
    )
    synthetic_disclaimer()
    st.error(f"Could not load or process the dataset: {e}")
    st.stop()

if not val_report.is_usable:
    st.error(
        "The uploaded CSV is missing critical identifier columns "
        "(Order_ID / SKU_ID) and cannot be analyzed. Missing columns: "
        f"{', '.join(val_report.missing_required_columns)}"
    )
    st.stop()

REQUIRED_ANALYTICAL_COLS = {
    "Distance_From_Dispatch_m", "Picker_Travel_Distance_m", "Picking_Time_sec",
    "Current_Zone", "Current_Aisle", "Current_Rack", "Current_Bin_Capacity_Units",
}
missing_analytical = REQUIRED_ANALYTICAL_COLS - set(cleaned_df.columns)

# ---- Build core analytical tables (cached) — single source of truth ----
sku_base = cached_sku_profile(cleaned_df, slotting_weights)
sku_classified = cached_abc_xyz(sku_base, abc_thresholds, xyz_thresholds)
sku_clustered, cluster_diag = cached_clustering(sku_classified, k_min, k_max, manual_k_val)
pairs_df = cached_association(cleaned_df, min_support, 25)
affinity = c.build_affinity_scores(pairs_df, sku_clustered["SKU_ID"].tolist())
sku_clustered["Affinity_Score"] = affinity.values
sku_clustered = c.compute_slotting_priority_score(sku_clustered, slotting_weights)
sku_ranked = c.slot_relocation_candidates(sku_clustered)
kpis, zone_stats, aisle_stats = cached_warehouse(cleaned_df)

if "optimization_result" not in st.session_state or run_optimization_btn:
    with st.spinner("Solving SKU-to-location slotting optimization (MILP)..."):
        sku_opt, opt_diag = cached_optimization(sku_ranked, cleaned_df, travel_weight, congestion_weight, 60)
        sku_opt, proximity_df = c.association_aware_adjustment(sku_opt, pairs_df)
        impact = c.impact_analysis(cleaned_df, sku_opt)
        st.session_state["optimization_result"] = (sku_opt, opt_diag, proximity_df, impact)

sku_opt, opt_diag, proximity_df, impact = st.session_state["optimization_result"]

# ---- Single source-of-truth financial outputs, built from the SHARED
#      assumptions in session_state. Every tab that shows Annual Labour
#      Savings / ROI / Payback / Feasibility reads from these same three
#      objects — nothing is recalculated independently downstream. ----
_cost_assumptions, _benefit_assumptions = get_shared_assumption_dicts()
if impact.get("status") == "ok":
    cba = c.cost_benefit_analysis(impact, cleaned_df, _cost_assumptions, _benefit_assumptions)
    feas = c.feasibility_assessment(cba, opt_diag, proximity_df)
    recommendations = c.generate_recommendations(sku_opt, impact, proximity_df, zone_stats, cba)
else:
    cba = {"status": "no_optimization_result"}
    feas = None
    recommendations = []

# ============================================================================
# HEADER
# ============================================================================
st.markdown(
    '<div class="app-header">'
    '<div class="app-header-title">📦 AI-Driven Warehouse Slotting &amp; Order-Picking Optimization</div>'
    '<div class="app-header-subtitle">A Decision Support Framework for BigBasket</div>'
    '</div>',
    unsafe_allow_html=True,
)
synthetic_disclaimer()

tab_names = [
    "🏠 Executive Control Tower", "🧪 Scenario Simulator", "💰 Cost-Benefit & Feasibility",
    "🧹 Data Quality & Preparation", "📊 SKU Analytics", "🔤 ABC/XYZ Analysis",
    "🤖 AI Clustering", "🔗 Association Analysis", "🏭 Current Warehouse",
    "🎯 Slotting Optimization", "📈 Impact Analysis", "📋 Recommendations",
    "📖 Methodology & Data Dictionary",
]
tabs = st.tabs(tab_names)

TAB_EXEC, TAB_SCENARIO, TAB_CBA, TAB_DQ, TAB_SKU, TAB_ABCXYZ, TAB_CLUSTER, \
    TAB_ASSOC, TAB_WAREHOUSE, TAB_OPT, TAB_IMPACT, TAB_REC, TAB_METHOD = range(13)

# ============================================================================
# TAB — EXECUTIVE CONTROL TOWER
# ============================================================================
with tabs[TAB_EXEC]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Facility Scale", "Live figures computed from the loaded dataset and current analytical settings.")
    kpi_row([
        {"label": "Unique SKUs", "value": f"{sku_opt['SKU_ID'].nunique():,}", "tone": "neutral"},
        {"label": "Unique Orders", "value": f"{kpis['unique_orders']:,}", "tone": "neutral"},
        {"label": "Order Lines", "value": f"{kpis['total_order_lines']:,}", "tone": "neutral"},
        {"label": "Warehouse Locations Modeled", "value": f"{c.TOTAL_LOCATIONS}", "tone": "neutral"},
    ])
    st.markdown('</div>', unsafe_allow_html=True)

    if impact.get("status") == "ok":
        shared_caption(
            f"Shared assumptions in effect: ₹{st.session_state['shared_labour_cost']:.0f}/hr labour cost, "
            f"{st.session_state['shared_orders_per_day']:,.0f} orders/day, "
            f"{st.session_state['shared_operating_days']:,.0f} operating days/year — "
            f"edit these in Scenario Simulator or Cost-Benefit & Feasibility."
        )

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Current vs Optimized Layout — Operational Impact")
        kpi_row([
            {"label": "Total Picker Travel", "value": f"{impact['current_total_travel_m']:,.0f} m → {impact['optimized_total_travel_m']:,.0f} m",
             "sublabel": f"-{impact['travel_reduction_pct']:.1f}% reduction", "tone": "positive"},
            {"label": "Total Picking Time", "value": f"{impact['current_total_time_sec']/3600:,.1f} h → {impact['optimized_total_time_sec']/3600:,.1f} h",
             "sublabel": f"-{impact['time_reduction_pct']:.1f}% reduction", "tone": "positive"},
            {"label": "Est. Picking Productivity", "value": f"{impact['current_picks_per_hour']:.0f} → {impact['optimized_picks_per_hour']:.0f} picks/hr",
             "sublabel": f"+{impact['estimated_productivity_gain_pct']:.1f}% gain", "tone": "positive"},
        ])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Financial Outcome", "Calculated from the shared cost/benefit assumptions above — identical to Cost-Benefit & Feasibility.")
        roi_val = cba.get("roi_pct")
        roi_tone = "positive" if (roi_val is not None and roi_val >= 0) else ("negative" if roi_val is not None else "neutral")
        pb_val = cba.get("payback_months")
        kpi_row([
            {"label": "Annual Labour Savings (est.)", "value": f"₹{cba['annual_labour_savings_inr']:,.0f}", "tone": "positive"},
            {"label": "Year-1 ROI (est.)", "value": f"{roi_val:.0f}%" if roi_val is not None else "N/A", "tone": roi_tone},
            {"label": "Payback Period (est.)", "value": f"{pb_val:.1f} mo" if pb_val else "N/A", "tone": "neutral"},
        ])
        st.markdown('</div>', unsafe_allow_html=True)

        n_moved = len(sku_opt[sku_opt["Optimized_Distance_From_Dispatch_m"] < sku_opt["Distance_From_Dispatch_m"]])
        insight_box(
            f"Re-slotting <b>{n_moved} high-priority SKUs</b> closer to dispatch and reducing zone-level congestion "
            f"is modeled to cut total picker travel by <b>{impact['travel_reduction_pct']:.1f}%</b> and total picking "
            f"time by <b>{impact['time_reduction_pct']:.1f}%</b> at the current order volume — worth an estimated "
            f"<b>₹{cba['annual_labour_savings_inr']:,.0f}/year</b> in labour savings. Adjust cost/benefit assumptions "
            f"in the Scenario Simulator or Cost-Benefit tab to reflect your facility's actual scale."
        )

        with st.expander("🔧 Optimization Solver Status"):
            st.json(opt_diag, expanded=False)
    else:
        st.warning("Optimization has not produced a result yet. Click 'Run / Refresh Optimization' in the sidebar.")

# ============================================================================
# TAB — SCENARIO SIMULATOR
# ============================================================================
with tabs[TAB_SCENARIO]:
    st.subheader("Scenario Simulator")
    if impact.get("status") != "ok":
        st.warning("Run the optimization first (sidebar button) to use the scenario simulator.")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Scenario Assumptions")
        c1, c2, c3 = st.columns(3)
        demand_growth = c1.slider("Demand growth (%)", 0, 100, 0, 5)
        peak_multiplier = c2.slider("Peak-demand multiplier", 1.0, 3.0, 1.0, 0.1)
        with c3:
            labour_cost = shared_number_input(
                "Labour cost (INR / hour)", "shared_labour_cost", "labour_cost_scenario_widget",
                min_value=50.0, max_value=2000.0, step=10.0,
            )
            shared_caption("Shared with Cost-Benefit & Executive Control Tower")
        st.markdown('</div>', unsafe_allow_html=True)

        sim = c.run_scenario_simulation(impact, demand_growth, peak_multiplier, labour_cost)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Projected Outcome Under This Scenario")
        kpi_row([
            {"label": "Optimized Total Travel", "value": f"{sim['optimized_total_travel_m']:,.0f} m", "tone": "neutral"},
            {"label": "Picking Hours Saved", "value": f"{sim['picking_hours_saved']:,.1f} h", "tone": "positive"},
            {"label": "Est. Labour Cost Savings", "value": f"₹{sim['labour_cost_savings_inr']:,.0f}", "tone": "positive"},
        ])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("##### Preset Scenario Comparison")
        scen_table = c.run_scenario_table(impact, labour_cost)
        st.dataframe(scen_table, use_container_width=True)
        st.download_button("⬇️ Download scenario results (CSV)", c.to_csv_download(scen_table), "scenario_results.csv", "text/csv")

        fig = px.bar(scen_table, x="Scenario", y=["Current_Travel_m", "Optimized_Travel_m"], barmode="group",
                     color_discrete_sequence=[ACCENT_COLOR, PRIMARY_COLOR])
        fig.update_layout(height=400, margin=dict(t=20), yaxis_title="Total Picker Travel (m)",
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB — COST-BENEFIT & FEASIBILITY
# ============================================================================
with tabs[TAB_CBA]:
    st.subheader("Cost-Benefit Analysis")
    if impact.get("status") != "ok":
        st.warning("Run the optimization first (sidebar button) for cost-benefit analysis.")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Editable Cost Assumptions (INR)", "One-time implementation costs — shared with the Executive Control Tower.")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            reslot_cost = shared_number_input("Re-slotting / movement cost", "shared_reslot_cost", "reslot_cost_widget", step=5000.0)
        with cc2:
            software_cost = shared_number_input("Software / analytics cost", "shared_software_cost", "software_cost_widget", step=5000.0)
        with cc3:
            training_cost = shared_number_input("Training cost", "shared_training_cost", "training_cost_widget", step=2000.0)
        cc1, cc2 = st.columns(2)
        with cc1:
            disruption_cost = shared_number_input("Temporary disruption cost", "shared_disruption_cost", "disruption_cost_widget", step=2000.0)
        with cc2:
            relabel_cost = shared_number_input("Relabelling / signage cost", "shared_relabel_cost", "relabel_cost_widget", step=2000.0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Editable Benefit Assumptions")
        shared_caption("Shared assumption — changes apply across Scenario Simulator, Cost-Benefit and Executive Control Tower")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            labour_cost_cba = shared_number_input(
                "Labour cost (INR/hour)", "shared_labour_cost", "labour_cost_cba_widget", step=10.0,
            )
        with bc2:
            override_orders = shared_number_input(
                "Assumed orders/day", "shared_orders_per_day", "orders_per_day_widget",
                min_value=1.0, step=10.0,
                help="Default reflects a representative facility scale. Override with your facility's real "
                     "daily order volume for a realistic annualized estimate.",
            )
        with bc3:
            operating_days = shared_number_input(
                "Operating days / year", "shared_operating_days", "operating_days_widget", step=5.0,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # cba/feas were already computed above (single source of truth) from
        # these exact shared session_state values, so we simply reuse them.
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Results")
        roi_val = cba.get("roi_pct")
        roi_tone = "positive" if (roi_val is not None and roi_val >= 0) else ("negative" if roi_val is not None else "neutral")
        pb_val = cba.get("payback_months")
        kpi_row([
            {"label": "Total Implementation Cost", "value": f"₹{cba['total_implementation_cost_inr']:,.0f}", "tone": "neutral"},
            {"label": "Annual Labour Savings (est.)", "value": f"₹{cba['annual_labour_savings_inr']:,.0f}", "tone": "positive"},
            {"label": "Year-1 ROI (est.)", "value": f"{roi_val:.0f}%" if roi_val is not None else "N/A", "tone": roi_tone},
            {"label": "Payback Period (est.)", "value": f"{pb_val:.1f} mo" if pb_val else "N/A", "tone": "neutral"},
        ])
        st.caption(
            f"Observed orders/day in dataset: {cba['observed_orders_per_day']} | "
            f"Assumed orders/day used: {cba['assumed_orders_per_day']} | "
            f"Annual orders assumed: {cba['annual_orders_assumed']:,}"
        )
        st.info(f"ℹ️ {cba['note']}")
        st.markdown('</div>', unsafe_allow_html=True)

        cba_df = pd.DataFrame([{k: v for k, v in cba.items() if k not in ("cost_assumptions", "benefit_assumptions", "note")}])
        st.download_button("⬇️ Download cost-benefit results (CSV)", c.to_csv_download(cba_df), "cost_benefit_results.csv", "text/csv")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Feasibility Assessment")
        c1, c2, c3, c4, c5 = st.columns(5)
        dims = list(feas["dimension_scores"].items())
        for col, (dim, score) in zip([c1, c2, c3, c4, c5], dims):
            tone = "positive" if score >= 4 else ("warning" if score == 3 else "negative")
            kpi_card(col, dim, f"{score}/5", tone=tone)
        kpi_card(st, "Overall Feasibility Score", f"{feas['overall_score_out_of_5']}/5", tone="neutral")

        st.markdown("##### Feasibility Reasoning")
        for dim, reason in feas["reasons"].items():
            st.markdown(f"**{dim}** — {reason}")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB — DATA QUALITY & PREPARATION
# ============================================================================
with tabs[TAB_DQ]:
    st.subheader("Dataset Overview")
    kpi_row([
        {"label": "Rows", "value": f"{val_report.row_count:,}", "tone": "neutral"},
        {"label": "Columns", "value": f"{val_report.column_count}", "tone": "neutral"},
        {"label": "Unique Orders", "value": f"{val_report.unique_orders:,}", "tone": "neutral"},
        {"label": "Unique SKUs", "value": f"{val_report.unique_skus}", "tone": "neutral"},
    ])

    if missing_analytical:
        st.warning(f"Missing columns needed for some analyses: {', '.join(missing_analytical)}. Those modules will be limited.")

    st.markdown("#### Missing Values (raw data)")
    if val_report.missing_values:
        st.dataframe(pd.DataFrame(list(val_report.missing_values.items()), columns=["Column", "Missing_Count"]), use_container_width=True)
    else:
        st.success("No missing values detected in the raw dataset.")

    st.markdown("#### Duplicate Rows")
    st.write(f"{val_report.duplicate_rows} exact duplicate row(s) detected in raw data.")

    st.markdown("#### Numeric Anomalies")
    if val_report.numeric_anomalies:
        st.json(val_report.numeric_anomalies)
    else:
        st.success("No numeric anomalies (negative values / non-numeric entries) detected.")

    st.markdown("#### Preprocessing Actions Performed")
    for line in prep_log:
        st.write(f"- {line}")

    st.markdown("#### Cleaned Analytical Dataset (preview)")
    st.dataframe(cleaned_df.head(50), use_container_width=True)
    st.download_button("⬇️ Download cleaned dataset (CSV)", c.to_csv_download(cleaned_df), "cleaned_order_lines.csv", "text/csv")

# ============================================================================
# TAB — SKU ANALYTICS
# ============================================================================
with tabs[TAB_SKU]:
    st.subheader("SKU-Level Analytical Profile")
    st.caption("Slotting Priority Score combines business value, velocity, demand stability, current travel burden, criticality and co-purchase affinity — weights adjustable in the sidebar.")

    display_cols = [
        "SKU_ID", "Category", "Subcategory", "Storage_Type", "Criticality",
        "Total_Quantity_Demanded", "Revenue_Contribution_INR", "Demand_Share_Pct",
        "Picking_Frequency_Weekly", "Demand_CV", "Avg_Picker_Travel_m",
        "Avg_Picking_Time_sec", "Distance_From_Dispatch_m", "Package_Volume_cm3",
        "Weight_kg", "Current_Zone", "Current_Aisle", "Current_Rack",
        "Slotting_Priority_Score",
    ]
    display_cols = [col for col in display_cols if col in sku_opt.columns]

    with st.expander("🔎 Filters"):
        cat_filter = st.multiselect("Category", sorted(sku_opt["Category"].dropna().unique()))
        storage_filter = st.multiselect("Storage Type", sorted(sku_opt["Storage_Type"].dropna().unique()))

    filtered = sku_opt.copy()
    if cat_filter:
        filtered = filtered[filtered["Category"].isin(cat_filter)]
    if storage_filter:
        filtered = filtered[filtered["Storage_Type"].isin(storage_filter)]

    st.dataframe(filtered[display_cols].sort_values("Slotting_Priority_Score", ascending=False).round(2), use_container_width=True, height=420)
    st.download_button("⬇️ Download SKU analytics (CSV)", c.to_csv_download(filtered[display_cols]), "sku_analytics.csv", "text/csv")

    st.markdown("#### Priority Score Distribution")
    fig = px.histogram(sku_opt, x="Slotting_Priority_Score", nbins=25, color_discrete_sequence=[PRIMARY_COLOR])
    fig.update_layout(height=350, margin=dict(t=20), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB — ABC / XYZ ANALYSIS
# ============================================================================
with tabs[TAB_ABCXYZ]:
    st.subheader("ABC / XYZ Classification")
    st.caption(
        f"ABC thresholds (cumulative revenue contribution): A ≤ {abc_thresholds['A']}%, "
        f"B ≤ {abc_thresholds['B']}%, C = remainder. "
        f"XYZ thresholds (demand coefficient of variation): X ≤ {xyz_thresholds['X']}, "
        f"Y ≤ {xyz_thresholds['Y']}, Z = above."
    )

    summary = c.abc_xyz_summary(sku_classified)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### ABC Distribution")
        abc_df = pd.DataFrame(list(summary["abc_distribution"].items()), columns=["Class", "SKU_Count"]).sort_values("Class")
        fig = px.bar(abc_df, x="Class", y="SKU_Count", color="Class", color_discrete_sequence=PLOTLY_SEQUENCE)
        fig.update_layout(height=320, margin=dict(t=20), showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("##### XYZ Distribution")
        xyz_df = pd.DataFrame(list(summary["xyz_distribution"].items()), columns=["Class", "SKU_Count"]).sort_values("Class")
        fig = px.bar(xyz_df, x="Class", y="SKU_Count", color="Class", color_discrete_sequence=PLOTLY_SEQUENCE)
        fig.update_layout(height=320, margin=dict(t=20), showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### ABC × XYZ Matrix (SKU count)")
    fig = px.imshow(
        summary["matrix"], text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="XYZ Class", y="ABC Class", color="SKU Count"),
    )
    fig.update_layout(height=350, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Segment Interpretation")
    for seg, text in c.ABC_XYZ_INTERPRETATION.items():
        if seg in sku_classified["Computed_ABC_XYZ_Class"].values:
            st.markdown(f"**{seg}** — {text}")

    st.markdown("##### SKU-Level Classification Table")
    show_cols = ["SKU_ID", "Category", "Revenue_Contribution_INR", "Cumulative_Value_Pct",
                 "Demand_CV", "Computed_ABC_Class", "Computed_XYZ_Class", "Computed_ABC_XYZ_Class"]
    show_cols = [col for col in show_cols if col in sku_classified.columns]
    st.dataframe(sku_classified[show_cols].round(2), use_container_width=True, height=380)
    st.download_button("⬇️ Download ABC/XYZ results (CSV)", c.to_csv_download(sku_classified[show_cols]), "abc_xyz_results.csv", "text/csv")

# ============================================================================
# TAB — AI CLUSTERING
# ============================================================================
with tabs[TAB_CLUSTER]:
    st.subheader("AI-Based SKU Clustering (K-Means)")
    kpi_row([
        {"label": "Selected K", "value": f"{cluster_diag['best_k']}", "tone": "neutral"},
        {"label": "Silhouette Score", "value": f"{cluster_diag['best_silhouette']:.3f}" if not np.isnan(cluster_diag["best_silhouette"]) else "N/A", "tone": "neutral"},
    ])

    sil_df = pd.DataFrame(list(cluster_diag["silhouette_scores"].items()), columns=["K", "Silhouette_Score"]).sort_values("K")
    fig = px.line(sil_df, x="K", y="Silhouette_Score", markers=True, title="Silhouette Score by K",
                  color_discrete_sequence=[PRIMARY_COLOR])
    fig.add_vline(x=cluster_diag["best_k"], line_dash="dash", line_color=ACCENT_COLOR)
    fig.update_layout(height=320, margin=dict(t=40), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Features used: {', '.join(cluster_diag['features_used'])}")

    st.markdown("##### Cluster Visualization (Velocity vs Travel Burden)")
    if {"Picking_Frequency_Weekly", "Distance_From_Dispatch_m"}.issubset(sku_clustered.columns):
        fig = px.scatter(
            sku_clustered, x="Distance_From_Dispatch_m", y="Picking_Frequency_Weekly",
            color=sku_clustered["Cluster"].astype(str), hover_data=["SKU_ID", "Category"],
            labels={"color": "Cluster"}, color_discrete_sequence=PLOTLY_SEQUENCE,
        )
        fig.update_layout(height=420, margin=dict(t=20), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Managerial Cluster Profiles")
    st.dataframe(cluster_diag["cluster_profile"], use_container_width=True)

    st.markdown("##### SKU-to-Cluster Assignment")
    cl_cols = ["SKU_ID", "Category", "Cluster"] + [f for f in cluster_diag["features_used"]]
    st.dataframe(sku_clustered[cl_cols].round(2), use_container_width=True, height=350)
    st.download_button("⬇️ Download clustering results (CSV)", c.to_csv_download(sku_clustered[cl_cols]), "clustering_results.csv", "text/csv")

# ============================================================================
# TAB — ASSOCIATION ANALYSIS
# ============================================================================
with tabs[TAB_ASSOC]:
    st.subheader("Co-Purchase / Association Analysis")
    st.caption("Order_ID-level market-basket analysis. Support, confidence and lift computed from actual order co-occurrence — no fabricated relationships.")

    if pairs_df.empty:
        st.warning("No SKU pairs met the minimum support threshold. Lower the threshold in the sidebar.")
    else:
        with st.expander("🔎 Filters"):
            min_lift = st.slider("Minimum Lift", 0.0, float(max(pairs_df["Lift"].fillna(0).max(), 1.0)), 0.0, 0.1)
        f_pairs = pairs_df[pairs_df["Lift"].fillna(0) >= min_lift]

        st.markdown("##### Top Co-Purchased SKU Pairs")
        st.dataframe(f_pairs, use_container_width=True, height=400)
        st.download_button("⬇️ Download association results (CSV)", c.to_csv_download(f_pairs), "association_results.csv", "text/csv")

        st.markdown("##### Top Pairs by Lift")
        top_chart = f_pairs.head(15).copy()
        top_chart["Pair"] = top_chart["SKU_A"] + " + " + top_chart["SKU_B"]
        fig = px.bar(top_chart.sort_values("Lift"), x="Lift", y="Pair", orientation="h", color="Support",
                     color_continuous_scale="Blues")
        fig.update_layout(height=450, margin=dict(t=20), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Category-Level Relationships")
        if {"Category_A", "Category_B"}.issubset(f_pairs.columns):
            cat_pairs = f_pairs.groupby(["Category_A", "Category_B"]).size().reset_index(name="Pair_Count")
            st.dataframe(cat_pairs.sort_values("Pair_Count", ascending=False), use_container_width=True)

# ============================================================================
# TAB — CURRENT WAREHOUSE
# ============================================================================
with tabs[TAB_WAREHOUSE]:
    st.subheader("Current (Baseline) Warehouse Layout Analysis")
    kpi_row([
        {"label": "Total Picker Travel", "value": f"{kpis['total_picker_travel_m']:,.0f} m", "tone": "neutral"},
        {"label": "Total Picking Time", "value": f"{kpis['total_picking_time_sec']/3600:,.1f} h", "tone": "neutral"},
        {"label": "Est. Picks / Hour", "value": f"{kpis['estimated_picks_per_hour']:.0f}", "tone": "neutral"},
    ])
    kpi_row([
        {"label": "Avg Travel / Order Line", "value": f"{kpis['avg_travel_per_line_m']:.1f} m", "tone": "neutral"},
        {"label": "Avg Travel / Order", "value": f"{kpis['avg_travel_per_order_m']:.1f} m", "tone": "neutral"},
        {"label": "Avg Picking Time / Order", "value": f"{kpis['avg_picking_time_per_order_sec']:.1f} s", "tone": "neutral"},
    ])

    st.markdown("##### Warehouse Heatmap")
    st.caption(
        "'Congestion Risk Proxy' is an analytical proxy (0-100) combining order-line density, "
        "active-SKU concentration, and picking-frequency concentration within a zone — it does not "
        "represent a physical congestion measurement."
    )
    metric_choice = st.radio("Heatmap metric", ["Pick Frequency", "Picking Time", "Travel Distance", "Congestion Risk"], horizontal=True)
    hm = c.heatmap_matrix(cleaned_df, metric_choice)
    fig = px.imshow(hm, text_auto=".1f", color_continuous_scale="OrRd", aspect="auto",
                     labels=dict(x="Aisle", y="Zone", color=metric_choice))
    fig.update_layout(height=450, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Zone-Level Performance")
    st.dataframe(zone_stats.round(2), use_container_width=True)

    st.markdown("##### Aisle-Level Performance")
    st.dataframe(aisle_stats.round(2), use_container_width=True, height=300)

# ============================================================================
# TAB — SLOTTING OPTIMIZATION
# ============================================================================
with tabs[TAB_OPT]:
    st.subheader("Slot Priority & Slotting Optimization")

    st.markdown("##### Relocation Priority Ranking")
    pr_cols = ["SKU_ID", "Computed_ABC_XYZ_Class", "Relocation_Priority", "Slotting_Priority_Score", "Relocation_Reason"]
    pr_cols = [col for col in pr_cols if col in sku_ranked.columns]
    priority_filter = st.multiselect("Filter by priority", ["High Priority", "Medium Priority", "Low Priority"], default=["High Priority"])
    pr_view = sku_ranked[sku_ranked["Relocation_Priority"].isin(priority_filter)] if priority_filter else sku_ranked
    st.dataframe(pr_view[pr_cols].round(2), use_container_width=True, height=320)

    st.markdown("---")
    st.markdown("##### Optimization Model")
    st.markdown(
        """
        - **Decision:** assign each SKU to exactly one eligible warehouse location (zone/aisle/rack).
        - **Objective:** minimize a priority-weighted combination of travel distance and zone congestion risk.
        - **Constraints:** one location per SKU, one SKU per location, bin-capacity feasibility, storage-type
          zone compatibility.
        - **Method:** Mixed-Integer Linear Program solved with `scipy.optimize.milp` (HiGHS branch-and-bound
          solver), using a documented candidate-location filter to keep the model tractable on Streamlit Cloud.
        """
    )
    with st.expander("🔧 Solver Diagnostics"):
        st.json({k: v for k, v in opt_diag.items()}, expanded=False)

    if opt_diag.get("status") == "optimal":
        relocated = sku_opt[sku_opt["Optimized_Location_ID"] != (sku_opt["Current_Zone"] + "-" + sku_opt["Current_Aisle"] + "-" + sku_opt["Current_Rack"])]
        kpi_row([{"label": "SKUs Recommended for Relocation", "value": f"{len(relocated)}", "tone": "positive"}])

        st.markdown("##### Before / After Location Table")
        comp_cols = ["SKU_ID", "Current_Zone", "Current_Aisle", "Current_Rack", "Distance_From_Dispatch_m",
                     "Optimized_Zone", "Optimized_Aisle", "Optimized_Rack", "Optimized_Distance_From_Dispatch_m"]
        comp_cols = [col for col in comp_cols if col in sku_opt.columns]
        st.dataframe(sku_opt[comp_cols].round(2), use_container_width=True, height=380)
        st.download_button("⬇️ Download optimized slotting assignment (CSV)", c.to_csv_download(sku_opt[comp_cols]), "optimized_slotting.csv", "text/csv")

        st.markdown("##### Top Recommended Relocations (by priority score)")
        top_reloc = relocated.sort_values("Slotting_Priority_Score", ascending=False).head(10)
        st.dataframe(top_reloc[comp_cols + ["Slotting_Priority_Score"]].round(2), use_container_width=True)

        st.markdown("##### Association-Aware Slotting: Proximity Outcomes")
        st.caption("Whether top co-purchased SKU pairs became physically closer after optimization.")
        if not proximity_df.empty:
            st.dataframe(proximity_df, use_container_width=True, height=300)
            improved_pct = 100 * proximity_df["Proximity_Improved"].mean()
            kpi_row([{"label": "Pairs with Improved Proximity", "value": f"{improved_pct:.0f}%", "tone": "positive"}])
        else:
            st.info("No association pairs available to evaluate (try lowering the minimum support threshold).")
    else:
        st.error(f"Optimization did not converge to a feasible solution: {opt_diag.get('solver_message')}")

# ============================================================================
# TAB — IMPACT ANALYSIS
# ============================================================================
with tabs[TAB_IMPACT]:
    st.subheader("Current vs Optimized Impact Analysis")
    if impact.get("status") != "ok":
        st.warning("Run the optimization first (sidebar button) to see impact analysis.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(data=[go.Bar(
                x=["Current", "Optimized"],
                y=[impact["current_total_travel_m"], impact["optimized_total_travel_m"]],
                marker_color=[ACCENT_COLOR, PRIMARY_COLOR],
                text=[f"{impact['current_total_travel_m']:,.0f} m", f"{impact['optimized_total_travel_m']:,.0f} m"],
                textposition="auto",
            )])
            fig.update_layout(title="Total Picker Travel Distance (m)", height=380, margin=dict(t=40),
                               plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(data=[go.Bar(
                x=["Current", "Optimized"],
                y=[impact["current_total_time_sec"] / 3600, impact["optimized_total_time_sec"] / 3600],
                marker_color=[ACCENT_COLOR, PRIMARY_COLOR],
                text=[f"{impact['current_total_time_sec']/3600:,.1f} h", f"{impact['optimized_total_time_sec']/3600:,.1f} h"],
                textposition="auto",
            )])
            fig.update_layout(title="Total Picking Time (hours)", height=380, margin=dict(t=40),
                               plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Productivity Implications")
        kpi_row([
            {"label": "Picks / Hour (current → optimized)", "value": f"{impact['current_picks_per_hour']:.0f} → {impact['optimized_picks_per_hour']:.0f}", "tone": "positive"},
            {"label": "Productivity Gain (est.)", "value": f"{impact['estimated_productivity_gain_pct']:.1f}%", "tone": "positive"},
            {"label": "Picking-Stage Fulfillment Time Reduction (est.)", "value": f"{impact['fulfillment_picking_stage_reduction_pct']:.1f}%", "tone": "positive"},
        ])
        st.caption(
            "The fulfillment-time figure reflects only the picking stage's contribution to order lead time "
            "(the stage this model directly optimizes) — not total dock-to-door fulfillment time."
        )

        st.markdown("##### Distribution of Travel Distance: Current vs Optimized (order-line level)")
        lines = impact["line_level_detail"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=lines["Picker_Travel_Distance_m"], name="Current", opacity=0.6, marker_color=ACCENT_COLOR))
        fig.add_trace(go.Histogram(x=lines["Optimized_Picker_Travel_m"], name="Optimized", opacity=0.6, marker_color=PRIMARY_COLOR))
        fig.update_layout(barmode="overlay", height=380, margin=dict(t=20), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Before / After Summary Table")
        summary_tbl = pd.DataFrame([
            {"Metric": "Total Picker Travel (m)", "Current": round(impact["current_total_travel_m"]), "Optimized": round(impact["optimized_total_travel_m"]), "Reduction %": round(impact["travel_reduction_pct"], 1)},
            {"Metric": "Total Picking Time (hrs)", "Current": round(impact["current_total_time_sec"] / 3600, 1), "Optimized": round(impact["optimized_total_time_sec"] / 3600, 1), "Reduction %": round(impact["time_reduction_pct"], 1)},
            {"Metric": "Avg Picking Time / Order (s)", "Current": round(impact["current_avg_time_per_order_sec"], 1), "Optimized": round(impact["optimized_avg_time_per_order_sec"], 1), "Reduction %": round(impact["fulfillment_picking_stage_reduction_pct"], 1)},
            {"Metric": "Picks / Hour", "Current": round(impact["current_picks_per_hour"]), "Optimized": round(impact["optimized_picks_per_hour"]), "Reduction %": -round(impact["estimated_productivity_gain_pct"], 1)},
        ])
        st.dataframe(summary_tbl, use_container_width=True)
        st.download_button("⬇️ Download before/after KPI table (CSV)", c.to_csv_download(summary_tbl), "before_after_kpis.csv", "text/csv")

# ============================================================================
# TAB — MANAGERIAL RECOMMENDATIONS  (visual-first redesign)
# ============================================================================
with tabs[TAB_REC]:
    st.subheader("Managerial Recommendations")
    if impact.get("status") != "ok":
        st.warning("Run the optimization first (sidebar button) to generate recommendations.")
    else:
        # ---- Recommendation priority (impact-ranked) overview cards ----
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Recommendation Priority Overview", "Ranked by expected operational/financial impact of this run.")
        impact_scores = []
        for rec in recommendations:
            text_blob = (rec.get("expected_impact", "") + " " + rec.get("evidence", "")).lower()
            score = 1
            if "%" in text_blob:
                import re as _re
                nums = [float(x) for x in _re.findall(r"(\d+(?:\.\d+)?)%", text_blob)]
                score = max(nums) if nums else 1
            impact_scores.append(score)
        rank_df = pd.DataFrame({
            "Recommendation": [r["title"] for r in recommendations],
            "Impact_Score": impact_scores,
        }).sort_values("Impact_Score", ascending=True)
        fig = px.bar(rank_df, x="Impact_Score", y="Recommendation", orientation="h",
                     color="Impact_Score", color_continuous_scale="Blues",
                     labels={"Impact_Score": "Relative Impact Indicator"})
        fig.update_layout(height=280 + 20 * len(recommendations), margin=dict(t=20), showlegend=False,
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Top relocation opportunities ----
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Top Relocation Opportunities", "High-priority SKUs recommended for physical re-slotting.")
        if "Optimized_Distance_From_Dispatch_m" in sku_opt.columns:
            top_moves = sku_opt[sku_opt["Optimized_Distance_From_Dispatch_m"] < sku_opt["Distance_From_Dispatch_m"]] \
                .sort_values("Slotting_Priority_Score", ascending=False).head(10)
            if not top_moves.empty:
                top_moves = top_moves.copy()
                top_moves["Distance_Saved_m"] = top_moves["Distance_From_Dispatch_m"] - top_moves["Optimized_Distance_From_Dispatch_m"]
                fig = px.bar(top_moves.sort_values("Distance_Saved_m"), x="Distance_Saved_m", y="SKU_ID",
                             orientation="h", color="Slotting_Priority_Score", color_continuous_scale="Blues",
                             labels={"Distance_Saved_m": "Dispatch-Distance Reduction (m)"})
                fig.update_layout(height=380, margin=dict(t=20), plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No relocation opportunities identified in this run.")
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Current vs optimized impact recap ----
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Current vs Optimized Impact")
        kpi_row([
            {"label": "Travel Reduction", "value": f"{impact['travel_reduction_pct']:.1f}%", "tone": "positive"},
            {"label": "Picking-Time Reduction", "value": f"{impact['time_reduction_pct']:.1f}%", "tone": "positive"},
            {"label": "Productivity Gain", "value": f"{impact['estimated_productivity_gain_pct']:.1f}%", "tone": "positive"},
        ])
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Co-purchase recommendation ----
        if proximity_df is not None and not proximity_df.empty:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            section_header("Co-Purchase Recommendation", "Strongest co-purchase relationships and proximity outcomes after optimization.")
            improved_pct = 100 * proximity_df["Proximity_Improved"].mean()
            kpi_row([{"label": "Top Pairs with Improved Proximity", "value": f"{improved_pct:.0f}%", "tone": "positive"}])
            st.dataframe(proximity_df.head(10), use_container_width=True, height=260)
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- Congestion recommendation ----
        if zone_stats is not None and not zone_stats.empty:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            section_header("Congestion Recommendation", "Highest-risk zones by Congestion Risk Proxy (0-100).")
            top_zones = zone_stats.sort_values("Congestion_Risk_Proxy", ascending=False).head(6)
            fig = px.bar(top_zones, x="Current_Zone", y="Congestion_Risk_Proxy", color="Congestion_Risk_Proxy",
                         color_continuous_scale="OrRd", labels={"Congestion_Risk_Proxy": "Congestion Risk Proxy (0-100)"})
            fig.update_layout(height=320, margin=dict(t=20), showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- Implementation roadmap ----
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Implementation Roadmap")
        roadmap = [
            ("Immediate", "Re-slot the highest-priority SKUs identified by the optimizer; update WMS location records and bin labels."),
            ("Short Term", "Co-locate high-lift co-purchase pairs where capacity allows; retrain pickers on new pick paths."),
            ("Medium Term", "Rebalance high-congestion zones; monitor Congestion Risk Proxy after each re-slotting cycle."),
            ("Continuous Review", "Re-run ABC/XYZ, clustering and slotting optimization quarterly (or after seasonal shifts)."),
        ]
        cols = st.columns(len(roadmap))
        for i, (col, (stage, desc)) in enumerate(zip(cols, roadmap)):
            with col:
                st.markdown(
                    f'<div class="roadmap-step"><div class="roadmap-step-title">{stage}</div>'
                    f'<div style="font-size:0.85rem;color:#33415C;">{desc}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Detailed recommendation cards (concise, in expanders) ----
        st.markdown("#### Full Recommendation Detail")
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec['title']}"):
                st.markdown(f"**Issue:** {rec['issue']}")
                st.markdown(f"**Evidence:** {rec['evidence']}")
                st.markdown(f"**Recommendation:** {rec['recommendation']}")
                st.markdown(f"**Expected Impact:** {rec['expected_impact']}")
                st.markdown(f"**Implementation Consideration:** {rec['implementation_consideration']}")

        rec_df = pd.DataFrame(recommendations)
        st.download_button("⬇️ Download recommendations (CSV)", c.to_csv_download(rec_df), "recommendations.csv", "text/csv")

# ============================================================================
# TAB — METHODOLOGY / DATA DICTIONARY  (visual-first redesign)
# ============================================================================
with tabs[TAB_METHOD]:
    st.subheader("Methodology & Data Dictionary")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Data Source")
    synthetic_disclaimer()
    st.markdown(
        "The dataset contains order-line-level records with order, SKU, and warehouse-placement "
        "attributes, plus derived analytical fields (ABC/XYZ class, observed demand/frequency)."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Analytical Pipeline — Methodology Flow")
    flow_steps = [
        "DATA", "Data Validation & Preparation", "SKU Profiling", "ABC / XYZ",
        "AI Clustering", "Association Analysis", "Warehouse Diagnosis",
        "Slotting Priority", "MILP Optimization", "Current vs Optimized Impact",
        "Scenario Simulation", "Cost-Benefit & Feasibility", "Managerial Recommendations",
    ]
    for i, step in enumerate(flow_steps):
        st.markdown(f'<div class="flow-box">{step}</div>', unsafe_allow_html=True)
        if i < len(flow_steps) - 1:
            st.markdown('<div class="flow-arrow">↓</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Dataset Overview")
    kpi_row([
        {"label": "Orders", "value": f"{val_report.unique_orders:,}", "tone": "neutral"},
        {"label": "Order Lines", "value": f"{val_report.row_count:,}", "tone": "neutral"},
        {"label": "SKUs", "value": f"{val_report.unique_skus:,}", "tone": "neutral"},
    ])
    kpi_row([
        {"label": "Warehouse Locations", "value": f"{c.TOTAL_LOCATIONS}", "tone": "neutral"},
        {"label": "Features / Columns", "value": f"{val_report.column_count}", "tone": "neutral"},
        {"label": "Zones × Aisles × Racks", "value": f"{len(c.ZONES)} × {len(c.AISLES)} × {len(c.RACKS)}", "tone": "neutral"},
    ])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Analytical Methods")
    method_cards = [
        ("ABC / XYZ", "SKUs classified by cumulative revenue contribution (ABC) and demand coefficient of variation (XYZ), against configurable thresholds."),
        ("K-Means Clustering", "Standardized features clustered with K auto-selected via best silhouette score across a configurable range."),
        ("Association Analysis", "Order-level market-basket analysis: support, confidence and lift computed from actual co-occurrence."),
        ("MILP Optimization", "scipy.optimize.milp (HiGHS) assigns each SKU to one eligible location, minimizing weighted travel + congestion."),
        ("Scenario Analysis", "Linear scaling of travel/time totals under demand-growth and peak-multiplier assumptions."),
    ]
    cols = st.columns(len(method_cards))
    for col, (title, desc) in zip(cols, method_cards):
        with col:
            st.markdown(
                f'<div class="roadmap-step"><div class="roadmap-step-title">{title}</div>'
                f'<div style="font-size:0.8rem;color:#33415C;">{desc}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Optimization Logic")
    c1, c2, c3, c4, c5 = st.columns([2, 0.3, 2, 0.3, 2])
    with c1:
        st.markdown('<div class="flow-box" style="min-height:70px;">Objective<br><span style="font-weight:400;font-size:0.75rem;">Minimize weighted travel + congestion</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="text-align:center;font-size:1.3rem;padding-top:1rem;">+</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="flow-box" style="min-height:70px;">Constraints<br><span style="font-weight:400;font-size:0.75rem;">Capacity + zone compatibility</span></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div style="text-align:center;font-size:1.3rem;padding-top:1rem;">=</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="flow-box" style="min-height:70px;">Optimized SKU Location Assignment</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📜 Detailed Methodology Text (ABC/XYZ, Clustering, Association, Optimization)"):
        st.markdown(
            """
            **ABC/XYZ:** SKUs sorted by revenue contribution (descending), cumulative % computed, then classified
            against user-configurable thresholds (default A ≤ 70%, B ≤ 90%, C = remainder). XYZ classified by
            demand coefficient of variation against user-configurable thresholds (default X ≤ 0.5, Y ≤ 1.0, Z = above).

            **Clustering:** K-Means on standardized features (picking frequency, demand CV, average travel, package
            volume, weight, picking time, distance from dispatch). K selected automatically via best silhouette
            score across a configurable range, with manual override available.

            **Association Analysis:** Market-basket analysis at the Order_ID level. For SKU pair (A, B):
            support = orders containing both / total orders; confidence(A→B) = orders containing both / orders
            containing A; lift = support(A,B) / (support(A) × support(B)). Lift > 1 indicates a positive purchase
            association.

            **Optimization — decision variables:** x[i,j] ∈ {0,1} — SKU i assigned to candidate location j.
            **Objective:** minimize Σ x[i,j] × (travel_weight × PriorityScore[i] × Distance[j] +
            congestion_weight × PriorityScore[i] × BaselineZoneCongestion[zone(j)]).
            **Constraints:** each SKU assigned exactly one location; each location holds at most one SKU;
            eligible locations filtered by bin-capacity feasibility and storage-type zone compatibility.
            **Method:** scipy.optimize.milp (HiGHS branch-and-bound MILP solver). Because a full 120-SKU ×
            240-location assignment is computationally larger than needed, a documented candidate-location
            filter keeps each SKU's choice set to its nearest eligible locations (dynamically widened per
            eligibility group to guarantee solver feasibility) — a standard, defensible simplification for
            deployment on limited compute (Streamlit Cloud).

            **Association-aware adjustment:** rather than embedding pairwise co-purchase terms directly into the
            MILP (which would add combinatorial complexity), the top co-purchased SKU pairs are evaluated
            *after* the primary optimization to measure how many pairs became physically closer — a lightweight,
            transparent way of showing association's influence on the result.

            **Solver limitations:** the model uses linear (not queueing/simulation) travel and congestion
            approximations, a static congestion baseline (not re-computed with the new assignment), and treats
            each SKU as requiring exactly one storage location.
            """
        )

    with st.expander("📋 Assumptions, Limitations & Responsible AI"):
        st.markdown(
            """
            #### Assumptions
            - Distance-from-dispatch is deterministic by (zone, aisle, rack), estimated from the supplied data.
            - Bin capacity is deterministic by rack, estimated from the supplied data.
            - Picker travel distance and picking time are projected for new locations using simple linear models
              fitted on the *supplied* order-line data (Picker_Travel_Distance_m and Picking_Time_sec as a
              function of Distance_From_Dispatch_m).
            - Storage-type zone compatibility is inferred from which zones currently host each non-Ambient
              storage type in the baseline data (a modelling simplification, not a stated warehouse rule).
            - Financial estimates use user-editable assumptions and the dataset's short observation window;
              override "Assumed orders/day" in the Cost-Benefit tab to reflect real facility scale.

            #### Limitations
            - Synthetic/proxy dataset — absolute figures are illustrative, not actual BigBasket performance.
            - No live re-optimization of congestion after reassignment (single-pass model).
            - Clustering and priority-score weighting involve subjective analytical choices, made transparent
              and adjustable in the sidebar.
            - Co-purchase influence on slotting is evaluated post-hoc rather than jointly optimized.

            #### Responsible AI
            - **Synthetic data limitations:** results should be validated against real WMS data before
              operational use.
            - **Model assumptions:** linear travel/time projection, static congestion baseline, and configurable
              priority weights are simplifications appropriate for decision support, not guarantees.
            - **Clustering subjectivity:** K selection and feature choice affect cluster composition; silhouette
              score guides but does not guarantee the "correct" K.
            - **Feature-weighting bias:** the Slotting Priority Score's weights are user-adjustable specifically
              so no single fixed weighting is presented as objectively correct.
            - **Managerial validation required:** all recommendations should be reviewed by warehouse operations
              management before physical re-slotting.
            - **Data privacy:** a real deployment would require access controls on live demand and location data
              consistent with company data-privacy policy.
            """
        )

    st.markdown("#### Data Dictionary (key fields)")
    dict_df = pd.DataFrame({
        "Field": ["Order_ID", "SKU_ID", "Picking_Frequency_Weekly", "Demand_CV", "Distance_From_Dispatch_m",
                  "Picker_Travel_Distance_m", "Picking_Time_sec", "Current_Bin_Capacity_Units", "ABC_XYZ_Class"],
        "Description": [
            "Unique customer order identifier", "Unique SKU identifier",
            "Number of times the SKU is picked per week", "Coefficient of variation of demand (variability)",
            "Baseline distance from the dispatch point to the SKU's current location (m)",
            "Actual picker travel distance recorded for the order line (m)",
            "Actual picking time recorded for the order line (seconds)",
            "Storage capacity of the SKU's current bin/location (units)",
            "Combined ABC (value) and XYZ (variability) classification supplied with the dataset",
        ],
    })
    st.dataframe(dict_df, use_container_width=True)
