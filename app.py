"""
app.py
======
AI-Driven Warehouse Slotting and Order-Picking Optimization
A Decision Support Framework for BigBasket — Streamlit Application

MBA Working-with-AI (WAI) project, Supply Chain Management, IIM Ranchi.

This file builds the complete interactive dashboard. All analytical logic
lives in core.py; this file focuses on layout, state, and visualization.

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

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="BigBasket Warehouse Slotting Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "bigbasket_warehouse_slotting_2000_order_lines.csv"

PRIMARY_COLOR = "#1F6FEB"
ACCENT_COLOR = "#F59E0B"
NEUTRAL_BG = "#0E1117"

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.55rem; }
    .disclaimer-box {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #F59E0B;
        padding: 0.75rem 1rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def synthetic_disclaimer():
    st.markdown(
        f'<div class="disclaimer-box">⚠️ <b>Academic Disclaimer:</b> {c.SYNTHETIC_DATA_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# CACHED DATA PIPELINE
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# SIDEBAR — GLOBAL CONTROLS
# ----------------------------------------------------------------------
st.sidebar.title("📦 Control Panel")
st.sidebar.caption("AI-Driven Warehouse Slotting — BigBasket WAI Project")

uploaded_file = None
if not DATA_PATH.exists():
    st.sidebar.warning(
        "Default dataset not found at `data/`. Upload the supplied CSV to continue."
    )
    uploaded_file = st.sidebar.file_uploader(
        "Upload bigbasket_warehouse_slotting_2000_order_lines.csv", type=["csv"]
    )

st.sidebar.markdown("---")
st.sidebar.subheader("SKU Priority Score Weights")
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
st.sidebar.subheader("ABC / XYZ Thresholds")
abc_a = st.sidebar.slider("A-class cumulative value cutoff (%)", 40, 90, 70)
abc_b = st.sidebar.slider("B-class cumulative value cutoff (%)", int(abc_a) + 1, 99, max(90, int(abc_a) + 5))
abc_thresholds = {"A": abc_a, "B": abc_b}
xyz_x = st.sidebar.slider("X-class max demand CV", 0.1, 1.0, 0.5, 0.05)
xyz_y = st.sidebar.slider("Y-class max demand CV", xyz_x + 0.05, 2.0, 1.0, 0.05)
xyz_thresholds = {"X": xyz_x, "Y": xyz_y}

st.sidebar.markdown("---")
st.sidebar.subheader("Clustering")
k_min, k_max = st.sidebar.slider("K search range", 2, 10, (2, 7))
manual_k = st.sidebar.selectbox("Manual K override", ["Auto (best silhouette)"] + list(range(2, 11)))
manual_k_val = None if manual_k == "Auto (best silhouette)" else int(manual_k)

st.sidebar.markdown("---")
st.sidebar.subheader("Association Analysis")
min_support = st.sidebar.slider("Minimum support", 0.001, 0.05, 0.005, 0.001, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.subheader("Optimization Objective")
travel_weight = st.sidebar.slider("Weight: travel distance", 0.0, 1.0, 0.7, 0.05)
congestion_weight = round(1.0 - travel_weight, 2)
st.sidebar.caption(f"Weight: congestion risk = {congestion_weight}")

st.sidebar.markdown("---")
run_optimization_btn = st.sidebar.button("🚀 Run / Refresh Optimization", use_container_width=True)


# ----------------------------------------------------------------------
# LOAD & PREPARE DATA
# ----------------------------------------------------------------------
file_bytes = uploaded_file.read() if uploaded_file is not None else None
path_str = str(DATA_PATH)

if file_bytes is None and not DATA_PATH.exists():
    st.title("📦 BigBasket Warehouse Slotting & Order-Picking Optimization")
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
    st.title("📦 BigBasket Warehouse Slotting & Order-Picking Optimization")
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

# ---- Build core analytical tables (cached) ----
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

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.title("📦 AI-Driven Warehouse Slotting & Order-Picking Optimization")
st.caption("A Decision Support Framework for BigBasket — MBA WAI Project, Supply Chain Management, IIM Ranchi")
synthetic_disclaimer()

tab_names = [
    "🏠 Executive Control Tower", "🧹 Data Quality & Preparation", "📊 SKU Analytics",
    "🔤 ABC/XYZ Analysis", "🤖 AI Clustering", "🔗 Association Analysis",
    "🏭 Current Warehouse", "🎯 Slotting Optimization", "📈 Impact Analysis",
    "🧪 Scenario Simulator", "💰 Cost-Benefit & Feasibility", "📋 Recommendations",
    "📖 Methodology & Data Dictionary",
]
tabs = st.tabs(tab_names)

# ============================================================================
# TAB 1 — EXECUTIVE CONTROL TOWER
# ============================================================================
with tabs[0]:
    st.subheader("Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unique SKUs", f"{sku_opt['SKU_ID'].nunique():,}")
    c2.metric("Unique Orders", f"{kpis['unique_orders']:,}")
    c3.metric("Order Lines", f"{kpis['total_order_lines']:,}")
    c4.metric("Warehouse Locations Modeled", f"{c.TOTAL_LOCATIONS}")

    st.markdown("#### Current vs Optimized Layout")
    if impact.get("status") == "ok":
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Picker Travel", f"{impact['current_total_travel_m']:,.0f} m → {impact['optimized_total_travel_m']:,.0f} m",
                       delta=f"-{impact['travel_reduction_pct']:.1f}%")
        with c2:
            st.metric("Total Picking Time", f"{impact['current_total_time_sec']/3600:,.1f} h → {impact['optimized_total_time_sec']/3600:,.1f} h",
                       delta=f"-{impact['time_reduction_pct']:.1f}%")
        with c3:
            st.metric("Est. Picking Productivity", f"{impact['current_picks_per_hour']:.0f} → {impact['optimized_picks_per_hour']:.0f} picks/hr",
                       delta=f"+{impact['estimated_productivity_gain_pct']:.1f}%")

        cba_quick = c.cost_benefit_analysis(impact, cleaned_df)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Annual Labour Savings (est.)", f"₹{cba_quick['annual_labour_savings_inr']:,.0f}")
        with c2:
            roi_disp = f"{cba_quick['roi_pct']:.0f}%" if cba_quick['roi_pct'] is not None else "N/A"
            st.metric("Year-1 ROI (est.)", roi_disp)
        with c3:
            pb_disp = f"{cba_quick['payback_months']:.1f} mo" if cba_quick['payback_months'] else "N/A"
            st.metric("Payback Period (est.)", pb_disp)

        st.info(
            f"**Key managerial message:** Re-slotting {len(sku_opt[sku_opt['Optimized_Distance_From_Dispatch_m'] < sku_opt['Distance_From_Dispatch_m']])} "
            f"high-priority SKUs closer to dispatch and reducing zone-level congestion is modeled to cut total picker "
            f"travel by {impact['travel_reduction_pct']:.1f}% and total picking time by {impact['time_reduction_pct']:.1f}% "
            f"on the analyzed order volume. Adjust cost/benefit assumptions in the sidebar and the Cost-Benefit tab to "
            f"reflect your facility's actual order volume."
        )
    else:
        st.warning("Optimization has not produced a result yet. Click 'Run / Refresh Optimization' in the sidebar.")

    st.markdown("#### Optimization Solver Status")
    st.json(opt_diag, expanded=False)

# ============================================================================
# TAB 2 — DATA QUALITY & PREPARATION
# ============================================================================
with tabs[1]:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{val_report.row_count:,}")
    c2.metric("Columns", val_report.column_count)
    c3.metric("Unique Orders", f"{val_report.unique_orders:,}")
    c4.metric("Unique SKUs", val_report.unique_skus)

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
# TAB 3 — SKU ANALYTICS
# ============================================================================
with tabs[2]:
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
    fig.update_layout(height=350, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4 — ABC / XYZ ANALYSIS
# ============================================================================
with tabs[3]:
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
        fig = px.bar(abc_df, x="Class", y="SKU_Count", color="Class", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=320, margin=dict(t=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("##### XYZ Distribution")
        xyz_df = pd.DataFrame(list(summary["xyz_distribution"].items()), columns=["Class", "SKU_Count"]).sort_values("Class")
        fig = px.bar(xyz_df, x="Class", y="SKU_Count", color="Class", color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=320, margin=dict(t=20), showlegend=False)
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
# TAB 5 — AI CLUSTERING
# ============================================================================
with tabs[4]:
    st.subheader("AI-Based SKU Clustering (K-Means)")
    c1, c2 = st.columns(2)
    c1.metric("Selected K", cluster_diag["best_k"])
    c2.metric("Silhouette Score", f"{cluster_diag['best_silhouette']:.3f}" if not np.isnan(cluster_diag["best_silhouette"]) else "N/A")

    sil_df = pd.DataFrame(list(cluster_diag["silhouette_scores"].items()), columns=["K", "Silhouette_Score"]).sort_values("K")
    fig = px.line(sil_df, x="K", y="Silhouette_Score", markers=True, title="Silhouette Score by K")
    fig.add_vline(x=cluster_diag["best_k"], line_dash="dash", line_color=ACCENT_COLOR)
    fig.update_layout(height=320, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Features used: {', '.join(cluster_diag['features_used'])}")

    st.markdown("##### Cluster Visualization (Velocity vs Travel Burden)")
    if {"Picking_Frequency_Weekly", "Distance_From_Dispatch_m"}.issubset(sku_clustered.columns):
        fig = px.scatter(
            sku_clustered, x="Distance_From_Dispatch_m", y="Picking_Frequency_Weekly",
            color=sku_clustered["Cluster"].astype(str), hover_data=["SKU_ID", "Category"],
            labels={"color": "Cluster"},
        )
        fig.update_layout(height=420, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Managerial Cluster Profiles")
    st.dataframe(cluster_diag["cluster_profile"], use_container_width=True)

    st.markdown("##### SKU-to-Cluster Assignment")
    cl_cols = ["SKU_ID", "Category", "Cluster"] + [f for f in cluster_diag["features_used"]]
    st.dataframe(sku_clustered[cl_cols].round(2), use_container_width=True, height=350)
    st.download_button("⬇️ Download clustering results (CSV)", c.to_csv_download(sku_clustered[cl_cols]), "clustering_results.csv", "text/csv")

# ============================================================================
# TAB 6 — ASSOCIATION ANALYSIS
# ============================================================================
with tabs[5]:
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
        fig = px.bar(top_chart.sort_values("Lift"), x="Lift", y="Pair", orientation="h", color="Support", color_continuous_scale="Viridis")
        fig.update_layout(height=450, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Category-Level Relationships")
        if {"Category_A", "Category_B"}.issubset(f_pairs.columns):
            cat_pairs = f_pairs.groupby(["Category_A", "Category_B"]).size().reset_index(name="Pair_Count")
            st.dataframe(cat_pairs.sort_values("Pair_Count", ascending=False), use_container_width=True)

# ============================================================================
# TAB 7 — CURRENT WAREHOUSE
# ============================================================================
with tabs[6]:
    st.subheader("Current (Baseline) Warehouse Layout Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Picker Travel", f"{kpis['total_picker_travel_m']:,.0f} m")
    c2.metric("Total Picking Time", f"{kpis['total_picking_time_sec']/3600:,.1f} h")
    c3.metric("Est. Picks / Hour", f"{kpis['estimated_picks_per_hour']:.0f}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Travel / Order Line", f"{kpis['avg_travel_per_line_m']:.1f} m")
    c2.metric("Avg Travel / Order", f"{kpis['avg_travel_per_order_m']:.1f} m")
    c3.metric("Avg Picking Time / Order", f"{kpis['avg_picking_time_per_order_sec']:.1f} s")

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
# TAB 8 — SLOTTING OPTIMIZATION
# ============================================================================
with tabs[7]:
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
    st.json({k: v for k, v in opt_diag.items()}, expanded=False)

    if opt_diag.get("status") == "optimal":
        relocated = sku_opt[sku_opt["Optimized_Location_ID"] != (sku_opt["Current_Zone"] + "-" + sku_opt["Current_Aisle"] + "-" + sku_opt["Current_Rack"])]
        st.metric("SKUs Recommended for Relocation", len(relocated))

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
            st.metric("Pairs with Improved Proximity", f"{improved_pct:.0f}%")
        else:
            st.info("No association pairs available to evaluate (try lowering the minimum support threshold).")
    else:
        st.error(f"Optimization did not converge to a feasible solution: {opt_diag.get('solver_message')}")

# ============================================================================
# TAB 9 — IMPACT ANALYSIS
# ============================================================================
with tabs[8]:
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
            fig.update_layout(title="Total Picker Travel Distance (m)", height=380, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(data=[go.Bar(
                x=["Current", "Optimized"],
                y=[impact["current_total_time_sec"] / 3600, impact["optimized_total_time_sec"] / 3600],
                marker_color=[ACCENT_COLOR, PRIMARY_COLOR],
                text=[f"{impact['current_total_time_sec']/3600:,.1f} h", f"{impact['optimized_total_time_sec']/3600:,.1f} h"],
                textposition="auto",
            )])
            fig.update_layout(title="Total Picking Time (hours)", height=380, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Productivity Implications")
        c1, c2, c3 = st.columns(3)
        c1.metric("Picks / Hour (current → optimized)", f"{impact['current_picks_per_hour']:.0f} → {impact['optimized_picks_per_hour']:.0f}")
        c2.metric("Productivity Gain (est.)", f"{impact['estimated_productivity_gain_pct']:.1f}%")
        c3.metric("Picking-Stage Fulfillment Time Reduction (est.)", f"{impact['fulfillment_picking_stage_reduction_pct']:.1f}%")
        st.caption(
            "The fulfillment-time figure reflects only the picking stage's contribution to order lead time "
            "(the stage this model directly optimizes) — not total dock-to-door fulfillment time."
        )

        st.markdown("##### Distribution of Travel Distance: Current vs Optimized (order-line level)")
        lines = impact["line_level_detail"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=lines["Picker_Travel_Distance_m"], name="Current", opacity=0.6, marker_color=ACCENT_COLOR))
        fig.add_trace(go.Histogram(x=lines["Optimized_Picker_Travel_m"], name="Optimized", opacity=0.6, marker_color=PRIMARY_COLOR))
        fig.update_layout(barmode="overlay", height=380, margin=dict(t=20))
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
# TAB 10 — SCENARIO SIMULATOR
# ============================================================================
with tabs[9]:
    st.subheader("Scenario Simulator")
    if impact.get("status") != "ok":
        st.warning("Run the optimization first (sidebar button) to use the scenario simulator.")
    else:
        c1, c2, c3 = st.columns(3)
        demand_growth = c1.slider("Demand growth (%)", 0, 100, 0, 5)
        peak_multiplier = c2.slider("Peak-demand multiplier", 1.0, 3.0, 1.0, 0.1)
        labour_cost = c3.number_input("Labour cost (INR / hour)", min_value=50.0, max_value=2000.0, value=150.0, step=10.0)

        sim = c.run_scenario_simulation(impact, demand_growth, peak_multiplier, labour_cost)
        c1, c2, c3 = st.columns(3)
        c1.metric("Optimized Total Travel", f"{sim['optimized_total_travel_m']:,.0f} m")
        c2.metric("Picking Hours Saved", f"{sim['picking_hours_saved']:,.1f} h")
        c3.metric("Est. Labour Cost Savings", f"₹{sim['labour_cost_savings_inr']:,.0f}")

        st.markdown("##### Preset Scenario Comparison")
        scen_table = c.run_scenario_table(impact, labour_cost)
        st.dataframe(scen_table, use_container_width=True)
        st.download_button("⬇️ Download scenario results (CSV)", c.to_csv_download(scen_table), "scenario_results.csv", "text/csv")

        fig = px.bar(scen_table, x="Scenario", y=["Current_Travel_m", "Optimized_Travel_m"], barmode="group",
                     color_discrete_sequence=[ACCENT_COLOR, PRIMARY_COLOR])
        fig.update_layout(height=400, margin=dict(t=20), yaxis_title="Total Picker Travel (m)")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 11 — COST-BENEFIT & FEASIBILITY
# ============================================================================
with tabs[10]:
    st.subheader("Cost-Benefit Analysis")
    if impact.get("status") != "ok":
        st.warning("Run the optimization first (sidebar button) for cost-benefit analysis.")
    else:
        st.markdown("##### Editable Cost Assumptions (INR)")
        cc1, cc2, cc3 = st.columns(3)
        reslot_cost = cc1.number_input("Re-slotting / movement cost", value=float(c.DEFAULT_COST_ASSUMPTIONS["reslotting_cost_inr"]), step=5000.0)
        software_cost = cc2.number_input("Software / analytics cost", value=float(c.DEFAULT_COST_ASSUMPTIONS["software_analytics_cost_inr"]), step=5000.0)
        training_cost = cc3.number_input("Training cost", value=float(c.DEFAULT_COST_ASSUMPTIONS["training_cost_inr"]), step=2000.0)
        cc1, cc2 = st.columns(2)
        disruption_cost = cc1.number_input("Temporary disruption cost", value=float(c.DEFAULT_COST_ASSUMPTIONS["disruption_cost_inr"]), step=2000.0)
        relabel_cost = cc2.number_input("Relabelling / signage cost", value=float(c.DEFAULT_COST_ASSUMPTIONS["relabelling_cost_inr"]), step=2000.0)

        st.markdown("##### Editable Benefit Assumptions")
        bc1, bc2, bc3 = st.columns(3)
        labour_cost_cba = bc1.number_input("Labour cost (INR/hour)", value=150.0, step=10.0)
        override_orders = bc2.number_input(
            "Assumed orders/day (0 = use observed rate from dataset)", min_value=0.0, value=0.0, step=10.0,
            help="The synthetic dataset spans a short window; override this with your facility's real daily order volume for a realistic annualized estimate."
        )
        operating_days = bc3.number_input("Operating days / year", value=330, step=5)

        cost_assumptions = {
            "reslotting_cost_inr": reslot_cost, "software_analytics_cost_inr": software_cost,
            "training_cost_inr": training_cost, "disruption_cost_inr": disruption_cost,
            "relabelling_cost_inr": relabel_cost,
        }
        benefit_assumptions = {
            "labour_cost_per_hour_inr": labour_cost_cba,
            "orders_per_day": override_orders if override_orders > 0 else None,
            "operating_days_per_year": operating_days,
        }
        cba = c.cost_benefit_analysis(impact, cleaned_df, cost_assumptions, benefit_assumptions)

        st.markdown("##### Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Implementation Cost", f"₹{cba['total_implementation_cost_inr']:,.0f}")
        c2.metric("Annual Labour Savings (est.)", f"₹{cba['annual_labour_savings_inr']:,.0f}")
        roi_disp = f"{cba['roi_pct']:.0f}%" if cba['roi_pct'] is not None else "N/A"
        c3.metric("Year-1 ROI (est.)", roi_disp)
        pb_disp = f"{cba['payback_months']:.1f} mo" if cba['payback_months'] else "N/A"
        c4.metric("Payback Period (est.)", pb_disp)

        st.caption(f"Observed orders/day in dataset: {cba['observed_orders_per_day']} | Assumed orders/day used: {cba['assumed_orders_per_day']} | Annual orders assumed: {cba['annual_orders_assumed']:,}")
        st.info(f"ℹ️ {cba['note']}")

        cba_df = pd.DataFrame([{k: v for k, v in cba.items() if k not in ("cost_assumptions", "benefit_assumptions", "note")}])
        st.download_button("⬇️ Download cost-benefit results (CSV)", c.to_csv_download(cba_df), "cost_benefit_results.csv", "text/csv")

        st.markdown("---")
        st.subheader("Feasibility Assessment")
        feas = c.feasibility_assessment(cba, opt_diag, proximity_df)
        c1, c2, c3, c4, c5 = st.columns(5)
        dims = list(feas["dimension_scores"].items())
        for col, (dim, score) in zip([c1, c2, c3, c4, c5], dims):
            col.metric(dim, f"{score}/5")
        st.metric("Overall Feasibility Score", f"{feas['overall_score_out_of_5']}/5")

        st.markdown("##### Feasibility Reasoning")
        for dim, reason in feas["reasons"].items():
            st.markdown(f"**{dim}** — {reason}")

# ============================================================================
# TAB 12 — MANAGERIAL RECOMMENDATIONS
# ============================================================================
with tabs[11]:
    st.subheader("Managerial Recommendations")
    if impact.get("status") != "ok":
        st.warning("Run the optimization first (sidebar button) to generate recommendations.")
    else:
        cba_for_rec = c.cost_benefit_analysis(impact, cleaned_df)
        recommendations = c.generate_recommendations(sku_opt, impact, proximity_df, zone_stats, cba_for_rec)

        for i, rec in enumerate(recommendations, 1):
            with st.container(border=True):
                st.markdown(f"#### {i}. {rec['title']}")
                st.markdown(f"**Issue:** {rec['issue']}")
                st.markdown(f"**Evidence:** {rec['evidence']}")
                st.markdown(f"**Recommendation:** {rec['recommendation']}")
                st.markdown(f"**Expected Impact:** {rec['expected_impact']}")
                st.markdown(f"**Implementation Consideration:** {rec['implementation_consideration']}")

        rec_df = pd.DataFrame(recommendations)
        st.download_button("⬇️ Download recommendations (CSV)", c.to_csv_download(rec_df), "recommendations.csv", "text/csv")

# ============================================================================
# TAB 13 — METHODOLOGY / DATA DICTIONARY
# ============================================================================
with tabs[12]:
    st.subheader("Methodology & Data Dictionary")

    st.markdown("#### Data Source")
    synthetic_disclaimer()
    st.markdown(
        "The dataset contains order-line-level records with order, SKU, and warehouse-placement "
        "attributes, plus derived analytical fields (ABC/XYZ class, observed demand/frequency)."
    )

    st.markdown("#### ABC/XYZ Methodology")
    st.markdown(
        """
        - **ABC:** SKUs sorted by revenue contribution (descending), cumulative % computed, then classified
          against user-configurable thresholds (default A ≤ 70%, B ≤ 90%, C = remainder).
        - **XYZ:** classified by demand coefficient of variation (Demand_CV) against user-configurable
          thresholds (default X ≤ 0.5, Y ≤ 1.0, Z = above).
        """
    )

    st.markdown("#### Clustering Methodology")
    st.markdown(
        """
        K-Means on standardized features (picking frequency, demand CV, average travel, package volume,
        weight, picking time, distance from dispatch). K is selected automatically via the best silhouette
        score across a configurable range, with manual override available.
        """
    )

    st.markdown("#### Association Analysis Methodology")
    st.markdown(
        """
        Market-basket analysis at the Order_ID level. For SKU pair (A, B): support = orders containing both /
        total orders; confidence(A→B) = orders containing both / orders containing A; lift = support(A,B) /
        (support(A) × support(B)). Lift > 1 indicates a positive purchase association.
        """
    )

    st.markdown("#### Optimization Methodology")
    st.markdown(
        """
        **Decision variables:** x[i,j] ∈ {0,1} — SKU i assigned to candidate location j.

        **Objective:** minimize Σ x[i,j] × (travel_weight × PriorityScore[i] × Distance[j] +
        congestion_weight × PriorityScore[i] × BaselineZoneCongestion[zone(j)]).

        **Constraints:** each SKU assigned exactly one location; each location holds at most one SKU;
        eligible locations filtered by bin-capacity feasibility and storage-type zone compatibility.

        **Method:** `scipy.optimize.milp` (HiGHS branch-and-bound MILP solver). Because a full 120-SKU ×
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

    st.markdown("#### Assumptions")
    st.markdown(
        """
        - Distance-from-dispatch is deterministic by (zone, aisle, rack), estimated from the supplied data.
        - Bin capacity is deterministic by rack, estimated from the supplied data.
        - Picker travel distance and picking time are projected for new locations using simple linear models
          fitted on the *supplied* order-line data (Picker_Travel_Distance_m and Picking_Time_sec as a
          function of Distance_From_Dispatch_m).
        - Storage-type zone compatibility is inferred from which zones currently host each non-Ambient
          storage type in the baseline data (a modelling simplification, not a stated warehouse rule).
        - Financial estimates use user-editable assumptions and the dataset's short observation window;
          override "orders/day" in the Cost-Benefit tab to reflect real facility scale.
        """
    )

    st.markdown("#### Limitations")
    st.markdown(
        """
        - Synthetic/proxy dataset — absolute figures are illustrative, not actual BigBasket performance.
        - No live re-optimization of congestion after reassignment (single-pass model).
        - Clustering and priority-score weighting involve subjective analytical choices, made transparent
          and adjustable in the sidebar.
        - Co-purchase influence on slotting is evaluated post-hoc rather than jointly optimized.
        """
    )

    st.markdown("#### Responsible AI")
    st.markdown(
        """
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
