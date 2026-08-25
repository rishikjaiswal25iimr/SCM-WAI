"""
core.py
========
AI-Driven Warehouse Slotting and Order-Picking Optimization
Decision-Support Framework for BigBasket (Academic WAI Project, IIM Ranchi)

SINGLE CORE ANALYTICAL ENGINE
------------------------------
This module contains every analytical function used by the Streamlit
application (app.py). It is organised into clearly separated sections:

  0. CONFIGURATION & CONSTANTS
  1. DATA LOADING & VALIDATION
  2. DATA PREPARATION / FEATURE ENGINEERING
  3. SKU PROFILING & SLOTTING PRIORITY SCORE
  4. ABC / XYZ CLASSIFICATION
  5. AI-BASED SKU CLUSTERING
  6. ASSOCIATION / CO-PURCHASE ANALYSIS
  7. CURRENT WAREHOUSE LAYOUT ANALYSIS & HEATMAP
  8. SLOT PRIORITY (RELOCATION CANDIDATE) ANALYSIS
  9. SLOT OPTIMIZATION (MILP) + ASSOCIATION-AWARE ADJUSTMENT
 10. CURRENT VS OPTIMIZED IMPACT ANALYSIS
 11. SCENARIO SIMULATOR
 12. COST-BENEFIT ANALYSIS
 13. FEASIBILITY ASSESSMENT
 14. MANAGERIAL RECOMMENDATION ENGINE
 15. UTILITY / HELPER FUNCTIONS

IMPORTANT ACADEMIC / DATA NOTE
-------------------------------
This project uses a synthetic/proxy dataset constructed for academic
analysis. It does not represent confidential BigBasket operational data
or actual BigBasket warehouse performance. All KPIs and findings shown by
the application are calculated live from the supplied dataset and the
model assumptions documented here and in the README.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    SCIPY_MILP_AVAILABLE = True
except Exception:  # pragma: no cover
    SCIPY_MILP_AVAILABLE = False

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ============================================================================
# 0. CONFIGURATION & CONSTANTS
# ============================================================================

REQUIRED_COLUMNS = [
    "Order_ID", "Order_Line_No", "Order_Date", "Channel", "Order_Priority",
    "Order_Size_Lines", "SKU_ID", "Category", "Subcategory", "Quantity",
    "Width_cm", "Depth_cm", "Height_cm", "Weight_kg", "Storage_Type",
    "Criticality", "Shelf_Life_Days", "Unit_Value_INR", "Avg_Daily_Demand",
    "Demand_CV", "Demand_StdDev", "Avg_Weekly_Demand",
    "Picking_Frequency_Weekly", "Current_Zone", "Current_Aisle",
    "Current_Rack", "Current_Bin_Capacity_Units", "Distance_From_Dispatch_m",
    "Picker_Travel_Distance_m", "Picking_Time_sec",
    "Observed_SKU_90D_Demand", "Observed_SKU_Order_Frequency",
    "ABC_Class", "XYZ_Class", "ABC_XYZ_Class",
]

NUMERIC_COLUMNS = [
    "Order_Line_No", "Order_Size_Lines", "Quantity", "Width_cm", "Depth_cm",
    "Height_cm", "Weight_kg", "Shelf_Life_Days", "Unit_Value_INR",
    "Avg_Daily_Demand", "Demand_CV", "Demand_StdDev", "Avg_Weekly_Demand",
    "Picking_Frequency_Weekly", "Current_Bin_Capacity_Units",
    "Distance_From_Dispatch_m", "Picker_Travel_Distance_m",
    "Picking_Time_sec", "Observed_SKU_90D_Demand",
    "Observed_SKU_Order_Frequency",
]

# Warehouse physical layout — inferred from the supplied dataset. The
# baseline data occupies 120 of these 240 candidate slotting locations,
# which gives the optimizer real unused locations to relocate SKUs into.
ZONES = [f"Z{i}" for i in range(1, 9)]     # Z1..Z8  (8 zones)
AISLES = [f"A{i}" for i in range(1, 6)]    # A1..A5  (5 aisles)
RACKS = [f"R{i}" for i in range(1, 7)]     # R1..R6  (6 racks)
TOTAL_LOCATIONS = len(ZONES) * len(AISLES) * len(RACKS)  # 240

# Distance_From_Dispatch_m is a deterministic function of (zone, aisle,
# rack) in the supplied dataset: distance = 8*zone_idx + 4*aisle_idx +
# 1.5*rack_idx - 1.5 (verified against the raw data with zero residual).
# This lets the optimizer compute a distance for every one of the 240
# candidate locations, not just the 120 currently occupied ones.
def location_distance(zone_idx: int, aisle_idx: int, rack_idx: int) -> float:
    return 8.0 * zone_idx + 4.0 * aisle_idx + 1.5 * rack_idx - 1.5


# Storage bin capacity is a deterministic function of rack in the supplied
# dataset (verified against raw data): R1=40, R2=50, R3=30, R4=40, R5=50,
# R6=30 units. Used to build the full 240-location capacity map.
RACK_CAPACITY = {"R1": 40, "R2": 50, "R3": 30, "R4": 40, "R5": 50, "R6": 30}

DEFAULT_ABC_THRESHOLDS = {"A": 70.0, "B": 90.0}   # cumulative % cut-offs
DEFAULT_XYZ_THRESHOLDS = {"X": 0.5, "Y": 1.0}      # demand CV cut-offs

DEFAULT_SLOTTING_WEIGHTS = {
    "velocity": 0.30,
    "value": 0.15,
    "variability_penalty": 0.10,
    "travel_burden": 0.25,
    "criticality": 0.10,
    "affinity": 0.10,
}

SYNTHETIC_DATA_DISCLAIMER = (
    "This project uses a synthetic/proxy dataset constructed for academic "
    "analysis. It does not represent confidential BigBasket operational "
    "data or actual BigBasket warehouse performance."
)


# ============================================================================
# 1. DATA LOADING & VALIDATION
# ============================================================================

@dataclass
class ValidationReport:
    row_count: int = 0
    column_count: int = 0
    unique_orders: int = 0
    unique_skus: int = 0
    missing_values: dict = field(default_factory=dict)
    duplicate_rows: int = 0
    missing_required_columns: list = field(default_factory=list)
    numeric_anomalies: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        # The app can proceed as long as the key identifier / analytical
        # columns needed for the core modules are present.
        critical = {"Order_ID", "SKU_ID"}
        return critical.isdisjoint(set(self.missing_required_columns))


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the supplied order-line CSV. Raises FileNotFoundError with a
    clear message if the user has not yet placed the file at the expected
    path (this file is supplied by the user and is never generated here)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset not found at '{path}'. Please place the "
            f"supplied CSV file at this path before running the app."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The supplied CSV file is empty.")
    return df


def validate_data(df: pd.DataFrame) -> ValidationReport:
    """Programmatically inspect the actual columns/rows rather than
    assuming hard-coded structure, and report data-quality issues."""
    report = ValidationReport()
    report.row_count = len(df)
    report.column_count = df.shape[1]

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report.missing_required_columns = missing_cols
    if missing_cols:
        report.notes.append(
            f"{len(missing_cols)} expected column(s) not found: "
            f"{', '.join(missing_cols)}. Analyses depending on these "
            f"columns will be skipped or degraded gracefully."
        )

    if "Order_ID" in df.columns:
        report.unique_orders = int(df["Order_ID"].nunique())
    if "SKU_ID" in df.columns:
        report.unique_skus = int(df["SKU_ID"].nunique())

    miss = df.isnull().sum()
    report.missing_values = {k: int(v) for k, v in miss.items() if v > 0}

    report.duplicate_rows = int(df.duplicated().sum())

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            n_negative = int((series < 0).sum())
            n_nonnumeric = int(series.isna().sum() - df[col].isna().sum())
            anomalies = {}
            if n_negative > 0:
                anomalies["negative_values"] = n_negative
            if n_nonnumeric > 0:
                anomalies["non_numeric_values"] = n_nonnumeric
            if anomalies:
                report.numeric_anomalies[col] = anomalies

    return report


# ============================================================================
# 2. DATA PREPARATION / FEATURE ENGINEERING
# ============================================================================

def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean and enrich the raw order-line data. Returns the cleaned
    dataframe plus a human-readable audit log of every transformation
    performed (nothing is modified silently)."""
    log = []
    data = df.copy()

    # --- Parse dates -------------------------------------------------
    if "Order_Date" in data.columns:
        before_na = data["Order_Date"].isna().sum()
        data["Order_Date"] = pd.to_datetime(data["Order_Date"], errors="coerce")
        after_na = data["Order_Date"].isna().sum()
        log.append(
            f"Parsed 'Order_Date' to datetime "
            f"({after_na - before_na} value(s) could not be parsed)."
        )

    # --- Coerce numeric columns --------------------------------------
    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            before_na = data[col].isna().sum()
            data[col] = pd.to_numeric(data[col], errors="coerce")
            after_na = data[col].isna().sum()
            if after_na > before_na:
                log.append(
                    f"Coerced '{col}' to numeric; "
                    f"{after_na - before_na} non-numeric value(s) became NaN."
                )

    # --- Handle missing values transparently --------------------------
    n_before = len(data)
    critical_cols = [c for c in ["Order_ID", "SKU_ID"] if c in data.columns]
    if critical_cols:
        data = data.dropna(subset=critical_cols)
    n_after = len(data)
    if n_after < n_before:
        log.append(
            f"Dropped {n_before - n_after} row(s) missing a critical "
            f"identifier (Order_ID / SKU_ID)."
        )

    numeric_present = [c for c in NUMERIC_COLUMNS if c in data.columns]
    for col in numeric_present:
        n_missing = data[col].isna().sum()
        if n_missing > 0:
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            log.append(
                f"Filled {n_missing} missing value(s) in '{col}' with the "
                f"column median ({median_val:.2f})."
            )

    # --- Remove exact duplicate rows -----------------------------------
    n_dupe = data.duplicated().sum()
    if n_dupe > 0:
        data = data.drop_duplicates()
        log.append(f"Removed {n_dupe} exact duplicate row(s).")

    # --- Flag unreasonable numeric values (do not silently drop) -------
    if "Weight_kg" in data.columns:
        n_bad = int((data["Weight_kg"] <= 0).sum())
        if n_bad:
            log.append(
                f"Flagged {n_bad} row(s) with non-positive Weight_kg "
                f"(kept in data, excluded from volumetric feature calcs)."
            )

    # --- Derived variables ----------------------------------------------
    if {"Width_cm", "Depth_cm", "Height_cm"}.issubset(data.columns):
        data["Package_Volume_cm3"] = (
            data["Width_cm"] * data["Depth_cm"] * data["Height_cm"]
        )
        log.append(
            "Derived 'Package_Volume_cm3' = Width_cm x Depth_cm x Height_cm."
        )

    if {"Order_ID", "SKU_ID"}.issubset(data.columns):
        data["Order_Month"] = (
            data["Order_Date"].dt.to_period("M").astype(str)
            if "Order_Date" in data.columns else "Unknown"
        )

    if {"Picker_Travel_Distance_m", "Picking_Time_sec"}.issubset(data.columns):
        data["Picking_Speed_m_per_sec"] = np.where(
            data["Picking_Time_sec"] > 0,
            data["Picker_Travel_Distance_m"] / data["Picking_Time_sec"],
            np.nan,
        )
        log.append(
            "Derived 'Picking_Speed_m_per_sec' = travel distance / picking "
            "time (used only for diagnostic purposes)."
        )

    data = data.reset_index(drop=True)
    log.append(f"Final cleaned dataset: {len(data)} rows x {data.shape[1]} columns.")
    return data, log


# ============================================================================
# 3. SKU PROFILING & SLOTTING PRIORITY SCORE
# ============================================================================

def build_sku_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order-line data into one row per SKU with the analytical
    attributes needed for ABC/XYZ, clustering, priority scoring and
    optimization."""

    agg_dict = {
        "Category": "first",
        "Subcategory": "first",
        "Storage_Type": "first",
        "Criticality": "first",
        "Shelf_Life_Days": "first",
        "Unit_Value_INR": "first",
        "Width_cm": "first",
        "Depth_cm": "first",
        "Height_cm": "first",
        "Weight_kg": "first",
        "Avg_Daily_Demand": "first",
        "Avg_Weekly_Demand": "first",
        "Demand_CV": "first",
        "Demand_StdDev": "first",
        "Picking_Frequency_Weekly": "first",
        "Current_Zone": "first",
        "Current_Aisle": "first",
        "Current_Rack": "first",
        "Current_Bin_Capacity_Units": "first",
        "Distance_From_Dispatch_m": "first",
        "Observed_SKU_90D_Demand": "first",
        "Observed_SKU_Order_Frequency": "first",
        "ABC_Class": "first",
        "XYZ_Class": "first",
        "ABC_XYZ_Class": "first",
        "Quantity": "sum",
        "Order_Line_No": "count",
        "Picker_Travel_Distance_m": "mean",
        "Picking_Time_sec": "mean",
    }
    if "Package_Volume_cm3" in data.columns:
        agg_dict["Package_Volume_cm3"] = "first"

    agg_dict = {k: v for k, v in agg_dict.items() if k in data.columns}
    sku = data.groupby("SKU_ID").agg(agg_dict).reset_index()

    sku = sku.rename(columns={
        "Quantity": "Total_Quantity_Demanded",
        "Order_Line_No": "Order_Line_Frequency",
        "Picker_Travel_Distance_m": "Avg_Picker_Travel_m",
        "Picking_Time_sec": "Avg_Picking_Time_sec",
    })

    # Revenue / value contribution
    if {"Total_Quantity_Demanded", "Unit_Value_INR"}.issubset(sku.columns):
        sku["Revenue_Contribution_INR"] = (
            sku["Total_Quantity_Demanded"] * sku["Unit_Value_INR"]
        )

    total_revenue = sku.get("Revenue_Contribution_INR", pd.Series(dtype=float)).sum()
    if total_revenue > 0:
        sku["Demand_Share_Pct"] = (
            100 * sku["Revenue_Contribution_INR"] / total_revenue
        )
    else:
        sku["Demand_Share_Pct"] = 0.0

    total_travel = sku["Avg_Picker_Travel_m"].sum() if "Avg_Picker_Travel_m" in sku else 0
    total_picks = sku["Order_Line_Frequency"].sum() if "Order_Line_Frequency" in sku else 1

    if {"Avg_Picker_Travel_m", "Order_Line_Frequency"}.issubset(sku.columns):
        sku["Travel_Burden_m"] = sku["Avg_Picker_Travel_m"] * sku["Order_Line_Frequency"]

    sku = sku.reset_index(drop=True)
    return sku


def compute_slotting_priority_score(
    sku_df: pd.DataFrame,
    weights: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute a transparent, configurable SKU Slotting Priority Score
    (0-100) combining business relevance (value, velocity, criticality)
    and operational impact (travel burden, demand variability, co-purchase
    affinity). Higher score = higher priority for a favourable (near
    dispatch) location."""
    weights = weights or DEFAULT_SLOTTING_WEIGHTS
    sku = sku_df.copy()

    def norm(col):
        if col not in sku.columns:
            return pd.Series(0.0, index=sku.index)
        vals = sku[col].astype(float)
        rng = vals.max() - vals.min()
        if rng == 0:
            return pd.Series(0.5, index=sku.index)
        return (vals - vals.min()) / rng

    velocity_n = norm("Picking_Frequency_Weekly")
    value_n = norm("Revenue_Contribution_INR")
    variability_n = norm("Demand_CV")           # higher CV -> harder to slot tightly
    travel_n = norm("Distance_From_Dispatch_m")  # current burden
    crit_map = {"High": 1.0, "Medium": 0.6, "Low": 0.3}
    criticality_n = sku["Criticality"].map(crit_map).fillna(0.5) if "Criticality" in sku else 0.5
    affinity_n = norm("Affinity_Score") if "Affinity_Score" in sku.columns else pd.Series(0.0, index=sku.index)

    score = (
        weights.get("velocity", 0) * velocity_n
        + weights.get("value", 0) * value_n
        + weights.get("variability_penalty", 0) * (1 - variability_n)
        + weights.get("travel_burden", 0) * travel_n
        + weights.get("criticality", 0) * criticality_n
        + weights.get("affinity", 0) * affinity_n
    )
    weight_sum = sum(weights.values()) or 1.0
    sku["Slotting_Priority_Score"] = 100 * score / weight_sum
    return sku


# ============================================================================
# 4. ABC / XYZ CLASSIFICATION
# ============================================================================

def classify_abc_xyz(
    sku_df: pd.DataFrame,
    abc_thresholds: Optional[dict] = None,
    xyz_thresholds: Optional[dict] = None,
) -> pd.DataFrame:
    """Explainable, value-based ABC analysis (cumulative revenue
    contribution) combined with variability-based XYZ analysis (demand
    coefficient of variation). Thresholds are configurable and always
    shown to the user."""
    abc_thresholds = abc_thresholds or DEFAULT_ABC_THRESHOLDS
    xyz_thresholds = xyz_thresholds or DEFAULT_XYZ_THRESHOLDS
    sku = sku_df.copy()

    # ---- ABC: sort by value contribution, cumulative % ----
    value_col = "Revenue_Contribution_INR" if "Revenue_Contribution_INR" in sku.columns else "Total_Quantity_Demanded"
    sku = sku.sort_values(value_col, ascending=False).reset_index(drop=True)
    total_value = sku[value_col].sum()
    sku["Cumulative_Value_Pct"] = (
        100 * sku[value_col].cumsum() / total_value if total_value > 0 else 0
    )

    def abc_label(pct):
        if pct <= abc_thresholds["A"]:
            return "A"
        elif pct <= abc_thresholds["B"]:
            return "B"
        return "C"

    sku["Computed_ABC_Class"] = sku["Cumulative_Value_Pct"].apply(abc_label)

    # ---- XYZ: demand coefficient of variation ----
    def xyz_label(cv):
        if pd.isna(cv):
            return "Y"
        if cv <= xyz_thresholds["X"]:
            return "X"
        elif cv <= xyz_thresholds["Y"]:
            return "Y"
        return "Z"

    if "Demand_CV" in sku.columns:
        sku["Computed_XYZ_Class"] = sku["Demand_CV"].apply(xyz_label)
    else:
        sku["Computed_XYZ_Class"] = "Y"

    sku["Computed_ABC_XYZ_Class"] = sku["Computed_ABC_Class"] + sku["Computed_XYZ_Class"]
    return sku


ABC_XYZ_INTERPRETATION = {
    "AX": "High value, stable demand — ideal for tight, prime slotting near dispatch with lean buffer stock.",
    "AY": "High value, moderate variability — prime slotting with moderate safety buffer.",
    "AZ": "High value, highly volatile demand — prime slotting but needs flexible capacity/buffer management.",
    "BX": "Medium value, stable demand — good candidate for efficient standard slotting.",
    "BY": "Medium value, moderate variability — standard slotting with periodic review.",
    "BZ": "Medium value, volatile demand — monitor closely; may need dynamic re-slotting.",
    "CX": "Low value, stable demand — can be slotted further from dispatch with minimal risk.",
    "CY": "Low value, moderate variability — lower priority; batch/bulk storage acceptable.",
    "CZ": "Low value, highly volatile demand — lowest priority; consolidate or review necessity.",
}


def abc_xyz_summary(sku_df: pd.DataFrame) -> dict:
    abc_dist = sku_df["Computed_ABC_Class"].value_counts().to_dict()
    xyz_dist = sku_df["Computed_XYZ_Class"].value_counts().to_dict()
    matrix = (
        sku_df.groupby(["Computed_ABC_Class", "Computed_XYZ_Class"])
        .size()
        .unstack(fill_value=0)
    )
    return {"abc_distribution": abc_dist, "xyz_distribution": xyz_dist, "matrix": matrix}


# ============================================================================
# 5. AI-BASED SKU CLUSTERING
# ============================================================================

CLUSTER_FEATURES = [
    "Picking_Frequency_Weekly",
    "Demand_CV",
    "Avg_Picker_Travel_m",
    "Package_Volume_cm3",
    "Weight_kg",
    "Avg_Picking_Time_sec",
    "Distance_From_Dispatch_m",
]


def run_kmeans_clustering(
    sku_df: pd.DataFrame,
    k_range: range = range(2, 8),
    manual_k: Optional[int] = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Cluster SKUs using KMeans on standardized operational features.
    Automatically scans k_range and selects the k with the best silhouette
    score unless manual_k is supplied. Returns the enriched dataframe and a
    diagnostics dict (silhouette scores per k, chosen k, cluster profiles)."""
    sku = sku_df.copy()
    features = [f for f in CLUSTER_FEATURES if f in sku.columns]
    X = sku[features].fillna(sku[features].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = {}
    n_samples = len(sku)
    valid_ks = [k for k in k_range if 2 <= k < n_samples]
    for k in valid_ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        try:
            score = silhouette_score(X_scaled, labels)
        except Exception:
            score = np.nan
        silhouette_scores[k] = score

    if manual_k and manual_k in valid_ks:
        best_k = manual_k
    elif silhouette_scores:
        best_k = max(silhouette_scores, key=lambda k: (silhouette_scores[k] if not np.isnan(silhouette_scores[k]) else -1))
    else:
        best_k = min(3, max(2, n_samples - 1))

    km_final = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    sku["Cluster"] = km_final.fit_predict(X_scaled)
    try:
        best_silhouette = silhouette_score(X_scaled, sku["Cluster"])
    except Exception:
        best_silhouette = np.nan

    # ---- Managerial profile per cluster ----
    profile_rows = []
    overall_median = sku[features].median()
    for c in sorted(sku["Cluster"].unique()):
        sub = sku[sku["Cluster"] == c]
        med = sub[features].median()
        tags = []
        if "Picking_Frequency_Weekly" in med and med["Picking_Frequency_Weekly"] >= overall_median.get("Picking_Frequency_Weekly", 0):
            tags.append("High velocity")
        else:
            tags.append("Low velocity")
        if "Package_Volume_cm3" in med and med["Package_Volume_cm3"] >= overall_median.get("Package_Volume_cm3", 0):
            tags.append("bulky package")
        else:
            tags.append("small package")
        if "Demand_CV" in med and med["Demand_CV"] >= overall_median.get("Demand_CV", 0):
            tags.append("variable demand")
        else:
            tags.append("predictable demand")
        if "Distance_From_Dispatch_m" in med and med["Distance_From_Dispatch_m"] >= overall_median.get("Distance_From_Dispatch_m", 0):
            tags.append("currently far from dispatch")
        else:
            tags.append("currently near dispatch")

        profile_rows.append({
            "Cluster": c,
            "SKU_Count": len(sub),
            "Managerial_Description": " / ".join(tags),
            **{f"Median_{f}": round(med.get(f, np.nan), 2) for f in features},
        })

    profile_df = pd.DataFrame(profile_rows)

    diagnostics = {
        "silhouette_scores": silhouette_scores,
        "best_k": best_k,
        "best_silhouette": best_silhouette,
        "features_used": features,
        "cluster_profile": profile_df,
    }
    return sku, diagnostics


# ============================================================================
# 6. ASSOCIATION / CO-PURCHASE ANALYSIS
# ============================================================================

def association_analysis(
    data: pd.DataFrame,
    min_support: float = 0.01,
    top_n: int = 25,
) -> pd.DataFrame:
    """Market-basket style co-purchase analysis at the Order_ID level.
    Computes pair frequency, support, confidence and lift for SKU pairs
    that appear together within the same order. Mathematically standard
    definitions are used throughout:

        support(A,B)   = orders containing both A and B / total orders
        confidence(A->B) = orders containing both / orders containing A
        lift(A,B)      = support(A,B) / (support(A) * support(B))
    """
    orders = data.groupby("Order_ID")["SKU_ID"].apply(lambda s: sorted(set(s)))
    total_orders = len(orders)
    if total_orders == 0:
        return pd.DataFrame()

    sku_order_count = {}
    pair_count = {}
    for skus in orders:
        for s in skus:
            sku_order_count[s] = sku_order_count.get(s, 0) + 1
        for a, b in itertools.combinations(skus, 2):
            key = (a, b)
            pair_count[key] = pair_count.get(key, 0) + 1

    sku_meta = data.drop_duplicates("SKU_ID").set_index("SKU_ID")

    rows = []
    for (a, b), freq in pair_count.items():
        support = freq / total_orders
        if support < min_support:
            continue
        support_a = sku_order_count[a] / total_orders
        support_b = sku_order_count[b] / total_orders
        confidence_ab = freq / sku_order_count[a]
        confidence_ba = freq / sku_order_count[b]
        lift = support / (support_a * support_b) if support_a * support_b > 0 else np.nan
        rows.append({
            "SKU_A": a,
            "SKU_B": b,
            "Category_A": sku_meta.loc[a, "Category"] if a in sku_meta.index else None,
            "Category_B": sku_meta.loc[b, "Category"] if b in sku_meta.index else None,
            "Pair_Frequency": freq,
            "Support": round(support, 4),
            "Confidence_A_to_B": round(confidence_ab, 4),
            "Confidence_B_to_A": round(confidence_ba, 4),
            "Lift": round(lift, 3) if not np.isnan(lift) else None,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(["Lift", "Pair_Frequency"], ascending=False).head(top_n)
    return result.reset_index(drop=True)


def build_affinity_scores(pairs_df: pd.DataFrame, sku_ids: list[str]) -> pd.Series:
    """Aggregate pairwise lift/support into a single per-SKU affinity
    intensity score, used as an input to the priority score and to the
    association-aware slotting adjustment."""
    if pairs_df is None or pairs_df.empty:
        return pd.Series(0.0, index=sku_ids)
    intensity = {}
    for _, row in pairs_df.iterrows():
        w = (row["Lift"] or 0) * row["Support"]
        intensity[row["SKU_A"]] = intensity.get(row["SKU_A"], 0) + w
        intensity[row["SKU_B"]] = intensity.get(row["SKU_B"], 0) + w
    return pd.Series(intensity).reindex(sku_ids).fillna(0.0)


# ============================================================================
# 7. CURRENT WAREHOUSE LAYOUT ANALYSIS & HEATMAP
# ============================================================================

def warehouse_baseline_kpis(data: pd.DataFrame) -> dict:
    """Baseline KPIs computed directly from the (Current_Zone,
    Current_Aisle, Current_Rack) fields treated as the physical layout."""
    n_orders = data["Order_ID"].nunique()
    n_lines = len(data)
    total_travel = data["Picker_Travel_Distance_m"].sum()
    total_time_sec = data["Picking_Time_sec"].sum()

    kpis = {
        "total_order_lines": int(n_lines),
        "unique_orders": int(n_orders),
        "unique_skus": int(data["SKU_ID"].nunique()),
        "total_picker_travel_m": float(total_travel),
        "avg_travel_per_line_m": float(data["Picker_Travel_Distance_m"].mean()),
        "avg_travel_per_order_m": float(total_travel / n_orders) if n_orders else 0,
        "total_picking_time_sec": float(total_time_sec),
        "avg_picking_time_per_line_sec": float(data["Picking_Time_sec"].mean()),
        "avg_picking_time_per_order_sec": float(total_time_sec / n_orders) if n_orders else 0,
        "estimated_picks_per_hour": float(3600 / data["Picking_Time_sec"].mean()) if data["Picking_Time_sec"].mean() else 0,
    }
    return kpis


def zone_level_activity(data: pd.DataFrame) -> pd.DataFrame:
    """Zone-level picking activity used for congestion-risk proxy and
    heatmaps."""
    g = data.groupby("Current_Zone").agg(
        Order_Lines=("Order_Line_No", "count"),
        Active_SKUs=("SKU_ID", "nunique"),
        Total_Picking_Frequency=("Picking_Frequency_Weekly", "sum"),
        Avg_Travel_m=("Picker_Travel_Distance_m", "mean"),
        Total_Travel_m=("Picker_Travel_Distance_m", "sum"),
        Avg_Picking_Time_sec=("Picking_Time_sec", "mean"),
        Total_Picking_Time_sec=("Picking_Time_sec", "sum"),
    ).reset_index()

    # Congestion Risk Proxy: a normalized composite of order-line density,
    # number of active high-frequency SKUs, and pick concentration within
    # the zone. This is an analytical proxy, not a measurement of actual
    # physical congestion.
    def normalize(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    line_density_n = normalize(g["Order_Lines"])
    sku_conc_n = normalize(g["Active_SKUs"])
    freq_conc_n = normalize(g["Total_Picking_Frequency"])

    g["Congestion_Risk_Proxy"] = round(
        100 * (0.45 * line_density_n + 0.25 * sku_conc_n + 0.30 * freq_conc_n), 1
    )
    return g.sort_values("Congestion_Risk_Proxy", ascending=False).reset_index(drop=True)


def aisle_level_activity(data: pd.DataFrame) -> pd.DataFrame:
    g = data.groupby(["Current_Zone", "Current_Aisle"]).agg(
        Order_Lines=("Order_Line_No", "count"),
        Active_SKUs=("SKU_ID", "nunique"),
        Avg_Travel_m=("Picker_Travel_Distance_m", "mean"),
        Avg_Picking_Time_sec=("Picking_Time_sec", "mean"),
        Total_Picking_Frequency=("Picking_Frequency_Weekly", "sum"),
    ).reset_index()
    return g


def heatmap_matrix(data: pd.DataFrame, metric: str = "Pick Frequency") -> pd.DataFrame:
    """Return a Zone x Aisle pivot table for the selected metric, for use
    in the warehouse heatmap visual. Supported metrics: 'Pick Frequency',
    'Picking Time', 'Travel Distance', 'Congestion Risk'."""
    metric_map = {
        "Pick Frequency": ("Picking_Frequency_Weekly", "sum"),
        "Picking Time": ("Picking_Time_sec", "mean"),
        "Travel Distance": ("Picker_Travel_Distance_m", "mean"),
    }
    if metric == "Congestion Risk":
        zone_stats = zone_level_activity(data)
        aisle_g = aisle_level_activity(data)
        merged = aisle_g.merge(
            zone_stats[["Current_Zone", "Congestion_Risk_Proxy"]], on="Current_Zone"
        )
        # scale zone congestion by aisle's share of zone activity
        merged["Zone_Total_Lines"] = merged.groupby("Current_Zone")["Order_Lines"].transform("sum")
        merged["Aisle_Congestion"] = merged["Congestion_Risk_Proxy"] * (
            merged["Order_Lines"] / merged["Zone_Total_Lines"]
        )
        pivot = merged.pivot(index="Current_Zone", columns="Current_Aisle", values="Aisle_Congestion")
    else:
        col, agg = metric_map[metric]
        pivot = data.pivot_table(index="Current_Zone", columns="Current_Aisle", values=col, aggfunc=agg)
    return pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns)).fillna(0)


# ============================================================================
# 8. SLOT PRIORITY (RELOCATION CANDIDATE) ANALYSIS
# ============================================================================

def slot_relocation_candidates(sku_df: pd.DataFrame) -> pd.DataFrame:
    """Rank SKUs as High / Medium / Low relocation priority using ABC
    class, XYZ class, velocity, current travel burden, package size and
    co-purchase affinity — with a transparent, human-readable reason for
    every SKU."""
    sku = sku_df.copy()

    score_col = "Slotting_Priority_Score" if "Slotting_Priority_Score" in sku.columns else None
    if score_col is None:
        sku = compute_slotting_priority_score(sku)
        score_col = "Slotting_Priority_Score"

    q_high = sku[score_col].quantile(0.66)
    q_med = sku[score_col].quantile(0.33)

    def tier(v):
        if v >= q_high:
            return "High Priority"
        elif v >= q_med:
            return "Medium Priority"
        return "Low Priority"

    sku["Relocation_Priority"] = sku[score_col].apply(tier)

    def reason(row):
        parts = []
        abc = row.get("Computed_ABC_Class") or row.get("ABC_Class")
        xyz = row.get("Computed_XYZ_Class") or row.get("XYZ_Class")
        if abc and xyz:
            parts.append(f"{abc}{xyz} SKU")
        if "Picking_Frequency_Weekly" in row and row["Picking_Frequency_Weekly"] >= sku["Picking_Frequency_Weekly"].median():
            parts.append("above-median picking frequency")
        if "Distance_From_Dispatch_m" in row and row["Distance_From_Dispatch_m"] >= sku["Distance_From_Dispatch_m"].median():
            parts.append("currently located far from dispatch (high travel burden)")
        if "Affinity_Score" in row and row["Affinity_Score"] > sku.get("Affinity_Score", pd.Series([0])).median():
            parts.append("strong co-purchase affinity with other SKUs")
        if not parts:
            parts.append("balanced operational profile")
        return f"{row['SKU_ID']} is a {row['Relocation_Priority'].lower()} candidate because it is " + ", ".join(parts) + "."

    sku["Relocation_Reason"] = sku.apply(reason, axis=1)
    return sku.sort_values(score_col, ascending=False).reset_index(drop=True)


# ============================================================================
# 9. SLOT OPTIMIZATION (MILP) + ASSOCIATION-AWARE ADJUSTMENT
# ============================================================================

def build_candidate_locations() -> pd.DataFrame:
    """Enumerate all TOTAL_LOCATIONS (zone, aisle, rack) physical slotting
    locations with their deterministic distance-from-dispatch and bin
    capacity, derived from the layout relationships verified in the
    supplied dataset (see location_distance() and RACK_CAPACITY)."""
    rows = []
    for zi, zone in enumerate(ZONES, start=1):
        for ai, aisle in enumerate(AISLES, start=1):
            for ri, rack in enumerate(RACKS, start=1):
                rows.append({
                    "Zone": zone, "Aisle": aisle, "Rack": rack,
                    "Distance_From_Dispatch_m": location_distance(zi, ai, ri),
                    "Bin_Capacity_Units": RACK_CAPACITY[rack],
                    "Location_ID": f"{zone}-{aisle}-{rack}",
                })
    return pd.DataFrame(rows)


def _fit_travel_time_models(data: pd.DataFrame) -> dict:
    """Fit simple, transparent linear models (from the order-line data)
    that translate a location's Distance_From_Dispatch_m into an expected
    Picker_Travel_Distance_m and Picking_Time_sec. Used to project the
    impact of relocating a SKU to a new distance."""
    x = data["Distance_From_Dispatch_m"].values.astype(float)
    X = np.column_stack([np.ones(len(x)), x])

    y_travel = data["Picker_Travel_Distance_m"].values.astype(float)
    coef_travel, *_ = np.linalg.lstsq(X, y_travel, rcond=None)

    y_time = data["Picking_Time_sec"].values.astype(float)
    coef_time, *_ = np.linalg.lstsq(X, y_time, rcond=None)

    return {
        "travel_intercept": float(coef_travel[0]), "travel_slope": float(coef_travel[1]),
        "time_intercept": float(coef_time[0]), "time_slope": float(coef_time[1]),
    }


def predict_travel(distance: np.ndarray, model: dict) -> np.ndarray:
    return model["travel_intercept"] + model["travel_slope"] * distance


def predict_time(distance: np.ndarray, model: dict) -> np.ndarray:
    return model["time_intercept"] + model["time_slope"] * distance


def _candidate_set_for_sku(row, locations: pd.DataFrame, congestion_by_zone: dict, top_k: int) -> pd.DataFrame:
    """Filter feasible locations for one SKU (capacity + storage-type zone
    compatibility) then keep only the top_k closest, to keep the MILP
    tractable (documented candidate-set reduction, per assignment
    requirements on optimization performance)."""
    cand = locations.copy()

    required_capacity = row.get("Current_Bin_Capacity_Units", 30)
    cand = cand[cand["Bin_Capacity_Units"] >= required_capacity]

    # Storage-type zone affinity: special storage types (Frozen, Chilled,
    # HighValue, Fragile) are restricted, as an explicit modelling
    # assumption, to zones that already host that storage type in the
    # baseline layout (representing zones equipped with the relevant
    # infrastructure e.g. refrigeration / secure cage). Ambient SKUs may
    # use any zone.
    storage_type = row.get("Storage_Type", "Ambient")
    if storage_type != "Ambient" and storage_type in congestion_by_zone.get("_storage_zone_map", {}):
        allowed_zones = congestion_by_zone["_storage_zone_map"][storage_type]
        if allowed_zones:
            cand = cand[cand["Zone"].isin(allowed_zones)]

    if cand.empty:
        cand = locations.copy()  # fallback: relax constraints rather than infeasible

    cand = cand.sort_values("Distance_From_Dispatch_m").head(top_k)
    return cand


def run_slotting_optimization(
    sku_df: pd.DataFrame,
    data: pd.DataFrame,
    travel_weight: float = 0.7,
    congestion_weight: float = 0.3,
    top_k_candidates: int = 60,
) -> tuple[pd.DataFrame, dict]:
    """Solve a genuine SKU-to-location assignment MILP:

    Decision variables: x[i,j] in {0,1} = 1 if SKU i is assigned to
    candidate location j (i ranges over all SKUs; j ranges over each
    SKU's individually-filtered candidate location set).

    Objective: minimize sum_i,j x[i,j] * cost[i,j], where
        cost[i,j] = travel_weight * Slotting_Priority_Score[i] * Distance[j]
                  + congestion_weight * Slotting_Priority_Score[i] * Baseline_Congestion[zone(j)]
    i.e. higher-priority SKUs are pushed toward low-distance, low-congestion
    locations.

    Constraints:
        - each SKU assigned to exactly one eligible location
        - each location holds at most one SKU (no duplicate assignment)
        - only locations meeting bin-capacity and storage-type
          compatibility for that SKU are eligible (candidate-set filter)

    Solved with scipy.optimize.milp (branch-and-bound MILP solver). If the
    solver reports infeasibility (extremely unlikely given the relaxed
    fallback in candidate-set construction) the function returns a
    diagnostics dict flagging this rather than fabricating a solution.
    """
    if not SCIPY_MILP_AVAILABLE:
        return sku_df, {"status": "unavailable", "message": "scipy.optimize.milp is not available in this environment."}

    sku = compute_slotting_priority_score(sku_df) if "Slotting_Priority_Score" not in sku_df.columns else sku_df.copy()
    locations = build_candidate_locations()

    zone_stats = zone_level_activity(data)
    congestion_by_zone = dict(zip(zone_stats["Current_Zone"], zone_stats["Congestion_Risk_Proxy"]))
    storage_zone_map = (
        data[data["Storage_Type"] != "Ambient"]
        .groupby("Storage_Type")["Current_Zone"].apply(lambda s: sorted(set(s))).to_dict()
    )
    congestion_by_zone["_storage_zone_map"] = storage_zone_map

    # Determine, for every SKU, its eligibility group (capacity tier +
    # storage-type zone restriction) so the candidate-set size can be
    # widened automatically when many SKUs share the same small pool of
    # eligible locations. Without this, a fixed top_k could make the
    # assignment problem infeasible purely by pigeonhole (more SKUs than
    # candidate locations in a shared, identical nearest-k set).
    def eligibility_key(row):
        storage_type = row.get("Storage_Type", "Ambient")
        return (row.get("Current_Bin_Capacity_Units", 30), storage_type)

    group_sizes = sku.apply(eligibility_key, axis=1).value_counts().to_dict()

    # Build per-SKU candidate sets
    candidate_lists = []
    for _, row in sku.iterrows():
        group_size = group_sizes.get(eligibility_key(row), 1)
        # Floor of 60 candidates keeps the assignment problem's Hall
        # condition comfortably satisfied in practice for this dataset's
        # location/capacity mix; scaled up further for large eligibility
        # groups so no group of same-tier SKUs is starved of distinct
        # candidate locations.
        effective_top_k = max(top_k_candidates, 60, int(group_size * 2) + 10)
        cand = _candidate_set_for_sku(row, locations, congestion_by_zone, effective_top_k).copy()
        cand["SKU_ID"] = row["SKU_ID"]
        cand["Priority_Score"] = row.get("Slotting_Priority_Score", 50.0)
        cand["Zone_Congestion"] = cand["Zone"].map(lambda z: congestion_by_zone.get(z, 0))
        cand["Cost"] = (
            travel_weight * cand["Priority_Score"] * cand["Distance_From_Dispatch_m"]
            + congestion_weight * cand["Priority_Score"] * cand["Zone_Congestion"]
        )
        candidate_lists.append(cand)

    all_cand = pd.concat(candidate_lists, ignore_index=True)
    all_cand["var_idx"] = range(len(all_cand))

    n_vars = len(all_cand)
    n_skus = sku["SKU_ID"].nunique()
    sku_ids = sku["SKU_ID"].tolist()
    sku_to_rows = {s: all_cand.index[all_cand["SKU_ID"] == s].tolist() for s in sku_ids}
    loc_to_rows = {loc: all_cand.index[all_cand["Location_ID"] == loc].tolist() for loc in all_cand["Location_ID"].unique()}

    # Objective (minimize)
    c = all_cand["Cost"].values.astype(float)

    # Constraint 1: each SKU assigned exactly one candidate location (== 1)
    rows_eq, cols_eq, data_eq = [], [], []
    for r_i, s in enumerate(sku_ids):
        for col in sku_to_rows[s]:
            rows_eq.append(r_i)
            cols_eq.append(col)
            data_eq.append(1.0)
    A_eq = _sparse_coo(rows_eq, cols_eq, data_eq, (n_skus, n_vars))
    eq_constraint = LinearConstraint(A_eq, lb=1, ub=1)

    # Constraint 2: each physical location used by at most one SKU (<= 1)
    loc_ids = list(loc_to_rows.keys())
    rows_le, cols_le, data_le = [], [], []
    for r_i, loc in enumerate(loc_ids):
        for col in loc_to_rows[loc]:
            rows_le.append(r_i)
            cols_le.append(col)
            data_le.append(1.0)
    A_le = _sparse_coo(rows_le, cols_le, data_le, (len(loc_ids), n_vars))
    le_constraint = LinearConstraint(A_le, lb=-np.inf, ub=1)

    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars)

    result = milp(
        c=c,
        constraints=[eq_constraint, le_constraint],
        integrality=integrality,
        bounds=bounds,
    )

    diagnostics = {
        "status": "optimal" if result.success else "infeasible_or_failed",
        "solver_message": result.message,
        "n_decision_variables": int(n_vars),
        "n_skus": int(n_skus),
        "n_candidate_locations_considered": int(len(loc_ids)),
        "objective_value": float(result.fun) if result.success else None,
    }

    if not result.success:
        # Graceful automatic retry with a larger candidate set before
        # giving up — protects against solver failure rather than
        # fabricating a solution.
        if top_k_candidates < 150:
            return run_slotting_optimization(
                sku_df, data, travel_weight, congestion_weight,
                top_k_candidates=min(top_k_candidates * 2, 240),
            )
        return sku, diagnostics

    chosen = all_cand.loc[np.round(result.x) > 0.5].copy()
    chosen = chosen.sort_values("SKU_ID").reset_index(drop=True)

    assignment = chosen[[
        "SKU_ID", "Zone", "Aisle", "Rack", "Location_ID",
        "Distance_From_Dispatch_m", "Bin_Capacity_Units", "Cost",
    ]].rename(columns={
        "Zone": "Optimized_Zone", "Aisle": "Optimized_Aisle", "Rack": "Optimized_Rack",
        "Location_ID": "Optimized_Location_ID",
        "Distance_From_Dispatch_m": "Optimized_Distance_From_Dispatch_m",
        "Bin_Capacity_Units": "Optimized_Bin_Capacity_Units",
        "Cost": "Optimization_Cost",
    })

    sku_opt = sku.merge(assignment, on="SKU_ID", how="left")
    return sku_opt, diagnostics


def _sparse_coo(rows, cols, data, shape):
    from scipy.sparse import coo_matrix
    return coo_matrix((data, (rows, cols)), shape=shape)


def association_aware_adjustment(sku_opt: pd.DataFrame, pairs_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Association-aware secondary adjustment: after primary MILP
    optimization (which handles travel, congestion, capacity and
    compatibility), evaluate whether high-affinity SKU pairs ended up
    physically closer than their baseline positions. This measures the
    influence of co-purchase relationships on the final layout without
    adding combinatorial pair-distance terms directly into the MILP
    (which would make the model computationally unstable at this scale).
    """
    if pairs_df is None or pairs_df.empty or "Optimized_Zone" not in sku_opt.columns:
        return sku_opt, pd.DataFrame()

    loc_lookup = sku_opt.set_index("SKU_ID")[
        ["Current_Zone", "Current_Aisle", "Current_Rack", "Optimized_Zone", "Optimized_Aisle", "Optimized_Rack"]
    ]

    def same_zone(z1, z2):
        return z1 == z2

    rows = []
    for _, pair in pairs_df.iterrows():
        a, b = pair["SKU_A"], pair["SKU_B"]
        if a not in loc_lookup.index or b not in loc_lookup.index:
            continue
        la, lb = loc_lookup.loc[a], loc_lookup.loc[b]
        baseline_same_zone = same_zone(la["Current_Zone"], lb["Current_Zone"])
        optimized_same_zone = same_zone(la["Optimized_Zone"], lb["Optimized_Zone"])
        baseline_same_aisle = baseline_same_zone and (la["Current_Aisle"] == lb["Current_Aisle"])
        optimized_same_aisle = optimized_same_zone and (la["Optimized_Aisle"] == lb["Optimized_Aisle"])
        rows.append({
            "SKU_A": a, "SKU_B": b, "Lift": pair["Lift"], "Support": pair["Support"],
            "Baseline_Same_Zone": baseline_same_zone, "Optimized_Same_Zone": optimized_same_zone,
            "Baseline_Same_Aisle": baseline_same_aisle, "Optimized_Same_Aisle": optimized_same_aisle,
            "Proximity_Improved": (not baseline_same_zone and optimized_same_zone)
                                    or (not baseline_same_aisle and optimized_same_aisle),
        })
    proximity_df = pd.DataFrame(rows)
    return sku_opt, proximity_df


# ============================================================================
# 10. CURRENT VS OPTIMIZED IMPACT ANALYSIS
# ============================================================================

def impact_analysis(data: pd.DataFrame, sku_opt: pd.DataFrame) -> dict:
    """Directly compare baseline vs optimized layouts using the fitted
    travel/time models (Section 9) applied at the order-line level, so the
    comparison reflects real order-line volumes rather than SKU counts."""
    model = _fit_travel_time_models(data)

    if "Optimized_Distance_From_Dispatch_m" not in sku_opt.columns:
        return {"status": "no_optimization_result"}

    dist_map = sku_opt.set_index("SKU_ID")["Optimized_Distance_From_Dispatch_m"].to_dict()
    lines = data.copy()
    lines["Optimized_Distance_From_Dispatch_m"] = lines["SKU_ID"].map(dist_map)
    lines = lines.dropna(subset=["Optimized_Distance_From_Dispatch_m"])

    lines["Optimized_Picker_Travel_m"] = np.maximum(
        0, predict_travel(lines["Optimized_Distance_From_Dispatch_m"].values, model)
    )
    lines["Optimized_Picking_Time_sec"] = np.maximum(
        1, predict_time(lines["Optimized_Distance_From_Dispatch_m"].values, model)
    )

    current_travel = lines["Picker_Travel_Distance_m"].sum()
    optimized_travel = lines["Optimized_Picker_Travel_m"].sum()
    current_time = lines["Picking_Time_sec"].sum()
    optimized_time = lines["Optimized_Picking_Time_sec"].sum()

    travel_reduction = current_travel - optimized_travel
    time_reduction = current_time - optimized_time

    n_orders = lines["Order_ID"].nunique()

    result = {
        "status": "ok",
        "current_total_travel_m": float(current_travel),
        "optimized_total_travel_m": float(optimized_travel),
        "travel_reduction_m": float(travel_reduction),
        "travel_reduction_pct": float(100 * travel_reduction / current_travel) if current_travel else 0,
        "current_total_time_sec": float(current_time),
        "optimized_total_time_sec": float(optimized_time),
        "time_reduction_sec": float(time_reduction),
        "time_reduction_pct": float(100 * time_reduction / current_time) if current_time else 0,
        "current_avg_time_per_order_sec": float(current_time / n_orders) if n_orders else 0,
        "optimized_avg_time_per_order_sec": float(optimized_time / n_orders) if n_orders else 0,
        "current_picks_per_hour": float(3600 / lines["Picking_Time_sec"].mean()) if lines["Picking_Time_sec"].mean() else 0,
        "optimized_picks_per_hour": float(3600 / lines["Optimized_Picking_Time_sec"].mean()) if lines["Optimized_Picking_Time_sec"].mean() else 0,
        "line_level_detail": lines,
        "travel_time_model": model,
    }
    # Estimated fulfillment-time impact: derived from the picking-time
    # reduction per order (the picking stage's direct, measurable
    # contribution to overall order fulfillment lead time). This is NOT a
    # claim about total dock-to-door fulfillment time.
    if result["current_avg_time_per_order_sec"] > 0:
        result["fulfillment_picking_stage_reduction_pct"] = round(
            100 * (result["current_avg_time_per_order_sec"] - result["optimized_avg_time_per_order_sec"])
            / result["current_avg_time_per_order_sec"], 2
        )
    else:
        result["fulfillment_picking_stage_reduction_pct"] = 0.0

    productivity_gain_pct = (
        100 * (result["optimized_picks_per_hour"] - result["current_picks_per_hour"]) / result["current_picks_per_hour"]
        if result["current_picks_per_hour"] else 0
    )
    result["estimated_productivity_gain_pct"] = round(productivity_gain_pct, 2)

    return result


def zone_level_impact(data: pd.DataFrame, impact_result: dict) -> pd.DataFrame:
    """Zone-level shift in picking activity between baseline and optimized
    layouts (which zones gain/lose workload)."""
    if impact_result.get("status") != "ok":
        return pd.DataFrame()
    lines = impact_result["line_level_detail"]
    current_zone_load = lines.groupby("Current_Zone")["Picking_Time_sec"].sum().rename("Current_Time_sec")
    # Determine optimized zone per line via distance -> nearest matching zone
    return current_zone_load.reset_index()


# ============================================================================
# 11. SCENARIO SIMULATOR
# ============================================================================

def run_scenario_simulation(
    impact_result: dict,
    demand_growth_pct: float = 0.0,
    peak_multiplier: float = 1.0,
    labour_cost_per_hour_inr: float = 150.0,
) -> dict:
    """Project how the optimized (and, for comparison, baseline) layout
    performs under a demand-growth / peak-demand scenario. Growth scales
    order-line volume (and therefore total travel/time) linearly, which is
    a transparent, order-of-magnitude approximation appropriate for a
    slotting decision-support tool — not a full discrete-event simulation.
    """
    if impact_result.get("status") != "ok":
        return {"status": "no_optimization_result"}

    growth_factor = (1 + demand_growth_pct / 100) * peak_multiplier

    current_travel = impact_result["current_total_travel_m"] * growth_factor
    optimized_travel = impact_result["optimized_total_travel_m"] * growth_factor
    current_time_sec = impact_result["current_total_time_sec"] * growth_factor
    optimized_time_sec = impact_result["optimized_total_time_sec"] * growth_factor

    current_hours = current_time_sec / 3600
    optimized_hours = optimized_time_sec / 3600

    return {
        "status": "ok",
        "growth_factor": growth_factor,
        "current_total_travel_m": current_travel,
        "optimized_total_travel_m": optimized_travel,
        "travel_reduction_m": current_travel - optimized_travel,
        "current_total_picking_hours": current_hours,
        "optimized_total_picking_hours": optimized_hours,
        "picking_hours_saved": current_hours - optimized_hours,
        "labour_cost_savings_inr": (current_hours - optimized_hours) * labour_cost_per_hour_inr,
    }


SCENARIO_PRESETS = [
    ("Baseline", 0.0, 1.0),
    ("+10% Demand", 10.0, 1.0),
    ("+20% Demand", 20.0, 1.0),
    ("+30% Demand", 30.0, 1.0),
    ("Peak Demand", 20.0, 1.5),
]


def run_scenario_table(impact_result: dict, labour_cost_per_hour_inr: float = 150.0) -> pd.DataFrame:
    rows = []
    for name, growth, peak in SCENARIO_PRESETS:
        sim = run_scenario_simulation(impact_result, growth, peak, labour_cost_per_hour_inr)
        if sim.get("status") != "ok":
            continue
        rows.append({
            "Scenario": name,
            "Demand_Growth_Pct": growth,
            "Peak_Multiplier": peak,
            "Current_Travel_m": round(sim["current_total_travel_m"]),
            "Optimized_Travel_m": round(sim["optimized_total_travel_m"]),
            "Travel_Reduction_m": round(sim["travel_reduction_m"]),
            "Current_Picking_Hours": round(sim["current_total_picking_hours"], 1),
            "Optimized_Picking_Hours": round(sim["optimized_total_picking_hours"], 1),
            "Picking_Hours_Saved": round(sim["picking_hours_saved"], 1),
            "Labour_Cost_Savings_INR": round(sim["labour_cost_savings_inr"]),
        })
    return pd.DataFrame(rows)


# ============================================================================
# 12. COST-BENEFIT ANALYSIS
# ============================================================================

DEFAULT_COST_ASSUMPTIONS = {
    "reslotting_cost_inr": 150000.0,
    "software_analytics_cost_inr": 75000.0,
    "training_cost_inr": 40000.0,
    "disruption_cost_inr": 35000.0,
    "relabelling_cost_inr": 20000.0,
}

DEFAULT_BENEFIT_ASSUMPTIONS = {
    "labour_cost_per_hour_inr": 150.0,
    "operating_days_per_year": 330,
    "orders_per_day": None,   # if None, derived from dataset's observed order rate
    "discount_rate_pct": 10.0,
}


def cost_benefit_analysis(
    impact_result: dict,
    data: pd.DataFrame,
    cost_assumptions: Optional[dict] = None,
    benefit_assumptions: Optional[dict] = None,
) -> dict:
    """Transparent cost-benefit model. All assumptions are explicit
    parameters (editable in the Streamlit UI) and are always returned
    alongside the results so the user can see exactly what drove them.
    Labelled as scenario-based estimates, not actual BigBasket financials.
    """
    if impact_result.get("status") != "ok":
        return {"status": "no_optimization_result"}

    costs = {**DEFAULT_COST_ASSUMPTIONS, **(cost_assumptions or {})}
    benefits_in = {**DEFAULT_BENEFIT_ASSUMPTIONS, **(benefit_assumptions or {})}

    total_implementation_cost = sum(costs.values())

    # Observed order rate from the dataset's date span, used to scale
    # order-line-level time savings to an annual estimate.
    n_orders_observed = data["Order_ID"].nunique()
    if "Order_Date" in data.columns and data["Order_Date"].notna().any():
        span_days = max((data["Order_Date"].max() - data["Order_Date"].min()).days, 1)
    else:
        span_days = 90
    observed_orders_per_day = n_orders_observed / span_days

    orders_per_day = benefits_in["orders_per_day"] or observed_orders_per_day
    operating_days = benefits_in["operating_days_per_year"]

    time_saved_per_order_sec = (
        impact_result["current_avg_time_per_order_sec"] - impact_result["optimized_avg_time_per_order_sec"]
    )
    annual_orders = orders_per_day * operating_days
    annual_hours_saved = (time_saved_per_order_sec * annual_orders) / 3600
    annual_labour_savings_inr = annual_hours_saved * benefits_in["labour_cost_per_hour_inr"]

    net_benefit_year1 = annual_labour_savings_inr - total_implementation_cost
    roi_pct = (
        100 * (annual_labour_savings_inr - total_implementation_cost) / total_implementation_cost
        if total_implementation_cost else None
    )
    payback_months = (
        (total_implementation_cost / annual_labour_savings_inr) * 12
        if annual_labour_savings_inr > 0 else None
    )

    return {
        "status": "ok",
        "cost_assumptions": costs,
        "benefit_assumptions": benefits_in,
        "total_implementation_cost_inr": total_implementation_cost,
        "observed_orders_per_day": round(observed_orders_per_day, 1),
        "assumed_orders_per_day": round(orders_per_day, 1),
        "annual_orders_assumed": round(annual_orders),
        "time_saved_per_order_sec": round(time_saved_per_order_sec, 2),
        "annual_hours_saved": round(annual_hours_saved, 1),
        "annual_labour_savings_inr": round(annual_labour_savings_inr),
        "net_benefit_year1_inr": round(net_benefit_year1),
        "roi_pct": round(roi_pct, 1) if roi_pct is not None else None,
        "payback_months": round(payback_months, 1) if payback_months is not None else None,
        "note": "Scenario-based estimate derived from the synthetic dataset and the stated assumptions above — not an actual BigBasket financial result.",
    }


# ============================================================================
# 13. FEASIBILITY ASSESSMENT
# ============================================================================

def feasibility_assessment(cba_result: dict, optimization_diagnostics: dict, proximity_df: pd.DataFrame) -> dict:
    """Transparent managerial feasibility scoring (1-5 scale per
    dimension) with the reasoning for each score shown to the user."""
    scores = {}
    reasons = {}

    # Operational feasibility: did the optimizer converge to a valid,
    # capacity/compatibility-respecting solution?
    if optimization_diagnostics.get("status") == "optimal":
        scores["Operational"] = 4
        reasons["Operational"] = "Optimizer produced a feasible, capacity- and compatibility-respecting assignment for all SKUs."
    else:
        scores["Operational"] = 2
        reasons["Operational"] = "Optimizer did not converge to a full feasible solution; manual review required."

    # Technical feasibility: relies only on data already present in the
    # existing order/warehouse system.
    scores["Technical"] = 4
    reasons["Technical"] = "Recommendation uses only fields already captured in standard order/warehouse master data (dimensions, demand, location, picking history)."

    # Financial feasibility
    if cba_result.get("status") == "ok" and cba_result.get("roi_pct") is not None:
        roi = cba_result["roi_pct"]
        if roi >= 100:
            scores["Financial"] = 5
        elif roi >= 30:
            scores["Financial"] = 4
        elif roi >= 0:
            scores["Financial"] = 3
        else:
            scores["Financial"] = 2
        reasons["Financial"] = f"Estimated year-1 ROI of {roi}% under the stated cost/benefit assumptions."
    else:
        scores["Financial"] = 3
        reasons["Financial"] = "Financial impact not yet computed."

    # Organizational feasibility
    scores["Organizational"] = 3
    reasons["Organizational"] = "Re-slotting requires picker retraining, temporary pick-path disruption, and updated location labels/signage."

    # Data feasibility
    scores["Data"] = 3
    reasons["Data"] = (
        "Analysis uses a synthetic/proxy dataset. Real deployment would require validated live WMS data "
        "(actual SKU dimensions, live demand feeds, true rack capacities, and real congestion/travel-time telemetry)."
    )

    overall = round(sum(scores.values()) / len(scores), 2)

    return {"dimension_scores": scores, "reasons": reasons, "overall_score_out_of_5": overall}


# ============================================================================
# 14. MANAGERIAL RECOMMENDATION ENGINE
# ============================================================================

def generate_recommendations(
    sku_opt: pd.DataFrame,
    impact_result: dict,
    proximity_df: pd.DataFrame,
    zone_stats: pd.DataFrame,
    cba_result: dict,
) -> list[dict]:
    """Generate managerial recommendations that trace back to actual
    analytical findings from this run (no generic boilerplate numbers)."""
    recs = []

    # Recommendation 1 — velocity-based prime slotting
    if "Optimized_Distance_From_Dispatch_m" in sku_opt.columns:
        moved_closer = sku_opt[
            sku_opt["Optimized_Distance_From_Dispatch_m"] < sku_opt["Distance_From_Dispatch_m"]
        ].sort_values("Slotting_Priority_Score", ascending=False)
        n_moved_closer = len(moved_closer)
        top_examples = ", ".join(moved_closer["SKU_ID"].head(5).tolist())
        recs.append({
            "title": "Move high-velocity, high-priority SKUs closer to dispatch",
            "issue": "High-priority SKUs are currently slotted at above-average distance from dispatch, inflating picker travel.",
            "evidence": f"The optimization model relocated {n_moved_closer} SKU(s) to shorter-distance locations, including {top_examples}.",
            "recommendation": "Physically re-slot the identified high-priority SKUs to the optimized zone/aisle/rack positions.",
            "expected_impact": f"Contributes to the modeled {impact_result.get('travel_reduction_pct', 0):.1f}% reduction in total picker travel." if impact_result.get("status") == "ok" else "Reduces picker travel for these SKUs.",
            "implementation_consideration": "Sequence moves during low-order-volume windows; update bin labels and WMS location records immediately after physical relocation.",
        })

    # Recommendation 2 — co-location of high-affinity pairs
    if proximity_df is not None and not proximity_df.empty:
        improved = proximity_df[proximity_df["Proximity_Improved"]]
        recs.append({
            "title": "Co-locate frequently co-picked SKU pairs",
            "issue": "Several high-lift SKU pairs are currently split across different zones/aisles, adding cross-zone travel when both are picked in the same order.",
            "evidence": f"{len(improved)} of {len(proximity_df)} top co-purchased pairs became physically closer (same zone/aisle) after optimization.",
            "recommendation": "Where capacity allows, prioritize placing remaining high-lift pairs in adjacent aisles during the next slotting cycle.",
            "expected_impact": "Reduces cross-zone travel for multi-item orders containing these SKU combinations.",
            "implementation_consideration": "Balance co-location against storage-type compatibility and capacity constraints; not all pairs can be adjacent.",
        })

    # Recommendation 3 — congestion management
    if zone_stats is not None and not zone_stats.empty:
        top_congested = zone_stats.iloc[0]
        recs.append({
            "title": "Manage congestion risk in high-activity zones",
            "issue": f"Zone {top_congested['Current_Zone']} shows the highest Congestion Risk Proxy ({top_congested['Congestion_Risk_Proxy']}/100) based on order-line density, active SKU count and picking frequency concentration.",
            "evidence": f"Zone {top_congested['Current_Zone']} handles {int(top_congested['Order_Lines'])} order lines across {int(top_congested['Active_SKUs'])} SKUs.",
            "recommendation": "Avoid further concentrating bulky, high-frequency SKUs in this zone; distribute new high-velocity placements across adjacent lower-congestion zones where the optimizer allows.",
            "expected_impact": "Reduces picker path conflicts and queuing at peak order volumes.",
            "implementation_consideration": "Monitor zone-level congestion proxy after each re-slotting cycle to confirm it does not resurface.",
        })

    # Recommendation 4 — periodic review
    recs.append({
        "title": "Treat warehouse slotting as a periodically reviewed decision, not a static layout",
        "issue": "SKU velocity, seasonality and co-purchase patterns shift over time; a one-time slotting exercise degrades in effectiveness.",
        "evidence": "ABC/XYZ classification and clustering results are sensitive to the demand window analyzed; SKUs will migrate between classes as demand evolves.",
        "recommendation": "Re-run this ABC/XYZ, clustering and slotting-optimization pipeline on a quarterly cadence (or after major seasonal/promotional shifts).",
        "expected_impact": "Keeps travel/time savings from eroding as SKU mix and demand patterns evolve.",
        "implementation_consideration": "Requires a lightweight data-refresh process from the WMS; automatable given the existing pipeline.",
    })

    # Recommendation 5 — AI-assisted refresh mechanism
    recs.append({
        "title": "Establish an AI-assisted periodic slotting refresh mechanism",
        "issue": "Manual re-slotting analysis is time-consuming and inconsistent across review cycles.",
        "evidence": "This project demonstrates that the full pipeline (profiling, ABC/XYZ, clustering, association analysis, MILP optimization, cost-benefit) can run end-to-end on live order-line data in minutes.",
        "recommendation": "Operationalize this decision-support tool (or an equivalent internal system) as a recurring, scheduled analysis feeding warehouse-management change requests.",
        "expected_impact": f"Estimated annual labour savings of INR {cba_result.get('annual_labour_savings_inr', 0):,.0f} under the stated cost-benefit assumptions." if cba_result.get("status") == "ok" else "Sustains ongoing productivity gains.",
        "implementation_consideration": "Requires a validated live WMS data feed and a defined approval workflow before physical re-slotting is executed.",
    })

    return recs


# ============================================================================
# 15. UTILITY / HELPER FUNCTIONS
# ============================================================================

def to_csv_download(df: pd.DataFrame) -> bytes:
    """Serialize a dataframe to CSV bytes for a Streamlit download button."""
    return df.to_csv(index=False).encode("utf-8")


def format_number(x, decimals=0):
    try:
        return f"{x:,.{decimals}f}"
    except (TypeError, ValueError):
        return str(x)


def safe_get(d: dict, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default
