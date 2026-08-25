# AI-Driven Warehouse Slotting and Order-Picking Optimization
### A Decision Support Framework for BigBasket

MBA Working-with-AI (WAI) Individual Project — Supply Chain Management, IIM Ranchi.

---

## 1. Project Overview

This project is an interactive, AI-assisted **decision-support application** that helps a
grocery e-commerce fulfillment centre (using BigBasket as the case-company context) decide
**where each SKU should be physically placed inside the warehouse** to minimize picker
travel, picking time, and congestion risk — while respecting capacity and storage-type
constraints.

The application performs the entire analytical workflow end-to-end on a supplied order-line
dataset: data validation → cleaning → SKU profiling → ABC/XYZ classification → AI clustering
→ co-purchase association analysis → warehouse heatmap diagnosis → MILP-based slotting
optimization → current-vs-optimized impact analysis → scenario simulation → cost-benefit and
feasibility analysis → managerial recommendations.

## 2. Business Problem

Poor SKU placement causes excessive picker travel, longer picking times, aisle congestion,
and higher operating cost. The central managerial question this project answers is:

> **Where should each SKU be physically placed inside a warehouse so that picker travel,
> picking time, and congestion are minimized, while respecting warehouse and SKU
> constraints?**

This is a **warehouse slotting / SKU-to-location assignment** problem — not an inventory
forecasting or reorder-point optimization project.

## 3. Research / Managerial Question

Given a set of SKUs with heterogeneous demand velocity, physical dimensions, storage
requirements, and co-purchase relationships, and a fixed warehouse layout of zones, aisles
and racks, what assignment of SKUs to locations minimizes total picker travel and picking
time while respecting capacity and compatibility constraints, and is that assignment
financially and operationally worthwhile to implement?

## 4. SCM Concepts Used

- **Warehouse slotting** — SKU-to-location assignment decisions
- **ABC/XYZ analysis** — value- and variability-based SKU prioritization
- **Order-picking optimization** — minimizing picker travel and picking time
- **Warehouse efficiency vs responsiveness trade-offs**
- **Capacity utilization** — bin-capacity-constrained location assignment
- **Cost-benefit trade-off analysis** for operational change

## 5. AI / Analytics Used

- **K-Means clustering** (scikit-learn) with standardized features and silhouette-score-based
  K selection, for SKU segmentation into managerially interpretable groups.
- **Market-basket / association analysis** (support, confidence, lift) for co-purchase
  relationships, computed directly from order-line co-occurrence.
- **Mixed-Integer Linear Programming (MILP)** (`scipy.optimize.milp`, HiGHS solver) for the
  core SKU-to-location slotting optimization.
- **Scenario simulation** for demand-growth and peak-demand what-if analysis.

## 6. Dataset

**This project uses a synthetic/proxy dataset constructed for academic analysis. It does not
represent confidential BigBasket operational data or actual BigBasket warehouse
performance.** BigBasket is used only as the case-company context; all numerical findings
shown by the application are calculated from the supplied dataset and the model assumptions
documented in the app's Methodology tab.

The dataset must be supplied separately (it is **not** included in this repository) and
placed at:

```
data/bigbasket_warehouse_slotting_2000_order_lines.csv
```

It contains order-line-level records (~2,000 lines, ~547 orders, 120 unique SKUs) with
order-level fields (Order_ID, Order_Date, Channel, Order_Priority, ...), SKU-level fields
(dimensions, weight, storage type, criticality, demand statistics, ...), warehouse-placement
fields (Current_Zone, Current_Aisle, Current_Rack, bin capacity, distance from dispatch), and
picking-performance fields (Picker_Travel_Distance_m, Picking_Time_sec), plus supplied
ABC/XYZ classification fields. The application validates the actual columns present at
runtime rather than assuming a fixed schema, and degrades gracefully if expected columns are
missing.

## 7. Project Structure

```
bigbasket-warehouse-slotting/
│
├── app.py               # Streamlit dashboard / user interface
├── core.py               # Single core analytical engine (all SCM/ML/optimization logic)
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── data/
│   └── bigbasket_warehouse_slotting_2000_order_lines.csv   # supplied by user, not included
│
└── outputs/                # reserved for runtime-generated downloads (empty until app runs)
```

### File responsibilities

- **`app.py`** — builds the complete 13-tab Streamlit interface, loads/calls functions from
  `core.py`, manages sidebar controls and session state, displays KPIs/tables/charts,
  provides download buttons, and triggers the optimization run.
- **`core.py`** — the single analytical engine. Internally organised into 15 clearly
  commented sections (configuration, data loading/validation, data preparation, SKU
  profiling, ABC/XYZ, clustering, association analysis, warehouse/heatmap analysis, slot
  priority ranking, MILP optimization + association-aware adjustment, impact analysis,
  scenario simulation, cost-benefit analysis, feasibility assessment, recommendation engine,
  and utility helpers).

## 8. How to Run Locally

```bash
pip install -r requirements.txt
# place the supplied CSV at data/bigbasket_warehouse_slotting_2000_order_lines.csv
streamlit run app.py
```

The app performs the full analysis automatically on launch. If the CSV is not found at the
expected path, the sidebar offers a file-uploader fallback so the app still runs.

## 9. Streamlit Deployment Instructions

1. Push this repository (including the supplied CSV at `data/...csv`) to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io), sign in, and click
   "New app".
3. Select this repository, branch, and set the main file path to `app.py`.
4. Deploy. No Docker, database, or external/paid API credentials are required — the app runs
   entirely offline against the supplied CSV once deployed.

## 10. Methodology Summary

Full methodology, formulas, and assumptions are documented **live in the app's "Methodology &
Data Dictionary" tab** (Tab 13) so they always match the actual implementation. In summary:

- **ABC analysis**: SKUs ranked by revenue contribution, cumulative % thresholds (default
  A ≤ 70%, B ≤ 90%, C = remainder), user-configurable in the sidebar.
- **XYZ analysis**: classified by demand coefficient of variation (default X ≤ 0.5, Y ≤ 1.0,
  Z = above), user-configurable.
- **Clustering**: K-Means on standardized operational features; K auto-selected by silhouette
  score across a configurable range, with manual override.
- **Association analysis**: Order_ID-level market-basket analysis; support, confidence and
  lift computed with standard definitions.
- **Optimization model**:
  - *Decision variables*: `x[i,j] ∈ {0,1}` — SKU *i* assigned to candidate location *j*.
  - *Objective*: minimize `Σ x[i,j] × (travel_weight × PriorityScore[i] × Distance[j] +
    congestion_weight × PriorityScore[i] × BaselineZoneCongestion[zone(j)])`.
  - *Constraints*: each SKU assigned exactly one location; each location holds at most one
    SKU; eligible locations filtered by bin-capacity feasibility and storage-type zone
    compatibility.
  - *Solver*: `scipy.optimize.milp` (HiGHS branch-and-bound MILP solver).
  - *Performance*: a documented, dynamically-sized candidate-location filter (nearest
    eligible locations per SKU, widened per eligibility group) keeps the model tractable
    without ever fabricating a result — if the solver reports infeasibility, the app retries
    with a larger candidate set and, failing that, reports the solver status transparently
    rather than presenting a fake optimum.
  - *Association-aware slotting*: rather than embedding pairwise co-purchase terms directly
    into the MILP (computationally unstable at this scale), the top co-purchased SKU pairs
    are evaluated **after** primary optimization to measure how many became physically
    closer — a transparent way of showing association's influence on the final layout.
- **Impact analysis**: current vs optimized picker travel and picking time are compared using
  simple linear models fitted on the supplied order-line data (travel/time as a function of
  distance-from-dispatch), applied at the order-line level — not an arbitrary percentage.
- **Scenario simulator**: linearly scales order-line volume by demand-growth/peak multipliers
  — a transparent, order-of-magnitude approximation appropriate for slotting decision
  support, not a full discrete-event simulation.
- **Cost-benefit analysis**: fully editable cost and benefit assumptions (implementation
  costs, labour cost/hour, assumed daily order volume); outputs annualized benefit, net
  benefit, ROI, and payback period, always labelled as **scenario-based estimates**.
- **Feasibility assessment**: transparent 1–5 scoring across Operational, Technical,
  Financial, Organizational and Data dimensions, each with a stated reason.

## 11. Key Assumptions

- Distance-from-dispatch is deterministic by (zone, aisle, rack) and bin capacity is
  deterministic by rack — both relationships were empirically verified against the supplied
  dataset (zero residual linear fit) and used to project distances/capacities for all 240
  candidate locations, not just the 120 currently occupied ones.
- Storage-type zone compatibility (for Frozen/Chilled/HighValue/Fragile SKUs) is inferred
  from which zones currently host that storage type in the baseline data — an explicit
  modelling simplification representing zones equipped with the relevant infrastructure.
- Financial projections use the dataset's short observation window by default; the
  Cost-Benefit tab lets the user override "assumed orders/day" to reflect a real facility's
  actual scale.

## 12. Limitations

- The supplied dataset is synthetic/proxy — absolute KPI values are illustrative for this
  academic exercise, not actual BigBasket performance.
- The optimization model is a single-pass MILP; it does not iteratively re-optimize
  congestion after reassignment.
- Clustering K-selection and Slotting Priority Score weights involve analytical judgment;
  both are made transparent and user-adjustable rather than hidden.
- Co-purchase influence on slotting is evaluated post-hoc (association-aware adjustment)
  rather than jointly optimized within the MILP, for tractability.

## 13. Responsible AI / Ethics

- **Synthetic-data limitations**: all findings should be validated against real WMS data
  before any operational decision is made.
- **Model assumptions**: linear travel/time projection and a static congestion baseline are
  simplifications appropriate for decision support, not guarantees of exact real-world
  impact.
- **Clustering subjectivity**: K and feature choice affect cluster composition; the
  silhouette score guides but does not certify a single "correct" K.
- **Feature-weighting bias**: the Slotting Priority Score's weights are exposed as sidebar
  controls precisely so no single fixed weighting is presented as objectively correct.
- **Hallucination risk**: no BigBasket-specific operational facts, costs, or performance
  figures are asserted anywhere in this project; every number shown is computed live from the
  supplied dataset and the stated assumptions.
- **Managerial validation required**: every recommendation is intended for review by
  warehouse operations management before physical re-slotting is executed.
- **Data privacy**: a real deployment would require access controls on live demand and
  location data consistent with company data-privacy policy.

## 14. Academic Disclaimer

This is an academic MBA Working-with-AI (WAI) project submitted for the Supply Chain
Management course at IIM Ranchi. It is not an official BigBasket product, analysis, or
endorsement, and it must not be represented as containing confidential BigBasket data or
actual BigBasket performance figures.
