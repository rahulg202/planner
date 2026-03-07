"""
Integrated Cost Optimization Model
===================================
Standalone Python script that plans weekly production by minimizing a weighted
composite cost function combining penalty, overtime, and capacity utilization costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Core parameter dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntegratedParams:
    """All configuration for the integrated cost optimizer."""

    # Production constraints
    horizon_weeks: int = 52
    min_batch_produced: int = 2
    max_batch_produced: int = 16
    test_discard_per_batch: int = 1
    normal_max_batches: int = 2
    overtime_max_batches: int = 3

    # Cost rates
    penalty_rate: float = 7000.0          # USD per unit-week early inventory
    late_penalty_multiplier: float = 100.0  # multiplier for backlog penalty
    overtime_rate: float = 2000.0          # USD per overtime week (3rd batch)
    capacity_rate: float = 15000.0             # USD per unused good unit slot per week

    # Weights (0.0 to 1.0)
    w_penalty: float = 1.0
    w_overtime: float = 1.0
    w_capacity: float = 1.0

    # ROW constraint
    row_cap: int = 2

    def __post_init__(self) -> None:
        _validate_weights(self.w_penalty, self.w_overtime, self.w_capacity)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def late_penalty_rate(self) -> float:
        """Penalty rate applied to backlog (last-resort late delivery)."""
        return self.penalty_rate * self.late_penalty_multiplier

    @property
    def max_good_per_batch(self) -> int:
        """Good units produced per batch (after discarding 1 test unit)."""
        return self.max_batch_produced - self.test_discard_per_batch  # 15

    @property
    def normal_max_good_week(self) -> int:
        """Maximum good units in a normal week (2 batches × 15)."""
        return self.normal_max_batches * self.max_good_per_batch  # 30

    @property
    def overtime_max_good_week(self) -> int:
        """Maximum good units in an overtime week (3 batches × 15)."""
        return self.overtime_max_batches * self.max_good_per_batch  # 45


# ---------------------------------------------------------------------------
# Weight validation helper
# ---------------------------------------------------------------------------

def _validate_weights(w_penalty: float, w_overtime: float, w_capacity: float) -> None:
    """
    Validate that all weights are in [0.0, 1.0] and at least one is non-zero.

    Raises
    ------
    ValueError
        If any weight is outside [0.0, 1.0] or all weights are 0.0.
    """
    weights = {"w_penalty": w_penalty, "w_overtime": w_overtime, "w_capacity": w_capacity}
    for name, value in weights.items():
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"Weight '{name}' must be in [0.0, 1.0], got {value}."
            )
    if w_penalty == 0.0 and w_overtime == 0.0 and w_capacity == 0.0:
        raise ValueError(
            "All weights (w_penalty, w_overtime, w_capacity) are 0.0. "
            "At least one weight must be non-zero to define an optimization objective."
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLS = ["site_id", "active", "next_demand_week", "interval_weeks"]
ROW_COUNTRIES = {"denmark", "uk", "netherlands", "sweden"}  # case-insensitive


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase stripped strings."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def read_sites(path: str, sites_sheet: str = "Sites") -> pd.DataFrame:
    """
    Read site data from an Excel (.xlsx) or CSV file.

    Parameters
    ----------
    path : str
        Path to the input file.
    sites_sheet : str
        Sheet name to read when the file is Excel (ignored for CSV).

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with normalized (lowercase) column names.

    Raises
    ------
    ValueError
        If any required column is missing after normalization.
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sites_sheet)

    df = _norm_cols(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found: {list(df.columns)}"
        )
    return df


def clean_sites(
    df: pd.DataFrame, params: IntegratedParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate and clean the raw sites DataFrame.

    Rules applied
    -------------
    - Only rows where Active is Y/YES/TRUE/1 are kept.
    - The ``country`` column is optional; absent → empty string (non-ROW).
    - Duplicate Site_IDs among active rows are reported as issues and excluded.
    - Next_Demand_Week values outside 1..horizon_weeks are reported as issues.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by :func:`read_sites`.
    params : IntegratedParams
        Model parameters (used for horizon_weeks).

    Returns
    -------
    active : pd.DataFrame
        Cleaned active sites with columns:
        site_id, next_demand_week, interval_weeks, country, is_row
    issues_df : pd.DataFrame
        Data-quality issues with columns: row_index, site_id, issue
    """
    d = df.copy()

    d["site_id"] = d["site_id"].astype(str).str.strip()
    d["active"] = d["active"].astype(str).str.strip().str.upper()
    d["is_active"] = d["active"].isin(["Y", "YES", "TRUE", "1"])

    # Optional country column — default to empty (non-ROW) if absent
    if "country" in d.columns:
        d["country"] = d["country"].astype(str).str.strip().str.lower()
    else:
        d["country"] = ""

    d["next_demand_week_num"] = pd.to_numeric(d["next_demand_week"], errors="coerce")
    d["interval_weeks_num"] = pd.to_numeric(d["interval_weeks"], errors="coerce")

    issues: List[Tuple[int, str, str]] = []
    active = d.loc[d["is_active"]].copy()

    for idx, r in active.iterrows():
        sid = r["site_id"]
        ndw = r["next_demand_week_num"]
        itv = r["interval_weeks_num"]

        if not sid or str(sid).lower() == "nan":
            issues.append((idx, str(sid), "Missing Site_ID"))
            continue
        if pd.isna(ndw) or pd.isna(itv):
            issues.append((idx, str(sid), "Missing Next_Demand_Week or Interval_Weeks"))
            continue
        if ndw < 1 or ndw > params.horizon_weeks:
            issues.append(
                (idx, str(sid), f"Next_Demand_Week out of range 1..{params.horizon_weeks}")
            )
        if itv < 1:
            issues.append((idx, str(sid), "Interval_Weeks must be >= 1"))

    # Report and exclude duplicate Site_IDs
    dupes = active["site_id"][active["site_id"].duplicated(keep=False)]
    if not dupes.empty:
        for sid in sorted(dupes.unique()):
            issues.append((-1, str(sid), "Duplicate Site_ID among active rows"))
        active = active[~active["site_id"].isin(dupes.unique())].copy()

    issues_df = pd.DataFrame(
        issues, columns=["row_index", "site_id", "issue"]
    ).sort_values(["issue", "site_id", "row_index"])

    active["next_demand_week"] = active["next_demand_week_num"].astype(int)
    active["interval_weeks"] = active["interval_weeks_num"].astype(int)
    active["is_row"] = active["country"].isin(ROW_COUNTRIES)

    keep = ["site_id", "next_demand_week", "interval_weeks", "country", "is_row"]
    active = active[keep].reset_index(drop=True)
    return active, issues_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demand building
# ---------------------------------------------------------------------------

def build_weekly_demand(
    active: pd.DataFrame, params: IntegratedParams
) -> List[int]:
    """
    Build a 1-indexed demand array across the planning horizon.

    Each active site contributes 1 unit of demand at its ``next_demand_week``
    and then every ``interval_weeks`` thereafter, wrapping within the horizon.

    Parameters
    ----------
    active : pd.DataFrame
        Cleaned active sites from :func:`clean_sites`.
        Must have columns: next_demand_week, interval_weeks.
    params : IntegratedParams
        Model parameters (used for horizon_weeks).

    Returns
    -------
    List[int]
        demand[t] for t = 1..horizon_weeks (index 0 unused, index 1 = week 1).
        demand[0] is always 0.
    """
    demand = [0] * (params.horizon_weeks + 1)  # 1-indexed; index 0 unused

    for _, row in active.iterrows():
        week = int(row["next_demand_week"])
        interval = int(row["interval_weeks"])
        while week <= params.horizon_weeks:
            demand[week] += 1
            week += interval

    return demand


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def compute_weekly_cost(
    inv_end: int,
    good_prod: int,
    week_type: str,
    params: IntegratedParams,
) -> float:
    """
    Compute the weighted composite cost for a single week.

    Parameters
    ----------
    inv_end : int
        Net inventory at end of week (>= 0 = early units held, < 0 = backlog).
    good_prod : int
        Good units produced this week (after test discard).
    week_type : str
        One of "Normal", "Partial", or "Shutdown".
    params : IntegratedParams
        Model parameters including rates and weights.

    Returns
    -------
    float
        Weighted composite cost = w_penalty × penalty + w_overtime × overtime + w_capacity × capacity.
    """
    # Penalty component (Requirements 2.1, 2.2)
    if inv_end >= 0:
        penalty = params.penalty_rate * inv_end
    else:
        penalty = params.late_penalty_rate * abs(inv_end)

    # Overtime component (Requirements 3.1, 3.4)
    overtime = params.overtime_rate if good_prod > params.normal_max_good_week else 0.0

    # Capacity utilization component (Requirements 4.1, 4.4, 4.5, 4.6)
    if week_type == "Shutdown":
        capacity = 0.0
    elif week_type == "Partial":
        ceiling = params.max_good_per_batch  # 15
        capacity = params.capacity_rate * max(0, ceiling - good_prod)
    else:  # Normal
        ceiling = params.normal_max_good_week  # 30
        capacity = params.capacity_rate * max(0, ceiling - good_prod)

    # Weighted composite (Requirement 1.1)
    return (
        params.w_penalty * penalty
        + params.w_overtime * overtime
        + params.w_capacity * capacity
    )


def build_weekly_row_demand(
    active: pd.DataFrame, params: IntegratedParams
) -> List[int]:
    """
    Build a 1-indexed ROW demand array across the planning horizon.

    Only sites where ``is_row`` is True (Denmark, UK, Netherlands, Sweden)
    contribute to this array.  The structure mirrors :func:`build_weekly_demand`.

    Parameters
    ----------
    active : pd.DataFrame
        Cleaned active sites from :func:`clean_sites`.
        Must have columns: next_demand_week, interval_weeks, is_row.
    params : IntegratedParams
        Model parameters (used for horizon_weeks).

    Returns
    -------
    List[int]
        row_demand[t] for t = 1..horizon_weeks (index 0 unused, index 1 = week 1).
        row_demand[0] is always 0.
    """
    row_demand = [0] * (params.horizon_weeks + 1)  # 1-indexed; index 0 unused

    row_sites = active[active["is_row"]]
    for _, row in row_sites.iterrows():
        week = int(row["next_demand_week"])
        interval = int(row["interval_weeks"])
        while week <= params.horizon_weeks:
            row_demand[week] += 1
            week += interval

    return row_demand


# ---------------------------------------------------------------------------
# DP Solver
# ---------------------------------------------------------------------------

def compute_inventory_bounds(
    demand: List[int],
    cap_max: List[int],
    params: IntegratedParams,
) -> Tuple[List[int], List[int]]:
    """
    Compute per-week inventory lower and upper bounds for DP pruning.

    Upper bound: no point holding more inventory than remaining demand.
    Lower bound: can go negative (backlog allowed as last resort) — the minimum
    is how far short we could be if we produce as much as possible from here on.

    Parameters
    ----------
    demand : List[int]
        1-indexed demand array (index 0 unused).
    cap_max : List[int]
        1-indexed max good units per week (index 0 unused).
    params : IntegratedParams
        Model parameters (used for horizon_weeks).

    Returns
    -------
    lb : List[int]
        Lower bound on inventory at end of each week (can be negative).
    ub : List[int]
        Upper bound on inventory at end of each week.
    """
    T = params.horizon_weeks
    lb = [0] * (T + 1)
    ub = [0] * (T + 1)

    # Suffix sums: remaining demand and remaining capacity after week t
    suffix_demand = [0] * (T + 2)   # suffix_demand[t] = sum(demand[t..T])
    suffix_cap = [0] * (T + 2)      # suffix_cap[t]    = sum(cap_max[t..T])

    for t in range(T, 0, -1):
        suffix_demand[t] = demand[t] + suffix_demand[t + 1]
        suffix_cap[t] = cap_max[t] + suffix_cap[t + 1]

    for t in range(1, T + 1):
        # Upper bound: remaining demand after this week (no point holding more)
        ub[t] = suffix_demand[t + 1]

        # Lower bound: worst case — we produce max from t+1..T but still can't
        # cover remaining demand → backlog = remaining_demand - remaining_cap
        remaining_demand = suffix_demand[t + 1]
        remaining_cap = suffix_cap[t + 1]
        lb[t] = remaining_demand - remaining_cap  # can be negative

    # Terminal week must end at exactly 0
    lb[T] = 0
    ub[T] = 0

    return lb, ub


def solve_plan_integrated(
    demand: List[int],
    shutdown_weeks: List[int],
    partial_shutdown_weeks: List[int],
    row_demand: List[int],
    row_cap: int,
    params: IntegratedParams,
) -> Tuple["pd.DataFrame", dict]:
    """
    DP forward pass to find the globally optimal 52-week production schedule.

    State: net inventory (integer, can be negative for backlog).
    DP value: (composite_cost, overtime_weeks, total_batches) tuple for tie-breaking.

    Parameters
    ----------
    demand : List[int]
        1-indexed total demand array.
    shutdown_weeks : List[int]
        Weeks with zero production.
    partial_shutdown_weeks : List[int]
        Weeks with max 1 batch (15 good units).
    row_demand : List[int]
        1-indexed ROW demand array.
    row_cap : int
        Max ROW units fulfilled per week.
    params : IntegratedParams
        Model parameters.

    Returns
    -------
    plan_df : pd.DataFrame
        Weekly plan with all required columns.
    summary : dict
        Cost summary dictionary.

    Raises
    ------
    RuntimeError
        If no feasible states exist at any week, or no solution at week 52 with inv=0.
    """
    T = params.horizon_weeks
    shutdown_set = set(shutdown_weeks)
    partial_set = set(partial_shutdown_weeks)

    # Build cap_max per week
    cap_max = [0] * (T + 1)
    week_types = [""] * (T + 1)
    for t in range(1, T + 1):
        if t in shutdown_set:
            cap_max[t] = 0
            week_types[t] = "Shutdown"
        elif t in partial_set:
            cap_max[t] = params.max_good_per_batch  # 15
            week_types[t] = "Partial"
        else:
            cap_max[t] = params.overtime_max_good_week  # 45
            week_types[t] = "Normal"

    lb, ub = compute_inventory_bounds(demand, cap_max, params)

    INF = (float("inf"), float("inf"), float("inf"))

    # dp[inv] = (composite_cost, overtime_weeks, total_batches)
    # We use a dict to only track reachable states
    dp: dict[int, tuple] = {0: (0.0, 0, 0)}
    prev: list[dict[int, tuple]] = [{}] * (T + 1)  # prev[t][inv] = (prev_inv, y)

    for t in range(1, T + 1):
        new_dp: dict[int, tuple] = {}
        new_prev: dict[int, tuple] = {}

        wt = week_types[t]
        d_t = demand[t]
        c_max = cap_max[t]

        for inv_prev, val_prev in dp.items():
            cost_prev, ot_prev, bat_prev = val_prev

            # Enforce minimum production: never create avoidable backlog.
            # The solver must produce at least enough so inv_new >= lb[t],
            # but also at least enough to cover demand when capacity allows.
            # y_min = max(0, d_t - inv_prev) ensures inv_new >= 0 when possible.
            # We only enforce this when capacity is available (c_max > 0).
            if c_max > 0:
                y_min = max(0, d_t - inv_prev)
                # Cap y_min at c_max — can't produce more than capacity allows
                y_min = min(y_min, c_max)
            else:
                y_min = 0

            # Enumerate all feasible production levels for this week.
            # Each batch yields 1..15 good units (batch size 2..16 minus 1 test discard).
            # With up to 3 batches, y can be any integer in [0, cap_max[t]].
            for y in range(y_min, c_max + 1):
                inv_new = inv_prev + y - d_t

                # Prune: outside inventory bounds
                if inv_new < lb[t] or inv_new > ub[t]:
                    continue

                # Compute cost for this week
                cost_week = compute_weekly_cost(inv_new, y, wt, params)

                # Overtime and batch tracking for tie-breaking
                ot_flag = 1 if y > params.normal_max_good_week else 0
                batches = math.ceil(y / params.max_good_per_batch) if y > 0 else 0

                new_cost = cost_prev + cost_week
                new_ot = ot_prev + ot_flag
                new_bat = bat_prev + batches
                candidate = (new_cost, new_ot, new_bat)

                if inv_new not in new_dp or candidate < new_dp[inv_new]:
                    new_dp[inv_new] = candidate
                    new_prev[inv_new] = (inv_prev, y)

        if not new_dp:
            raise RuntimeError(
                f"No feasible production states at week {t}. "
                "Check shutdown weeks and demand — total capacity may be insufficient."
            )

        dp = new_dp
        prev[t] = new_prev

    # Check terminal condition: inv must be 0 at week T
    if 0 not in dp:
        raise RuntimeError(
            f"No feasible solution with Net_Inventory_End = 0 at week {T}. "
            "Total demand cannot be satisfied within the planning horizon."
        )

    # Backward reconstruction
    y_plan = [0] * (T + 1)
    inv_plan = [0] * (T + 1)

    inv_cur = 0
    for t in range(T, 0, -1):
        inv_prev_val, y_t = prev[t][inv_cur]
        y_plan[t] = y_t
        inv_plan[t] = inv_cur
        inv_cur = inv_prev_val

    # Build plan DataFrame
    plan_df, summary = _build_plan_df(
        y_plan, inv_plan, demand, row_demand, row_cap,
        week_types, cap_max, params
    )
    return plan_df, summary


def _build_plan_df(
    y_plan: List[int],
    inv_plan: List[int],
    demand: List[int],
    row_demand: List[int],
    row_cap: int,
    week_types: List[str],
    cap_max: List[int],
    params: IntegratedParams,
) -> Tuple["pd.DataFrame", dict]:
    """
    Build the weekly plan DataFrame from reconstructed y and inv arrays.

    Parameters
    ----------
    y_plan : List[int]
        Good units produced per week (1-indexed).
    inv_plan : List[int]
        Net inventory at end of each week (1-indexed).
    demand : List[int]
        Total demand per week (1-indexed).
    row_demand : List[int]
        ROW demand per week (1-indexed).
    row_cap : int
        Max ROW units fulfilled per week.
    week_types : List[str]
        Week type per week (1-indexed).
    cap_max : List[int]
        Max good units per week (1-indexed).
    params : IntegratedParams
        Model parameters.

    Returns
    -------
    plan_df : pd.DataFrame
    summary : dict
    """
    T = params.horizon_weeks
    rows = []
    cumulative_cost = 0.0
    row_inv = 0  # ROW inventory carried forward

    total_penalty = 0.0
    total_overtime = 0.0
    total_capacity = 0.0
    total_composite = 0.0
    total_ot_weeks = 0

    for t in range(1, T + 1):
        y = y_plan[t]
        inv = inv_plan[t]
        wt = week_types[t]
        d = demand[t]
        rd = row_demand[t]

        # Batch breakdown — each batch yields 1..15 good units
        batches = math.ceil(y / params.max_good_per_batch) if y > 0 else 0
        # Distribute good units across batches (fill earlier batches first)
        rem = y
        batch_goods = []
        for _ in range(batches):
            alloc = min(rem, params.max_good_per_batch)
            batch_goods.append(alloc)
            rem -= alloc
        while len(batch_goods) < 3:
            batch_goods.append(0)
        batch1, batch2, batch3 = batch_goods[0], batch_goods[1], batch_goods[2]
        produced_total = y + batches * params.test_discard_per_batch
        testing_discard = batches * params.test_discard_per_batch
        overtime_used = 1 if y > params.normal_max_good_week else 0

        # Inventory split
        early_held = max(0, inv)
        late_backlog = max(0, -inv)

        # ROW fulfillment
        row_fulfilled = min(rd + row_inv, row_cap)
        row_inv = max(0, row_inv + rd - row_fulfilled)

        # Cost breakdown
        if inv >= 0:
            penalty_cost = params.penalty_rate * inv
        else:
            penalty_cost = params.late_penalty_rate * abs(inv)

        overtime_cost = params.overtime_rate if y > params.normal_max_good_week else 0.0

        if wt == "Shutdown":
            capacity_cost = 0.0
        elif wt == "Partial":
            capacity_cost = params.capacity_rate * max(0, params.max_good_per_batch - y)
        else:
            capacity_cost = params.capacity_rate * max(0, params.normal_max_good_week - y)

        composite_cost = (
            params.w_penalty * penalty_cost
            + params.w_overtime * overtime_cost
            + params.w_capacity * capacity_cost
        )
        cumulative_cost += composite_cost

        total_penalty += penalty_cost
        total_overtime += overtime_cost
        total_capacity += capacity_cost
        total_composite += composite_cost
        if overtime_used:
            total_ot_weeks += 1

        rows.append({
            "Week": t,
            "Week_Type": wt,
            "Demand_Due": d,
            "Good_Production": y,
            "Batch_Count": batches,
            "Batch1_Produced": batch1,
            "Batch2_Produced": batch2,
            "Batch3_Produced": batch3,
            "Produced_Total": produced_total,
            "Testing_Discard": testing_discard,
            "Overtime_Used": overtime_used,
            "Net_Inventory_End": inv,
            "Early_Units_Held": early_held,
            "Late_Units_Backlog": late_backlog,
            "ROW_Demand_Due": rd,
            "ROW_Fulfilled": row_fulfilled,
            "ROW_Inventory": row_inv,
            "Penalty_Cost_USD": penalty_cost,
            "Overtime_Cost_USD": overtime_cost,
            "Capacity_Utilization_Cost_USD": capacity_cost,
            "Composite_Cost_USD": composite_cost,
            "Cumulative_Composite_Cost_USD": cumulative_cost,
        })

    plan_df = pd.DataFrame(rows)

    summary = {
        "total_composite_cost": total_composite,
        "total_penalty_cost": total_penalty,
        "total_overtime_cost": total_overtime,
        "total_capacity_cost": total_capacity,
        "overtime_weeks": total_ot_weeks,
        "w_penalty": params.w_penalty,
        "w_overtime": params.w_overtime,
        "w_capacity": params.w_capacity,
    }

    return plan_df, summary


# ---------------------------------------------------------------------------
# Batch utilities  (Requirement 6.6)
# ---------------------------------------------------------------------------

def batches_needed(good_units: int, params: IntegratedParams) -> int:
    """
    Return the number of batches required to produce ``good_units`` good units.

    Each batch yields 1..15 good units (batch size 2..16 minus 1 test discard).
    0 good units → 0 batches.

    Parameters
    ----------
    good_units : int
        Target good units (must be non-negative).
    params : IntegratedParams
        Model parameters.

    Returns
    -------
    int
        Number of batches (0, 1, 2, or 3).

    Raises
    ------
    ValueError
        If good_units is negative.
    """
    if good_units < 0:
        raise ValueError(f"good_units must be >= 0, got {good_units}.")
    if good_units == 0:
        return 0
    return math.ceil(good_units / params.max_good_per_batch)


def split_good_into_batches(
    good_units: int, params: IntegratedParams
) -> List[int]:
    """
    Split ``good_units`` into individual batch sizes (good units per batch).

    Each batch yields 1..15 good units. Earlier batches are filled to the
    maximum (15) first; the last batch takes the remainder.
    Returns a list of length ``batches_needed(good_units, params)``.

    Parameters
    ----------
    good_units : int
        Total good units to split (non-negative).
    params : IntegratedParams
        Model parameters.

    Returns
    -------
    List[int]
        List of good-unit counts per batch, e.g. [15, 15] for 30 good units,
        [15, 7] for 22 good units.
        Empty list when good_units == 0.

    Raises
    ------
    ValueError
        If good_units is negative (see :func:`batches_needed`).
    """
    n = batches_needed(good_units, params)
    if n == 0:
        return []
    result = []
    rem = good_units
    for _ in range(n):
        alloc = min(rem, params.max_good_per_batch)
        result.append(alloc)
        rem -= alloc
    return result


# ---------------------------------------------------------------------------
# Excel export  (Requirements 9.1–9.6)
# ---------------------------------------------------------------------------

def export_excel(
    output_path: str,
    plan_df: "pd.DataFrame",
    active_df: "pd.DataFrame",
    issues_df: "pd.DataFrame",
    params: IntegratedParams,
    summary: dict,
) -> None:
    """
    Write the optimization results to an Excel workbook.

    Sheets written
    --------------
    Weekly_Plan   — one row per week with full cost breakdown (Req 9.1, 9.2, 9.3)
    Sites_Clean   — validated active sites used in the run (Req 9.5)
    Input_Issues  — data-quality problems found during input validation (Req 9.6)
    Model_Params  — all parameters and weights used in this run (Req 9.4)

    Parameters
    ----------
    output_path : str
        Destination .xlsx file path.
    plan_df : pd.DataFrame
        Weekly plan DataFrame returned by :func:`solve_plan_integrated`.
    active_df : pd.DataFrame
        Cleaned active sites from :func:`clean_sites`.
    issues_df : pd.DataFrame
        Input issues from :func:`clean_sites`.
    params : IntegratedParams
        Model parameters used for this run.
    summary : dict
        Cost summary dict returned by :func:`solve_plan_integrated`.
    """
    # Build Model_Params sheet rows
    param_rows = [
        ("horizon_weeks", params.horizon_weeks, "Planning horizon in weeks"),
        ("penalty_rate", params.penalty_rate, "USD per unit-week early inventory"),
        ("late_penalty_multiplier", params.late_penalty_multiplier, "Multiplier on penalty_rate for backlog"),
        ("late_penalty_rate", params.late_penalty_rate, "Effective backlog penalty rate (derived)"),
        ("overtime_rate", params.overtime_rate, "USD per overtime week (3rd batch)"),
        ("capacity_rate", params.capacity_rate, "USD per unused good unit slot per week"),
        ("w_penalty", params.w_penalty, "Weight for penalty cost component"),
        ("w_overtime", params.w_overtime, "Weight for overtime cost component"),
        ("w_capacity", params.w_capacity, "Weight for capacity utilization cost component"),
        ("row_cap", params.row_cap, "Max ROW units fulfilled per week"),
        ("min_batch_produced", params.min_batch_produced, "Min units produced per batch (incl. test discard)"),
        ("max_batch_produced", params.max_batch_produced, "Max units produced per batch (incl. test discard)"),
        ("test_discard_per_batch", params.test_discard_per_batch, "Test units discarded per batch"),
        ("normal_max_batches", params.normal_max_batches, "Max batches in a normal week"),
        ("overtime_max_batches", params.overtime_max_batches, "Max batches in an overtime week"),
        ("normal_max_good_week", params.normal_max_good_week, "Max good units in a normal week (derived)"),
        ("overtime_max_good_week", params.overtime_max_good_week, "Max good units in an overtime week (derived)"),
        # Summary stats
        ("total_composite_cost_usd", summary.get("total_composite_cost", ""), "Total composite cost across horizon"),
        ("total_penalty_cost_usd", summary.get("total_penalty_cost", ""), "Total penalty cost across horizon"),
        ("total_overtime_cost_usd", summary.get("total_overtime_cost", ""), "Total overtime cost across horizon"),
        ("total_capacity_cost_usd", summary.get("total_capacity_cost", ""), "Total capacity utilization cost across horizon"),
        ("overtime_weeks", summary.get("overtime_weeks", ""), "Number of weeks with 3rd batch"),
    ]
    params_df = pd.DataFrame(param_rows, columns=["Parameter", "Value", "Description"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        plan_df.to_excel(writer, sheet_name="Weekly_Plan", index=False)
        active_df.to_excel(writer, sheet_name="Sites_Clean", index=False)
        issues_df.to_excel(writer, sheet_name="Input_Issues", index=False)
        params_df.to_excel(writer, sheet_name="Model_Params", index=False)


# ---------------------------------------------------------------------------
# CLI — main entry point  (Requirements 10.1, 10.2, 1.4, 9.7)
# ---------------------------------------------------------------------------

def _parse_week_list(value: str) -> List[int]:
    """Parse a comma-separated string of week numbers into a sorted list of ints."""
    if not value or not value.strip():
        return []
    return sorted(int(w.strip()) for w in value.split(",") if w.strip())


def print_summary(summary: dict, active_count: int) -> None:
    """
    Print a console summary of the optimization run (Requirement 9.7).

    Parameters
    ----------
    summary : dict
        Cost summary dict returned by :func:`solve_plan_integrated`.
    active_count : int
        Number of active sites used in the plan.
    """
    print("\n=== Integrated Cost Optimization — Summary ===")
    print(f"  Active sites          : {active_count}")
    print(f"  Weights               : w_penalty={summary['w_penalty']:.3f}  "
          f"w_overtime={summary['w_overtime']:.3f}  "
          f"w_capacity={summary['w_capacity']:.3f}")
    print(f"  Total composite cost  : ${summary['total_composite_cost']:,.2f}")
    print(f"    Penalty component   : ${summary['total_penalty_cost']:,.2f}")
    print(f"    Overtime component  : ${summary['total_overtime_cost']:,.2f}")
    print(f"    Capacity component  : ${summary['total_capacity_cost']:,.2f}")
    print(f"  Overtime weeks        : {summary['overtime_weeks']}")
    print("==============================================\n")


def main() -> None:
    """
    CLI entry point for the integrated cost optimizer.

    All parameters are configurable via command-line arguments (Requirement 10.1).
    Validates weights before running the solver (Requirements 1.4, 1.5, 1.6, 10.2).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Integrated Cost Optimization Model — minimize weighted composite cost."
    )

    # Required arguments
    parser.add_argument("--input", required=True, help="Path to sites Excel or CSV file.")
    parser.add_argument("--output", required=True, help="Path to output Excel file.")

    # Optional input arguments
    parser.add_argument("--sites-sheet", default="Sites",
                        help="Sheet name for Excel input (default: Sites).")
    parser.add_argument("--shutdown-weeks", default="",
                        help="Comma-separated full shutdown week numbers (e.g. '1,2,3').")
    parser.add_argument("--partial-shutdown-weeks", default="",
                        help="Comma-separated partial shutdown week numbers (e.g. '4,5').")

    # Weight arguments
    parser.add_argument("--w-penalty", type=float, default=1.0,
                        help="Weight for penalty cost component [0.0–1.0] (default: 1.0).")
    parser.add_argument("--w-overtime", type=float, default=1.0,
                        help="Weight for overtime cost component [0.0–1.0] (default: 1.0).")
    parser.add_argument("--w-capacity", type=float, default=0.0,
                        help="Weight for capacity utilization cost component [0.0–1.0] (default: 0.0).")

    # Cost rate arguments
    parser.add_argument("--penalty-rate", type=float, default=7000.0,
                        help="USD per unit-week early inventory (default: 7000).")
    parser.add_argument("--late-penalty-multiplier", type=float, default=10.0,
                        help="Multiplier on penalty-rate for backlog weeks (default: 10).")
    parser.add_argument("--overtime-rate", type=float, default=2000.0,
                        help="USD per overtime week (default: 2000).")
    parser.add_argument("--capacity-rate", type=float, default=0.0,
                        help="USD per unused good unit slot per week (default: 0).")

    # Other optional arguments
    parser.add_argument("--row-cap", type=int, default=2,
                        help="Max ROW units fulfilled per week (default: 2).")
    parser.add_argument("--horizon", type=int, default=52,
                        help="Planning horizon in weeks (default: 52).")
    parser.add_argument("--print-summary", action="store_true",
                        help="Print a console summary after optimization.")

    args = parser.parse_args()

    # Validate weights before constructing params (Requirements 1.4, 1.5, 1.6, 10.2)
    _validate_weights(args.w_penalty, args.w_overtime, args.w_capacity)

    # Build params dataclass (will also validate via __post_init__)
    params = IntegratedParams(
        horizon_weeks=args.horizon,
        penalty_rate=args.penalty_rate,
        late_penalty_multiplier=args.late_penalty_multiplier,
        overtime_rate=args.overtime_rate,
        capacity_rate=args.capacity_rate,
        w_penalty=args.w_penalty,
        w_overtime=args.w_overtime,
        w_capacity=args.w_capacity,
        row_cap=args.row_cap,
    )

    # Parse shutdown week lists
    shutdown_weeks = _parse_week_list(args.shutdown_weeks)
    partial_shutdown_weeks = _parse_week_list(args.partial_shutdown_weeks)

    # Load and clean sites
    raw_df = read_sites(args.input, sites_sheet=args.sites_sheet)
    active_df, issues_df = clean_sites(raw_df, params)

    # Build demand arrays
    demand = build_weekly_demand(active_df, params)
    row_demand = build_weekly_row_demand(active_df, params)

    # Run solver
    plan_df, summary = solve_plan_integrated(
        demand, shutdown_weeks, partial_shutdown_weeks,
        row_demand, params.row_cap, params
    )

    # Export results
    export_excel(args.output, plan_df, active_df, issues_df, params, summary)
    print(f"Output written to: {args.output}")

    # Optional console summary (Requirement 9.7)
    if args.print_summary:
        print_summary(summary, active_count=len(active_df))


if __name__ == "__main__":
    main()
