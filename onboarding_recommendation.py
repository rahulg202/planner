"""
Onboarding Recommendation Engine
=================================
Evaluates candidate start weeks for generator onboarding by running the full
52-week DP optimizer with and without new sites, then ranking candidates by
marginal cost difference (penalty, overtime, capacity).

Approach (mirrors the original ``recommend_onboarding`` from the legacy planner):
1. Run baseline optimizer with existing sites only → ``base_summary``
2. For each candidate start week, inject new site demand and re-run → ``cand_summary``
3. Marginal cost = ``cand_summary`` − ``base_summary`` for each cost component
4. Rank candidates by marginal penalty, marginal overtime, marginal capacity
"""

from __future__ import annotations

import warnings
from typing import List

import pandas as pd

from integrated_cost_optimizer import (
    IntegratedParams,
    ROW_COUNTRIES,
    batches_needed,
    clean_sites,
    build_weekly_demand,
    build_weekly_row_demand,
    solve_plan_integrated,
)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_onboarding_inputs(
    total_generators: int, start_week: int, end_week: int
) -> list[str]:
    """Validate user inputs and return a list of error strings (empty if valid)."""
    errors: list[str] = []
    if total_generators < 1:
        errors.append("Total generators must be at least 1.")
    if start_week < 1:
        errors.append("Start week must be at least 1.")
    if start_week >= end_week:
        errors.append("Start week must be less than end week.")
    return errors


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------

def enumerate_candidates(start_week: int, end_week: int) -> list[int]:
    """Return all candidate start weeks in [start_week, end_week] inclusive."""
    return list(range(start_week, end_week + 1))


# ---------------------------------------------------------------------------
# Demand injection helpers
# ---------------------------------------------------------------------------

def add_new_sites_demand(
    base_demand: list[int],
    new_sites: list[dict],
    candidate_start_week: int,
    params: IntegratedParams,
) -> list[int]:
    """Add recurring demand for new sites starting at *candidate_start_week*.

    Each site in *new_sites* must have ``interval_weeks``.  The site's first
    demand occurs at ``candidate_start_week`` and repeats every
    ``interval_weeks`` through the horizon.

    Returns a new demand array (does not mutate *base_demand*).
    """
    d = base_demand.copy()
    T = params.horizon_weeks
    for site in new_sites:
        interval = int(site["interval_weeks"])
        w = candidate_start_week
        while 1 <= w <= T:
            d[w] += 1
            w += interval
    return d


def add_new_sites_row_demand(
    base_row_demand: list[int],
    new_sites: list[dict],
    candidate_start_week: int,
    params: IntegratedParams,
) -> list[int]:
    """Add recurring ROW demand for new sites that are in ROW countries."""
    rd = base_row_demand.copy()
    T = params.horizon_weeks
    for site in new_sites:
        country = str(site.get("country", "")).strip().lower()
        if country not in ROW_COUNTRIES:
            continue
        interval = int(site["interval_weeks"])
        w = candidate_start_week
        while 1 <= w <= T:
            rd[w] += 1
            w += interval
    return rd


# ---------------------------------------------------------------------------
# Core evaluation: baseline + marginal cost per candidate
# ---------------------------------------------------------------------------

_COST_KEYS = ("penalty", "overtime", "capacity")


def run_baseline(
    active_df: pd.DataFrame,
    params: IntegratedParams,
    shutdown_weeks: list[int] | None = None,
    partial_shutdown_weeks: list[int] | None = None,
) -> dict:
    """Run the 52-week optimizer on existing sites only.

    Returns the summary dict from ``solve_plan_integrated``.
    """
    shutdown_weeks = shutdown_weeks or []
    partial_shutdown_weeks = partial_shutdown_weeks or []

    demand = build_weekly_demand(active_df, params)
    row_demand = build_weekly_row_demand(active_df, params)

    _, summary = solve_plan_integrated(
        demand=demand,
        shutdown_weeks=shutdown_weeks,
        partial_shutdown_weeks=partial_shutdown_weeks,
        row_demand=row_demand,
        row_cap=params.row_cap,
        params=params,
    )
    return summary


def evaluate_candidate(
    active_df: pd.DataFrame,
    new_sites: list[dict],
    candidate_start_week: int,
    params: IntegratedParams,
    base_summary: dict,
    shutdown_weeks: list[int] | None = None,
    partial_shutdown_weeks: list[int] | None = None,
) -> dict | None:
    """Run the full optimizer with new sites injected at *candidate_start_week*.

    Returns a result dict with absolute and marginal costs, or *None* if
    the candidate is infeasible.
    """
    shutdown_weeks = shutdown_weeks or []
    partial_shutdown_weeks = partial_shutdown_weeks or []

    base_demand = build_weekly_demand(active_df, params)
    base_row_demand = build_weekly_row_demand(active_df, params)

    cand_demand = add_new_sites_demand(
        base_demand, new_sites, candidate_start_week, params,
    )
    cand_row_demand = add_new_sites_row_demand(
        base_row_demand, new_sites, candidate_start_week, params,
    )

    try:
        plan_df, cand_summary = solve_plan_integrated(
            demand=cand_demand,
            shutdown_weeks=shutdown_weeks,
            partial_shutdown_weeks=partial_shutdown_weeks,
            row_demand=cand_row_demand,
            row_cap=params.row_cap,
            params=params,
        )
    except RuntimeError:
        return None

    return {
        "candidate_start_week": candidate_start_week,
        "feasible": True,
        # Absolute costs
        "total_penalty": cand_summary["total_penalty_cost"],
        "total_overtime": cand_summary["total_overtime_cost"],
        "total_capacity": cand_summary["total_capacity_cost"],
        "total_composite": cand_summary["total_composite_cost"],
        "overtime_weeks": cand_summary["overtime_weeks"],
        # Marginal costs (delta vs baseline)
        "delta_penalty": cand_summary["total_penalty_cost"] - base_summary["total_penalty_cost"],
        "delta_overtime": cand_summary["total_overtime_cost"] - base_summary["total_overtime_cost"],
        "delta_capacity": cand_summary["total_capacity_cost"] - base_summary["total_capacity_cost"],
        "delta_composite": cand_summary["total_composite_cost"] - base_summary["total_composite_cost"],
        "delta_overtime_weeks": cand_summary["overtime_weeks"] - base_summary["overtime_weeks"],
        # Full plan for display
        "plan_df": plan_df,
    }


def evaluate_all_candidates(
    active_df: pd.DataFrame,
    new_sites: list[dict],
    start_week: int,
    end_week: int,
    params: IntegratedParams,
    shutdown_weeks: list[int] | None = None,
    partial_shutdown_weeks: list[int] | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate every candidate start week in [start_week, end_week].

    Returns ``(base_summary, results)`` where *results* is a list of dicts
    from :func:`evaluate_candidate`.  Infeasible candidates are excluded
    with a warning.
    """
    base_summary = run_baseline(
        active_df, params, shutdown_weeks, partial_shutdown_weeks,
    )

    results: list[dict] = []
    for cand in enumerate_candidates(start_week, end_week):
        r = evaluate_candidate(
            active_df, new_sites, cand, params, base_summary,
            shutdown_weeks, partial_shutdown_weeks,
        )
        if r is None:
            warnings.warn(
                f"Candidate start week {cand} is infeasible; excluding."
            )
        else:
            results.append(r)

    return base_summary, results


# ---------------------------------------------------------------------------
# Ranking and selection
# ---------------------------------------------------------------------------

def rank_and_select_top5(
    results: list[dict],
) -> dict[str, list[dict]]:
    """Rank candidates by marginal penalty, overtime, and capacity.

    Ties on the primary delta are broken by ascending delta_composite,
    then by ascending candidate_start_week.

    Returns a dict with keys ``"penalty"``, ``"overtime"``, ``"capacity"``,
    each mapping to the top-5 candidates sorted by ascending marginal cost.
    """
    top5: dict[str, list[dict]] = {}
    for key in _COST_KEYS:
        sorted_list = sorted(
            results,
            key=lambda r, k=key: (
                r[f"delta_{k}"],
                r["delta_composite"],
                r["candidate_start_week"],
            ),
        )
        top5[key] = sorted_list[:5]
    return top5


# ---------------------------------------------------------------------------
# Batch metrics and cost formatting
# ---------------------------------------------------------------------------

def compute_batch_metrics(
    plan_df: pd.DataFrame, params: IntegratedParams
) -> dict:
    """Return counts of weeks with 1, 2, or 3 batches."""
    weeks_1 = weeks_2 = weeks_3 = 0
    col = "Good_Production" if "Good_Production" in plan_df.columns else "Good_Units_Produced"
    for good in plan_df[col]:
        nb = batches_needed(int(good), params)
        if nb == 1:
            weeks_1 += 1
        elif nb == 2:
            weeks_2 += 1
        elif nb >= 3:
            weeks_3 += 1
    return {
        "weeks_1_batch": weeks_1,
        "weeks_2_batch": weeks_2,
        "weeks_3_batch": weeks_3,
    }


def format_cost_thousands(cost: float) -> str:
    """Format a cost value as ``"$<integer>K"`` (e.g. 28000.0 → ``"$28K"``)."""
    return f"${round(cost / 1000)}K"


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def export_recommendation_excel(
    top5: dict[str, list[dict]],
    base_summary: dict,
    params: IntegratedParams,
) -> bytes:
    """Write recommendation results to an in-memory Excel workbook.

    Produces a Summary sheet with marginal costs for each candidate,
    plus one sheet per objective with the top-5 candidates' details.
    """
    import io

    buf = io.BytesIO()
    obj_sheet_names = {"penalty": "By_Penalty", "overtime": "By_Overtime", "capacity": "By_Capacity"}

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # --- Summary sheet ---
        summary_rows: list[dict] = []
        for obj_key, sheet_name in obj_sheet_names.items():
            for idx, r in enumerate(top5.get(obj_key, [])):
                metrics = compute_batch_metrics(r["plan_df"], params)
                summary_rows.append({
                    "Ranked_By": obj_key.title(),
                    "Rank": idx + 1,
                    "Start_Week": r["candidate_start_week"],
                    "Delta_Penalty": format_cost_thousands(r["delta_penalty"]),
                    "Delta_Overtime": format_cost_thousands(r["delta_overtime"]),
                    "Delta_Capacity": format_cost_thousands(r["delta_capacity"]),
                    "Delta_Composite": format_cost_thousands(r["delta_composite"]),
                    "Total_Penalty": format_cost_thousands(r["total_penalty"]),
                    "Total_Overtime": format_cost_thousands(r["total_overtime"]),
                    "Total_Capacity": format_cost_thousands(r["total_capacity"]),
                    "Overtime_Weeks": r["overtime_weeks"],
                    "Weeks_1_Batch": metrics["weeks_1_batch"],
                    "Weeks_2_Batch": metrics["weeks_2_batch"],
                    "Weeks_3_Batch": metrics["weeks_3_batch"],
                })

        # Baseline row
        summary_rows.insert(0, {
            "Ranked_By": "BASELINE",
            "Rank": 0,
            "Start_Week": "-",
            "Delta_Penalty": "-",
            "Delta_Overtime": "-",
            "Delta_Capacity": "-",
            "Delta_Composite": "-",
            "Total_Penalty": format_cost_thousands(base_summary["total_penalty_cost"]),
            "Total_Overtime": format_cost_thousands(base_summary["total_overtime_cost"]),
            "Total_Capacity": format_cost_thousands(base_summary["total_capacity_cost"]),
            "Overtime_Weeks": base_summary["overtime_weeks"],
            "Weeks_1_Batch": "-",
            "Weeks_2_Batch": "-",
            "Weeks_3_Batch": "-",
        })

        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        # --- Per-objective sheets with weekly plan columns ---
        for obj_key, sheet_name in obj_sheet_names.items():
            options = top5.get(obj_key, [])
            if not options:
                pd.DataFrame({"Info": ["No feasible options"]}).to_excel(
                    writer, sheet_name=sheet_name, index=False,
                )
                continue

            first_plan = options[0]["plan_df"]
            combined = first_plan[["Week"]].copy()

            for idx, r in enumerate(options):
                pdf = r["plan_df"]
                label = f"Opt{idx+1}_Wk{r['candidate_start_week']}"
                combined[f"{label}_Good_Prod"] = pdf["Good_Production"].values
                combined[f"{label}_Batches"] = pdf["Batch_Count"].values
                combined[f"{label}_Inv"] = pdf["Net_Inventory_End"].values

            combined.to_excel(writer, sheet_name=sheet_name, index=False)

    return buf.getvalue()
