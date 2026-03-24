"""Streamlit UI for the Integrated Cost Optimizer."""

import io
from typing import Any
import pandas as pd
import streamlit as st
from integrated_cost_optimizer import (
    IntegratedParams,
    clean_sites,
    build_weekly_demand,
    build_weekly_row_demand,
    solve_plan_integrated,
    _norm_cols,
    REQUIRED_COLS,
    ROW_COUNTRIES,
)
from onboarding_recommendation import (
    validate_onboarding_inputs,
    evaluate_all_candidates,
    rank_and_select_top5,
    compute_batch_metrics,
    format_cost_thousands,
    export_recommendation_excel,
)

st.set_page_config(
    page_title="Production Cost Optimizer",
    layout="wide",
)

# Session state initialization
for key in [
    "file_bytes",
    "file_name",
    "sheet_names",
    "last_plan_df",
    "last_active_df",
    "last_issues_df",
    "last_summary",
    "last_xlsx_bytes",
    "ob_group_results",
    "ob_params",
    "ob_file_bytes_saved",
    "ob_file_name_saved",
    "ob_sheet_saved",
    "ob_full_plan_df",
    "ob_full_summary",
    "ob_full_issues_df",
    "ob_full_xlsx",
    "ob_full_selections",
]:
    if key not in st.session_state:
        st.session_state[key] = None

st.title("Production Cost Optimizer")

tab_hyperparams, tab_optimizer, tab_onboarding = st.tabs(
    ["⚙️ Settings", "Cost Optimizer", "Onboarding Recommendation"]
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_week_list(text: str) -> tuple[list[int], str | None]:
    """Parse a comma-separated string of week numbers.

    Returns (parsed_list, error_message_or_None).
    Each token must be a positive integer.
    """
    if not text or not text.strip():
        return [], None
    tokens = [t.strip() for t in text.split(",")]
    result: list[int] = []
    for token in tokens:
        if not token:
            return [], "Invalid format: contains empty entries between commas."
        try:
            val = int(token)
        except ValueError:
            return [], f"'{token}' is not a valid integer week number."
        if val <= 0:
            return [], f"Week number must be a positive integer, got {val}."
        result.append(val)
    return sorted(result), None


def validate_inputs(
    file_bytes: bytes | None,
    sheet_name: str,
    sheet_names: list[str] | None,
    min_batch: int,
    max_batch: int,
    normal_max_batches: int,
    overtime_max_batches: int,
    w_penalty: float,
    w_overtime: float,
    w_capacity: float,
    horizon_weeks: int,
    shutdown_str: str,
    partial_str: str,
) -> tuple[list[str], list[str]]:
    """Validate all UI inputs. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    # File validation
    if file_bytes is None:
        errors.append("No file uploaded. Please upload an Excel workbook.")
    else:
        if sheet_names is None:
            errors.append("Uploaded file could not be parsed as a valid Excel workbook.")
        elif sheet_name not in sheet_names:
            errors.append(
                f"Sheet '{sheet_name}' not found in workbook. "
                f"Available sheets: {', '.join(sheet_names)}."
            )

    # Batch size cross-field validation
    if max_batch < min_batch:
        errors.append(
            f"max_batch_produced ({max_batch}) must be >= min_batch_produced ({min_batch})."
        )

    # Overtime batch cross-field validation
    if overtime_max_batches < normal_max_batches:
        errors.append(
            f"overtime_max_batches ({overtime_max_batches}) must be >= "
            f"normal_max_batches ({normal_max_batches})."
        )

    # Weight validation
    if w_penalty == 0.0 and w_overtime == 0.0 and w_capacity == 0.0:
        errors.append("At least one objective weight (w_penalty, w_overtime, w_capacity) must be non-zero.")

    # Shutdown week parsing
    shutdown_weeks, shutdown_err = parse_week_list(shutdown_str)
    if shutdown_err:
        errors.append(f"Shutdown weeks parse error: {shutdown_err}")
    else:
        for w in shutdown_weeks:
            if w > horizon_weeks:
                warnings.append(
                    f"Shutdown week {w} exceeds horizon_weeks ({horizon_weeks}) and will be ignored."
                )

    partial_weeks, partial_err = parse_week_list(partial_str)
    if partial_err:
        errors.append(f"Partial shutdown weeks parse error: {partial_err}")
    else:
        for w in partial_weeks:
            if w > horizon_weeks:
                warnings.append(
                    f"Partial shutdown week {w} exceeds horizon_weeks ({horizon_weeks}) and will be ignored."
                )

    return errors, warnings


def build_params(widget_values: dict[str, Any]) -> IntegratedParams:
    """Construct an IntegratedParams from collected widget values."""
    return IntegratedParams(
        horizon_weeks=widget_values["horizon_weeks"],
        min_batch_produced=widget_values["min_batch_produced"],
        max_batch_produced=widget_values["max_batch_produced"],
        test_discard_per_batch=widget_values["test_discard_per_batch"],
        normal_max_batches=widget_values["normal_max_batches"],
        overtime_max_batches=widget_values["overtime_max_batches"],
        penalty_rate=widget_values["penalty_rate"],
        late_penalty_multiplier=widget_values["late_penalty_multiplier"],
        overtime_rate=widget_values["overtime_rate"],
        capacity_rate=widget_values["capacity_rate"],
        w_penalty=widget_values["w_penalty"],
        w_overtime=widget_values["w_overtime"],
        w_capacity=widget_values["w_capacity"],
        row_cap=widget_values["row_cap"],
    )


def _shared_params_from_session() -> IntegratedParams:
    """Build IntegratedParams from the shared Hyperparameters tab session state."""
    return IntegratedParams(
        horizon_weeks=int(st.session_state.get("hp_horizon_weeks", 52)),
        min_batch_produced=int(st.session_state.get("hp_min_batch_produced", 2)),
        max_batch_produced=int(st.session_state.get("hp_max_batch_produced", 16)),
        test_discard_per_batch=int(st.session_state.get("hp_test_discard_per_batch", 1)),
        normal_max_batches=int(st.session_state.get("hp_normal_max_batches", 2)),
        overtime_max_batches=int(st.session_state.get("hp_overtime_max_batches", 3)),
        penalty_rate=float(st.session_state.get("hp_penalty_rate", 7000.0)),
        late_penalty_multiplier=float(st.session_state.get("hp_late_penalty_multiplier", 100.0)),
        overtime_rate=float(st.session_state.get("hp_overtime_rate", 2000.0)),
        capacity_rate=float(st.session_state.get("hp_capacity_rate", 15000.0)),
        w_penalty=float(st.session_state.get("hp_w_penalty", 1.0)),
        w_overtime=float(st.session_state.get("hp_w_overtime", 1.0)),
        w_capacity=float(st.session_state.get("hp_w_capacity", 1.0)),
        row_cap=int(st.session_state.get("hp_row_cap", 2)),
    )


def export_excel_bytes(
    plan_df: pd.DataFrame,
    active_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    params: IntegratedParams,
    summary: dict,
) -> bytes:
    """Write optimization results to an in-memory Excel workbook and return bytes."""
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
        ("min_batch_produced", params.min_batch_produced, "Min units produced per batch"),
        ("max_batch_produced", params.max_batch_produced, "Max units produced per batch"),
        ("test_discard_per_batch", params.test_discard_per_batch, "Test units discarded per batch"),
        ("normal_max_batches", params.normal_max_batches, "Max batches in a normal week"),
        ("overtime_max_batches", params.overtime_max_batches, "Max batches in an overtime week"),
        ("total_composite_cost_usd", summary.get("total_composite_cost", ""), "Total composite cost"),
        ("total_penalty_cost_usd", summary.get("total_penalty_cost", ""), "Total penalty cost"),
        ("total_overtime_cost_usd", summary.get("total_overtime_cost", ""), "Total overtime cost"),
        ("total_capacity_cost_usd", summary.get("total_capacity_cost", ""), "Total capacity cost"),
        ("overtime_weeks", summary.get("overtime_weeks", ""), "Number of weeks with 3rd batch"),
    ]
    params_df = pd.DataFrame(param_rows, columns=["Parameter", "Value", "Description"])

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        plan_df.to_excel(writer, sheet_name="Weekly_Plan", index=False)
        active_df.to_excel(writer, sheet_name="Sites_Clean", index=False)
        issues_df.to_excel(writer, sheet_name="Input_Issues", index=False)
        params_df.to_excel(writer, sheet_name="Model_Params", index=False)
    return buf.getvalue()


def run_optimizer(
    file_bytes: bytes,
    filename: str,
    sheet_name: str,
    params: IntegratedParams,
    shutdown_weeks: list[int],
    partial_weeks: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, bytes]:
    """Run the full optimizer pipeline and return results.

    Returns (plan_df, active_df, issues_df, summary, xlsx_bytes).
    Raises on any optimizer error.
    """
    from integrated_cost_optimizer import _norm_cols, REQUIRED_COLS
    file_like = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        raw_df = pd.read_csv(file_like)
    else:
        raw_df = pd.read_excel(file_like, sheet_name=sheet_name)
    raw_df = _norm_cols(raw_df)
    missing = [c for c in REQUIRED_COLS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(raw_df.columns)}")
    active_df, issues_df = clean_sites(raw_df, params)
    demand = build_weekly_demand(active_df, params)
    row_demand = build_weekly_row_demand(active_df, params)
    plan_df, summary = solve_plan_integrated(
        demand=demand,
        shutdown_weeks=shutdown_weeks,
        partial_shutdown_weeks=partial_weeks,
        row_demand=row_demand,
        row_cap=params.row_cap,
        params=params,
    )
    xlsx_bytes = export_excel_bytes(plan_df, active_df, issues_df, params, summary)
    return plan_df, active_df, issues_df, summary, xlsx_bytes


def render_results(
    summary: dict,
    plan_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    xlsx_bytes: bytes,
) -> None:
    """Render metric cards, plan table, issues section, and download button."""
    st.subheader("3. Results")

    # Metric cards (Requirements 8.1–8.4)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Cost (USD)", f"${summary.get('total_composite_cost', 0):,.0f}")
    col2.metric("Penalty Cost (USD)", f"${summary.get('total_penalty_cost', 0):,.0f}")
    col3.metric("Overtime Cost (USD)", f"${summary.get('total_overtime_cost', 0):,.0f}")
    col4.metric("Capacity Cost (USD)", f"${summary.get('total_capacity_cost', 0):,.0f}")
    col5.metric("Overtime Weeks", summary.get("overtime_weeks", 0))

    # Plan table (Requirement 8.5)
    st.markdown("**Weekly Production Plan**")
    st.dataframe(plan_df, use_container_width=True)

    # Collapsible issues section (Requirement 10.4)
    with st.expander(f"Data Quality Issues ({len(issues_df)} row(s))", expanded=False):
        if issues_df.empty:
            st.info("No data quality issues found.")
        else:
            st.dataframe(issues_df, use_container_width=True)

    # Download button (Requirements 9.1–9.3)
    st.download_button(
        label="Download Results",
        data=xlsx_bytes,
        file_name="plan_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ---------------------------------------------------------------------------
# Cost Optimizer Tab
# ---------------------------------------------------------------------------

with tab_optimizer:
    st.caption("Upload an Excel workbook, configure parameters, and run the optimizer.")

    st.subheader("1. Upload Input File")

    uploaded_file = st.file_uploader(
        "Upload your input file",
        type=["xlsx", "xls", "csv"],
        help="Select the Excel file containing site data.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        is_csv = uploaded_file.name.lower().endswith(".csv")
        st.session_state["file_bytes"] = file_bytes
        st.session_state["file_name"] = uploaded_file.name
        if is_csv:
            st.session_state["sheet_names"] = None
            st.success(f"File loaded: **{uploaded_file.name}** (CSV)")
        else:
            try:
                xf = pd.ExcelFile(io.BytesIO(file_bytes))
                sheet_names = xf.sheet_names
                st.session_state["sheet_names"] = sheet_names
                st.success(f"File loaded: **{uploaded_file.name}** ({len(sheet_names)} sheet(s) found)")
            except Exception as exc:
                st.session_state["sheet_names"] = None
                st.error(f"Could not parse the uploaded file as a valid Excel workbook: {exc}")
    else:
        st.session_state["file_bytes"] = None
        st.session_state["file_name"] = None
        st.session_state["sheet_names"] = None
        st.info("Upload an Excel or CSV file to get started. The file must contain site data.")

    # Sheet selection — only shown for Excel files (not CSV)
    _file_name = st.session_state.get("file_name") or ""
    _is_csv = _file_name.lower().endswith(".csv")

    if not _is_csv and st.session_state["sheet_names"] is not None:
        sheet_names = st.session_state["sheet_names"]
        default_index = sheet_names.index("Sites") if "Sites" in sheet_names else 0
        selected_sheet = st.selectbox(
            "Select the Sites sheet",
            options=sheet_names,
            index=default_index,
            help="Choose the worksheet that contains site data (default: 'Sites').",
        )
    else:
        selected_sheet = "Sites"

    # --- Parameter Widgets (shared from Hyperparameters tab) ---

    widget_values: dict = {
        "horizon_weeks": int(st.session_state.get("hp_horizon_weeks", 52)),
        "min_batch_produced": int(st.session_state.get("hp_min_batch_produced", 2)),
        "max_batch_produced": int(st.session_state.get("hp_max_batch_produced", 16)),
        "test_discard_per_batch": int(st.session_state.get("hp_test_discard_per_batch", 1)),
        "normal_max_batches": int(st.session_state.get("hp_normal_max_batches", 2)),
        "overtime_max_batches": int(st.session_state.get("hp_overtime_max_batches", 3)),
        "row_cap": int(st.session_state.get("hp_row_cap", 2)),
        "penalty_rate": float(st.session_state.get("hp_penalty_rate", 7000.0)),
        "late_penalty_multiplier": float(st.session_state.get("hp_late_penalty_multiplier", 100.0)),
        "overtime_rate": float(st.session_state.get("hp_overtime_rate", 2000.0)),
        "capacity_rate": float(st.session_state.get("hp_capacity_rate", 15000.0)),
        "w_penalty": float(st.session_state.get("hp_w_penalty", 1.0)),
        "w_overtime": float(st.session_state.get("hp_w_overtime", 1.0)),
        "w_capacity": float(st.session_state.get("hp_w_capacity", 1.0)),
        "shutdown_str": st.session_state.get("hp_shutdown_str", ""),
        "partial_str": st.session_state.get("hp_partial_str", ""),
    }

    # --- Validation feedback and Run button ---
    st.subheader("2. Validate & Run")

    errors, warnings = validate_inputs(
        file_bytes=st.session_state["file_bytes"],
        sheet_name=selected_sheet,
        sheet_names=[selected_sheet] if _is_csv and st.session_state["file_bytes"] else st.session_state["sheet_names"],
        min_batch=widget_values["min_batch_produced"],
        max_batch=widget_values["max_batch_produced"],
        normal_max_batches=widget_values["normal_max_batches"],
        overtime_max_batches=widget_values["overtime_max_batches"],
        w_penalty=widget_values["w_penalty"],
        w_overtime=widget_values["w_overtime"],
        w_capacity=widget_values["w_capacity"],
        horizon_weeks=widget_values["horizon_weeks"],
        shutdown_str=widget_values["shutdown_str"],
        partial_str=widget_values["partial_str"],
    )

    for error in errors:
        st.error(error)
    for warning in warnings:
        st.warning(warning)

    run_clicked = st.button(
        "Run Optimization",
        disabled=bool(errors),
        type="primary",
        help="Fix all validation errors above before running." if errors else "Click to run the optimizer.",
    )

    if run_clicked:
        st.session_state["last_plan_df"] = None
        st.session_state["last_active_df"] = None
        st.session_state["last_issues_df"] = None
        st.session_state["last_summary"] = None
        st.session_state["last_xlsx_bytes"] = None

        with st.spinner("Running optimization — this may take a moment..."):
            try:
                shutdown_weeks, _ = parse_week_list(widget_values["shutdown_str"])
                partial_weeks, _ = parse_week_list(widget_values["partial_str"])
                params = build_params(widget_values)
                plan_df, active_df, issues_df, summary, xlsx_bytes = run_optimizer(
                    file_bytes=st.session_state["file_bytes"],
                    filename=st.session_state.get("file_name", ""),
                    sheet_name=selected_sheet,
                    params=params,
                    shutdown_weeks=shutdown_weeks,
                    partial_weeks=partial_weeks,
                )
            except Exception as exc:
                st.error(f"Optimizer error: {exc}")
                st.stop()

        st.session_state["last_plan_df"] = plan_df
        st.session_state["last_active_df"] = active_df
        st.session_state["last_issues_df"] = issues_df
        st.session_state["last_summary"] = summary
        st.session_state["last_xlsx_bytes"] = xlsx_bytes

    if st.session_state["last_summary"] is not None:
        render_results(
            summary=st.session_state["last_summary"],
            plan_df=st.session_state["last_plan_df"],
            issues_df=st.session_state["last_issues_df"],
            xlsx_bytes=st.session_state["last_xlsx_bytes"],
        )


# ---------------------------------------------------------------------------
# Onboarding Recommendation Tab
# ---------------------------------------------------------------------------

with tab_onboarding:
    st.caption("Determine the optimal start week for onboarding new generators, then generate a full 52-week plan.")

    # --- 1. Sites file upload ---
    st.subheader("1. Upload Existing Sites File")
    ob_file = st.file_uploader(
        "Upload your sites file",
        type=["xlsx", "xls", "csv"],
        help="The same sites Excel file used in the Cost Optimizer tab.",
        key="ob_file_uploader",
    )

    ob_file_bytes = None
    ob_file_name = ""
    ob_sheet_names: list[str] | None = None

    if ob_file is not None:
        ob_file_bytes = ob_file.read()
        ob_file_name = ob_file.name
        if ob_file_name.lower().endswith(".csv"):
            ob_sheet_names = None
            st.success(f"File loaded: **{ob_file_name}** (CSV)")
        else:
            try:
                xf = pd.ExcelFile(io.BytesIO(ob_file_bytes))
                ob_sheet_names = xf.sheet_names
                st.success(f"File loaded: **{ob_file_name}** ({len(ob_sheet_names)} sheet(s))")
            except Exception as exc:
                st.error(f"Could not parse file: {exc}")

    # Sheet selector for Excel files
    if ob_sheet_names is not None:
        ob_default_idx = ob_sheet_names.index("Sites") if "Sites" in ob_sheet_names else 0
        ob_selected_sheet = st.selectbox(
            "Select the Sites sheet",
            options=ob_sheet_names,
            index=ob_default_idx,
            key="ob_sheet_select",
        )
    else:
        ob_selected_sheet = "Sites"

    # --- 2. Add new sites dynamically ---
    st.subheader("2. Define New Sites to Onboard")
    # --- Single new site form ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ob_earliest = st.number_input(
            "Earliest Start Week", min_value=1, max_value=52, value=1, step=1,
            key="ob_site_earliest",
            help="The earliest week the new site can begin onboarding.",
        )
    with col2:
        ob_latest = st.number_input(
            "Must Onboard By", min_value=2, max_value=52, value=10, step=1,
            key="ob_site_latest",
            help="The last week by which the new site must be onboarded.",
        )
    with col3:
        ob_interval = st.number_input(
            "Interval Weeks", min_value=1, max_value=52, value=7, step=1,
            key="ob_site_interval",
            help="How often (in weeks) this site needs a new generator.",
        )
    with col4:
        ob_country = st.text_input(
            "Country", value="usa",
            key="ob_site_country",
            help="Country of the new site (e.g. usa, denmark, uk).",
        ).strip().lower()

    # Build the single-site list (backend still supports multiple)
    new_sites = [{
        "earliest_start": ob_earliest,
        "latest_start": ob_latest,
        "interval_weeks": ob_interval,
        "country": ob_country,
    }]

    # --- 3. Run recommendation ---
    st.subheader("3. Run Recommendation")

    ob_run = st.button("Run Recommendation", type="primary", key="ob_run",
                       disabled=(ob_file_bytes is None))

    if ob_file_bytes is None:
        st.warning("Upload a sites file above — the engine needs existing demand as baseline.")

    if ob_run and ob_file_bytes is not None:
        # Validate the site
        site_errors = []
        s = new_sites[0]
        if s["earliest_start"] >= s["latest_start"]:
            site_errors.append(
                f"Earliest Start ({s['earliest_start']}) "
                f"must be < Must Onboard By ({s['latest_start']})."
            )
        if site_errors:
            for e in site_errors:
                st.error(e)
        else:
            params = _shared_params_from_session()

            # Parse the uploaded sites file to get active_df
            file_like = io.BytesIO(ob_file_bytes)
            if ob_file_name.lower().endswith(".csv"):
                raw_df = pd.read_csv(file_like)
            else:
                raw_df = pd.read_excel(file_like, sheet_name=ob_selected_sheet)
            raw_df = _norm_cols(raw_df)
            active_df, _ = clean_sites(raw_df, params)

            # Read shutdown weeks from shared settings
            _ob_shutdown, _ = parse_week_list(
                st.session_state.get("hp_shutdown_str", ""))
            _ob_partial, _ = parse_week_list(
                st.session_state.get("hp_partial_str", ""))

            s = new_sites[0]
            with st.spinner("Running full optimizer for each candidate week (this may take a moment)..."):
                base_summary, results = evaluate_all_candidates(
                    active_df=active_df,
                    new_sites=new_sites,
                    start_week=s["earliest_start"],
                    end_week=s["latest_start"],
                    params=params,
                    shutdown_weeks=_ob_shutdown,
                    partial_shutdown_weeks=_ob_partial,
                )
                t5 = rank_and_select_top5(results)
                group_results = [{
                    "start": s["earliest_start"], "end": s["latest_start"],
                    "count": 1, "sites": new_sites,
                    "top5": t5, "base_summary": base_summary,
                    "all_results": results,
                }]

            st.session_state["ob_group_results"] = group_results
            st.session_state["ob_params"] = params
            st.session_state["ob_file_bytes_saved"] = ob_file_bytes
            st.session_state["ob_file_name_saved"] = ob_file_name
            st.session_state["ob_sheet_saved"] = ob_selected_sheet
            # Clear old full-plan results
            for k in ["ob_full_plan_df", "ob_full_summary", "ob_full_issues_df",
                       "ob_full_xlsx", "ob_full_selections"]:
                st.session_state[k] = None

    # Display results if available
    if st.session_state.get("ob_group_results") is not None:
        group_results = st.session_state["ob_group_results"]
        params = st.session_state.get("ob_params") or _shared_params_from_session()

        for g_idx, g in enumerate(group_results):
            t5 = g["top5"]
            base_summary = g["base_summary"]
            has_results = any(len(v) > 0 for v in t5.values())

            st.divider()
            st.markdown(
                f"**New site: weeks {g['start']}–{g['end']}**"
            )

            # Show baseline costs
            bl_cols = st.columns(4)
            bl_cols[0].metric("Baseline Penalty", format_cost_thousands(base_summary["total_penalty_cost"]))
            bl_cols[1].metric("Baseline Overtime", format_cost_thousands(base_summary["total_overtime_cost"]))
            bl_cols[2].metric("Baseline Capacity", format_cost_thousands(base_summary["total_capacity_cost"]))
            bl_cols[3].metric("Baseline Composite", format_cost_thousands(
                base_summary["total_penalty_cost"] + base_summary["total_overtime_cost"] + base_summary["total_capacity_cost"]
            ))

            if not has_results:
                st.info("No feasible schedule found for this group.")
                continue

            obj_keys = ["penalty", "overtime", "capacity"]
            sub_tabs = st.tabs([
                "Top 5 by Δ Penalty", "Top 5 by Δ Overtime",
                "Top 5 by Δ Capacity",
            ])

            for sub_tab, obj_key in zip(sub_tabs, obj_keys):
                with sub_tab:
                    options = t5.get(obj_key, [])
                    if not options:
                        st.info(f"No feasible options for {obj_key}.")
                        continue
                    # Build a summary table with one row per candidate
                    table_rows = []
                    for option in options:
                        cw = option["candidate_start_week"]
                        metrics = compute_batch_metrics(option["plan_df"], params)
                        table_rows.append({
                            "Start Week": cw,
                            "Δ Penalty": format_cost_thousands(option["delta_penalty"]),
                            "Δ Overtime": format_cost_thousands(option["delta_overtime"]),
                            "Δ Capacity": format_cost_thousands(option["delta_capacity"]),
                            "Δ Composite": format_cost_thousands(option["delta_composite"]),
                            "OT Weeks": option["overtime_weeks"],
                            "1-Batch Wks": metrics["weeks_1_batch"],
                            "2-Batch Wks": metrics["weeks_2_batch"],
                            "3-Batch Wks": metrics["weeks_3_batch"],
                        })
                    st.dataframe(
                        pd.DataFrame(table_rows),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Start Week": st.column_config.NumberColumn(
                                "Start Week",
                                help="The candidate week to begin onboarding the new site(s).",
                            ),
                            "Δ Penalty": st.column_config.TextColumn(
                                "Δ Penalty",
                                help="Change in early-inventory holding cost vs baseline. Negative means adding the site reduces penalty (new demand absorbs excess inventory).",
                            ),
                            "Δ Overtime": st.column_config.TextColumn(
                                "Δ Overtime",
                                help="Change in overtime cost vs baseline. Positive means more overtime weeks are needed to meet the extra demand.",
                            ),
                            "Δ Capacity": st.column_config.TextColumn(
                                "Δ Capacity",
                                help="Change in unused-capacity cost vs baseline. Negative means the new demand fills idle production slots, reducing waste.",
                            ),
                            "Δ Composite": st.column_config.TextColumn(
                                "Δ Composite",
                                help="Weighted sum of Δ Penalty + Δ Overtime + Δ Capacity. Negative means the overall plan becomes cheaper with the new site.",
                            ),
                            "OT Weeks": st.column_config.NumberColumn(
                                "OT Weeks",
                                help="Total number of overtime weeks (3rd batch used) in the full plan with the new site(s) added.",
                            ),
                            "1-Batch Wks": st.column_config.NumberColumn(
                                "1-Batch Wks",
                                help="Number of weeks where only 1 production batch is run.",
                            ),
                            "2-Batch Wks": st.column_config.NumberColumn(
                                "2-Batch Wks",
                                help="Number of weeks where 2 production batches are run (normal max).",
                            ),
                            "3-Batch Wks": st.column_config.NumberColumn(
                                "3-Batch Wks",
                                help="Number of weeks where 3 production batches are run (overtime).",
                            ),
                        },
                    )

            xlsx = export_recommendation_excel(t5, base_summary, params)
            st.download_button(
                label="Download Recommendation",
                data=xlsx,
                file_name="onboarding_recommendation.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_rec_g{g_idx}",
            )

    # --- 4. Generate Full 52-Week Plan ---
    if st.session_state.get("ob_group_results") is not None:
        st.divider()
        st.subheader("4. Generate Full 52-Week Plan (with new sites)")
        st.caption("Select a candidate start week for each group, then generate the combined plan.")

        group_results = st.session_state["ob_group_results"]
        group_selections: dict[int, int] = {}

        for g_idx, g in enumerate(group_results):
            cands = set()
            for obj_options in g["top5"].values():
                for opt in obj_options:
                    cands.add(opt["candidate_start_week"])
            cand_list = sorted(cands)
            if cand_list:
                sel = st.selectbox(
                    "Select candidate start week for the new site",
                    options=cand_list, key=f"sel_cand_g{g_idx}",
                )
                group_selections[g_idx] = sel
            else:
                st.warning(f"Group {g_idx+1}: No feasible candidates.")

        gen_plan = st.button("Generate Full Plan", type="primary", key="gen_full_plan",
                             disabled=len(group_selections) == 0)

        if gen_plan:
            saved_bytes = st.session_state.get("ob_file_bytes_saved")
            if saved_bytes is None:
                st.error("No sites file. Upload one and re-run the recommendation.")
            else:
                with st.spinner("Running full 52-week optimizer with new sites..."):
                    try:
                        full_params = _shared_params_from_session()
                        saved_name = st.session_state.get("ob_file_name_saved", "")
                        saved_sheet = st.session_state.get("ob_sheet_saved", "Sites")

                        file_like = io.BytesIO(saved_bytes)
                        if saved_name.lower().endswith(".csv"):
                            raw_df = pd.read_csv(file_like)
                        else:
                            raw_df = pd.read_excel(file_like, sheet_name=saved_sheet)
                        raw_df = _norm_cols(raw_df)
                        active_df, _ = clean_sites(raw_df, full_params)

                        _ob_shutdown, _ = parse_week_list(
                            st.session_state.get("hp_shutdown_str", ""))
                        _ob_partial, _ = parse_week_list(
                            st.session_state.get("hp_partial_str", ""))

                        # Build combined demand with all groups' selected candidate weeks
                        from onboarding_recommendation import (
                            add_new_sites_demand, add_new_sites_row_demand,
                        )
                        demand = build_weekly_demand(active_df, full_params)
                        row_demand = build_weekly_row_demand(active_df, full_params)

                        for g_idx, g in enumerate(group_results):
                            if g_idx not in group_selections:
                                continue
                            selected_week = group_selections[g_idx]
                            demand = add_new_sites_demand(
                                demand, g["sites"], selected_week, full_params,
                            )
                            row_demand = add_new_sites_row_demand(
                                row_demand, g["sites"], selected_week, full_params,
                            )

                        plan_df, summary = solve_plan_integrated(
                            demand=demand,
                            shutdown_weeks=_ob_shutdown,
                            partial_shutdown_weeks=_ob_partial,
                            row_demand=row_demand,
                            row_cap=full_params.row_cap,
                            params=full_params,
                        )

                        # Build issues_df from combined sites for export
                        all_new_rows = []
                        site_counter = 0
                        for g_idx, g in enumerate(group_results):
                            if g_idx not in group_selections:
                                continue
                            selected_week = group_selections[g_idx]
                            for s in g["sites"]:
                                site_counter += 1
                                all_new_rows.append({
                                    "site_id": f"NEW_{site_counter:04d}",
                                    "active": "Y",
                                    "next_demand_week": selected_week,
                                    "interval_weeks": s["interval_weeks"],
                                    "country": s["country"],
                                })
                        if all_new_rows:
                            new_df = pd.DataFrame(all_new_rows)
                            combined_raw = pd.concat([raw_df, new_df], ignore_index=True)
                        else:
                            combined_raw = raw_df
                        combined_active, issues_df = clean_sites(combined_raw, full_params)

                        full_xlsx = export_excel_bytes(
                            plan_df, combined_active, issues_df, full_params, summary,
                        )

                        st.session_state["ob_full_plan_df"] = plan_df
                        st.session_state["ob_full_summary"] = summary
                        st.session_state["ob_full_issues_df"] = issues_df
                        st.session_state["ob_full_xlsx"] = full_xlsx

                    except Exception as exc:
                        st.error(f"Optimizer error: {exc}")
                        st.stop()

        # Display full plan results
        if st.session_state.get("ob_full_summary") is not None:
            summary = st.session_state["ob_full_summary"]
            plan_df = st.session_state["ob_full_plan_df"]
            issues_df = st.session_state["ob_full_issues_df"]
            full_xlsx = st.session_state["ob_full_xlsx"]

            st.markdown("**Full 52-week plan with new sites included**")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Cost", f"${summary.get('total_composite_cost', 0):,.0f}")
            c2.metric("Penalty Cost", f"${summary.get('total_penalty_cost', 0):,.0f}")
            c3.metric("Overtime Cost", f"${summary.get('total_overtime_cost', 0):,.0f}")
            c4.metric("Capacity Cost", f"${summary.get('total_capacity_cost', 0):,.0f}")
            c5.metric("Overtime Weeks", summary.get("overtime_weeks", 0))

            st.dataframe(plan_df, use_container_width=True)

            with st.expander(f"Data Quality Issues ({len(issues_df)} row(s))", expanded=False):
                if issues_df.empty:
                    st.info("No data quality issues found.")
                else:
                    st.dataframe(issues_df, use_container_width=True)

            st.download_button(
                label="Download Full Plan (with new sites)",
                data=full_xlsx,
                file_name="full_plan_with_onboarding.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_full_plan",
            )

# ---------------------------------------------------------------------------
# Hyperparameters Tab (shared by Cost Optimizer & Onboarding)
# ---------------------------------------------------------------------------

with tab_hyperparams:
    st.caption(
        "Configure hyperparameters used by both the Cost Optimizer and the "
        "Onboarding Recommendation engine."
    )

    with st.expander("Production Constraints", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.number_input(
                "Horizon Weeks", min_value=1, max_value=104, value=52, step=1,
                help="Planning horizon in weeks (1–104).",
                key="hp_horizon_weeks",
            )
            st.number_input(
                "Min Batch Produced", min_value=1, value=2, step=1,
                help="Minimum units produced per batch.",
                key="hp_min_batch_produced",
            )
            st.number_input(
                "Max Batch Produced", min_value=1, value=16, step=1,
                help="Maximum units produced per batch.",
                key="hp_max_batch_produced",
            )

        with col2:
            st.number_input(
                "Test Discard per Batch", min_value=0, value=1, step=1,
                help="Units discarded for testing per batch.",
                key="hp_test_discard_per_batch",
            )
            st.number_input(
                "Normal Max Batches", min_value=1, value=2, step=1,
                help="Maximum batches in a normal (non-overtime) week.",
                key="hp_normal_max_batches",
            )
            st.number_input(
                "Overtime Max Batches", min_value=1, value=3, step=1,
                help="Maximum batches in an overtime week.",
                key="hp_overtime_max_batches",
            )

        with col3:
            st.number_input(
                "Row Cap", min_value=0, value=2, step=1,
                help="Maximum ROW units fulfilled per week.",
                key="hp_row_cap",
            )

    with st.expander("Cost Rates", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.number_input(
                "Penalty Rate (USD / unit-week)", min_value=0.0, value=7000.0,
                step=100.0, format="%.2f",
                help="Cost per unit per week of early inventory.",
                key="hp_penalty_rate",
            )
            st.number_input(
                "Late Penalty Multiplier", min_value=1.0, value=100.0,
                step=1.0, format="%.2f",
                help="Multiplier applied to penalty_rate for backlog.",
                key="hp_late_penalty_multiplier",
            )

        with col2:
            st.number_input(
                "Overtime Rate (USD / overtime week)", min_value=0.0, value=2000.0,
                step=100.0, format="%.2f",
                help="Cost per overtime week (3rd batch).",
                key="hp_overtime_rate",
            )
            st.number_input(
                "Capacity Rate (USD / unused slot / week)", min_value=0.0, value=15000.0,
                step=100.0, format="%.2f",
                help="Cost per unused good unit slot per week.",
                key="hp_capacity_rate",
            )

        _hp_penalty = float(st.session_state.get("hp_penalty_rate", 7000.0))
        _hp_late_mult = float(st.session_state.get("hp_late_penalty_multiplier", 100.0))
        st.metric(
            label="Late Penalty Rate (derived, read-only)",
            value=f"${_hp_penalty * _hp_late_mult:,.2f}",
            help="Computed as penalty_rate × late_penalty_multiplier.",
        )

    with st.expander("Objective Weights", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.slider(
                "w_penalty", min_value=0.0, max_value=1.0, value=1.0, step=0.05,
                help="Weight for the penalty cost component.",
                key="hp_w_penalty",
            )
        with col2:
            st.slider(
                "w_overtime", min_value=0.0, max_value=1.0, value=1.0, step=0.05,
                help="Weight for the overtime cost component.",
                key="hp_w_overtime",
            )
        with col3:
            st.slider(
                "w_capacity", min_value=0.0, max_value=1.0, value=1.0, step=0.05,
                help="Weight for the capacity utilization cost component.",
                key="hp_w_capacity",
            )

    with st.expander("Shutdown Weeks", expanded=False):
        st.text_input(
            "Shutdown Weeks", value="", placeholder="e.g. 1,2,3",
            help="Comma-separated list of week numbers with full production shutdown.",
            key="hp_shutdown_str",
        )
        st.text_input(
            "Partial Shutdown Weeks", value="", placeholder="e.g. 4,5",
            help="Comma-separated list of week numbers with partial production shutdown.",
            key="hp_partial_str",
        )
