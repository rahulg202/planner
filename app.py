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
]:
    if key not in st.session_state:
        st.session_state[key] = None

st.title("Production Cost Optimizer")
st.caption("Upload an Excel workbook, configure parameters, and run the optimizer.")


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
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    result: list[int] = []
    for token in tokens:
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


# ---------------------------------------------------------------------------
# File Upload and Sheet Selection UI (Requirements 1, 2)
# ---------------------------------------------------------------------------

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
    # Clear stale file state when uploader is cleared
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
    selected_sheet = "Sites"  # unused for CSV but kept as a safe default


# ---------------------------------------------------------------------------
# Parameter Widgets (Requirements 3, 4, 5, 6)
# ---------------------------------------------------------------------------

st.subheader("2. Configure Parameters")

# --- Production Constraint Parameters (Requirement 3) ---
with st.expander("Production Constraints", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        horizon_weeks = st.number_input(
            "Horizon Weeks",
            min_value=1,
            max_value=104,
            value=52,
            step=1,
            help="Planning horizon in weeks (1–104).",
        )
        min_batch_produced = st.number_input(
            "Min Batch Produced",
            min_value=1,
            value=2,
            step=1,
            help="Minimum units produced per batch.",
        )
        max_batch_produced = st.number_input(
            "Max Batch Produced",
            min_value=1,
            value=16,
            step=1,
            help="Maximum units produced per batch.",
        )

    with col2:
        test_discard_per_batch = st.number_input(
            "Test Discard per Batch",
            min_value=0,
            value=1,
            step=1,
            help="Units discarded for testing per batch.",
        )
        normal_max_batches = st.number_input(
            "Normal Max Batches",
            min_value=1,
            value=2,
            step=1,
            help="Maximum batches in a normal (non-overtime) week.",
        )
        overtime_max_batches = st.number_input(
            "Overtime Max Batches",
            min_value=1,
            value=3,
            step=1,
            help="Maximum batches in an overtime week.",
        )

    with col3:
        row_cap = st.number_input(
            "Row Cap",
            min_value=0,
            value=2,
            step=1,
            help="Maximum ROW units fulfilled per week.",
        )

# --- Cost Rate Parameters (Requirement 4) ---
with st.expander("Cost Rates", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        penalty_rate = st.number_input(
            "Penalty Rate (USD / unit-week)",
            min_value=0.0,
            value=7000.0,
            step=100.0,
            format="%.2f",
            help="Cost per unit per week of early inventory.",
        )
        late_penalty_multiplier = st.number_input(
            "Late Penalty Multiplier",
            min_value=1.0,
            value=100.0,
            step=1.0,
            format="%.2f",
            help="Multiplier applied to penalty_rate for backlog.",
        )

    with col2:
        overtime_rate = st.number_input(
            "Overtime Rate (USD / overtime week)",
            min_value=0.0,
            value=2000.0,
            step=100.0,
            format="%.2f",
            help="Cost per overtime week (3rd batch).",
        )
        capacity_rate = st.number_input(
            "Capacity Rate (USD / unused slot / week)",
            min_value=0.0,
            value=15000.0,
            step=100.0,
            format="%.2f",
            help="Cost per unused good unit slot per week.",
        )

    # Derived read-only field (Requirement 4.5)
    late_penalty_rate = penalty_rate * late_penalty_multiplier
    st.metric(
        label="Late Penalty Rate (derived, read-only)",
        value=f"${late_penalty_rate:,.2f}",
        help="Computed as penalty_rate × late_penalty_multiplier.",
    )

# --- Objective Weight Parameters (Requirement 5) ---
with st.expander("Objective Weights", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        w_penalty = st.slider(
            "w_penalty",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="Weight for the penalty cost component.",
        )

    with col2:
        w_overtime = st.slider(
            "w_overtime",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="Weight for the overtime cost component.",
        )

    with col3:
        w_capacity = st.slider(
            "w_capacity",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="Weight for the capacity utilization cost component.",
        )

# --- Shutdown Week Configuration (Requirement 6) ---
with st.expander("Shutdown Weeks", expanded=False):
    shutdown_str = st.text_input(
        "Shutdown Weeks",
        value="",
        placeholder="e.g. 1,2,3",
        help="Comma-separated list of week numbers with full production shutdown.",
    )
    partial_str = st.text_input(
        "Partial Shutdown Weeks",
        value="",
        placeholder="e.g. 4,5",
        help="Comma-separated list of week numbers with partial production shutdown.",
    )

# Collect all widget values into a dict for downstream use
widget_values: dict = {
    "horizon_weeks": int(horizon_weeks),
    "min_batch_produced": int(min_batch_produced),
    "max_batch_produced": int(max_batch_produced),
    "test_discard_per_batch": int(test_discard_per_batch),
    "normal_max_batches": int(normal_max_batches),
    "overtime_max_batches": int(overtime_max_batches),
    "row_cap": int(row_cap),
    "penalty_rate": float(penalty_rate),
    "late_penalty_multiplier": float(late_penalty_multiplier),
    "overtime_rate": float(overtime_rate),
    "capacity_rate": float(capacity_rate),
    "w_penalty": float(w_penalty),
    "w_overtime": float(w_overtime),
    "w_capacity": float(w_capacity),
    "shutdown_str": shutdown_str,
    "partial_str": partial_str,
}


# ---------------------------------------------------------------------------
# Validation feedback and Run button (Requirements 7, 10)
# ---------------------------------------------------------------------------

st.subheader("3. Validate & Run")

errors, warnings = validate_inputs(
    file_bytes=st.session_state["file_bytes"],
    sheet_name=selected_sheet,
    # For CSV files there are no sheets — skip the sheet-existence check
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


# ---------------------------------------------------------------------------
# Run logic and results display (Requirements 7, 8, 9)
# ---------------------------------------------------------------------------

if run_clicked:
    # 6.1 Clear previous results from session state
    st.session_state["last_plan_df"] = None
    st.session_state["last_active_df"] = None
    st.session_state["last_issues_df"] = None
    st.session_state["last_summary"] = None
    st.session_state["last_xlsx_bytes"] = None

    # 6.2 Run optimizer inside spinner; surface exceptions via st.error
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

    # 6.3 Store results in session state
    st.session_state["last_plan_df"] = plan_df
    st.session_state["last_active_df"] = active_df
    st.session_state["last_issues_df"] = issues_df
    st.session_state["last_summary"] = summary
    st.session_state["last_xlsx_bytes"] = xlsx_bytes


def render_results(
    summary: dict,
    plan_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    xlsx_bytes: bytes,
) -> None:
    """Render metric cards, plan table, issues section, and download button."""
    st.subheader("4. Results")

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


# Render results if a successful run exists in session state
if st.session_state["last_summary"] is not None:
    render_results(
        summary=st.session_state["last_summary"],
        plan_df=st.session_state["last_plan_df"],
        issues_df=st.session_state["last_issues_df"],
        xlsx_bytes=st.session_state["last_xlsx_bytes"],
    )
