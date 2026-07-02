from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# Financial Analysis Panel - SOFP / SOCI Flux Analysis
# Version 13 online repository data + SOCI by PL Sort + no point-click
# Built around Ryan's standard JE database columns:
#   BS Sort, PL Sort, BS Section A, BS Section B,
#   BS Report FSLI, PL Section, FSLI
# ============================================================

st.set_page_config(page_title="Financial Analysis Panel", page_icon="📊", layout="wide")

# -----------------------------
# Online / repository data setup
# -----------------------------
APP_DIR = Path(__file__).resolve().parent

# Put your sanitized standard JE database beside this .py file in GitHub.
# You may change this default name below, or type another file name in the sidebar.
DEFAULT_REPO_DATA_FILE = "Interactive Project Data 01 vSafe.xlsx"

SUPPORTED_DATA_EXTENSIONS = [".xlsx", ".xlsm", ".xls", ".csv"]

# -----------------------------
# Standard database columns
# -----------------------------
COL_BS_SORT = "BS Sort"
COL_PL_SORT = "PL Sort"
COL_JE = "JE Number"
COL_DATE = "Date"
COL_ACCOUNT = "Account Name CN"
COL_CURRENCY = "Currency"
COL_AMOUNT = "Amount"
COL_EXC = "Exc"
COL_RMB = "RMB Amount"
COL_ABSTRACT = "Abstract"
COL_DIMENSION = "Dimension"
COL_CF_SANKEY = "CF Sankey B"

COL_BS_SECTION_A = "BS Section A"
COL_BS_SECTION_B = "BS Section B"
COL_BS_FSLI = "BS Report FSLI"

COL_PL_SECTION = "PL Section"
COL_ACCOUNT_PROPERTY = "Account Property"
COL_FSLI = "FSLI"

DETAIL_COLS = [
    COL_DATE,
    COL_ACCOUNT,
    COL_CURRENCY,
    COL_AMOUNT,
    COL_EXC,
    COL_RMB,
    COL_ABSTRACT,
    COL_DIMENSION,
    COL_CF_SANKEY,
]

AMOUNT_EPS = 1e-9

# Office-like chart colors
OFFICE_BLUE = "#4472C4"
OFFICE_ORANGE = "#ED7D31"
OFFICE_GREY = "#A5A5A5"
OFFICE_GREEN = "#70AD47"
OFFICE_YELLOW = "#FFC000"
OFFICE_LIGHT_BLUE = "#5B9BD5"
OFFICE_PALETTE = [OFFICE_BLUE, OFFICE_ORANGE, OFFICE_GREY, OFFICE_GREEN, OFFICE_YELLOW, OFFICE_LIGHT_BLUE]

DEFAULT_DISPLAY_ROWS = 300
DEFAULT_SCATTER_POINTS = 5000

# -----------------------------
# General helpers
# -----------------------------
def clean_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace({"nan": "", "None": "", "NaT": ""})


def required_cols_missing(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c not in df.columns]


def safe_pct(numerator: float, denominator: float) -> float:
    if pd.isna(denominator) or abs(float(denominator)) < AMOUNT_EPS:
        return np.nan
    return float(numerator) / float(denominator)


def fmt_amount(x) -> str:
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)


def fmt_pct(x) -> str:
    if pd.isna(x):
        return "N/A"
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "N/A"


def parse_month_label(label: str) -> pd.Timestamp:
    return pd.to_datetime(label + "-01")


def month_end(label: str) -> pd.Timestamp:
    return parse_month_label(label) + pd.offsets.MonthEnd(0)


def normalize_key(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
        .str.replace("&", "and", regex=False)
    )


def natural_sort_value(x):
    if pd.isna(x) or str(x).strip() == "":
        return 999999999
    try:
        return float(x)
    except Exception:
        return str(x)


def normalize_sort_code(value) -> str:
    """Normalize report sort code such as 6001 or 6001.0 into a stable text key."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass
    # Keep non-numeric sort keys as-is, but remove a trailing .0 from Excel imports.
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def add_row_role(df: pd.DataFrame) -> pd.DataFrame:
    """Marks row 1 inside each JE Number as Debit and row 2 as Credit.
    For JE groups with more than 2 rows, rows 3+ are tagged Other.
    """
    out = df.copy()
    out["__row_order"] = np.arange(len(out))
    if COL_JE in out.columns:
        out["__je_row_no"] = out.groupby(out[COL_JE].astype(str), sort=False).cumcount() + 1
    else:
        out["__je_row_no"] = 1
    out["__side"] = np.select(
        [out["__je_row_no"].eq(1), out["__je_row_no"].eq(2)],
        ["Debit", "Credit"],
        default="Other",
    )
    return out


def standardize_database(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning rules no matter whether data comes from upload or repo file."""
    # Trim column headers but keep original names after trimming.
    df.columns = [str(c).strip() for c in df.columns]

    if COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    if COL_RMB in df.columns:
        df[COL_RMB] = pd.to_numeric(df[COL_RMB], errors="coerce").fillna(0.0)
    if COL_AMOUNT in df.columns:
        df[COL_AMOUNT] = pd.to_numeric(df[COL_AMOUNT], errors="coerce")

    for c in [
        COL_ACCOUNT,
        COL_BS_SECTION_A,
        COL_BS_SECTION_B,
        COL_BS_FSLI,
        COL_PL_SECTION,
        COL_ACCOUNT_PROPERTY,
        COL_FSLI,
        COL_CURRENCY,
        COL_ABSTRACT,
        COL_DIMENSION,
        COL_CF_SANKEY,
        COL_JE,
    ]:
        if c in df.columns:
            df[c] = clean_text_series(df[c])

    df = add_row_role(df)
    return df


def read_database_file(file_obj_or_path, suffix: str) -> pd.DataFrame:
    suffix = suffix.lower()
    if suffix in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(file_obj_or_path)
    if suffix == ".csv":
        return pd.read_csv(file_obj_or_path)
    raise ValueError("Please use xlsx, xlsm, xls, or csv file.")


@st.cache_data(show_spinner=False)
def load_database_from_upload(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    df = read_database_file(uploaded_file, suffix)
    return standardize_database(df)


@st.cache_data(show_spinner=False)
def load_database_from_repo(file_path: str, file_mtime: float) -> pd.DataFrame:
    """Load data committed beside the app file.

    file_mtime is included only to refresh Streamlit cache after the data file changes.
    """
    path = Path(file_path)
    df = read_database_file(path, path.suffix.lower())
    return standardize_database(df)


def find_default_repo_data_file() -> Optional[Path]:
    preferred = APP_DIR / DEFAULT_REPO_DATA_FILE
    if preferred.exists() and preferred.suffix.lower() in SUPPORTED_DATA_EXTENSIONS:
        return preferred

    candidates = []
    for ext in SUPPORTED_DATA_EXTENSIONS:
        candidates.extend(APP_DIR.glob(f"*{ext}"))

    # Avoid accidentally reading temporary Excel lock files.
    candidates = [p for p in candidates if not p.name.startswith("~$")]
    if not candidates:
        return None

    # Prefer obvious data file names.
    priority_words = ["standard", "database", "je", "data", "analysis", "sanitized"]
    def score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        return (-sum(word in name for word in priority_words), name)

    return sorted(candidates, key=score)[0]


# -----------------------------
# Display / styling helpers
# -----------------------------
def prepare_display_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Return a display-safe copy. Keeping dataframe native is much faster than heavy Styler bars."""
    out = df.copy()
    if max_rows is not None and len(out) > max_rows:
        out = out.head(max_rows).copy()
    return out


def show_amount_dataframe(
    df: pd.DataFrame,
    amount_cols: list[str],
    pct_cols: Optional[list[str]] = None,
    max_rows: Optional[int] = None,
    caption_if_truncated: str = "",
):
    pct_cols = pct_cols or []
    display = prepare_display_table(df, max_rows=max_rows)
    column_config = {}
    for c in amount_cols:
        if c in display.columns:
            column_config[c] = st.column_config.NumberColumn(c, format="%,.2f")
    for c in pct_cols:
        if c in display.columns:
            column_config[c] = st.column_config.NumberColumn(c, format="%.2f%%")
            # Convert ratio to percentage points for Streamlit formatting.
            display[c] = display[c] * 100
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )
    if max_rows is not None and len(df) > max_rows:
        st.caption(caption_if_truncated or f"Showing first {max_rows:,} rows out of {len(df):,} rows for performance.")


def top_abs_rows(df: pd.DataFrame, value_col: str, n: int) -> pd.DataFrame:
    if df.empty or value_col not in df.columns or len(df) <= n:
        return df
    return df.sort_values(value_col, key=lambda s: pd.to_numeric(s, errors="coerce").abs(), ascending=False).head(n).copy()

# -----------------------------
# Outlier helpers
# -----------------------------
def outlier_iqr_thresholds(series: pd.Series) -> dict:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    result = {
        "n": int(len(valid)),
        "q1": np.nan,
        "q3": np.nan,
        "iqr": np.nan,
        "lower": np.nan,
        "upper": np.nan,
        "rule_applied": False,
        "reason": "Need at least 8 valid points to apply the IQR rule.",
    }
    if len(valid) < 8:
        return result
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    result.update({"q1": q1, "q3": q3, "iqr": iqr})
    if pd.isna(iqr) or abs(iqr) < AMOUNT_EPS:
        result["reason"] = "IQR is zero, so the data is too concentrated to identify statistical outliers."
        return result
    result.update(
        {
            "lower": q1 - 1.5 * iqr,
            "upper": q3 + 1.5 * iqr,
            "rule_applied": True,
            "reason": "Applied 1.5 × IQR rule.",
        }
    )
    return result


def identify_outliers_iqr(series: pd.Series) -> pd.Series:
    thresholds = outlier_iqr_thresholds(series)
    if not thresholds["rule_applied"]:
        return pd.Series(False, index=series.index)
    values = pd.to_numeric(series, errors="coerce")
    return (values < thresholds["lower"]) | (values > thresholds["upper"])


def show_outlier_rule_window(plot_df: pd.DataFrame, y_col: str, key: str):
    with st.expander("How outliers are identified", expanded=False):
        st.write(
            "Outliers are flagged using the 1.5 × IQR rule on `RMB Amount`. "
            "The rule is applied only when there are at least 8 valid transaction points."
        )
        thresholds = outlier_iqr_thresholds(plot_df[y_col] if y_col in plot_df.columns else pd.Series(dtype=float))
        summary = pd.DataFrame(
            [
                {"Item": "Valid points", "Value": thresholds["n"]},
                {"Item": "Q1", "Value": thresholds["q1"]},
                {"Item": "Q3", "Value": thresholds["q3"]},
                {"Item": "IQR", "Value": thresholds["iqr"]},
                {"Item": "Lower bound", "Value": thresholds["lower"]},
                {"Item": "Upper bound", "Value": thresholds["upper"]},
                {"Item": "Rule status", "Value": thresholds["reason"]},
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)


# -----------------------------
# JE detail and chart helpers
# -----------------------------
def show_je_detail(df: pd.DataFrame, je_number: str):
    if not je_number:
        return
    detail = df[df[COL_JE].astype(str) == str(je_number)].sort_values("__row_order")
    if detail.empty:
        st.info("No JE detail found for the selected point.")
        return
    st.markdown(f"#### Original JE group: `{je_number}`")
    cols = [c for c in DETAIL_COLS if c in detail.columns]
    column_config = {}
    for c in [COL_RMB, COL_AMOUNT, COL_EXC]:
        if c in detail.columns:
            column_config[c] = st.column_config.NumberColumn(c, format="%,.2f")
    st.dataframe(
        detail[cols],
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )


def scatter_and_select(
    plot_df: pd.DataFrame,
    title: str,
    x_col: str,
    y_col: str = COL_RMB,
    color_col: Optional[str] = None,
    hover_cols: Optional[list[str]] = None,
    key: str = "scatter",
    max_points: int = DEFAULT_SCATTER_POINTS,
    chart_theme: str = "Dark Office",
) -> Optional[str]:
    """Smooth, colored scatter with fast JE selector.

    Point-click events are intentionally removed. streamlit-plotly-events triggers
    full reruns and can distort displayed values on large transaction datasets.
    The chart is now view-only, and JE inspection is handled by a stable selector.
    """
    if plot_df.empty:
        st.info("No transaction found for this selection.")
        show_outlier_rule_window(plot_df, y_col=y_col, key=key + "_rule")
        return None

    hover_cols = hover_cols or []
    plot_df = plot_df.copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df[plot_df[x_col].notna() & plot_df[y_col].notna()].copy()
    if plot_df.empty:
        st.info("No valid transaction amount/date found for the scatter chart.")
        show_outlier_rule_window(plot_df, y_col=y_col, key=key + "_rule")
        return None

    plot_df["Outlier"] = identify_outliers_iqr(plot_df[y_col]).map({True: "Outlier", False: "Normal"})

    original_points = len(plot_df)
    if max_points and original_points > max_points:
        outliers = plot_df[plot_df["Outlier"].eq("Outlier")].copy()
        normal = plot_df[plot_df["Outlier"].ne("Outlier")].copy()
        keep_normal_n = max(0, max_points - len(outliers))
        if len(normal) > keep_normal_n:
            normal = normal.sample(keep_normal_n, random_state=42)
        plot_df = pd.concat([outliers, normal], ignore_index=True)
        st.caption(f"Scatter sampled to {len(plot_df):,} points from {original_points:,} transactions. All outliers are retained.")

    if color_col and color_col in plot_df.columns:
        color = color_col
        color_map = None
    else:
        color = "Outlier"
        color_map = {"Normal": OFFICE_BLUE, "Outlier": OFFICE_ORANGE}

    hover_data = [c for c in hover_cols if c in plot_df.columns]

    dark = chart_theme.startswith("Dark")
    template = "plotly_dark" if dark else "plotly_white"
    paper_bg = "#0E1117" if dark else "white"
    plot_bg = "#0E1117" if dark else "white"
    font_color = "#FAFAFA" if dark else "#262730"
    grid_color = "rgba(250,250,250,0.16)" if dark else "rgba(0,0,0,0.10)"
    zero_color = "rgba(250,250,250,0.45)" if dark else "rgba(0,0,0,0.35)"
    marker_line = "rgba(255,255,255,0.85)" if dark else "white"

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color,
        color_discrete_sequence=OFFICE_PALETTE,
        color_discrete_map=color_map,
        hover_data=hover_data,
        title=title,
        render_mode="webgl",
        template=template,
    )
    fig.update_traces(
        marker={"size": 9, "opacity": 0.82, "line": {"width": 0.7, "color": marker_line}},
        selector={"mode": "markers"},
    )
    fig.update_layout(
        height=520,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend_title_text="",
        hovermode="closest",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font={"color": font_color},
        title={"x": 0.02, "xanchor": "left"},
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor=zero_color)
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor=zero_color)

    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": True, "scrollZoom": True})

    outlier_jes = plot_df.loc[plot_df["Outlier"].eq("Outlier"), COL_JE].astype(str).drop_duplicates().tolist()
    top_jes = (
        plot_df.sort_values(y_col, key=lambda s: s.abs(), ascending=False)[COL_JE]
        .astype(str)
        .drop_duplicates()
        .head(100)
        .tolist()
    )
    candidates = []
    for je in outlier_jes + top_jes:
        if je and je not in candidates:
            candidates.append(je)

    selected_je = None
    if candidates:
        selected_je = st.selectbox(
            "Inspect JE group",
            [""] + candidates,
            key=key + "_fast_je",
            help="Outlier JEs and top absolute-amount JEs are listed first.",
        )

    show_outlier_rule_window(plot_df, y_col=y_col, key=key + "_rule")
    return selected_je


# -----------------------------
# SOFP calculations
# -----------------------------
@st.cache_data(show_spinner=False)
def sofp_base(df: pd.DataFrame) -> pd.DataFrame:
    needed = [COL_DATE, COL_RMB, COL_BS_SORT, COL_BS_SECTION_A, COL_BS_SECTION_B, COL_BS_FSLI]
    missing = required_cols_missing(df, needed)
    if missing:
        st.error("SOFP missing required columns: " + ", ".join(missing))
        return pd.DataFrame()
    base = df.copy()
    base = base[base[COL_DATE].notna()].copy()
    base = base[clean_text_series(base[COL_BS_FSLI]).ne("")].copy()
    base = base[clean_text_series(base[COL_BS_SECTION_A]).ne("")].copy()
    return base


@st.cache_data(show_spinner=False)
def build_sofp_statement(df: pd.DataFrame, date1: pd.Timestamp, date2: pd.Timestamp) -> pd.DataFrame:
    base = sofp_base(df)
    if base.empty:
        return pd.DataFrame()

    b1 = base[base[COL_DATE] <= date1].groupby(COL_BS_FSLI, dropna=False)[COL_RMB].sum()
    b2 = base[base[COL_DATE] <= date2].groupby(COL_BS_FSLI, dropna=False)[COL_RMB].sum()

    structure = (
        base[[COL_BS_SORT, COL_BS_SECTION_A, COL_BS_SECTION_B, COL_BS_FSLI]]
        .drop_duplicates()
        .copy()
    )
    structure["__sort"] = structure[COL_BS_SORT].map(natural_sort_value)
    structure = structure.sort_values([COL_BS_SECTION_A, COL_BS_SECTION_B, "__sort", COL_BS_FSLI])

    rows = []
    for section_a, dfa in structure.groupby(COL_BS_SECTION_A, sort=False):
        sec_fslis = dfa[COL_BS_FSLI].dropna().unique().tolist()
        sec_b1 = float(b1.reindex(sec_fslis).fillna(0).sum())
        sec_b2 = float(b2.reindex(sec_fslis).fillna(0).sum())
        rows.append({
            "Statement of Financial Position": section_a,
            "Date 1 Balance": sec_b1,
            "Date 2 Balance": sec_b2,
            "Flux": sec_b2 - sec_b1,
            "Flux %": safe_pct(sec_b2 - sec_b1, sec_b1),
            "__level": 0,
            "__fsli": "",
        })
        for section_b, dfb in dfa.groupby(COL_BS_SECTION_B, sort=False):
            sub_fslis = dfb[COL_BS_FSLI].dropna().unique().tolist()
            sub_b1 = float(b1.reindex(sub_fslis).fillna(0).sum())
            sub_b2 = float(b2.reindex(sub_fslis).fillna(0).sum())
            rows.append({
                "Statement of Financial Position": "   " + section_b,
                "Date 1 Balance": sub_b1,
                "Date 2 Balance": sub_b2,
                "Flux": sub_b2 - sub_b1,
                "Flux %": safe_pct(sub_b2 - sub_b1, sub_b1),
                "__level": 1,
                "__fsli": "",
            })
            for _, r in dfb.sort_values(["__sort", COL_BS_FSLI]).iterrows():
                fsli = r[COL_BS_FSLI]
                v1 = float(b1.get(fsli, 0.0))
                v2 = float(b2.get(fsli, 0.0))
                rows.append({
                    "Statement of Financial Position": "      " + str(fsli),
                    "Date 1 Balance": v1,
                    "Date 2 Balance": v2,
                    "Flux": v2 - v1,
                    "Flux %": safe_pct(v2 - v1, v1),
                    "__level": 2,
                    "__fsli": fsli,
                })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def sofp_account_bridge(df: pd.DataFrame, fsli: str, date1: pd.Timestamp, date2: pd.Timestamp) -> pd.DataFrame:
    base = sofp_base(df)
    if base.empty:
        return pd.DataFrame()
    base = base[base[COL_BS_FSLI].eq(fsli)].copy()
    if base.empty:
        return pd.DataFrame()

    bal1 = base[base[COL_DATE] <= date1].groupby(COL_ACCOUNT, dropna=False)[COL_RMB].sum()
    bal2 = base[base[COL_DATE] <= date2].groupby(COL_ACCOUNT, dropna=False)[COL_RMB].sum()
    movement = base[(base[COL_DATE] > date1) & (base[COL_DATE] <= date2)].copy()
    debit = movement[movement["__side"].eq("Debit")].groupby(COL_ACCOUNT, dropna=False)[COL_RMB].sum()
    credit = movement[movement["__side"].eq("Credit")].groupby(COL_ACCOUNT, dropna=False)[COL_RMB].sum()

    accounts = sorted(set(bal1.index).union(bal2.index).union(debit.index).union(credit.index), key=str)
    out = pd.DataFrame({COL_ACCOUNT: accounts})
    out["Date 1 Balance"] = out[COL_ACCOUNT].map(bal1).fillna(0.0)
    out["Debit Amount"] = out[COL_ACCOUNT].map(debit).fillna(0.0)
    out["Credit Amount"] = out[COL_ACCOUNT].map(credit).fillna(0.0)
    out["Date 2 Balance"] = out[COL_ACCOUNT].map(bal2).fillna(0.0)
    out["Check Difference"] = out["Date 1 Balance"] + out["Debit Amount"] + out["Credit Amount"] - out["Date 2 Balance"]
    return out.sort_values("Date 2 Balance", key=lambda s: s.abs(), ascending=False)



# -----------------------------
# SOCI custom structure helpers
# -----------------------------
# The online/sanitized database may anonymize FSLI names.
# Therefore, SOCI is structured and calculated by PL Sort code,
# while the displayed line name is read dynamically from the current FSLI column.
SOCI_STRUCTURE = [
    {
        "section": "Revenues",
        "line_sorts": ["6001", "6002", "6003"],
        "subtotal": "Total revenues",
        "subtotal_key": "total_revenues",
    },
    {
        "section": "Costs",
        "line_sorts": ["6401", "6402", "6403", "6404", "6405", "6406"],
        "subtotal": "Total costs",
        "subtotal_key": "total_costs",
    },
    {
        "calculation": "Gross profit/(loss)",
        "key": "gross_profit",
        "formula_keys": ["total_revenues", "total_costs"],
    },
    {
        "section": "Expenses",
        "line_sorts": ["6601", "6602", "6603", "6604"],
        "subtotal": "Total expenses",
        "subtotal_key": "total_expenses",
    },
    {
        "calculation": "Operating profit/(loss)",
        "key": "operating_profit",
        "formula_keys": ["gross_profit", "total_expenses"],
    },
    {
        "section": "Other income/(expenses)",
        "line_sorts": ["6701", "6702", "6703", "6704"],
        "subtotal": None,
    },
    {
        "calculation": "Net income/(loss)",
        "key": "net_income",
        "formula_keys": ["operating_profit", "6701", "6702", "6703", "6704"],
    },
    {
        "section": "Other comprehensive income/(loss)",
        "line_sorts": ["7001", "7002"],
        "subtotal": None,
    },
    {
        "calculation": "Comprehensive income/(loss)",
        "key": "comprehensive_income",
        "formula_keys": ["net_income", "7001", "7002"],
    },
    {"rate": "Gross rate", "numerator": "gross_profit", "denominator": "total_revenues"},
    {"rate": "Operating rate", "numerator": "operating_profit", "denominator": "total_revenues"},
    {"rate": "Net rate", "numerator": "net_income", "denominator": "total_revenues"},
]


def safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(denominator) or abs(float(denominator)) < AMOUNT_EPS:
        return np.nan
    return float(numerator) / float(denominator)


def fmt_mixed_soci_value(x, is_rate: bool = False) -> str:
    if pd.isna(x):
        return "N/A" if is_rate else ""
    try:
        return f"{float(x):.2%}" if is_rate else f"{float(x):,.2f}"
    except Exception:
        return str(x)


def show_soci_statement_dataframe(df: pd.DataFrame):
    """Small SOCI table with mixed amount rows and percentage-rate rows.

    Display rule:
    - Section header rows show blank numeric cells.
    - Rate rows show percentages in Period 1/2 and Flux, but blank Flux %.
    - Ordinary amount rows keep the normal Flux % calculation.
    """
    display = df.drop(columns=["__level", "__fsli", "__pl_sort", "__option_label", "__is_rate", "__is_section"], errors="ignore").copy()
    if display.empty:
        st.dataframe(display, use_container_width=True, hide_index=True)
        return

    is_rate = df.get("__is_rate", pd.Series(False, index=df.index)).fillna(False).astype(bool).reset_index(drop=True)
    is_section = df.get("__is_section", pd.Series(False, index=df.index)).fillna(False).astype(bool).reset_index(drop=True)

    for c in ["Period 1 Amount", "Period 2 Amount", "Flux"]:
        if c in display.columns:
            formatted = []
            for v, r, sec in zip(display[c].tolist(), is_rate.tolist(), is_section.tolist()):
                if sec:
                    formatted.append("")
                else:
                    formatted.append(fmt_mixed_soci_value(v, bool(r)))
            display[c] = formatted

    if "Flux %" in display.columns:
        formatted = []
        for v, r, sec in zip(df["Flux %"].tolist(), is_rate.tolist(), is_section.tolist()):
            if sec or r:
                formatted.append("")
            else:
                formatted.append(fmt_pct(v))
        display["Flux %"] = formatted

    st.dataframe(display, use_container_width=True, hide_index=True)

# -----------------------------
# SOCI calculations
# -----------------------------
@st.cache_data(show_spinner=False)
def soci_base(df: pd.DataFrame) -> pd.DataFrame:
    needed = [COL_DATE, COL_RMB, COL_PL_SORT, COL_PL_SECTION, COL_FSLI]
    missing = required_cols_missing(df, needed)
    if missing:
        st.error("SOCI missing required columns: " + ", ".join(missing))
        return pd.DataFrame()

    base = df.copy()
    base = base[base[COL_DATE].notna()].copy()
    base[COL_PL_SECTION] = clean_text_series(base[COL_PL_SECTION])
    base[COL_FSLI] = clean_text_series(base[COL_FSLI])
    base["__pl_sort_code"] = base[COL_PL_SORT].map(normalize_sort_code)

    base = base[base[COL_PL_SECTION].ne("")].copy()
    base = base[base[COL_FSLI].ne("")].copy()
    base = base[base["__pl_sort_code"].ne("")].copy()

    # Optional extra filter: if Account Property exists, keep rows marked as PL & OCI.
    # If this column is absent or not populated consistently, PL Section + FSLI + PL Sort still drives the SOCI.
    if COL_ACCOUNT_PROPERTY in base.columns:
        prop_norm = normalize_key(base[COL_ACCOUNT_PROPERTY])
        prop_mask = prop_norm.str.contains("pl", na=False) & prop_norm.str.contains("oci", na=False)
        if prop_mask.any():
            base = base[prop_mask].copy()

    return base


def build_pl_sort_label_map(base: pd.DataFrame) -> dict[str, str]:
    """Map PL Sort code to the current/sanitized FSLI display name."""
    if base.empty:
        return {}
    labels = (
        base[["__pl_sort_code", COL_FSLI]]
        .dropna()
        .drop_duplicates()
        .sort_values(["__pl_sort_code", COL_FSLI])
    )
    mapping = {}
    for sort_code, group in labels.groupby("__pl_sort_code", sort=False):
        names = [str(x).strip() for x in group[COL_FSLI].tolist() if str(x).strip()]
        mapping[str(sort_code)] = names[0] if names else str(sort_code)
    return mapping


@st.cache_data(show_spinner=False)
def build_soci_statement(
    df: pd.DataFrame,
    p1_start: pd.Timestamp,
    p1_end: pd.Timestamp,
    p2_start: pd.Timestamp,
    p2_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build the custom SOCI structure by PL Sort code.

    Display convention: all SOCI line items use -RMB Amount.
    PL Sort is used as the stable mapping key so sanitized FSLI names will not break the report.
    """
    base = soci_base(df)
    if base.empty:
        return pd.DataFrame()

    p1 = base[(base[COL_DATE] >= p1_start) & (base[COL_DATE] <= p1_end)]
    p2 = base[(base[COL_DATE] >= p2_start) & (base[COL_DATE] <= p2_end)]

    # Requested presentation amount = -RMB Amount for all SOCI rows.
    p1_sum = -p1.groupby("__pl_sort_code", dropna=False)[COL_RMB].sum()
    p2_sum = -p2.groupby("__pl_sort_code", dropna=False)[COL_RMB].sum()
    label_map = build_pl_sort_label_map(base)

    rows = []
    values_p1: dict[str, float] = {}
    values_p2: dict[str, float] = {}

    def add_row(
        label: str,
        v1: float,
        v2: float,
        level: int,
        pl_sort: str = "",
        fsli: str = "",
        is_rate: bool = False,
        is_section: bool = False,
    ):
        option_label = f"{pl_sort} - {fsli}" if pl_sort and fsli else ""
        rows.append({
            "Statement of Comprehensive Income": label,
            "Period 1 Amount": v1,
            "Period 2 Amount": v2,
            "Flux": v2 - v1 if not (pd.isna(v1) or pd.isna(v2)) else np.nan,
            "Flux %": np.nan if is_rate else safe_pct(v2 - v1, v1),
            "__level": level,
            "__fsli": fsli,
            "__pl_sort": pl_sort,
            "__option_label": option_label,
            "__is_rate": is_rate,
            "__is_section": is_section,
        })

    def get_line_values(pl_sort: str) -> tuple[float, float]:
        code = normalize_sort_code(pl_sort)
        return float(p1_sum.get(code, 0.0)), float(p2_sum.get(code, 0.0))

    for block in SOCI_STRUCTURE:
        if "section" in block:
            section = block["section"]
            line_sorts = block.get("line_sorts", [])
            subtotal_name = block.get("subtotal")
            subtotal_key = block.get("subtotal_key")

            add_row(section, np.nan, np.nan, level=0, is_section=True)
            subtotal_p1 = 0.0
            subtotal_p2 = 0.0

            for pl_sort in line_sorts:
                code = normalize_sort_code(pl_sort)
                fsli_label = label_map.get(code, code)
                v1, v2 = get_line_values(code)
                subtotal_p1 += v1
                subtotal_p2 += v2
                values_p1[code] = v1
                values_p2[code] = v2
                add_row("   " + fsli_label, v1, v2, level=1, pl_sort=code, fsli=fsli_label)

            if subtotal_name:
                values_p1[subtotal_key] = subtotal_p1
                values_p2[subtotal_key] = subtotal_p2
                add_row(subtotal_name, subtotal_p1, subtotal_p2, level=1)

        elif "calculation" in block:
            name = block["calculation"]
            key = block.get("key", "")
            formula_keys = block.get("formula_keys", [])
            v1 = sum(values_p1.get(k, 0.0) for k in formula_keys)
            v2 = sum(values_p2.get(k, 0.0) for k in formula_keys)
            if key:
                values_p1[key] = v1
                values_p2[key] = v2
            add_row(name, v1, v2, level=0)

        elif "rate" in block:
            name = block["rate"]
            numerator = block["numerator"]
            denominator = block["denominator"]
            v1 = safe_divide(values_p1.get(numerator, np.nan), values_p1.get(denominator, np.nan))
            v2 = safe_divide(values_p2.get(numerator, np.nan), values_p2.get(denominator, np.nan))
            add_row(name, v1, v2, level=0, is_rate=True)

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def soci_account_bridge(
    df: pd.DataFrame,
    pl_sorts: list[str],
    p1_start: pd.Timestamp,
    p1_end: pd.Timestamp,
    p2_start: pd.Timestamp,
    p2_end: pd.Timestamp,
) -> pd.DataFrame:
    base = soci_base(df)
    if base.empty or not pl_sorts:
        return pd.DataFrame()

    selected_codes = [normalize_sort_code(x) for x in pl_sorts if normalize_sort_code(x)]
    base = base[base["__pl_sort_code"].isin(selected_codes)].copy()

    p1 = base[(base[COL_DATE] >= p1_start) & (base[COL_DATE] <= p1_end)]
    p2 = base[(base[COL_DATE] >= p2_start) & (base[COL_DATE] <= p2_end)]

    def side_sum(data: pd.DataFrame, side: str) -> pd.Series:
        return data[data["__side"].eq(side)].groupby(COL_ACCOUNT, dropna=False)[COL_RMB].sum()

    p1_debit = side_sum(p1, "Debit")
    p1_credit = side_sum(p1, "Credit")
    p2_debit = side_sum(p2, "Debit")
    p2_credit = side_sum(p2, "Credit")
    accounts = sorted(
        set(p1_debit.index).union(p1_credit.index).union(p2_debit.index).union(p2_credit.index),
        key=str,
    )
    out = pd.DataFrame({COL_ACCOUNT: accounts})
    out["Period 1 Debit Amount"] = out[COL_ACCOUNT].map(p1_debit).fillna(0.0)
    out["Period 1 Credit Amount"] = out[COL_ACCOUNT].map(p1_credit).fillna(0.0)
    out["Period 2 Debit Amount"] = out[COL_ACCOUNT].map(p2_debit).fillna(0.0)
    out["Period 2 Credit Amount"] = out[COL_ACCOUNT].map(p2_credit).fillna(0.0)
    out["Period 1 Total"] = out["Period 1 Debit Amount"] + out["Period 1 Credit Amount"]
    out["Period 2 Total"] = out["Period 2 Debit Amount"] + out["Period 2 Credit Amount"]
    out["Flux"] = out["Period 2 Total"] - out["Period 1 Total"]
    return out.sort_values("Flux", key=lambda s: s.abs(), ascending=False)


def add_period_day_no(data: pd.DataFrame, start_date: pd.Timestamp, label: str) -> pd.DataFrame:
    out = data.copy()
    out["Period"] = label
    out["Day No"] = (out[COL_DATE] - start_date).dt.days + 1
    return out


# -----------------------------
# App UI
# -----------------------------
st.title("Financial Analysis Panel")
st.caption("SOFP and SOCI comparison panel based on the standard JE database. Online mode can read a sanitized data file committed beside this app.")

with st.sidebar:
    st.header("Data source")
    source_mode = st.radio(
        "Source mode",
        ["Repository file", "Upload manually"],
        index=0,
        help="For online publishing, keep the sanitized data file in the same GitHub repository folder as this .py file.",
    )

    repo_file = None
    uploaded = None

    if source_mode == "Repository file":
        auto_file = find_default_repo_data_file()
        default_name = auto_file.name if auto_file else DEFAULT_REPO_DATA_FILE
        repo_file_name = st.text_input("Repository data file name", value=default_name)
        repo_file = APP_DIR / repo_file_name
        st.caption(f"App folder: `{APP_DIR}`")
    else:
        uploaded = st.file_uploader("Upload standard JE database", type=["xlsx", "xlsm", "xls", "csv"])

    st.header("Performance")
    fast_mode = st.toggle("Fast mode", value=True, help="Avoid heavy dataframe styling and limit chart points for smoother interaction.")
    max_display_rows = st.number_input("Max rows shown in expansion tables", min_value=50, max_value=5000, value=DEFAULT_DISPLAY_ROWS, step=50)
    max_scatter_points = st.number_input("Max scatter points rendered", min_value=500, max_value=50000, value=2500, step=500)
    chart_theme = st.selectbox("Scatter chart theme", ["Dark Office", "Light Office"], index=0)

try:
    if source_mode == "Repository file":
        if repo_file is None or not repo_file.exists():
            st.error(
                "Repository data file not found. Put your sanitized database beside this .py file, "
                "or type the exact file name in the sidebar."
            )
            with st.expander("Files currently found beside the app", expanded=False):
                local_files = sorted(
                    [p.name for p in APP_DIR.iterdir() if p.is_file() and not p.name.startswith("~$")]
                )
                st.write(local_files if local_files else "No files found.")
            st.stop()
        if repo_file.suffix.lower() not in SUPPORTED_DATA_EXTENSIONS:
            st.error("Repository data file must be xlsx, xlsm, xls, or csv.")
            st.stop()
        df = load_database_from_repo(str(repo_file), repo_file.stat().st_mtime)
    else:
        if not uploaded:
            st.info("Upload your standard JE database to start the analysis.")
            st.stop()
        df = load_database_from_upload(uploaded)
except Exception as e:
    st.error(f"Failed to load database: {e}")
    st.stop()

missing_core = required_cols_missing(df, [COL_DATE, COL_JE, COL_ACCOUNT, COL_RMB])
if missing_core:
    st.error("Missing core columns: " + ", ".join(missing_core))
    st.stop()

df = df[df[COL_DATE].notna()].copy()
if df.empty:
    st.error("No valid Date found in the uploaded database.")
    st.stop()

with st.sidebar:
    st.success(f"Loaded {len(df):,} rows")
    st.write(f"Date range: {df[COL_DATE].min().date()} to {df[COL_DATE].max().date()}")

page = st.sidebar.radio("Page", ["Statement of Financial Position", "Statement of Comprehensive Income"])

# -----------------------------
# SOFP page
# -----------------------------
if page == "Statement of Financial Position":
    st.header("Statement of Financial Position")

    base = sofp_base(df)
    if base.empty:
        st.stop()

    available_dates = sorted(base[COL_DATE].dt.date.dropna().unique())
    if "sofp_date1" not in st.session_state:
        st.session_state["sofp_date1"] = available_dates[max(0, len(available_dates) - 2)]
    if "sofp_date2" not in st.session_state:
        st.session_state["sofp_date2"] = available_dates[-1]

    with st.form("sofp_date_form", border=False):
        c1, c2, c3 = st.columns([1, 1, 0.45])
        with c1:
            date1_input = st.selectbox("Date 1", available_dates, index=available_dates.index(st.session_state["sofp_date1"]) if st.session_state["sofp_date1"] in available_dates else max(0, len(available_dates)-2))
        with c2:
            date2_input = st.selectbox("Date 2", available_dates, index=available_dates.index(st.session_state["sofp_date2"]) if st.session_state["sofp_date2"] in available_dates else len(available_dates)-1)
        with c3:
            st.write("")
            st.write("")
            apply_sofp = st.form_submit_button("Apply")
        if apply_sofp:
            st.session_state["sofp_date1"] = date1_input
            st.session_state["sofp_date2"] = date2_input

    date1_ts = pd.Timestamp(st.session_state["sofp_date1"])
    date2_ts = pd.Timestamp(st.session_state["sofp_date2"])
    if date2_ts < date1_ts:
        st.warning("Date 2 is earlier than Date 1. Flux will still follow Date 2 minus Date 1.")

    statement = build_sofp_statement(df, date1_ts, date2_ts)
    if statement.empty:
        st.info("No SOFP rows identified from BS Report FSLI / BS Section columns.")
        st.stop()

    display = statement.drop(columns=["__level", "__fsli"], errors="ignore")
    show_amount_dataframe(display, ["Date 1 Balance", "Date 2 Balance", "Flux"], ["Flux %"])

    fsli_options = statement.loc[statement["__fsli"].astype(str).ne(""), "__fsli"].drop_duplicates().tolist()
    st.subheader("Flux expansion")
    fsli = st.selectbox("Select FSLI to expand", fsli_options)

    bridge = sofp_account_bridge(df, fsli, date1_ts, date2_ts)
    st.markdown(f"#### Account bridge - {fsli}")
    if bridge.empty:
        st.info("No account movement found for this FSLI.")
    else:
        show_amount_dataframe(
            bridge,
            ["Date 1 Balance", "Debit Amount", "Credit Amount", "Date 2 Balance", "Check Difference"],
            max_rows=int(max_display_rows) if fast_mode else None,
        )

        account_options = bridge[COL_ACCOUNT].dropna().astype(str).tolist()
        account = st.selectbox("Select account for transaction scatter", account_options)
        side_choice = st.radio("Transaction side", ["Debit", "Credit", "Both"], horizontal=True)

        tx = base[
            base[COL_BS_FSLI].eq(fsli)
            & base[COL_ACCOUNT].astype(str).eq(str(account))
            & (base[COL_DATE] > date1_ts)
            & (base[COL_DATE] <= date2_ts)
        ].copy()
        if side_choice != "Both":
            tx = tx[tx["__side"].eq(side_choice)].copy()

        hover = [c for c in [COL_DATE, COL_ACCOUNT, COL_RMB, COL_ABSTRACT, COL_DIMENSION, COL_CF_SANKEY, "__side"] if c in tx.columns]
        render_sofp_scatter = st.toggle("Render / refresh scatter chart", value=False, key="render_sofp_scatter")
        if render_sofp_scatter:
            with st.expander("Transaction scatter", expanded=True):
                clicked_je = scatter_and_select(
                    tx,
                    title=f"{account} transactions from {date1_ts.date()}+1 to {date2_ts.date()}",
                    x_col=COL_DATE,
                    y_col=COL_RMB,
                    color_col=None,
                    hover_cols=hover,
                    key="sofp_scatter",
                    max_points=int(max_scatter_points) if fast_mode else 0,
                    chart_theme=chart_theme,
                )
                show_je_detail(df, clicked_je)
        else:
            st.caption("Scatter chart is paused for smoother navigation. Turn on the toggle above when you want to inspect transactions visually.")

# -----------------------------
# SOCI page
# -----------------------------
else:
    st.header("Statement of Comprehensive Income")

    base = soci_base(df)
    if base.empty:
        st.info("No SOCI rows identified. Please check PL Section, FSLI, PL Sort, and Account Property columns.")
        st.stop()

    min_month = base[COL_DATE].min().to_period("M")
    max_month = base[COL_DATE].max().to_period("M")
    month_labels = [str(p) for p in pd.period_range(min_month, max_month, freq="M")]

    st.markdown("#### Period selection")
    if "soci_p1_start" not in st.session_state:
        st.session_state["soci_p1_start"] = month_labels[0]
    if "soci_p1_end" not in st.session_state:
        st.session_state["soci_p1_end"] = month_labels[min(2, len(month_labels) - 1)]
    if "soci_p2_start" not in st.session_state:
        st.session_state["soci_p2_start"] = month_labels[max(0, len(month_labels) - 3)]
    if "soci_p2_end" not in st.session_state:
        st.session_state["soci_p2_end"] = month_labels[-1]

    with st.form("soci_period_form", border=False):
        p1c1, p1c2, p2c1, p2c2, p_apply = st.columns([1, 1, 1, 1, 0.45])
        with p1c1:
            p1_start_input = st.selectbox("Period 1 start", month_labels, index=month_labels.index(st.session_state["soci_p1_start"]) if st.session_state["soci_p1_start"] in month_labels else 0)
        with p1c2:
            p1_end_input = st.selectbox("Period 1 end", month_labels, index=month_labels.index(st.session_state["soci_p1_end"]) if st.session_state["soci_p1_end"] in month_labels else min(2, len(month_labels)-1))
        with p2c1:
            p2_start_input = st.selectbox("Period 2 start", month_labels, index=month_labels.index(st.session_state["soci_p2_start"]) if st.session_state["soci_p2_start"] in month_labels else max(0, len(month_labels)-3))
        with p2c2:
            p2_end_input = st.selectbox("Period 2 end", month_labels, index=month_labels.index(st.session_state["soci_p2_end"]) if st.session_state["soci_p2_end"] in month_labels else len(month_labels)-1)
        with p_apply:
            st.write("")
            st.write("")
            apply_soci = st.form_submit_button("Apply")
        if apply_soci:
            st.session_state["soci_p1_start"] = p1_start_input
            st.session_state["soci_p1_end"] = p1_end_input
            st.session_state["soci_p2_start"] = p2_start_input
            st.session_state["soci_p2_end"] = p2_end_input

    p1_start_label = st.session_state["soci_p1_start"]
    p1_end_label = st.session_state["soci_p1_end"]
    p2_start_label = st.session_state["soci_p2_start"]
    p2_end_label = st.session_state["soci_p2_end"]

    p1_start = parse_month_label(p1_start_label)
    p1_end = month_end(p1_end_label)
    p2_start = parse_month_label(p2_start_label)
    p2_end = month_end(p2_end_label)

    if p1_end < p1_start or p2_end < p2_start:
        st.error("Period end month cannot be earlier than start month.")
        st.stop()

    p1_months = (p1_end.to_period("M") - p1_start.to_period("M")).n + 1
    p2_months = (p2_end.to_period("M") - p2_start.to_period("M")).n + 1
    if p1_months != p2_months:
        st.error(
            f"Period length must be aligned before comparison. Period 1 has {p1_months} month(s), "
            f"while Period 2 has {p2_months} month(s)."
        )
        st.stop()

    statement = build_soci_statement(df, p1_start, p1_end, p2_start, p2_end)
    if statement.empty:
        st.info("No SOCI rows found for the selected periods.")
        st.stop()

    show_soci_statement_dataframe(statement)

    st.subheader("Flux expansion")
    line_options_df = (
        statement.loc[statement["__pl_sort"].astype(str).ne(""), ["__pl_sort", "__option_label"]]
        .drop_duplicates()
        .copy()
    )
    option_labels = line_options_df["__option_label"].tolist()
    label_to_sort = dict(zip(line_options_df["__option_label"], line_options_df["__pl_sort"]))

    selected_labels = st.multiselect(
        "Select FSLI(s) to expand",
        option_labels,
        default=option_labels[:1],
    )
    selected_pl_sorts = [label_to_sort[label] for label in selected_labels if label in label_to_sort]

    bridge = soci_account_bridge(df, selected_pl_sorts, p1_start, p1_end, p2_start, p2_end)
    if bridge.empty:
        st.info("No account movement found for the selected FSLI(s).")
    else:
        show_amount_dataframe(
            bridge,
            [
                "Period 1 Debit Amount",
                "Period 1 Credit Amount",
                "Period 2 Debit Amount",
                "Period 2 Credit Amount",
                "Period 1 Total",
                "Period 2 Total",
                "Flux",
            ],
            max_rows=int(max_display_rows) if fast_mode else None,
        )

        selected_accounts = st.multiselect(
            "Select account(s) for scatter chart",
            bridge[COL_ACCOUNT].dropna().astype(str).tolist(),
            default=bridge[COL_ACCOUNT].dropna().astype(str).tolist()[:1],
        )
        side_choice = st.radio("Transaction side", ["Debit", "Credit", "Both"], horizontal=True, key="soci_side")

        p1_tx = base[
            base["__pl_sort_code"].isin(selected_pl_sorts)
            & base[COL_ACCOUNT].astype(str).isin(selected_accounts)
            & (base[COL_DATE] >= p1_start)
            & (base[COL_DATE] <= p1_end)
        ].copy()
        p2_tx = base[
            base["__pl_sort_code"].isin(selected_pl_sorts)
            & base[COL_ACCOUNT].astype(str).isin(selected_accounts)
            & (base[COL_DATE] >= p2_start)
            & (base[COL_DATE] <= p2_end)
        ].copy()
        if side_choice != "Both":
            p1_tx = p1_tx[p1_tx["__side"].eq(side_choice)].copy()
            p2_tx = p2_tx[p2_tx["__side"].eq(side_choice)].copy()

        p1_plot = add_period_day_no(p1_tx, p1_start, f"Period 1: {p1_start_label} to {p1_end_label}")
        p2_plot = add_period_day_no(p2_tx, p2_start, f"Period 2: {p2_start_label} to {p2_end_label}")
        plot_df = pd.concat([p1_plot, p2_plot], ignore_index=True)

        hover = [c for c in ["Period", COL_DATE, COL_ACCOUNT, COL_PL_SORT, COL_FSLI, COL_RMB, COL_ABSTRACT, COL_DIMENSION, COL_CF_SANKEY, "__side"] if c in plot_df.columns]
        render_soci_scatter = st.toggle("Render / refresh scatter chart", value=False, key="render_soci_scatter")
        if render_soci_scatter:
            with st.expander("Transaction scatter", expanded=True):
                clicked_je = scatter_and_select(
                    plot_df,
                    title="SOCI selected account transactions by day number inside each period",
                    x_col="Day No",
                    y_col=COL_RMB,
                    color_col="Period",
                    hover_cols=hover,
                    key="soci_scatter",
                    max_points=int(max_scatter_points) if fast_mode else 0,
                    chart_theme=chart_theme,
                )
                show_je_detail(df, clicked_je)
        else:
            st.caption("Scatter chart is paused for smoother navigation. Turn on the toggle above when you want to inspect transactions visually.")
