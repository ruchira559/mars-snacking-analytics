import os
import re
import time
import logging
import datetime
import hashlib
import streamlit as st
import pandas as pd
import io
import json
from pathlib import Path
import streamlit.components.v1 as components
from html import escape
import plotly.express as px
import plotly.graph_objects as go
from databricks import sql as dbsql
from databricks.sdk.core import Config
from databricks.sdk import WorkspaceClient

log = logging.getLogger("mars_snacking")
logging.basicConfig(level=logging.INFO)

# ── Configuration ─────────────────────────────────────────────────────────────
TABLE = "uc_danda_sandbox.bi_kitchen.mars_snacking_rls_test"
ADMIN_TABLE = "uc_danda_sandbox.bi_kitchen.mars_snacking_rls_admins"
AGENCY_MAP_TABLE = "uc_danda_sandbox.bi_kitchen.mars_snacking_agency_map"

# ── Viewer Authentication & Data Access ───────────────────────────────────────
def get_viewer_token() -> str:
    """Extract the viewer's OAuth token forwarded by the Databricks Apps proxy.

    The ``x-forwarded-access-token`` header is set by the Databricks reverse
    proxy *after* the viewer completes OAuth authentication.  It cannot be
    injected or overridden by the client.  Fail closed when the header is
    absent — never fall back to the service principal."""
    token = st.context.headers.get("x-forwarded-access-token")
    if not token:
        st.error(
            "Unable to verify your identity. This app requires Databricks "
            "user authorization with the **sql** scope enabled. Please "
            "access the app through its Databricks Apps URL."
        )
        st.stop()
    return token


_WAREHOUSE_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_HTTP_PATH_RE = re.compile(r"^/sql/1\.0/warehouses/[0-9a-f]{16}$")


def _resolve_http_path(raw: str) -> str:
    """Normalise the SQL-warehouse env var into a valid HTTP path.

    Databricks Apps ``valueFrom: sql-warehouse`` may resolve to either a
    bare warehouse ID (``2e466592f73554e0``) or a full HTTP path
    (``/sql/1.0/warehouses/2e466592f73554e0``).  Accept both; reject
    anything else (fail-closed)."""
    val = raw.strip()
    if not val:
        return ""
    if _HTTP_PATH_RE.match(val):
        return val
    if _WAREHOUSE_ID_RE.match(val):
        resolved = f"/sql/1.0/warehouses/{val}"
        log.info("Resolved bare warehouse ID to http_path=%s", resolved)
        return resolved
    raise ConnectionError(
        f"DATABRICKS_SQL_WAREHOUSE_HTTP_PATH has unexpected format: {val!r}\n"
        "Expected a 16-char hex warehouse ID or /sql/1.0/warehouses/<id>."
    )


def load_data(viewer_token: str) -> pd.DataFrame:
    """Query the source table as the authenticated viewer.

    Unity Catalog row filters restrict rows server-side based on
    ``CURRENT_USER()``.  No application-layer RLS filtering is performed;
    enforcement is entirely in the database layer.

    A fresh connection is created per call — no cross-user connection or
    data caching is used."""
    cfg = Config()
    server_hostname = (
        (cfg.host or "").replace("https://", "").replace("http://", "").rstrip("/")
    )
    http_path_raw = os.environ.get("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH", "")
    http_path = _resolve_http_path(http_path_raw)

    # ── Diagnose missing or malformed configuration.  Never expose tokens. ──
    problems = []
    if not server_hostname:
        problems.append("DATABRICKS_HOST is not set")
    if not http_path:
        problems.append(
            "DATABRICKS_SQL_WAREHOUSE_HTTP_PATH is empty — "
            "check app.yaml `valueFrom` and the sql-warehouse app resource"
        )
    if not viewer_token:
        problems.append("viewer access token is absent (x-forwarded-access-token)")
    if problems:
        detail = "\n".join(f"• {p}" for p in problems)
        raise ConnectionError(
            f"SQL connection cannot be established:\n{detail}"
        )

    log.info("Connecting: host=%s, http_path=%s, token=<present>",
             server_hostname, http_path)

    # ── Connect with a bounded timeout so the page never hangs silently. ──
    t0 = time.time()
    try:
        conn = dbsql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=viewer_token,
            _socket_timeout=30,
        )
    except Exception as e:
        elapsed = time.time() - t0
        raise ConnectionError(
            f"dbsql.connect() failed after {elapsed:.1f}s.\n"
            f"Host: {server_hostname}\nHTTP path: {http_path}\n"
            f"Error: {e}"
        ) from e
    log.info("Connected in %.1fs", time.time() - t0)

    try:
        with conn.cursor() as cur:
            t1 = time.time()
            cur.execute(f"SELECT * FROM {TABLE}")
            log.info("Query executed in %.1fs", time.time() - t1)

            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            log.info("Fetched %d rows in %.1fs", len(rows), time.time() - t1)

        df = pd.DataFrame(rows, columns=cols)
        # Convert Decimal → float so Plotly charts render correctly
        if "Hours" in df.columns:
            df["Hours"] = pd.to_numeric(df["Hours"], errors="coerce")
        return df
    finally:
        conn.close()


def _check_is_admin(viewer_token: str) -> bool:
    """Check whether CURRENT_USER() is listed in the admin table.

    Uses the viewer's own OAuth token so the identity resolves to the
    actual viewer, not the app service principal.  Fail-closed: any
    exception (network, permissions, timeout) returns False."""
    try:
        cfg = Config()
        hostname = (
            (cfg.host or "").replace("https://", "").replace("http://", "").rstrip("/")
        )
        http_path = _resolve_http_path(
            os.environ.get("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH", "")
        )
        if not hostname or not http_path:
            log.warning("Admin check skipped — missing host or http_path")
            return False
        conn = dbsql.connect(
            server_hostname=hostname,
            http_path=http_path,
            access_token=viewer_token,
            _socket_timeout=30,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT 1 FROM {ADMIN_TABLE} "
                    "WHERE user_email = CURRENT_USER() LIMIT 1"
                )
                return cur.fetchone() is not None
        finally:
            conn.close()
    except Exception:
        log.warning("Admin check failed — treating viewer as non-admin",
                    exc_info=True)
        return False


def _run_security_query(viewer_token: str, query: str) -> pd.DataFrame:
    """Execute a read-only SELECT as the authenticated viewer.

    Used by the Security administration page.  A fresh connection is
    created per call — no cross-user caching."""
    cfg = Config()
    hostname = (
        (cfg.host or "")
        .replace("https://", "")
        .replace("http://", "")
        .rstrip("/")
    )
    http_path = _resolve_http_path(
        os.environ.get("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH", "")
    )
    if not hostname or not http_path:
        raise ConnectionError("Missing host or warehouse path")
    conn = dbsql.connect(
        server_hostname=hostname,
        http_path=http_path,
        access_token=viewer_token,
        _socket_timeout=30,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


def _exec_secure_sql(viewer_token: str, sql: str, params=None):
    """Execute parameterized SQL as the authenticated viewer.

    Used for DML (INSERT / DELETE) and parameterized SELECT checks
    on the agency mapping table.  Returns *(rowcount, rows)*.
    A fresh connection is opened and closed per call."""
    cfg = Config()
    hostname = (
        (cfg.host or "")
        .replace("https://", "")
        .replace("http://", "")
        .rstrip("/")
    )
    http_path = _resolve_http_path(
        os.environ.get("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH", "")
    )
    if not hostname or not http_path:
        raise ConnectionError("Missing host or warehouse path")
    conn = dbsql.connect(
        server_hostname=hostname,
        http_path=http_path,
        access_token=viewer_token,
        _socket_timeout=30,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(sql, parameters=params)
            try:
                rows = cur.fetchall()
            except Exception:
                rows = []
            rc = getattr(cur, "rowcount", 0)
            return (rc if rc and rc >= 0 else 0), rows
    finally:
        conn.close()


def _scrub_sql_error(exc) -> str:
    """Return a safe error message, stripping credentials."""
    msg = str(exc)
    if any(
        s in msg.lower()
        for s in ("token", "bearer", "credential", "secret")
    ):
        return "A permission or connection error occurred."
    return msg


# ── Mars Snacking Theme ────────────────────────────────────────────────────────
MARS_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding-top: 1rem !important; }
    /* ── Header ── */
    .mars-header {
        background: linear-gradient(135deg, #1B365D 0%, #2C4A7C 100%);
        padding: 0.7rem 1.5rem; border-radius: 10px;
        margin-bottom: 0.6rem; color: white;
    }
    .mars-header h1 {
        font-size: 1.35rem; font-weight: 700; margin: 0; color: white;
        letter-spacing: 0.02em;
    }
    .mars-header .subtitle { font-size: 0.78rem; color: #B8C9E0; margin-top: 0.1rem; }
    .mars-header .period-badge {
        display: inline-block; background: rgba(255,255,255,0.15);
        padding: 0.1rem 0.55rem; border-radius: 4px; font-weight: 600; color: #fff;
    }
    /* ── KPI Cards ── */
    .kpi-card {
        border-radius: 10px; padding: 0.7rem 0.9rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        display: flex; align-items: center; gap: 0.65rem;
    }
    .kpi-icon { font-size: 1.5rem; line-height: 1; }
    .kpi-value {
        font-size: 1.65rem; font-weight: 700; color: #1B365D;
        line-height: 1.15; margin: 0;
    }
    .kpi-label {
        font-size: 0.68rem; color: #6B7280; text-transform: uppercase;
        letter-spacing: 0.06em; margin: 0;
    }
    /* ── Ranking Panels ── */
    .rank-panel {
        background: #fff; border-radius: 10px; padding: 0.7rem 0.9rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    .rank-panel h3 {
        font-size: 0.82rem; font-weight: 600; color: #1B365D;
        margin: 0 0 0.4rem 0; padding-bottom: 0.3rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .rank-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
    .rank-table th {
        text-align: left; color: #9CA3AF; font-weight: 600;
        padding: 0.25rem 0.4rem; font-size: 0.68rem;
        text-transform: uppercase; letter-spacing: 0.04em;
        border-bottom: 1px solid #F3F4F6;
    }
    .rank-table td {
        padding: 0.3rem 0.4rem; border-bottom: 1px solid #F3F4F6; color: #374151;
    }
    .rank-table .hours-col { text-align: right; font-weight: 600; color: #1B365D; }
    .rank-table tr:hover td { background: #F9FAFB; }
    .rank-footer { font-size: 0.68rem; color: #9CA3AF; margin-top: 0.35rem; }
    /* ── Chart Cards ── */
    [data-testid="stPlotlyChart"] {
        background: white; border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07); padding: 0.3rem;
    }
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 2px solid #E5E7EB; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0; padding: 0.35rem 1.2rem;
        font-weight: 600; font-size: 0.85rem;
    }
    /* ── Product Details ── */
    .pd-ctrl label { font-size: 0.8rem !important; }
    /* ── Sidebar ── */
    .sidebar-brand {
        font-size: 0.85rem; font-weight: 600; color: #1B365D;
        padding-bottom: 0.25rem; margin-bottom: 0.6rem;
        border-bottom: 2px solid #E11D3F;
    }
    /* Phase 3.1: compact navigation and polished filter trigger */
    section[data-testid="stSidebar"] { background:#f1f4f9; border-right:1px solid #dce3ed; }
    section[data-testid="stSidebar"] [role="radiogroup"] { gap:0.4rem; }
    section[data-testid="stSidebar"] [role="radiogroup"] label {
        background:#fff; border:1px solid #e3e8f1; border-radius:10px;
        padding:0.55rem 0.65rem; transition:background .15s ease;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#e8eef8; }
    div[data-testid="stPopover"] button { border:1px solid #cad5e4; border-radius:9px;
        color:#1b365d; font-weight:650; background:#fff; }
    div[data-testid="stPopover"] button:hover { border-color:#e11d3f; color:#e11d3f; }
</style>
"""

MARS_NAVY = "#1B365D"
LOB_COLORS = {"Project": "#1B365D", "Retainer": "#E11D3F", "Incremental": "#F5A623", "(Blank)": "#D1D5DB"}


def compute_reporting_period(month_series: pd.Series) -> str:
    """Derive 'MMM YYYY' from the latest valid MM/YYYY value in the series."""
    parsed = pd.to_datetime(month_series, format="%m/%Y", errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return "N/A"
    return valid.max().strftime("%b %Y")


KPI_STYLES = {
    "Agencies":  {"bg": "#EFF6FF", "accent": "#3B82F6", "icon": "\U0001F3E2"},
    "Brands":    {"bg": "#FFF1F2", "accent": "#E11D3F", "icon": "\U0001F3F7\uFE0F"},
    "Campaigns": {"bg": "#FFFBEB", "accent": "#F59E0B", "icon": "\U0001F4E2"},
    "Products":  {"bg": "#F0FDFA", "accent": "#0D9488", "icon": "\U0001F4E6"},
}


def render_kpi_card(value, label: str) -> str:
    s = KPI_STYLES.get(label, {"bg": "#F7F8FA", "accent": "#6B7280", "icon": "\U0001F4CA"})
    return (
        f'<div class="kpi-card" style="background:{s["bg"]};border-left:4px solid {s["accent"]}">'
        f'<div class="kpi-icon">{s["icon"]}</div>'
        '<div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        '</div></div>'
    )


def render_ranking_panel(title: str, data: pd.DataFrame, name_col: str,
                         value_col: str, max_rows: int = 15) -> str:
    """Self-contained ranking panel: title + table + optional footer."""
    rows_html = ""
    for _, row in data.head(max_rows).iterrows():
        name = row[name_col]
        name = "(Blank)" if pd.isna(name) or str(name).strip() == "" else name
        rows_html += f'<tr><td>{name}</td><td class="hours-col">{row[value_col]:,.0f}</td></tr>'
    footer = ""
    if len(data) > max_rows:
        footer = f'<div class="rank-footer">Showing top {max_rows} of {len(data)}</div>'
    return (
        f'<div class="rank-panel"><h3>{title}</h3>'
        f'<table class="rank-table"><thead>'
        f'<tr><th>{name_col}</th><th style="text-align:right">Hours</th></tr>'
        f'</thead><tbody>{rows_html}</tbody></table>{footer}</div>'
    )


def _safe_options(series: pd.Series) -> list:
    return sorted(series.dropna().unique().tolist())


# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Mars Snacking | Time Utilization", page_icon="\U0001F36B", layout="wide")
st.markdown(MARS_CSS, unsafe_allow_html=True)

# ── Authenticate & Load Data ─────────────────────────────────────────────────
viewer_token = get_viewer_token()

# Per-session data cache — avoids redundant SQL loads when navigating between
# Dashboard / Genie / Security within a single browser session.  Gated on a
# viewer-token fingerprint so data is never reused across identity changes.
# This is NOT cross-user, NOT cross-session, and NOT @st.cache_data.
_token_fp = hashlib.sha256(viewer_token.encode()).hexdigest()[:16]
_DATA_TTL = 300  # seconds — auto-refresh data every 5 minutes
_cache_age = time.time() - st.session_state.get("_data_loaded_at", 0)
_needs_load = (
    st.session_state.get("_data_token_fp") != _token_fp
    or "_cached_df" not in st.session_state
    or st.session_state.get("_force_reload", False)
    or _cache_age > _DATA_TTL
)

if _needs_load:
    with st.spinner("Connecting to SQL warehouse and loading data\u2026"):
        try:
            t_start = time.time()
            df = load_data(viewer_token)
            _elapsed = time.time() - t_start
            log.info("load_data() completed: %.1fs, %d rows (cache miss)",
                     _elapsed, len(df))
        except Exception as exc:
            log.error("load_data() failed: %s", exc, exc_info=True)
            st.error(
                f"Could not load data from `{TABLE}`.\n\n{exc}\n\n"
                "Check: (1) SQL warehouse is running, (2) your account has "
                "USE CATALOG / USE SCHEMA / SELECT grants, (3) app resource "
                "`sql-warehouse` is configured."
            )
            st.stop()
    st.session_state["_cached_df"] = df
    st.session_state["_data_token_fp"] = _token_fp
    st.session_state["_data_loaded_at"] = time.time()
    st.session_state["_force_reload"] = False
else:
    df = st.session_state["_cached_df"]
    log.debug("load_data() skipped — session cache hit (fp=%s)", _token_fp[:8])

if df.empty:
    st.warning(
        "No data is available for your account. If you believe this is an "
        "error, contact an administrator to verify your agency mapping."
    )
    st.stop()

# ── Admin check (fail-closed, cached per-session per-viewer) ────────────────
if st.session_state.get("_admin_token_fp") != _token_fp:
    is_admin = _check_is_admin(viewer_token)
    st.session_state["_is_admin"] = is_admin
    st.session_state["_admin_token_fp"] = _token_fp
    log.info("Admin check executed: is_admin=%s", is_admin)
else:
    is_admin = st.session_state.get("_is_admin", False)

# ── Parse helper columns (display only, never written back) ──────────────────
df["_month_dt"] = pd.to_datetime(df["Month"], format="%m/%Y", errors="coerce")
df["_year"] = df["_month_dt"].dt.year.astype("Int64")
df["_month_label"] = df["_month_dt"].dt.strftime("%b %Y")

# ── Application navigation: sidebar is for sections, not data filters ────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">MARS SNACKING</div>', unsafe_allow_html=True)
    _nav_options = ["Dashboard", "Genie", "Security"] if is_admin else ["Dashboard"]
    if st.session_state.get("mars_section") not in _nav_options:
        st.session_state["mars_section"] = "Dashboard"
    section = st.radio(
        "Workspace", _nav_options,
        index=0, key="mars_section", label_visibility="collapsed",
    )
    st.caption("Time Utilization Analytics")
    if st.button("\u21bb Refresh data", use_container_width=True,
                 help="Reload data from the warehouse"):
        st.session_state["_force_reload"] = True
        st.rerun()

# ── Shared dashboard filter popover ─────────────────────────────────────────
# Native Streamlit widgets maintain a single filter state across both tabs.
# The popover opens above the dashboard without taking permanent sidebar space.
year_opts = sorted(df["_year"].dropna().unique().tolist())
month_map = (
    df[["Month", "_month_label"]].dropna(subset=["_month_label"])
    .drop_duplicates().set_index("Month")["_month_label"].to_dict()
)
# Phase 3.2: native dialog avoids clipped nested dropdowns in popovers.
# Persistent filter state — manually managed keys, never bound to dialog widgets.
# Dialog widgets use default= to load and Done to save, avoiding fragment-scoped
# key cleanup that clears widget-bound keys when the dialog closes.
_FILTER_KEYS = ("filter_year", "filter_month", "filter_product", "filter_project",
                "filter_dept", "filter_title", "filter_emp")
for _fk in _FILTER_KEYS:
    st.session_state.setdefault(_fk, [])

@st.dialog("Filter dashboard", width="large")
def show_filters():
    st.caption("Selections apply to Summary and Product Details. Close when finished.")
    _yr_opts = year_opts
    _mo_opts = sorted(month_map.keys())
    _pr_opts = _safe_options(df["Product"])
    _pj_opts = _safe_options(df["Project"])
    _dp_opts = _safe_options(df["Department"])
    _tt_opts = _safe_options(df["Employee Title"])
    _em_opts = _safe_options(df["Employee Name"])
    year = st.multiselect("Year", _yr_opts,
                          default=[v for v in st.session_state["filter_year"] if v in _yr_opts],
                          format_func=lambda x: str(int(x)))
    month = st.multiselect("Month", _mo_opts,
                           default=[v for v in st.session_state["filter_month"] if v in _mo_opts],
                           format_func=lambda x: month_map.get(x, x))
    product = st.multiselect("Product", _pr_opts,
                             default=[v for v in st.session_state["filter_product"] if v in _pr_opts])
    project = st.multiselect("Project", _pj_opts,
                             default=[v for v in st.session_state["filter_project"] if v in _pj_opts])
    dept = st.multiselect("Department", _dp_opts,
                          default=[v for v in st.session_state["filter_dept"] if v in _dp_opts])
    title = st.multiselect("Employee Title", _tt_opts,
                           default=[v for v in st.session_state["filter_title"] if v in _tt_opts])
    emp = st.multiselect("Employee Name", _em_opts,
                         default=[v for v in st.session_state["filter_emp"] if v in _em_opts])
    left, right = st.columns(2)
    with left:
        if st.button("Clear all filters", use_container_width=True):
            for _fk in _FILTER_KEYS:
                st.session_state[_fk] = []
            st.rerun()
    with right:
        if st.button("Done", type="primary", use_container_width=True):
            st.session_state["filter_year"] = year
            st.session_state["filter_month"] = month
            st.session_state["filter_product"] = product
            st.session_state["filter_project"] = project
            st.session_state["filter_dept"] = dept
            st.session_state["filter_title"] = title
            st.session_state["filter_emp"] = emp
            st.rerun()

if section == "Dashboard":
    filter_spacer, filter_action = st.columns([5, 1])
    with filter_action:
        if st.button("⚙ Filters", use_container_width=True):
            show_filters()
    sel_year = st.session_state.get("filter_year", [])
    sel_month = st.session_state.get("filter_month", [])
    sel_product = st.session_state.get("filter_product", [])
    sel_project = st.session_state.get("filter_project", [])
    sel_dept = st.session_state.get("filter_dept", [])
    sel_title = st.session_state.get("filter_title", [])
    sel_emp = st.session_state.get("filter_emp", [])
else:
    # Preserve filter selections when switching sections.
    sel_year = st.session_state.get("filter_year", [])
    sel_month = st.session_state.get("filter_month", [])
    sel_product = st.session_state.get("filter_product", [])
    sel_project = st.session_state.get("filter_project", [])
    sel_dept = st.session_state.get("filter_dept", [])
    sel_title = st.session_state.get("filter_title", [])
    sel_emp = st.session_state.get("filter_emp", [])

def _display_label(value):
    return "(Blank)" if pd.isna(value) or str(value).strip() == "" else str(value)


# ── Apply Filters ─────────────────────────────────────────────────────────────
filtered = df.copy()
if sel_year:
    filtered = filtered[filtered["_year"].isin(sel_year)]
if sel_month:
    filtered = filtered[filtered["Month"].isin(sel_month)]
if sel_product:
    filtered = filtered[filtered["Product"].isin(sel_product)]
if sel_project:
    filtered = filtered[filtered["Project"].isin(sel_project)]
if sel_dept:
    filtered = filtered[filtered["Department"].isin(sel_dept)]
if sel_title:
    filtered = filtered[filtered["Employee Title"].isin(sel_title)]
if sel_emp:
    filtered = filtered[filtered["Employee Name"].isin(sel_emp)]



# ── Section routing ─────────────────────────────────────────────────────────
if section == "Genie":
    # ── Admin-only Genie route guard (live SQL recheck, fail-closed) ───────
    if not _check_is_admin(viewer_token):
        st.error("Access denied. Genie requires an administrator account.")
        st.stop()

    # ── Genie Space configuration ──────────────────────────────────────────
    _genie_space_id = os.environ.get("DATABRICKS_GENIE_SPACE_ID", "").strip()
    if not _genie_space_id:
        st.warning(
            "Genie Space is not configured. An administrator must add the "
            "`genie-space` app resource and redeploy."
        )
        st.stop()

    # ── Identity isolation — clear conversation when viewer changes ────────
    _viewer_fp = hashlib.sha256(viewer_token.encode()).hexdigest()[:16]
    if st.session_state.get("_genie_fp") != _viewer_fp:
        for _k in list(st.session_state.keys()):
            if _k.startswith("_genie_"):
                del st.session_state[_k]
        st.session_state["_genie_fp"] = _viewer_fp
    st.session_state.setdefault("_genie_history", [])
    st.session_state.setdefault("_genie_conv_id", None)

    # ── Header ─────────────────────────────────────────────────────────────
    _gh_col, _ga_col = st.columns([5, 1])
    with _gh_col:
        st.markdown(
            '<div class="mars-header">'
            '<h1>Genie</h1>'
            '<div class="subtitle">Conversational Analytics &mdash; Admin Only</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with _ga_col:
        if st.button("New conversation", use_container_width=True):
            st.session_state["_genie_history"] = []
            st.session_state["_genie_conv_id"] = None
            st.rerun()

    # ── Render conversation history ────────────────────────────────────────
    for _msg in st.session_state["_genie_history"]:
        with st.chat_message(_msg["role"]):
            st.markdown(_msg["content"])
            if _msg.get("sql"):
                with st.expander("Generated SQL", expanded=False):
                    st.code(_msg["sql"], language="sql")
            if _msg.get("result_data"):
                st.dataframe(
                    pd.DataFrame(_msg["result_data"]),
                    use_container_width=True,
                )
            if _msg.get("suggestions"):
                st.caption("Suggested: " + " \u00b7 ".join(_msg["suggestions"]))

    # ── Chat input ─────────────────────────────────────────────────────────
    st.caption(
        "\U0001f4a1 For complete results, include NULL and blank categories "
        "in your question."
    )
    if _user_q := st.chat_input("Ask a question about Mars Snacking data\u2026"):
        # ── Re-verify admin authorization before every Genie API call ──────
        if not _check_is_admin(viewer_token):
            st.error(
                "Administrator authorization could not be verified. "
                "Your request has been blocked."
            )
            st.stop()

        # ── Render user message ────────────────────────────────────────────
        st.session_state["_genie_history"].append(
            {"role": "user", "content": _user_q}
        )
        with st.chat_message("user"):
            st.markdown(_user_q)

        # ── Call Genie Conversation API as the viewer ──────────────────────
        with st.chat_message("assistant"):
            with st.spinner("Thinking\u2026"):
                try:
                    # auth_type="pat" suppresses OAuth M2M auto-discovery.
                    # Without it the SDK sees BOTH the viewer token (PAT)
                    # AND the SP env-vars DATABRICKS_CLIENT_ID/SECRET,
                    # then rejects the ambiguity.
                    _genie_ws = WorkspaceClient(
                        host=os.environ.get("DATABRICKS_HOST", ""),
                        token=viewer_token,
                        auth_type="pat",
                    )
                    _conv_id = st.session_state.get("_genie_conv_id")

                    if _conv_id:
                        _genie_result = (
                            _genie_ws.genie.create_message_and_wait(
                                space_id=_genie_space_id,
                                conversation_id=_conv_id,
                                content=_user_q,
                                timeout=datetime.timedelta(seconds=120),
                            )
                        )
                    else:
                        _genie_result = (
                            _genie_ws.genie.start_conversation_and_wait(
                                space_id=_genie_space_id,
                                content=_user_q,
                                timeout=datetime.timedelta(seconds=120),
                            )
                        )

                    st.session_state["_genie_conv_id"] = (
                        _genie_result.conversation_id
                    )

                    # ── Extract response components ────────────────────────
                    _text_parts = []
                    _sql_text = None
                    _suggestions = []

                    _query_att_id = None
                    _query_stmt_id = None
                    for _att in _genie_result.attachments or []:
                        # Text attachment
                        _txt = getattr(_att, "text", None)
                        if _txt and getattr(_txt, "content", None):
                            _text_parts.append(_txt.content)
                        # Query attachment — defensive getattr for SDK
                        # compat (0.33 uses QueryAttachment without
                        # some fields present in later versions).
                        _qa = getattr(_att, "query", None)
                        if _qa:
                            _q_sql = getattr(_qa, "query", None)
                            if _q_sql:
                                _sql_text = _q_sql
                            _q_desc = getattr(_qa, "description", None)
                            if _q_desc:
                                _text_parts.insert(0, _q_desc)
                            _meta = getattr(_qa, "query_result_metadata", None)
                            if _meta and getattr(_meta, "row_count", None) is not None:
                                _text_parts.append(
                                    f"*{_meta.row_count} rows returned*"
                                )
                            if not _query_att_id:
                                _query_att_id = getattr(
                                    _att, "attachment_id", None
                                )
                            if not _query_stmt_id:
                                _query_stmt_id = getattr(
                                    _qa, "statement_id", None
                                )
                        # Suggested follow-ups
                        _sq = getattr(_att, "suggested_questions", None)
                        if _sq and getattr(_sq, "questions", None):
                            _suggestions.extend(_sq.questions)

                    # SDK 0.33: statement_id and row_count live on
                    # GenieMessage.query_result (a Result object), not
                    # on QueryAttachment.  Later SDKs moved them.
                    if not _query_stmt_id:
                        _qr_msg = getattr(
                            _genie_result, "query_result", None
                        )
                        if _qr_msg:
                            _query_stmt_id = getattr(
                                _qr_msg, "statement_id", None
                            )
                            if (
                                not any(
                                    "rows returned" in p
                                    for p in _text_parts
                                )
                                and getattr(
                                    _qr_msg, "row_count", None
                                ) is not None
                            ):
                                _text_parts.append(
                                    f"*{_qr_msg.row_count}"
                                    f" rows returned*"
                                )

                    # ── Fetch query result table ───────────────────────────
                    # Primary: Genie attachment API (needs attachment_id).
                    # Fallback: SQL Statement API via statement_id —
                    # required for SDK 0.33 where attachment_id is absent.
                    _result_records = None
                    _table_warning = None
                    _sr = None  # StatementResponse from either path
                    if _query_att_id or _query_stmt_id:
                        try:
                            if _query_att_id:
                                _qr = (
                                    _genie_ws.genie
                                    .get_message_query_result_by_attachment(
                                        space_id=_genie_space_id,
                                        conversation_id=(
                                            _genie_result.conversation_id
                                        ),
                                        message_id=(
                                            getattr(
                                                _genie_result,
                                                "message_id", None,
                                            )
                                            or _genie_result.id
                                        ),
                                        attachment_id=_query_att_id,
                                    )
                                )
                                _sr = getattr(
                                    _qr, "statement_response", None
                                )
                            elif _query_stmt_id:
                                log.info(
                                    "attachment_id unavailable; "
                                    "falling back to statement_id=%s",
                                    _query_stmt_id,
                                )
                                _sr = (
                                    _genie_ws.statement_execution
                                    .get_statement(_query_stmt_id)
                                )
                        except Exception as _fetch_err:
                            log.warning(
                                "Genie result fetch failed: %s",
                                type(_fetch_err).__name__,
                                exc_info=True,
                            )

                    # Parse the StatementResponse (same shape from both paths)
                    if _sr:
                        try:
                            _manifest = getattr(_sr, "manifest", None)
                            _rdata = getattr(_sr, "result", None)
                            _schema = (
                                getattr(_manifest, "schema", None)
                                if _manifest else None
                            )
                            _darr = (
                                getattr(_rdata, "data_array", None)
                                if _rdata else None
                            )
                            if _schema and _darr:
                                _cols = [
                                    getattr(c, "name", f"col_{i}")
                                    for i, c in enumerate(
                                        getattr(_schema, "columns", [])
                                        or []
                                    )
                                ]
                                _result_records = [
                                    dict(zip(_cols, r))
                                    for r in _darr[:200]
                                ]
                                log.info(
                                    "Genie result table: %d cols, "
                                    "%d rows",
                                    len(_cols), len(_result_records),
                                )
                        except Exception as _parse_err:
                            log.warning(
                                "Genie result parse failed: %s",
                                type(_parse_err).__name__,
                                exc_info=True,
                            )

                    if (
                        _sql_text
                        and not _result_records
                        and not _genie_result.error
                    ):
                        _table_warning = (
                            "Result table could not be loaded. "
                            "The text answer and SQL are shown above."
                        )

                    if _genie_result.error:
                        _err_detail = getattr(
                            _genie_result.error, "message",
                            "Genie returned an error.",
                        )
                        _resp_text = f"\u26a0\ufe0f {_err_detail}"
                    elif _text_parts:
                        _resp_text = "\n\n".join(_text_parts)
                    else:
                        _resp_text = (
                            "Genie did not return a response "
                            "for this question."
                        )

                    # ── Render ─────────────────────────────────────────────
                    st.markdown(_resp_text)
                    if _sql_text:
                        with st.expander("Generated SQL", expanded=False):
                            st.code(_sql_text, language="sql")
                    if _result_records:
                        st.dataframe(
                            pd.DataFrame(_result_records),
                            use_container_width=True,
                        )
                    elif _table_warning:
                        st.caption(_table_warning)
                    if _suggestions:
                        st.caption(
                            "Suggested: "
                            + " \u00b7 ".join(_suggestions[:3])
                        )

                    # ── Persist to conversation history ────────────────────
                    st.session_state["_genie_history"].append({
                        "role": "assistant",
                        "content": _resp_text,
                        "sql": _sql_text,
                        "result_data": _result_records,
                        "suggestions": (
                            _suggestions[:3] if _suggestions else None
                        ),
                    })

                except Exception as _genie_err:
                    _err_type = type(_genie_err).__name__
                    _err_code = getattr(_genie_err, "error_code", None)
                    _err_status = getattr(_genie_err, "status_code", None)
                    _op_name = (
                        "create_message" if _conv_id
                        else "start_conversation"
                    )
                    log.warning(
                        "Genie API error: type=%s, code=%s, "
                        "http=%s, op=%s",
                        _err_type, _err_code, _err_status, _op_name,
                        exc_info=True,
                    )
                    _safe_msg = str(_genie_err)
                    # Detect scope errors first — give actionable guidance
                    if "required scopes" in _safe_msg.lower():
                        _safe_msg = (
                            "Your browser session does not include the "
                            "required **genie** authorization scope.\n\n"
                            "This usually happens after the app\u2019s OAuth "
                            "permissions were updated. To fix it:\n\n"
                            "1. Clear your browser cookies for this app "
                            "URL\n"
                            "2. Close this tab\n"
                            "3. Re-open the app from its Databricks URL\n\n"
                            "This triggers a fresh authorization that "
                            "includes the new scope."
                        )
                    # Never expose tokens or credentials in error output
                    elif any(
                        s in _safe_msg.lower()
                        for s in ("token", "bearer", "credential", "secret")
                    ):
                        _safe_msg = (
                            "An authentication error occurred. "
                            "Please refresh the page and try again."
                        )
                    st.error(
                        f"Genie encountered an error:\n\n{_safe_msg}"
                    )
elif section == "Security":
    # ── Security route guard (live SQL recheck, fail-closed) ───────────────
    if not _check_is_admin(viewer_token):
        st.error("Access denied. Security administration requires an "
                 "administrator account.")
        st.stop()
    # ── Security Administration Console ─────────────────────────
    _PROTECTED_ADMIN = "ruchir.agrawal@omc.com"

    st.markdown(
        '<div class="mars-header">'
        '<h1>Security Administration</h1>'
        '<div class="subtitle">Admin Only &mdash; '
        'Agency Access Management</div></div>',
        unsafe_allow_html=True,
    )

    # ── Feedback from previous action ────────────────────────────
    if "_sec_feedback" in st.session_state:
        _fb_t, _fb_m = st.session_state.pop("_sec_feedback")
        getattr(st, _fb_t, st.info)(_fb_m)

    # ── Fetch security data ──────────────────────────────────────
    _sec_errors = []
    try:
        _df_mappings = _run_security_query(
            viewer_token,
            f"SELECT user_email, agency_name FROM {AGENCY_MAP_TABLE} "
            "ORDER BY user_email, agency_name",
        )
    except Exception as _e:
        _sec_errors.append(("User-Agency Mappings", _e))
        _df_mappings = pd.DataFrame(columns=["user_email", "agency_name"])

    try:
        _df_sec_admins = _run_security_query(
            viewer_token,
            f"SELECT user_email FROM {ADMIN_TABLE} ORDER BY user_email",
        )
    except Exception as _e:
        _sec_errors.append(("Administrators", _e))
        _df_sec_admins = pd.DataFrame(columns=["user_email"])

    try:
        _df_avail = _run_security_query(
            viewer_token,
            f"SELECT DISTINCT `Agency Name` AS agency_name FROM {TABLE} "
            "ORDER BY agency_name",
        )
    except Exception as _e:
        _sec_errors.append(("Available Agencies", _e))
        _df_avail = pd.DataFrame(columns=["agency_name"])

    _agency_opts = sorted(
        v for v in _df_avail["agency_name"].dropna().tolist()
        if isinstance(v, str) and v.strip()
    )

    # ── KPI cards ──────────────────────────────────────────────────
    _n_users = (
        _df_mappings["user_email"].nunique()
        if not _df_mappings.empty
        else 0
    )
    _n_maps = len(_df_mappings)
    _n_avail_ct = len(_df_avail)
    _n_sec_admins = len(_df_sec_admins)

    _sec_kpis = [
        (_n_users, "Mapped Users", "#EFF6FF", "#3B82F6", "\U0001F465"),
        (_n_maps, "Agency Mappings", "#FFF1F2", "#E11D3F", "\U0001F517"),
        (_n_avail_ct, "Available Agencies", "#FFFBEB", "#F59E0B", "\U0001F3E2"),
        (_n_sec_admins, "Administrators", "#F0FDFA", "#0D9488", "\U0001F6E1\uFE0F"),
    ]
    _kc = st.columns(4)
    for _i, (_val, _lbl, _bg, _acc, _ico) in enumerate(_sec_kpis):
        _kc[_i].markdown(
            f'<div class="kpi-card" style="background:{_bg};'
            f'border-left:4px solid {_acc}">'
            f'<div class="kpi-icon">{_ico}</div><div>'
            f'<div class="kpi-value">{_val}</div>'
            f'<div class="kpi-label">{_lbl}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════
    #  SECTION 1 — User-to-Agency Access
    # ══════════════════════════════════════════════════════════════
    st.markdown("#### User-to-Agency Access")
    st.caption(
        "Each row shows a user and every agency they can access. "
        "Click \u00d7 beside an agency to revoke that single mapping."
    )
    # ── Confirmation: revoke mapping ────────────────────────────
    if st.session_state.get("_sec_confirm_del"):
        _pd = st.session_state["_sec_confirm_del"]
        st.warning(
            f"Revoke **{_pd['agency']}** access from "
            f"**{_pd['email']}**?"
        )
        _cd1, _cd2, _ = st.columns([1, 1, 6])
        if _cd1.button("\u2713 Confirm", key="_sec_del_ok"):
            if not _check_is_admin(viewer_token):
                st.session_state["_sec_feedback"] = (
                    "error", "Authorization failed.",
                )
            else:
                try:
                    _exec_secure_sql(
                        viewer_token,
                        f"DELETE FROM {AGENCY_MAP_TABLE} "
                        "WHERE user_email = %(email)s "
                        "AND agency_name = %(agency)s",
                        {"email": _pd["email"],
                         "agency": _pd["agency"]},
                    )
                    st.session_state["_sec_feedback"] = (
                        "success",
                        f"Revoked {_pd['agency']} from "
                        f"{_pd['email']}.",
                    )
                except Exception as _e:
                    st.session_state["_sec_feedback"] = (
                        "error",
                        "Delete failed: "
                        + _scrub_sql_error(_e),
                    )
            del st.session_state["_sec_confirm_del"]
            st.rerun()
        if _cd2.button("\u2717 Cancel", key="_sec_del_no"):
            del st.session_state["_sec_confirm_del"]
            st.rerun()

    # ── Render user-agency table ──────────────────────────────
    if _df_mappings.empty:
        st.info(
            "No user-to-agency mappings configured yet. "
            "Use the form below to grant agency access."
        )
    else:
        _grouped = _df_mappings.groupby("user_email", sort=True)
        for _user, _grp in _grouped:
            _agencies = _grp["agency_name"].tolist()
            _cols = st.columns([3] + [2] * len(_agencies))
            with _cols[0]:
                st.markdown(f"**{_user}**")
            for _j, _ag in enumerate(_agencies):
                with _cols[_j + 1]:
                    _hk = hashlib.md5(
                        f"{_user}|{_ag}".encode()
                    ).hexdigest()[:8]
                    if st.button(
                        f"\u00d7 {_ag}",
                        key=f"_rx_{_hk}",
                    ):
                        st.session_state[
                            "_sec_confirm_del"
                        ] = {"email": _user, "agency": _ag}
                        st.rerun()

    # ── Add user access (multiselect) ─────────────────────────
    st.markdown("")
    with st.expander("Grant Agency Access", expanded=False):
        if st.session_state.get("_sec_confirm_add"):
            _pa = st.session_state["_sec_confirm_add"]
            _ag_list = ", ".join(
                f"**{a}**" for a in _pa["agencies"]
            )
            st.info(
                f"Grant {_ag_list} to **{_pa['email']}**?"
            )
            _ca1, _ca2, _ = st.columns([1, 1, 6])
            if _ca1.button(
                "\u2713 Confirm", key="_sec_add_ok"
            ):
                if not _check_is_admin(viewer_token):
                    st.session_state["_sec_feedback"] = (
                        "error", "Authorization failed.",
                    )
                else:
                    _req_agencies = list(dict.fromkeys(_pa["agencies"]))
                    _added = []
                    _skipped = []
                    _fail = None
                    try:
                        _sel_params = {"email": _pa["email"]}
                        _sel_placeholders = []
                        for _ix, _ag in enumerate(_req_agencies):
                            _pname = f"agency_{_ix}"
                            _sel_params[_pname] = _ag
                            _sel_placeholders.append(f"%({_pname})s")
                        _, _existing_rows = _exec_secure_sql(
                            viewer_token,
                            f"SELECT agency_name FROM {AGENCY_MAP_TABLE} "
                            "WHERE user_email = %(email)s AND agency_name IN ("
                            + ", ".join(_sel_placeholders)
                            + ")",
                            _sel_params,
                        )
                        _existing = {
                            _r[0] if isinstance(_r, (list, tuple)) else _r
                            for _r in _existing_rows
                        }
                        _skipped = [
                            _ag for _ag in _req_agencies if _ag in _existing
                        ]
                        _added = [
                            _ag for _ag in _req_agencies if _ag not in _existing
                        ]
                        if _added:
                            _ins_parts = []
                            _ins_params = {"email": _pa["email"]}
                            for _ix, _ag in enumerate(_added):
                                _pname = f"agency_new_{_ix}"
                                _ins_params[_pname] = _ag
                                _ins_parts.append(
                                    f"SELECT %(email)s, %({_pname})s"
                                )
                            _exec_secure_sql(
                                viewer_token,
                                f"INSERT INTO {AGENCY_MAP_TABLE} "
                                "(user_email, agency_name) "
                                + " UNION ALL ".join(_ins_parts),
                                _ins_params,
                            )
                    except Exception as _e:
                        _fail = _scrub_sql_error(_e)
                    _parts = []
                    if _added:
                        _parts.append(
                            f"Added: {', '.join(_added)}"
                        )
                    if _skipped:
                        _parts.append(
                            f"Already existed: "
                            f"{', '.join(_skipped)}"
                        )
                    if _fail:
                        _parts.append(f"Error: {_fail}")
                    st.session_state["_sec_feedback"] = (
                        "error" if _fail and not _added
                        else "warning" if _fail
                        else "success" if _added
                        else "warning",
                        (
                            f"{_pa['email']}: " + "; ".join(_parts)
                            if _parts else "No changes."
                        ),
                    )
                del st.session_state["_sec_confirm_add"]
                st.rerun()
            if _ca2.button(
                "\u2717 Cancel", key="_sec_add_no"
            ):
                del st.session_state["_sec_confirm_add"]
                st.rerun()
        else:
            if not _agency_opts:
                st.caption("No agencies available.")
            else:
                with st.form(
                    "_sec_add_form", clear_on_submit=True
                ):
                    _inp_email = st.text_input(
                        "User email",
                        placeholder="user@example.com",
                    )
                    _inp_agencies = st.multiselect(
                        "Agencies",
                        options=_agency_opts,
                    )
                    _add_sub = st.form_submit_button(
                        "Grant Access"
                    )
                if _add_sub:
                    _clean = _inp_email.strip().lower()
                    if not _clean or "@" not in _clean:
                        st.session_state["_sec_feedback"] = (
                            "error",
                            "Invalid email \u2014 "
                            "must contain @.",
                        )
                        st.rerun()
                    elif not _inp_agencies:
                        st.session_state["_sec_feedback"] = (
                            "error",
                            "Select at least one agency.",
                        )
                        st.rerun()
                    else:
                        st.session_state[
                            "_sec_confirm_add"
                        ] = {
                            "email": _clean,
                            "agencies": list(_inp_agencies),
                        }
                        st.rerun()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════
    #  SECTION 2 — Administrators
    # ══════════════════════════════════════════════════════════════
    st.markdown("#### Administrators")
    st.caption(
        "Administrators have full access to every agency. "
        "The original administrator cannot be removed."
    )

    # ── Confirmation: remove admin ─────────────────────────────
    if st.session_state.get("_sec_confirm_del_admin"):
        _pa_adm = st.session_state["_sec_confirm_del_admin"]
        st.warning(
            f"Remove administrator **{_pa_adm['email']}**?"
        )
        _ad1, _ad2, _ = st.columns([1, 1, 6])
        if _ad1.button("\u2713 Confirm", key="_sec_adm_del_ok"):
            if not _check_is_admin(viewer_token):
                st.session_state["_sec_feedback"] = (
                    "error", "Authorization failed.",
                )
            else:
                try:
                    _live_admins = _run_security_query(
                        viewer_token,
                        f"SELECT user_email FROM {ADMIN_TABLE} "
                        "ORDER BY user_email",
                    )
                    _live_admin_list = [
                        str(_v).strip().lower()
                        for _v in _live_admins["user_email"].tolist()
                    ]
                    if _pa_adm["email"] == _PROTECTED_ADMIN:
                        st.session_state["_sec_feedback"] = (
                            "error",
                            "The original administrator cannot "
                            "be removed.",
                        )
                    elif _pa_adm["email"] not in _live_admin_list:
                        st.session_state["_sec_feedback"] = (
                            "warning",
                            "That administrator was already removed.",
                        )
                    elif len(_live_admin_list) <= 1:
                        st.session_state["_sec_feedback"] = (
                            "error",
                            "Cannot remove the last administrator.",
                        )
                    else:
                        _exec_secure_sql(
                            viewer_token,
                            f"DELETE FROM {ADMIN_TABLE} "
                            "WHERE user_email = %(email)s",
                            {"email": _pa_adm["email"]},
                        )
                        st.session_state["_sec_feedback"] = (
                            "success",
                            f"Removed administrator: "
                            f"{_pa_adm['email']}.",
                        )
                except Exception as _e:
                    st.session_state["_sec_feedback"] = (
                        "error",
                        "Delete failed: "
                        + _scrub_sql_error(_e),
                    )
            del st.session_state["_sec_confirm_del_admin"]
            st.rerun()
        if _ad2.button("\u2717 Cancel", key="_sec_adm_del_no"):
            del st.session_state["_sec_confirm_del_admin"]
            st.rerun()

    # ── Confirmation: add admin ───────────────────────────────
    if st.session_state.get("_sec_confirm_add_admin"):
        _pa_adm = st.session_state["_sec_confirm_add_admin"]
        st.warning(
            f"Add **{_pa_adm['email']}** as administrator?"
        )
        _aa1, _aa2, _ = st.columns([1, 1, 6])
        if _aa1.button("\u2713 Confirm", key="_sec_adm_add_ok"):
            if not _check_is_admin(viewer_token):
                st.session_state["_sec_feedback"] = (
                    "error", "Authorization failed.",
                )
            else:
                _dup = False
                try:
                    _, _dr = _exec_secure_sql(
                        viewer_token,
                        f"SELECT 1 FROM {ADMIN_TABLE} "
                        "WHERE user_email = %(email)s "
                        "LIMIT 1",
                        {"email": _pa_adm["email"]},
                    )
                    _dup = bool(_dr)
                except Exception:
                    pass
                if _dup:
                    st.session_state["_sec_feedback"] = (
                        "warning",
                        "Already an administrator.",
                    )
                else:
                    try:
                        _exec_secure_sql(
                            viewer_token,
                            f"INSERT INTO {ADMIN_TABLE} "
                            "(user_email) "
                            "VALUES (%(email)s)",
                            {"email": _pa_adm["email"]},
                        )
                        st.session_state["_sec_feedback"] = (
                            "success",
                            f"Added administrator: "
                            f"{_pa_adm['email']}.",
                        )
                    except Exception as _e:
                        st.session_state["_sec_feedback"] = (
                            "error",
                            "Insert failed: "
                            + _scrub_sql_error(_e),
                        )
            del st.session_state["_sec_confirm_add_admin"]
            st.rerun()
        if _aa2.button("\u2717 Cancel", key="_sec_adm_add_no"):
            del st.session_state["_sec_confirm_add_admin"]
            st.rerun()

    # ── Render admin table ───────────────────────────────────
    if _df_sec_admins.empty:
        st.warning("No administrators found.")
    else:
        for _, _row in _df_sec_admins.iterrows():
            _ae = _row["user_email"]
            _ac1, _ac2 = st.columns([6, 1])
            with _ac1:
                st.markdown(f"\U0001F6E1\uFE0F `{_ae}`")
            with _ac2:
                if _ae == _PROTECTED_ADMIN:
                    st.markdown(
                        '<span title="Protected">'
                        "\U0001F512</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(
                        "\u00d7",
                        key=f"_adm_rm_{hashlib.md5(_ae.encode()).hexdigest()[:8]}",
                    ):
                        st.session_state[
                            "_sec_confirm_del_admin"
                        ] = {"email": _ae}
                        st.rerun()

    # ── Add administrator ───────────────────────────────────
    with st.expander("Add Administrator", expanded=False):
        with st.form("_sec_add_admin_form", clear_on_submit=True):
            _adm_email = st.text_input(
                "Administrator email",
                placeholder="admin@example.com",
            )
            _adm_sub = st.form_submit_button("Add Administrator")
        if _adm_sub:
            _adm_clean = _adm_email.strip().lower()
            if not _adm_clean or "@" not in _adm_clean:
                st.session_state["_sec_feedback"] = (
                    "error",
                    "Invalid email \u2014 must contain @.",
                )
                st.rerun()
            else:
                if _adm_clean in [
                    str(_v).strip().lower()
                    for _v in _df_sec_admins["user_email"].tolist()
                ]:
                    st.session_state["_sec_feedback"] = (
                        "warning",
                        "Already an administrator.",
                    )
                    st.rerun()
                else:
                    st.session_state[
                        "_sec_confirm_add_admin"
                    ] = {"email": _adm_clean}
                    st.rerun()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════
    #  SECTION 3 — Available Agencies
    # ══════════════════════════════════════════════════════════════
    st.markdown("#### Available Agencies")
    if _df_avail.empty:
        st.warning("No agencies found in the source table.")
    else:
        _ag_display = []
        for _, _row in _df_avail.iterrows():
            _a = _row["agency_name"]
            _ag_display.append(
                "(Blank)"
                if pd.isna(_a) or str(_a).strip() == ""
                else str(_a)
            )
        st.dataframe(
            pd.DataFrame({"Agency Name": _ag_display}),
            use_container_width=True,
            hide_index=True,
        )

    # ── Query errors (credentials scrubbed) ──────────────────────
    if _sec_errors:
        st.markdown("---")
        for _label, _exc in _sec_errors:
            _msg = str(_exc)
            if any(
                s in _msg.lower()
                for s in ("token", "bearer", "credential", "secret")
            ):
                _msg = (
                    "A connection error occurred. "
                    "Check warehouse availability and permissions."
                )
            st.error(f"Could not load {_label}: {_msg}")

    # ── Refresh button ───────────────────────────────────────────
    st.markdown("")
    st.button(
        "\u21bb Refresh Security Data",
        key="_sec_refresh",
    )
else:
    # ── Header ────────────────────────────────────────────────────────────────────
    period_label = compute_reporting_period(filtered["Month"])
    st.markdown(
        f'<div class="mars-header">'
        f'<h1>Mars Snacking</h1>'
        f'<div class="subtitle">Time Utilization &nbsp;|&nbsp; '
        f'<span class="period-badge">{period_label}</span>'
        f' &nbsp;|&nbsp; {len(filtered):,} records</div></div>',
        unsafe_allow_html=True,
    )

    # ── Navigation Tabs ─────────────────────────────────────────────────────────
    tab_summary, tab_details = st.tabs(["Summary", "Product Details"])

    with tab_summary:
        # Display-only HTML dashboard rendered via components.html().
        # Data is injected as JSON into the template; no bidirectional
        # component messaging. Cross-filtering is postponed.
        _filter_display = [
            ("Year", [str(int(v)) for v in sel_year]),
            ("Month", [month_map.get(v, v) for v in sel_month]),
            ("Product", list(sel_product)),
            ("Project", list(sel_project)),
            ("Department", list(sel_dept)),
            ("Employee Title", list(sel_title)),
            ("Employee Name", list(sel_emp)),
        ]
        _breadcrumb_chips = []
        for _fb_label, _fb_vals in _filter_display:
            if not _fb_vals:
                continue
            _fb_esc = [escape(str(v)) for v in _fb_vals]
            if len(_fb_esc) <= 2:
                _fb_text = ", ".join(_fb_esc)
            else:
                _fb_text = f"{_fb_esc[0]}, {_fb_esc[1]}, +{len(_fb_esc) - 2} more"
            _breadcrumb_chips.append(
                f'<span style="display:inline-block;background:#edf2fa;'
                f'border:1px solid #d4dfef;border-radius:16px;padding:4px 12px;'
                f'font-size:.75rem;color:#1b365d;white-space:nowrap">'
                f'<strong style="font-weight:650;margin-right:3px">'
                f'{escape(_fb_label)}:</strong> {_fb_text}</span>'
            )
        if _breadcrumb_chips:
            st.markdown(
                '<div style="display:flex;flex-wrap:wrap;gap:6px;margin:0 0 8px">'
                + " ".join(_breadcrumb_chips) + '</div>',
                unsafe_allow_html=True,
            )
        _summary_rows = []
        for _, rec in filtered.iterrows():
            _summary_rows.append({
                "lob": _display_label(rec["Line of Business"]),
                "tier": _display_label(rec["Project Tier"]),
                "product": _display_label(rec["Product"]),
                "agency": _display_label(rec["Agency Name"]),
                "brand": _display_label(rec["Brand"]),
                "campaign": _display_label(rec["Campaign"]),
                "hours": float(rec["Hours"]) if pd.notna(rec["Hours"]) else 0.0,
            })
        _summary_payload = json.dumps(
            {"rows": _summary_rows}, ensure_ascii=False, default=str
        ).replace("<", "\\u003c")
        _summary_tpl = (
            Path(__file__).resolve().parent / "components" / "summary" / "index.html"
        ).read_text(encoding="utf-8")
        components.html(
            _summary_tpl.replace("/*__SUMMARY_DATA__*/null", _summary_payload),
            height=950, scrolling=False)

    # ── Product Details helpers ────────────────────────────────────────────────────
    HIERARCHY = ["Product", "Project", "Department", "Employee Title", "Employee Name"]
    BLANK = "(Blank)"


    def _display(val):
        """Return a display-safe string; blanks become (Blank)."""
        if pd.isna(val) or str(val).strip() == "":
            return BLANK
        return str(val)


    def _period_columns(series: pd.Series, mode: str):
        """Return (ordered column names, mapping from raw Month to col name)."""
        parsed = pd.to_datetime(series, format="%m/%Y", errors="coerce")
        raw_vals = series.unique()
        mapping = {}  # raw Month value -> display column name
        for raw in raw_vals:
            dt = pd.to_datetime(raw, format="%m/%Y", errors="coerce")
            if pd.isna(dt):
                mapping[raw] = "Unknown"
            elif mode == "month":
                mapping[raw] = dt.strftime("%b %Y")
            elif mode == "quarter":
                mapping[raw] = f"Q{dt.quarter} {dt.year}"
            else:  # year
                mapping[raw] = str(dt.year)
        # Determine chronological order
        valid = parsed.dropna().sort_values().dt
        if mode == "month":
            ordered = valid.strftime("%b %Y").unique().tolist()
        elif mode == "quarter":
            ordered = ("Q" + valid.quarter.astype(str) + " " + valid.year.astype(str)).unique().tolist()
        else:
            ordered = valid.year.astype(str).unique().tolist()
        if "Unknown" in mapping.values() and "Unknown" not in ordered:
            ordered.append("Unknown")
        return ordered, mapping


    def _build_matrix(data: pd.DataFrame, mode: str):
        """Build the hierarchical matrix DataFrame.

        Returns (matrix_rows: list[dict], period_cols: list[str]).
        Each dict has: level, label, is_leaf, and one key per period_col + 'Total'.
        """
        period_cols, month_to_col = _period_columns(data["Month"], mode)
        data = data.copy()
        data["_pcol"] = data["Month"].map(month_to_col)
        rows = []

        def _agg(subset, pcols):
            """Aggregate Hours by period column."""
            result = {c: 0.0 for c in pcols}
            if subset.empty:
                result["Total"] = 0.0
                return result
            for col_name, grp in subset.groupby("_pcol"):
                if col_name in result:
                    result[col_name] = grp["Hours"].sum()
            result["Total"] = subset["Hours"].sum()
            return result

        # Grand total
        gt = _agg(data, period_cols)
        gt_row = {"level": -1, "label": "Grand Total", "is_leaf": False}
        gt_row.update(gt)

        for prod, g1 in data.groupby(HIERARCHY[0], dropna=False):
            agg1 = _agg(g1, period_cols)
            r1 = {"level": 0, "label": _display(prod), "is_leaf": False}
            r1.update(agg1)
            rows.append(r1)
            for proj, g2 in g1.groupby(HIERARCHY[1], dropna=False):
                agg2 = _agg(g2, period_cols)
                r2 = {"level": 1, "label": _display(proj), "is_leaf": False}
                r2.update(agg2)
                rows.append(r2)
                for dept, g3 in g2.groupby(HIERARCHY[2], dropna=False):
                    agg3 = _agg(g3, period_cols)
                    r3 = {"level": 2, "label": _display(dept), "is_leaf": False}
                    r3.update(agg3)
                    rows.append(r3)
                    for title, g4 in g3.groupby(HIERARCHY[3], dropna=False):
                        agg4 = _agg(g4, period_cols)
                        r4 = {"level": 3, "label": _display(title), "is_leaf": False}
                        r4.update(agg4)
                        rows.append(r4)
                        for emp, g5 in g4.groupby(HIERARCHY[4], dropna=False):
                            agg5 = _agg(g5, period_cols)
                            r5 = {"level": 4, "label": _display(emp), "is_leaf": True}
                            r5.update(agg5)
                            rows.append(r5)

        rows.append(gt_row)  # grand total at bottom
        return rows, period_cols


    def _render_interactive_matrix(rows: list, period_cols: list, expand_level: int) -> None:
        """Embed a client-side interactive tree; only viewer-authorized aggregates cross into it."""
        template = (Path(__file__).resolve().parent / "components" / "matrix.html").read_text(encoding="utf-8")
        payload = json.dumps({"rows": rows, "periods": period_cols, "expandLevel": expand_level},
                             ensure_ascii=False, default=str).replace("<", "\\u003c")
        components.html(template.replace("/*__MATRIX_DATA__*/null", payload), height=660, scrolling=False)


    def _matrix_to_csv(rows: list, period_cols: list) -> str:
        """Convert matrix rows to CSV string for download."""
        HCOLS = ["Product", "Project", "Department", "Employee Title", "Employee Name"]
        buf = io.StringIO()
        header = HCOLS + period_cols + ["Total"]
        buf.write(",".join(header) + "\n")
        for r in rows:
            lvl = r["level"]
            h_cells = [""] * 5
            if lvl == -1:
                h_cells[0] = '"Grand Total"'
            else:
                h_cells[min(lvl, 4)] = f'"{r["label"]}"'
            vals = h_cells.copy()
            for pc in period_cols:
                vals.append(f'{r.get(pc, 0):.2f}')
            vals.append(f'{r.get("Total", 0):.2f}')
            buf.write(",".join(vals) + "\n")
        return buf.getvalue()


    with tab_details:
        # ── Controls: time grouping at left, CSV export at top right ──
        ctrl1, ctrl2 = st.columns([1, 3])
        with ctrl1:
            time_mode = st.selectbox("Time grouping", ["Month", "Quarter", "Year"],
                                      index=0, key="pd_time_mode")

        # ── Build matrix ──
        matrix_rows, pcols = _build_matrix(filtered, time_mode.lower())

        with ctrl2:
            _, download_col = st.columns([3, 1])
            csv_data = _matrix_to_csv(matrix_rows, pcols)
            with download_col:
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name=f"mars_snacking_product_details_{time_mode.lower()}.csv",
                    mime="text/csv",
                    key="pd_csv_dl",
                    use_container_width=True,
                )

        # ── Render: default to Product + Project, with per-row controls in JS ──
        _render_interactive_matrix(matrix_rows, pcols, expand_level=1)

        # ── Totals verification line ──
        gt = matrix_rows[-1]  # grand total row
        st.caption(f"Grand Total: {gt['Total']:,.2f} Hours | "
                   f"{len(filtered):,} records | Filters applied to both tabs")
