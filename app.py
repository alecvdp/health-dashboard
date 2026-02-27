import os
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
MANUAL_LOG = DATA_DIR / "manual_log.csv"
OURA_BASE = "https://api.ouraring.com/v2/usercollection"

MANUAL_COLUMNS = [
    "date", "nicotine_pouches", "vape_puffs", "caffeine_mg",
    "cpap_ahi", "cpap_hours", "cpap_leak_95", "notes",
]

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tighten top padding */
    .block-container { padding-top: 1.5rem; }

    /* Metric card */
    [data-testid="metric-container"] {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #2a2a3e;
    }

    /* Tab strip */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 6px 18px;
    }
</style>
""", unsafe_allow_html=True)


# ── Data helpers ──────────────────────────────────────────────────────────────
def init_manual_log() -> pd.DataFrame:
    if not MANUAL_LOG.exists():
        pd.DataFrame(columns=MANUAL_COLUMNS).to_csv(MANUAL_LOG, index=False)
    df = pd.read_csv(MANUAL_LOG, dtype=str)
    for col in MANUAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[MANUAL_COLUMNS]


def save_manual_log(df: pd.DataFrame):
    df.to_csv(MANUAL_LOG, index=False)


def upsert_row(df: pd.DataFrame, new_row: dict) -> pd.DataFrame:
    """Insert or update a row by date."""
    mask = df["date"] == str(new_row["date"])
    if mask.any():
        for k, v in new_row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def parse_duration_hours(value) -> float | None:
    """Convert HH:MM:SS, HH:MM, or plain numeric strings to decimal hours."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return None
    # Already a plain number
    try:
        return float(s)
    except ValueError:
        pass
    # HH:MM:SS or HH:MM
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
            return h + m / 60 + sec / 3600
        elif len(parts) == 2:
            h, m = int(parts[0]), int(parts[1])
            return h + m / 60
    except Exception:
        pass
    return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_oura(endpoint: str, token: str, start: str, end: str):
    try:
        r = requests.get(
            f"{OURA_BASE}/{endpoint}",
            headers={"Authorization": f"Bearer {token}"},
            params={"start_date": start, "end_date": end},
            timeout=12,
        )
        if r.status_code == 200:
            return r.json().get("data", [])
        elif r.status_code == 401:
            return {"error": "Invalid API token — check your token at cloud.ouraring.com"}
        return {"error": f"Oura API returned {r.status_code}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out — check your internet connection"}
    except Exception as e:
        return {"error": str(e)}


def oura_records_to_df(records, extra_fields: list[str]) -> pd.DataFrame:
    """Flatten Oura API records into a DataFrame."""
    if isinstance(records, dict) and "error" in records:
        st.error(f"⚠️ Oura: {records['error']}")
        return pd.DataFrame()
    if not records:
        return pd.DataFrame()
    rows = []
    for d in records:
        day_str = d.get("day") or (d.get("timestamp") or "")[:10]
        row = {"date": pd.Timestamp(day_str)}
        for f in extra_fields:
            val = d.get(f)
            # Flatten contributor sub-dicts
            if isinstance(val, dict):
                for k, v in val.items():
                    row[f"{f}_{k}"] = v
            else:
                row[f] = val
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def line_chart(df, x, y, title, color="#60a5fa", yunit="", yrange=None):
    fig = px.line(df, x=x, y=y, markers=True,
                  color_discrete_sequence=[color], template="plotly_dark")
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(text=title, font_size=14),
        xaxis_title="", yaxis_title=yunit,
        showlegend=False,
    )
    if yrange:
        layout["yaxis"] = dict(range=yrange, title=yunit)
    fig.update_layout(**layout)
    fig.update_traces(line_width=2)
    return fig


def bar_chart(df, x, y, title, color="#60a5fa", yunit="", hline=None, hline_label=""):
    fig = px.bar(df, x=x, y=y, title=title,
                 color_discrete_sequence=[color], template="plotly_dark")
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="#6b7280",
                      annotation_text=hline_label, annotation_position="top right")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=36, b=0),
        xaxis_title="", yaxis_title=yunit,
    )
    return fig


def score_bands(fig):
    """Add green/yellow/red score bands to a figure with a 0–100 y axis."""
    fig.add_hrect(y0=85, y1=100, fillcolor="#22c55e", opacity=0.06, line_width=0)
    fig.add_hrect(y0=70, y1=85, fillcolor="#eab308", opacity=0.06, line_width=0)
    fig.add_hrect(y0=0, y1=70, fillcolor="#ef4444", opacity=0.06, line_width=0)
    return fig


def latest_val(df, col):
    if df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s.iloc[-1] if len(s) else None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Seed from .env on first load
    if "oura_token" not in st.session_state:
        st.session_state["oura_token"] = os.getenv("OURA_TOKEN", "")

    oura_token = st.text_input(
        "Oura API Token",
        type="password",
        value=st.session_state["oura_token"],
        placeholder="Paste your personal access token",
        help="Or set OURA_TOKEN in a .env file to load automatically",
    )
    if oura_token:
        st.session_state["oura_token"] = oura_token

    st.markdown("---")
    st.markdown("### 📅 Date Range")
    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("From", value=date.today() - timedelta(days=30))
    with col_b:
        end_date = st.date_input("To", value=date.today())

    st.markdown("---")
    st.markdown("### ✏️ Log Entry")

    with st.form("manual_entry", clear_on_submit=True):
        log_date = st.date_input("Date", value=date.today())
        nicotine = st.number_input("Nicotine pouches", min_value=0, max_value=99, step=1, value=0)
        vape_puffs = st.number_input("Vape puffs", min_value=0, max_value=9999, step=1, value=0)
        caffeine = st.number_input("Caffeine (mg)", min_value=0, max_value=3000, step=25, value=0)
        cpap_ahi = st.number_input("CPAP AHI", min_value=0.0, max_value=999.0, step=0.1,
                                    format="%.1f", value=0.0)
        cpap_hours = st.number_input("CPAP hours used", min_value=0.0, max_value=24.0, step=0.25,
                                      format="%.2f", value=0.0)
        cpap_leak = st.number_input("Leak 95th %tile (L/min)", min_value=0.0, max_value=999.0,
                                     step=0.5, format="%.1f", value=0.0)
        notes = st.text_input("Notes")
        submitted = st.form_submit_button("💾 Save", use_container_width=True, type="primary")

    if submitted:
        df_log = init_manual_log()
        new_row = {
            "date": str(log_date),
            "nicotine_pouches": nicotine if nicotine > 0 else None,
            "vape_puffs": vape_puffs if vape_puffs > 0 else None,
            "caffeine_mg": caffeine if caffeine > 0 else None,
            "cpap_ahi": cpap_ahi if cpap_ahi > 0 else None,
            "cpap_hours": cpap_hours if cpap_hours > 0 else None,
            "cpap_leak_95": cpap_leak if cpap_leak > 0 else None,
            "notes": notes or None,
        }
        df_log = upsert_row(df_log, new_row)
        save_manual_log(df_log)
        st.success(f"Saved {log_date}")
        st.cache_data.clear()


# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🩺 Health Dashboard")

start_str = start_date.isoformat()
end_str = end_date.isoformat()

# ── Fetch Oura ────────────────────────────────────────────────────────────────
token = st.session_state.get("oura_token", "")
sleep_raw, readiness_raw, activity_raw = [], [], []

if token:
    with st.spinner("Fetching Oura data…"):
        sleep_raw = fetch_oura("daily_sleep", token, start_str, end_str)
        readiness_raw = fetch_oura("daily_readiness", token, start_str, end_str)
        activity_raw = fetch_oura("daily_activity", token, start_str, end_str)
else:
    st.info("👈  Enter your Oura API token in the sidebar to load sleep, readiness, and activity data.")

df_sleep = oura_records_to_df(sleep_raw, ["score", "contributors"])
df_readiness = oura_records_to_df(readiness_raw, ["score", "contributors", "temperature_deviation"])
df_activity = oura_records_to_df(activity_raw, ["score", "steps", "active_calories", "equivalent_walking_distance"])

# ── Load manual data filtered to date range ───────────────────────────────────
df_all_manual = init_manual_log()
df_all_manual["date"] = pd.to_datetime(df_all_manual["date"], errors="coerce")
df_manual = df_all_manual[
    (df_all_manual["date"] >= pd.Timestamp(start_date)) &
    (df_all_manual["date"] <= pd.Timestamp(end_date))
].copy().sort_values("date").reset_index(drop=True)

for col in ["nicotine_pouches", "vape_puffs", "caffeine_mg", "cpap_ahi", "cpap_hours", "cpap_leak_95"]:
    df_manual[col] = pd.to_numeric(df_manual[col], errors="coerce")


# ── Tabs ──────────────────────────────────────────────────────────────────────
(tab_overview, tab_sleep, tab_readiness,
 tab_activity, tab_lifestyle, tab_cpap, tab_data) = st.tabs([
    "📊 Overview", "😴 Sleep", "⚡ Readiness",
    "🏃 Activity", "☕ Lifestyle", "😮‍💨 CPAP", "🗃️ Data",
])


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Most Recent Values")

    last_manual = df_manual.dropna(subset=["date"]).iloc[-1] if not df_manual.empty else None

    def safe_int(v):
        try:
            return int(float(v))
        except Exception:
            return None

    def safe_float(v, decimals=1):
        try:
            return round(float(v), decimals)
        except Exception:
            return None

    sleep_score = safe_int(latest_val(df_sleep, "score"))
    readiness_score = safe_int(latest_val(df_readiness, "score"))
    steps = safe_int(latest_val(df_activity, "steps"))
    nicotine_latest = safe_int(last_manual["nicotine_pouches"]) if last_manual is not None else None
    vape_latest = safe_int(last_manual["vape_puffs"]) if last_manual is not None else None
    caffeine_latest = safe_int(last_manual["caffeine_mg"]) if last_manual is not None else None
    ahi_latest = safe_float(last_manual["cpap_ahi"]) if last_manual is not None else None

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("😴 Sleep Score", sleep_score if sleep_score is not None else "—")
    m2.metric("⚡ Readiness", readiness_score if readiness_score is not None else "—")
    m3.metric("👟 Steps", f"{steps:,}" if steps is not None else "—")
    m4.metric("🫧 Pouches", f"{nicotine_latest}" if nicotine_latest is not None else "—")
    m5.metric("💨 Vape puffs", f"{vape_latest}" if vape_latest is not None else "—")
    m6.metric("☕ Caffeine", f"{caffeine_latest} mg" if caffeine_latest is not None else "—")
    m7.metric("😮‍💨 CPAP AHI", f"{ahi_latest}" if ahi_latest is not None else "—")

    st.divider()
    st.subheader(f"Trends — {start_date.strftime('%b %-d')} → {end_date.strftime('%b %-d, %Y')}")

    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)

    with r1c1:
        if not df_sleep.empty and "score" in df_sleep:
            st.plotly_chart(line_chart(df_sleep, "date", "score", "Sleep Score",
                                       "#818cf8", yrange=[0, 100]), use_container_width=True)
        else:
            st.caption("Sleep — no data")

    with r1c2:
        if not df_readiness.empty and "score" in df_readiness:
            st.plotly_chart(line_chart(df_readiness, "date", "score", "Readiness Score",
                                       "#4ade80", yrange=[0, 100]), use_container_width=True)
        else:
            st.caption("Readiness — no data")

    with r1c3:
        if not df_activity.empty and "steps" in df_activity:
            st.plotly_chart(bar_chart(df_activity, "date", "steps", "Daily Steps",
                                      "#f59e0b", "steps", hline=10000, hline_label="10k"), use_container_width=True)
        else:
            st.caption("Steps — no data")

    with r2c1:
        if not df_manual.empty and df_manual["nicotine_pouches"].notna().any():
            st.plotly_chart(bar_chart(df_manual, "date", "nicotine_pouches", "Pouches",
                                      "#f87171", "pouches/day"), use_container_width=True)
        else:
            st.caption("Pouches — no data logged")

    with r2c2:
        if not df_manual.empty and df_manual["vape_puffs"].notna().any():
            st.plotly_chart(bar_chart(df_manual, "date", "vape_puffs", "Vape Puffs",
                                      "#fb923c", "puffs/day"), use_container_width=True)
        else:
            st.caption("Vape — no data logged")

    with r2c3:
        if not df_manual.empty and df_manual["caffeine_mg"].notna().any():
            st.plotly_chart(bar_chart(df_manual, "date", "caffeine_mg", "Caffeine",
                                      "#fbbf24", "mg/day", hline=400, hline_label="400mg"), use_container_width=True)
        else:
            st.caption("Caffeine — no data logged")

    with r2c4:
        if not df_manual.empty and df_manual["cpap_ahi"].notna().any():
            st.plotly_chart(line_chart(df_manual, "date", "cpap_ahi", "CPAP AHI",
                                       "#60a5fa", "events/hr"), use_container_width=True)
        else:
            st.caption("CPAP — no data logged")


# ══════════════════════════════════════════════════════════════════════════════
# SLEEP
# ══════════════════════════════════════════════════════════════════════════════
with tab_sleep:
    if df_sleep.empty:
        st.info("No sleep data — add your Oura API token in the sidebar.")
    else:
        # Score over time
        fig = line_chart(df_sleep, "date", "score", "Sleep Score", "#818cf8", yrange=[0, 100])
        score_bands(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Contributors
        contrib_map = {
            "contributors_deep_sleep": "Deep Sleep",
            "contributors_rem_sleep": "REM Sleep",
            "contributors_sleep_efficiency": "Efficiency",
            "contributors_restfulness": "Restfulness",
            "contributors_sleep_latency": "Latency",
            "contributors_sleep_timing": "Timing",
            "contributors_total_sleep": "Total Sleep",
        }
        avail_contribs = [c for c in contrib_map if c in df_sleep.columns
                          and df_sleep[c].notna().any()]

        if avail_contribs:
            st.subheader("Sleep Contributors")
            colors = px.colors.qualitative.Pastel
            fig2 = go.Figure()
            for i, col in enumerate(avail_contribs):
                fig2.add_trace(go.Scatter(
                    x=df_sleep["date"], y=df_sleep[col],
                    name=contrib_map[col],
                    mode="lines+markers",
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0, 100], title="Contributor Score"),
                xaxis_title="",
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Raw Data")
        disp = df_sleep.copy()
        disp["date"] = disp["date"].dt.date
        st.dataframe(disp.sort_values("date", ascending=False).set_index("date"),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# READINESS
# ══════════════════════════════════════════════════════════════════════════════
with tab_readiness:
    if df_readiness.empty:
        st.info("No readiness data — add your Oura API token in the sidebar.")
    else:
        fig = line_chart(df_readiness, "date", "score", "Readiness Score", "#4ade80", yrange=[0, 100])
        score_bands(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Temperature deviation
        if "temperature_deviation" in df_readiness.columns and df_readiness["temperature_deviation"].notna().any():
            st.subheader("Body Temperature Deviation")
            df_temp = df_readiness[df_readiness["temperature_deviation"].notna()].copy()
            df_temp["temperature_deviation"] = pd.to_numeric(df_temp["temperature_deviation"], errors="coerce")
            fig_temp = px.bar(
                df_temp, x="date", y="temperature_deviation",
                color="temperature_deviation",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                template="plotly_dark",
            )
            fig_temp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="°C from baseline", xaxis_title="",
                margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False,
            )
            st.plotly_chart(fig_temp, use_container_width=True)

        # Readiness contributors
        contrib_map_r = {
            "contributors_activity_balance": "Activity Balance",
            "contributors_body_temperature": "Body Temp",
            "contributors_hrv_balance": "HRV Balance",
            "contributors_previous_day_activity": "Prev Day Activity",
            "contributors_previous_night": "Previous Night",
            "contributors_recovery_index": "Recovery Index",
            "contributors_resting_heart_rate": "Resting HR",
            "contributors_sleep_balance": "Sleep Balance",
        }
        avail_r = [c for c in contrib_map_r if c in df_readiness.columns
                   and df_readiness[c].notna().any()]
        if avail_r:
            st.subheader("Readiness Contributors")
            colors = px.colors.qualitative.Pastel
            fig_r = go.Figure()
            for i, col in enumerate(avail_r):
                fig_r.add_trace(go.Scatter(
                    x=df_readiness["date"], y=df_readiness[col],
                    name=contrib_map_r[col], mode="lines+markers",
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
            fig_r.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0, 100], title="Contributor Score"),
                xaxis_title="", margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        st.subheader("Raw Data")
        disp = df_readiness.copy()
        disp["date"] = disp["date"].dt.date
        st.dataframe(disp.sort_values("date", ascending=False).set_index("date"),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ACTIVITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_activity:
    if df_activity.empty:
        st.info("No activity data — add your Oura API token in the sidebar.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            if "steps" in df_activity.columns:
                df_act_steps = df_activity[pd.to_numeric(df_activity["steps"], errors="coerce").notna()].copy()
                df_act_steps["steps"] = pd.to_numeric(df_act_steps["steps"])
                fig_steps = bar_chart(df_act_steps, "date", "steps", "Daily Steps",
                                      "#f59e0b", "steps", hline=10000, hline_label="10k goal")
                st.plotly_chart(fig_steps, use_container_width=True)

        with c2:
            if "active_calories" in df_activity.columns:
                df_cal = df_activity[pd.to_numeric(df_activity["active_calories"], errors="coerce").notna()].copy()
                df_cal["active_calories"] = pd.to_numeric(df_cal["active_calories"])
                fig_cal = bar_chart(df_cal, "date", "active_calories", "Active Calories",
                                    "#f87171", "kcal")
                st.plotly_chart(fig_cal, use_container_width=True)

        if "score" in df_activity.columns:
            df_activity["score"] = pd.to_numeric(df_activity["score"], errors="coerce")
            fig_act_score = line_chart(df_activity, "date", "score", "Activity Score",
                                       "#34d399", yrange=[0, 100])
            score_bands(fig_act_score)
            st.plotly_chart(fig_act_score, use_container_width=True)

        st.subheader("Raw Data")
        disp = df_activity.copy()
        disp["date"] = disp["date"].dt.date
        st.dataframe(disp.sort_values("date", ascending=False).set_index("date"),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# LIFESTYLE (Nicotine + Caffeine)
# ══════════════════════════════════════════════════════════════════════════════
with tab_lifestyle:
    has_nic = not df_manual.empty and df_manual["nicotine_pouches"].notna().any()
    has_vape = not df_manual.empty and df_manual["vape_puffs"].notna().any()
    has_caf = not df_manual.empty and df_manual["caffeine_mg"].notna().any()

    if not has_nic and not has_vape and not has_caf:
        st.info("No lifestyle data yet — use the sidebar form to log nicotine and caffeine.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("🫧 Pouches")
            if has_nic:
                df_nic = df_manual[df_manual["nicotine_pouches"].notna()]
                st.plotly_chart(bar_chart(df_nic, "date", "nicotine_pouches",
                                          "Pouches per Day", "#f87171", "pouches"),
                                use_container_width=True)
                ma, mb = st.columns(2)
                ma.metric("Avg / day", f"{df_nic['nicotine_pouches'].mean():.1f}")
                mb.metric("Total", f"{int(df_nic['nicotine_pouches'].sum())}")
            else:
                st.caption("No data in this date range.")

        with c2:
            st.subheader("💨 Vape Puffs")
            if has_vape:
                df_vape = df_manual[df_manual["vape_puffs"].notna()]
                st.plotly_chart(bar_chart(df_vape, "date", "vape_puffs",
                                          "Puffs per Day", "#fb923c", "puffs"),
                                use_container_width=True)
                ma, mb = st.columns(2)
                ma.metric("Avg / day", f"{df_vape['vape_puffs'].mean():.0f}")
                mb.metric("Total", f"{int(df_vape['vape_puffs'].sum())}")
            else:
                st.caption("No data in this date range.")

        with c3:
            st.subheader("☕ Caffeine")
            if has_caf:
                df_caf = df_manual[df_manual["caffeine_mg"].notna()]
                st.plotly_chart(bar_chart(df_caf, "date", "caffeine_mg",
                                          "Caffeine per Day", "#fbbf24", "mg",
                                          hline=400, hline_label="400mg daily max"),
                                use_container_width=True)
                ma, mb = st.columns(2)
                ma.metric("Avg / day", f"{df_caf['caffeine_mg'].mean():.0f} mg")
                mb.metric("Peak day", f"{df_caf['caffeine_mg'].max():.0f} mg")
            else:
                st.caption("No data in this date range.")


# ══════════════════════════════════════════════════════════════════════════════
# CPAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_cpap:
    st.subheader("😮‍💨 CPAP Data")

    with st.expander("📂 Import from OSCAR CSV export"):
        st.markdown(
            "In OSCAR: open a session → **File → Export → Daily Summary CSV**. "
            "Then upload that file here."
        )
        uploaded = st.file_uploader("Upload OSCAR daily summary CSV", type=["csv"],
                                    key="oscar_upload")
        if uploaded:
            try:
                df_oscar = pd.read_csv(uploaded)
                st.write("**Preview (first 5 rows):**")
                st.dataframe(df_oscar.head())

                all_cols = ["(skip)"] + list(df_oscar.columns)

                def best_match(keywords):
                    for kw in keywords:
                        for i, c in enumerate(all_cols):
                            if kw in c.lower():
                                return i
                    return 0

                col_date = st.selectbox("Date column", all_cols,
                                        index=best_match(["date", "day"]))
                col_ahi = st.selectbox("AHI column", all_cols,
                                       index=best_match(["ahi"]))
                col_hours = st.selectbox("Usage hours column", all_cols,
                                         index=best_match(["hour", "duration", "usage"]))
                col_leak = st.selectbox("Leak 95th %tile column", all_cols,
                                        index=best_match(["leak", "95"]))

                if st.button("⬆️ Import OSCAR Data", type="primary"):
                    df_log = init_manual_log()
                    imported = 0
                    for _, row in df_oscar.iterrows():
                        if col_date == "(skip)":
                            continue
                        try:
                            entry_date = str(pd.Timestamp(row[col_date]).date())
                        except Exception:
                            continue
                        updates: dict = {"date": entry_date}
                        if col_ahi != "(skip)":
                            updates["cpap_ahi"] = row.get(col_ahi)
                        if col_hours != "(skip)":
                            updates["cpap_hours"] = parse_duration_hours(row.get(col_hours))
                        if col_leak != "(skip)":
                            updates["cpap_leak_95"] = row.get(col_leak)
                        df_log = upsert_row(df_log, updates)
                        imported += 1
                    save_manual_log(df_log)
                    st.success(f"Imported {imported} OSCAR records!")
                    st.cache_data.clear()
                    st.rerun()

            except Exception as e:
                st.error(f"Error reading file: {e}")

    # Charts
    df_cpap = df_manual[
        df_manual["cpap_ahi"].notna() |
        df_manual["cpap_hours"].notna() |
        df_manual["cpap_leak_95"].notna()
    ]

    if df_cpap.empty:
        st.info("No CPAP data yet. Import from OSCAR above or log entries via the sidebar.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("AHI Over Time")
            df_ahi = df_cpap[df_cpap["cpap_ahi"].notna()]
            if not df_ahi.empty:
                fig_ahi = px.line(df_ahi, x="date", y="cpap_ahi", markers=True,
                                  color_discrete_sequence=["#60a5fa"], template="plotly_dark")
                fig_ahi.add_hrect(y0=0, y1=5, fillcolor="#22c55e", opacity=0.07, line_width=0,
                                  annotation_text="Normal (<5)", annotation_position="top right")
                fig_ahi.add_hrect(y0=5, y1=15, fillcolor="#eab308", opacity=0.07, line_width=0,
                                  annotation_text="Mild (5–15)", annotation_position="bottom right")
                fig_ahi.add_hrect(y0=15, y1=30, fillcolor="#ef4444", opacity=0.07, line_width=0)
                fig_ahi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title="Events/hr", xaxis_title="",
                )
                st.plotly_chart(fig_ahi, use_container_width=True)
                avg_ahi = df_ahi["cpap_ahi"].mean()
                st.metric("Average AHI", f"{avg_ahi:.2f}", help="Lower is better. Goal: <5")

        with c2:
            st.subheader("Usage Hours")
            df_hrs = df_cpap[df_cpap["cpap_hours"].notna()]
            if not df_hrs.empty:
                fig_hrs = bar_chart(df_hrs, "date", "cpap_hours", "Usage per Night",
                                    "#818cf8", "hours", hline=4, hline_label="4hr minimum")
                st.plotly_chart(fig_hrs, use_container_width=True)
                avg_hrs = df_hrs["cpap_hours"].mean()
                st.metric("Average hours/night", f"{avg_hrs:.1f}")

        if df_cpap["cpap_leak_95"].notna().any():
            st.subheader("Leak Rate — 95th Percentile")
            df_leak = df_cpap[df_cpap["cpap_leak_95"].notna()]
            fig_leak = line_chart(df_leak, "date", "cpap_leak_95",
                                  "Leak Rate (95th %tile)", "#a78bfa", "L/min")
            fig_leak.add_hline(y=24, line_dash="dash", line_color="#6b7280",
                               annotation_text="24 L/min threshold",
                               annotation_position="top right")
            st.plotly_chart(fig_leak, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Manual Log — All Entries")
    st.caption("Edit cells directly, then click **Save Changes**.")

    df_editor_src = init_manual_log().copy()
    df_editor_src["date"] = pd.to_datetime(df_editor_src["date"], errors="coerce").dt.date
    for col in ["nicotine_pouches", "vape_puffs", "caffeine_mg", "cpap_ahi", "cpap_hours", "cpap_leak_95"]:
        df_editor_src[col] = pd.to_numeric(df_editor_src[col], errors="coerce")

    edited = st.data_editor(
        df_editor_src.sort_values("date", ascending=False).reset_index(drop=True),
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "date": st.column_config.DateColumn("Date", required=True),
            "nicotine_pouches": st.column_config.NumberColumn("Pouches", min_value=0, max_value=99, step=1),
            "vape_puffs": st.column_config.NumberColumn("Vape Puffs", min_value=0, max_value=9999, step=1),
            "caffeine_mg": st.column_config.NumberColumn("Caffeine (mg)", min_value=0, max_value=3000, step=25),
            "cpap_ahi": st.column_config.NumberColumn("CPAP AHI", min_value=0.0, format="%.1f"),
            "cpap_hours": st.column_config.NumberColumn("CPAP Hours", min_value=0.0, max_value=24.0, format="%.2f"),
            "cpap_leak_95": st.column_config.NumberColumn("Leak 95th %tile", min_value=0.0, format="%.1f"),
            "notes": st.column_config.TextColumn("Notes"),
        },
    )

    if st.button("💾 Save Changes", type="primary"):
        edited["date"] = edited["date"].astype(str)
        save_manual_log(edited)
        st.success("Saved!")
        st.rerun()

    st.divider()
    csv_export = init_manual_log().to_csv(index=False)
    st.download_button(
        "⬇️ Download full log as CSV",
        data=csv_export,
        file_name=f"health_log_{date.today()}.csv",
        mime="text/csv",
    )
