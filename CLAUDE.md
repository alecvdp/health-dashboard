# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal health tracking dashboard built with Python/Streamlit. Integrates Oura Ring API data, manual health logging (nicotine pouches, vape puffs, caffeine, weight), CPAP data import from OSCAR, and habit streak tracking. All data stored locally in CSV/JSON files (no database).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app (opens at http://localhost:8501)
streamlit run app.py
```

No test suite or linter is configured.

## Architecture

**Single-file Streamlit app** (`app.py`, ~925 lines) with everything in one module. No separate modules, no class-based structure — all functions and UI code live at the top level.

### Core Functions

1. **Data helpers** — CSV persistence in `data/manual_log.csv`:
   - `init_manual_log()` — Creates the CSV if missing, reads it, deduplicates by date (keeps last), ensures all expected columns exist.
   - `save_manual_log(df)` — Writes a DataFrame back to CSV.
   - `upsert_row(df, new_row)` — Insert or update a row keyed by date string. Returns the modified DataFrame (does not save).
   - `parse_duration_hours(value)` — Converts HH:MM:SS, HH:MM, or plain numeric strings to decimal hours. Used for OSCAR CSV imports.
   - `latest_val(df, col)` — Returns the last non-null numeric value from a DataFrame column.

2. **Oura API** — `fetch_oura(endpoint, token, start, end)` with 5-minute `@st.cache_data` TTL. Hits `api.ouraring.com/v2/usercollection/{endpoint}`. Returns a list of records on success or `{"error": "..."}` dict on failure. `oura_records_to_df(records, extra_fields)` flattens API records into a DataFrame, expanding nested contributor dicts into `{parent}_{key}` columns.

3. **Streak system** — JSON-based (`data/streaks.json`). Each streak has `name`, `start_date`, `emoji`. Managed via `load_streaks()`, `save_streaks()`, `streak_duration()`. Displayed with milestone cards on the Overview tab and managed on the Data tab.

4. **Charting** — Plotly wrappers with dark theme (`plotly_dark` template, transparent backgrounds):
   - `line_chart(df, x, y, title, color, yunit, yrange)` — Line + markers via `px.line`.
   - `bar_chart(df, x, y, title, color, yunit, hline, hline_label)` — Bar chart with optional reference line.
   - `score_bands(fig)` — Adds green (85-100), yellow (70-85), red (0-70) horizontal bands to score charts.

5. **7 tabs** — Overview, Sleep, Readiness, Activity, Lifestyle, CPAP, Data. Sidebar handles Oura token input, date range selection, and manual log entry form.

### Data Flow

```
Sidebar inputs (token, date range, manual entry form)
  -> Fetch Oura API (cached 5 min) + load local CSV/JSON
  -> Transform to DataFrames (df_sleep, df_readiness, df_activity, df_manual)
  -> Plotly charts rendered in tabs
  -> Manual edits/imports persist back to CSV/JSON via save_manual_log() / save_streaks()
```

### Key Patterns

- **Upsert semantics**: Manual log entries are keyed by date. One row per day; re-logging the same date updates the existing row.
- **`@st.fragment`**: The Lifestyle tab and Data tab editor use `@st.fragment` to isolate reruns — interactions within these fragments don't cause a full app rerun.
- **Session state**: `st.session_state["oura_token"]` is seeded from the `.env` file on first load and updated when the user types a new token in the sidebar.
- **Cache clearing**: After manual log saves or OSCAR imports, `st.cache_data.clear()` is called to force fresh data on the next rerun.
- **Weight forward-fill**: `weight_lbs` uses `.ffill()` so gaps in weight entries carry the last known value forward in charts.
- **Custom CSS**: Injected at the top of the app via `st.markdown(unsafe_allow_html=True)` for metric card and tab styling.

## Data Files

All data files live in the `data/` directory (gitignored):

- **`data/manual_log.csv`** — Primary data store. Columns (defined in `MANUAL_COLUMNS`):
  `date`, `nicotine_pouches`, `vape_puffs`, `caffeine_mg`, `weight_lbs`, `cpap_ahi`, `cpap_hours`, `cpap_leak_95`, `notes`

- **`data/streaks.json`** — Array of `{name, start_date, emoji}` objects.

- **`.env`** — Contains `OURA_TOKEN` for API access (see `.env.example`).

## Key Dependencies

- `streamlit` (>=1.33) — App framework, UI components, caching, fragments
- `pandas` (>=2.0) — DataFrame operations, CSV I/O
- `plotly` (>=5.18) — All charts (line, bar, scatter via `plotly.express` and `plotly.graph_objects`)
- `requests` (>=2.31) — Oura API HTTP calls
- `python-dotenv` (>=1.0) — Loads `.env` file for `OURA_TOKEN`

## Conventions

- **No classes**: All code is procedural/functional. Keep it that way unless there's a strong reason to refactor.
- **Single file**: Everything lives in `app.py`. Don't split into modules unless the file becomes unmanageable.
- **Section markers**: Code is organized with comment banners (`# ── Section ──` and `# ═══ TAB NAME ═══`) — maintain this style when adding new sections.
- **Dark theme**: All Plotly charts use `template="plotly_dark"` with `paper_bgcolor="rgba(0,0,0,0)"` and `plot_bgcolor="rgba(0,0,0,0)"`. Maintain this for visual consistency.
- **Null handling**: Manual log fields use `None` (not 0) for empty entries. Zero is a valid value (e.g., 0 pouches). The sidebar form only saves non-zero values; zero → `None`.
- **Date strings**: Dates are stored as `YYYY-MM-DD` strings in CSV. Converted to `pd.Timestamp` for filtering/charting.
- **Error returns**: `fetch_oura()` returns a dict with an `"error"` key on failure rather than raising exceptions. Callers check with `isinstance(records, dict) and "error" in records`.
