# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal health tracking dashboard built with Python/Streamlit. Integrates Oura Ring API data, manual health logging (nicotine, vape, caffeine), CPAP data import from OSCAR, and habit streak tracking. All data stored locally in CSV/JSON files (no database).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app (opens at http://localhost:8501)
streamlit run app.py
```

No test suite or linter is configured.

## Architecture

**Single-file Streamlit app** (`app.py`) with everything in one module:

1. **Data helpers** — `init_manual_log()`, `save_manual_log()`, `upsert_row()` manage CSV persistence in `data/manual_log.csv`. Rows are keyed by date (one row per day, upsert semantics).
2. **Oura API** — `fetch_oura(token, endpoint, params)` with 5-minute `@st.cache_data` TTL. Hits `api.ouraring.com/v2/usercollection/{endpoint}`. Responses transformed via `oura_records_to_df()`.
3. **Streak system** — JSON-based (`data/streaks.json`). Each streak has name, start_date, emoji. Displayed with milestone badges on Overview tab.
4. **Charting** — `line_chart()` and `bar_chart()` wrappers around Plotly. `score_bands()` adds colored reference regions.
5. **7 tabs** — Overview, Sleep, Readiness, Activity, Lifestyle, CPAP, Data. Sidebar handles Oura token input, date range, and manual log entry form.

**Data flow:** Sidebar inputs → fetch Oura API + load local CSV/JSON → transform to DataFrames → Plotly charts in tabs → persist changes back to CSV/JSON.

## Data Files

- `data/manual_log.csv` — Primary data store. Columns: date, nicotine_pouches, vape_puffs, caffeine_mg, cpap_ahi, cpap_hours, cpap_leak_95, notes.
- `data/streaks.json` — Array of `{name, start_date, emoji}` objects.
- `.env` — Contains `OURA_TOKEN` for API access.

## Helper Scripts

- `scripts/log_nicotine.py` — CLI for logging from macOS/iOS Shortcuts: `python3 scripts/log_nicotine.py <pouches> <vape_puffs>`

## Key Dependencies

streamlit (>=1.32), pandas, plotly, requests, python-dotenv
