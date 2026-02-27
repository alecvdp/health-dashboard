# 🩺 Health Dashboard

A local Streamlit app for tracking sleep, readiness, steps (Oura), nicotine, caffeine, and CPAP data.

## Quick Start

```bash
# 1. Install dependencies (one time)
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

---

## Setup

### Oura API Token
1. Go to [cloud.ouraring.com/personal-access-tokens](https://cloud.ouraring.com/personal-access-tokens)
2. Create a **Personal Access Token**
3. Paste it in the sidebar when the app is open

Sleep, readiness, and activity data will auto-load for the selected date range.

### CPAP Data (OSCAR)
1. Open OSCAR and load a session
2. **File → Export → Daily Summary CSV**
3. In the dashboard: go to the **CPAP** tab → expand **Import from OSCAR CSV**
4. Upload the file and map the columns

### Nicotine & Caffeine
Use the **sidebar form** to log daily entries. You can also edit the raw CSV directly at `data/manual_log.csv`.

---

## Data Storage

All manually entered data lives in `data/manual_log.csv` — plain CSV, easy to back up or edit in Excel/Numbers.

| Column | Description |
|--------|-------------|
| `date` | YYYY-MM-DD |
| `nicotine_pouches` | Number of pouches |
| `caffeine_mg` | Total mg caffeine |
| `cpap_ahi` | Apnea-Hypopnea Index |
| `cpap_hours` | Hours of CPAP use |
| `cpap_leak_95` | 95th percentile leak rate (L/min) |
| `notes` | Free text notes |

---

## Tips

- **Date range**: Use the sidebar date pickers to zoom in/out on any time period
- **Refresh Oura data**: Oura data is cached for 5 minutes — restart the app or wait to force a refresh
- **Edit past entries**: Go to the **Data** tab to edit any row in the table directly
- **Export**: The Data tab has a download button for the full CSV log
