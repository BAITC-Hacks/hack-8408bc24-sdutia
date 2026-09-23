# UI integration requests

## Version-specific metadata — resolved

- **What:** `windagent.api.load_run`, schema R `kpis`, `models_used`, `weather_source`.
- **Resolution:** The core now supplies optional `versions[].kpis`, `models_used` and `runs_used`. Forecast and agent cards read the selected version's KPIs; absent legacy values show `n/a` rather than borrowing another version's summary. Weather charts and provenance use schema W's version association and the selected display clock.
- **Reproduce:** `WINDAGENT_OUTPUTS_DIR=tests/fixtures/outputs streamlit run app/streamlit_app.py`; switch between v1 and v2. On PowerShell set the environment variable separately.
- **Severity:** resolved.

## Verification environment — resolved

- **What:** local `.venv` interpreter.
- **Expected vs actual:** The project contract supports Python 3.11–3.13. The original environment ran Python 3.14.5. The isolated `hackaton-ui` checkout now has its own Python 3.12.14 environment, installed directly from the unchanged requirements file. Offline UI tests cover all five tabs, real API fixture loading, empty outputs, translated selections, clocks, version labels and KPIs, agent timelines, and analytical safeguards.
- **Reproduce:** `.venv\Scripts\python.exe --version`.
- **Severity:** resolved; no dependency files changed by UI.
