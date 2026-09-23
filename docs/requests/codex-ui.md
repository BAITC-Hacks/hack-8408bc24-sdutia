# UI integration requests

## Issue metadata is not version-specific

- **What:** `windagent.api.load_run`, schema R `kpis`, `models_used`, `weather_source`; schema W version association.
- **Expected vs actual:** Selecting v1 or v2 changes the plotted forecast correctly, but the frozen interface provides issue-level KPIs and provenance only. Weather rows can include several initialization times without a forecast-version key. The UI labels KPIs as issue metadata and draws weather runs separately; it does not invent selected-version figures.
- **Reproduce:** `WINDAGENT_OUTPUTS_DIR=tests/fixtures/outputs streamlit run app/streamlit_app.py`; switch between v1 and v2. On PowerShell set the environment variable separately.
- **Severity:** minor; the current contract is supported. A future optional `kpis` and weather-run references inside each `versions[]` entry would permit exact version-specific cards and provenance.

## Verification environment — resolved

- **What:** local `.venv` interpreter.
- **Expected vs actual:** The project contract supports Python 3.11–3.13. The original environment ran Python 3.14.5. The isolated `hackaton-ui` checkout now has its own Python 3.12.14 environment, installed directly from the unchanged requirements file. All 36 offline UI tests pass there. UI modules also parse with Python 3.11 syntax rules.
- **Reproduce:** `.venv\Scripts\python.exe --version`.
- **Severity:** resolved; no dependency files changed by UI.
