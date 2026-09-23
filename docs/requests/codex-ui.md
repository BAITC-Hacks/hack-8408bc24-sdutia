# UI integration requests

## Issue metadata is not version-specific

- **What:** `windagent.api.load_run`, schema R `kpis`, `models_used`, `weather_source`; schema W version association.
- **Expected vs actual:** Selecting v1 or v2 changes the plotted forecast correctly, but the frozen interface provides issue-level KPIs and provenance only. Weather rows can include several initialization times without a forecast-version key. The UI labels KPIs as issue metadata and draws weather runs separately; it does not invent selected-version figures.
- **Reproduce:** `WINDAGENT_OUTPUTS_DIR=tests/fixtures/outputs streamlit run app/streamlit_app.py`; switch between v1 and v2. On PowerShell set the environment variable separately.
- **Severity:** minor; the current contract is supported. A future optional `kpis` and weather-run references inside each `versions[]` entry would permit exact version-specific cards and provenance.

## Verification environment

- **What:** local `.venv` interpreter.
- **Expected vs actual:** The project contract supports Python 3.11–3.13. The provided local environment runs Python 3.14.5. The UI uses Python 3.11-compatible syntax, and its offline tests pass in the available environment; supported-version CI remains the core/QA owner's responsibility.
- **Reproduce:** `.venv\Scripts\python.exe --version`.
- **Severity:** minor; no dependency files changed by UI.
