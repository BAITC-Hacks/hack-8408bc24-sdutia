# WindAgent dashboard

The Streamlit presentation layer uses `windagent.api` for all project data and agent actions. It does not read prediction files directly or implement forecasting logic.

## Run from the UI checkout

Use `C:\Users\AsusROG\Desktop\hackaton-ui` for UI work. The core engineer's `hackaton` checkout is separate.

```powershell
cd C:\Users\AsusROG\Desktop\hackaton-ui
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The default view loads real outputs in Russian, with the data clock (UTC+6) and the February 1 issue when available. Missing data produces guidance rather than substituting example forecasts. The sidebar selects RU/EN and fixed UTC+6, UTC+5 or UTC display clocks; issue labels always retain their data-clock date.

The Forecast tab explains the project and validation evidence, distinguishes the initial forecast from its weather-driven update, and offers test-period, personal (`adhoc`) and live runs. Cards use per-version metadata; the hourly table shows percentages while CSV downloads preserve raw API columns. The Agent tab includes date presets, merged tool-call/result steps, per-version analysis and a friendly rules fallback when no LLM key is configured. Validation separates held-out calibrated coverage from uncalibrated full-period coverage. The How it works tab includes exact README reproduction commands in a **Run locally** section.

To review synthetic fixtures explicitly:

```powershell
$env:WINDAGENT_OUTPUTS_DIR = 'tests/fixtures/outputs'
$env:WINDAGENT_OFFLINE = '1'
$env:LLM_PROVIDER = 'none'
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Fixture forecasts and validation values are labelled as synthetic. To return to real outputs, remove `WINDAGENT_OUTPUTS_DIR` and restart Streamlit.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ui -q
```

Tests use temporary copies of fixtures, block network access and remove LLM credentials. They cover all five tabs, empty outputs, callback timelines, RU/EN, manual clock selection after a live run, chart gaps, exact revision matching, complete-day energy aggregation, API failures, and rate limits.

Verified on Python 3.12.14: 91 UI tests pass. Tests include the browser's stale translated-label wire protocol, selected-version KPIs, `adhoc` routing, readable percentage tables, calibrated coverage scopes, and elapsed-hour revision gaps. Browser verification covers language and clock switching, both forecast versions and real agent execution with cached weather.

Browser captures: [introduction](../docs/figures/ui/intro.png), [forecast](../docs/figures/ui/forecast.png), [hourly values](../docs/figures/ui/hourly-values.png), [agent run](../docs/figures/ui/agent.png), [validation](../docs/figures/ui/validation.png), and [test period](../docs/figures/ui/test-period.png). These show real project outputs.

## Publish completed UI tasks

Commit only `app/`, `.streamlit/config.toml`, `tests/ui/`, `tests/fixtures/`, `docs/figures/ui/` and `docs/requests/codex-ui.md`, as applicable. Do not commit credentials or runtime rate-limit state.

Before each push, pull with rebase from `origin main`, reread `docs/CONTRACTS.md`, and check that the outgoing diff contains only UI-owned paths. Push to `origin main` without force. Core integration requests belong in `docs/requests/codex-ui.md`.
