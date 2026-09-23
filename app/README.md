# WindAgent dashboard

The Streamlit presentation layer uses `windagent.api` for all project data and agent actions. It does not read prediction files directly or implement forecasting logic.

## Run from the UI checkout

Use `C:\Users\AsusROG\Desktop\hackaton-ui` for UI work. The core engineer's `hackaton` checkout is separate.

```powershell
cd C:\Users\AsusROG\Desktop\hackaton-ui
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The default view loads real outputs. Missing data produces guidance rather than substituting example forecasts. The sidebar selects RU/EN and fixed UTC+6, UTC+5 or UTC display clocks.

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

## Publish completed UI tasks

Commit only `app/`, `.streamlit/config.toml`, `tests/ui/`, `tests/fixtures/`, `docs/figures/ui/` and `docs/requests/codex-ui.md`, as applicable. Do not commit credentials or runtime rate-limit state.

Before each push, pull with rebase from `origin main`, reread `docs/CONTRACTS.md`, and check that the outgoing diff contains only UI-owned paths. Push to `origin main` without force. Core integration requests belong in `docs/requests/codex-ui.md`.
