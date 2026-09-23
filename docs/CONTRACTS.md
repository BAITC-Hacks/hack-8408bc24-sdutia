# Integration contracts

This document is authoritative. It fixes the interfaces between the core engine (`src/windagent`) and the presentation layer (`app/`).
Only the core owner changes it. Change requests go to `docs/requests/codex.md`.

## 1. Environment

- Python 3.11–3.13. The package `windagent` lives in `src/windagent/`.
- Install: `pip install -r requirements.txt` (it includes `-e .`).
- The UI must also work when the package is not installed. `app/streamlit_app.py` puts `<repo>/src` on `sys.path` before importing `windagent`:

  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
  ```

**Environment variables** (all optional)

| Variable | Default | Meaning |
|---|---|---|
| `WINDAGENT_OUTPUTS_DIR` | `<repo>/outputs` | Root of generated outputs. UI tests point it to `tests/fixtures/outputs`. |
| `WINDAGENT_DATA_DIR` | `<repo>/data` | Raw SCADA data and the weather cache |
| `WINDAGENT_OFFLINE` | `0` | `1` means never call external APIs; use the cache only |
| `LLM_PROVIDER` | `auto` | `auto` \| `openai` \| `nvidia` \| `none`. `auto` tries OpenAI, then NVIDIA, then falls back to rules |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | — | OpenAI access |
| `NVIDIA_API_KEY`, `NVIDIA_MODEL`, `NVIDIA_BASE_URL` | — | NVIDIA NIM (OpenAI-compatible) |

## 2. Time and clocks

- **Internally, every datetime is timezone-aware UTC.** Never call a naive `datetime.now()`.
- **Data clock = fixed UTC+6.** This is the clock of the provided SCADA CSVs. The logger did not follow Kazakhstan's 2024-03-01 change.
- **Official Kazakhstan time = UTC+5.**
- Never use named timezones such as `Asia/Almaty`.

**How times are written in files**

| Column pattern | Format | Example |
|---|---|---|
| `*_utc` | ISO-8601 with `Z` | `2026-02-01T18:00:00Z` |
| `*_data_clock`, `*_kz_official` | `YYYY-MM-DD HH:MM` (the clock is named in the column) | `2026-02-02 00:00` |

- Hours are labelled by their **start**. For example, `00:00` means the mean over 00:00–00:59.
- File and folder names never contain `:`, because Windows forbids it.

## 3. Identifiers

| Name | Values |
|---|---|
| `site` | Slug from `config/sites.yaml`, e.g. `shelek` |
| `entity` | `farm` (mean of the site's turbines) or a turbine id: `t1`, `t2`, … |
| `issue_id` | Issue time on the data clock as `YYYY-MM-DD_HHMM`, e.g. `2026-02-10_0000` |
| `version` | `1` = issued at issue time; `2`, `3`, … = recomputed after a newer weather run was published |
| `product` | `intraday` (lead 0–23 h), `day_ahead` (lead 24–47 h), `extended` (lead ≥ 48 h) |
| Power values | Normalized power in [0, 1], the same unit as the dataset. Columns starting `mw_` hold MW and are empty unless `rated_mw` is configured |

## 4. Output layout (under `WINDAGENT_OUTPUTS_DIR`)

```
<site>/
  runs/<issue_id>/forecast.csv      schema F  (all versions of this issue)
  runs/<issue_id>/weather.csv       schema W  (weather inputs actually used)
  runs/<issue_id>/trace.jsonl       schema T  (one JSON event per line)
  runs/<issue_id>/analysis.md       markdown analysis written by the agent
  runs/<issue_id>/run.json          schema R
  live/<issue_id>/...               same files, produced by `forecast --now`
  test_period/submission_day_ahead.csv   schema S (jury-facing, wide)
  test_period/all_issues.csv             schema F (every issue, every version)
  validation/metrics.json                schema M
  validation/predictions.csv             schema F + column `actual`
  evaluation/evaluation.json             schema E (only after `evaluate`)
```

### Schema F: forecast, long format

| Column | Type | Notes |
|---|---|---|
| `site` | str | |
| `issue_id` | str | |
| `version` | int | |
| `issue_time_utc` | str | ISO Z |
| `target_time_utc` | str | ISO Z |
| `target_time_data_clock` | str | UTC+6 |
| `target_time_kz_official` | str | UTC+5 |
| `lead_h` | int | 0 … horizon−1 |
| `product` | str | intraday / day_ahead / extended |
| `entity` | str | farm / t1 / t2 … |
| `mean` | float | Point forecast |
| `p10`, `p50`, `p90` | float | Quantiles, with `p10 ≤ p50 ≤ p90` |
| `mw_mean` | float or empty | Only if `rated_mw` is set |

### Schema W: weather inputs

| Column | Type | Notes |
|---|---|---|
| `issue_id` | str | |
| `target_time_utc` | str | |
| `model` | str | e.g. `ecmwf_aifs025_single` |
| `init_time_utc` | str | Start time of the weather-model run used |
| `offset_days` | int | Previous-runs offset N |
| `ws10`, `ws100` | float | Wind speed at 10 m and 100 m, m/s |
| `wd100` | float | Wind direction at 100 m, degrees |
| `t2m` | float | Temperature at 2 m, °C |
| `sp` | float | Surface pressure, hPa |
| `source` | str | `live` or `cache` |

### Schema T: trace event (one JSON object per line)

```json
{"ts_utc": "2026-02-09T18:00:01Z", "run_id": "shelek/2026-02-10_0000", "seq": 3,
 "type": "tool_call", "policy": "llm", "provider": "openai", "model": "gpt-…",
 "tool": "fetch_weather", "args": {"models": ["ecmwf_aifs025_single"]},
 "summary": "Fetched 4 models, 48 h, 0 missing values", "data": {}, "duration_ms": 812, "ok": true}
```

- `type` is one of: `run_start`, `plan`, `tool_call`, `tool_result`, `decision`, `llm_message`, `warning`, `error`, `publish`, `run_end`.
- `policy` is `llm` or `rules`. `provider` is `openai`, `nvidia` or `none`.
- `tool`, `args`, `data`, `duration_ms` and `model` may be `null`.

### Schema R: `run.json`

```json
{"site": "shelek", "issue_id": "2026-02-10_0000", "issue_time_utc": "2026-02-09T18:00:00Z",
 "created_at_utc": "…", "policy": "rules", "provider": "none", "model": null, "horizon_h": 48,
 "versions": [{"version": 1, "as_of_utc": "2026-02-09T18:00:00Z", "reason": "initial"},
              {"version": 2, "as_of_utc": "2026-02-10T00:00:00Z", "reason": "new runs: …"}],
 "models_used": ["ecmwf_ifs025", "ecmwf_aifs025_single", "icon_seamless", "gfs_seamless"],
 "excluded_models": [{"model": "…", "reason": "…"}],
 "runs_used": {"ecmwf_aifs025_single": ["2026-02-09T06:00:00Z"]},
 "weather_source": "live|cache|mixed", "warnings": [],
 "kpis": {"energy_norm_h": 0.0, "capacity_factor": 0.0, "max_ramp_3h": 0.0,
          "mean_band_width": 0.0, "revision_mae_v2_vs_v1": null},
 "actuals_available": false, "errors": null}
```

`errors` is `{"mae": …, "rmse": …}` when actuals exist for the forecast window.

### Schema S: `submission_day_ahead.csv` (jury-facing)

- One row per target hour of the test period. It's the day-ahead product (`lead_h` 24–47), version 1.
- Columns: `target_time_data_clock`, `target_time_utc`, `target_time_kz_official`, `issue_id`, `issue_time_utc`, `lead_h`, `farm_mean`, `farm_p10`, `farm_p50`, `farm_p90`, then `<turbine>_mean` for each turbine (`t1_mean`, `t2_mean`), then `models_used` (separated by `;`).

### Schema M: `metrics.json`

```json
{"site": "shelek", "generated_at_utc": "…",
 "validation": {"start_utc": "…", "end_utc": "…", "n_issues": 0, "issue_hour_data_clock": 0},
 "by_product": {
   "day_ahead": {"model": {"mae": 0, "rmse": 0, "bias": 0, "r2": 0, "n": 0},
                 "persistence": {…}, "climatology": {…}, "nwp_powercurve": {…}},
   "intraday":  {…}},
 "by_lead": [{"lead_h": 0, "model": 0, "persistence": 0, "climatology": 0, "nwp_powercurve": 0}],
 "intervals": {"day_ahead": {"coverage_p10_p90": 0, "mean_width": 0}},
 "skill_pct": {"day_ahead_vs_climatology": 0, "day_ahead_vs_persistence": 0},
 "per_month": [{"month": "2025-10", "model": 0, "climatology": 0, "persistence": 0}],
 "leave_one_turbine_out": {"train": "t1", "test": "t2", "mae": 0, "mae_reference": 0}}
```

- The `by_lead` and `per_month` values are MAE.
- **Consumers must tolerate missing keys.** Show "n/a" rather than crashing.

### Schema E: `evaluation.json`

```json
{"forecast_file": "…", "actuals_files": {"t1": "…"}, "period": {"start_utc": "…", "end_utc": "…"},
 "best_lag_h": 0, "lag_corr": {"-3": 0, "…": 0, "3": 0}, "warning": null,
 "metrics": {"farm": {"mae": 0, "rmse": 0, "bias": 0, "r2": 0, "n": 0}, "t1": {…}, "t2": {…}},
 "metrics_realigned": null, "by_day": [{"date_data_clock": "2026-02-01", "farm_mae": 0}]}
```

## 5. Python API: `windagent.api` (the only interface the UI may use)

```python
class WindAgentError(Exception):            # base class; has .user_message_ru and .user_message_en
class InputError(WindAgentError): ...       # invalid user input (bad date, unknown site, …)
class DataUnavailableError(WindAgentError): ...  # outputs or data not generated / missing

def list_sites() -> list[str]
def site_info(site: str) -> dict
    # {"site", "name", "turbines": [{"id", "lat", "lon", "rated_mw"}],
    #  "data_clock_utc_offset_h": 6, "official_utc_offset_h": 5,
    #  "test_period": {"start": "2026-02-01", "end": "2026-02-28"},
    #  "issue_range_data_clock": {"min": "YYYY-MM-DD HH:MM", "max": "YYYY-MM-DD HH:MM"}}
def list_runs(site: str, kind: str = "runs") -> pd.DataFrame
    # kind in {"runs", "live"}; columns: issue_id, issue_time_utc, versions, policy, provider, created_at_utc
def load_run(site: str, issue_id: str, kind: str = "runs") -> dict
    # {"forecast": DataFrame(F), "weather": DataFrame(W), "trace": list[dict(T)], "analysis": str, "meta": dict(R)}
def load_submission(site: str) -> pd.DataFrame         # schema S
def load_all_issues(site: str) -> pd.DataFrame         # schema F
def load_metrics(site: str) -> dict                    # schema M, or {} if not generated
def load_validation_predictions(site: str) -> pd.DataFrame
def load_actuals(site: str, start_utc=None, end_utc=None) -> pd.DataFrame
    # columns: target_time_utc (datetime64[ns, UTC]), entity, actual (float), available (bool)
def llm_status() -> dict    # {"provider": "openai"|"nvidia"|"none", "model": str|None, "configured": bool}
def system_info() -> dict
    # {"version": str, "issue_hour_data_clock": 0, "horizon_h": 48,
    #  "weather_models": [{"id", "label", "latency_h", "archive_from"}],
    #  "policies": ["auto", "llm", "rules"], "tools": [{"name", "description"}]}
    # The UI's "How it works" tab must take these facts from here, not hard-code them.
def run_agent(site: str, issue_time_data_clock: str, policy: str = "auto",
              on_event=None, horizon_h: int = 48, offline: bool | None = None) -> dict
    # Runs the full agent loop and persists to runs/<issue_id>/. Returns the same dict as load_run.
    # on_event(event: dict) is called synchronously for each trace event (schema T), for live display.
def forecast_now(site: str, policy: str = "auto", on_event=None) -> dict   # persists to live/<issue_id>/
```

- In returned DataFrames, `*_utc` columns are parsed as timezone-aware UTC datetimes. `*_data_clock` and `*_kz_official` stay strings.
- Loaders raise `DataUnavailableError` if files are missing. The UI shows the error's `user_message_*`, never a traceback.

## 6. CLI: `python -m windagent <command>`

| Command | Purpose |
|---|---|
| `all [--site S] [--offline] [--policy auto\|llm\|rules]` | End-to-end reproduction |
| `fetch-history [--site S]` | Bulk archived weather → cache |
| `train [--site S]` | Train the models |
| `validate [--site S]` | Walk-forward validation → `metrics.json` |
| `backtest [--site S] [--start 2026-01-31] [--end 2026-02-28] [--policy …] [--offline] [--outputs-dir DIR]` | Rolling agent run over the test period |
| `forecast [--site S] (--issue "YYYY-MM-DD HH:MM" \| --now) [--policy …] [--offline]` | A single issue, or live now |
| `evaluate [--site S] --actuals t1=PATH --actuals t2=PATH [--forecast PATH]` | Score against actuals in the organizers' CSV format |
| `detect-clock [--site S]` | SCADA clock-offset forensics |
| `report` | Insert metrics into the README |

- `--issue` is given on the **data clock**.
- **Exit codes:** `0` OK, `2` invalid input, `3` data unavailable, `4` external service failed and no cache, `1` unexpected.
- Errors print one line, `ERROR: <message>`, to stderr. A traceback appears only with `--debug`.

## 7. Streamlit app

- Entry point: `app/streamlit_app.py`. Docker, CI and Streamlit Cloud all reference this path.
- **It must start and stay usable** with no outputs generated, no LLM keys, and no internet. In each of those cases it shows guidance instead of crashing.
- Secrets: on Streamlit Cloud, keys come from `st.secrets`. The app copies them into environment variables before calling `windagent.api`. Keys are never displayed or logged.
