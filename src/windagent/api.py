"""Public Python API. The ONLY interface the UI may use (see docs/CONTRACTS.md, section 5)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config, scada
from .errors import DataUnavailableError, ExternalServiceError, InputError, WindAgentError  # noqa: F401
from .timeutil import clock_str

__all__ = [
    "WindAgentError", "InputError", "DataUnavailableError", "ExternalServiceError",
    "list_sites", "site_info", "list_runs", "load_run", "load_submission", "load_all_issues",
    "load_metrics", "load_validation_predictions", "load_actuals", "llm_status", "system_info",
    "run_agent", "forecast_now",
]

_UTC_SUFFIX = "_utc"


def _site_dir(site: str) -> Path:
    config.get_site(site)  # validates the site name
    return config.outputs_dir() / site


def _parse_utc_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c.endswith(_UTC_SUFFIX):
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


def _read_csv(path: Path, what_en: str, what_ru: str) -> pd.DataFrame:
    if not path.exists():
        raise DataUnavailableError(
            f"{what_en} not found ({path}). Generate outputs first: python -m windagent all --offline",
            f"{what_ru} не найден ({path}). Сначала сгенерируйте результаты: python -m windagent all --offline",
        )
    return _parse_utc_columns(pd.read_csv(path, encoding="utf-8"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def list_sites() -> list[str]:
    return sorted(config.load_sites())


def site_info(site: str) -> dict:
    s = config.get_site(site)
    info = {
        "site": s.site,
        "name": s.name,
        "turbines": [{"id": t.id, "lat": t.lat, "lon": t.lon, "rated_mw": None} for t in s.turbines],
        "rated_mw": s.rated_mw,
        "data_clock_utc_offset_h": s.data_clock_utc_offset_h,
        "official_utc_offset_h": s.official_utc_offset_h,
        "test_period": s.test_period,
        "issue_range_data_clock": None,
    }
    try:
        from .asof import issue_range_utc  # available once the weather layer is in place

        lo, hi = issue_range_utc(s)
        info["issue_range_data_clock"] = {"min": clock_str(lo, s.data_clock_utc_offset_h),
                                          "max": clock_str(hi, s.data_clock_utc_offset_h)}
    except ImportError:
        pass
    return info


def list_runs(site: str, kind: str = "runs") -> pd.DataFrame:
    if kind not in {"runs", "live", "adhoc"}:
        raise InputError(f"kind must be 'runs', 'adhoc' or 'live', got '{kind}'",
                         f"kind должен быть 'runs', 'adhoc' или 'live', получено '{kind}'")
    root = _site_dir(site) / kind
    cols = ["issue_id", "issue_time_utc", "versions", "policy", "provider", "created_at_utc"]
    rows = []
    if root.exists():
        for meta_path in sorted(root.glob("*/run.json")):
            try:
                m = _read_json(meta_path)
            except (OSError, json.JSONDecodeError):
                continue
            rows.append({
                "issue_id": m.get("issue_id", meta_path.parent.name),
                "issue_time_utc": m.get("issue_time_utc"),
                "versions": len(m.get("versions", [])) or 1,
                "policy": m.get("policy"),
                "provider": m.get("provider"),
                "created_at_utc": m.get("created_at_utc"),
            })
    return _parse_utc_columns(pd.DataFrame(rows, columns=cols))


def load_run(site: str, issue_id: str, kind: str = "runs") -> dict:
    run_dir = _site_dir(site) / kind / issue_id
    if not (run_dir / "run.json").exists():
        raise DataUnavailableError(f"Run '{issue_id}' not found for site '{site}'",
                                   f"Прогноз '{issue_id}' для площадки '{site}' не найден")
    trace = []
    trace_path = run_dir / "trace.jsonl"
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trace.append(json.loads(line))
    analysis_path = run_dir / "analysis.md"
    weather_path = run_dir / "weather.csv"
    return {
        "forecast": _read_csv(run_dir / "forecast.csv", "Forecast file", "Файл прогноза"),
        "weather": _parse_utc_columns(pd.read_csv(weather_path, encoding="utf-8")) if weather_path.exists() else pd.DataFrame(),
        "trace": trace,
        "analysis": analysis_path.read_text(encoding="utf-8") if analysis_path.exists() else "",
        "meta": _read_json(run_dir / "run.json"),
    }


def load_submission(site: str) -> pd.DataFrame:
    return _read_csv(_site_dir(site) / "test_period" / "submission_day_ahead.csv", "Submission file", "Файл сабмита")


def load_all_issues(site: str) -> pd.DataFrame:
    return _read_csv(_site_dir(site) / "test_period" / "all_issues.csv", "All-issues file", "Файл всех прогнозов")


def load_metrics(site: str) -> dict:
    path = _site_dir(site) / "validation" / "metrics.json"
    return _read_json(path) if path.exists() else {}


def load_validation_predictions(site: str) -> pd.DataFrame:
    return _read_csv(_site_dir(site) / "validation" / "predictions.csv", "Validation predictions", "Прогнозы валидации")


def load_actuals(site: str, start_utc=None, end_utc=None) -> pd.DataFrame:
    return scada.actuals(site, start_utc, end_utc)


def llm_status() -> dict:
    try:
        from .agent.llm import resolve_provider

        return resolve_provider()
    except ImportError:
        wanted = os.environ.get("LLM_PROVIDER", "auto").lower()
        for name, key_var, model_var in (("openai", "OPENAI_API_KEY", "OPENAI_MODEL"),
                                         ("nvidia", "NVIDIA_API_KEY", "NVIDIA_MODEL")):
            if wanted in {"auto", name} and os.environ.get(key_var):
                return {"provider": name, "model": os.environ.get(model_var), "configured": True}
        return {"provider": "none", "model": None, "configured": False}


def system_info() -> dict:
    tools = []
    try:
        from .agent.tools import TOOL_SPECS

        tools = [{"name": t["name"], "description": t["description"]} for t in TOOL_SPECS]
    except ImportError:
        pass
    return {
        "version": config.__version__,
        "issue_hour_data_clock": config.ISSUE_HOUR_DATA_CLOCK,
        "horizon_h": config.HORIZON_H,
        "weather_models": [dict(m) for m in config.WEATHER_MODELS],
        "policies": ["auto", "llm", "rules"],
        "tools": tools,
    }


def run_agent(site: str, issue_time_data_clock: str, policy: str = "auto",
              on_event: Callable[[dict], None] | None = None, horizon_h: int = config.HORIZON_H,
              offline: bool | None = None, kind: str = "adhoc") -> dict:
    try:
        from .agent.runner import run_issue
    except ImportError as exc:
        raise DataUnavailableError("The agent is not available in this build yet.",
                                   "Агент пока недоступен в этой сборке.") from exc
    if kind not in {"adhoc", "runs"}:
        raise InputError(f"kind must be 'adhoc' or 'runs', got '{kind}'", f"kind должен быть 'adhoc' или 'runs', получено '{kind}'")
    return run_issue(site=site, issue_time_data_clock=issue_time_data_clock, policy=policy,
                     on_event=on_event, horizon_h=horizon_h, offline=offline, kind=kind)


def forecast_now(site: str, policy: str = "auto", on_event: Callable[[dict], None] | None = None) -> dict:
    try:
        from .agent.runner import run_now
    except ImportError as exc:
        raise DataUnavailableError("Live forecasting is not available in this build yet.",
                                   "Прогноз в реальном времени пока недоступен в этой сборке.") from exc
    return run_now(site=site, policy=policy, on_event=on_event)
