"""Open-Meteo archived-forecast client (Previous Runs API) with a parquet cache.

For a valid hour v, the variable `<var>_previous_dayN` holds the value from the model run
initialised at floor_6h(v) - N days (verified value-by-value against exact single runs).
All requests use timezone=UTC and every response is checked for utc_offset_seconds == 0.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config
from .errors import DataUnavailableError, ExternalServiceError

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
BASE_VARS = {
    "wind_speed_10m": "ws10",
    "wind_speed_100m": "ws100",
    "wind_direction_100m": "wd100",
    "temperature_2m": "t2m",
    "surface_pressure": "sp",
}
OFFSETS = (1, 2, 3, 4)          # supports horizons up to 72 h for any issue hour
CHUNK_DAYS = 120
_TIMEOUT_S = 90

_session: requests.Session | None = None


def _http() -> requests.Session:
    """One shared session, sequential use, retries with backoff (parallel calls hit WinError 10013)."""
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(total=5, connect=5, read=3, backoff_factor=1.5,
                      status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers["User-Agent"] = "windagent/0.1 (hackathon research; open-meteo.com)"
        _session = s
    return _session


def model_ids() -> list[str]:
    return [m["id"] for m in config.WEATHER_MODELS]


def floor_6h(ts: pd.Series | pd.DatetimeIndex):
    return ts.dt.floor("6h") if isinstance(ts, pd.Series) else ts.floor("6h")


def fetch_previous_runs(model: str, lat: float, lon: float, start_date: str, end_date: str,
                        offsets=OFFSETS) -> pd.DataFrame:
    """Fetch one model for a UTC date range → long DataFrame:
    valid_time_utc, model, offset_days, init_time_utc, ws10, ws100, wd100, t2m, sp."""
    hourly = [f"{v}_previous_day{n}" for n in offsets for v in BASE_VARS]
    params = {
        "latitude": lat, "longitude": lon, "hourly": ",".join(hourly), "models": model,
        "start_date": start_date, "end_date": end_date, "timezone": "UTC", "wind_speed_unit": "ms",
    }
    try:
        r = _http().get(PREVIOUS_RUNS_URL, params=params, timeout=_TIMEOUT_S)
    except requests.RequestException as exc:
        raise ExternalServiceError(f"Open-Meteo request failed for {model}: {exc}",
                                   f"Запрос к Open-Meteo не удался ({model}): {exc}") from exc
    if r.status_code != 200:
        reason = r.json().get("reason", r.text[:200]) if r.headers.get("content-type", "").startswith("application/json") else r.text[:200]
        raise ExternalServiceError(f"Open-Meteo HTTP {r.status_code} for {model}: {reason}",
                                   f"Open-Meteo вернул HTTP {r.status_code} ({model}): {reason}")
    payload = r.json()
    if payload.get("utc_offset_seconds", 0) != 0:
        raise ExternalServiceError("Open-Meteo returned non-UTC timestamps; refusing to use them.",
                                   "Open-Meteo вернул время не в UTC; такие данные не используются.")
    h = payload.get("hourly") or {}
    times = pd.to_datetime(pd.Series(h.get("time", [])), utc=True)
    frames = []
    for n in offsets:
        part = pd.DataFrame({"valid_time_utc": times, "model": model, "offset_days": n})
        for api_name, short in BASE_VARS.items():
            vals = h.get(f"{api_name}_previous_day{n}")
            part[short] = pd.to_numeric(pd.Series(vals, dtype="object"), errors="coerce") if vals is not None else np.nan
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out["init_time_utc"] = floor_6h(out["valid_time_utc"]) - pd.to_timedelta(out["offset_days"], unit="D")
    out = out.dropna(subset=["ws10", "ws100"], how="all")
    return out[["valid_time_utc", "model", "offset_days", "init_time_utc", *BASE_VARS.values()]]


def cache_path(site_key: str, model: str) -> Path:
    return config.data_dir() / "cache" / site_key / f"nwp_{model}.parquet"


def load_cache(site_key: str, model: str) -> pd.DataFrame:
    p = cache_path(site_key, model)
    if not p.exists():
        raise DataUnavailableError(
            f"No weather cache for {model} ({p}). Run: python -m windagent fetch-history",
            f"Нет кэша погоды для {model} ({p}). Выполните: python -m windagent fetch-history",
        )
    df = pd.read_parquet(p)
    for c in ("valid_time_utc", "init_time_utc"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def save_cache(site_key: str, model: str, df: pd.DataFrame) -> Path:
    p = cache_path(site_key, model)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(["valid_time_utc", "offset_days"], keep="last").sort_values(["valid_time_utc", "offset_days"])
    df.to_parquet(p, index=False)
    return p


def fetch_history(site: config.Site, start: str, end: str, models: list[str] | None = None,
                  log=print) -> dict[str, int]:
    """Bulk-download archived forecasts for the site location into the cache (chunked, sequential)."""
    counts = {}
    for model in models or model_ids():
        meta = next(m for m in config.WEATHER_MODELS if m["id"] == model)
        lo = max(pd.Timestamp(start), pd.Timestamp(meta["archive_from"]))
        hi = pd.Timestamp(end)
        parts = []
        cur = lo
        while cur <= hi:
            chunk_end = min(cur + pd.Timedelta(days=CHUNK_DAYS - 1), hi)
            df = fetch_previous_runs(model, site.lat, site.lon, cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
            parts.append(df)
            log(f"  {model}: {cur:%Y-%m-%d}..{chunk_end:%Y-%m-%d} -> {len(df)} rows")
            cur = chunk_end + pd.Timedelta(days=1)
            time.sleep(0.5)
        full = pd.concat(parts, ignore_index=True)
        save_cache(site.site, model, full)
        counts[model] = len(full)
    return counts


def window_for_issue(site: config.Site, issue_time_utc: pd.Timestamp, horizon_h: int,
                     models: list[str] | None = None, offline: bool | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Weather rows needed for one issue: valid times [T, T + horizon] (+1 h for interval alignment).

    Tries a small live request per model (unless offline), falls back to the cache.
    Returns (long DataFrame, {model: 'live'|'cache'|'missing'}).
    """
    offline = config.offline() if offline is None else offline
    t0 = issue_time_utc
    t1 = issue_time_utc + pd.Timedelta(hours=horizon_h + 1)
    start, end = (t0 - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), (t1 + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames, source = [], {}
    for model in models or model_ids():
        df = None
        if not offline:
            try:
                df = fetch_previous_runs(model, site.lat, site.lon, start, end)
                source[model] = "live"
            except ExternalServiceError:
                df = None
        if df is None:
            try:
                df = load_cache(site.site, model)
                source[model] = "cache"
            except DataUnavailableError:
                source[model] = "missing"
                continue
        frames.append(df[(df["valid_time_utc"] >= t0) & (df["valid_time_utc"] <= t1)])
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        where = "the weather cache (offline mode)" if offline else "the API or the weather cache"
        raise ExternalServiceError(
            f"No archived weather forecasts for {t0:%Y-%m-%d %H:%M} UTC in {where}. The committed cache covers "
            f"2024-02 … 2026-03-03; for later dates run online.",
            f"Нет архивных прогнозов погоды на {t0:%Y-%m-%d %H:%M} UTC ({'кэш, офлайн-режим' if offline else 'API или кэш'}). "
            f"Кэш в репозитории покрывает 2024-02 … 2026-03-03; для более поздних дат запустите онлайн.")
    return out, source
