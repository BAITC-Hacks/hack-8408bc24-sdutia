"""Training/inference rows: as-of weather → features × turbines (+ SCADA targets)."""

from __future__ import annotations

import pandas as pd

from . import asof, config, features, scada, weather
from .timeutil import parse_clock_time


def load_weather_cache(site: config.Site, models: list[str] | None = None) -> pd.DataFrame:
    return pd.concat([weather.load_cache(site.site, m) for m in (models or weather.model_ids())], ignore_index=True)


def daily_issue_times(site: config.Site, start_data_clock: str, end_data_clock: str,
                      hour: int = config.ISSUE_HOUR_DATA_CLOCK) -> list[pd.Timestamp]:
    lo = parse_clock_time(f"{start_data_clock[:10]} {hour:02d}:00", site.data_clock_utc_offset_h)
    hi = parse_clock_time(f"{end_data_clock[:10]} {hour:02d}:00", site.data_clock_utc_offset_h)
    return list(pd.date_range(lo, hi, freq="1D"))


def build_rows(site: config.Site, issue_times, horizon_h: int, weather_long: pd.DataFrame,
               with_target: bool = True) -> pd.DataFrame:
    sel = asof.select_asof_many(weather_long, issue_times, horizon_h)
    feats = features.make_features(sel, horizon_h)
    rows = features.with_turbines(feats, site.turbine_ids)
    if with_target:
        act = scada.site_hourly_long(site)
        act = act[act["entity"] != "farm"][["target_time_utc", "entity", "actual", "available", "ws"]]
        act = act.rename(columns={"ws": "ws_obs"})
        rows = rows.merge(act, on=["target_time_utc", "entity"], how="left")
    return rows


def training_rows(rows: pd.DataFrame, cutoff_utc: pd.Timestamp) -> pd.DataFrame:
    """Only targets strictly before the cutoff, only hours when the turbine was available."""
    r = rows[(rows["target_time_utc"] < cutoff_utc) & (rows["issue_time_utc"] < cutoff_utc)]
    return r[r["actual"].notna() & (r["available"] == True)]  # noqa: E712
