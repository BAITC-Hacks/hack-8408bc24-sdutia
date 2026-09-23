"""SCADA loading, hourly aggregation and availability flags.

Input format (as provided by the organizers), columns by position:
ID, statistical time (data clock), mean wind speed m/s, normalized active power, mean ambient temperature °C.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Site, get_site
from .errors import DataUnavailableError, InputError
from .timeutil import UTC, clock_to_utc

MIN_RECORDS_PER_HOUR = 4          # of 6 ten-minute records
UNAVAILABLE_MAX_POWER = 0.05      # hourly mean power below this ...
UNAVAILABLE_MIN_WIND = 6.0        # ... while hourly mean wind is above this → outage / curtailment


def read_scada_csv(path: Path, data_clock_utc_offset_h: int) -> pd.DataFrame:
    """Read one turbine file → DataFrame indexed by UTC time with columns ws, p, temp (10-minute)."""
    path = Path(path)
    if not path.exists():
        raise DataUnavailableError(f"SCADA file not found: {path}", f"Файл SCADA не найден: {path}")
    try:
        raw = pd.read_csv(path, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - surface any parse problem as a user error
        raise InputError(f"Cannot read SCADA CSV {path.name}: {exc}", f"Не удалось прочитать CSV {path.name}: {exc}")
    if raw.shape[1] < 5:
        raise InputError(
            f"{path.name}: expected 5 columns (ID, time, wind m/s, normalized power, temperature °C), got {raw.shape[1]}",
            f"{path.name}: ожидается 5 столбцов (ID, время, ветер м/с, нормализованная мощность, температура °C), найдено {raw.shape[1]}",
        )
    df = raw.iloc[:, 1:5].copy()
    df.columns = ["time", "ws", "p", "temp"]
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ("ws", "p", "temp"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df["time"].isna().sum()
    if bad > 0.01 * len(df):
        raise InputError(f"{path.name}: {bad} rows have an unparseable time", f"{path.name}: у {bad} строк не распознано время")
    df = df.dropna(subset=["time"]).drop_duplicates("time").sort_values("time")
    df["time_utc"] = clock_to_utc(df["time"], data_clock_utc_offset_h)
    return df.set_index("time_utc")[["ws", "p", "temp"]]


def to_hourly(df10: pd.DataFrame) -> pd.DataFrame:
    """10-minute → hourly means (hour labelled by its start). Hours with < 4 records are dropped."""
    g = df10.resample("1h")
    hourly = g[["ws", "p", "temp"]].mean()
    hourly["n_records"] = g["p"].count()
    hourly = hourly[hourly["n_records"] >= MIN_RECORDS_PER_HOUR].copy()
    hourly["available"] = ~((hourly["p"] < UNAVAILABLE_MAX_POWER) & (hourly["ws"] > UNAVAILABLE_MIN_WIND))
    return hourly


@lru_cache(maxsize=16)
def _turbine_hourly_cached(site_key: str, turbine_id: str) -> pd.DataFrame:
    site = get_site(site_key)
    turbine = next((t for t in site.turbines if t.id == turbine_id), None)
    if turbine is None:
        raise InputError(f"Unknown turbine '{turbine_id}' for site '{site_key}'",
                         f"Неизвестная турбина '{turbine_id}' на площадке '{site_key}'")
    if turbine.scada_path() is None:
        raise DataUnavailableError(f"No SCADA history configured for turbine '{turbine_id}'",
                                   f"Для турбины '{turbine_id}' не задана история SCADA")
    return to_hourly(read_scada_csv(turbine.scada_path(), site.data_clock_utc_offset_h))


def turbine_hourly(site: Site | str, turbine_id: str) -> pd.DataFrame:
    key = site if isinstance(site, str) else site.site
    return _turbine_hourly_cached(key, turbine_id).copy()


def site_hourly_long(site: Site | str) -> pd.DataFrame:
    """All turbines + farm, long format: target_time_utc, entity, actual, ws, temp, available.

    farm = mean of the turbines' normalized power, only for hours where every turbine has data.
    """
    s = get_site(site) if isinstance(site, str) else site
    frames = []
    for t in s.turbines:
        if t.scada_path() is None:
            continue
        h = turbine_hourly(s, t.id)
        frames.append(pd.DataFrame({
            "target_time_utc": h.index, "entity": t.id, "actual": h["p"].to_numpy(),
            "ws": h["ws"].to_numpy(), "temp": h["temp"].to_numpy(), "available": h["available"].to_numpy(),
        }))
    if not frames:
        raise DataUnavailableError(f"No SCADA data for site '{s.site}'", f"Нет данных SCADA для площадки '{s.site}'")
    long = pd.concat(frames, ignore_index=True)
    wide_p = long.pivot(index="target_time_utc", columns="entity", values="actual")
    wide_a = long.pivot(index="target_time_utc", columns="entity", values="available")
    wide_ws = long.pivot(index="target_time_utc", columns="entity", values="ws")
    complete = wide_p.notna().all(axis=1)
    farm = pd.DataFrame({
        "target_time_utc": wide_p.index[complete],
        "entity": "farm",
        "actual": wide_p[complete].mean(axis=1).to_numpy(),
        "ws": wide_ws[complete].mean(axis=1).to_numpy(),
        "temp": np.nan,
        "available": wide_a[complete].astype(bool).all(axis=1).to_numpy(),
    })
    out = pd.concat([long, farm], ignore_index=True)
    out["target_time_utc"] = pd.to_datetime(out["target_time_utc"], utc=True)
    return out.sort_values(["target_time_utc", "entity"]).reset_index(drop=True)


def last_observation_time(site: Site | str) -> pd.Timestamp:
    long = site_hourly_long(site)
    return long["target_time_utc"].max()


def actuals(site: Site | str, start_utc=None, end_utc=None) -> pd.DataFrame:
    long = site_hourly_long(site)
    if start_utc is not None:
        long = long[long["target_time_utc"] >= pd.Timestamp(start_utc).tz_convert(UTC)]
    if end_utc is not None:
        long = long[long["target_time_utc"] <= pd.Timestamp(end_utc).tz_convert(UTC)]
    return long[["target_time_utc", "entity", "actual", "available"]].reset_index(drop=True)
