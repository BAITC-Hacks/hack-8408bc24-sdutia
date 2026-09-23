"""Evidence for two assumptions of the as-of engine (needs internet; results are saved to outputs/provenance/).

1. Publication delay per weather model: Open-Meteo meta.json gives the time the latest run became
   available; delay = last_run_availability_time - last_run_initialisation_time. Our configured
   latencies must be >= the measured delay (safety margin).
2. Previous Runs semantics: the value of `<var>_previous_dayN` at valid time v comes from the run
   initialised at floor_6h(v) - N days. Checked against exact runs from the Single Runs API for
   ECMWF IFS HRES (the one model archived there for 2026).
"""

from __future__ import annotations

import json

import pandas as pd

from . import config, weather
from .timeutil import iso_z, now_utc

META_IDS = {"ecmwf_aifs025_single": "ecmwf_aifs025_single", "ecmwf_ifs025": "ecmwf_ifs025",
            "icon_seamless": "dwd_icon", "gfs_seamless": "ncep_gfs025"}
SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"


def measure_latency() -> list[dict]:
    out = []
    for m in config.WEATHER_MODELS:
        meta_id = META_IDS.get(m["id"])
        r = weather._http().get(f"https://api.open-meteo.com/data/{meta_id}/static/meta.json", timeout=30)
        j = r.json()
        init = pd.Timestamp(j["last_run_initialisation_time"], unit="s", tz="UTC")
        avail = pd.Timestamp(j["last_run_availability_time"], unit="s", tz="UTC")
        measured = round((avail - init) / pd.Timedelta(hours=1), 2)
        out.append({"model": m["id"], "meta_id": meta_id, "last_run_utc": iso_z(init), "available_utc": iso_z(avail),
                    "measured_delay_h": measured, "configured_latency_h": m["latency_h"],
                    "margin_h": round(m["latency_h"] - measured, 2), "ok": bool(m["latency_h"] >= measured)})
    return out


def check_offset_semantics(site: config.Site, day: str = "2026-01-21", runs_back: int = 4) -> dict:
    """Compare previous_day1/2 values of ECMWF IFS HRES with the exact runs they should come from."""
    s = weather._http()
    hv = "wind_speed_100m_previous_day1,wind_speed_100m_previous_day2"
    pr = s.get(weather.PREVIOUS_RUNS_URL, params=dict(latitude=site.lat, longitude=site.lon, hourly=hv, models="ecmwf_ifs",
                                                      start_date=day, end_date=day, timezone="UTC", wind_speed_unit="ms"), timeout=60).json()
    prev = pd.DataFrame(pr["hourly"])
    prev["time"] = pd.to_datetime(prev["time"], utc=True)
    runs = {}
    d0 = pd.Timestamp(day, tz="UTC")
    for k in range(1, runs_back * 4 + 1):                      # every 6-hourly run in the previous days
        init = d0 - pd.Timedelta(hours=6 * k)
        rr = s.get(SINGLE_RUNS_URL, params=dict(latitude=site.lat, longitude=site.lon, hourly="wind_speed_100m", models="ecmwf_ifs",
                                                run=init.strftime("%Y-%m-%dT%H:%M"), forecast_days=4, timezone="UTC",
                                                wind_speed_unit="ms"), timeout=60)
        if rr.status_code == 200:
            x = pd.DataFrame(rr.json()["hourly"])
            runs[init] = pd.Series(x["wind_speed_100m"].to_numpy(), index=pd.to_datetime(x["time"], utc=True))
    checked, matched = 0, 0
    for n in (1, 2):
        col = f"wind_speed_100m_previous_day{n}"
        for _, row in prev.iterrows():
            expected_init = row["time"].floor("6h") - pd.Timedelta(days=n)
            if expected_init in runs and row["time"] in runs[expected_init].index and pd.notna(row[col]):
                checked += 1
                matched += int(abs(runs[expected_init][row["time"]] - row[col]) < 0.015)
    return {"model": "ecmwf_ifs (HRES, Single Runs API)", "day": day, "values_checked": checked, "values_matching": matched,
            "rule": "value at v from run floor_6h(v) - N days", "ok": bool(checked > 0 and matched == checked)}


def run(site_key: str = "shelek") -> dict:
    site = config.get_site(site_key)
    res = {"generated_at_utc": iso_z(now_utc()), "latency": measure_latency(), "previous_runs_semantics": check_offset_semantics(site)}
    out = config.outputs_dir() / "provenance"
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res
