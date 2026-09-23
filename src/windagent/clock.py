"""SCADA clock forensics (offline): which fixed UTC offset does the logger use, and did it change?

1. Continuity at known time-zone changes (Kazakhstan: 2024-03-01 00:00, UTC+6 regions moved to UTC+5):
   a logger following the wall clock would show a duplicated hour or a gap there.
2. "Sun check": clock time of the daily temperature peak (first 24 h harmonic) per year for the
   same months. A clock change of 1 h moves it by ~60 min; natural variation is minutes.
3. Weather cross-correlation: SCADA wind vs archived weather forecasts (hour H ↔ mean of H and H+1)
   scanned over candidate offsets, per quarter. Gives the absolute offset (season adds ±0.5 h noise).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, weather

KNOWN_SWITCHES = [{"local_time": "2024-03-01 00:00", "note": "Kazakhstan unified time: UTC+6 regions moved to UTC+5"}]
SUN_MONTHS = {"Apr-Sep": (4, 5, 6, 7, 8, 9), "March": (3,), "Jan-Feb": (1, 2)}


def _raw_local(site: config.Site, turbine_idx: int = 0) -> pd.DataFrame:
    t = site.turbines[turbine_idx]
    raw = pd.read_csv(t.scada_path(), encoding="utf-8")
    df = raw.iloc[:, 1:5].copy()
    df.columns = ["time", "ws", "p", "temp"]
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time")


def continuity(df: pd.DataFrame, local_time: str, window_h: int = 3) -> dict:
    t = pd.Timestamp(local_time)
    w = df[(df["time"] >= t - pd.Timedelta(hours=window_h)) & (df["time"] <= t + pd.Timedelta(hours=window_h))]
    steps = w["time"].diff().dropna()
    return {"records": int(len(w)), "expected": window_h * 12 + 1, "duplicates": int(w["time"].duplicated().sum()),
            "irregular_steps": int((steps != pd.Timedelta(minutes=10)).sum())}


def sun_peaks(df: pd.DataFrame) -> dict:
    out = {}
    s = df.set_index("time")["temp"].dropna()
    for label, months in SUN_MONTHS.items():
        per_year = {}
        for year in sorted(s.index.year.unique()):
            x = s[(s.index.year == year) & (s.index.month.isin(months))]
            # compare like with like: every month of the group must be (mostly) present that year
            per_month = x.groupby(x.index.month).size()
            if len(x) < 2000 or set(per_month.index) != set(months) or (per_month < 2500).any():
                continue
            hrs = x.index.hour + x.index.minute / 60
            anom = x - x.groupby(x.index.date).transform("mean")
            a = float(np.sum(anom * np.cos(2 * np.pi * hrs / 24)))
            b = float(np.sum(anom * np.sin(2 * np.pi * hrs / 24)))
            peak_h = (np.degrees(np.arctan2(b, a)) % 360) / 15
            per_year[str(year)] = f"{int(peak_h):02d}:{int(round((peak_h % 1) * 60)) % 60:02d}"
        out[label] = per_year
    return out


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def nwp_offsets_by_quarter(site: config.Site, df: pd.DataFrame, models=("icon_seamless", "gfs_seamless", "ecmwf_ifs025")) -> dict:
    frames = []
    for m in models:
        try:
            c = weather.load_cache(site.site, m)
        except Exception:  # noqa: BLE001
            continue
        frames.append(c[c["offset_days"] == 1].set_index("valid_time_utc")["ws100"].rename(m))
    if not frames:
        return {}
    ref = pd.concat(frames, axis=1).mean(axis=1)
    ref_mid = (ref + ref.shift(-1)) / 2                                  # interval mean of [H, H+1)
    s = df.set_index("time")["ws"]
    out = {}
    for q, x in s.groupby(s.index.to_period("Q")):
        if len(x) < 5000:
            continue
        best, best_c = None, -2.0
        for L in np.arange(3.0, 9.01, 1 / 6):
            y = x.copy()
            y.index = (y.index - pd.Timedelta(minutes=round(L * 60))).tz_localize("UTC")
            hh = y.resample("1h").mean()
            j = pd.concat([hh.rename("obs"), ref_mid.rename("nwp")], axis=1, join="inner").dropna()
            if len(j) < 500:
                break
            c = j["obs"].corr(j["nwp"])
            if c > best_c:
                best, best_c = L, c
        if best is not None:
            out[str(q)] = {"best_offset_h": round(float(best), 2), "corr": round(float(best_c), 3)}
    return out


def detect_clock(site_key: str = "shelek") -> dict:
    site = config.get_site(site_key)
    df = _raw_local(site)
    cont = {sw["local_time"]: {**continuity(df, sw["local_time"]), "note": sw["note"]} for sw in KNOWN_SWITCHES}
    sun = sun_peaks(df)
    shifts = []
    for label, per_year in sun.items():
        mins = [_minutes(v) for v in per_year.values()]
        if len(mins) >= 2:
            shifts.append(max(mins) - min(mins))
    max_shift = int(max(shifts)) if shifts else None
    nwp = nwp_offsets_by_quarter(site, df)
    est = int(round(float(np.median([v["best_offset_h"] for v in nwp.values()])))) if nwp else None
    no_switch = all(c["duplicates"] == 0 and c["irregular_steps"] == 0 for c in cont.values()) and (max_shift is not None and max_shift < 30)
    conclusion = (f"Fixed UTC+{est} logger clock; no change at the 2024-03-01 switch "
                  f"(no duplicate/gap at the switch, temperature-peak time stable within {max_shift} min across years)."
                  if no_switch and est is not None else "Inconclusive: inspect the details.")
    report = {"site": site.site, "configured_offset_h": site.data_clock_utc_offset_h, "estimated_offset_h": est,
              "matches_config": est == site.data_clock_utc_offset_h, "switch_continuity": cont,
              "sun_peak_clock_time_by_year": sun, "sun_peak_max_shift_min": max_shift,
              "nwp_best_offset_by_quarter": nwp, "conclusion": conclusion}
    out = config.outputs_dir() / site.site / "clock"
    out.mkdir(parents=True, exist_ok=True)
    (out / "clock_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
