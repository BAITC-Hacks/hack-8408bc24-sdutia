"""The as-of ("time travel") guard.

A weather value from a run initialised at t0 is usable at issue time T only if
    t0 + latency_h(model) <= T.
For every (issue, model, valid hour) we pick the SMALLEST offset N (the freshest run) that
satisfies the rule and has data. Training rows and live forecasts use this same function.

With previous-runs data, the run behind offset N at valid time v is floor_6h(v) - N days, so
the minimal usable offset is N_min = max(1, ceil((floor_6h(v) + latency - T) / 24 h)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

MIN_ISSUE_DATA_CLOCK = "2024-06-01 00:00"   # enough weather archive + SCADA history before it
_VARS = ["ws10", "ws100", "wd100", "t2m", "sp"]


def latency_h(model: str) -> int:
    for m in config.WEATHER_MODELS:
        if m["id"] == model:
            return int(m["latency_h"])
    raise KeyError(model)


def usable(init_time_utc: pd.Series, model: pd.Series | str, issue_time_utc) -> pd.Series:
    lat_h = latency_h(model) if isinstance(model, str) else model.map(latency_h)
    return (init_time_utc + pd.to_timedelta(lat_h, unit="h")) <= issue_time_utc


def select_asof_many(weather_long: pd.DataFrame, issue_times_utc, horizon_h: int) -> pd.DataFrame:
    """For every issue T and model, rows for valid hours v in [T, T + horizon] (inclusive, for
    interval alignment) from the freshest run published by T. Columns:
    issue_time_utc, model, valid_time_utc, offset_days, init_time_utc, lead_from_init_h, ws10..sp."""
    issues = pd.DatetimeIndex(pd.to_datetime(pd.Series(issue_times_utc), utc=True)).unique().sort_values()
    steps = np.arange(horizon_h + 1)
    grid = pd.DataFrame({
        "issue_time_utc": np.repeat(issues, len(steps)),
        "valid_time_utc": np.repeat(issues, len(steps)) + pd.to_timedelta(np.tile(steps, len(issues)), unit="h"),
    })
    offsets = sorted(weather_long["offset_days"].unique())
    out = []
    for model, wm in weather_long.groupby("model", sort=True):
        lat = pd.Timedelta(hours=latency_h(model))
        g = grid.copy()
        g["model"] = model
        need = (g["valid_time_utc"].dt.floor("6h") + lat - g["issue_time_utc"]) / pd.Timedelta(days=1)
        g["n_min"] = np.maximum(1, np.ceil(need.to_numpy() - 1e-9)).astype(int)
        wm = wm.set_index(["valid_time_utc", "offset_days"])[["init_time_utc", *_VARS]]
        chosen = None
        for extra in range(0, 3):                 # fall back to older runs if the freshest is missing
            cand = g[["issue_time_utc", "valid_time_utc"]].copy()
            cand["offset_days"] = g["n_min"] + extra
            cand = cand.join(wm, on=["valid_time_utc", "offset_days"])
            cand = cand[cand["offset_days"].isin(offsets)]
            cand = cand.dropna(subset=["ws100", "ws10"], how="all")
            cand["_rank"] = extra
            chosen = cand if chosen is None else pd.concat([chosen, cand])
        chosen = chosen.sort_values("_rank").drop_duplicates(["issue_time_utc", "valid_time_utc"], keep="first")
        chosen["model"] = model
        out.append(chosen.drop(columns="_rank"))
    sel = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["issue_time_utc", "model", "valid_time_utc"])
    sel["lead_from_init_h"] = (sel["valid_time_utc"] - sel["init_time_utc"]) / pd.Timedelta(hours=1)
    assert_asof(sel)
    cols = ["issue_time_utc", "model", "valid_time_utc", "offset_days", "init_time_utc", "lead_from_init_h", *_VARS]
    return sel[cols].sort_values(["issue_time_utc", "model", "valid_time_utc"]).reset_index(drop=True)


def select_asof(weather_long: pd.DataFrame, issue_time_utc, horizon_h: int) -> pd.DataFrame:
    return select_asof_many(weather_long, [pd.Timestamp(issue_time_utc)], horizon_h)


def assert_asof(selected: pd.DataFrame) -> None:
    """Runtime guard: fail loudly if any selected value comes from a run not yet published at T."""
    if selected.empty:
        return
    ok = usable(selected["init_time_utc"], selected["model"], selected["issue_time_utc"])
    if not bool(ok.all()):
        bad = selected.loc[~ok, ["issue_time_utc", "model", "valid_time_utc", "init_time_utc"]].head(3).to_dict("records")
        raise AssertionError(f"as-of violation: {bad}")


def issue_range_utc(site: config.Site) -> tuple[pd.Timestamp, pd.Timestamp]:
    from .timeutil import now_utc, parse_clock_time

    lo = parse_clock_time(MIN_ISSUE_DATA_CLOCK, site.data_clock_utc_offset_h)
    return lo, now_utc().floor("h")
