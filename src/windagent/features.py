"""Feature engineering from as-of weather selections.

SCADA hours are MEANS over [H, H+1); weather values are INSTANTANEOUS at full hours, so each
weather feature for hour H is the average of the values at H and H+1 (interval alignment).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_SHORT = {
    "ecmwf_aifs025_single": "aifs",
    "ecmwf_ifs025": "ifs",
    "icon_seamless": "icon",
    "gfs_seamless": "gfs",
}
_INTERVAL_VARS = ["ws10", "ws100", "u100", "v100", "t2m", "sp", "lead_from_init_h"]
KEYS = ["issue_time_utc", "target_time_utc"]


def _interval_mean(sel: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    s = sel.copy()
    rad = np.deg2rad(s["wd100"].astype(float))
    s["u100"] = -s["ws100"] * np.sin(rad)
    s["v100"] = -s["ws100"] * np.cos(rad)
    cur = s[["issue_time_utc", "model", "valid_time_utc", *_INTERVAL_VARS]]
    nxt = cur.copy()
    nxt["valid_time_utc"] = nxt["valid_time_utc"] - pd.Timedelta(hours=1)
    m = cur.merge(nxt, on=["issue_time_utc", "model", "valid_time_utc"], how="left", suffixes=("", "_next"))
    for v in _INTERVAL_VARS:
        m[v] = m[[v, f"{v}_next"]].mean(axis=1)          # falls back to the start value if next is missing
    m = m.rename(columns={"valid_time_utc": "target_time_utc"})
    lead = (m["target_time_utc"] - m["issue_time_utc"]) / pd.Timedelta(hours=1)
    return m[(lead >= 0) & (lead < horizon_h)][["issue_time_utc", "model", "target_time_utc", *_INTERVAL_VARS]]


def make_features(sel: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """sel: output of asof.select_asof_many → one row per (issue, target hour) with wide features."""
    iv = _interval_mean(sel, horizon_h)
    iv["m"] = iv["model"].map(MODEL_SHORT).fillna(iv["model"])
    wide = iv.pivot_table(index=KEYS, columns="m", values=_INTERVAL_VARS, aggfunc="first")
    wide.columns = [f"{m}_{v}" for v, m in wide.columns]
    wide = wide.reset_index()
    for short in sorted(set(iv["m"])):
        ws100, ws10 = wide.get(f"{short}_ws100"), wide.get(f"{short}_ws10")
        if ws100 is None:
            continue
        spd = np.hypot(wide[f"{short}_u100"], wide[f"{short}_v100"]).replace(0, np.nan)
        wide[f"{short}_dir_sin"] = -wide[f"{short}_u100"] / spd
        wide[f"{short}_dir_cos"] = -wide[f"{short}_v100"] / spd
        wide[f"{short}_shear"] = (ws100 / ws10.clip(lower=0.3)).clip(0.3, 6.0)
        wide[f"{short}_rho"] = wide[f"{short}_sp"] * 100.0 / (287.05 * (wide[f"{short}_t2m"] + 273.15))
        wide = wide.drop(columns=[f"{short}_u100", f"{short}_v100"])
    ws_cols = [c for c in wide.columns if c.endswith("_ws100")]
    ens = wide[ws_cols]
    wide["ens_ws100_mean"] = ens.mean(axis=1)
    wide["ens_ws100_std"] = ens.std(axis=1)
    wide["ens_ws100_min"] = ens.min(axis=1)
    wide["ens_ws100_max"] = ens.max(axis=1)
    wide["ens_ws10_mean"] = wide[[c for c in wide.columns if c.endswith("_ws10")]].mean(axis=1)
    wide["ens_n_models"] = ens.notna().sum(axis=1)
    wide["lead_h"] = ((wide["target_time_utc"] - wide["issue_time_utc"]) / pd.Timedelta(hours=1)).astype(int)
    hour = wide["target_time_utc"].dt.hour + 0.5
    doy = wide["target_time_utc"].dt.dayofyear
    wide["hour_sin"], wide["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    wide["doy_sin"], wide["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return wide.sort_values(KEYS).reset_index(drop=True)


def with_turbines(feats: pd.DataFrame, turbine_ids: list[str]) -> pd.DataFrame:
    t = pd.DataFrame({"entity": turbine_ids})
    return feats.merge(t, how="cross")
