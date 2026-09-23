"""Single-issue forecasting: model choice (as-of cutoff), weather → features → P10/P50/P90,
intraday blend with fresh SCADA, farm aggregation, output rows in the contract schemas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import asof, config, dataset, features, scada
from .model import ForecastModel
from .timeutil import clock_str, iso_z, issue_id

NOWCAST_TAU_H = 2.0              # blend weight w = exp(-(lead + 1) / tau), tuned on validation
FRESH_SCADA_MAX_AGE_H = 2.0      # SCADA older than this is not used for the intraday blend
TRAIN_START_DATA_CLOCK = "2024-03-10"
PRODUCTION_CUTOFF_DATA_CLOCK = "2026-01-31 00:00"   # the first test-period issue time


def production_model_path(site: config.Site) -> Path:
    return config.models_dir() / site.site / "model_production.pkl"


def _asof_model_path(site: config.Site, cutoff_utc: pd.Timestamp) -> Path:
    return config.models_dir() / site.site / f"model_cutoff_{cutoff_utc:%Y%m%dT%H%M}Z.pkl"


def train_model(site: config.Site, cutoff_utc: pd.Timestamp, log=print, weather_long: pd.DataFrame | None = None) -> ForecastModel:
    """Train on every daily issue before the cutoff (targets strictly before the cutoff)."""
    from .timeutil import clock_str as _cs
    w = weather_long if weather_long is not None else dataset.load_weather_cache(site)
    end_clock = _cs(cutoff_utc - pd.Timedelta(days=1), site.data_clock_utc_offset_h)
    issues = dataset.daily_issue_times(site, TRAIN_START_DATA_CLOCK, end_clock)
    rows = dataset.build_rows(site, issues, config.HORIZON_H, w)
    train = dataset.training_rows(rows, cutoff_utc)
    if len(train) < 5000:
        from .errors import DataUnavailableError
        raise DataUnavailableError(f"Not enough history before {iso_z(cutoff_utc)} to train a model ({len(train)} rows).",
                                   f"Недостаточно истории до {iso_z(cutoff_utc)} для обучения модели ({len(train)} строк).")
    m = ForecastModel(turbine_ids=site.turbine_ids).fit(train)
    m.cutoff_utc = iso_z(cutoff_utc)
    metrics_path = config.outputs_dir() / site.site / "validation" / "metrics.json"
    if metrics_path.exists():
        cal = json.loads(metrics_path.read_text(encoding="utf-8")).get("interval_calibration") or {}
        m.interval_scale = float(cal.get("k", 1.0))
        m.calibration = cal
    log(f"trained model: cutoff {m.cutoff_utc}, {m.n_train} rows, interval scale {m.interval_scale}")
    return m


def model_for_issue(site: config.Site, issue_time_utc: pd.Timestamp, log=print) -> tuple[ForecastModel, str]:
    """Newest model whose training cutoff is <= the issue time; trains one if none exists.
    Returns (model, decision text for the trace)."""
    from .timeutil import parse_clock_time
    prod_cutoff = parse_clock_time(PRODUCTION_CUTOFF_DATA_CLOCK, site.data_clock_utc_offset_h)
    if issue_time_utc >= prod_cutoff:
        p = production_model_path(site)
        if p.exists():
            try:
                return ForecastModel.load(p), f"production model (cutoff {iso_z(prod_cutoff)})"
            except Exception:  # noqa: BLE001 - incompatible pickle → retrain below
                pass
        m = train_model(site, prod_cutoff, log)
        m.save(p)
        return m, f"production model trained now (cutoff {iso_z(prod_cutoff)})"
    candidates = []
    for p in (config.models_dir() / site.site).glob("model_cutoff_*Z.pkl"):
        try:
            c = pd.Timestamp(p.stem.replace("model_cutoff_", ""), tz="UTC")
        except ValueError:
            continue
        if c <= issue_time_utc:
            candidates.append((c, p))
    if candidates:
        c, p = max(candidates)
        try:
            return ForecastModel.load(p), f"as-of model (cutoff {iso_z(c)} <= issue time)"
        except Exception:  # noqa: BLE001
            pass
    cutoff = issue_time_utc.floor("D")
    m = train_model(site, cutoff, log)
    m.save(_asof_model_path(site, cutoff))
    return m, f"issue time precedes the production cutoff: trained an as-of model (cutoff {iso_z(cutoff)})"


def forecast_rows(site: config.Site, model: ForecastModel, weather_long: pd.DataFrame, issue_time_utc: pd.Timestamp,
                  as_of_utc: pd.Timestamp, window_end_utc: pd.Timestamp, exclude_models=()) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Forecast target hours [as_of, window_end) with weather published by as_of.
    Returns (per-entity predictions incl. farm, weather rows used, info)."""
    horizon = int((window_end_utc - as_of_utc) / pd.Timedelta(hours=1))
    w = weather_long[~weather_long["model"].isin(list(exclude_models))]
    sel = asof.select_asof(w, as_of_utc, horizon)
    if sel.empty:
        from .errors import DataUnavailableError
        raise DataUnavailableError("No usable weather forecasts for this issue time.",
                                   "Нет доступных прогнозов погоды для этого времени выпуска.")
    feats = features.make_features(sel, horizon)
    rows = features.with_turbines(feats, site.turbine_ids)
    pred = pd.concat([rows[["issue_time_utc", "target_time_utc", "entity", "lead_h"]], model.predict(rows)], axis=1)

    # intraday blend with the latest SCADA observation, only if it is fresh at as_of
    info = {"nowcast": None}
    obs = scada.site_hourly_long(site)
    obs = obs[(obs["entity"] != "farm") & (obs["target_time_utc"] + pd.Timedelta(hours=1) <= as_of_utc)]
    if not obs.empty:
        last_t = obs["target_time_utc"].max()
        age_h = (as_of_utc - (last_t + pd.Timedelta(hours=1))) / pd.Timedelta(hours=1)
        if age_h <= FRESH_SCADA_MAX_AGE_H:
            last = obs[obs["target_time_utc"] == last_t].set_index("entity")["actual"]
            lead_asof = (pred["target_time_utc"] - as_of_utc) / pd.Timedelta(hours=1)
            wgt = np.exp(-(lead_asof + 1) / NOWCAST_TAU_H)
            p_last = pred["entity"].map(last)
            delta = (wgt * (p_last - pred["mean"])).fillna(0.0)
            for c in ("mean", "p10", "p50", "p90"):
                pred[c] = (pred[c] + delta).clip(0.0, 1.0)
            info["nowcast"] = {"last_obs_utc": iso_z(last_t), "age_h": round(float(age_h), 2), "tau_h": NOWCAST_TAU_H}

    farm = pred.groupby(["issue_time_utc", "target_time_utc", "lead_h"], as_index=False)[["mean", "p10", "p50", "p90"]].mean()
    farm["entity"] = "farm"
    out = pd.concat([pred, farm], ignore_index=True)
    info["models_used"] = sorted(sel["model"].unique())
    info["runs_used"] = {m: sorted(iso_z(pd.Series(g["init_time_utc"].unique())).tolist())
                         for m, g in sel.groupby("model")}
    return out, sel, info


def to_schema_f(pred: pd.DataFrame, site: config.Site, issue_time_utc: pd.Timestamp, version: int,
                as_of_utc: pd.Timestamp) -> pd.DataFrame:
    df = pd.DataFrame({
        "site": site.site,
        "issue_id": issue_id(issue_time_utc, site.data_clock_utc_offset_h),
        "version": version,
        "issue_time_utc": iso_z(issue_time_utc),
        "version_as_of_utc": iso_z(as_of_utc),
        "target_time_utc": iso_z(pred["target_time_utc"]),
        "target_time_data_clock": clock_str(pred["target_time_utc"], site.data_clock_utc_offset_h),
        "target_time_kz_official": clock_str(pred["target_time_utc"], site.official_utc_offset_h),
    })
    lead = ((pred["target_time_utc"] - issue_time_utc) / pd.Timedelta(hours=1)).astype(int)
    df["lead_h"] = lead.to_numpy()
    df["product"] = np.where(lead < 24, "intraday", np.where(lead < 48, "day_ahead", "extended"))
    df["entity"] = pred["entity"].to_numpy()
    for c in ("mean", "p10", "p50", "p90"):
        df[c] = pred[c].round(4).to_numpy()
    df["mw_mean"] = (pred["mean"] * site.rated_mw).round(3).to_numpy() if site.rated_mw else np.nan
    order = {"farm": 0, **{t: i + 1 for i, t in enumerate(site.turbine_ids)}}
    return df.assign(_o=df["entity"].map(order)).sort_values(["version", "_o", "target_time_utc"]).drop(columns="_o").reset_index(drop=True)


def to_schema_w(sel: pd.DataFrame, site: config.Site, issue_time_utc: pd.Timestamp, version: int,
                source: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "issue_id": issue_id(issue_time_utc, site.data_clock_utc_offset_h),
        "version": version,
        "target_time_utc": iso_z(sel["valid_time_utc"]),
        "model": sel["model"].to_numpy(),
        "init_time_utc": iso_z(sel["init_time_utc"]),
        "offset_days": sel["offset_days"].astype(int).to_numpy(),
    })
    for c in ("ws10", "ws100", "wd100", "t2m", "sp"):
        df[c] = sel[c].round(2).to_numpy()
    df["source"] = sel["model"].map(source).fillna("cache").to_numpy()
    return df
