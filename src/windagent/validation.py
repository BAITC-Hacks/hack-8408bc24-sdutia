"""Walk-forward validation against baselines, with the same as-of pipeline as production."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import config, dataset, scada
from .model import ForecastModel
from .timeutil import iso_z, now_utc, parse_clock_time

FOLD_MONTHS = ["2025-10", "2025-11", "2025-12", "2026-01"]
TRAIN_START_DATA_CLOCK = "2024-03-10"


def product_of(lead_h: pd.Series) -> pd.Series:
    return pd.Series(np.where(lead_h < 24, "intraday", np.where(lead_h < 48, "day_ahead", "extended")), index=lead_h.index)


def add_baselines(rows: pd.DataFrame, site: config.Site, train: pd.DataFrame, model: ForecastModel) -> pd.DataFrame:
    """persistence (last complete hour before T), climatology (month × hour mean of training data),
    nwp_powercurve (isotonic curve of the best single weather model)."""
    out = rows.copy()
    act = scada.site_hourly_long(site)
    act = act[act["entity"] != "farm"][["target_time_utc", "entity", "actual"]]
    last = act.rename(columns={"target_time_utc": "t_last", "actual": "persistence"})
    out["t_last"] = out["issue_time_utc"] - pd.Timedelta(hours=1)
    out = out.merge(last, on=["t_last", "entity"], how="left").drop(columns="t_last")
    clim = train.assign(month=train["target_time_utc"].dt.month, hour=train["target_time_utc"].dt.hour)
    clim = clim.groupby(["entity", "month", "hour"])["actual"].mean().rename("climatology").reset_index()
    out = out.assign(month=out["target_time_utc"].dt.month, hour=out["target_time_utc"].dt.hour)
    out = out.merge(clim, on=["entity", "month", "hour"], how="left").drop(columns=["month", "hour"])
    out["nwp_powercurve"] = model.predict_powercurve(out).to_numpy()
    return out


def to_farm(pred: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Farm = mean of turbines; farm actual only where every turbine has an actual."""
    g = pred.groupby(["issue_time_utc", "target_time_utc"])
    farm = g[value_cols].mean()
    if "actual" in pred:
        n_tur = pred["entity"].nunique()
        complete = g["actual"].count() == n_tur
        farm["actual"] = g["actual"].mean().where(complete)
        farm["available"] = g["available"].apply(lambda s: bool(s.fillna(False).all()))
    farm = farm.reset_index()
    farm["entity"] = "farm"
    farm["lead_h"] = ((farm["target_time_utc"] - farm["issue_time_utc"]) / pd.Timedelta(hours=1)).astype(int)
    return farm


def _scores(y: np.ndarray, f: np.ndarray) -> dict:
    ok = ~(np.isnan(y) | np.isnan(f))
    y, f = y[ok], f[ok]
    if len(y) == 0:
        return {"mae": None, "rmse": None, "bias": None, "r2": None, "n": 0}
    err = f - y
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"mae": round(float(np.mean(np.abs(err))), 4), "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "bias": round(float(np.mean(err)), 4), "r2": round(1 - float(np.sum(err ** 2)) / ss, 4) if ss > 0 else None,
            "n": int(len(y))}


def run_validation(site_key: str = "shelek", horizon_h: int = config.HORIZON_H, log=print) -> dict:
    site = config.get_site(site_key)
    t0 = time.time()
    w = dataset.load_weather_cache(site)
    last_obs = scada.last_observation_time(site)
    issues = dataset.daily_issue_times(site, TRAIN_START_DATA_CLOCK, "2026-01-31")
    log(f"building rows for {len(issues)} daily issues…")
    rows = dataset.build_rows(site, issues, horizon_h, w)
    rows["lead_h"] = ((rows["target_time_utc"] - rows["issue_time_utc"]) / pd.Timedelta(hours=1)).astype(int)
    preds = []
    loto = {}
    for month in FOLD_MONTHS:
        start = parse_clock_time(f"{month}-01 00:00", site.data_clock_utc_offset_h)
        end = parse_clock_time(f"{(pd.Period(month) + 1).strftime('%Y-%m')}-01 00:00", site.data_clock_utc_offset_h)
        train = dataset.training_rows(rows, start)
        test = rows[(rows["issue_time_utc"] >= start) & (rows["issue_time_utc"] < end)].copy()
        m = ForecastModel(turbine_ids=site.turbine_ids).fit(train)
        m.cutoff_utc = iso_z(start)
        m.save(config.models_dir() / site.site / f"model_cutoff_{month}.pkl")
        p = m.predict(test)
        test = pd.concat([test, p], axis=1)
        test = add_baselines(test, site, train, m)
        test["fold"] = month
        preds.append(test)
        log(f"  fold {month}: train rows {len(train)}, test rows {len(test)}, {time.time() - t0:.0f}s")
        if month == FOLD_MONTHS[-1] and len(site.turbine_ids) >= 2:
            a, b = site.turbine_ids[0], site.turbine_ids[1]
            m1 = ForecastModel(turbine_ids=site.turbine_ids).fit(train[train["entity"] == a])
            tb = test[test["entity"] == b]
            pb = m1.predict(tb.drop(columns=["mean", "p10", "p50", "p90"]))
            da = (tb["lead_h"] >= 24) & (tb["lead_h"] < 48)
            loto = {"train": a, "test": b, "fold": month,
                    "mae": _scores(tb.loc[da, "actual"].to_numpy(float), pb.loc[da, "mean"].to_numpy(float))["mae"],
                    "mae_reference": _scores(tb.loc[da, "actual"].to_numpy(float), tb.loc[da, "mean"].to_numpy(float))["mae"],
                    "note": f"day-ahead MAE on {b} when trained only on {a} (mae) vs trained on all turbines (mae_reference)"}
    allp = pd.concat(preds, ignore_index=True)
    cols = ["mean", "p10", "p50", "p90", "persistence", "climatology", "nwp_powercurve"]
    farm = to_farm(allp, cols)
    farm["product"] = product_of(farm["lead_h"])
    metrics = _metrics(farm, site, issues_n=int(allp["issue_time_utc"].nunique()))
    metrics["leave_one_turbine_out"] = loto
    metrics["validation"]["last_observation_utc"] = iso_z(last_obs)
    _write(site, metrics, allp, farm)
    log(f"validation done in {time.time() - t0:.0f}s")
    return metrics


def _metrics(farm: pd.DataFrame, site: config.Site, issues_n: int) -> dict:
    ev = farm[farm["actual"].notna()]
    by_product, by_product_avail = {}, {}
    for prod in ("intraday", "day_ahead"):
        d = ev[ev["product"] == prod]
        da = d[d["available"]]
        by_product[prod] = {k: _scores(d["actual"].to_numpy(float), d[c].to_numpy(float))
                            for k, c in [("model", "mean"), ("persistence", "persistence"),
                                         ("climatology", "climatology"), ("nwp_powercurve", "nwp_powercurve")]}
        by_product_avail[prod] = {"model": _scores(da["actual"].to_numpy(float), da["mean"].to_numpy(float))}
    by_lead = []
    for lead, d in ev.groupby("lead_h"):
        by_lead.append({"lead_h": int(lead), **{k: _scores(d["actual"].to_numpy(float), d[c].to_numpy(float))["mae"]
                                               for k, c in [("model", "mean"), ("persistence", "persistence"),
                                                            ("climatology", "climatology"), ("nwp_powercurve", "nwp_powercurve")]}})
    da = ev[ev["product"] == "day_ahead"]
    inside = ((da["actual"] >= da["p10"]) & (da["actual"] <= da["p90"])).mean()
    per_month = []
    issue_month = (da["issue_time_utc"] + pd.Timedelta(hours=site.data_clock_utc_offset_h)).dt.strftime("%Y-%m")
    for month, d in da.groupby(issue_month):
        per_month.append({"month": month, **{k: _scores(d["actual"].to_numpy(float), d[c].to_numpy(float))["mae"]
                                             for k, c in [("model", "mean"), ("climatology", "climatology"),
                                                          ("persistence", "persistence"), ("nwp_powercurve", "nwp_powercurve")]}})
    m_da = by_product["day_ahead"]
    skill = {}
    for ref in ("climatology", "persistence", "nwp_powercurve"):
        if m_da[ref]["mae"]:
            skill[f"day_ahead_vs_{ref}"] = round(100 * (1 - m_da["model"]["mae"] / m_da[ref]["mae"]), 1)
    return {
        "site": site.site, "generated_at_utc": iso_z(now_utc()),
        "validation": {"folds": FOLD_MONTHS, "start_utc": iso_z(ev["issue_time_utc"].min()),
                       "end_utc": iso_z(ev["target_time_utc"].max()), "n_issues": issues_n,
                       "issue_hour_data_clock": config.ISSUE_HOUR_DATA_CLOCK, "entity": "farm",
                       "method": "walk-forward: for each month, train on all earlier data, then issue daily forecasts "
                                 "with the same as-of weather selection as production"},
        "by_product": by_product,
        "by_product_available_hours_only": by_product_avail,
        "by_lead": by_lead,
        "intervals": {"day_ahead": {"coverage_p10_p90": round(float(inside), 3),
                                    "mean_width": round(float((da["p90"] - da["p10"]).mean()), 3)}},
        "skill_pct": skill,
        "per_month": per_month,
    }


def _write(site: config.Site, metrics: dict, allp: pd.DataFrame, farm: pd.DataFrame) -> None:
    from .timeutil import clock_str

    out = config.outputs_dir() / site.site / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    keep = ["issue_time_utc", "target_time_utc", "lead_h", "entity", "mean", "p10", "p50", "p90", "actual",
            "persistence", "climatology", "nwp_powercurve"]
    both = pd.concat([allp[keep], farm[keep]], ignore_index=True)
    both["target_time_data_clock"] = clock_str(both["target_time_utc"], site.data_clock_utc_offset_h)
    both["product"] = product_of(both["lead_h"])
    for c in ("issue_time_utc", "target_time_utc"):
        both[c] = iso_z(both[c])
    both.round(4).to_csv(out / "predictions.csv", index=False, encoding="utf-8")
