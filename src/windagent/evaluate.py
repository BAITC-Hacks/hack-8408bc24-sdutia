"""Score a forecast against actuals given in the organizers' CSV format (10-minute, data clock).

Includes a clock-alignment check: if the best correlation between forecast and actuals is at a
non-zero lag (e.g. actuals exported in official UTC+5 instead of the UTC+6 data clock), metrics
are reported both as-is and re-aligned, with a warning.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, scada
from .errors import InputError
from .timeutil import clock_str, iso_z


def load_actuals_csv(path: Path, data_clock_utc_offset_h: int) -> pd.Series:
    """Organizers' 10-minute file → hourly mean normalized power indexed by UTC hour."""
    h = scada.to_hourly(scada.read_scada_csv(Path(path), data_clock_utc_offset_h))
    return h["p"]


def _forecast_long(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        from .errors import DataUnavailableError
        raise DataUnavailableError(f"Forecast file not found: {path}", f"Файл прогноза не найден: {path}")
    f = pd.read_csv(path, encoding="utf-8")
    if "farm_mean" in f.columns:                                   # schema S (wide submission)
        parts = [pd.DataFrame({"target_time_utc": f["target_time_utc"], "entity": "farm", "mean": f["farm_mean"],
                               "p10": f.get("farm_p10"), "p90": f.get("farm_p90")})]
        for c in f.columns:
            if c.endswith("_mean") and c != "farm_mean":
                parts.append(pd.DataFrame({"target_time_utc": f["target_time_utc"], "entity": c[:-5], "mean": f[c]}))
        long = pd.concat(parts, ignore_index=True)
    elif {"entity", "mean", "target_time_utc"} <= set(f.columns):  # schema F (long)
        if "version" in f.columns:
            f = f[f["version"] == 1]
        if "product" in f.columns and (f["product"] == "day_ahead").any():
            f = f[f["product"] == "day_ahead"]
        long = f[[c for c in ("target_time_utc", "entity", "mean", "p10", "p90") if c in f.columns]].copy()
    else:
        raise InputError(f"{Path(path).name}: not a forecast file (need schema S or F columns).",
                         f"{Path(path).name}: это не файл прогноза (нужны столбцы схемы S или F).")
    long["target_time_utc"] = pd.to_datetime(long["target_time_utc"], utc=True)
    return long.drop_duplicates(["target_time_utc", "entity"], keep="first")


def detect_lag_hours(fc: pd.Series, act: pd.Series, max_lag_h: int = 3) -> tuple[int, dict]:
    """Best integer lag L maximizing corr(forecast(t), actual(t + L h))."""
    corr = {}
    for lag in range(-max_lag_h, max_lag_h + 1):
        shifted = act.copy()
        shifted.index = shifted.index - pd.Timedelta(hours=lag)
        j = pd.concat([fc.rename("f"), shifted.rename("a")], axis=1, join="inner").dropna()
        corr[lag] = round(float(j["f"].corr(j["a"])), 4) if len(j) > 24 else None
    valid = {k: v for k, v in corr.items() if v is not None}
    return (max(valid, key=valid.get) if valid else 0), corr


def _scores(y, f) -> dict:
    y, f = np.asarray(y, float), np.asarray(f, float)
    ok = ~(np.isnan(y) | np.isnan(f))
    y, f = y[ok], f[ok]
    if len(y) == 0:
        return {"mae": None, "rmse": None, "bias": None, "r2": None, "n": 0}
    e = f - y
    ss = float(((y - y.mean()) ** 2).sum())
    return {"mae": round(float(np.abs(e).mean()), 4), "rmse": round(float(np.sqrt((e ** 2).mean())), 4),
            "bias": round(float(e.mean()), 4), "r2": round(1 - float((e ** 2).sum()) / ss, 4) if ss > 0 else None, "n": int(len(y))}


def _metrics(fc: pd.DataFrame, actuals: dict[str, pd.Series], shift_h: int = 0) -> tuple[dict, pd.DataFrame]:
    acts = {}
    for ent, s in actuals.items():
        s2 = s.copy()
        s2.index = s2.index - pd.Timedelta(hours=shift_h)
        acts[ent] = s2
    wide = pd.DataFrame(acts)
    if len(acts) > 1:
        acts["farm"] = wide.mean(axis=1).where(wide.notna().all(axis=1))
    out, rows = {}, []
    for ent, a in acts.items():
        f = fc[fc["entity"] == ent].set_index("target_time_utc")
        if f.empty:
            continue
        j = f.join(a.rename("actual"), how="inner")
        m = _scores(j["actual"], j["mean"])
        if ent == "farm" and "p10" in j and j["p10"].notna().any():
            jj = j.dropna(subset=["actual", "p10", "p90"])
            m["coverage_p10_p90"] = round(float(((jj["actual"] >= jj["p10"]) & (jj["actual"] <= jj["p90"])).mean()), 3) if len(jj) else None
        out[ent] = m
        rows.append(j.reset_index().assign(entity=ent))
    return out, (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())


def evaluate(site_key: str, actual_paths: dict[str, str], forecast_path: str | None = None, out_dir: Path | None = None) -> dict:
    site = config.get_site(site_key)
    if not actual_paths:
        raise InputError("Give at least one --actuals <turbine>=<path>.", "Укажите хотя бы один --actuals <турбина>=<путь>.")
    unknown = [t for t in actual_paths if t not in site.turbine_ids]
    if unknown:
        raise InputError(f"Unknown turbine ids {unknown}; expected {site.turbine_ids}.",
                         f"Неизвестные турбины {unknown}; ожидаются {site.turbine_ids}.")
    fpath = Path(forecast_path) if forecast_path else config.outputs_dir() / site.site / "test_period" / "submission_day_ahead.csv"
    fc = _forecast_long(fpath)
    actuals = {t: load_actuals_csv(Path(p), site.data_clock_utc_offset_h) for t, p in actual_paths.items()}
    ref_ent = "farm" if len(actuals) > 1 and (fc["entity"] == "farm").any() else next(iter(actuals))
    ref_act = pd.DataFrame(actuals).mean(axis=1) if ref_ent == "farm" else actuals[ref_ent]
    ref_fc = fc[fc["entity"] == ref_ent].set_index("target_time_utc")["mean"]
    best_lag, corr = detect_lag_hours(ref_fc, ref_act)
    metrics, joined = _metrics(fc, actuals)
    if not any(m.get("n") for m in metrics.values()):
        raise InputError("No overlapping hours between the forecast and the actuals.",
                         "Нет пересекающихся часов между прогнозом и фактом.")
    realigned, warning = None, None
    if best_lag != 0:
        realigned, _ = _metrics(fc, actuals, shift_h=best_lag)
        warning = (f"Actuals correlate best with the forecast when shifted by {best_lag:+d} h. The actuals may use a different "
                   f"clock (the dataset clock is UTC+{site.data_clock_utc_offset_h}; official Kazakhstan time is "
                   f"UTC+{site.official_utc_offset_h}). Metrics are reported as-is and re-aligned.")
    by_day = []
    if len(joined):
        fj = joined[joined["entity"] == ref_ent].copy()
        fj["day"] = clock_str(fj["target_time_utc"], site.data_clock_utc_offset_h).str[:10]
        for day, d in fj.groupby("day"):
            by_day.append({"date_data_clock": day, f"{ref_ent}_mae": _scores(d["actual"], d["mean"])["mae"]})
    period = {"start_utc": iso_z(joined["target_time_utc"].min()), "end_utc": iso_z(joined["target_time_utc"].max())} if len(joined) else {}
    res = {"forecast_file": str(fpath), "actuals_files": {k: str(v) for k, v in actual_paths.items()}, "period": period,
           "best_lag_h": int(best_lag), "lag_corr": {str(k): v for k, v in corr.items()}, "warning": warning,
           "metrics": metrics, "metrics_realigned": realigned, "by_day": by_day}
    out_dir = out_dir or config.outputs_dir() / site.site / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    if len(joined):
        j = joined.copy()
        j["target_time_data_clock"] = clock_str(j["target_time_utc"], site.data_clock_utc_offset_h)
        j["target_time_utc"] = iso_z(j["target_time_utc"])
        j.round(4).to_csv(out_dir / "evaluation_by_hour.csv", index=False, encoding="utf-8")
    return res
