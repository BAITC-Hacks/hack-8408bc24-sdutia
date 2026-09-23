"""Agent tools. Each tool takes the run state + validated arguments and returns
{"ok": bool, "summary": str, "data": {...}}. The issue time is fixed by the runner: no tool can
move it, so an LLM cannot request future data. Forecast numbers come only from the ML model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import asof, config, forecast, scada, weather
from ..errors import WindAgentError
from ..model import ForecastModel
from ..timeutil import clock_str, iso_z, issue_id

RAMP_THRESHOLD = 0.30          # |Δ farm power| over 3 h, normalized
HIGH_WIND_MS = 20.0            # forecast 100 m wind: cut-out risk zone
OUTLIER_MIN_MS = 3.0           # an ensemble member this far (m/s) from the others is suspicious


@dataclass
class RunState:
    site: config.Site
    issue_time_utc: pd.Timestamp
    horizon_h: int
    offline: bool
    kind: str = "runs"
    as_of_utc: pd.Timestamp | None = None
    model: ForecastModel | None = None
    model_note: str = ""
    weather_raw: pd.DataFrame | None = None
    weather_source: dict = field(default_factory=dict)
    excluded: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    scada_info: dict = field(default_factory=dict)
    current: dict | None = None
    versions: list = field(default_factory=list)
    analysis: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    pending_reason: str = "initial"

    def __post_init__(self):
        if self.as_of_utc is None:
            self.as_of_utc = self.issue_time_utc

    @property
    def window_end_utc(self) -> pd.Timestamp:
        return self.issue_time_utc + pd.Timedelta(hours=self.horizon_h)

    @property
    def issue_id(self) -> str:
        return issue_id(self.issue_time_utc, self.site.data_clock_utc_offset_h)

    def dc(self, ts) -> str:
        return clock_str(ts, self.site.data_clock_utc_offset_h)


def _ok(summary: str, **data) -> dict:
    return {"ok": True, "summary": summary, "data": data}


def _err(summary: str, **data) -> dict:
    return {"ok": False, "summary": summary, "data": data}


def _active_models(state: RunState) -> list[str]:
    return [m for m in weather.model_ids() if m not in state.excluded]


# ---- tools ------------------------------------------------------------------------------------

def inspect_scada(state: RunState) -> dict:
    obs = scada.site_hourly_long(state.site)
    tur = obs[(obs["entity"] != "farm") & (obs["target_time_utc"] + pd.Timedelta(hours=1) <= state.as_of_utc)]
    if tur.empty:
        state.scada_info = {"fresh": False, "last_obs_data_clock": None, "age_h": None}
        return _ok("No SCADA observations before the issue time: weather-only forecast.", **state.scada_info)
    last_t = tur["target_time_utc"].max()
    age_h = float((state.as_of_utc - (last_t + pd.Timedelta(hours=1))) / pd.Timedelta(hours=1))
    week = tur[tur["target_time_utc"] > last_t - pd.Timedelta(days=7)]
    per = {t: {"hours_last_7d": int((week["entity"] == t).sum()),
               "availability_last_7d": round(float(week.loc[week["entity"] == t, "available"].mean()), 3)
               if (week["entity"] == t).any() else None} for t in state.site.turbine_ids}
    fresh = age_h <= forecast.FRESH_SCADA_MAX_AGE_H
    state.scada_info = {"fresh": fresh, "last_obs_data_clock": state.dc(last_t), "age_h": round(age_h, 1), "turbines": per}
    use = "used for the intraday blend" if fresh else "too old for the intraday blend: weather-only forecast"
    return _ok(f"Last SCADA hour {state.dc(last_t)} (data clock), age {age_h:.0f} h, {use}.", **state.scada_info)


def list_weather_runs(state: RunState) -> dict:
    runs = []
    for m in config.WEATHER_MODELS:
        if m["id"] in state.excluded:
            continue
        lat = pd.Timedelta(hours=m["latency_h"])
        newest = (state.as_of_utc - lat).floor("6h")
        archived = state.as_of_utc >= pd.Timestamp(m["archive_from"], tz="UTC") + pd.Timedelta(days=2)
        runs.append({"model": m["id"], "label": m["label"], "latency_h": m["latency_h"],
                     "newest_usable_run_utc": iso_z(newest), "published_by_utc": iso_z(newest + lat), "archived": bool(archived)})
    usable = [r["model"] for r in runs if r["archived"]]
    return _ok(f"{len(usable)} weather models usable at {state.dc(state.as_of_utc)} (runs published by then only).",
               runs=runs, usable_models=usable)


def fetch_weather(state: RunState, models: list[str] | None = None) -> dict:
    wanted = [m for m in (models or _active_models(state)) if m not in state.excluded]
    df, source = weather.window_for_issue(state.site, state.issue_time_utc, state.horizon_h, wanted, state.offline)
    state.weather_raw = df
    state.weather_source.update(source)
    counts = df.groupby("model").size().to_dict()
    live = sorted(m for m, s in source.items() if s == "live")
    cache = sorted(m for m, s in source.items() if s == "cache")
    missing = sorted(m for m, s in source.items() if s == "missing")
    txt = f"Fetched archived forecasts: live={live or '-'}, cache={cache or '-'}"
    if missing:
        txt += f", missing={missing}"
    return _ok(txt + ".", source=source, rows=counts)


def validate_weather(state: RunState) -> dict:
    if state.weather_raw is None:
        return _err("No weather data yet: call fetch_weather first.")
    remaining = int((state.window_end_utc - state.as_of_utc) / pd.Timedelta(hours=1))
    sel = asof.select_asof(state.weather_raw[~state.weather_raw["model"].isin(list(state.excluded))], state.as_of_utc, remaining)
    need = remaining + 1
    per, flags = {}, []
    for m in _active_models(state):
        g = sel[sel["model"] == m]
        bad = int(((g["ws100"] < 0) | (g["ws100"] > 60) | (g["t2m"] < -60) | (g["t2m"] > 60)).sum())
        per[m] = {"coverage": round(len(g) / need, 3), "implausible": bad,
                  "ws100_mean": round(float(g["ws100"].mean()), 2) if len(g) else None,
                  "ws100_max": round(float(g["ws100"].max()), 2) if len(g) else None,
                  "newest_run_utc": iso_z(g["init_time_utc"].max()) if len(g) else None}
    piv = sel.pivot_table(index="valid_time_utc", columns="model", values="ws100")
    spread = float(piv.std(axis=1).mean()) if piv.shape[1] > 1 else None
    dev = {}
    if piv.shape[1] >= 3:
        for m in piv.columns:
            others = piv.drop(columns=m).median(axis=1)
            dev[m] = float((piv[m] - others).abs().mean())
    typical = float(np.median(list(dev.values()))) if dev else None
    outliers = [m for m, d in dev.items() if d > max(OUTLIER_MIN_MS, 2.5 * (typical or 0))]
    insufficient = [m for m, p in per.items() if p["coverage"] < 0.5]
    for m in insufficient:
        flags.append(f"{m}: only {per[m]['coverage']:.0%} of hours available")
    for m in outliers:
        flags.append(f"{m}: deviates {dev[m]:.1f} m/s on average from the other models (typical {typical:.1f})")
    for m, p in per.items():
        if p["implausible"]:
            flags.append(f"{m}: {p['implausible']} implausible values")
    keep = [m for m in per if m not in set(outliers) | set(insufficient)]
    recommend = sorted(set(outliers) | set(insufficient)) if len(keep) >= 2 else sorted(insufficient)
    state.validation = {"per_model": per, "ensemble_spread_ms": round(spread, 2) if spread is not None else None,
                        "deviation_ms": {k: round(v, 2) for k, v in dev.items()}, "flags": flags,
                        "recommend_exclude": recommend}
    txt = f"{len(per)} models checked, ensemble spread {spread:.1f} m/s" if spread is not None else f"{len(per)} model(s) checked"
    txt += f"; flags: {'; '.join(flags)}" if flags else "; no issues found"
    return _ok(txt + ".", **state.validation)


def run_forecast(state: RunState, exclude_models: list[str] | None = None) -> dict:
    exclude_models = list(exclude_models or [])
    unknown = [m for m in exclude_models if m not in weather.model_ids()]
    if unknown:
        return _err(f"Unknown models {unknown}. Valid: {weather.model_ids()}")
    remaining = [m for m in _active_models(state) if m not in exclude_models]
    if not remaining:
        return _err("Cannot exclude every weather model; keep at least one.")
    if state.weather_raw is None:
        return _err("No weather data yet: call fetch_weather first.")
    for m in exclude_models:
        reason = next((f for f in state.validation.get("flags", []) if f.startswith(m)), "excluded by the agent")
        state.excluded[m] = reason
    if state.model is None:
        state.model, state.model_note = forecast.model_for_issue(state.site, state.issue_time_utc)
    pred, sel, info = forecast.forecast_rows(state.site, state.model, state.weather_raw, state.issue_time_utc,
                                             state.as_of_utc, state.window_end_utc, exclude_models=list(state.excluded))
    version = len(state.versions) + 1
    state.current = {
        "version": version, "as_of_utc": state.as_of_utc, "pred": pred, "info": info,
        "forecast": forecast.to_schema_f(pred, state.site, state.issue_time_utc, version, state.as_of_utc),
        "weather": forecast.to_schema_w(sel, state.site, state.issue_time_utc, version, state.weather_source),
    }
    farm = pred[pred["entity"] == "farm"].sort_values("target_time_utc")
    peak = farm.loc[farm["mean"].idxmax()]
    data = {"version": version, "hours": int(len(farm)), "models_used": info["models_used"],
            "energy_norm_h": round(float(farm["mean"].sum()), 2), "capacity_factor": round(float(farm["mean"].mean()), 3),
            "peak": round(float(peak["mean"]), 3), "peak_time_data_clock": state.dc(peak["target_time_utc"]),
            "mean_band_width": round(float((farm["p90"] - farm["p10"]).mean()), 3),
            "nowcast": info["nowcast"], "model": state.model_note}
    return _ok(f"v{version}: {len(farm)} h, mean capacity factor {data['capacity_factor']:.2f}, peak {data['peak']:.2f} "
               f"at {data['peak_time_data_clock']}, P10–P90 width {data['mean_band_width']:.2f}, models {len(info['models_used'])}.", **data)


def analyze_forecast(state: RunState) -> dict:
    cur = state.current or (state.versions[-1] if state.versions else None)
    if cur is None:
        return _err("Nothing to analyze: call run_forecast first.")
    pred = cur["pred"]
    farm = pred[pred["entity"] == "farm"].sort_values("target_time_utc").reset_index(drop=True)
    d3 = farm["mean"].diff(3).abs()
    ramp_idx = farm.index[d3 > RAMP_THRESHOLD].tolist()
    ramps = [state.dc(farm.loc[i, "target_time_utc"]) for i in ramp_idx][:6]
    calm = int((farm["mean"] < 0.05).sum())
    width = farm["p90"] - farm["p10"]
    w = cur["weather"]
    high_wind = int(w.groupby("target_time_utc")["ws100"].max().gt(HIGH_WIND_MS).sum()) if len(w) else 0
    n_models = len(cur["info"]["models_used"])
    spread = state.validation.get("ensemble_spread_ms")
    # climatology of the same month-hours from SCADA history before the issue
    hist = scada.site_hourly_long(state.site)
    hist = hist[(hist["entity"] == "farm") & (hist["target_time_utc"] < state.issue_time_utc)]
    key = hist["target_time_utc"].dt.month * 100 + hist["target_time_utc"].dt.hour
    clim = hist.groupby(key)["actual"].mean()
    fkey = farm["target_time_utc"].dt.month * 100 + farm["target_time_utc"].dt.hour
    cf_clim = float(fkey.map(clim).mean()) if len(clim) else None
    # revision vs previous version in this run, or vs the previous issue
    revision = None
    if len(state.versions) >= 1 and cur is state.current:
        prev = state.versions[-1]["pred"]
        revision = _revision(prev, farm, "previous version")
    else:
        revision = _revision_prev_issue(state, farm)
    # NOTE: no actuals from the forecast window are shown to the agent (they do not exist at the
    # issue time). Post-hoc verification is done by the runner after publishing (run.json "errors").
    mw = float(width.mean())
    confidence = "high" if (mw < 0.35 and n_models >= 3 and (spread or 0) < 2.0) else \
                 "low" if (mw > 0.55 or n_models <= 2 or (spread or 0) > 3.5) else "medium"
    data = {"version": cur["version"], "capacity_factor": round(float(farm["mean"].mean()), 3),
            "climatology_capacity_factor": round(cf_clim, 3) if cf_clim is not None else None,
            "energy_norm_h": round(float(farm["mean"].sum()), 2), "max_ramp_3h": round(float(d3.max()), 3),
            "ramp_hours_data_clock": ramps, "calm_hours": calm, "high_wind_hours": high_wind,
            "mean_band_width": round(mw, 3), "wide_band_hours": int((width > 0.5).sum()), "n_models": n_models,
            "ensemble_spread_ms": spread, "revision": revision, "confidence": confidence,
            "excluded_models": dict(state.excluded)}
    state.analysis = data
    bits = [f"confidence {confidence}", f"capacity factor {data['capacity_factor']:.2f}"
            + (f" vs climatology {cf_clim:.2f}" if cf_clim is not None else ""),
            f"max 3 h ramp {data['max_ramp_3h']:.2f}"]
    if revision:
        bits.append(f"revision vs {revision['against']}: MAE {revision['mae']:.3f} over {revision['hours']} h")
    return _ok("; ".join(bits) + ".", **data)


def _revision(prev_pred: pd.DataFrame, farm: pd.DataFrame, label: str) -> dict | None:
    p = prev_pred[prev_pred["entity"] == "farm"].set_index("target_time_utc")["mean"]
    c = farm.set_index("target_time_utc")["mean"]
    j = pd.concat([p.rename("prev"), c.rename("cur")], axis=1, join="inner").dropna()
    if j.empty:
        return None
    return {"against": label, "hours": int(len(j)), "mae": round(float((j["cur"] - j["prev"]).abs().mean()), 4),
            "mean_change": round(float((j["cur"] - j["prev"]).mean()), 4)}


def _revision_prev_issue(state: RunState, farm: pd.DataFrame) -> dict | None:
    prev_id = issue_id(state.issue_time_utc - pd.Timedelta(days=1), state.site.data_clock_utc_offset_h)
    path = config.outputs_dir() / state.site.site / state.kind / prev_id / "forecast.csv"
    if not path.exists():
        return None
    prev = pd.read_csv(path, encoding="utf-8")
    prev = prev[(prev["entity"] == "farm") & (prev["version"] == prev["version"].max())]
    prev["target_time_utc"] = pd.to_datetime(prev["target_time_utc"], utc=True)
    return _revision(prev.assign(entity="farm"), farm, f"previous issue {prev_id}")


def check_for_updates(state: RunState) -> dict:
    if not state.versions:
        return _err("Nothing published yet.")
    last_w = state.versions[-1]["weather"]
    newest_used = last_w.groupby("model")["init_time_utc"].max().to_dict()
    updates = []
    for m in config.WEATHER_MODELS:
        if m["id"] in state.excluded or m["id"] not in newest_used:
            continue
        newest_now = (state.as_of_utc - pd.Timedelta(hours=m["latency_h"])).floor("6h")
        if newest_now > pd.Timestamp(newest_used[m["id"]]):
            updates.append({"model": m["id"], "published_run_utc": iso_z(newest_now), "used_run_utc": newest_used[m["id"]]})
    has = bool(updates)
    state.pending_reason = ("new runs: " + ", ".join(f"{u['model']} {u['published_run_utc']}" for u in updates)) if has else ""
    txt = f"At {state.dc(state.as_of_utc)}: " + (f"{len(updates)} model(s) published newer runs." if has else "no newer runs.")
    return _ok(txt, has_updates=has, updates=updates, as_of_data_clock=state.dc(state.as_of_utc))


def publish_forecast(state: RunState, summary: str = "") -> dict:
    if state.current is None:
        return _err("Nothing to publish: call run_forecast first.")
    v = state.current
    v["reason"] = "initial" if v["version"] == 1 else (state.pending_reason or "recomputed")
    v["summary_text"] = (summary or "").strip()[:4000]
    v["analysis"] = dict(state.analysis) if state.analysis.get("version") == v["version"] else {}
    state.versions.append(v)
    state.current = None
    return _ok(f"Published v{v['version']} ({len(v['forecast'])} rows).", version=v["version"])


TOOLS = {
    "inspect_scada": inspect_scada,
    "list_weather_runs": list_weather_runs,
    "fetch_weather": fetch_weather,
    "validate_weather": validate_weather,
    "run_forecast": run_forecast,
    "analyze_forecast": analyze_forecast,
    "check_for_updates": check_for_updates,
    "publish_forecast": publish_forecast,
}

_MODEL_ENUM = {"type": "array", "items": {"type": "string", "enum": [m["id"] for m in config.WEATHER_MODELS]}}
TOOL_SPECS = [
    {"name": "inspect_scada", "description": "Check the latest turbine SCADA data before the issue time: age, availability, whether it is fresh enough for the intraday blend.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "list_weather_runs", "description": "List weather models and the newest run of each that was already PUBLISHED at the issue time (as-of rule: run start + latency <= issue time).",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "fetch_weather", "description": "Download archived weather forecasts (Open-Meteo) for the site and forecast window. Falls back to the local cache if the API is unreachable.",
     "parameters": {"type": "object", "properties": {"models": _MODEL_ENUM}, "additionalProperties": False}},
    {"name": "validate_weather", "description": "Quality-check the as-of weather inputs: coverage, implausible values, ensemble spread, and outlier models that disagree strongly with the others. Returns recommend_exclude.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "run_forecast", "description": "Run the ML model (gradient boosting + NWP power curves) to produce hourly P10/P50/P90 per turbine and for the farm. Optionally exclude weather models.",
     "parameters": {"type": "object", "properties": {"exclude_models": _MODEL_ENUM}, "additionalProperties": False}},
    {"name": "analyze_forecast", "description": "Analyze the latest forecast: capacity factor vs climatology, ramps, calm and high-wind hours, uncertainty, revision vs the previous forecast, errors where actuals exist, confidence level.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "check_for_updates", "description": "Check whether newer weather runs have been published since the last published version (at the current simulated clock).",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "publish_forecast", "description": "Publish the latest computed forecast as a new version, with a short analysis for operators (in Russian, then one line in English).",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string", "description": "3-6 sentences in Russian + 1 line in English. Cite only numbers returned by tools."}},
                    "required": ["summary"], "additionalProperties": False}},
]


def call_tool(state: RunState, name: str, args: dict | None, tracer) -> dict:
    """Validated, traced tool execution. Unknown tools and bad arguments return an error result."""
    from .trace import Timer

    args = dict(args or {})
    if name not in TOOLS:
        res = _err(f"Unknown tool '{name}'. Available: {list(TOOLS)}")
        tracer.emit("error", res["summary"], tool=name, args=args, ok=False)
        return res
    spec = next(s for s in TOOL_SPECS if s["name"] == name)
    allowed = set(spec["parameters"]["properties"])
    extra = set(args) - allowed
    if extra:
        res = _err(f"Unexpected arguments {sorted(extra)} for {name}.")
        tracer.emit("error", res["summary"], tool=name, args=args, ok=False)
        return res
    tracer.emit("tool_call", f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})", tool=name, args=args)
    with Timer() as t:
        try:
            res = TOOLS[name](state, **args)
        except WindAgentError as exc:
            res = _err(exc.user_message_en)
        except AssertionError as exc:          # as-of guard violation: never continue silently
            res = _err(f"as-of guard: {exc}")
    tracer.emit("tool_result", res["summary"], tool=name, data=res.get("data"), duration_ms=t.ms, ok=res["ok"])
    return res
