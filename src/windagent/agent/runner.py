"""Runs the agent for one issue (or a rolling backtest) and writes the outputs (docs/CONTRACTS.md §4)."""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd

from .. import asof, config, scada
from ..errors import InputError
from ..timeutil import clock_str, iso_z, now_utc, parse_clock_time
from . import llm as llm_mod
from .rules import RulesPolicy
from .tools import RunState
from .trace import Tracer

UPDATE_AFTER_H = 6          # simulated "later the same night": newer weather runs have been published
MAX_HORIZON_H = 72


def _choose_policy(policy: str, tracer_notes: list[str]):
    policy = (policy or "auto").strip().lower()
    if policy not in {"auto", "llm", "rules"}:
        raise InputError(f"Unknown policy '{policy}'. Use auto, llm or rules.",
                         f"Неизвестная политика '{policy}'. Используйте auto, llm или rules.")
    if policy == "rules":
        return [RulesPolicy()]
    cands = llm_mod.candidates()
    if not cands:
        if policy == "llm":
            tracer_notes.append("No LLM key configured (OPENAI_API_KEY / NVIDIA_API_KEY): using the rules policy.")
        return [RulesPolicy()]
    out = []
    for c in cands:
        try:
            out.append(llm_mod.LLMPolicy(c))
        except Exception as exc:  # noqa: BLE001
            tracer_notes.append(f"LLM provider {c['provider']} unavailable: {exc}")
    return out + [RulesPolicy()]


def _run_phase(phase: str, chain: list, state: RunState, tracer: Tracer) -> object:
    """Run a phase with the first policy in the chain; on LLM failure fall back to the next one."""
    for i, pol in enumerate(chain):
        tracer.switch_policy(pol.name, getattr(pol, "provider", "none"), getattr(pol, "model", None))
        try:
            getattr(pol, phase)(state, tracer)
            return pol
        except llm_mod.LLMUnavailable as exc:
            nxt = chain[i + 1] if i + 1 < len(chain) else None
            tracer.emit("warning", f"LLM failed ({exc}); falling back to {getattr(nxt, 'provider', None) or nxt.name}.", ok=False)
            state.current = None if phase == "initial" and not state.versions else state.current
    return chain[-1]


def _posthoc_errors(state: RunState) -> dict | None:
    """Verification against actuals AFTER the run (never shown to the agent). Backtest only."""
    if not state.versions:
        return None
    v1 = state.versions[0]["pred"]
    farm = v1[v1["entity"] == "farm"].set_index("target_time_utc")["mean"]
    obs = scada.site_hourly_long(state.site)
    obs = obs[obs["entity"] == "farm"].set_index("target_time_utc")["actual"]
    j = pd.concat([farm.rename("f"), obs.rename("a")], axis=1, join="inner").dropna()
    if j.empty:
        return None
    e = j["f"] - j["a"]
    return {"version": 1, "hours": int(len(j)), "mae": round(float(e.abs().mean()), 4),
            "rmse": round(float(np.sqrt((e ** 2).mean())), 4), "note": "post-hoc verification; not available to the agent"}


def _reason_ru(reason: str) -> str:
    from ..config import humanize_models
    if reason == "initial":
        return "первичный прогноз"
    r = humanize_models(reason).replace("newer runs for ", "вышли новые прогоны погоды: ")
    return r.replace(" of hours, run ", " часов, прогон ").replace("recomputed", "пересчитан")


def _analysis_md(state: RunState, errors: dict | None) -> str:
    lines = [f"### Прогноз от {state.dc(state.issue_time_utc)} · {state.site.name}",
             f"Время выпуска: {state.dc(state.issue_time_utc)} (время данных, UTC+{state.site.data_clock_utc_offset_h}) · "
             f"горизонт {state.horizon_h} ч", ""]
    for v in state.versions:
        if v["version"] == 1:
            lines.append(f"#### Версия 1 — первичный прогноз ({state.dc(v['as_of_utc'])})")
        else:
            lines.append(f"#### Версия {v['version']} — уточнение ({state.dc(v['as_of_utc'])})")
            lines.append(f"Почему: {_reason_ru(v['reason'])}")
            lines.append("")
        lines.append(v.get("summary_text") or "—")
        a = v.get("analysis") or {}
        if a:
            lines.append("")
            lines.append(f"- КИУМ: {a.get('capacity_factor')} (норма {a.get('climatology_capacity_factor')}); "
                         f"макс. рампа 3 ч: {a.get('max_ramp_3h')}; ширина P10–P90: {a.get('mean_band_width')}; "
                         f"уверенность: {a.get('confidence')}; моделей: {a.get('n_models')}")
        lines.append("")
    if state.warnings:
        lines += ["#### Предупреждения", *[f"- {w}" for w in state.warnings], ""]
    if errors:
        lines += ["#### Пост-проверка по факту (после публикации)",
                  f"По {errors['hours']} ч с фактом: MAE {errors['mae']}, RMSE {errors['rmse']}. "
                  "Эти данные не были доступны агенту в момент выпуска.", ""]
    return "\n".join(lines)


def _write_outputs(state: RunState, tracer: Tracer, policy_used, errors: dict | None) -> dict:
    out_dir = config.outputs_dir() / state.site.site / state.kind / state.issue_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fc = pd.concat([v["forecast"] for v in state.versions], ignore_index=True)
    wx = pd.concat([v["weather"] for v in state.versions], ignore_index=True)
    fc.to_csv(out_dir / "forecast.csv", index=False, encoding="utf-8")
    wx.to_csv(out_dir / "weather.csv", index=False, encoding="utf-8")
    last_a = state.versions[-1].get("analysis") or state.analysis or {}
    src = set(state.weather_source.values())
    meta = {
        "site": state.site.site, "issue_id": state.issue_id, "issue_time_utc": iso_z(state.issue_time_utc),
        "issue_time_data_clock": state.dc(state.issue_time_utc), "created_at_utc": iso_z(now_utc()),
        "policy": policy_used.name, "provider": getattr(policy_used, "provider", "none"),
        "model": getattr(policy_used, "model", None), "horizon_h": state.horizon_h,
        "versions": [{"version": v["version"], "as_of_utc": iso_z(v["as_of_utc"]), "reason": v["reason"],
                      "models_used": v["info"]["models_used"], "runs_used": v["info"]["runs_used"],
                      "kpis": {k: (v.get("analysis") or {}).get(k) for k in
                               ("energy_norm_h", "capacity_factor", "max_ramp_3h", "mean_band_width", "confidence")}}
                     for v in state.versions],
        "models_used": state.versions[-1]["info"]["models_used"],
        "excluded_models": [{"model": m, "reason": r} for m, r in state.excluded.items()],
        "runs_used": state.versions[0]["info"]["runs_used"],
        "weather_source": "mixed" if len(src) > 1 else (src.pop() if src else "cache"),
        "forecast_model": state.model_note,
        "nowcast": state.versions[0]["info"].get("nowcast"),
        "warnings": state.warnings,
        "kpis": {"energy_norm_h": last_a.get("energy_norm_h"), "capacity_factor": last_a.get("capacity_factor"),
                 "max_ramp_3h": last_a.get("max_ramp_3h"), "mean_band_width": last_a.get("mean_band_width"),
                 "confidence": last_a.get("confidence"),
                 "revision_mae_v2_vs_v1": (state.versions[-1].get("analysis") or {}).get("revision", {}).get("mae")
                 if len(state.versions) > 1 and (state.versions[-1].get("analysis") or {}).get("revision") else None},
        "actuals_available": errors is not None, "errors": errors,
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "analysis.md").write_text(_analysis_md(state, errors), encoding="utf-8")
    tracer.write(out_dir / "trace.jsonl")
    return meta


def run_issue(site: str, issue_time_data_clock: str, policy: str = "auto", on_event: Callable[[dict], None] | None = None,
              horizon_h: int = config.HORIZON_H, offline: bool | None = None, kind: str = "runs",
              update_after_h: int | None = UPDATE_AFTER_H) -> dict:
    s = config.get_site(site)
    if not isinstance(horizon_h, int) or not 1 <= horizon_h <= MAX_HORIZON_H:
        raise InputError(f"Horizon must be an integer from 1 to {MAX_HORIZON_H} hours, got {horizon_h}.",
                         f"Горизонт должен быть целым числом от 1 до {MAX_HORIZON_H} ч, получено {horizon_h}.")
    T = parse_clock_time(issue_time_data_clock, s.data_clock_utc_offset_h)
    if T.minute or T.second:
        raise InputError("Issue time must be on a full hour.", "Время выпуска должно быть на начало часа.")
    lo, hi = asof.issue_range_utc(s)
    if not lo <= T <= hi:
        raise InputError(
            f"Issue time must be between {clock_str(lo, s.data_clock_utc_offset_h)} and {clock_str(hi, s.data_clock_utc_offset_h)} "
            f"(data clock), got {clock_str(T, s.data_clock_utc_offset_h)}.",
            f"Время выпуска должно быть в диапазоне {clock_str(lo, s.data_clock_utc_offset_h)} — "
            f"{clock_str(hi, s.data_clock_utc_offset_h)} (время данных), получено {clock_str(T, s.data_clock_utc_offset_h)}.")
    offline = config.offline() if offline is None else offline
    notes: list[str] = []
    chain = _choose_policy(policy, notes)
    state = RunState(site=s, issue_time_utc=T, horizon_h=horizon_h, offline=offline, kind=kind)
    first = chain[0]
    tracer = Tracer(f"{s.site}/{state.issue_id}", first.name, getattr(first, "provider", "none"), getattr(first, "model", None), on_event)
    tracer.emit("run_start", f"Issue {state.issue_id}: forecast {horizon_h} h from {state.dc(T)} (data clock, "
                f"UTC+{s.data_clock_utc_offset_h}); policy {first.name}.",
                data={"issue_time_utc": iso_z(T), "horizon_h": horizon_h, "offline": offline})
    for n in notes:
        tracer.emit("warning", n)
        state.warnings.append(n)
    tracer.emit("plan", "inspect SCADA → list published weather runs → fetch → validate → forecast → analyze → publish v1 → "
                        f"re-check for newer runs after {update_after_h or 0} h → recompute if needed.")
    used = _run_phase("initial", chain, state, tracer)
    if not state.versions:                       # absolute guarantee that v1 exists
        tracer.emit("warning", "No version published by the policy; completing with the rules policy.")
        used = RulesPolicy()
        tracer.switch_policy("rules")
        used.initial(state, tracer)
    if not state.versions:                       # still nothing: report the first tool error cleanly
        from ..errors import ExternalServiceError
        first = state.errors[0] if state.errors else None
        reason = first.user_message_en if first else "unknown reason"
        cls = type(first) if first else ExternalServiceError
        raise cls(f"No forecast could be produced for {state.dc(T)}: {reason}",
                  f"Не удалось построить прогноз на {state.dc(T)}: {first.user_message_ru if first else reason}")
    if update_after_h:
        as_of = T + pd.Timedelta(hours=update_after_h)
        if as_of <= now_utc() and as_of < state.window_end_utc:
            state.as_of_utc = as_of
            tracer.emit("plan", f"Clock advances to {state.dc(as_of)}: checking for newer weather runs.")
            chain_u = [p for p in chain if p is used] + [RulesPolicy()] if used.name != "rules" else [RulesPolicy()]
            _run_phase("update", chain_u, state, tracer)
    errors = _posthoc_errors(state) if kind in ("runs", "adhoc") else None
    tracer.emit("run_end", f"Published {len(state.versions)} version(s).", data={"versions": len(state.versions)})
    _write_outputs(state, tracer, used, errors)
    from .. import api
    return api.load_run(s.site, state.issue_id, kind)


def run_now(site: str, policy: str = "auto", on_event: Callable[[dict], None] | None = None,
            horizon_h: int = config.HORIZON_H, offline: bool | None = None) -> dict:
    s = config.get_site(site)
    T = now_utc().floor("h")
    return run_issue(site, clock_str(T, s.data_clock_utc_offset_h), policy=policy, on_event=on_event,
                     horizon_h=horizon_h, offline=offline, kind="live", update_after_h=None)


def backtest(site: str = "shelek", start: str = "2026-01-31", end: str = "2026-02-28", policy: str = "rules",
             offline: bool | None = None, log=print) -> dict:
    s = config.get_site(site)
    try:
        lo, hi = pd.Timestamp(str(start)), pd.Timestamp(str(end))
    except (ValueError, TypeError) as exc:
        raise InputError(f"Invalid backtest dates '{start}' / '{end}': expected YYYY-MM-DD.",
                         f"Неверные даты бэктеста '{start}' / '{end}': ожидается ГГГГ-ММ-ДД.") from exc
    if lo > hi:
        raise InputError("--start must not be after --end.", "--start не может быть позже --end.")
    days = pd.date_range(lo.normalize(), hi.normalize(), freq="1D")
    for d in days:
        issue = f"{d:%Y-%m-%d} {config.ISSUE_HOUR_DATA_CLOCK:02d}:00"
        r = run_issue(site, issue, policy=policy, offline=offline)
        m = r["meta"]
        log(f"  {issue}  v{len(m['versions'])}  policy={m['policy']}/{m['provider']}  models={len(m['models_used'])}  "
            f"CF={m['kpis']['capacity_factor']}  source={m['weather_source']}")
    return build_test_period(s)


def build_test_period(s: config.Site) -> dict:
    runs_dir = config.outputs_dir() / s.site / "runs"
    tp = s.test_period or {"start": "2026-02-01", "end": "2026-02-28"}
    frames = []
    for p in sorted(runs_dir.glob("*/forecast.csv")):
        f = pd.read_csv(p, encoding="utf-8")
        frames.append(f)
    if not frames:
        raise InputError("No runs found to build the test-period files.", "Нет прогнозов для сборки файлов тестового периода.")
    all_issues = pd.concat(frames, ignore_index=True)
    all_issues = all_issues[all_issues["issue_id"].str.endswith(f"_{config.ISSUE_HOUR_DATA_CLOCK:02d}00")]
    day = all_issues["target_time_data_clock"].str[:10]
    in_tp = (day >= tp["start"]) & (day <= tp["end"])
    out = config.outputs_dir() / s.site / "test_period"
    out.mkdir(parents=True, exist_ok=True)
    first_issue_day = (pd.Timestamp(tp["start"]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    issue_day = all_issues["issue_id"].str[:10]
    keep = (issue_day >= first_issue_day) & (issue_day <= tp["end"])
    all_issues[keep].to_csv(out / "all_issues.csv", index=False, encoding="utf-8")
    da = all_issues[keep & in_tp & (all_issues["product"] == "day_ahead") & (all_issues["version"] == 1)]
    base = da[da["entity"] == "farm"][["target_time_data_clock", "target_time_utc", "target_time_kz_official", "issue_id",
                                        "issue_time_utc", "lead_h", "mean", "p10", "p50", "p90"]]
    base = base.rename(columns={"mean": "farm_mean", "p10": "farm_p10", "p50": "farm_p50", "p90": "farm_p90"})
    for t in s.turbine_ids:
        tt = da[da["entity"] == t][["target_time_utc", "mean"]].rename(columns={"mean": f"{t}_mean"})
        base = base.merge(tt, on="target_time_utc", how="left")
    models = {}
    for iid in base["issue_id"].unique():
        meta = json.loads((runs_dir / iid / "run.json").read_text(encoding="utf-8"))
        models[iid] = ";".join(meta.get("models_used", []))
    base["models_used"] = base["issue_id"].map(models)
    base = base.sort_values("target_time_utc").reset_index(drop=True)
    base.to_csv(out / "submission_day_ahead.csv", index=False, encoding="utf-8")
    expected = (pd.Timestamp(tp["end"]) - pd.Timestamp(tp["start"])).days * 24 + 24
    return {"submission_rows": int(len(base)), "expected_rows": int(expected), "all_issue_rows": int(keep.sum()),
            "path": str(out / "submission_day_ahead.csv")}
