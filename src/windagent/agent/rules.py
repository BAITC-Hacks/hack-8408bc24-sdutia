"""Deterministic playbook policy (used when no LLM key is configured, or as the LLM fallback).
Same tools and same decisions structure as the LLM policy."""

from __future__ import annotations

from .tools import RunState, call_tool


def _fmt_share(x) -> str:
    return "н/д" if x is None else f"{x:.2f}"


def summary_ru(state: RunState) -> str:
    a = state.analysis or {}
    cur = state.current or {}
    info = cur.get("info", {})
    models = ", ".join(info.get("models_used", [])) or "н/д"
    parts = [
        f"Ожидаемый средний коэффициент использования мощности (КИУМ) ВЭС: {_fmt_share(a.get('capacity_factor'))} "
        f"(климатическая норма для этих часов: {_fmt_share(a.get('climatology_capacity_factor'))}).",
        f"Максимальный перепад мощности за 3 ч: {_fmt_share(a.get('max_ramp_3h'))}"
        + (f", крупные рампы около {', '.join(a['ramp_hours_data_clock'][:3])} (время данных)." if a.get("ramp_hours_data_clock") else "."),
        f"Средняя ширина интервала P10–P90: {_fmt_share(a.get('mean_band_width'))}; уверенность: "
        f"{ {'high': 'высокая', 'medium': 'средняя', 'low': 'низкая'}.get(a.get('confidence'), 'н/д') }.",
        f"Использованы модели погоды: {models}.",
    ]
    if state.excluded:
        parts.append("Исключены: " + "; ".join(f"{m} ({r})" for m, r in state.excluded.items()) + ".")
    rev = a.get("revision")
    if rev:
        parts.append(f"Пересмотр относительно {rev['against']}: средн. абс. изменение {rev['mae']:.3f} за {rev['hours']} ч.")
    parts.append(f"EN: capacity factor {_fmt_share(a.get('capacity_factor'))}, max 3 h ramp {_fmt_share(a.get('max_ramp_3h'))}, "
                 f"confidence {a.get('confidence', 'n/a')}.")
    return " ".join(parts)


class RulesPolicy:
    name, provider, model = "rules", "none", None

    def initial(self, state: RunState, tracer) -> None:
        call_tool(state, "inspect_scada", {}, tracer)
        call_tool(state, "list_weather_runs", {}, tracer)
        call_tool(state, "fetch_weather", {}, tracer)
        v = call_tool(state, "validate_weather", {}, tracer)
        excl = (v.get("data") or {}).get("recommend_exclude") or []
        if excl:
            tracer.emit("decision", f"Exclude {excl}: flagged by validation ({'; '.join(v['data']['flags'])}).")
        else:
            tracer.emit("decision", "Use all available weather models: validation found no outliers.")
        r = call_tool(state, "run_forecast", {"exclude_models": excl}, tracer)
        if not r["ok"] and excl:
            tracer.emit("decision", "Forecast failed with exclusions; retrying with all models.")
            state.excluded.clear()
            call_tool(state, "run_forecast", {"exclude_models": []}, tracer)
        a = call_tool(state, "analyze_forecast", {}, tracer)
        if (a.get("data") or {}).get("confidence") == "low":
            tracer.emit("decision", "Confidence is low: publishing with a wide uncertainty band and a warning.")
            state.warnings.append("low confidence: wide P10–P90 band or strong model disagreement")
        call_tool(state, "publish_forecast", {"summary": summary_ru(state)}, tracer)

    def update(self, state: RunState, tracer) -> None:
        u = call_tool(state, "check_for_updates", {}, tracer)
        if not (u.get("data") or {}).get("has_updates"):
            tracer.emit("decision", "No newer weather runs: the published forecast stays current.")
            return
        tracer.emit("decision", "Newer weather runs are available: recomputing the remaining hours of the window.")
        call_tool(state, "fetch_weather", {}, tracer)
        v = call_tool(state, "validate_weather", {}, tracer)
        excl = [m for m in ((v.get("data") or {}).get("recommend_exclude") or []) if m not in state.excluded]
        call_tool(state, "run_forecast", {"exclude_models": excl}, tracer)
        call_tool(state, "analyze_forecast", {}, tracer)
        call_tool(state, "publish_forecast", {"summary": summary_ru(state)}, tracer)
