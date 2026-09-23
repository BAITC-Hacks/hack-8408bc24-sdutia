"""LLM policy over OpenAI-compatible chat completions with tool calling (OpenAI or NVIDIA NIM).

Guardrails: whitelisted tools with validated arguments, fixed issue time, step limit, timeouts;
forecast numbers come only from tools. Any LLM failure raises LLMUnavailable → rules fallback."""

from __future__ import annotations

import json
import os

from .tools import TOOL_SPECS, RunState, call_tool

MAX_STEPS = 14
_DEFAULTS = {
    "openai": {"key": "OPENAI_API_KEY", "model_var": "OPENAI_MODEL", "model": "gpt-5.4-mini", "base_url": None},
    "nvidia": {"key": "NVIDIA_API_KEY", "model_var": "NVIDIA_MODEL", "model": "meta/llama-3.3-70b-instruct",
               "base_url": "https://integrate.api.nvidia.com/v1"},
}


class LLMUnavailable(RuntimeError):
    pass


def resolve_provider(wanted: str | None = None) -> dict:
    """Which LLM would be used: explicit LLM_PROVIDER, else OpenAI, else NVIDIA, else none."""
    wanted = (wanted or os.environ.get("LLM_PROVIDER", "auto")).strip().lower()
    order = ["openai", "nvidia"] if wanted in ("auto", "") else [wanted] if wanted in _DEFAULTS else []
    for name in order:
        d = _DEFAULTS[name]
        if os.environ.get(d["key"], "").strip():
            base = os.environ.get("NVIDIA_BASE_URL", d["base_url"]) if name == "nvidia" else d["base_url"]
            return {"provider": name, "model": os.environ.get(d["model_var"], "").strip() or d["model"],
                    "configured": True, "base_url": base}
    return {"provider": "none", "model": None, "configured": False, "base_url": None}


def candidates() -> list[dict]:
    """All configured providers in fallback order (OpenAI → NVIDIA)."""
    if os.environ.get("LLM_PROVIDER", "auto").strip().lower() == "none":
        return []
    out = []
    for name in ("openai", "nvidia"):
        r = resolve_provider(name)
        if r["configured"]:
            out.append(r)
    wanted = os.environ.get("LLM_PROVIDER", "auto").strip().lower()
    if wanted in _DEFAULTS:
        out = [r for r in out if r["provider"] == wanted]
    return out


def _client(p: dict):
    from openai import OpenAI

    return OpenAI(api_key=os.environ[_DEFAULTS[p["provider"]]["key"]], base_url=p["base_url"], timeout=60, max_retries=1)


SYSTEM_PROMPT = """You are WindAgent, an autonomous forecasting agent for a wind farm (site: {site_name}, turbines: {turbines}).
Goal: publish an hourly generation forecast for the {horizon} hours after the issue time {issue_dc} (data clock, UTC+{off}),
using ONLY the tools. The issue time is fixed; you cannot see or request data published after it.

Protocol:
1. inspect_scada, 2. list_weather_runs, 3. fetch_weather,
4. validate_weather: if it flags outlier or insufficient models, decide whether to exclude them (keep at least 2 models) and say why,
5. run_forecast (with your exclusions), 6. analyze_forecast: if confidence is low and a model was flagged, you may re-run without it,
7. publish_forecast with a summary for grid/farm operators: 3-6 sentences in Russian (expected output profile, ramps and their
   times, uncertainty, data issues, the decisions you made), then one line in English.
Rules: cite only numbers returned by tools; never invent values; times you mention must be data-clock times from tool results;
keep tool calls purposeful; stop after publishing."""


def _compact(res: dict, limit: int = 3500) -> str:
    s = json.dumps({"ok": res.get("ok"), "summary": res.get("summary"), "data": res.get("data")}, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…(truncated)"


class LLMPolicy:
    name = "llm"

    def __init__(self, provider: dict):
        self.p = provider
        self.provider, self.model = provider["provider"], provider["model"]
        self.client = _client(provider)
        self._temperature_ok = not str(self.model).startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
        self.messages: list[dict] = []
        self.tools = [{"type": "function", "function": s} for s in TOOL_SPECS]

    def _loop(self, state: RunState, tracer, stop_on_publish: bool = True) -> bool:
        published = False
        nudged = False
        for _ in range(MAX_STEPS):
            kwargs = dict(model=self.model, messages=self.messages, tools=self.tools, tool_choice="auto")
            if self._temperature_ok:
                kwargs["temperature"] = 0.2
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - any provider error → fallback
                if self._temperature_ok and "temperature" in str(exc).lower():
                    self._temperature_ok = False          # reasoning models accept only the default temperature
                    try:
                        resp = self.client.chat.completions.create(**{k: v for k, v in kwargs.items() if k != "temperature"})
                    except Exception as exc2:  # noqa: BLE001
                        raise LLMUnavailable(f"{self.provider}/{self.model}: {type(exc2).__name__}: {str(exc2)[:200]}") from exc2
                else:
                    raise LLMUnavailable(f"{self.provider}/{self.model}: {type(exc).__name__}: {str(exc)[:200]}") from exc
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            entry = {"role": "assistant", "content": msg.content or ""}
            if calls:
                entry["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}} for c in calls]
            self.messages.append(entry)
            if msg.content:
                tracer.emit("llm_message", msg.content.strip()[:1500])
            if not calls:
                text = (msg.content or "").strip()
                if published or nudged or text.upper().startswith("KEEP"):
                    return published
                # the model answered in prose instead of acting: nudge once to act or to decide explicitly
                nudged = True
                self.messages.append({"role": "user", "content": (
                    "You replied without calling any tool. If you decided to (re)compute, call the tools now "
                    "(validate_weather → run_forecast → analyze_forecast → publish_forecast). If you decided to keep the "
                    "current version, reply with KEEP followed by your reason.")})
                continue
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                    res = call_tool(state, c.function.name, args, tracer)
                except (json.JSONDecodeError, ValueError) as exc:
                    res = {"ok": False, "summary": f"Invalid arguments: {exc}", "data": None}
                    tracer.emit("error", res["summary"], tool=c.function.name, ok=False)
                self.messages.append({"role": "tool", "tool_call_id": c.id, "content": _compact(res)})
                if c.function.name == "publish_forecast" and res.get("ok"):
                    published = True
            if published and stop_on_publish:
                return True
        tracer.emit("warning", f"LLM reached the step limit ({MAX_STEPS}).")
        return published

    def initial(self, state: RunState, tracer) -> None:
        s = state.site
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(site_name=s.name, turbines=", ".join(s.turbine_ids),
                                                              horizon=state.horizon_h, issue_dc=state.dc(state.issue_time_utc),
                                                              off=s.data_clock_utc_offset_h)},
            {"role": "user", "content": f"Issue time {state.dc(state.issue_time_utc)} (data clock). Produce and publish the forecast."},
        ]
        if not self._loop(state, tracer):
            raise LLMUnavailable("the LLM finished without publishing a forecast")

    def update(self, state: RunState, tracer) -> None:
        self.messages.append({"role": "user", "content": (
            f"The clock is now {state.dc(state.as_of_utc)} (data clock). Call check_for_updates: it re-fetches the weather and "
            "reports, per model, the share of remaining hours whose inputs would come from a newer run (the fresh data is already "
            "loaded). Decide: if the update is material, call validate_weather, run_forecast, analyze_forecast and publish a new "
            "version whose summary explains what changed and why you recomputed; if there are no newer runs, or the change is "
            "negligible, keep the current version and reply briefly with your reason, without publishing.")})
        self._loop(state, tracer)
