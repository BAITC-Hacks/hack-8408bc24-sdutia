"""Agent controls and live event display over the public API."""
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import streamlit as st
from app.components.forecast import render_revisions
from app.i18n import tr
from app.rate_limit import reserve_policy
from app.services import ROOT, call_api, clear_loaders

ICONS = {"run_start": "▶", "plan": "◇", "tool_call": "↗", "tool_result": "✓", "decision": "◆",
         "llm_message": "✦", "warning": "⚠", "error": "✕", "publish": "●", "run_end": "■"}

def render_event(event: dict, lang: str) -> None:
    kind = event.get("type", "tool_result")
    title = tr("event_" + kind, lang)
    tool = event.get("tool")
    label = f"{ICONS.get(kind, '·')} {title}" + (f" · {tool}" if tool else "")
    st.text(label)
    if event.get("summary"):
        st.text(str(event["summary"]))
    if event.get("duration_ms") is not None:
        st.caption(tr("event_ms", lang, value=event["duration_ms"]))

def _official_clock() -> None:
    st.session_state["display_clock"] = "official"

def render_agent(site: str, info: dict, system: dict, saved: dict | None, lang: str,
                 offset: int, clock_label: str) -> None:
    st.subheader(tr("agent_heading", lang))
    st.caption(tr("agent_note", lang))
    data_tz = timezone(timedelta(hours=int(info.get("data_clock_utc_offset_h", 6))))
    bounds = info.get("issue_range_data_clock") or {"min": "2026-01-31 00:00", "max": "2026-02-28 23:59"}
    try:
        low = datetime.fromisoformat(bounds["min"]).replace(tzinfo=data_tz)
        high = datetime.fromisoformat(bounds["max"]).replace(tzinfo=data_tz)
    except (ValueError, KeyError, TypeError):
        st.error(tr("invalid_time", lang))
        return
    if st.session_state.get("agent_site") != site:
        st.session_state["agent_site"] = site
        st.session_state.pop("agent_issue_date", None)
        st.session_state.pop("agent_issue_time", None)
    cols = st.columns([2, 1, 2])
    selected_date = cols[0].date_input(tr("issue_date", lang), value=low.date(), min_value=low.date(), max_value=high.date(), key="agent_issue_date")
    selected_time = cols[1].time_input(tr("issue_time", lang), value=low.timetz(), step=3600, key="agent_issue_time")
    policies = [p for p in system.get("policies", ["auto", "llm", "rules"]) if p in {"auto", "llm", "rules"}]
    policy = cols[2].selectbox(tr("policy", lang), policies or ["rules"], key="agent_policy", format_func=lambda p: tr("policy_" + p, lang))
    st.caption(tr("bounds", lang, start=bounds["min"], end=bounds["max"]))
    buttons = st.columns([1, 1, 2])
    historical = buttons[0].button(tr("run_agent", lang), key="run_agent", type="primary", width="stretch")
    live = buttons[1].button(tr("forecast_now", lang), key="forecast_now", on_click=_official_clock, width="stretch")
    st.caption(tr("rate_note", lang))
    state_key, events_key = f"agent_result_{site}", f"agent_events_{site}"
    executed = False
    if historical or live:
        issue = datetime.combine(selected_date, selected_time).replace(tzinfo=data_tz)
        if historical and not low <= issue <= high:
            st.error(tr("invalid_time", lang))
        else:
            effective, notice = reserve_policy(policy, st.session_state.get("llm_runs", 0), Path(os.environ.get("WINDAGENT_OUTPUTS_DIR", ROOT / "outputs")))
            if notice:
                st.info(tr(notice, lang))
            if effective in {"auto", "llm"}:
                st.session_state["llm_runs"] = st.session_state.get("llm_runs", 0) + 1
            events = []
            with st.status(tr("running", lang), expanded=True) as status:
                def callback(event):
                    events.append(dict(event))
                    render_event(event, lang)
                if live:
                    result = call_api("forecast_now", site, effective, on_event=callback, lang=lang, cached=False)
                else:
                    result = call_api("run_agent", site, issue.strftime("%Y-%m-%d %H:%M"), effective, on_event=callback, lang=lang, cached=False)
                status.update(label=tr("complete" if result else "failed", lang), state="complete" if result else "error", expanded=True)
            executed = True
            st.session_state[events_key] = events
            st.session_state[state_key] = result
            st.session_state[f"agent_live_{site}"] = live
            clear_loaders()
    result = st.session_state.get(state_key) if state_key in st.session_state else saved
    if not executed:
        events = st.session_state.get(events_key, (result or {}).get("trace") or [])
        if events:
            with st.expander(tr("timeline", lang), expanded=False):
                for event in events:
                    render_event(event, lang)
        else:
            st.info(tr("no_trace", lang))
    if not result:
        return
    meta = result.get("meta") or {}
    if meta.get("synthetic_fixture"):
        st.warning(tr("synthetic", lang))
    if st.session_state.get(f"agent_live_{site}") and st.session_state.get("display_clock") == "official":
        st.caption(tr("live_clock", lang))
    left, right = st.columns([3, 2])
    with left:
        st.subheader(tr("analysis", lang))
        if result.get("analysis"):
            st.markdown(result["analysis"], unsafe_allow_html=False)
        else:
            st.info(tr("no_analysis", lang))
    with right:
        for warning in meta.get("warnings") or []:
            st.warning(str(warning))
        if meta.get("excluded_models"):
            st.caption(tr("excluded", lang))
            st.dataframe(pd.DataFrame(meta["excluded_models"]), hide_index=True, width="stretch")
    st.subheader(tr("revisions", lang))
    render_revisions(result, lang, offset, clock_label)
