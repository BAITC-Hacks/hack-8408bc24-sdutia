"""Agent controls and readable live events over the public API."""
from __future__ import annotations

from collections import defaultdict, deque
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.forecast import render_kpis, render_revisions
from app.copy_agent import tool_title, tr
from app.presentation import human_reason, model_name, version_label, version_record
from app.rate_limit import reserve_policy
from app.services import ROOT, call_api, clear_loaders
from app.widgets import get_choice, option_selectbox, set_choice

ICONS = {"run_start": "▶", "plan": "◇", "decision": "◆", "llm_message": "✦",
         "warning": "⚠", "error": "✕", "publish": "✓", "run_end": "✓"}
WEATHER_IDS = ("ecmwf_aifs025_single", "ecmwf_ifs025", "icon_seamless", "gfs_seamless")
VERSION_HEADING = re.compile(r"^\s*(#{1,6})\s+(?:Версия|Version)\s+(\d+)\b", re.IGNORECASE)


def default_issue(low: datetime, high: datetime) -> datetime:
    """Use the first evaluation issue, constrained to the API's site bounds."""
    preferred = datetime(2026, 1, 31, tzinfo=low.tzinfo)
    return min(max(preferred, low), high)


def coalesce_events(events: list[dict]) -> list[dict]:
    """Pair calls/results by run and tool without merging repeats or losing errors."""
    rows: list[dict] = []
    pending: dict[tuple, deque] = defaultdict(deque)
    step = 0
    for raw in events:
        if not isinstance(raw, dict):
            continue
        event = dict(raw)
        kind, tool = event.get("type"), event.get("tool")
        key = (event.get("run_id"), tool)
        if kind == "tool_result" and pending[key]:
            row = rows[pending[key].popleft()]
            row["events"].append(event)
            row.update(summary=event.get("summary"), duration_ms=event.get("duration_ms"),
                       ok=event.get("ok"), pending=False)
        elif kind in {"tool_call", "tool_result"}:
            step += 1
            rows.append({**event, "type": "tool", "number": step,
                         "pending": kind == "tool_call", "events": [event]})
            if kind == "tool_call":
                pending[key].append(len(rows) - 1)
        else:
            rows.append({**event, "events": [event]})
    return rows


def no_key_warning(message: object) -> bool:
    value = str(message).lower()
    return ("no llm key" in value or "no api key" in value or
            "llm-ключ не" in value or "llm ключ не" in value)


def readable_models(text: object, lang: str) -> str:
    value = str(text)
    for identifier in WEATHER_IDS:
        value = value.replace(identifier, model_name(identifier, lang))
    return value


def soften_markdown(text: str, lang: str) -> str:
    """Keep original facts while preventing backend headings from taking over UI."""
    text = readable_models(text, lang)
    text = re.sub(r"(?i)причина:\s*initial\b", "первичный прогноз", text)
    text = re.sub(r"(?i)reason:\s*initial\b", "initial forecast", text)
    output, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        if not fenced:
            line = re.sub(r"^(\s*)(#{1,6})(\s+)", lambda m: m[1] + "#" * min(6, len(m[2]) + 3) + m[3], line)
        output.append(line)
    return "\n".join(output).strip()


def split_analysis(text: str) -> tuple[str, dict[int, str], str]:
    """Separate version narratives from the issue intro and post-check notes."""
    intro, appendix = [], []
    sections: dict[int, list[str]] = {}
    current = None
    version_level = None
    in_appendix = False
    for line in str(text or "").splitlines():
        match = VERSION_HEADING.match(line)
        heading = re.match(r"^\s*(#{1,6})\s+", line)
        if match:
            current, in_appendix = int(match[2]), False
            version_level = len(match[1])
            sections.setdefault(current, [])
        elif current is not None and heading and len(heading[1]) <= version_level:
            current, in_appendix = None, True
            appendix.append(line)
        elif current is not None:
            sections[current].append(line)
        elif in_appendix:
            appendix.append(line)
        else:
            intro.append(line)
    return "\n".join(intro).strip(), {number: "\n".join(lines).strip() for number, lines in sections.items()}, "\n".join(appendix).strip()


def localized_analysis(text: str, lang: str) -> str:
    """Use an English summary already supplied by the agent, never invent one."""
    lines = []
    for line in text.splitlines():
        # These generated metric bullets are displayed once in the version chips.
        if re.match(r"^\s*[-*]\s+(?:КИУМ|CF):", line):
            continue
        # The version card already explains this reason using its API metadata.
        if re.match(r"^\s*(?:Почему|Why):", line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    english = re.search(r"(?<!\w)(?:English|_?EN):\s*(.+)", cleaned, flags=re.DOTALL)
    if english:
        return english[1].strip().strip("_") if lang == "EN" else cleaned[:english.start()].strip()
    english_lines = [line for line in cleaned.splitlines() if line.strip() and not re.search(r"[А-Яа-яЁё]", line)]
    russian_lines = [line for line in cleaned.splitlines() if re.search(r"[А-Яа-яЁё]", line)]
    if english_lines and russian_lines:
        return "\n".join(english_lines if lang == "EN" else russian_lines).strip()
    return cleaned


def _friendly_no_key(lang: str, tool_count: int = 0) -> None:
    st.info(tr("no_key_count", lang, count=tool_count) if tool_count else tr("no_key", lang))


def render_timeline(events: list[dict], lang: str) -> None:
    for row in coalesce_events(events):
        kind = row.get("type", "tool_result")
        summary = readable_models(row.get("summary") or "", lang)
        if no_key_warning(summary):
            _friendly_no_key(lang)
        elif kind == "tool":
            icon = "◌" if row["pending"] else ("✗" if row.get("ok") is False else "✓")
            label = f"{icon} {row['number']}. {tool_title(str(row.get('tool') or ''), lang)}"
            if row.get("duration_ms") is not None:
                label += " · " + tr("event_ms", lang, value=row["duration_ms"])
            st.markdown(f"**{label}**")
            if row["pending"]:
                st.caption(tr("waiting_result", lang))
                # Keep a semantic progress summary, but hide raw tool(args) lines.
                tool_call = rf"^\s*{re.escape(str(row.get('tool') or ''))}\s*\("
                if summary and not re.match(tool_call, summary):
                    st.caption(summary)
            elif summary:
                st.caption(summary)
        elif kind in {"decision", "llm_message"}:
            st.info(f"{ICONS[kind]} {tr('agent_decision', lang)}: {summary}")
        elif kind == "error":
            st.error(summary)
        elif kind == "warning":
            st.warning(summary)
        else:
            st.markdown(f"**{ICONS.get(kind, '·')} {tr('event_' + kind, lang)}**")
            if summary:
                st.caption(summary)
        with st.expander(tr("details", lang), expanded=False):
            st.json(row["events"], expanded=False)


def _official_clock() -> None:
    set_choice("display_clock", "official")


def _set_issue(value: datetime) -> None:
    st.session_state["agent_issue_date"] = value.date()
    st.session_state["agent_issue_time"] = value.time()


def _issue_date_label(meta: dict, data_offset: int) -> str:
    try:
        issued = pd.Timestamp(meta.get("issue_time_utc"))
        if pd.isna(issued) or issued.tzinfo is None:
            raise ValueError("Missing aware issue timestamp")
        return issued.tz_convert(timezone(timedelta(hours=data_offset))).strftime("%d.%m.%Y")
    except (ValueError, TypeError):
        return str(meta.get("issue_id", "n/a")).split("_")[0]


def _render_narrative(text: str, lang: str) -> None:
    localized = localized_analysis(text, lang)
    if lang == "EN" and re.search(r"[А-Яа-яЁё]", localized):
        st.caption(tr("original_language", lang))
    st.markdown(soften_markdown(localized, lang), unsafe_allow_html=False)


def render_analysis(result: dict, lang: str, data_offset: int) -> None:
    meta = result.get("meta") or {}
    raw_analysis = str(result.get("analysis") or "")
    intro, narratives, appendix = split_analysis(raw_analysis)
    versions = [item.get("version") for item in meta.get("versions", []) if isinstance(item, dict) and item.get("version") is not None]
    st.subheader(tr("analysis", lang))
    if intro and not narratives:
        st.caption(tr("general_analysis", lang))
        _render_narrative(intro, lang)
    if not versions:
        if not raw_analysis:
            st.info(tr("no_analysis", lang))
        elif narratives:
            _render_narrative(raw_analysis, lang)
    for version in versions:
        record = version_record(meta, version)
        with st.container(border=True):
            st.markdown(f"#### {version_label(meta, version, lang, data_offset)}")
            if version != 1 and record.get("reason"):
                st.caption(tr("why_updated", lang, reason=human_reason(record["reason"], lang)))
            text = narratives.get(version, "")
            if text:
                _render_narrative(text, lang)
            else:
                st.caption(tr("no_analysis", lang))
            render_kpis(meta, lang, version=version)
    if appendix:
        with st.expander(tr("additional_analysis", lang)):
            _render_narrative(appendix, lang)
    if raw_analysis:
        with st.expander(tr("original_analysis", lang)):
            st.code(raw_analysis, language="markdown", wrap_lines=True)


def render_agent(site: str, info: dict, system: dict, saved: dict | None, lang: str,
                 offset: int, clock_label: str) -> None:
    st.subheader(tr("agent_heading", lang))
    st.caption(tr("agent_note", lang))
    data_offset = int(info.get("data_clock_utc_offset_h", 6))
    data_tz = timezone(timedelta(hours=data_offset))
    bounds = info.get("issue_range_data_clock") or {"min": "2026-01-31 00:00", "max": "2026-02-28 23:59"}
    try:
        low = datetime.fromisoformat(bounds["min"]).replace(tzinfo=data_tz)
        high = datetime.fromisoformat(bounds["max"]).replace(tzinfo=data_tz)
        if high < low:
            raise ValueError("Inverted date range")
    except (ValueError, KeyError, TypeError):
        st.error(tr("invalid_time", lang))
        return
    if st.session_state.get("agent_site") != site:
        st.session_state["agent_site"] = site
        _set_issue(default_issue(low, high))
    if "agent_issue_date" not in st.session_state or not low.date() <= st.session_state["agent_issue_date"] <= high.date():
        _set_issue(default_issue(low, high))

    quick = st.columns([2, 1, 1])
    for column, key, day in ((quick[0], "quick_first", datetime(2026, 1, 31, tzinfo=data_tz)),
                             (quick[1], "quick_february", datetime(2026, 2, 10, tzinfo=data_tz))):
        column.button(tr(key, lang), key=key, disabled=not low <= day <= high,
                      on_click=_set_issue, args=(day,), width="stretch")
    live = quick[2].button(tr("forecast_now", lang), key="forecast_now", on_click=_official_clock, width="stretch")
    st.caption(tr("quick_note", lang))
    cols = st.columns([2, 1, 2])
    selected_date = cols[0].date_input(tr("issue_date", lang), min_value=low.date(), max_value=high.date(), key="agent_issue_date")
    selected_time = cols[1].time_input(tr("issue_time", lang), step=3600, key="agent_issue_time")
    policies = [p for p in system.get("policies", ["auto", "llm", "rules"]) if p in {"auto", "llm", "rules"}]
    policy = option_selectbox(tr("policy", lang), policies or ["rules"], key="agent_policy", format_func=lambda p: tr("policy_" + p, lang), ui=cols[2])
    st.caption(tr("bounds", lang, start=bounds["min"], end=bounds["max"]))
    historical = st.button(tr("run_agent", lang), key="run_agent", type="primary")
    st.caption(tr("rate_note", lang))
    provider = call_api("llm_status", lang=lang, default={}, quiet=True) or {}
    if policy in {"auto", "llm"} and provider.get("configured") is False:
        _friendly_no_key(lang, len(system.get("tools") or []))

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
                timeline = st.empty()
                def callback(event):
                    events.append(dict(event))
                    with timeline.container():
                        render_timeline(events, lang)
                if live:
                    result = call_api("forecast_now", site, effective, on_event=callback, lang=lang, cached=False)
                else:
                    result = call_api("run_agent", site, issue.strftime("%Y-%m-%d %H:%M"), effective, on_event=callback, lang=lang, cached=False)
                if not events and result:
                    events = result.get("trace") or []
                    with timeline.container():
                        render_timeline(events, lang)
                status.update(label=tr("complete" if result else "failed", lang), state="complete" if result else "error", expanded=True)
            executed = True
            st.session_state[events_key] = events
            st.session_state[state_key] = result
            st.session_state[f"agent_live_{site}"] = live
            clear_loaders()
    own_run = state_key in st.session_state
    result = st.session_state.get(state_key) if own_run else saved
    if not own_run:
        st.info(tr("analysis_pending", lang))
    if result:
        label = "your_result" if own_run else "saved_example"
        st.caption(tr(label, lang, date=_issue_date_label(result.get("meta") or {}, data_offset)))
    if not executed:
        events = st.session_state.get(events_key, (result or {}).get("trace") or [])
        if events:
            with st.expander(tr("timeline", lang), expanded=False):
                render_timeline(events, lang)
        else:
            st.info(tr("no_trace", lang))
    if not result:
        return
    meta = result.get("meta") or {}
    if meta.get("synthetic_fixture"):
        st.warning(tr("synthetic", lang))
    if st.session_state.get(f"agent_live_{site}") and get_choice("display_clock") == "official":
        st.caption(tr("live_clock", lang))
    render_analysis(result, lang, data_offset)
    for warning in meta.get("warnings") or []:
        if no_key_warning(warning):
            _friendly_no_key(lang, len(system.get("tools") or []))
        else:
            st.warning(readable_models(warning, lang))
    if meta.get("excluded_models"):
        st.caption(tr("excluded", lang))
        for item in meta["excluded_models"]:
            if isinstance(item, dict):
                st.write(f"{model_name(item.get('model', ''), lang)}: {human_reason(item.get('reason', ''), lang)}")
            else:
                st.write(readable_models(item, lang))
    st.subheader(tr("revisions", lang))
    render_revisions(result, lang, offset, clock_label)
