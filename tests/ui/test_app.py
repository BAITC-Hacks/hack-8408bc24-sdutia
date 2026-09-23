"""Offline UI acceptance tests against the public API and labelled fixtures.

The app never sees test data unless WINDAGENT_OUTPUTS_DIR explicitly selects it.
Agent doubles stay in this module; no production prediction is fabricated.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import socket
import sys
from pathlib import Path

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from windagent import api  # noqa: E402

APP = REPO / "app" / "streamlit_app.py"
FIXTURE_OUTPUTS = REPO / "tests" / "fixtures" / "outputs"


@pytest.fixture(autouse=True)
def isolated_offline_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Keep tests independent of credentials, SCADA files and external services."""
    for name in ("OPENAI_API_KEY", "NVIDIA_API_KEY", "OPENAI_MODEL", "NVIDIA_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("WINDAGENT_OFFLINE", "1")
    data_dir = tmp_path / "empty_data"
    data_dir.mkdir()
    monkeypatch.setenv("WINDAGENT_DATA_DIR", str(data_dir))
    monkeypatch.setattr(st, "secrets", {})

    def reject_network(*args, **kwargs):
        raise AssertionError("UI tests must not access the network")

    monkeypatch.setattr(requests.sessions.Session, "request", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def fixture_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Copy committed fixtures so a run cannot modify repository test data."""
    target = tmp_path / "outputs"
    shutil.copytree(FIXTURE_OUTPUTS, target)
    monkeypatch.setenv("WINDAGENT_OUTPUTS_DIR", str(target))
    return target


def start_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=15).run()
    assert not app.exception, [element.value for element in app.exception]
    return app


def choice(app: AppTest, key: str, widget_type: str = "selectbox"):
    return next(widget for widget in app.get(widget_type) if widget.key.startswith(key + "__"))


def visible_text(app: AppTest) -> str:
    """Read human-facing text without coupling tests to layout containers."""
    text = []
    for element_type in ("markdown", "text", "caption", "code", "info", "warning", "error", "success"):
        text.extend(str(element.value) for element in app.get(element_type))
    return "\n".join(text)


def test_fixtures_render_all_five_tabs(fixture_outputs: Path):
    app = start_app()

    labels = [tab.label for tab in app.tabs]
    assert len(labels) == 5
    for expected in ("Прогноз", "Агент", "Валидация", "Тестовый период", "Как это работает"):
        assert any(expected in label for label in labels), labels
    # Forecast, weather, validation and test-period charts must actually exist.
    assert len(app.get("plotly_chart")) >= 4
    assert len(app.dataframe) >= 2


def test_empty_outputs_keep_app_usable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    empty_outputs = tmp_path / "empty_outputs"
    empty_outputs.mkdir()
    monkeypatch.setenv("WINDAGENT_OUTPUTS_DIR", str(empty_outputs))

    app = start_app()

    assert len(app.tabs) == 5
    assert "python -m windagent all --offline" in visible_text(app)
    assert app.button(key="run_agent") is not None
    assert app.selectbox(key="language") is not None


def test_agent_streams_five_callback_events(fixture_outputs: Path, monkeypatch: pytest.MonkeyPatch):
    result = api.load_run("shelek", "2026-02-01_0000")
    events = [
        {
            "ts_utc": f"2026-01-31T18:00:0{index}Z",
            "run_id": "shelek/2026-02-01_0000",
            "seq": index + 1,
            "type": event_type,
            "policy": "rules",
            "provider": "none",
            "model": None,
            "tool": "fetch_weather" if event_type == "tool_call" else None,
            "args": {},
            "summary": f"TEST_ONLY_TIMELINE_EVENT_{index + 1}",
            "data": {},
            "duration_ms": 10 + index,
            "ok": True,
        }
        for index, event_type in enumerate(("run_start", "plan", "tool_call", "publish", "run_end"))
    ]
    calls = []

    def fake_run_agent(site, issue_time_data_clock, policy="auto", on_event=None, **kwargs):
        calls.append((site, issue_time_data_clock, policy))
        assert on_event is not None
        for event in events:
            on_event(event)
        return {**result, "trace": events}

    monkeypatch.setattr(api, "run_agent", fake_run_agent)
    app = start_app()
    app.date_input(key="agent_issue_date").set_value(dt.date(2026, 2, 1))
    app.time_input(key="agent_issue_time").set_value(dt.time(0, 0))
    choice(app, "agent_policy").set_value("rules")
    app.button(key="run_agent").click().run()

    assert not app.exception, [element.value for element in app.exception]
    assert calls == [("shelek", "2026-02-01 00:00", "rules")]
    rendered = visible_text(app)
    for event in events:
        assert event["summary"] in rendered
    assert len(app.status) >= 1


def test_english_toggle_translates_known_labels(fixture_outputs: Path):
    app = start_app()
    assert any("Прогноз" in tab.label for tab in app.tabs)

    app.selectbox(key="language").set_value("EN").run()

    assert not app.exception, [element.value for element in app.exception]
    labels = [tab.label for tab in app.tabs]
    assert "Forecast" in labels
    assert "Agent" in labels
    assert "Validation" in labels
    assert not any("Прогноз" in label for label in labels)


def test_live_forecast_defaults_to_official_clock_but_allows_manual_override(
    fixture_outputs: Path, monkeypatch: pytest.MonkeyPatch,
):
    result = api.load_run("shelek", "2026-02-01_0000")
    calls = []

    def fake_forecast_now(site, policy="auto", on_event=None):
        calls.append((site, policy))
        return result

    def revision_chart(app):
        element = next(chart for chart in app.get("plotly_chart") if chart.proto.id.endswith("-agent_revisions"))
        return json.loads(element.proto.spec)

    monkeypatch.setattr(api, "forecast_now", fake_forecast_now)
    app = start_app()
    app.selectbox(key="language").set_value("EN")
    choice(app, "agent_policy").set_value("rules")
    app.button(key="forecast_now").click().run()

    assert not app.exception, [element.value for element in app.exception]
    assert calls == [("shelek", "rules")]
    assert choice(app, "display_clock").value == "official"
    assert "official Kazakhstan time" in visible_text(app)
    assert revision_chart(app)["layout"]["xaxis"]["title"]["text"] == "Official Kazakhstan time (UTC+5)"

    for selected_clock, expected_axis in (("utc", "UTC"), ("data", "Data clock (UTC+6)")):
        choice(app, "display_clock").set_value(selected_clock).run()

        assert not app.exception, [element.value for element in app.exception]
        assert choice(app, "display_clock").value == selected_clock
        assert "The live forecast is shown in official Kazakhstan time." not in visible_text(app)
        assert revision_chart(app)["layout"]["xaxis"]["title"]["text"] == expected_axis
    assert calls == [("shelek", "rules")], "Changing a display clock must not launch another forecast"


def test_language_switch_and_stale_browser_labels_never_reach_api(
    fixture_outputs: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Exercise Streamlit's formatted-string wire protocol, including stale labels.

    Public AppTest set_value takes raw Python options and can miss the browser
    failure. _run lets this regression submit actual WidgetState string values.
    """
    seen_kinds, seen_issues, seen_policies = [], [], []
    original_list_runs, original_load_run = api.list_runs, api.load_run
    result = original_load_run("shelek", "2026-02-01_0000")

    def list_runs(site, kind="runs"):
        seen_kinds.append(kind)
        return original_list_runs(site, kind)

    def load_run(site, issue_id, kind="runs"):
        seen_issues.append(issue_id)
        return original_load_run(site, issue_id, kind)

    def run_agent(site, issue_time_data_clock, policy="auto", on_event=None, **kwargs):
        seen_policies.append(policy)
        return result

    monkeypatch.setattr(api, "list_runs", list_runs)
    monkeypatch.setattr(api, "load_run", load_run)
    monkeypatch.setattr(api, "run_agent", run_agent)
    app = start_app()
    choice(app, "display_clock").set_value("official")
    choice(app, "agent_policy").set_value("rules")
    choice(app, "issue_shelek_runs").set_value("2026-02-01_0000").run()
    old_keys = {key: choice(app, key, kind).key for key, kind in (
        ("display_clock", "selectbox"), ("agent_policy", "selectbox"),
        ("issue_shelek_runs", "selectbox"), ("run_kind", "radio"),
    )}

    app.selectbox(key="language").set_value("EN").run()

    assert not app.exception
    assert not app.error
    for key, kind in (("display_clock", "selectbox"), ("agent_policy", "selectbox"),
                      ("issue_shelek_runs", "selectbox"), ("run_kind", "radio")):
        assert choice(app, key, kind).key != old_keys[key]
    assert choice(app, "run_kind", "radio").options == ["Historical", "Live"]
    assert choice(app, "display_clock").value == "official"
    assert choice(app, "agent_policy").value == "rules"
    assert choice(app, "issue_shelek_runs").value == "2026-02-01_0000"

    wire = app._tree.get_widget_states()
    stale = {
        choice(app, "run_kind", "radio").id: "Исторические",
        choice(app, "display_clock").id: "Официальное время РК (UTC+5)",
        choice(app, "agent_policy").id: "Правила",
        choice(app, "issue_shelek_runs").id: "2026-01-31 23:00 · Официальное время РК (UTC+5)",
    }
    for widget in wire.widgets:
        if widget.id in stale:
            widget.string_value = stale[widget.id]
        elif widget.id == app.button(key="refresh").id:
            widget.trigger_value = True
    app._run(wire)

    assert not app.exception, [element.value for element in app.exception]
    assert not app.error, [element.value for element in app.error]
    assert choice(app, "run_kind", "radio").value == "runs"
    assert choice(app, "display_clock").value == "official"
    assert choice(app, "agent_policy").value == "rules"
    assert choice(app, "issue_shelek_runs").value == "2026-02-01_0000"
    app.button(key="run_agent").click().run()
    assert not app.exception
    assert not app.error
    assert seen_kinds and set(seen_kinds) == {"runs"}
    assert seen_issues and set(seen_issues) <= {"2026-02-01_0000", "2026-02-02_0000"}
    assert seen_policies == ["rules"]


def test_issue_option_label_updates_when_display_clock_changes(fixture_outputs: Path):
    app = start_app()
    original = choice(app, "issue_shelek_runs")
    original_value, original_key = original.value, original.key
    choice(app, "display_clock").set_value("utc").run()

    issue = choice(app, "issue_shelek_runs")
    assert not app.exception
    assert issue.value == original_value
    assert issue.key != original_key
    assert all(label.endswith(" · UTC") for label in issue.options)
