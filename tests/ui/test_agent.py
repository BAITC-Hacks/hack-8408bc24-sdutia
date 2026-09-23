"""Pure presentation regressions for live steps and versioned agent prose."""
from datetime import datetime, timedelta, timezone

from app.components.agent import (
    coalesce_events, default_issue, localized_analysis, no_key_warning,
    soften_markdown, split_analysis,
)


def event(kind, tool=None, run="one", **values):
    return {"type": kind, "tool": tool, "run_id": run, "summary": kind, **values}


def test_tool_result_updates_one_row_without_losing_interleaved_decision():
    call = event("tool_call", "fetch_weather", args={"models": ["example"]})
    decision = event("decision", summary="Keep the published run")
    result = event("tool_result", "fetch_weather", summary="4 models available", duration_ms=277, ok=True)
    rows = coalesce_events([call, decision, result])
    assert len(rows) == 2
    assert rows[0]["number"] == 1
    assert rows[0]["events"] == [call, result]
    assert rows[0]["summary"] == "4 models available"
    assert rows[0]["duration_ms"] == 277
    assert rows[0]["pending"] is False
    assert rows[1]["type"] == "decision"
    assert call["type"] == "tool_call", "UI grouping must not modify raw audit events"


def test_repeated_tools_and_other_runs_do_not_overwrite_each_other():
    events = [event("tool_call", "fetch_weather"), event("tool_call", "fetch_weather", run="two"),
              event("tool_result", "fetch_weather", run="two", summary="second run"),
              event("tool_result", "fetch_weather", summary="first run"),
              event("tool_call", "fetch_weather"), event("tool_result", "fetch_weather", ok=False, summary="failed")]
    rows = coalesce_events(events)
    assert [row["number"] for row in rows] == [1, 2, 3]
    assert [row["summary"] for row in rows] == ["first run", "second run", "failed"]
    assert rows[-1]["ok"] is False


def test_pending_and_orphan_result_remain_visible():
    rows = coalesce_events([event("tool_call", "run_forecast"), event("tool_result", "fetch_weather", ok=True)])
    assert len(rows) == 2
    assert rows[0]["pending"] is True
    assert rows[1]["pending"] is False


def test_analysis_parser_keeps_versions_and_postcheck_separate():
    raw = """# Прогноз выпуска
Intro
## Версия 1 (as-of 2026-01-31 00:00, причина: initial)
First facts: 0.364
### Detail
Keep this detail.
## Версия 2 (as-of 2026-01-31 06:00, причина: newer runs)
Second facts: 0.343
## Пост-проверка (только бэктест)
Actuals were unavailable at issue time.
"""
    intro, versions, appendix = split_analysis(raw)
    assert "Intro" in intro
    assert set(versions) == {1, 2}
    assert "0.364" in versions[1] and "0.343" not in versions[1]
    assert "Keep this detail" in versions[1]
    assert "0.343" in versions[2]
    assert "as-of" not in "".join(versions.values())
    assert "Actuals were unavailable" in appendix
    assert "Actuals" not in versions[2]


def test_analysis_uses_existing_bilingual_summary_without_inventing_translation():
    text = "Средняя загрузка 0.36. English: Expected capacity factor 0.36.\n\n- КИУМ: 0.36"
    assert localized_analysis(text, "EN") == "Expected capacity factor 0.36."
    assert localized_analysis(text, "RU") == "Средняя загрузка 0.36."
    assert localized_analysis("Только исходный текст.", "EN") == "Только исходный текст."
    assert localized_analysis("КИУМ 0.36.\n\n_EN: capacity factor 0.36._", "EN") == "capacity factor 0.36."


def test_current_core_fourth_level_version_headings_end_before_warnings():
    raw = """### Прогноз от 2026-01-31
#### Версия 1 — первичный прогноз (2026-01-31 00:00)
Первый расчёт.
#### Версия 2 — уточнение (2026-01-31 06:00)
Почему: новые прогнозы погоды
Уточнённый расчёт.
#### Предупреждения
Недоступна модель.
#### Пост-проверка по факту
MAE 0.2
"""
    _, versions, appendix = split_analysis(raw)
    assert set(versions) == {1, 2}
    assert "Недоступна модель" not in versions[2]
    assert "Недоступна модель" in appendix
    assert "MAE 0.2" in appendix
    assert localized_analysis(versions[2], "RU") == "Уточнённый расчёт."


def test_markdown_demotes_headings_and_labels_models_but_preserves_code():
    text = "# Heading\n## Subheading\n###### Small\necmwf_aifs025_single\nпричина: initial\n```python\n# comment\n```"
    result = soften_markdown(text, "RU")
    assert result.startswith("#### Heading\n##### Subheading\n###### Small")
    assert "ecmwf_aifs025_single" not in result
    assert "ECMWF AIFS" in result
    assert "первичный прогноз" in result
    assert "\n# comment\n" in result


def test_default_date_uses_first_issue_and_clamps_to_api_limits():
    tz = timezone(timedelta(hours=6))
    first = datetime(2026, 1, 31, tzinfo=tz)
    assert default_issue(datetime(2024, 6, 1, tzinfo=tz), datetime(2026, 2, 28, tzinfo=tz)) == first
    low = datetime(2026, 2, 3, 6, tzinfo=tz)
    assert default_issue(low, datetime(2026, 2, 28, tzinfo=tz)) == low
    high = datetime(2025, 12, 31, 18, tzinfo=tz)
    assert default_issue(datetime(2024, 6, 1, tzinfo=tz), high) == high


def test_only_missing_key_messages_receive_friendly_rule_mode_copy():
    assert no_key_warning("No LLM key configured (OPENAI_API_KEY / NVIDIA_API_KEY): using the rules policy.")
    assert not no_key_warning("Weather service failed; using cached inputs.")
