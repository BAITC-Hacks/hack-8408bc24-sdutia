"""Presentation safeguards: missing hours and ambiguous revisions stay visible."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.analytics import (  # noqa: E402
    clean_submission, coverage_summary, daily_energy, revision_pairs,
    submission_summary, validation_period, validation_table,
)
from app.copy_analytics import tr  # noqa: E402


def local_day() -> pd.DataFrame:
    return pd.DataFrame({
        "target_time_utc": pd.date_range("2026-02-01T18:00:00Z", periods=24, freq="h"),
        "farm_mean": [0.5] * 24,
        "lead_h": range(24, 48),
    })


def revision(version: int, mean: float, issue_id: str = "2026-02-01_0000") -> dict:
    return {"site": "shelek", "issue_id": issue_id, "issue_time_utc": "2026-01-31T18:00:00Z",
            "target_time_utc": "2026-02-01T18:00:00Z", "lead_h": 24, "product": "day_ahead",
            "entity": "farm", "version": version, "mean": mean}


def test_daily_energy_uses_chosen_fixed_clock():
    energy, partial = daily_energy(local_day(), clock_offset=6)
    assert energy.to_dict("records") == [{"day": "2026-02-02", "energy_norm_h": 12.0}]
    assert partial == 0

    # The same 24 samples cover portions of two official UTC+5 calendar days.
    energy, partial = daily_energy(local_day(), clock_offset=5)
    assert energy.empty
    assert partial == 2


def test_missing_hour_does_not_look_like_complete_day_energy():
    energy, partial = daily_energy(local_day().drop(index=7), clock_offset=6)
    assert energy.empty
    assert partial == 1


def test_duplicate_submission_hours_are_all_excluded_without_changing_input():
    original = local_day()
    duplicate = pd.concat([original, original.iloc[[0]]], ignore_index=True)
    before = duplicate.copy(deep=True)

    cleaned, rejected = clean_submission(duplicate)

    assert rejected == 2
    assert len(cleaned) == 23
    pd.testing.assert_frame_equal(duplicate, before)
    energy, partial = daily_energy(duplicate, clock_offset=6)
    assert energy.empty
    assert partial == 1


def test_revision_delta_pairs_same_issue_and_entity():
    rows = [revision(1, 0.4), revision(2, 0.55), {**revision(2, 0.9), "entity": "t1"}]
    pairs, rejected = revision_pairs(pd.DataFrame(rows))

    assert len(pairs) == 1
    assert pairs.iloc[0]["delta"] == pytest.approx(0.15)
    assert rejected == 0


def test_revision_does_not_pair_different_issues():
    pairs, rejected = revision_pairs(pd.DataFrame([
        revision(1, 0.4, "2026-02-01_0000"), revision(2, 0.6, "2026-02-02_0000"),
    ]))
    assert pairs.empty
    assert rejected == 2


def test_duplicate_revision_is_not_arbitrarily_selected():
    pairs, rejected = revision_pairs(pd.DataFrame([revision(1, 0.4), revision(2, 0.5), revision(2, 0.6)]))
    assert pairs.empty
    assert rejected == 3


def test_two_issues_for_same_heatmap_cell_are_not_silently_averaged():
    pairs, rejected = revision_pairs(pd.DataFrame([
        revision(1, 0.4, "issue_a"), revision(2, 0.5, "issue_a"),
        revision(1, 0.2, "issue_b"), revision(2, 0.8, "issue_b"),
    ]))
    assert pairs.empty
    assert rejected == 4


def test_calibrated_coverage_keeps_heldout_and_full_period_scopes_separate():
    result = coverage_summary({
        "intervals": {"day_ahead": {"coverage_p10_p90": 0.716}},
        "interval_calibration": {"coverage_check_after": 0.799, "coverage_check_before": 0.713,
                                 "check_month": "2026-01", "fitted_on_months": ["2025-10", "2025-11", "2025-12"]},
    })
    assert result["calibrated"] == 0.799
    assert result["before_all"] == 0.716
    assert result["before_check"] == 0.713
    assert result["check_month"] == "2026-01"


def test_absent_calibration_is_not_relabelled_as_calibrated():
    result = coverage_summary({"intervals": {"day_ahead": {"coverage_p10_p90": 0.71}}})
    assert result["calibrated"] is None
    assert result["before_all"] == 0.71
    assert coverage_summary({"interval_calibration": {"coverage_check_after": 1.2}})["calibrated"] is None


def test_validation_period_uses_data_clock_month_boundary():
    metrics = {"validation": {"start_utc": "2025-09-30T18:00:00Z", "end_utc": "2026-01-31T17:00:00Z"}}
    assert validation_period(metrics, "EN") == "October 2025 – January 2026"
    assert validation_period(metrics, "RU") == "октябрь 2025 – январь 2026"
    assert validation_period({}, "RU") == "n/a"


def test_table_uses_percent_units_and_computes_skill_without_dividing_by_zero():
    metrics = {"by_product": {"intraday": {
        "model": {"mae": 0.2, "bias": -0.01}, "model_with_nowcast": {"mae": 0.15},
        "persistence": {"mae": 0.4}, "climatology": {"mae": 0},
    }}}
    table = validation_table(metrics, "EN")
    model = table.loc[table[tr("method", "EN")].eq(tr("model", "EN")) & table[tr("product", "EN")].eq(tr("intraday", "EN"))].iloc[0]
    assert model[tr("table_mae", "EN")] == "20.0"
    assert model[tr("table_bias", "EN")] == "-1.0"
    persistence = table.loc[table[tr("method", "EN")].eq(tr("persistence", "EN"))].iloc[-1]
    assert persistence[tr("table_skill", "EN")] == "50.0%"
    climatology = table.loc[table[tr("method", "EN")].eq(tr("climatology", "EN"))].iloc[-1]
    assert climatology[tr("table_skill", "EN")] == "n/a"
    nowcast = table.loc[table[tr("method", "EN")].eq(tr("model_with_nowcast", "EN"))].iloc[0]
    assert nowcast[tr("table_mae", "EN")] == "15.0"
    assert metrics["by_product"]["intraday"]["model"]["mae"] == 0.2


def test_all_product_revision_grid_preserves_elapsed_hours_and_overlapping_issues():
    rows = []
    for day in (1, 2):
        issue = pd.Timestamp(f"2026-02-{day:02d}T00:00:00Z")
        for version in (1, 2):
            for lead in range(0 if version == 1 else 6, 48):
                rows.append({"site": "shelek", "issue_id": f"issue_{day}", "issue_time_utc": issue,
                             "version_as_of_utc": issue + pd.Timedelta(hours=0 if version == 1 else 6),
                             "target_time_utc": issue + pd.Timedelta(hours=lead), "lead_h": lead,
                             "product": "intraday" if lead < 24 else "day_ahead", "entity": "farm",
                             "version": version, "mean": 0.4 if version == 1 else 0.5})
    pairs, rejected = revision_pairs(pd.DataFrame(rows), product=None)
    assert len(pairs) == 84
    assert pairs.lead_h.min() == 6
    assert pairs.lead_h.max() == 47
    assert pairs.issue_id.nunique() == 2
    assert rejected == 0  # Expected elapsed hours are not mislabeled as bad data.


def test_submission_headline_counts_actual_api_rows_not_expected_month_size():
    frame = local_day()
    frame["issue_id"] = "2026-02-01_0000"
    summary = submission_summary(frame, "EN")
    assert summary == {"period": "February 2026", "issues": 1, "hours": 24}
    assert submission_summary(frame.drop(index=0), "EN")["hours"] == 23
    assert submission_summary(None, "RU")["hours"] == 0


@pytest.mark.parametrize("lang", ["RU", "EN"])
def test_plotly_hover_tokens_survive_translation(lang):
    result = tr("revision_hover", lang)
    assert "%{y}" in result and "%{x}" in result and "%{z:+.1%}" in result


@pytest.mark.parametrize("lang", ["RU", "EN"])
def test_analytics_panels_render_fixture_api_data_and_percent_axes(monkeypatch, lang):
    import json
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("WINDAGENT_OUTPUTS_DIR", str(root / "tests" / "fixtures" / "outputs"))
    code = f'''
from windagent import api
from app.components.analytics import render_validation, render_test_period, render_how_it_works
render_validation(api.load_metrics("shelek"), {lang!r})
render_test_period(api.load_submission("shelek"), api.load_all_issues("shelek"), {lang!r}, 6, "UTC+6")
render_how_it_works(api.system_info(), api.site_info("shelek"), {lang!r})
'''
    app = AppTest.from_string(code).run(timeout=20)
    assert not app.exception
    figures = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
    assert len(figures) == 5
    for index in (0, 1, 2):
        assert figures[index]["layout"]["yaxis"]["tickformat"] == ".0%"
        assert figures[index]["layout"]["legend"]["y"] < 0
    assert figures[-1]["data"][0]["colorbar"]["tickformat"] == ".0%"
    assert figures[-1]["data"][0]["x"] == list(range(48))


def test_run_locally_lists_reproduction_commands_without_executing_them():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string('from app.components.analytics import render_run_locally\nrender_run_locally("EN", key="test_local")').run()
    assert not app.exception
    commands = [element.value for element in app.code]
    assert commands == [
        "pip install -r requirements.txt", "python -m windagent verify",
        'python -m windagent forecast --issue "2026-01-31 00:00" --policy rules --offline -v',
        "streamlit run app/streamlit_app.py", "docker compose up --build",
    ]
    assert any("4 × PASS" in element.value for element in app.markdown)
