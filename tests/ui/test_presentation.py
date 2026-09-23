"""Presentation keeps dates, version evidence and KPI provenance unambiguous."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import presentation  # noqa: E402


@pytest.fixture
def meta():
    return {
        "issue_time_utc": "2026-01-31T18:00:00Z",
        "versions": [
            {"version": 2, "as_of_utc": "2026-02-01T00:00:00Z", "reason": "new runs: ecmwf_aifs025_single for 28% of remaining hours",
             "kpis": {"capacity_factor": 0.8, "energy_norm_h": 33.6}},
            {"version": 1, "as_of_utc": "2026-01-31T18:00:00Z", "reason": "initial",
             "kpis": {"capacity_factor": 0.2, "energy_norm_h": 9.6}},
        ],
        "kpis": {"capacity_factor": 0.99, "energy_norm_h": 47.52},
    }


def test_version_record_matches_identifier_not_list_position(meta):
    assert presentation.version_record(meta, 1)["reason"] == "initial"
    assert presentation.version_record(meta, 2)["kpis"]["capacity_factor"] == 0.8
    assert presentation.version_record(meta, 3) == {}


@pytest.mark.parametrize("lang, initial_word, update_word", [("RU", "первич", "уточнен"), ("EN", "initial", "update")])
def test_version_labels_show_actual_data_clock_and_purpose(meta, lang, initial_word, update_word):
    initial = presentation.version_label(meta, 1, lang)
    update = presentation.version_label(meta, 2, lang)

    assert "00:00" in initial and initial_word in initial.lower()
    assert "06:00" in update and update_word in update.lower()
    assert "23:00" in presentation.version_label(meta, 1, lang, data_offset=5)


def test_selected_version_kpis_never_use_another_version_or_issue_summary(meta):
    original = deepcopy(meta)
    assert presentation.version_kpis(meta, 1) == {"capacity_factor": 0.2, "energy_norm_h": 9.6}
    assert presentation.version_kpis(meta, 2) == {"capacity_factor": 0.8, "energy_norm_h": 33.6}
    assert presentation.version_kpis(meta, 3) == {}
    assert presentation.version_kpis(meta, None) == meta["kpis"]
    assert meta == original
    del meta["versions"][0]["kpis"]
    assert presentation.version_kpis(meta, 2) == {}


@pytest.mark.parametrize("lang", ["RU", "EN"])
def test_revision_reason_keeps_evidence_but_humanizes_model_name(meta, lang):
    rendered = presentation.human_reason(meta["versions"][0]["reason"], lang)
    assert "28%" in rendered
    assert "ECMWF AIFS" in rendered
    assert "ecmwf_aifs025_single" not in rendered


@pytest.mark.parametrize("model, display", [("ecmwf_aifs025_single", "ECMWF AIFS"), ("ecmwf_ifs025", "ECMWF IFS"),
                                           ("icon_seamless", "ICON"), ("gfs_seamless", "GFS")])
def test_weather_model_ids_have_readable_names(model, display):
    for lang in ("RU", "EN"):
        result = presentation.model_name(model, lang)
        assert display in result
        assert model not in result


def test_issue_label_identifies_data_date_and_forecast_month_boundary():
    issue = pd.Timestamp("2026-02-27T18:00:00Z")
    english = presentation.issue_label(issue, "EN")
    russian = presentation.issue_label(issue, "RU")

    assert "28 Feb 2026, 00:00" in english
    assert "1 Mar" in english
    assert "28 февраля 2026, 00:00" in russian
    assert "мар" in russian


def test_default_issue_prefers_february_first_over_newest_issue():
    ordered = pd.DataFrame({
        "issue_id": ["2026-02-28_0000", "2026-02-02_0000", "2026-02-01_0000", "2026-01-31_0000"],
        "issue_time_utc": pd.to_datetime(["2026-02-27T18:00:00Z", "2026-02-01T18:00:00Z", "2026-01-31T18:00:00Z", "2026-01-30T18:00:00Z"]),
    })
    info = {"test_period": {"start": "2026-02-01", "end": "2026-02-28"}, "data_clock_utc_offset_h": 6}

    assert presentation.default_issue(ordered, info) == "2026-02-01_0000"
    assert presentation.default_issue(ordered.drop(index=2), info) == "2026-02-02_0000"
