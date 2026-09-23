"""Fixtures preserve the revised issue clock, version clock and original horizon."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from app.components.forecast import csv_bytes, forecast_figure  # noqa: E402
from app.i18n import tr  # noqa: E402
from windagent import api  # noqa: E402

FORECAST_COLUMNS = [
    "site", "issue_id", "version", "issue_time_utc", "version_as_of_utc", "target_time_utc",
    "target_time_data_clock", "target_time_kz_official", "lead_h", "product", "entity",
    "mean", "p10", "p50", "p90", "mw_mean",
]


@pytest.fixture(params=["2026-02-01_0000", "2026-02-02_0000"])
def fixture_run(request, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setenv("WINDAGENT_OUTPUTS_DIR", str(REPO / "tests" / "fixtures" / "outputs"))
    monkeypatch.setenv("WINDAGENT_OFFLINE", "1")
    return api.load_run("shelek", request.param)


def test_fixture_versions_keep_original_issue_horizon_and_lead_semantics(fixture_run: dict):
    frame, meta = fixture_run["forecast"], fixture_run["meta"]
    assert frame.columns.tolist() == FORECAST_COLUMNS
    assert meta["synthetic_fixture"] is True
    assert set(frame["entity"]) == {"farm", "t1", "t2"}
    assert frame["issue_id"].eq(meta["issue_id"]).all()
    original_issue = pd.Timestamp(meta["issue_time_utc"])
    assert frame["issue_time_utc"].eq(original_issue).all()
    version_times = {item["version"]: pd.Timestamp(item["as_of_utc"]) for item in meta["versions"]}

    for version, expected_rows, expected_first_lead in ((1, 48, 0), (2, 42, 6)):
        selected = frame.loc[frame["version"].eq(version)]
        assert len(selected) == expected_rows * 3
        assert selected.groupby("entity").size().to_dict() == {"farm": expected_rows, "t1": expected_rows, "t2": expected_rows}
        assert selected["version_as_of_utc"].eq(version_times[version]).all()
        assert selected["target_time_utc"].ge(selected["version_as_of_utc"]).all()
        assert selected["target_time_utc"].max() == original_issue + pd.Timedelta(hours=47)
        assert sorted(selected["lead_h"].unique()) == list(range(expected_first_lead, 48))
        measured_lead = (selected["target_time_utc"] - selected["issue_time_utc"]) / pd.Timedelta(hours=1)
        assert measured_lead.eq(selected["lead_h"]).all()
        assert selected.loc[selected["lead_h"].lt(24), "product"].eq("intraday").all()
        assert selected.loc[selected["lead_h"].ge(24), "product"].eq("day_ahead").all()
    assert version_times[1] == original_issue
    assert version_times[2] == original_issue + pd.Timedelta(hours=6)


def test_second_version_chart_only_contains_remaining_hours(fixture_run: dict):
    selected = fixture_run["forecast"].loc[fixture_run["forecast"]["version"].eq(2)]

    chart = forecast_figure(selected, pd.DataFrame(), "EN", 0, "UTC")
    farm_trace = next(trace for trace in chart.data if trace.name == tr("farm", "EN"))

    assert len(farm_trace.x) == 42
    assert pd.Timestamp(farm_trace.x[0], tz="UTC") == selected["version_as_of_utc"].iloc[0]
    assert pd.Timestamp(farm_trace.x[-1], tz="UTC") == selected["issue_time_utc"].iloc[0] + pd.Timedelta(hours=47)


def test_csv_export_preserves_version_as_of_with_iso_z_timestamps(fixture_run: dict):
    frame = fixture_run["forecast"]
    before = frame.copy(deep=True)

    exported = pd.read_csv(StringIO(csv_bytes(frame).decode("utf-8-sig")))

    assert exported.columns.tolist() == FORECAST_COLUMNS
    assert len(exported) == len(frame)
    for column in ("issue_time_utc", "version_as_of_utc", "target_time_utc"):
        assert exported[column].str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z").all()
        assert pd.to_datetime(exported[column], utc=True).eq(frame[column]).all()
    assert exported["lead_h"].eq(frame["lead_h"]).all()
    assert exported["issue_id"].eq(frame["issue_id"]).all()
    pd.testing.assert_frame_equal(frame, before)
