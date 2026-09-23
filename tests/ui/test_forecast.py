"""Regression checks that forecast plots never connect missing observations."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.forecast import forecast_figure, hourly_rows  # noqa: E402
from app.i18n import tr  # noqa: E402


def test_hourly_rows_inserts_missing_hour_without_interpolation_or_mutation():
    source = pd.DataFrame({
        "target_time_utc": ["2026-02-01T20:00:00Z", "2026-02-01T18:00:00Z"],
        "mean": [0.6, 0.2],
    })
    original = source.copy(deep=True)

    rows = hourly_rows(source)

    assert rows["target_time_utc"].tolist() == list(pd.date_range("2026-02-01T18:00:00Z", periods=3, freq="h"))
    assert rows.loc[0, "mean"] == 0.2
    assert pd.isna(rows.loc[1, "mean"])
    assert rows.loc[2, "mean"] == 0.6
    pd.testing.assert_frame_equal(source, original)


def test_hourly_rows_keeps_ambiguous_duplicate_hour_as_gap():
    rows = hourly_rows(pd.DataFrame({
        "target_time_utc": ["2026-02-01T18:00:00Z", "2026-02-01T19:00:00Z",
                            "2026-02-01T19:00:00Z", "2026-02-01T20:00:00Z", "bad timestamp"],
        "mean": [0.2, 0.3, 0.7, 0.6, 0.9],
    }))

    assert len(rows) == 3
    assert pd.isna(rows.loc[1, "mean"])
    assert rows["mean"].dropna().tolist() == [0.2, 0.6]


def test_forecast_chart_preserves_missing_and_unavailable_actual_gaps():
    hours = pd.date_range("2026-02-01T18:00:00Z", periods=4, freq="h")
    forecast = pd.DataFrame({"target_time_utc": hours, "entity": "farm", "mean": 0.4})
    actuals = pd.DataFrame({
        "target_time_utc": hours[[0, 1, 3]], "entity": "farm",
        "actual": [0.2, 0.99, 0.6], "available": [True, False, True],
    })

    figure = forecast_figure(forecast, actuals, "EN", 6, "UTC+6")
    trace = next(trace for trace in figure.data if trace.name == tr("actuals", "EN"))

    assert list(trace.x) == ["2026-02-02 00:00", "2026-02-02 01:00", "2026-02-02 02:00", "2026-02-02 03:00"]
    assert trace.y[0] == 0.2
    assert pd.isna(trace.y[1])  # A supplied value explicitly marked unavailable.
    assert pd.isna(trace.y[2])  # An hour absent from the API response altogether.
    assert trace.y[3] == 0.6
    assert trace.connectgaps is not True


def test_farm_and_turbine_forecasts_preserve_hourly_gaps():
    rows = [{"target_time_utc": timestamp, "entity": entity, "mean": value}
            for entity in ("farm", "t1")
            for timestamp, value in (("2026-02-01T18:00:00Z", 0.2), ("2026-02-01T20:00:00Z", 0.6))]

    figure = forecast_figure(pd.DataFrame(rows), pd.DataFrame(), "EN", 0, "UTC", show_turbines=True)

    assert {trace.name for trace in figure.data} == {tr("farm", "EN"), "t1"}
    for trace in figure.data:
        assert len(trace.x) == 3
        assert pd.isna(trace.y[1])
        assert trace.connectgaps is not True
