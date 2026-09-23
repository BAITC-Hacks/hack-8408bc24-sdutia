"""Regression checks that forecast plots never connect missing observations."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components import forecast as forecast_module  # noqa: E402
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

    assert len(figure.data) == 2
    assert any(trace.name == tr("farm", "EN") for trace in figure.data)
    assert any("1" in trace.name for trace in figure.data if trace.name != tr("farm", "EN"))
    for trace in figure.data:
        assert len(trace.x) == 3
        assert pd.isna(trace.y[1])
        assert trace.connectgaps is not True


@pytest.mark.parametrize("selected_version", [1, 2])
def test_weather_selection_never_mixes_forecast_versions(selected_version: int):
    weather = pd.DataFrame({
        "version": [1, 2, 1, 2], "model": ["model_a", "model_a", "model_b", "model_b"],
        "target_time_utc": pd.to_datetime(["2026-02-01T00:00:00Z"] * 4),
        "ws100": [3.0, 8.0, 4.0, 9.0],
    })
    original = weather.copy(deep=True)

    selected = forecast_module.select_weather_for_version(weather, selected_version)

    pd.testing.assert_frame_equal(selected, weather.loc[weather["version"].eq(selected_version)])
    pd.testing.assert_frame_equal(weather, original)


def test_absent_weather_version_stays_empty_instead_of_using_other_version():
    weather = pd.DataFrame({"version": [1, 1], "model": ["model_a", "model_b"], "ws100": [3.0, 4.0]})

    selected = forecast_module.select_weather_for_version(weather, 2)

    assert selected.empty
    assert selected.columns.tolist() == weather.columns.tolist()


def test_legacy_weather_without_version_column_remains_available():
    legacy = pd.DataFrame({"model": ["model_a", "model_b"], "ws100": [3.0, 4.0]})

    selected = forecast_module.select_weather_for_version(legacy, 2)

    pd.testing.assert_frame_equal(selected, legacy)


def test_weather_missing_version_values_are_not_assigned_to_requested_version():
    weather = pd.DataFrame({"version": [1, None, 2], "ws100": [3.0, 99.0, 8.0]})

    selected = forecast_module.select_weather_for_version(weather, 2)

    assert selected["ws100"].tolist() == [8.0]


def test_percentage_axis_preserves_fractional_prediction_values():
    source = pd.DataFrame({
        "target_time_utc": pd.date_range("2026-02-01T18:00:00Z", periods=2, freq="h"),
        "entity": "farm", "mean": [0.25, 0.75],
    })

    figure = forecast_figure(source, pd.DataFrame(), "EN", 6, "UTC+6")
    farm = next(trace for trace in figure.data if trace.name == tr("farm", "EN"))

    assert list(farm.y) == [0.25, 0.75]
    assert figure.layout.yaxis.tickformat.endswith("%")
    assert figure.layout.legend.y < 0


@pytest.mark.parametrize("lang, time_column, mean_column, turbine_column, part_column", [
    ("EN", "Time", "Farm forecast, %", "Turbine 1, %", "Part"),
    ("RU", "Время", "Прогноз ВЭС, %", "Турбина 1, %", "Часть"),
])
def test_readable_hourly_table_uses_selected_clock_and_percentages_without_changing_raw_data(
    lang, time_column, mean_column, turbine_column, part_column,
):
    rows = [{"target_time_utc": "2026-01-31T18:00:00Z", "entity": entity, "mean": value,
             "p10": 0.12, "p90": 0.63, "product": "intraday", "issue_id": "2026-02-01_0000", "version": 1}
            for entity, value in (("farm", 0.375), ("t1", 0.35), ("t2", 0.4))]
    source = pd.DataFrame(rows)
    original = source.copy(deep=True)

    readable = forecast_module.readable_hourly(source, lang, 5)

    assert len(readable) == 1
    assert readable.iloc[0][time_column] == "2026-01-31 23:00"
    assert readable.iloc[0][mean_column] == pytest.approx(37.5)
    assert readable.iloc[0][turbine_column] == pytest.approx(35.0)
    assert readable.iloc[0]["P10–P90, %"] == "12–63"
    assert readable.iloc[0][part_column].lower() == ("today" if lang == "EN" else "сегодня")
    assert not {"issue_id", "version", "entity", "target_time_utc"}.intersection(readable.columns)
    pd.testing.assert_frame_equal(source, original)
