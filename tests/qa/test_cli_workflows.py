"""Exercise public commands and inspect files, without importing windagent."""

import csv
import json
import shutil

import numpy as np
import pandas as pd


def assert_success(result):
    assert result.returncode == 0, result.stdout + result.stderr
    assert not result.stderr.strip(), result.stderr


def test_help_and_info_are_available_offline(cli):
    result = cli("--help")
    assert_success(result)
    assert all(name in result.stdout for name in ("forecast", "backtest", "evaluate"))
    result = cli("info")
    assert_success(result)
    assert "shelek" in result.stdout
    assert "t1" in result.stdout and "t2" in result.stdout
    assert not cli.network_log.exists()


def test_forecast_writes_adhoc_output_and_tracks_provenance(cli):
    original = cli.repo / "outputs/shelek/runs/2026-01-31_0000"
    protected = cli.outputs / "shelek/runs/2026-01-31_0000"
    shutil.copytree(original, protected)
    original_bytes = {p.name: p.read_bytes() for p in protected.iterdir() if p.is_file()}
    result = cli("forecast", "--issue", "2026-01-31 00:00", "-v")
    assert_success(result)
    assert "run_start" in result.stdout and "run_end" in result.stdout
    assert "policy rules/none" in result.stdout
    run = cli.outputs / "shelek" / "adhoc" / "2026-01-31_0000"
    expected_files = {"forecast.csv", "weather.csv", "trace.jsonl", "analysis.md", "run.json"}
    assert expected_files <= {p.name for p in run.iterdir()}
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert meta["policy"] == "rules" and meta["provider"] == "none"
    assert meta["weather_source"] == "cache"
    assert meta["issue_time_utc"] == "2026-01-30T18:00:00Z"
    forecast = pd.read_csv(run / "forecast.csv")
    keys = ["version", "entity", "target_time_utc"]
    values = ["mean", "p10", "p50", "p90"]
    assert {p.name: p.read_bytes() for p in protected.iterdir() if p.is_file()} == original_bytes
    assert len(forecast) == 270
    assert not forecast.duplicated(keys).any()
    assert np.isfinite(forecast[values].to_numpy()).all()
    assert forecast[values].stack().between(0, 1).all()
    assert (forecast.p10 <= forecast.p50).all() and (forecast.p50 <= forecast.p90).all()
    assert set(forecast.entity) == {"farm", "t1", "t2"}
    assert forecast.mw_mean.isna().all()
    utc = pd.to_datetime(forecast.target_time_utc, utc=True).dt.tz_localize(None)
    assert (pd.to_datetime(forecast.target_time_data_clock) - utc == pd.Timedelta(hours=6)).all()
    assert (pd.to_datetime(forecast.target_time_kz_official) - utc == pd.Timedelta(hours=5)).all()
    events = [json.loads(line) for line in (run / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[0]["type"] == "run_start" and events[-1]["type"] == "run_end"
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert not cli.network_log.exists()


def test_offline_forecast_matches_committed_artifact(cli):
    result = cli("forecast", "--issue", "2026-01-31 00:00", "--policy", "rules", "--offline")
    assert_success(result)
    columns = ["version", "entity", "target_time_utc", "mean", "p10", "p50", "p90"]
    forecast = pd.read_csv(cli.outputs / "shelek/adhoc/2026-01-31_0000/forecast.csv")
    reference = pd.read_csv(cli.repo / "outputs/shelek/runs/2026-01-31_0000/forecast.csv")
    pd.testing.assert_frame_equal(forecast[columns], reference[columns], atol=2e-4, rtol=0)


def test_forecast_prints_the_effective_output_directory(cli):
    result = cli("forecast", "--issue", "2026-02-05 00:00", "--policy", "rules")
    assert_success(result)
    expected = cli.outputs / "shelek/adhoc/2026-02-05_0000"
    assert expected.exists()
    assert expected.as_posix() in result.stdout.replace("\\", "/")


def test_explicit_llm_without_provider_reports_rules_fallback(cli):
    result = cli("forecast", "--issue", "2026-02-01 00:00", "--policy", "llm", "--offline", "-v")
    assert_success(result)
    meta = json.loads((cli.outputs / "shelek/adhoc/2026-02-01_0000/run.json").read_text(encoding="utf-8"))
    assert meta["policy"] == "rules" and meta["provider"] == "none"
    assert meta["warnings"], "The fallback must be disclosed in the saved run"
    assert "warning" in result.stdout
    assert not cli.network_log.exists()


def test_short_backtest_exports_only_initial_day_ahead_hours(cli):
    result = cli("backtest", "--start", "2026-01-31", "--end", "2026-02-01", "--policy", "rules", "--offline")
    assert_success(result)
    output = cli.outputs / "shelek/test_period"
    submission = pd.read_csv(output / "submission_day_ahead.csv")
    all_issues = pd.read_csv(output / "all_issues.csv")
    assert len(submission) == 48 and submission.target_time_utc.nunique() == 48
    assert set(submission.issue_id) == {"2026-01-31_0000", "2026-02-01_0000"}
    assert submission.lead_h.between(24, 47).all()
    assert submission.target_time_data_clock.min() == "2026-02-01 00:00"
    assert submission.target_time_data_clock.max() == "2026-02-02 23:00"
    initial = all_issues.query("entity == 'farm' and version == 1 and product == 'day_ahead'")
    paired = submission.merge(initial, on=["issue_id", "target_time_utc"], validate="one_to_one")
    assert len(initial) == len(submission) == len(paired)
    np.testing.assert_allclose(paired.farm_mean, paired["mean"], atol=1e-8)
    assert not cli.network_log.exists()


def test_custom_horizon_keeps_the_original_target_window(cli):
    result = cli("forecast", "--issue", "2026-02-05 00:00", "--horizon", "24", "--policy", "rules")
    assert_success(result)
    forecast = pd.read_csv(cli.outputs / "shelek/adhoc/2026-02-05_0000/forecast.csv")
    initial = forecast.query("version == 1")
    revised = forecast.query("version == 2")
    assert set(initial.entity) == set(revised.entity) == {"farm", "t1", "t2"}
    assert initial.groupby("entity").size().eq(24).all()
    assert revised.groupby("entity").size().eq(18).all()
    assert (initial.lead_h.min(), initial.lead_h.max()) == (0, 23)
    assert (revised.lead_h.min(), revised.lead_h.max()) == (6, 23)
    assert initial.target_time_utc.max() == revised.target_time_utc.max()
    assert set(forecast["product"]) == {"intraday"}
    assert not cli.network_log.exists()


def test_evaluate_scores_known_hourly_error(cli, tmp_path):
    # Synthetic public input files with analytically known error, independent of core readers.
    hours = pd.date_range("2026-01-01", periods=72, freq="h")
    powers = 0.4 + 0.2 * np.sin(np.arange(len(hours)) * 0.73)
    actuals = []
    for turbine in ("t1", "t2"):
        path = tmp_path / f"{turbine} actual generation.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "time", "wind", "power", "temperature"])
            for index, (hour, power) in enumerate(zip(hours, powers)):
                for minute in range(0, 60, 10):
                    writer.writerow([index * 6 + minute // 10, hour + pd.Timedelta(minutes=minute), 7, power, 10])
        actuals.extend(["--actuals", f"{turbine}={path}"])
    forecast = tmp_path / "forecast.csv"
    pd.DataFrame({
        "target_time_utc": (hours - pd.Timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "farm_mean": powers + 0.1, "farm_p10": 0, "farm_p90": 1,
        "t1_mean": powers + 0.1, "t2_mean": powers + 0.1,
    }).to_csv(forecast, index=False)
    result = cli("evaluate", "--forecast", forecast, *actuals)
    assert_success(result)
    report = json.loads((cli.outputs / "shelek/evaluation/evaluation.json").read_text(encoding="utf-8"))
    for entity in ("farm", "t1", "t2"):
        assert report["metrics"][entity]["mae"] == 0.1
        assert report["metrics"][entity]["rmse"] == 0.1
        assert report["metrics"][entity]["n"] == len(hours)
    assert report["best_lag_h"] == 0 and report["warning"] is None
    assert report["metrics"]["farm"]["coverage_p10_p90"] == 1
    assert not cli.network_log.exists()
