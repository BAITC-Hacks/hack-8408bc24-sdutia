"""Subprocess checks for the public CLI contract (docs/CONTRACTS.md, section 6)."""

import json
import shutil

import pytest


def assert_cli_error(result, expected_code):
    """Normal user errors must be useful without leaking a Python traceback."""
    assert result.returncode == expected_code, (result.stdout, result.stderr)
    lines = result.stderr.strip().splitlines()
    assert len(lines) == 1, result.stderr
    assert lines[0].startswith("ERROR: "), result.stderr
    assert "Traceback" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "args, message",
    [
        (("forecast",), "--issue"),
        (("forecast", "--issue", "31.01.2026"), "Invalid time"),
        (("forecast", "--issue", "2026-02-30 00:00"), "Invalid time"),
        (("forecast", "--issue", "2026-02-10 00:15"), "full hour"),
        (("forecast", "--issue", "2099-01-01 00:00"), "between"),
        (("forecast", "--issue", "1900-01-01 00:00"), "between"),
        (("forecast", "--issue", "2026-02-05 00:00", "--site", "missing-site"), "Unknown site"),
        (("backtest", "--site", "missing-site", "--offline"), "Unknown site"),
        (("evaluate",), "--actuals"),
        (("evaluate", "--actuals", "missing-equals.csv"), "--actuals must look like"),
        (("evaluate", "--actuals", "unknown-turbine=missing.csv"), "Unknown turbine"),
    ],
    ids=[
        "forecast-missing-time", "wrong-date-format", "impossible-calendar-date", "partial-hour",
        "future-issue", "past-issue", "forecast-unknown-site", "backtest-unknown-site",
        "evaluate-missing-actuals", "actuals-malformed-assignment", "actuals-unknown-turbine",
    ],
)
def test_invalid_user_input(cli, args, message):
    result = cli(*args)
    assert_cli_error(result, 2)
    assert message in result.stderr
    assert not list(cli.outputs.rglob("run.json"))


@pytest.mark.parametrize("horizon", ["-1", "0", "73", "500"])
def test_forecast_horizon_out_of_range(cli, horizon):
    result = cli("forecast", "--issue", "2026-02-05 00:00", "--horizon", horizon)
    assert_cli_error(result, 2)
    assert "Horizon" in result.stderr


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="CLI-ERROR-FORMAT: argparse bypasses the one-line ERROR contract")
@pytest.mark.parametrize(
    "args",
    [
        (),
        ("missing-command",),
        ("forecast", "--issue"),
        ("forecast", "--issue", "2026-02-05 00:00", "--horizon", "abc"),
        ("forecast", "--issue", "2026-02-05 00:00", "--policy", "missing-policy"),
    ],
    ids=["missing-command", "unknown-command", "missing-option-value", "horizon-nonnumeric", "unknown-policy"],
)
def test_parser_input_errors_use_documented_stderr_format(cli, args):
    assert_cli_error(cli(*args), 2)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="CLI-BACKTEST-DATE: malformed dates exit 1 instead of input-error code 2")
@pytest.mark.parametrize(
    "start,end",
    [("not-a-date", "2026-02-05"), ("2026-02-05", "not-a-date"), ("2026-02-30", "2026-02-30")],
    ids=["malformed-start", "malformed-end", "impossible-calendar-date"],
)
def test_backtest_invalid_dates_are_input_errors(cli, start, end):
    result = cli("backtest", "--start", start, "--end", end, "--offline", "--policy", "rules")
    assert_cli_error(result, 2)


def test_backtest_reversed_range_is_rejected(cli):
    result = cli("backtest", "--start", "2026-02-06", "--end", "2026-02-05", "--offline", "--policy", "rules")
    assert_cli_error(result, 2)
    assert not list(cli.outputs.rglob("submission_day_ahead.csv"))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="CLI-ISSUE-NOW: --now silently takes precedence over --issue")
def test_forecast_issue_and_now_are_mutually_exclusive(cli):
    # The socket audit guard is a backstop if --now accidentally wins over --issue.
    result = cli("forecast", "--issue", "2026-02-05 00:00", "--now", "--offline", "--policy", "rules")
    assert_cli_error(result, 2)
    assert "--issue" in result.stderr and "--now" in result.stderr
    assert not cli.network_log.exists() or not cli.network_log.read_text(encoding="utf-8").strip()
    assert not list(cli.outputs.rglob("run.json"))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="CLI-OFFLINE-NOW: the live runner attempts network despite offline mode")
def test_forecast_now_offline_never_attempts_network(cli):
    # Real live command, guarded only at the OS socket boundary by the fixture.
    result = cli("forecast", "--now", "--offline", "--policy", "rules")
    attempts = cli.network_log.read_text(encoding="utf-8") if cli.network_log.exists() else ""
    assert not attempts, (result.returncode, result.stderr, attempts)
    assert result.returncode in (0, 3), (result.stdout, result.stderr)


def test_evaluate_missing_forecast_has_data_exit_code(cli, tmp_path):
    missing = tmp_path / "missing-forecast.csv"
    result = cli("evaluate", "--actuals", "t1=missing-actuals.csv", "--forecast", str(missing))
    assert_cli_error(result, 3)
    assert "Forecast file not found" in result.stderr


def test_evaluate_missing_actuals_has_data_exit_code(cli, tmp_path):
    submission = cli.repo / "outputs" / "shelek" / "test_period" / "submission_day_ahead.csv"
    missing = tmp_path / "missing-actuals.csv"
    result = cli("evaluate", "--actuals", f"t1={missing}", "--forecast", str(submission))
    assert_cli_error(result, 3)
    assert "SCADA file not found" in result.stderr


def test_missing_configuration_has_data_exit_code(cli, tmp_path):
    result = cli("forecast", "--issue", "2026-02-05 00:00", env={"WINDAGENT_CONFIG": str(tmp_path / "missing.yaml")})
    assert_cli_error(result, 3)
    assert "Site config not found" in result.stderr


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="CLI-MISSING-DATA: absent SCADA ends in an empty-concatenation error")
def test_missing_scada_has_data_exit_code(cli, tmp_path):
    data = tmp_path / "data-without-scada"
    shutil.copytree(cli.repo / "data" / "cache", data / "cache")
    result = cli("forecast", "--issue", "2026-02-05 00:00", "--offline", "--policy", "rules",
                 env={"WINDAGENT_DATA_DIR": str(data)})
    assert_cli_error(result, 3)
    assert "SCADA" in result.stderr


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="CLI-MISSING-DATA: absent offline weather cache ends in an empty-concatenation error")
def test_offline_missing_weather_cache_has_data_exit_code(cli, tmp_path):
    data = tmp_path / "data-without-weather-cache"
    shutil.copytree(cli.repo / "data" / "raw", data / "raw")
    result = cli("forecast", "--issue", "2026-02-05 00:00", "--offline", "--policy", "rules",
                 env={"WINDAGENT_DATA_DIR": str(data)})
    assert not cli.network_log.exists() or not cli.network_log.read_text(encoding="utf-8").strip()
    assert_cli_error(result, 3)


def test_debug_opt_in_preserves_exit_code_and_shows_traceback(cli):
    result = cli("--debug", "forecast")
    assert result.returncode == 2
    assert "Traceback (most recent call last)" in result.stderr
    assert result.stderr.strip().splitlines()[-1].startswith("ERROR: ")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="CLI-OUTPUTS-DIR: documented --outputs-dir option is not implemented")
def test_backtest_outputs_dir_flag_writes_only_requested_directory(cli, tmp_path):
    destination = tmp_path / "requested-outputs"
    result = cli("backtest", "--start", "2026-02-05", "--end", "2026-02-05", "--policy", "rules",
                 "--offline", "--outputs-dir", str(destination))
    assert result.returncode == 0, (result.stdout, result.stderr)
    run = destination / "shelek" / "runs" / "2026-02-05_0000" / "run.json"
    assert run.exists()
    metadata = json.loads(run.read_text(encoding="utf-8"))
    assert metadata["policy"] == "rules"
    assert metadata["weather_source"] == "cache"
    assert not list(cli.outputs.rglob("run.json"))

