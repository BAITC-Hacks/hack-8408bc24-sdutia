import pandas as pd
import pytest

from windagent import api, config
from windagent.clock import detect_clock
from windagent.evaluate import evaluate


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("WINDAGENT_OUTPUTS_DIR", str(tmp_path))
    return tmp_path


def _forecast_from_actuals(tmp_path, noise=0.05):
    """A schema-S forecast built from January actuals (+ noise), so the expected MAE is known."""
    a = api.load_actuals("shelek", "2026-01-01T00:00Z", "2026-01-31T00:00Z")
    farm = a[a["entity"] == "farm"].copy()
    df = pd.DataFrame({"target_time_utc": farm["target_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "farm_mean": (farm["actual"] + noise).clip(0, 1), "farm_p10": 0.0, "farm_p90": 1.0})
    p = tmp_path / "fc.csv"
    df.to_csv(p, index=False)
    return p


def _shifted_actuals(tmp_path, turbine, hours):
    raw = pd.read_csv(config.data_dir() / "raw" / f"turbine_{turbine[-1]}.csv", encoding="utf-8")
    raw.iloc[:, 1] = (pd.to_datetime(raw.iloc[:, 1]) + pd.Timedelta(hours=hours)).dt.strftime("%Y-%m-%d %H:%M:%S")
    p = tmp_path / f"{turbine}_shift.csv"
    raw.to_csv(p, index=False, encoding="utf-8")
    return p


def test_evaluate_matches_known_error(tmp_path):
    fc = _forecast_from_actuals(tmp_path)
    raw = config.data_dir() / "raw"
    res = evaluate("shelek", {"t1": str(raw / "turbine_1.csv"), "t2": str(raw / "turbine_2.csv")}, str(fc))
    assert res["best_lag_h"] == 0 and res["warning"] is None
    assert res["metrics"]["farm"]["mae"] <= 0.05 + 1e-6 and res["metrics"]["farm"]["coverage_p10_p90"] == 1.0


def test_evaluate_detects_a_clock_mismatch(tmp_path):
    fc = _forecast_from_actuals(tmp_path, noise=0.0)
    paths = {t: str(_shifted_actuals(tmp_path, t, -1)) for t in ("t1", "t2")}   # as if exported in UTC+5
    res = evaluate("shelek", paths, str(fc))
    assert res["best_lag_h"] == -1 and res["warning"]
    assert res["metrics_realigned"]["farm"]["mae"] < res["metrics"]["farm"]["mae"]


def test_evaluate_rejects_unknown_turbine(tmp_path):
    with pytest.raises(api.InputError):
        evaluate("shelek", {"t9": "x.csv"}, None)


def test_detect_clock_finds_fixed_utc_plus_6():
    r = detect_clock("shelek")
    assert r["estimated_offset_h"] == 6 and r["matches_config"]
    assert r["switch_continuity"]["2024-03-01 00:00"]["duplicates"] == 0
    assert r["sun_peak_max_shift_min"] < 30
