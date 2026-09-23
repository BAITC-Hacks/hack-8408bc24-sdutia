import numpy as np
import pandas as pd
import pytest

from windagent import asof, config, dataset
from windagent.timeutil import parse_clock_time


@pytest.fixture(scope="module")
def weather_long():
    return dataset.load_weather_cache(config.get_site("shelek"))


def _latency(model):
    return pd.to_timedelta(asof.latency_h(model), unit="h")


def test_no_selected_value_comes_from_an_unpublished_run(weather_long):
    rng = np.random.default_rng(0)
    days = pd.date_range("2025-03-01", "2026-02-27", freq="D", tz="UTC")
    issues = [d + pd.Timedelta(hours=int(h)) for d, h in zip(rng.choice(days, 25), rng.integers(0, 24, 25))]
    sel = asof.select_asof_many(weather_long, issues, 48)
    assert len(sel) > 0
    published = sel["init_time_utc"] + sel["model"].map(_latency)
    assert (published <= sel["issue_time_utc"]).all()


def test_the_freshest_usable_run_is_chosen(weather_long):
    T = parse_clock_time("2026-02-10 00:00", 6)
    sel = asof.select_asof(weather_long, T, 48)
    w = weather_long[(weather_long["valid_time_utc"] >= T) & (weather_long["valid_time_utc"] <= T + pd.Timedelta(hours=48))]
    w = w[(w["init_time_utc"] + w["model"].map(_latency) <= T) & w["ws100"].notna()]
    freshest = w.groupby(["model", "valid_time_utc"])["init_time_utc"].max().rename("best").reset_index()
    m = sel.merge(freshest, on=["model", "valid_time_utc"])
    assert len(m) == len(sel)
    assert (m["init_time_utc"] == m["best"]).all()


def test_future_fallback_values_are_never_selected():
    # For future valid times the API may return the latest run under offset 1 even though the run
    # "floor_6h(v) - 1 day" does not exist yet; its computed init time is in the future and must be rejected.
    T = pd.Timestamp("2026-09-23T06:00:00Z")
    v = pd.Timestamp("2026-09-25T00:00:00Z")
    rows = pd.DataFrame({
        "valid_time_utc": [v, v], "model": ["icon_seamless"] * 2, "offset_days": [1, 2],
        "init_time_utc": [v.floor("6h") - pd.Timedelta(days=1), v.floor("6h") - pd.Timedelta(days=2)],
        "ws10": [5.0, 4.0], "ws100": [8.0, 7.0], "wd100": [200.0, 190.0], "t2m": [10.0, 10.0], "sp": [950.0, 950.0],
    })
    sel = asof.select_asof(rows, T, 48)
    assert list(sel["offset_days"]) == [2]


def test_assert_asof_raises_on_violation():
    bad = pd.DataFrame({"issue_time_utc": [pd.Timestamp("2026-02-01T00:00Z")], "model": ["ecmwf_ifs025"],
                        "valid_time_utc": [pd.Timestamp("2026-02-01T12:00Z")], "init_time_utc": [pd.Timestamp("2026-01-31T18:00Z")]})
    with pytest.raises(AssertionError):
        asof.assert_asof(bad)
