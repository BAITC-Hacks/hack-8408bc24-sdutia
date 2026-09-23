import json

import pandas as pd
import pytest

from windagent import api, weather
from windagent.agent import runner

F_COLUMNS = {"site", "issue_id", "version", "issue_time_utc", "version_as_of_utc", "target_time_utc",
             "target_time_data_clock", "target_time_kz_official", "lead_h", "product", "entity",
             "mean", "p10", "p50", "p90", "mw_mean"}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("WINDAGENT_OUTPUTS_DIR", str(tmp_path))
    monkeypatch.setenv("WINDAGENT_OFFLINE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    return tmp_path


def test_rules_agent_end_to_end_offline(isolated):
    r = runner.run_issue("shelek", "2026-02-05 00:00", policy="rules")
    f, meta = r["forecast"], r["meta"]
    assert F_COLUMNS <= set(f.columns)
    v1, v2 = f[f["version"] == 1], f[f["version"] == 2]
    assert (v1.groupby("entity").size() == 48).all() and set(v1["entity"]) == {"farm", "t1", "t2"}
    assert (v2.groupby("entity").size() == 42).all()
    assert (f["p10"] <= f["p50"]).all() and (f["p50"] <= f["p90"]).all()
    assert f[["mean", "p10", "p50", "p90"]].stack().between(0, 1).all()
    assert set(v1["product"]) == {"intraday", "day_ahead"}
    assert meta["weather_source"] == "cache" and meta["policy"] == "rules"
    types = [e["type"] for e in r["trace"]]
    assert types[0] == "run_start" and types[-1] == "run_end" and "publish" not in types
    assert sum(1 for e in r["trace"] if e.get("tool") == "publish_forecast" and e["type"] == "tool_result") == 2
    # the weather actually used respects the as-of rule for every version
    w = r["weather"]
    info = {m["id"]: m["latency_h"] for m in api.system_info()["weather_models"]}
    as_of = {v["version"]: pd.Timestamp(v["as_of_utc"]) for v in meta["versions"]}
    published = w["init_time_utc"] + pd.to_timedelta(w["model"].map(info), unit="h")
    assert (published <= w["version"].map(as_of)).all()


def test_outlier_model_is_excluded(isolated, monkeypatch):
    real = weather.window_for_issue

    def corrupted(*a, **k):
        df, src = real(*a, **k)
        df = df.copy()
        df.loc[df["model"] == "gfs_seamless", ["ws100", "ws10"]] += 15.0
        return df, src

    monkeypatch.setattr(weather, "window_for_issue", corrupted)
    r = runner.run_issue("shelek", "2026-02-05 00:00", policy="rules", update_after_h=None)
    assert [x["model"] for x in r["meta"]["excluded_models"]] == ["gfs_seamless"]
    assert "gfs_seamless" not in r["meta"]["models_used"]
    assert any(e["type"] == "decision" and "gfs_seamless" in e["summary"] for e in r["trace"])


def test_llm_policy_without_keys_falls_back_to_rules(isolated):
    r = runner.run_issue("shelek", "2026-02-06 00:00", policy="llm", update_after_h=None)
    assert r["meta"]["policy"] == "rules"
    assert any(e["type"] == "warning" and "No LLM key" in e["summary"] for e in r["trace"])


@pytest.mark.parametrize("issue,horizon", [("2030-01-01 00:00", 48), ("2026-02-05 00:30", 48), ("2026-02-05 00:00", 0),
                                           ("2026-02-05 00:00", 500), ("05.02.2026", 48)])
def test_invalid_requests_raise_input_error(issue, horizon):
    with pytest.raises(api.InputError):
        runner.run_issue("shelek", issue, policy="rules", horizon_h=horizon)


def test_run_files_are_written_and_loadable(isolated):
    runner.run_issue("shelek", "2026-02-07 00:00", policy="rules", update_after_h=None)
    runs = api.list_runs("shelek")
    assert list(runs["issue_id"]) == ["2026-02-07_0000"]
    meta = json.loads((isolated / "shelek" / "runs" / "2026-02-07_0000" / "run.json").read_text(encoding="utf-8"))
    assert meta["runs_used"] and meta["kpis"]["capacity_factor"] is not None
