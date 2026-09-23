"""Create deterministic SYNTHETIC UI fixtures; never use as forecast evidence.

Run from any directory with ``python tests/fixtures/make_fixtures.py``.
Uses only the standard library and always writes inside tests/fixtures/outputs.
The UI accesses the result through windagent.api with WINDAGENT_OUTPUTS_DIR set.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean


OUTPUTS = Path(__file__).resolve().parent / "outputs"
UTC = timezone.utc
DATA_CLOCK = timezone(timedelta(hours=6))
KZ_CLOCK = timezone(timedelta(hours=5))
MODELS = (
    "ecmwf_ifs025",
    "ecmwf_aifs025_single",
    "icon_seamless",
    "gfs_seamless",
)
ENTITIES = ("farm", "t1", "t2")
F_COLUMNS = (
    "site", "issue_id", "version", "issue_time_utc", "target_time_utc",
    "target_time_data_clock", "target_time_kz_official", "lead_h", "product",
    "entity", "mean", "p10", "p50", "p90", "mw_mean",
)
W_COLUMNS = (
    "issue_id", "target_time_utc", "model", "init_time_utc", "offset_days",
    "ws10", "ws100", "wd100", "t2m", "sp", "source",
)
S_COLUMNS = (
    "target_time_data_clock", "target_time_utc", "target_time_kz_official",
    "issue_id", "issue_time_utc", "lead_h", "farm_mean", "farm_p10",
    "farm_p50", "farm_p90", "t1_mean", "t2_mean", "models_used",
)
T_KEYS = {
    "ts_utc", "run_id", "seq", "type", "policy", "provider", "model",
    "tool", "args", "summary", "data", "duration_ms", "ok",
}


def iso(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def issue_time(day: str) -> datetime:
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=DATA_CLOCK).astimezone(UTC)


def issue_id(issue: datetime) -> str:
    return issue.astimezone(DATA_CLOCK).strftime("%Y-%m-%d_%H%M")


def clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def forecast_rows(issue: datetime, phase: float, versions: tuple[int, ...] = (1, 2)) -> list[dict]:
    rows = []
    for version in versions:
        for lead in range(48):
            target = issue + timedelta(hours=lead)
            base = 0.44 + 0.22 * math.sin((lead + phase) / 7) + 0.07 * math.cos(lead / 3)
            revision = (version - 1) * (0.022 * math.sin(lead / 6) - 0.008)
            powers = {
                "t1": clamp(base + revision + 0.025 * math.sin(lead / 4)),
                "t2": clamp(base + revision - 0.032 + 0.015 * math.cos(lead / 5)),
            }
            powers["farm"] = round(mean(powers.values()), 6)
            for entity in ENTITIES:
                point = powers[entity]
                width = 0.065 + lead * 0.0013 + (0.01 if entity != "farm" else 0)
                rows.append({
                    "site": "shelek", "issue_id": issue_id(issue), "version": version,
                    "issue_time_utc": iso(issue), "target_time_utc": iso(target),
                    "target_time_data_clock": target.astimezone(DATA_CLOCK).strftime("%Y-%m-%d %H:%M"),
                    "target_time_kz_official": target.astimezone(KZ_CLOCK).strftime("%Y-%m-%d %H:%M"),
                    "lead_h": lead, "product": "intraday" if lead < 24 else "day_ahead",
                    "entity": entity, "mean": point, "p10": clamp(point - width),
                    "p50": point, "p90": clamp(point + width), "mw_mean": "",
                })
    return rows


def weather_rows(issue: datetime, phase: float) -> list[dict]:
    rows = []
    # V1 uses 06 UTC runs, v2 uses 12 UTC runs, all available before their as-of.
    # W has no version column; init_time_utc identifies the weather run.
    for version in (1, 2):
        init = issue - timedelta(hours=12 if version == 1 else 6)
        for model_index, model in enumerate(MODELS):
            for lead in range(48):
                target = issue + timedelta(hours=lead)
                ws100 = 7.2 + 2.3 * math.sin((lead + phase) / 7) + 0.18 * model_index + 0.13 * (version - 1)
                rows.append({
                    "issue_id": issue_id(issue), "target_time_utc": iso(target),
                    "model": model, "init_time_utc": iso(init),
                    "offset_days": (target.date() - init.date()).days,
                    "ws10": round(ws100 * 0.73, 4), "ws100": round(ws100, 4),
                    "wd100": round((245 + 35 * math.sin(lead / 9) + model_index * 3) % 360, 4),
                    "t2m": round(-3.0 + 4.0 * math.sin((lead - 7) * math.pi / 12) + 0.1 * model_index, 4),
                    "sp": round(933 + 3 * math.cos(lead / 11) + model_index * 0.1, 4),
                    "source": "cache",
                })
    return rows


def trace_events(issue: datetime) -> list[dict]:
    specs = [
        ("run_start", None, "Synthetic fixture run started", 0),
        ("plan", None, "Check archived weather, forecast and publish", 1),
        ("tool_call", "fetch_weather", "Read synthetic cached weather inputs", 2),
        ("tool_result", "fetch_weather", "4 models, 48 hours, no missing values", 3),
        ("decision", "validate_weather", "Synthetic weather runs satisfy the as-of guard", 4),
        ("warning", None, "Synthetic test data: not a real forecast or evaluation", 5),
        ("tool_call", "predict", "Calculate synthetic forecast intervals", 6),
        ("tool_result", "predict", "48 hours for farm, t1 and t2", 7),
        ("publish", "publish", "Published synthetic version 1", 8),
        ("decision", "check_new_runs", "New synthetic weather run available; recompute", 21600),
        ("publish", "publish", "Published synthetic version 2", 21601),
        ("run_end", None, "Synthetic fixture run complete", 21602),
    ]
    events = []
    for seq, (event_type, tool, summary, seconds) in enumerate(specs, start=1):
        events.append({
            "ts_utc": iso(issue + timedelta(seconds=seconds)),
            "run_id": f"shelek/{issue_id(issue)}", "seq": seq, "type": event_type,
            "policy": "rules", "provider": "none", "model": None, "tool": tool,
            "args": {"models": list(MODELS)} if event_type == "tool_call" and tool == "fetch_weather" else None,
            "summary": summary,
            "data": {"version": 1 if seq == 9 else 2} if event_type == "publish" else {},
            "duration_ms": 40 + seq * 7 if tool else None, "ok": True,
        })
    return events


def run_meta(issue: datetime, rows: list[dict]) -> dict:
    farm_v1 = [r for r in rows if r["entity"] == "farm" and r["version"] == 1]
    farm_v2 = [r for r in rows if r["entity"] == "farm" and r["version"] == 2]
    points = [r["mean"] for r in farm_v2]
    return {
        "synthetic_fixture": True,
        "site": "shelek", "issue_id": issue_id(issue), "issue_time_utc": iso(issue),
        "created_at_utc": iso(issue + timedelta(hours=6, seconds=2)),
        "policy": "rules", "provider": "none", "model": None, "horizon_h": 48,
        "versions": [
            {"version": 1, "as_of_utc": iso(issue), "reason": "initial"},
            {"version": 2, "as_of_utc": iso(issue + timedelta(hours=6)), "reason": "new synthetic weather runs"},
        ],
        "models_used": list(MODELS), "excluded_models": [],
        "runs_used": {model: [iso(issue - timedelta(hours=12)), iso(issue - timedelta(hours=6))] for model in MODELS},
        "weather_source": "cache",
        "warnings": ["SYNTHETIC FIXTURE / СИНТЕТИЧЕСКИЕ ДАННЫЕ: UI tests only; not forecast evidence."],
        "kpis": {
            "energy_norm_h": round(sum(points), 6), "capacity_factor": round(mean(points), 6),
            "max_ramp_3h": round(max(abs(points[i] - points[i - 3]) for i in range(3, len(points))), 6),
            "mean_band_width": round(mean(r["p90"] - r["p10"] for r in farm_v2), 6),
            "revision_mae_v2_vs_v1": round(mean(abs(a["mean"] - b["mean"]) for a, b in zip(farm_v2, farm_v1)), 6),
        },
        "actuals_available": False, "errors": None,
    }


def submission_rows(all_rows: list[dict]) -> list[dict]:
    by_hour: dict[str, dict] = {}
    for row in all_rows:
        if row["version"] != 1 or row["product"] != "day_ahead":
            continue
        target = row["target_time_utc"]
        output = by_hour.setdefault(target, {key: row[key] for key in S_COLUMNS[:6]})
        output[f'{row["entity"]}_mean'] = row["mean"]
        if row["entity"] == "farm":
            for quantile in ("p10", "p50", "p90"):
                output[f"farm_{quantile}"] = row[quantile]
        output["models_used"] = ";".join(MODELS)
    return [by_hour[key] for key in sorted(by_hour)]


def metric_summary(predicted: list[float], actual: list[float]) -> dict:
    errors = [prediction - value for prediction, value in zip(predicted, actual)]
    centered_squares = sum((value - mean(actual)) ** 2 for value in actual)
    squared_errors = sum(error**2 for error in errors)
    return {
        "mae": round(mean(abs(error) for error in errors), 6),
        "rmse": round(math.sqrt(squared_errors / len(errors)), 6),
        "bias": round(mean(errors), 6),
        "r2": round(1 - squared_errors / centered_squares, 6) if centered_squares else 0.0,
        "n": len(errors),
    }


def validation_data() -> tuple[list[dict], dict]:
    rows = []
    for month, phase in ((10, 2.0), (11, 4.0), (12, 6.0)):
        current = forecast_rows(issue_time(f"2025-{month:02d}-01"), phase, versions=(1,))
        for row in current:
            # Synthetic outcomes allow every displayed metric to be recomputed.
            deviation = 0.09 * math.sin(row["lead_h"] / 4 + month) + 0.045 * math.cos(row["lead_h"] / 2)
            row["actual"] = clamp(row["mean"] + deviation)
        rows.extend(current)
    farm = [row for row in rows if row["entity"] == "farm"]

    def baselines(selected: list[dict]) -> dict:
        actual = [r["actual"] for r in selected]
        return {
            "model": metric_summary([r["mean"] for r in selected], actual),
            "persistence": metric_summary([0.48 for _ in selected], actual),
            "climatology": metric_summary([0.40 for _ in selected], actual),
            "nwp_powercurve": metric_summary([clamp(r["mean"] + 0.075 * math.sin(r["lead_h"] / 3) + 0.045) for r in selected], actual),
        }

    by_product = {product: baselines([r for r in farm if r["product"] == product]) for product in ("day_ahead", "intraday")}
    intervals = {}
    for product in by_product:
        selected = [r for r in farm if r["product"] == product]
        intervals[product] = {
            "coverage_p10_p90": round(mean(r["p10"] <= r["actual"] <= r["p90"] for r in selected), 6),
            "mean_width": round(mean(r["p90"] - r["p10"] for r in selected), 6),
        }
    day_ahead = by_product["day_ahead"]
    t2 = [r for r in rows if r["entity"] == "t2"]
    metrics = {
        "synthetic_fixture": True, "site": "shelek", "generated_at_utc": "2026-01-01T00:00:00Z",
        "validation": {"start_utc": rows[0]["target_time_utc"], "end_utc": rows[-1]["target_time_utc"], "n_issues": 3, "issue_hour_data_clock": 0},
        "by_product": by_product,
        "by_lead": [{"lead_h": lead, **{name: result["mae"] for name, result in baselines([r for r in farm if r["lead_h"] == lead]).items()}} for lead in range(48)],
        "intervals": intervals,
        "skill_pct": {f"day_ahead_vs_{baseline}": round(100 * (1 - day_ahead["model"]["mae"] / day_ahead[baseline]["mae"]), 6) for baseline in ("climatology", "persistence")},
        "per_month": [{"month": month, **{name: result["mae"] for name, result in baselines([r for r in farm if r["target_time_data_clock"].startswith(month)]).items() if name != "nwp_powercurve"}} for month in ("2025-10", "2025-11", "2025-12")],
        "leave_one_turbine_out": {
            "train": "t1", "test": "t2",
            "mae": metric_summary([clamp(r["mean"] + 0.02) for r in t2], [r["actual"] for r in t2])["mae"],
            "mae_reference": metric_summary([r["mean"] for r in t2], [r["actual"] for r in t2])["mae"],
        },
    }
    return rows, metrics


def validate_forecasts(rows: list[dict], *, actuals: bool = False) -> None:
    columns = set(F_COLUMNS) | ({"actual"} if actuals else set())
    seen = set()
    for row in rows:
        assert set(row) == columns
        key = (row["issue_id"], row["version"], row["target_time_utc"], row["entity"])
        assert key not in seen
        seen.add(key)
        assert 0 <= row["p10"] <= row["p50"] <= row["p90"] <= 1
        assert 0 <= row["mean"] <= 1
        assert row["mw_mean"] == ""
        target = datetime.fromisoformat(row["target_time_utc"].replace("Z", "+00:00"))
        issued = datetime.fromisoformat(row["issue_time_utc"].replace("Z", "+00:00"))
        assert target - issued == timedelta(hours=row["lead_h"])
        assert row["target_time_data_clock"] == target.astimezone(DATA_CLOCK).strftime("%Y-%m-%d %H:%M")
        assert row["target_time_kz_official"] == target.astimezone(KZ_CLOCK).strftime("%Y-%m-%d %H:%M")
        assert row["product"] == ("intraday" if row["lead_h"] < 24 else "day_ahead")
        if actuals:
            assert 0 <= row["actual"] <= 1


def generate() -> None:
    site = OUTPUTS / "shelek"
    all_forecasts = []
    for day, phase in (("2026-02-01", 0.0), ("2026-02-02", 3.0)):
        issue = issue_time(day)
        run = site / "runs" / issue_id(issue)
        forecasts = forecast_rows(issue, phase)
        weather = weather_rows(issue, phase)
        events = trace_events(issue)
        validate_forecasts(forecasts)
        assert len(forecasts) == 2 * 48 * len(ENTITIES)
        assert len(weather) == 2 * 48 * len(MODELS)
        assert all(set(row) == set(W_COLUMNS) for row in weather)
        assert all(set(event) == T_KEYS for event in events)
        write_csv(run / "forecast.csv", F_COLUMNS, forecasts)
        write_csv(run / "weather.csv", W_COLUMNS, weather)
        write_json(run / "run.json", run_meta(issue, forecasts))
        (run / "trace.jsonl").write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8", newline="\n")
        (run / "analysis.md").write_text(
            "<!-- synthetic_fixture: true; UI testing only, not forecast evidence. -->\n\n"
            "### Synthetic fixture / Синтетический пример\n\n"
            "These deterministic values exercise the UI. They do not represent measured weather, "
            "real model forecasts, validation skill, or a completed agent run.\n\n"
            "- Two synthetic revisions cover 48 hours for farm, t1 and t2.\n"
            "- Four synthetic weather series are labelled as cached inputs.\n"
            "- Version 2 illustrates a revision after a later weather run becomes available.\n"
            "- Normalized power has no MW conversion because rated capacity is unset.\n",
            encoding="utf-8", newline="\n",
        )
        all_forecasts.extend(forecasts)
    submission = submission_rows(all_forecasts)
    assert len(submission) == 48
    assert all(set(row) == set(S_COLUMNS) for row in submission)
    write_csv(site / "test_period" / "all_issues.csv", F_COLUMNS, all_forecasts)
    write_csv(site / "test_period" / "submission_day_ahead.csv", S_COLUMNS, submission)
    validation, metrics = validation_data()
    validate_forecasts(validation, actuals=True)
    write_csv(site / "validation" / "predictions.csv", (*F_COLUMNS, "actual"), validation)
    write_json(site / "validation" / "metrics.json", metrics)
    print(f"Generated 14 synthetic fixture files in {OUTPUTS}")


if __name__ == "__main__":
    generate()
