"""Command-line interface: python -m windagent <command>. See docs/CONTRACTS.md, section 6."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

from .errors import WindAgentError


def _site(args):
    from . import config
    return config.get_site(args.site)


def _apply_offline(args) -> None:
    if getattr(args, "offline", False):
        os.environ["WINDAGENT_OFFLINE"] = "1"


def _cmd_info(args) -> int:
    from . import api

    for site in api.list_sites():
        info = api.site_info(site)
        print(f"{site}: {info['name']} | turbines: {', '.join(t['id'] for t in info['turbines'])} "
              f"| data clock UTC+{info['data_clock_utc_offset_h']} | issue range {info['issue_range_data_clock']}")
    print("LLM:", api.llm_status())
    return 0


def _cmd_fetch_history(args) -> int:
    from . import weather

    counts = weather.fetch_history(_site(args), args.start, args.end)
    print(json.dumps(counts, indent=2))
    return 0


def _cmd_validate(args) -> int:
    from . import validation

    m = validation.run_validation(args.site)
    da = m["by_product"]["day_ahead"]
    print(f"day-ahead farm MAE: model {da['model']['mae']} | persistence {da['persistence']['mae']} | "
          f"climatology {da['climatology']['mae']} | NWP power curve {da['nwp_powercurve']['mae']}")
    print(f"skill: {m['skill_pct']} | interval calibration k={m['interval_calibration'].get('k')}")
    return 0


def _cmd_train(args) -> int:
    from . import forecast
    from .timeutil import parse_clock_time

    s = _site(args)
    cutoff = parse_clock_time(forecast.PRODUCTION_CUTOFF_DATA_CLOCK, s.data_clock_utc_offset_h)
    m = forecast.train_model(s, cutoff)
    print("saved", m.save(forecast.production_model_path(s)))
    return 0


def _cmd_backtest(args) -> int:
    from .agent import runner

    _apply_offline(args)
    t0 = time.time()
    res = runner.backtest(args.site, args.start, args.end, policy=args.policy, offline=args.offline or None)
    print(json.dumps(res, indent=2), f"\nbacktest finished in {time.time() - t0:.0f}s")
    return 0


def _cmd_forecast(args) -> int:
    from .agent import runner

    _apply_offline(args)
    printer = (lambda e: print(f"[{e['seq']:02d}] {e['type']:<11} {e.get('tool') or '':<18} {e['summary']}")) if args.verbose else None
    if args.now:
        r = runner.run_now(args.site, policy=args.policy, on_event=printer, horizon_h=args.horizon)
    else:
        if not args.issue:
            from .errors import InputError
            raise InputError("Give --issue \"YYYY-MM-DD HH:MM\" (data clock) or --now.",
                             "Укажите --issue \"ГГГГ-ММ-ДД ЧЧ:ММ\" (время данных) или --now.")
        r = runner.run_issue(args.site, args.issue, policy=args.policy, on_event=printer, horizon_h=args.horizon,
                             offline=args.offline or None, kind="adhoc")
    m = r["meta"]
    print(f"issue {m['issue_id']} | policy {m['policy']}/{m['provider']} | versions {len(m['versions'])} | "
          f"models {m['models_used']} | KPIs {m['kpis']}")
    print(f"written to outputs/{args.site}/{'live' if args.now else 'adhoc'}/{m['issue_id']}/")
    print(r["analysis"])
    return 0


def _cmd_all(args) -> int:
    t0 = time.time()
    print("== 1/3 validate (walk-forward)"), _cmd_validate(args)
    print("== 2/3 train production model"), _cmd_train(args)
    args.start, args.end = "2026-01-31", "2026-02-28"
    print("== 3/3 backtest test period"), _cmd_backtest(args)
    print(f"all done in {time.time() - t0:.0f}s")
    return 0


def _cmd_evaluate(args) -> int:
    from .errors import InputError
    from .evaluate import evaluate

    paths = {}
    for item in args.actuals or []:
        if "=" not in item:
            raise InputError(f"--actuals must look like t1=PATH, got '{item}'.", f"--actuals должен иметь вид t1=ПУТЬ, получено '{item}'.")
        k, v = item.split("=", 1)
        paths[k.strip()] = v.strip()
    res = evaluate(args.site, paths, args.forecast)
    for ent, m in res["metrics"].items():
        print(f"{ent:>5}: MAE {m['mae']} | RMSE {m['rmse']} | bias {m['bias']} | n {m['n']}"
              + (f" | P10–P90 coverage {m['coverage_p10_p90']}" if m.get("coverage_p10_p90") is not None else ""))
    print(f"best lag: {res['best_lag_h']:+d} h")
    if res["warning"]:
        print("WARNING:", res["warning"])
        for ent, m in (res["metrics_realigned"] or {}).items():
            print(f"  realigned {ent:>5}: MAE {m['mae']} | RMSE {m['rmse']}")
    print("written: outputs/<site>/evaluation/evaluation.json")
    return 0


def _cmd_detect_clock(args) -> int:
    from .clock import detect_clock

    r = detect_clock(args.site)
    print(f"configured UTC+{r['configured_offset_h']} | estimated UTC+{r['estimated_offset_h']} | matches: {r['matches_config']}")
    for t, c in r["switch_continuity"].items():
        print(f"at {t}: {c['records']}/{c['expected']} records, duplicates {c['duplicates']}, irregular steps {c['irregular_steps']}")
    for label, per_year in r["sun_peak_clock_time_by_year"].items():
        print(f"temperature peak ({label}): " + ", ".join(f"{y} {t}" for y, t in per_year.items()))
    print(r["conclusion"])
    return 0


def _cmd_report(args) -> int:
    from . import config
    from .report import render, update_readme

    for fname, lang in (("README.md", "ru"), ("README.en.md", "en")):
        ok = update_readme(config.REPO_ROOT / fname, render(args.site, lang))
        print(f"{fname}: {'updated' if ok else 'no METRICS markers (skipped)'}")
    return 0


def _cmd_verify(args) -> int:
    """The same checks as the CI workflow, in one local command (GitHub Actions may be unavailable)."""
    import subprocess
    import tempfile

    import pandas as pd

    from . import config
    from .clock import detect_clock
    from .report import compare

    root = config.REPO_ROOT
    results = []
    env = {**os.environ, "WINDAGENT_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}
    print("1/4 running the test suite (pytest)…", flush=True)
    t = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, env=env, capture_output=True, text=True, encoding="utf-8")
    last = (t.stdout.strip().splitlines() or ["no output"])[-1]
    results.append(("tests (pytest -q)", t.returncode == 0, last))
    print("2/4 re-running the February backtest offline (rules policy) in a temporary folder…", flush=True)
    import shutil

    tmp = tempfile.mkdtemp(prefix="windagent_verify_")
    tmp_models = os.path.join(tmp, "_models")
    shutil.copytree(config.models_dir(), tmp_models, ignore=shutil.ignore_patterns("model_cutoff_*"))
    val = config.outputs_dir() / args.site / "validation"
    shutil.copytree(val, os.path.join(tmp, args.site, "validation"))
    b = subprocess.run([sys.executable, "-m", "windagent", "backtest", "--offline", "--policy", "rules"], cwd=root,
                       env={**env, "WINDAGENT_OUTPUTS_DIR": tmp, "WINDAGENT_MODELS_DIR": tmp_models},
                       capture_output=True, text=True, encoding="utf-8")
    committed = config.outputs_dir() / args.site / "test_period" / "submission_day_ahead.csv"
    fresh = os.path.join(tmp, args.site, "test_period", "submission_day_ahead.csv")
    if b.returncode == 0 and os.path.exists(fresh):
        ok, msg = compare(str(committed), fresh)
    else:
        ok, msg = False, (b.stderr.strip().splitlines() or ["backtest failed"])[-1]
    results.append(("reproduce submission from scratch", ok, msg))
    print("3/4 checking the committed submission…", flush=True)
    s = pd.read_csv(committed, encoding="utf-8")
    q = s[["farm_p10", "farm_p50", "farm_p90"]]
    ok = (len(s) == 672 and s["farm_mean"].notna().all() and bool(q.stack().between(0, 1).all())
          and bool((q["farm_p10"] <= q["farm_p50"]).all() and (q["farm_p50"] <= q["farm_p90"]).all()))
    results.append(("submission schema (672 h, 0–1, P10≤P50≤P90)", ok, f"{len(s)} rows"))
    print("4/4 clock forensics…", flush=True)
    r = detect_clock(args.site)
    results.append(("SCADA clock is fixed UTC+6", bool(r["matches_config"]), r["conclusion"][:90]))
    print()
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<48} {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


def _cmd_provenance(args) -> int:
    from .provenance import run

    r = run(args.site)
    for x in r["latency"]:
        print(f"{x['model']:<22} measured delay {x['measured_delay_h']:>5} h | configured {x['configured_latency_h']} h | "
              f"margin {x['margin_h']:+} h | {'OK' if x['ok'] else 'TOO SMALL'}")
    s = r["previous_runs_semantics"]
    print(f"previous_dayN semantics ({s['model']}, {s['day']}): {s['values_matching']}/{s['values_checked']} values match "
          f"'{s['rule']}' -> {'OK' if s['ok'] else 'MISMATCH'}")
    print("written: outputs/provenance/provenance.json")
    return 0 if all(x["ok"] for x in r["latency"]) and s["ok"] else 1


def _cmd_compare(args) -> int:
    from .report import compare

    ok, msg = compare(args.a, args.b, args.atol)
    print(("OK: " if ok else "MISMATCH: ") + msg)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="windagent", description="Agentic AI wind-farm forecasting")
    p.add_argument("--debug", action="store_true", help="show tracebacks on errors")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, help_text, handler):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--site", default="shelek")
        sp.set_defaults(handler=handler)
        return sp

    add("info", "show configured sites and LLM status", _cmd_info)
    sp = add("fetch-history", "download archived weather forecasts into the cache", _cmd_fetch_history)
    sp.add_argument("--start", default="2024-02-01")
    sp.add_argument("--end", default="2026-03-03")
    add("train", "train the production model (cutoff = first test issue)", _cmd_train)
    add("validate", "walk-forward validation → outputs/<site>/validation", _cmd_validate)
    for name, handler, help_text in (("backtest", _cmd_backtest, "rolling agent run over the test period"),
                                     ("all", _cmd_all, "end-to-end: validate → train → backtest")):
        sp = add(name, help_text, handler)
        sp.add_argument("--start", default="2026-01-31")
        sp.add_argument("--end", default="2026-02-28")
        sp.add_argument("--policy", default="auto", choices=["auto", "llm", "rules"])
        sp.add_argument("--offline", action="store_true", help="use the committed weather cache only")
    sp = add("forecast", "single issue (--issue) or live (--now)", _cmd_forecast)
    sp.add_argument("--issue", help='issue time on the data clock, e.g. "2026-02-10 00:00"')
    sp.add_argument("--now", action="store_true")
    sp.add_argument("--horizon", type=int, default=48)
    sp.add_argument("--policy", default="auto", choices=["auto", "llm", "rules"])
    sp.add_argument("--offline", action="store_true")
    sp.add_argument("--verbose", "-v", action="store_true", help="print the agent trace live")
    sp = add("evaluate", "score a forecast against actuals (organizers' CSV format)", _cmd_evaluate)
    sp.add_argument("--actuals", action="append", metavar="TURBINE=PATH", help="repeat per turbine, e.g. --actuals t1=feb_t1.csv")
    sp.add_argument("--forecast", help="forecast CSV (default: outputs/<site>/test_period/submission_day_ahead.csv)")
    add("detect-clock", "SCADA clock-offset forensics", _cmd_detect_clock)
    add("report", "write the generated metrics block into README.md / README.en.md", _cmd_report)
    add("verify", "run all checks locally: tests, reproduce the submission, schema, clock", _cmd_verify)
    add("provenance", "re-measure weather publication delays and the archive's run mapping (needs internet)", _cmd_provenance)
    sp = sub.add_parser("compare", help="compare two submission CSVs (CI reproducibility check)")
    sp.add_argument("a")
    sp.add_argument("b")
    sp.add_argument("--atol", type=float, default=2e-4)
    sp.set_defaults(handler=_cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args) or 0)
    except WindAgentError as exc:
        if args.debug:
            traceback.print_exc()
        print(f"ERROR: {exc.user_message_en}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - last-resort guard, keeps the CLI contract
        if args.debug:
            traceback.print_exc()
        print(f"ERROR: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
