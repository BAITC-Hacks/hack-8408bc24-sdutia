"""Command-line interface: python -m windagent <command>. See docs/CONTRACTS.md, section 6."""

from __future__ import annotations

import argparse
import sys
import traceback

from .errors import WindAgentError


def _not_ready(name: str):
    def run(_args):
        raise WindAgentError(f"Command '{name}' is under construction in this build.")
    return run


def _cmd_info(args) -> int:
    from . import api

    for site in api.list_sites():
        info = api.site_info(site)
        print(f"{site}: {info['name']} | turbines: {', '.join(t['id'] for t in info['turbines'])} "
              f"| data clock UTC+{info['data_clock_utc_offset_h']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="windagent", description="Agentic AI wind-farm forecasting")
    p.add_argument("--debug", action="store_true", help="show tracebacks on errors")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, help_text, handler, site=True):
        sp = sub.add_parser(name, help=help_text)
        if site:
            sp.add_argument("--site", default="shelek")
        sp.set_defaults(handler=handler)
        return sp

    add("info", "show configured sites", _cmd_info, site=False)
    for name, help_text in [
        ("all", "end-to-end reproduction"),
        ("fetch-history", "download archived weather forecasts into the cache"),
        ("train", "train the forecasting models"),
        ("validate", "walk-forward validation"),
        ("backtest", "rolling agent run over the test period"),
        ("forecast", "single issue or live forecast"),
        ("evaluate", "score a forecast against actuals"),
        ("detect-clock", "SCADA clock-offset forensics"),
        ("report", "insert metrics into README"),
    ]:
        add(name, help_text, _not_ready(name))
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
