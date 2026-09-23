"""Trace events (schema T in docs/CONTRACTS.md), streamed to an optional callback."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from ..timeutil import iso_z, now_utc


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        if isinstance(x, dict):
            return {str(k): _jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_jsonable(v) for v in x]
        return str(x)


class Tracer:
    def __init__(self, run_id: str, policy: str, provider: str, model: str | None,
                 on_event: Callable[[dict], None] | None = None):
        self.run_id, self.policy, self.provider, self.model = run_id, policy, provider, model
        self.on_event = on_event
        self.events: list[dict] = []

    def emit(self, type_: str, summary: str, tool: str | None = None, args: dict | None = None,
             data: dict | None = None, duration_ms: int | None = None, ok: bool = True) -> dict:
        ev = {
            "ts_utc": iso_z(now_utc()), "run_id": self.run_id, "seq": len(self.events) + 1, "type": type_,
            "policy": self.policy, "provider": self.provider, "model": self.model, "tool": tool,
            "args": _jsonable(args) if args is not None else None, "summary": summary,
            "data": _jsonable(data) if data is not None else None, "duration_ms": duration_ms, "ok": ok,
        }
        self.events.append(ev)
        if self.on_event is not None:
            try:
                self.on_event(ev)
            except Exception:  # noqa: BLE001 - a broken UI callback must never break the run
                pass
        return ev

    def switch_policy(self, policy: str, provider: str = "none", model: str | None = None) -> None:
        self.policy, self.provider, self.model = policy, provider, model

    def write(self, path: Path) -> None:
        path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in self.events) + "\n", encoding="utf-8")


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = int((time.perf_counter() - self.t0) * 1000)
        return False
