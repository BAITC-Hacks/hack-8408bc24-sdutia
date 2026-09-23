"""Cross-process rolling-hour limit for potentially paid UI agent invocations."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from uuid import uuid4


def reserve_policy(policy: str, session_count: int, outputs_dir: Path, *, now: float | None = None) -> tuple[str, str | None]:
    """Reserve a slot before an auto/LLM call; fail closed to rules on I/O errors.

    Exclusive lock creation serializes workers. A crashed process may leave a lock;
    subsequent calls safely use rules instead of guessing whether it is live.
    """
    if policy == "rules":
        return policy, None
    if session_count >= 3:
        return "rules", "rate_session"
    timestamp = time.time() if now is None else now
    lock = outputs_dir / ".ratelimit.lock"
    state = outputs_dir / ".ratelimit.json"
    temp = outputs_dir / f".ratelimit-{uuid4().hex}.tmp"
    acquired = False
    try:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(10):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
                acquired = True
                break
            except FileExistsError:
                time.sleep(0.02)
        if not acquired:
            return "rules", "rate_unavailable"
        raw = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {"timestamps": []}
        values = raw["timestamps"]
        if not isinstance(values, list) or any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in values):
            return "rules", "rate_unavailable"
        recent = [t for t in values if t > timestamp - 3600]
        if len(recent) >= 30:
            return "rules", "rate_global"
        recent.append(timestamp)
        temp.write_text(json.dumps({"timestamps": recent}), encoding="utf-8")
        os.replace(temp, state)
        return policy, None
    except (OSError, ValueError, KeyError, TypeError):
        return "rules", "rate_unavailable"
    finally:
        if acquired:
            try:
                lock.unlink(missing_ok=True)
                temp.unlink(missing_ok=True)
            except OSError:
                pass
