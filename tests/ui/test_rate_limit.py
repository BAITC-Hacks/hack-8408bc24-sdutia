"""Boundaries and safe failures for the UI's paid-call budget."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import rate_limit  # noqa: E402


def read_timestamps(outputs: Path) -> list[float]:
    return json.loads((outputs / ".ratelimit.json").read_text(encoding="utf-8"))["timestamps"]


def test_fourth_session_attempt_falls_back_without_reserving(tmp_path: Path):
    for count in range(3):
        assert rate_limit.reserve_policy("auto", count, tmp_path, now=10000) == ("auto", None)

    assert rate_limit.reserve_policy("llm", 3, tmp_path, now=10000) == ("rules", "rate_session")
    assert len(read_timestamps(tmp_path)) == 3


def test_thirty_first_global_attempt_uses_rules(tmp_path: Path):
    for _ in range(30):
        assert rate_limit.reserve_policy("llm", 0, tmp_path, now=10000) == ("llm", None)

    assert rate_limit.reserve_policy("auto", 0, tmp_path, now=10001) == ("rules", "rate_global")
    assert len(read_timestamps(tmp_path)) == 30


def test_rolling_hour_expires_exact_boundary(tmp_path: Path):
    state = tmp_path / ".ratelimit.json"
    state.write_text(json.dumps({"timestamps": [6400.0] * 29 + [6400.001]}), encoding="utf-8")

    assert rate_limit.reserve_policy("auto", 0, tmp_path, now=10000) == ("auto", None)
    assert read_timestamps(tmp_path) == [6400.001, 10000]


@pytest.mark.parametrize("contents", ["not json", "{}", "[]", '{"timestamps": true}',
                                     '{"timestamps": [true]}', '{"timestamps": ["10000"]}',
                                     '{"timestamps": [NaN]}', '{"timestamps": [Infinity]}'])
def test_corrupted_state_fails_closed_and_preserves_file(tmp_path: Path, contents: str):
    state = tmp_path / ".ratelimit.json"
    state.write_text(contents, encoding="utf-8")

    assert rate_limit.reserve_policy("llm", 0, tmp_path, now=10000) == ("rules", "rate_unavailable")
    assert state.read_text(encoding="utf-8") == contents
    assert not (tmp_path / ".ratelimit.lock").exists()


def test_storage_failure_uses_rules_and_releases_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fail_replace(*args, **kwargs):
        raise PermissionError("Read-only output storage")

    monkeypatch.setattr(rate_limit.os, "replace", fail_replace)

    assert rate_limit.reserve_policy("auto", 0, tmp_path, now=10000) == ("rules", "rate_unavailable")
    assert not list(tmp_path.iterdir())


def test_explicit_rules_does_not_create_or_change_state(tmp_path: Path):
    absent_outputs = tmp_path / "never_created"
    assert rate_limit.reserve_policy("rules", 99, absent_outputs, now=10000) == ("rules", None)
    assert not absent_outputs.exists()

    state = tmp_path / ".ratelimit.json"
    state.write_text("invalid untouched state", encoding="utf-8")
    assert rate_limit.reserve_policy("rules", 0, tmp_path, now=10000) == ("rules", None)
    assert state.read_text(encoding="utf-8") == "invalid untouched state"


def test_contending_workers_never_overbook_global_budget(tmp_path: Path):
    def reserve(_):
        return rate_limit.reserve_policy("auto", 0, tmp_path, now=10000)

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(reserve, range(40)))

    accepted = sum(policy == "auto" for policy, _ in outcomes)
    assert 1 <= accepted <= 30
    assert len(read_timestamps(tmp_path)) == accepted
    assert all(policy in {"auto", "rules"} for policy, _ in outcomes)
    assert not (tmp_path / ".ratelimit.lock").exists()
