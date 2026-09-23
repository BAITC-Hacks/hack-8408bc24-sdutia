"""Verify the UI API boundary does not expose backend failures or credentials."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import streamlit as st

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from app import services  # noqa: E402
from windagent import api  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_service_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(st, "secrets", {})
    for name in services.SECRET_NAMES:
        monkeypatch.delenv(name, raising=False)
    services.clear_loaders()
    yield
    services.clear_loaders()


@pytest.mark.parametrize("language, expected", [("RU", "Нет результатов"), ("EN", "No outputs")])
def test_missing_outputs_show_localized_guidance(monkeypatch: pytest.MonkeyPatch, language: str, expected: str):
    shown = []

    def unavailable(site):
        raise api.DataUnavailableError("No outputs", "Нет результатов")

    monkeypatch.setattr(api, "load_submission", unavailable)
    monkeypatch.setattr(st, "info", shown.append)

    assert services.call_api("load_submission", "shelek", lang=language, cached=False, default="empty") == "empty"
    assert shown == [expected]


def test_unknown_backend_error_does_not_display_sensitive_details(monkeypatch: pytest.MonkeyPatch):
    shown = []

    def fail(site):
        raise RuntimeError("private-token-DO-NOT-RENDER traceback details")

    monkeypatch.setattr(api, "load_metrics", fail)
    monkeypatch.setattr(st, "error", shown.append)

    assert services.call_api("load_metrics", "shelek", lang="EN", cached=False, default={}) == {}
    assert len(shown) == 1
    assert "private-token" not in shown[0]
    assert "traceback details" not in shown[0]


def test_secrets_are_copied_before_api_invocation(monkeypatch: pytest.MonkeyPatch):
    values = {name: f"test-only-{name}" for name in services.SECRET_NAMES}
    monkeypatch.setattr(st, "secrets", values)
    seen = []

    def fake_status():
        seen.append({name: os.environ.get(name) for name in values})
        return {"configured": True}

    monkeypatch.setattr(api, "llm_status", fake_status)

    assert services.call_api("llm_status", cached=False) == {"configured": True}
    assert seen == [values]


def test_missing_secrets_file_is_harmless(monkeypatch: pytest.MonkeyPatch):
    class MissingSecrets:
        def __contains__(self, key):
            raise st.errors.StreamlitSecretNotFoundError("No secrets file")

    monkeypatch.setattr(st, "secrets", MissingSecrets())
    monkeypatch.setattr(api, "llm_status", lambda: {"configured": False})

    assert services.call_api("llm_status", cached=False) == {"configured": False}


def test_loader_cache_separates_output_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = []

    def load_metrics(site):
        root = os.environ["WINDAGENT_OUTPUTS_DIR"]
        calls.append(root)
        return {"source_root": root}

    monkeypatch.setattr(api, "load_metrics", load_metrics)
    first_root = str(tmp_path / "fixtures")
    second_root = str(tmp_path / "real_outputs")
    monkeypatch.setenv("WINDAGENT_OUTPUTS_DIR", first_root)

    assert services.call_api("load_metrics", "shelek")["source_root"] == first_root
    assert services.call_api("load_metrics", "shelek")["source_root"] == first_root
    monkeypatch.setenv("WINDAGENT_OUTPUTS_DIR", second_root)
    assert services.call_api("load_metrics", "shelek")["source_root"] == second_root
    assert calls == [first_root, second_root]
