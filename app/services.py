"""API boundary, secrets and caching. No output-file parsing in the UI."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from windagent import api
from app.i18n import tr

ROOT = Path(__file__).resolve().parents[1]
SECRET_NAMES = ("OPENAI_API_KEY", "OPENAI_MODEL", "NVIDIA_API_KEY", "NVIDIA_MODEL", "LLM_PROVIDER")


def sync_secrets() -> None:
    try:
        for name in SECRET_NAMES:
            if name in st.secrets:
                os.environ[name] = str(st.secrets[name])
    except (FileNotFoundError, KeyError, OSError, st.errors.StreamlitSecretNotFoundError):
        pass


@st.cache_data(ttl=60, show_spinner=False)
def _cached_load(name: str, args: tuple, outputs: str, data: str, config: str, offline: str):
    # Root paths are cache keys so changing fixture / real directories cannot mix data.
    sync_secrets()
    return getattr(api, name)(*args)


def call_api(name: str, *args, lang: str = "RU", cached: bool = True, default=None, quiet: bool = False, **kwargs):
    sync_secrets()
    try:
        if cached and not kwargs:
            return _cached_load(name, args, os.environ.get("WINDAGENT_OUTPUTS_DIR", str(ROOT / "outputs")),
                                os.environ.get("WINDAGENT_DATA_DIR", str(ROOT / "data")),
                                os.environ.get("WINDAGENT_CONFIG", str(ROOT / "config" / "sites.yaml")),
                                os.environ.get("WINDAGENT_OFFLINE", "0"))
        return getattr(api, name)(*args, **kwargs)
    except api.WindAgentError as exc:
        if not quiet:
            message = exc.user_message_en if lang == "EN" else exc.user_message_ru
            (st.info if isinstance(exc, api.DataUnavailableError) else st.error)(message)
    except Exception:
        # Unexpected backend failures must not expose stack traces or credentials.
        if not quiet:
            st.error(tr("unexpected_error", lang))
    return default


def clear_loaders() -> None:
    _cached_load.clear()
