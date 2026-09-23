"""Minimal placeholder UI (S0). The full UI is built by the UI engineer against docs/CONTRACTS.md."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from windagent import api  # noqa: E402

st.set_page_config(page_title="WindAgent", layout="wide")
st.title("WindAgent — Agentic AI прогноз выработки ВЭС")
try:
    for site in api.list_sites():
        info = api.site_info(site)
        st.write(f"**{info['name']}** — турбины: {', '.join(t['id'] for t in info['turbines'])}, "
                 f"время данных UTC+{info['data_clock_utc_offset_h']}")
except api.WindAgentError as exc:
    st.error(exc.user_message_ru)
st.info("Интерфейс в разработке. / UI under construction.")
