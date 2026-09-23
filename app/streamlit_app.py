"""WindAgent dashboard. Run: streamlit run app/streamlit_app.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from app.components.agent import render_agent
from app.components.analytics import render_how_it_works, render_test_period, render_validation
from app.components.forecast import display_times, render_forecast
from app.i18n import tr
from app.services import call_api, clear_loaders
from app.widgets import option_radio, option_selectbox

st.set_page_config(page_title="WindAgent", page_icon="◌", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
  .stApp {background:#F7F9F8;}
  .block-container {padding-top:4rem;padding-bottom:3rem;padding-left:2.2rem;padding-right:2.2rem;max-width:1480px;}
  [data-testid="stSidebar"] {background:#EDF2F0;border-right:1px solid #DBE5DF;}
  [data-testid="stSidebar"] .block-container {padding-top:2rem;}
  h1 {letter-spacing:-1.4px;font-weight:650!important;font-size:2.5rem!important;}
  h2,h3 {letter-spacing:-.45px;}
  [data-testid="stMetric"] {background:#fff;padding:18px 20px;border:1px solid #DDE7E2;border-radius:12px;}
  [data-testid="stMetricValue"] {font-size:clamp(1rem,2.2vw,1.75rem);font-weight:600;}
  [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {white-space:normal!important;overflow:visible!important;text-overflow:clip!important;overflow-wrap:anywhere;}
  [data-testid="stMetricLabel"] {color:#60766E;font-size:.8rem;}
  [data-testid="stMetricLabel"] p {white-space:normal;overflow:visible;text-overflow:clip;min-height:2.6em;}
  [data-testid="stVerticalBlockBorderWrapper"] {border-radius:12px;}
  button[data-baseweb="tab"] {font-weight:600;padding:12px 18px;}
  [data-baseweb="tab-list"] {gap:4px;border-bottom:1px solid #DDE7E2;}
  .brand {font-size:26px;font-weight:750;letter-spacing:-1px;color:#142A32;margin-bottom:4px;}
  .brand span {color:#007F73;}
  @media (max-width:800px) {.block-container {padding:1rem;} h1 {font-size:1.9rem!important;}}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand">◌ Wind<span>Agent</span></div>', unsafe_allow_html=True)
    lang = st.selectbox(tr("language"), ["RU", "EN"], key="language")
    st.caption(tr("workspace", lang))
    sites = call_api("list_sites", lang=lang, default=[])
    if not sites:
        st.info(tr("no_sites", lang))
        st.stop()
    site = st.selectbox(tr("site", lang), sites, key="site")
    info = call_api("site_info", site, lang=lang, default={})
    clocks = {"data": int(info.get("data_clock_utc_offset_h", 6)), "official": int(info.get("official_utc_offset_h", 5)), "utc": 0}
    labels = {"data": tr("clock_data", lang, offset=clocks["data"]),
              "official": tr("clock_official", lang, offset=clocks["official"]), "utc": tr("clock_utc", lang)}
    clock = option_selectbox(tr("clock", lang), list(clocks), format_func=lambda c: labels[c], key="display_clock")
    llm = call_api("llm_status", lang=lang, cached=False, default={})
    st.divider()
    if llm.get("configured"):
        st.caption(tr("llm_badge", lang, provider=llm.get("provider", "n/a"), model=llm.get("model") or "n/a"))
    else:
        st.caption(tr("rules_badge", lang))
    st.caption(tr("sidebar_note", lang))
    if st.button(tr("refresh", lang), key="refresh"):
        clear_loaders()
        st.rerun()

offset, clock_label = clocks[clock], labels[clock]
system = call_api("system_info", lang=lang, default={})
st.caption(f"{info.get('name', site)}  /  {clock_label}")
st.title(tr("title", lang))
st.caption(tr("subtitle", lang))
tabs = st.tabs([tr(key, lang) for key in ("tab_forecast", "tab_agent", "tab_validation", "tab_test", "tab_how")])

selected_run = None
with tabs[0]:
    st.subheader(tr("forecast_heading", lang))
    st.caption(tr("forecast_note", lang))
    kind = option_radio(tr("run_kind", lang), ["runs", "live"], horizontal=True,
                    format_func=lambda value: tr("historical" if value == "runs" else "live", lang), key="run_kind")
    runs = call_api("list_runs", site, kind, lang=lang, default=pd.DataFrame())
    if runs.empty:
        st.info(tr("no_runs", lang))
        st.code(tr("setup_command", lang), language="bash")
        st.caption(tr("offline_note", lang))
    else:
        ordered = runs.sort_values("issue_time_utc", ascending=False)
        issues = ordered["issue_id"].tolist()
        formatted = dict(zip(issues, display_times(ordered["issue_time_utc"], offset)))
        c1, c2, c3 = st.columns([3, 1, 2])
        issue_id = option_selectbox(tr("issue", lang), issues, format_func=lambda value: f"{formatted[value]} · {clock_label}", key=f"issue_{site}_{kind}", ui=c1)
        with st.spinner(tr("loading", lang)):
            selected_run = call_api("load_run", site, issue_id, kind, lang=lang)
        if selected_run:
            forecast = selected_run.get("forecast", pd.DataFrame())
            versions = sorted(forecast["version"].dropna().unique().tolist(), reverse=True) if "version" in forecast else []
            if versions:
                version = option_selectbox(tr("version", lang), versions, format_func=lambda v: f"v{v}", key=f"version_{site}_{issue_id}", ui=c2)
                show_turbines = c3.toggle(tr("show_turbines", lang), key="show_turbines")
                render_forecast(selected_run, site, version, lang, offset, clock_label, show_turbines)
            else:
                st.info(tr("incomplete", lang))
with tabs[1]:
    render_agent(site, info, system, selected_run, lang, offset, clock_label)
with tabs[2]:
    metrics = call_api("load_metrics", site, lang=lang, default={})
    render_validation(metrics, lang)
with tabs[3]:
    submission = call_api("load_submission", site, lang=lang, default=pd.DataFrame(), quiet=True)
    all_issues = call_api("load_all_issues", site, lang=lang, default=pd.DataFrame(), quiet=True)
    if (metrics or {}).get("synthetic_fixture") or (selected_run or {}).get("meta", {}).get("synthetic_fixture"):
        st.warning(tr("synthetic", lang))
    render_test_period(submission, all_issues, lang, offset, clock_label)
with tabs[4]:
    render_how_it_works(system, info, lang)
