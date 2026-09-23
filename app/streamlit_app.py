"""WindAgent dashboard. Run: streamlit run app/streamlit_app.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from app.components.agent import render_agent
from app.components.analytics import render_how_it_works, render_test_period, render_validation, render_validation_headlines
from app.components.forecast import display_times, render_forecast
from app.i18n import tr
from app.services import call_api, clear_loaders
from app.widgets import get_choice, set_choice, option_radio, option_selectbox
from app.presentation import default_issue, issue_label, human_reason, version_label, version_record

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
    site = option_selectbox(tr("site", lang), sites, key="site") if len(sites) > 1 else sites[0]
    info = call_api("site_info", site, lang=lang, default={})
    site_name = ("Shelek Wind Farm" if lang == "EN" else "ВЭС Шелек") if site == "shelek" else info.get("name", site)
    st.markdown("**" + tr("site_single", lang, name=site_name, count=len(info.get("turbines") or [])) + "**")
    clocks = {"data": int(info.get("data_clock_utc_offset_h", 6)), "official": int(info.get("official_utc_offset_h", 5)), "utc": 0}
    labels = {"data": tr("clock_data", lang, offset=clocks["data"]),
              "official": tr("clock_official", lang, offset=clocks["official"]), "utc": tr("clock_utc", lang)}
    clock = option_selectbox(tr("clock", lang), list(clocks), format_func=lambda c: labels[c], key="display_clock", help=tr("clock_help", lang))
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
st.caption(f"{site_name}  /  {clock_label}")
st.title(tr("title", lang))
st.caption(tr("subtitle", lang))
tabs = st.tabs([tr(key, lang) for key in ("tab_forecast", "tab_agent", "tab_validation", "tab_test", "tab_how")])

selected_run = None
metrics = call_api("load_metrics", site, lang=lang, default={})
with tabs[0]:
    first_view = not st.session_state.get("intro_seen", False)
    with st.expander(tr("intro_heading", lang), expanded=first_view):
        st.write(tr("intro_text", lang))
        render_validation_headlines(metrics, lang)
    st.session_state["intro_seen"] = True
    st.subheader(tr("forecast_heading", lang))
    st.caption(tr("forecast_note", lang))
    source_keys = {"runs": "historical", "adhoc": "adhoc", "live": "live"}
    kind = option_radio(tr("run_kind", lang), list(source_keys), horizontal=True,
                    format_func=lambda value: tr(source_keys[value], lang), key="run_kind")
    runs = call_api("list_runs", site, kind, lang=lang, default=pd.DataFrame())
    if runs.empty:
        st.info(tr("empty_" + kind if kind != "runs" else "no_runs", lang))
        if kind == "runs":
            st.code(tr("setup_command", lang), language="bash")
            st.caption(tr("offline_note", lang))
    else:
        ordered = runs.sort_values("issue_time_utc", ascending=False)
        issues = ordered["issue_id"].tolist()
        formatted = {row.issue_id: issue_label(row.issue_time_utc, lang, clocks["data"], system.get("horizon_h", 48)) for row in ordered.itertuples()}
        issue_key = f"issue_{site}_{kind}"
        if get_choice(issue_key) is None:
            set_choice(issue_key, default_issue(ordered, info) if kind == "runs" else issues[0])
        issue_id = option_selectbox(tr("issue", lang), issues, format_func=lambda value: formatted[value], key=issue_key)
        with st.spinner(tr("loading", lang)):
            selected_run = call_api("load_run", site, issue_id, kind, lang=lang)
        if selected_run:
            forecast = selected_run.get("forecast", pd.DataFrame())
            versions = sorted(forecast["version"].dropna().unique().tolist()) if "version" in forecast else []
            if versions:
                meta = selected_run.get("meta") or {}
                if len(versions) > 1:
                    version = option_radio(tr("version", lang), versions, format_func=lambda v: version_label(meta, v, lang, clocks["data"]), key=f"version_{site}_{issue_id}")
                else:
                    version = versions[0]
                    st.write(version_label(meta, version, lang, clocks["data"]))
                    st.info(tr("no_update_needed", lang))
                reason = version_record(meta, version).get("reason")
                if reason:
                    st.caption(tr("version_reason", lang, reason=human_reason(reason, lang)))
                show_turbines = st.toggle(tr("show_turbines", lang), key="show_turbines")
                render_forecast(selected_run, site, version, lang, offset, clock_label, show_turbines)
            else:
                st.info(tr("incomplete", lang))
with tabs[1]:
    render_agent(site, info, system, selected_run, lang, offset, clock_label)
with tabs[2]:
    render_validation(metrics, lang)
with tabs[3]:
    submission = call_api("load_submission", site, lang=lang, default=pd.DataFrame(), quiet=True)
    all_issues = call_api("load_all_issues", site, lang=lang, default=pd.DataFrame(), quiet=True)
    if (metrics or {}).get("synthetic_fixture") or (selected_run or {}).get("meta", {}).get("synthetic_fixture"):
        st.warning(tr("synthetic", lang))
    render_test_period(submission, all_issues, lang, offset, clock_label)
with tabs[4]:
    how_system = dict(system)
    asofs = pd.to_datetime([item.get("as_of_utc") for item in (selected_run or {}).get("meta", {}).get("versions", [])], utc=True, errors="coerce").dropna().sort_values()
    if len(asofs) > 1:
        how_system["update_delay_h"] = (asofs[1] - asofs[0]).total_seconds() / 3600
    render_how_it_works(how_system, info, lang)
