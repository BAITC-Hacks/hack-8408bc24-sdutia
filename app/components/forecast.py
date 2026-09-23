"""Forecast presentation only: no domain model or direct output-file access."""
from __future__ import annotations
from datetime import timedelta, timezone
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app.i18n import tr
from app.services import call_api

TEAL = "#007F73"
INK = "#142A32"
COLORS = [TEAL, "#D49743", "#6087B0", "#8C789B"]

def display_times(values, offset: int) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce").dt.tz_convert(timezone(timedelta(hours=int(offset)))).dt.strftime("%Y-%m-%d %H:%M")

def styled(fig: go.Figure, clock_label: str = "", height: int = 385) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial, sans-serif", color=INK, size=12),
        margin=dict(l=12, r=20, t=38, b=35), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0),
        xaxis=dict(title=clock_label, showgrid=False, type="date"), yaxis=dict(gridcolor="#DFE8E4", zeroline=False))
    return fig

def csv_bytes(frame: pd.DataFrame) -> bytes:
    export = frame.copy()
    for col in export.columns:
        if col.endswith("_utc"):
            export[col] = pd.to_datetime(export[col], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return export.to_csv(index=False).encode("utf-8-sig")

def number(value) -> str:
    try:
        return f"{float(value):.3f}" if pd.notna(value) else "n/a"
    except (TypeError, ValueError):
        return "n/a"

def hourly_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Insert missing hours for a chart; do not interpolate or invent values."""
    rows = frame.copy()
    rows["target_time_utc"] = pd.to_datetime(rows["target_time_utc"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["target_time_utc"]).sort_values("target_time_utc")
    if rows.empty:
        return rows
    rows = rows.loc[~rows["target_time_utc"].duplicated(keep=False)]
    if rows.empty:
        return rows
    index = pd.date_range(rows["target_time_utc"].min(), rows["target_time_utc"].max(), freq="h")
    return rows.set_index("target_time_utc").reindex(index).rename_axis("target_time_utc").reset_index()

def forecast_figure(frame: pd.DataFrame, actuals: pd.DataFrame, lang: str, offset: int,
                    clock_label: str, show_turbines: bool = False) -> go.Figure:
    fig = go.Figure()
    farm = hourly_rows(frame[frame["entity"] == "farm"])
    if farm.empty:
        return styled(fig, clock_label)
    x = display_times(farm["target_time_utc"], offset)
    if {"p10", "p90"}.issubset(farm.columns):
        fig.add_trace(go.Scatter(x=x, y=farm["p90"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=farm["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(0,127,115,0.14)", name=tr("band", lang), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=farm["mean"], name=tr("farm", lang), line=dict(color=TEAL, width=3),
                             hovertemplate="%{y:.3f}<extra>%{fullData.name}</extra>"))
    if show_turbines:
        for idx, (entity, rows) in enumerate(frame[frame["entity"] != "farm"].groupby("entity")):
            rows = hourly_rows(rows)
            fig.add_trace(go.Scatter(x=display_times(rows["target_time_utc"], offset), y=rows["mean"], name=str(entity),
                line=dict(color=COLORS[(idx + 1) % len(COLORS)], width=1.5), hovertemplate="%{y:.3f}<extra>%{fullData.name}</extra>"))
    if not actuals.empty and {"entity", "actual", "target_time_utc", "available"}.issubset(actuals.columns):
        valid = hourly_rows(actuals[actuals["entity"] == "farm"])
        valid["actual"] = valid["actual"].where(valid["available"].eq(True))
        if not valid.empty:
            fig.add_trace(go.Scatter(x=display_times(valid["target_time_utc"], offset), y=valid["actual"],
                name=tr("actuals", lang), line=dict(color=INK, width=2, dash="dot")))
    if "product" in farm:
        for product, color in (("intraday", "rgba(0,127,115,0.035)"), ("day_ahead", "rgba(212,151,67,0.075)")):
            rows = farm[farm["product"] == product]
            if not rows.empty:
                bounds = pd.Series([pd.to_datetime(rows["target_time_utc"], utc=True).min(),
                    pd.to_datetime(rows["target_time_utc"], utc=True).max() + pd.Timedelta(hours=1)])
                dates = display_times(bounds, offset)
                fig.add_vrect(x0=dates.iloc[0], x1=dates.iloc[1], fillcolor=color, opacity=1, line_width=0,
                    layer="below", annotation_text=tr(product, lang), annotation_position="top left", annotation_font_size=10)
    fig.update_yaxes(title=tr("power", lang), range=[-0.02, 1.05], tickformat=".1f")
    return styled(fig, clock_label)

def render_kpis(meta: dict, lang: str) -> None:
    keys = ["energy_norm_h", "capacity_factor", "max_ramp_3h", "mean_band_width"]
    for column, key in zip(st.columns(4), keys):
        column.metric(tr(key, lang), number((meta.get("kpis") or {}).get(key)))
    st.caption(tr("kpi_scope", lang))

def select_weather_for_version(weather: pd.DataFrame, version: int) -> pd.DataFrame:
    """Honor explicit version association; older files remain readable."""
    if "version" not in weather:
        return weather.copy()
    return weather.loc[pd.to_numeric(weather["version"], errors="coerce").eq(version)].copy()

def render_weather(weather: pd.DataFrame, lang: str, offset: int, clock_label: str, key: str) -> None:
    st.subheader(tr("weather", lang))
    st.caption(tr("weather_note", lang))
    if weather.empty or not {"target_time_utc", "model", "ws100"}.issubset(weather.columns):
        st.info(tr("incomplete", lang))
        return
    fig = go.Figure()
    groups = ["model", "init_time_utc"] if "init_time_utc" in weather else ["model"]
    models = list(weather["model"].dropna().unique())
    shown = set()
    for group, rows in weather.groupby(groups, dropna=False):
        values = group if isinstance(group, tuple) else (group,)
        model = str(values[0])
        rows = hourly_rows(rows.drop_duplicates())
        suffix = ""
        if len(values) > 1 and pd.notna(values[1]):
            suffix = " · " + str(pd.Timestamp(values[1]).strftime("%m-%d %H:%M UTC"))
        fig.add_trace(go.Scatter(x=display_times(rows["target_time_utc"], offset), y=rows["ws100"], name=model + suffix,
            legendgroup=model, showlegend=model not in shown,
            line=dict(width=2, color=COLORS[models.index(values[0]) % len(COLORS)], dash="dot" if model in shown else "solid")))
        shown.add(model)
    fig.update_yaxes(title=tr("wind_speed", lang), rangemode="tozero")
    st.plotly_chart(styled(fig, clock_label, 285), width="stretch", key=key)

def render_forecast(run: dict, site: str, version: int, lang: str, offset: int, clock_label: str,
                    show_turbines: bool = False) -> None:
    meta = run.get("meta") or {}
    if meta.get("synthetic_fixture"):
        st.warning(tr("synthetic", lang))
    frame = run.get("forecast", pd.DataFrame())
    if frame.empty or not {"entity", "mean", "version", "target_time_utc"}.issubset(frame.columns):
        st.info(tr("incomplete", lang))
        return
    selected = frame[frame["version"] == version].copy()
    if selected.empty:
        st.info(tr("incomplete", lang))
        return
    render_kpis(meta, lang)
    if "version_as_of_utc" in selected:
        asofs = pd.to_datetime(selected["version_as_of_utc"], utc=True, errors="coerce").dropna().unique()
        if len(asofs) == 1:
            st.caption(tr("version_scope", lang, version=version,
                time=display_times(pd.Series(asofs), offset).iloc[0], clock=clock_label,
                hours=selected.loc[selected["entity"].eq("farm"), "target_time_utc"].nunique()))
    with st.container(border=True):
        actuals = call_api("load_actuals", site, pd.to_datetime(selected["target_time_utc"], utc=True).min(),
            pd.to_datetime(selected["target_time_utc"], utc=True).max(), lang=lang, default=pd.DataFrame(), quiet=True)
        st.plotly_chart(forecast_figure(selected, actuals, lang, offset, clock_label, show_turbines), width="stretch", key="forecast_chart")
        if actuals.empty or "available" not in actuals or not actuals["available"].eq(True).any():
            st.caption(tr("no_actuals", lang))
    with st.container(border=True):
        weather = run.get("weather", pd.DataFrame())
        if "version" not in weather and not weather.empty:
            st.caption(tr("weather_legacy", lang))
        render_weather(select_weather_for_version(weather, version), lang, offset, clock_label, "weather_chart")
    with st.expander(tr("hourly_table", lang)):
        st.dataframe(selected, width="stretch", hide_index=True,
            column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("mean", "p10", "p50", "p90", "mw_mean") if c in selected})
    issue_id = str(meta.get("issue_id", "forecast")).replace(":", "-")
    st.download_button(tr("download_forecast", lang), csv_bytes(selected), file_name=f"{site}_{issue_id}_v{version}.csv",
        mime="text/csv", key="download_forecast")
    with st.expander(tr("provenance", lang)):
        record = next((v for v in meta.get("versions", []) if v.get("version") == version), {})
        asof = record.get("as_of_utc")
        rendered = display_times(pd.Series([asof]), offset).iloc[0] if asof else "n/a"
        st.write(f"{tr('as_of', lang)}: {rendered} · {clock_label}")
        st.write(f"{tr('source', lang)}: {meta.get('weather_source', 'n/a')}")
        st.write(f"{tr('models', lang)}: {', '.join(meta.get('models_used') or []) or 'n/a'}")

def render_revisions(run: dict, lang: str, offset: int, clock_label: str) -> None:
    frame = run.get("forecast", pd.DataFrame())
    if frame.empty or not {"version", "entity", "target_time_utc", "mean"}.issubset(frame):
        st.info(tr("no_revision", lang))
        return
    farm = frame[frame["entity"] == "farm"]
    if not {1, 2}.issubset(set(farm["version"])):
        st.info(tr("no_revision", lang))
        return
    fig = go.Figure()
    for version, color in ((1, "#A4B6B0"), (2, TEAL)):
        rows = hourly_rows(farm[farm["version"] == version])
        fig.add_trace(go.Scatter(x=display_times(rows["target_time_utc"], offset), y=rows["mean"], name=f"v{version}",
            line=dict(color=color, width=2.5, dash="dot" if version == 1 else "solid")))
    fig.update_yaxes(title=tr("power", lang), tickformat=".3f")
    st.plotly_chart(styled(fig, clock_label, 310), width="stretch", key="agent_revisions")
