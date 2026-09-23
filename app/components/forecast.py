"""Forecast presentation only: no domain model or direct output-file access."""
from __future__ import annotations
from datetime import timedelta, timezone
import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app.i18n import tr
from app.services import call_api
from app.presentation import human_reason, model_name, source_label, version_kpis, version_label, version_record

TEAL = "#007F73"
INK = "#142A32"
COLORS = [TEAL, "#D49743", "#6087B0", "#8C789B"]

def display_times(values, offset: int) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce").dt.tz_convert(timezone(timedelta(hours=int(offset)))).dt.strftime("%Y-%m-%d %H:%M")

def styled(fig: go.Figure, clock_label: str = "", height: int = 385) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial, sans-serif", color=INK, size=12),
        margin=dict(l=12, r=20, t=38, b=110), hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, x=0),
        xaxis=dict(title=clock_label, showgrid=False, type="date", tickformat="%d.%m<br>%H:%M"), yaxis=dict(gridcolor="#DFE8E4", zeroline=False))
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
                             hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>"))
    if show_turbines:
        for idx, (entity, rows) in enumerate(frame[frame["entity"] != "farm"].groupby("entity")):
            rows = hourly_rows(rows)
            fig.add_trace(go.Scatter(x=display_times(rows["target_time_utc"], offset), y=rows["mean"], name=turbine_label(entity, lang),
                line=dict(color=COLORS[(idx + 1) % len(COLORS)], width=1.2, dash="dash"), hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>"))
    if not actuals.empty and {"entity", "actual", "target_time_utc", "available"}.issubset(actuals.columns):
        valid = hourly_rows(actuals[actuals["entity"] == "farm"])
        valid["actual"] = valid["actual"].where(valid["available"].eq(True))
        if not valid.empty:
            fig.add_trace(go.Scatter(x=display_times(valid["target_time_utc"], offset), y=valid["actual"],
                name=tr("actuals", lang), line=dict(color="#000000", width=2), hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>"))
    if "product" in farm:
        for product, color in (("intraday", "rgba(0,127,115,0.035)"), ("day_ahead", "rgba(212,151,67,0.075)")):
            rows = farm[farm["product"] == product]
            if not rows.empty:
                bounds = pd.Series([pd.to_datetime(rows["target_time_utc"], utc=True).min(),
                    pd.to_datetime(rows["target_time_utc"], utc=True).max() + pd.Timedelta(hours=1)])
                dates = display_times(bounds, offset)
                fig.add_vrect(x0=dates.iloc[0], x1=dates.iloc[1], fillcolor=color, opacity=1, line_width=0,
                    layer="below", annotation_text=tr(product, lang), annotation_position="top left", annotation_font_size=10)
    fig.update_yaxes(title=tr("power", lang), range=[0, 1], tickformat=".0%")
    return styled(fig, clock_label)

def render_kpis(meta: dict, lang: str, version=None) -> None:
    kpis = version_kpis(meta, version)
    keys = ["capacity_factor", "energy_norm_h", "max_ramp_3h", "mean_band_width"]
    # Two columns remain readable alongside an open sidebar on a laptop.
    for start in (0, 2):
        for column, key in zip(st.columns(2), keys[start:start + 2]):
            raw = pd.to_numeric(kpis.get(key), errors="coerce")
            value = "n/a"
            valid = pd.notna(raw) and math.isfinite(raw) and raw >= 0 and (key == "energy_norm_h" or raw <= 1)
            if valid:
                value = f"{raw:.1f} " + ("h" if lang == "EN" else "ч") if key == "energy_norm_h" else f"{raw * 100:.1f}%"
                if key == "mean_band_width":
                    value = f"±{raw * 50:.1f}%"
            column.metric(tr(key, lang), value, help=tr(key + "_help", lang))


def turbine_label(entity, lang: str) -> str:
    name = str(entity)
    return tr("turbine", lang, number=name[1:]) if name.startswith("t") and name[1:].isdigit() else name


def readable_hourly(frame: pd.DataFrame, lang: str, offset: int) -> pd.DataFrame:
    farm = frame.loc[frame["entity"].eq("farm")].sort_values("target_time_utc")
    result = pd.DataFrame(index=farm.index)
    result[tr("table_time", lang)] = display_times(farm["target_time_utc"], offset)
    result[tr("table_forecast", lang)] = pd.to_numeric(farm["mean"], errors="coerce") * 100
    def band(row):
        low, high = pd.to_numeric(row.get("p10"), errors="coerce"), pd.to_numeric(row.get("p90"), errors="coerce")
        return f"{low * 100:.0f}–{high * 100:.0f}" if pd.notna(low) and pd.notna(high) else "n/a"
    result[tr("table_band", lang)] = farm.apply(band, axis=1)
    for entity, rows in frame.loc[~frame["entity"].eq("farm")].groupby("entity", sort=True):
        unique = rows.loc[~rows["target_time_utc"].duplicated(keep=False)].set_index("target_time_utc")["mean"]
        result[turbine_label(entity, lang) + ", %"] = pd.to_numeric(farm["target_time_utc"].map(unique), errors="coerce") * 100
    products = {"intraday": tr("today", lang), "day_ahead": tr("tomorrow", lang), "extended": tr("extended", lang)}
    result[tr("table_part", lang)] = farm.get("product", pd.Series(index=farm.index, dtype=str)).map(products).fillna("n/a")
    return result.reset_index(drop=True)

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
    weather = weather.loc[weather["model"].notna() & weather["model"].astype(str).str.strip().ne("")].copy()
    if weather.empty:
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
        run_text = "n/a"
        if len(values) > 1 and pd.notna(values[1]):
            run_text = display_times(pd.Series([values[1]]), offset).iloc[0]
            inits = pd.to_datetime(weather.loc[weather["model"].eq(model), "init_time_utc"], utc=True, errors="coerce").dropna()
            bounds = display_times(pd.Series([inits.min(), inits.max()]), offset)
            suffix = " · " + (bounds.iloc[0] if bounds.iloc[0] == bounds.iloc[1] else f"{bounds.iloc[0]} – {bounds.iloc[1]}")
        fig.add_trace(go.Scatter(x=display_times(rows["target_time_utc"], offset), y=rows["ws100"], name=model_name(model, lang) + suffix,
            legendgroup=model, showlegend=model not in shown,
            text=[f"{model_name(model, lang)} · {run_text} · {clock_label}"] * len(rows),
            hovertemplate="%{y:.1f} m/s<br>%{text}<extra></extra>",
            line=dict(width=2, color=COLORS[models.index(values[0]) % len(COLORS)], dash="dot" if model in shown else "solid")))
        shown.add(model)
    fig.update_yaxes(title=tr("wind_speed", lang), rangemode="tozero")
    fig = styled(fig, clock_label, 440)
    fig.update_layout(margin=dict(b=170), legend=dict(font=dict(size=10), y=-0.3))
    st.plotly_chart(fig, width="stretch", key=key)

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
    render_kpis(meta, lang, version)
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
        if show_turbines:
            turbines = selected.loc[~selected["entity"].eq("farm")]
            unique = turbines.loc[~turbines.duplicated(["target_time_utc", "entity"], keep=False)]
            wide = unique.pivot(index="target_time_utc", columns="entity", values="mean")
            if len(wide.columns) == 2:
                correlation = wide.iloc[:, 0].corr(wide.iloc[:, 1])
                if pd.notna(correlation):
                    st.caption(tr("turbine_correlation", lang, correlation=f"{correlation:.2f}"))
        if actuals.empty or "available" not in actuals or not actuals["available"].eq(True).any():
            targets = pd.to_datetime(selected["target_time_utc"], utc=True) + pd.Timedelta(hours=6)
            february = targets.dt.strftime("%Y-%m").eq("2026-02").all()
            st.caption(tr("no_actuals_february" if february else "no_actuals", lang))
    with st.container(border=True):
        weather = run.get("weather", pd.DataFrame())
        if "version" not in weather and not weather.empty:
            st.caption(tr("weather_legacy", lang))
        render_weather(select_weather_for_version(weather, version), lang, offset, clock_label, "weather_chart")
    with st.expander(tr("hourly_table", lang)):
        readable = readable_hourly(selected, lang, offset)
        st.dataframe(readable, width="stretch", hide_index=True,
            column_config={c: st.column_config.NumberColumn(format="%.1f") for c in readable.select_dtypes(include="number")})
    issue_id = str(meta.get("issue_id", "forecast")).replace(":", "-")
    st.download_button(tr("download_forecast", lang), csv_bytes(selected), file_name=f"{site}_{issue_id}_v{version}.csv",
        mime="text/csv", key="download_forecast")
    with st.expander(tr("provenance", lang)):
        record = version_record(meta, version)
        asof = record.get("as_of_utc")
        rendered = display_times(pd.Series([asof]), offset).iloc[0] if asof else "n/a"
        st.write(f"{tr('as_of', lang)}: {rendered} · {clock_label}")
        selected_weather = select_weather_for_version(run.get("weather", pd.DataFrame()), version)
        sources = selected_weather["source"].dropna().unique().tolist() if "source" in selected_weather else []
        source = sources[0] if len(sources) == 1 else "mixed" if sources else meta.get("weather_source")
        st.write(f"{tr('source', lang)}: {source_label(source, lang)}")
        used = record.get("models_used") or (selected_weather["model"].dropna().unique().tolist() if "model" in selected_weather else [])
        st.write(f"{tr('models', lang)}: {', '.join(model_name(m, lang) for m in used) or 'n/a'}")
        run_rows = []
        if {"model", "init_time_utc"}.issubset(selected_weather):
            for model, rows in selected_weather.groupby("model"):
                valid_times = pd.to_datetime(rows["init_time_utc"], utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
                times = display_times(valid_times, offset).dropna().tolist()
                run_rows.append({tr("weather_model", lang): model_name(model, lang), tr("weather_run", lang): ", ".join(times) or "n/a"})
        else:
            for model, timestamps in (record.get("runs_used") or {}).items():
                times = display_times(pd.Series(timestamps), offset).dropna().tolist()
                run_rows.append({tr("weather_model", lang): model_name(model, lang), tr("weather_run", lang): ", ".join(times) or "n/a"})
        if run_rows:
            st.caption(clock_label)
            st.dataframe(pd.DataFrame(run_rows), hide_index=True, width="stretch")
        st.caption(tr("version_reason", lang, reason=human_reason(record.get("reason"), lang)))

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
        fig.add_trace(go.Scatter(x=display_times(rows["target_time_utc"], offset), y=rows["mean"], name=version_label(run.get("meta") or {}, version, lang),
            line=dict(color=color, width=2.5, dash="dot" if version == 1 else "solid")))
    fig.update_yaxes(title=tr("power", lang), range=[0, 1], tickformat=".0%")
    st.plotly_chart(styled(fig, clock_label, 400), width="stretch", key="agent_revisions")
