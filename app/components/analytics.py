"""Read-only analytical panels for API-provided data.

No API calls or output-file reads belong here. Aggregations are display-only;
forecast generation and evaluation remain the responsibility of windagent.api.
"""

from __future__ import annotations

from datetime import timedelta, timezone
import json
import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.copy_analytics import MONTHS, tr
from app.components.forecast import csv_bytes


INK = "#142a32"
TEAL = "#007f73"
AMBER = "#d49743"
COLORS = [TEAL, AMBER, "#709bba", "#9f8ab9"]
METHODS = ("model", "persistence", "climatology", "nwp_powercurve")
REVISION_KEYS = ["site", "issue_id", "issue_time_utc", "target_time_utc", "lead_h", "product", "entity"]


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _records(value: Any) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Any, digits: int = 3, percentage: bool = False) -> str:
    result = _number(value)
    if result is None:
        return "n/a"
    return f"{result:.1%}" if percentage else f"{result:.{digits}f}"


def _zone(offset: Any) -> timezone:
    """Use explicit fixed UTC offsets, never a regional timezone database."""
    number = _number(offset)
    return timezone(timedelta(hours=number if number is not None and abs(number) < 24 else 0))


def _styled(fig: go.Figure, *, x_title: str = "", y_title: str = "", height: int = 320) -> go.Figure:
    has_legend = any(trace.name and trace.showlegend is not False for trace in fig.data)
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font={"color": INK, "family": "Arial, sans-serif", "size": 12},
        colorway=COLORS, margin={"l": 12, "r": 14, "t": 20, "b": 125 if has_legend else 45},
        height=height, hovermode="x unified",
        legend={"orientation": "h", "y": -0.4, "yanchor": "top", "x": 0},
        xaxis_title=x_title, yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(20,42,50,0.08)", zeroline=False)
    return fig


def _chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def _empty(lang: str) -> None:
    st.caption(tr("missing_chart", lang))


def month_label(value: Any, lang: str = "RU") -> str:
    try:
        stamp = pd.Timestamp(str(value), tz=timezone.utc)
        if pd.isna(stamp):
            return "n/a"
        return f"{MONTHS['EN' if str(lang).upper() == 'EN' else 'RU'][stamp.month - 1]} {stamp.year}"
    except (TypeError, ValueError):
        return "n/a"


def validation_period(metrics: dict, lang: str = "RU") -> str:
    period = _mapping(metrics.get("validation"))
    folds = period.get("folds")
    months = sorted(str(value) for value in folds if month_label(value, lang) != "n/a") if isinstance(folds, list) else []
    if not months:
        for key in ("start_utc", "end_utc"):
            stamp = pd.to_datetime(period.get(key), utc=True, errors="coerce")
            if stamp is not None and not pd.isna(stamp):
                months.append(stamp.tz_convert(_zone(6)).strftime("%Y-%m"))
    names = [month_label(value, lang) for value in months]
    return (names[0] if names[0] == names[-1] else f"{names[0]} – {names[-1]}") if names else "n/a"


def coverage_summary(metrics: dict) -> dict:
    calibration = _mapping(metrics.get("interval_calibration"))
    original = _mapping(_mapping(metrics.get("intervals")).get("day_ahead"))
    def coverage(value):
        result = _number(value)
        return result if result is not None and 0 <= result <= 1 else None
    return {"calibrated": coverage(calibration.get("coverage_check_after")),
            "before_all": coverage(original.get("coverage_p10_p90")),
            "before_check": coverage(calibration.get("coverage_check_before")),
            "check_month": calibration.get("check_month"),
            "fitted_on_months": calibration.get("fitted_on_months")}


def render_validation_headlines(metrics: dict | None, lang: str = "RU") -> None:
    """Three compact evidence cards, reusable in the Forecast introduction."""
    metrics = _mapping(metrics)
    if not metrics:
        st.info(tr("validation_empty", lang))
        return
    if metrics.get("synthetic_fixture"):
        st.warning(tr("synthetic", lang))
    period = _mapping(metrics.get("validation"))
    st.caption(tr("validation_period_plain", lang, period=validation_period(metrics, lang), issues=_fmt(period.get("n_issues"), 0)))
    if isinstance(period.get("method"), str) and "walk-forward" in period["method"].lower():
        st.caption(tr("walk_forward", lang))
    ahead = _mapping(_mapping(metrics.get("by_product")).get("day_ahead"))
    coverage = coverage_summary(metrics)
    columns = st.columns(3)
    columns[0].metric(tr("headline_mae", lang), _fmt(_mapping(ahead.get("model")).get("mae"), percentage=True))
    columns[0].caption(tr("headline_baseline", lang, value=_fmt(_mapping(ahead.get("climatology")).get("mae"), percentage=True)))
    skill = _number(_mapping(metrics.get("skill_pct")).get("day_ahead_vs_climatology"))
    columns[1].metric(tr("headline_skill", lang), f"{skill:.1f}%" if skill is not None else "n/a")
    calibrated = coverage["calibrated"] is not None
    columns[2].metric(tr("headline_coverage" if calibrated else "headline_raw_coverage", lang),
                      _fmt(coverage["calibrated"] if calibrated else coverage["before_all"], percentage=True))
    if calibrated:
        columns[2].caption(tr("calibration_check", lang, month=month_label(coverage["check_month"], lang)))


def validation_table(metrics: dict, lang: str = "RU") -> pd.DataFrame:
    """Return presentation percentages; source metrics remain unchanged."""
    products = _mapping(metrics.get("by_product"))
    rows = []
    for product in ["day_ahead", "intraday"] + [key for key in products if key not in {"day_ahead", "intraday"}]:
        product_metrics = _mapping(products.get(product))
        methods = list(METHODS)
        if "model_with_nowcast" in product_metrics:
            methods.insert(1, "model_with_nowcast")
        model_mae = _number(_mapping(product_metrics.get("model")).get("mae"))
        for method in methods:
            values = _mapping(product_metrics.get(method))
            mae = _number(values.get("mae"))
            skill = (1 - model_mae / mae) * 100 if method not in {"model", "model_with_nowcast"} and mae is not None and mae > 0 and model_mae is not None else None
            row = {tr("product", lang): tr(product, lang), tr("method", lang): tr(method, lang)}
            for key in ("mae", "rmse", "bias"):
                value = _number(values.get(key))
                row[tr(f"table_{key}", lang)] = f"{value * 100:.1f}" if value is not None else "n/a"
            row.update({"R²": _fmt(values.get("r2")), tr("table_n", lang): _fmt(values.get("n"), 0),
                        tr("table_skill", lang): f"{skill:.1f}%" if skill is not None else "n/a"})
            rows.append(row)
    return pd.DataFrame(rows)


def render_validation(metrics: dict | None, lang: str = "RU") -> None:
    """Display schema M with honest missing-data states."""
    metrics = _mapping(metrics)
    st.caption(tr("validation_intro", lang))
    if not metrics:
        st.info(tr("validation_empty", lang))
        return
    render_validation_headlines(metrics, lang)
    skills = _mapping(metrics.get("skill_pct"))
    skill_values = [_number(skills.get(key)) for key in ("day_ahead_vs_climatology", "day_ahead_vs_persistence")]
    if all(value is not None for value in skill_values):
        descriptions = [tr("lower" if value >= 0 else "higher", lang, value=f"{abs(value):.1f}") for value in skill_values]
        st.markdown(tr("skill_headline", lang, climate=descriptions[0], persistence=descriptions[1]))

    st.subheader(tr("comparison", lang))
    table = validation_table(metrics, lang)
    st.dataframe(table.style.apply(lambda row: ["background-color: #e6f3ee; font-weight: 600" if str(row[tr("method", lang)]).startswith("WindAgent") else "" for _ in row], axis=1), hide_index=True, width="stretch")
    st.caption(tr("table_skill_note", lang))
    for label in ("baseline_persistence", "baseline_climatology", "baseline_nwp"):
        st.caption(tr(label, lang))

    # Keep axis titles and legends distinct on a laptop with its sidebar open.
    lead_col, month_col = st.container(), st.container()
    with lead_col:
        st.subheader(tr("lead_title", lang))
        lead = pd.DataFrame(_records(metrics.get("by_lead")))
        fig = go.Figure()
        if "lead_h" in lead:
            lead["lead_h"] = pd.to_numeric(lead["lead_h"], errors="coerce")
            lead = lead.dropna(subset=["lead_h"]).sort_values("lead_h")
            for color, method in zip(COLORS, METHODS):
                if method in lead:
                    values = pd.to_numeric(lead[method], errors="coerce")
                    values = values.where(values.map(lambda value: _number(value) is not None))
                    if values.notna().any():
                        fig.add_trace(go.Scatter(x=lead["lead_h"], y=values, name=tr(f"chart_{method}", lang),
                                                 mode="lines", line={"color": color, "width": 2.5}, connectgaps=False,
                                                 hovertemplate="%{x} h · %{y:.1%}<extra>%{fullData.name}</extra>"))
        if fig.data:
            fig.update_yaxes(tickformat=".0%")
            _chart(_styled(fig, x_title=tr("lead_axis", lang), y_title=tr("mae_axis", lang)), "analytics_validation_lead")
        else:
            _empty(lang)
    with month_col:
        st.subheader(tr("month_title", lang))
        monthly = pd.DataFrame(_records(metrics.get("per_month")))
        fig = go.Figure()
        if "month" in monthly:
            monthly = monthly.dropna(subset=["month"]).sort_values("month")
            for color, method in zip(COLORS, METHODS):
                if method in monthly:
                    values = pd.to_numeric(monthly[method], errors="coerce")
                    values = values.where(values.map(lambda value: _number(value) is not None))
                    if values.notna().any():
                        fig.add_trace(go.Bar(x=monthly["month"], y=values, name=tr(f"chart_{method}", lang), marker_color=color,
                                            hovertemplate="%{x} · %{y:.1%}<extra>%{fullData.name}</extra>"))
        if fig.data:
            fig.update_layout(barmode="group")
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(tickvals=monthly["month"], ticktext=[month_label(m, lang) for m in monthly["month"]])
            _chart(_styled(fig, x_title=tr("month_axis", lang), y_title=tr("mae_axis", lang)), "analytics_validation_month")
        else:
            _empty(lang)

    st.subheader(tr("coverage_title", lang))
    coverage = coverage_summary(metrics)
    left, right = st.columns(2)
    left.metric(tr("headline_coverage", lang), _fmt(coverage["calibrated"], percentage=True))
    if coverage["check_month"]:
        left.caption(tr("calibration_check", lang, month=month_label(coverage["check_month"], lang)))
    left.caption(tr("coverage_before_check", lang, value=_fmt(coverage["before_check"], percentage=True)))
    right.metric(tr("headline_raw_coverage", lang), _fmt(coverage["before_all"], percentage=True))
    right.caption(tr("coverage_before_all", lang, value=_fmt(coverage["before_all"], percentage=True)))
    fitted = coverage["fitted_on_months"]
    if isinstance(fitted, list) and fitted:
        st.caption(tr("calibration_fit", lang, months=" – ".join([month_label(fitted[0], lang), month_label(fitted[-1], lang)])))
    st.caption(tr("coverage_target", lang))

    st.subheader(tr("nowcast_title", lang))
    intraday = _mapping(_mapping(metrics.get("by_product")).get("intraday"))
    st.metric(tr("nowcast_mae", lang), _fmt(_mapping(intraday.get("model_with_nowcast")).get("mae"), percentage=True))
    st.caption(tr("nowcast_note", lang, value=_fmt(_mapping(intraday.get("model")).get("mae"), percentage=True)))

    st.subheader(tr("loto_title", lang))
    loto = _mapping(metrics.get("leave_one_turbine_out"))
    columns = st.columns(4)
    for column, label, value in zip(columns, ("loto_train", "loto_test", "loto_mae", "loto_reference"),
                                    (str(loto.get("train") or "n/a"), str(loto.get("test") or "n/a"),
                                     _fmt(loto.get("mae"), percentage=True), _fmt(loto.get("mae_reference"), percentage=True))):
        column.metric(tr(label, lang), value)
    st.caption(tr("loto_note", lang))


def clean_submission(submission: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    """Keep unambiguous, finite hourly farm predictions; never modify the API frame."""
    if not isinstance(submission, pd.DataFrame) or submission.empty:
        return pd.DataFrame(), 0
    if not {"target_time_utc", "farm_mean"}.issubset(submission.columns):
        return pd.DataFrame(), len(submission)
    frame = submission.copy()
    frame["target_time_utc"] = pd.to_datetime(frame["target_time_utc"], utc=True, errors="coerce")
    frame["farm_mean"] = pd.to_numeric(frame["farm_mean"], errors="coerce")
    valid = frame["target_time_utc"].notna() & frame["farm_mean"].between(0, 1)
    valid &= frame["target_time_utc"].eq(frame["target_time_utc"].dt.floor("h"))
    valid &= ~frame["target_time_utc"].duplicated(keep=False)
    if "lead_h" in frame:
        lead = pd.to_numeric(frame["lead_h"], errors="coerce")
        valid &= lead.between(24, 47) & lead.eq(lead.round())
    return frame.loc[valid].sort_values("target_time_utc"), int((~valid).sum())


def daily_energy(submission: pd.DataFrame, clock_offset: float = 6) -> tuple[pd.DataFrame, int]:
    """Sum power for complete local days, after validating unique hourly samples."""
    frame, _ = clean_submission(submission)
    columns = ["day", "energy_norm_h"]
    if frame.empty:
        return pd.DataFrame(columns=columns), 0
    local = frame["target_time_utc"].dt.tz_convert(_zone(clock_offset))
    frame["day"] = local.dt.strftime("%Y-%m-%d")
    frame["hour"] = local.dt.hour
    rows, partial = [], 0
    for day, group in frame.groupby("day", sort=True):
        if len(group) == 24 and group["hour"].nunique() == 24:
            rows.append({"day": day, "energy_norm_h": float(group["farm_mean"].sum())})
        else:
            partial += 1
    return pd.DataFrame(rows, columns=columns), partial


def revision_pairs(all_issues: pd.DataFrame | None, entity: str = "farm", product: str | None = "day_ahead") -> tuple[pd.DataFrame, int]:
    """Pair exact v1/v2 forecasts without averaging duplicate or mismatched issues."""
    if not isinstance(all_issues, pd.DataFrame) or all_issues.empty:
        return pd.DataFrame(), 0
    if not set(REVISION_KEYS + ["version", "mean"]).issubset(all_issues.columns):
        return pd.DataFrame(), len(all_issues)
    frame = all_issues.copy()
    frame["version"] = pd.to_numeric(frame["version"], errors="coerce")
    products = frame["product"].eq(product) if product is not None else frame["product"].isin(["intraday", "day_ahead"])
    frame = frame.loc[frame["entity"].eq(entity) & products & frame["version"].isin([1, 2])].copy()
    if frame.empty:
        return pd.DataFrame(), 0
    initial_count = len(frame)
    for column in ("issue_time_utc", "target_time_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame["mean"] = pd.to_numeric(frame["mean"], errors="coerce")
    frame["lead_h"] = pd.to_numeric(frame["lead_h"], errors="coerce")
    valid = frame[REVISION_KEYS].notna().all(axis=1) & frame["mean"].between(0, 1)
    valid &= frame["lead_h"].between(24 if product == "day_ahead" else 0, 47) & frame["lead_h"].eq(frame["lead_h"].round())
    valid &= frame["target_time_utc"].eq(frame["target_time_utc"].dt.floor("h"))
    valid &= ~frame.duplicated(REVISION_KEYS + ["version"], keep=False)
    frame = frame.loc[valid]
    before = frame.loc[frame["version"].eq(1), REVISION_KEYS + ["mean"]].rename(columns={"mean": "v1"})
    extras = ["version_as_of_utc"] if "version_as_of_utc" in frame else []
    after = frame.loc[frame["version"].eq(2), REVISION_KEYS + ["mean"] + extras].rename(columns={"mean": "v2"})
    expected_elapsed = 0
    if product is None and extras:
        issue_keys = ["site", "issue_id", "issue_time_utc", "entity"]
        update_times = after[issue_keys + extras].copy()
        update_times["version_as_of_utc"] = pd.to_datetime(update_times["version_as_of_utc"], utc=True, errors="coerce")
        update_times = update_times.dropna(subset=extras).drop_duplicates()
        update_times = update_times.loc[~update_times.duplicated(issue_keys, keep=False)]
        candidates = before.merge(update_times, on=issue_keys, how="inner", validate="many_to_one")
        expected_elapsed = int(candidates["target_time_utc"].lt(candidates["version_as_of_utc"]).sum())
    pairs = before.merge(after, on=REVISION_KEYS, how="inner", validate="one_to_one")
    # Default preserves the target-day grid contract; the all-product view has
    # one row per issue and one column per lead hour, so overlapping issues stay separate.
    cell_keys = ["target_time_utc"] if product is not None else ["issue_id", "lead_h"]
    pairs = pairs.loc[~pairs.duplicated(cell_keys, keep=False)].copy()
    if "version_as_of_utc" in pairs:
        pairs["version_as_of_utc"] = pd.to_datetime(pairs["version_as_of_utc"], utc=True, errors="coerce")
        pairs = pairs.loc[pairs["version_as_of_utc"].notna() & pairs["target_time_utc"].ge(pairs["version_as_of_utc"])].copy()
    pairs["delta"] = pairs["v2"] - pairs["v1"]
    return pairs.sort_values("target_time_utc"), max(0, initial_count - 2 * len(pairs) - expected_elapsed)


def submission_summary(submission: pd.DataFrame | None, lang: str = "RU") -> dict:
    frame, _ = clean_submission(submission)
    if frame.empty:
        return {"period": "n/a", "issues": "n/a", "hours": 0}
    # The submission period is defined on the fixed data clock, independently
    # of the user's display clock (which may straddle adjacent calendar dates).
    local = frame["target_time_utc"].dt.tz_convert(_zone(6))
    months = sorted(local.dt.strftime("%Y-%m").unique())
    first, last = month_label(months[0], lang), month_label(months[-1], lang)
    return {"period": first if first == last else f"{first} – {last}",
            "issues": int(frame["issue_id"].dropna().nunique()) if "issue_id" in frame else "n/a", "hours": len(frame)}


def render_test_period(submission: pd.DataFrame | None, all_issues: pd.DataFrame | None,
                       lang: str = "RU", clock_offset: float = 6, clock_label: str = "UTC+6") -> None:
    """Render schema S plus exact day-ahead version revisions from schema F."""
    frame, invalid = clean_submission(submission)
    summary = submission_summary(submission, lang)
    if not frame.empty:
        st.markdown(tr("submission_headline", lang, **summary))
    st.caption(tr("submission_intro", lang))
    if isinstance(submission, pd.DataFrame) and not submission.empty:
        st.download_button(tr("download_submission_primary", lang, hours=len(submission)), csv_bytes(submission),
                           file_name="submission_day_ahead.csv", mime="text/csv", key="analytics_download_submission", type="primary")
    if invalid:
        st.warning(tr("submission_invalid", lang, count=invalid))
    if frame.empty:
        st.info(tr("submission_empty", lang))
    else:
        st.subheader(tr("submission_title", lang))
        # Reindexing inserts explicit gaps instead of drawing through absent hours.
        hourly_index = pd.date_range(frame["target_time_utc"].min(), frame["target_time_utc"].max(), freq="h")
        plot_frame = frame.set_index("target_time_utc").reindex(hourly_index)
        x = plot_frame.index.tz_convert(_zone(clock_offset)).strftime("%Y-%m-%d %H:%M")
        fig = go.Figure()
        if {"farm_p10", "farm_p50", "farm_p90"}.issubset(plot_frame.columns):
            quantiles = plot_frame[["farm_p10", "farm_p50", "farm_p90"]].apply(pd.to_numeric, errors="coerce")
            band_valid = quantiles.notna().all(axis=1) & quantiles.ge(0).all(axis=1) & quantiles.le(1).all(axis=1)
            band_valid &= quantiles["farm_p10"].le(quantiles["farm_p50"]) & quantiles["farm_p50"].le(quantiles["farm_p90"])
            fig.add_trace(go.Scatter(x=x, y=quantiles["farm_p90"].where(band_valid), mode="lines",
                                    line={"width": 0}, showlegend=False, hoverinfo="skip", connectgaps=False))
            fig.add_trace(go.Scatter(x=x, y=quantiles["farm_p10"].where(band_valid), mode="lines",
                                    line={"width": 0}, fill="tonexty", fillcolor="rgba(0,127,115,0.13)",
                                    name=tr("band", lang), hoverinfo="skip", connectgaps=False))
            if not band_valid.all():
                st.caption(tr("band_invalid", lang))
        else:
            st.caption(tr("band_invalid", lang))
        fig.add_trace(go.Scatter(x=x, y=plot_frame["farm_mean"], mode="lines", name=tr("farm", lang),
                                line={"color": TEAL, "width": 2.5}, connectgaps=False,
                                hovertemplate="%{x}<br>%{y:.1%}<extra></extra>"))
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        fig.update_xaxes(tickformat="%d.%m")
        _chart(_styled(fig, x_title=clock_label, y_title=tr("power_axis", lang), height=360), "analytics_submission")

        st.subheader(tr("energy_title_period", lang, period=summary["period"]))
        energy, partial = daily_energy(frame, clock_offset)
        st.caption(tr("energy_note", lang))
        if partial:
            st.caption(tr("energy_partial", lang, count=partial, clock=clock_label))
        if energy.empty:
            st.info(tr("energy_empty", lang))
        else:
            fig = go.Figure(go.Bar(x=energy["day"], y=energy["energy_norm_h"], marker_color=TEAL,
                                  hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"))
            fig.update_xaxes(tickformat="%d.%m")
            _chart(_styled(fig, x_title=tr("day_axis", lang), y_title=tr("energy_axis", lang), height=260), "analytics_energy")

    has_v2 = isinstance(all_issues, pd.DataFrame) and "version" in all_issues and pd.to_numeric(all_issues["version"], errors="coerce").eq(2).any()
    if has_v2:
        st.subheader(tr("revision_title", lang))
        entities = sorted(all_issues["entity"].dropna().astype(str).unique()) if "entity" in all_issues else []
        entity = "farm"
        if entities:
            entity = st.selectbox(tr("revision_entity", lang), entities, index=entities.index("farm") if "farm" in entities else 0,
                                  key="analytics_revision_entity")
        pairs, excluded = revision_pairs(all_issues, entity, product=None)
        st.caption(tr("revision_note", lang))
        st.caption(tr("revision_grid_note", lang))
        if not pairs.empty:
            local = pairs["issue_time_utc"].dt.tz_convert(_zone(clock_offset))
            pairs["issue_label"] = local.dt.strftime("%Y-%m-%d %H:%M")
            # Different issue identifiers must not produce one visible cell.
            duplicated = pairs.duplicated(["issue_label", "lead_h"], keep=False)
            excluded += 2 * int(duplicated.sum())
            pairs = pairs.loc[~duplicated].copy()
        if excluded:
            st.caption(tr("revision_invalid", lang, count=excluded))
        if pairs.empty:
            st.info(tr("revision_none", lang))
        else:
            matrix = pairs.pivot(index="issue_label", columns="lead_h", values="delta").reindex(columns=range(48))
            delay = None
            if "version_as_of_utc" in pairs:
                delays = ((pairs["version_as_of_utc"] - pairs["issue_time_utc"]) / pd.Timedelta(hours=1)).dropna().unique()
                if len(delays) == 1 and delays[0] >= 0:
                    delay = float(delays[0])
            st.caption(tr("revision_elapsed", lang, hours=f"{delay:g}") if delay is not None else tr("revision_elapsed_generic", lang))
            limit = max(float(pairs["delta"].abs().max()), 0.001)
            fig = go.Figure(go.Heatmap(x=list(matrix.columns), y=matrix.index,
                                      z=matrix.to_numpy(), zmin=-limit, zmax=limit, zmid=0,
                                      colorscale=[[0, "#397cb8"], [0.5, "#f8faf9"], [1, "#c75246"]],
                                      colorbar={"title": {"text": tr("revision_delta", lang)}, "tickformat": ".0%"},
                                      hoverongaps=False,
                                      hovertemplate=tr("revision_hover", lang)))
            fig = _styled(fig, x_title=tr("lead_axis", lang), y_title=tr("revision_issue_axis", lang, clock=clock_label),
                          height=max(240, min(650, 110 + 24 * len(matrix))))
            fig.update_layout(hovermode="closest", plot_bgcolor="#e5ece8")
            fig.update_yaxes(type="category", autorange="reversed")
            _chart(fig, "analytics_revisions")
    else:
        st.caption(tr("revision_none", lang))

    st.subheader(tr("downloads", lang))
    if isinstance(submission, pd.DataFrame) and not submission.empty:
        with st.expander(tr("table", lang)):
            st.dataframe(submission, width="stretch", hide_index=True,
                         column_config={c: st.column_config.NumberColumn(format="%.3f") for c in submission if c.endswith(("_mean", "_p10", "_p50", "_p90"))})
    if isinstance(all_issues, pd.DataFrame) and not all_issues.empty:
        st.download_button(tr("download_issues", lang), csv_bytes(all_issues),
                              file_name="all_issues.csv", mime="text/csv", key="analytics_download_issues")


def _dot_label(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_run_locally(lang: str = "RU", key: str = "analytics_run_locally") -> None:
    """README sections 7–8 commands, without running them or reading local files."""
    with st.container(key=key):
        st.subheader(tr("run_locally_title", lang))
        st.markdown(tr("run_locally_note", lang))
        st.caption(tr("run_context", lang))
        for label, command in (
            ("run_install", "pip install -r requirements.txt"),
            ("run_verify", "python -m windagent verify"),
            ("run_forecast", 'python -m windagent forecast --issue "2026-01-31 00:00" --policy rules --offline -v'),
            ("run_dashboard", "streamlit run app/streamlit_app.py"),
            ("run_docker", "docker compose up --build"),
        ):
            st.markdown(tr(label, lang))
            st.code(command, language="bash", wrap_lines=True)


def render_how_it_works(system: dict | None, site_info: dict | None, lang: str = "RU") -> None:
    """Describe contract-defined flow, with runtime facts exclusively from the API."""
    system, site_info = _mapping(system), _mapping(site_info)
    st.caption(tr("how_intro", lang))
    issue_hour = _number(system.get("issue_hour_data_clock"))
    hour_label = f"{int(issue_hour):02d}:00" if issue_hour is not None and issue_hour.is_integer() and 0 <= issue_hour < 24 else "n/a"
    delay = _number(system.get("update_delay_h"))
    steps = [("🌤", tr("how_step_weather", lang, hour=hour_label)),
             ("✓", tr("how_step_validate", lang)), ("↗", tr("how_step_model", lang)),
             ("▤", tr("how_step_publish", lang, hours=_fmt(system.get("horizon_h"), 0))),
             ("↻", tr("how_step_update_timed", lang, hours=f"{delay:g}") if delay is not None else tr("how_step_update", lang))]
    for index, (icon, label) in enumerate(steps, 1):
        st.markdown(f"**{icon} {index}. {label}**")
    st.caption(tr("how_clock", lang))
    st.subheader(tr("architecture", lang))
    labels = {name: _dot_label(tr(f"node_{name}", lang)) for name in ("scada", "weather", "guard", "features", "model", "agent", "outputs", "ui")}
    nodes = "\n".join(f'{name} [label={label}];' for name, label in labels.items())
    st.graphviz_chart('digraph { graph [rankdir=LR, bgcolor="transparent", pad="0.2"]; '
                      'node [shape=box, style="rounded,filled", fillcolor="#edf5f2", color="#bfd6cc", '
                      f'fontcolor="{INK}", fontname="Arial", fontsize=11, margin="0.12,0.08"]; '
                      f'edge [color="{TEAL}"]; {nodes} '
                      'scada -> features; weather -> guard; guard -> features; features -> model; '
                      'model -> agent; agent -> outputs; outputs -> ui; }', width="stretch")

    left, right = st.columns([1, 1.3])
    with left:
        st.subheader(tr("tools_title", lang))
        tools = _records(system.get("tools"))
        if tools:
            st.dataframe(pd.DataFrame([{tr("tool_name", lang): item.get("name") or "n/a",
                                        tr("tool_description", lang): item.get("description") or "n/a"} for item in tools]),
                         width="stretch", hide_index=True)
        else:
            _empty(lang)
    with right:
        st.subheader(tr("weather_title", lang))
        weather = _records(system.get("weather_models"))
        if weather:
            st.dataframe(pd.DataFrame([{tr("weather_model", lang): item.get("label") or item.get("id") or "n/a",
                                        tr("weather_id", lang): item.get("id") or "n/a",
                                        tr("latency", lang): _fmt(item.get("latency_h"), 1),
                                        tr("archive", lang): item.get("archive_from") or "n/a"} for item in weather]),
                         width="stretch", hide_index=True)
        else:
            _empty(lang)

    st.subheader(tr("asof_title", lang))
    st.markdown(tr("asof_rule", lang))
    st.graphviz_chart('digraph { graph [rankdir=LR, bgcolor="transparent"]; '
                      f'node [shape=box, style="rounded", color="{TEAL}", fontcolor="{INK}", fontname="Arial"]; '
                      f'edge [color="{AMBER}", fontcolor="{INK}", fontname="Arial"]; '
                      f'init [label={_dot_label(tr("run_start", lang))}]; '
                      f'published [label={_dot_label(tr("publication", lang))}]; '
                      f'issue [label={_dot_label(tr("issue", lang))}]; '
                      f'init -> published [label={_dot_label(tr("delay_label", lang))}]; '
                      f'published -> issue [label={_dot_label(tr("before_label", lang))}]; }}', width="stretch")

    st.subheader(tr("clock_title", lang))
    data_offset = _number(site_info.get("data_clock_utc_offset_h"))
    official_offset = _number(site_info.get("official_utc_offset_h"))
    if data_offset is not None and official_offset is not None and abs(data_offset) < 24 and abs(official_offset) < 24:
        st.caption(tr("clock_note", lang, data=data_offset, official=official_offset))
        issue_range = _mapping(site_info.get("issue_range_data_clock"))
        sample = issue_range.get("min")
        if not sample:
            start = _mapping(site_info.get("test_period")).get("start")
            hour = _number(system.get("issue_hour_data_clock"))
            if start and hour is not None and hour.is_integer() and 0 <= hour <= 23:
                sample = f"{start} {int(hour):02d}:00"
        try:
            instant = pd.Timestamp(sample, tz=_zone(data_offset)) if sample else None
        except (ValueError, TypeError):
            instant = None
        if instant is not None and not pd.isna(instant):
            st.caption(tr("clock_example", lang))
            st.dataframe(pd.DataFrame([{
                tr("clock_utc", lang): instant.tz_convert(timezone.utc).strftime("%Y-%m-%d %H:%M %z"),
                tr("clock_data", lang, offset=data_offset): instant.strftime("%Y-%m-%d %H:%M %z"),
                tr("clock_official", lang, offset=official_offset): instant.tz_convert(_zone(official_offset)).strftime("%Y-%m-%d %H:%M %z"),
            }]), width="stretch", hide_index=True)
        else:
            st.caption(tr("clock_missing", lang))
    else:
        st.caption(tr("clock_missing", lang))

    turbines = _records(site_info.get("turbines"))
    if turbines:
        with st.expander(tr("site_title", lang)):
            st.dataframe(pd.DataFrame([{tr("turbine", lang): item.get("id") or "n/a",
                                        tr("latitude", lang): _fmt(item.get("lat"), 5),
                                        tr("longitude", lang): _fmt(item.get("lon"), 5),
                                        tr("rated", lang): _fmt(item.get("rated_mw"))} for item in turbines]),
                         width="stretch", hide_index=True)
    st.subheader(tr("scaling_title", lang))
    st.markdown(tr("scaling_note", lang))
    render_run_locally(lang, key="how_run_locally")
