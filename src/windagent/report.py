"""Generated documentation blocks: metrics tables are written into the READMEs from the output files,
so no number in the README is typed by hand. Also `compare` for the CI reproducibility check."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

START, END = "<!-- METRICS:START -->", "<!-- METRICS:END -->"


def _load(site: str) -> dict:
    base = config.outputs_dir() / site
    m = json.loads((base / "validation" / "metrics.json").read_text(encoding="utf-8"))
    clock_p = base / "clock" / "clock_report.json"
    clock = json.loads(clock_p.read_text(encoding="utf-8")) if clock_p.exists() else {}
    runs = []
    for p in sorted((base / "runs").glob("*/run.json")):
        runs.append(json.loads(p.read_text(encoding="utf-8")))
    sub_p = base / "test_period" / "submission_day_ahead.csv"
    sub = pd.read_csv(sub_p, encoding="utf-8") if sub_p.exists() else pd.DataFrame()
    return {"m": m, "clock": clock, "runs": runs, "sub": sub}


def _fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def render(site: str = "shelek", lang: str = "ru") -> str:
    d = _load(site)
    m, clock, runs, sub = d["m"], d["clock"], d["runs"], d["sub"]
    ru = lang == "ru"
    names = {"model": "WindAgent (наша модель)" if ru else "WindAgent (our model)",
             "nwp_powercurve": "Кривая мощности по лучшей NWP-модели" if ru else "Power curve of the best single NWP model",
             "climatology": "Климатология (месяц × час)" if ru else "Climatology (month × hour)",
             "persistence": "Персистентность (последний час)" if ru else "Persistence (last hour)"}
    v = m["validation"]
    out = []
    out.append(("**Валидация walk-forward** (" if ru else "**Walk-forward validation** (")
               + f"{', '.join(v['folds'])}; {v['n_issues']} " + ("ежедневных выпусков, ВЭС = среднее двух турбин, нормированная мощность 0–1)" if ru
                                                                   else "daily issues, farm = mean of the two turbines, normalized power 0–1)") + ":")
    out.append("")
    hdr = ("| Метод | MAE сутки вперёд (24–47 ч) | RMSE сутки вперёд | MAE внутрисуточно (0–23 ч) |" if ru
           else "| Method | MAE day-ahead (24–47 h) | RMSE day-ahead | MAE intraday (0–23 h) |")
    out += [hdr, "|---|---|---|---|"]
    for k in ("model", "nwp_powercurve", "climatology", "persistence"):
        da, ia = m["by_product"]["day_ahead"][k], m["by_product"]["intraday"][k]
        bold = "**" if k == "model" else ""
        out.append(f"| {bold}{names[k]}{bold} | {bold}{_fmt(da['mae'])}{bold} | {_fmt(da['rmse'])} | {_fmt(ia['mae'])} |")
    s = m.get("skill_pct", {})
    out.append("")
    out.append(("Улучшение MAE на сутки вперёд: " if ru else "Day-ahead MAE improvement: ")
               + f"{s.get('day_ahead_vs_climatology')}% vs " + ("климатологии" if ru else "climatology")
               + f", {s.get('day_ahead_vs_persistence')}% vs " + ("персистентности" if ru else "persistence")
               + f", {s.get('day_ahead_vs_nwp_powercurve')}% vs " + ("кривой мощности NWP." if ru else "the raw NWP power curve."))
    cal = m.get("interval_calibration") or {}
    if cal:
        out.append(("Интервал P10–P90: покрытие до калибровки " if ru else "P10–P90 interval: coverage before calibration ")
                   + f"{_fmt(cal.get('coverage_fit_before'))}; " + ("коэффициент расширения" if ru else "widening factor")
                   + f" k = {cal.get('k')} (" + ("подобран на" if ru else "fitted on") + f" {', '.join(cal.get('fitted_on_months', []))}); "
                   + ("проверка на отложенном месяце" if ru else "out-of-sample check on") + f" {cal.get('check_month')}: "
                   + f"{_fmt(cal.get('coverage_check_before'))} → **{_fmt(cal.get('coverage_check_after'))}** (" + ("цель 0.80)." if ru else "target 0.80)."))
    lo = m.get("leave_one_turbine_out") or {}
    if lo:
        out.append(("Перенос на «новую» турбину: обучение только на " if ru else "Transfer to an unseen turbine: trained only on ")
                   + f"{lo['train']} → MAE {lo['test']} = {_fmt(lo['mae'])} (" + ("при обучении на обеих турбинах" if ru else "vs trained on both")
                   + f" {_fmt(lo['mae_reference'])}).")
    if runs:
        in_tp = [r for r in runs if "2026-01-31" <= r["issue_id"][:10] <= "2026-02-28"]
        pol = sorted({f"{r['policy']}/{r['provider']}" + (f" ({r['model']})" if r.get("model") else "") for r in in_tp})
        nv = sum(len(r["versions"]) for r in in_tp)
        excl = sum(len(r["excluded_models"]) for r in in_tp)
        out.append("")
        out.append(("**Тестовый период:** " if ru else "**Test period:** ") + f"{len(in_tp)} " +
                   ("выпусков (31.01–28.02.2026, 00:00 по времени данных), " if ru else "issues (2026-01-31 … 02-28, 00:00 data clock), ")
                   + f"{nv} " + ("опубликованных версий, политика агента" if ru else "published versions, agent policy")
                   + f": {', '.join(pol)}; " + ("исключений моделей погоды" if ru else "weather-model exclusions") + f": {excl}; "
                   + ("строк в сабмите" if ru else "submission rows") + f": {len(sub)}.")
        post = [r["errors"] for r in in_tp if r.get("errors")]
        if post:
            e = post[0]
            out.append(("Пост-проверка выпуска 31.01 (внутрисуточная часть, факт есть): " if ru else "Post-hoc check of the Jan 31 issue (intraday part, actuals exist): ")
                       + f"MAE {_fmt(e['mae'])} " + ("за" if ru else "over") + f" {e['hours']} " + ("ч." if ru else "h."))
    if clock:
        out.append("")
        out.append(("**Часы SCADA:** " if ru else "**SCADA clock:** ")
                   + (f"оценка UTC+{clock.get('estimated_offset_h')}, в конфигурации UTC+{clock.get('configured_offset_h')}; "
                      if ru else f"estimated UTC+{clock.get('estimated_offset_h')}, configured UTC+{clock.get('configured_offset_h')}; ")
                   + ("время пика температуры стабильно в пределах " if ru else "temperature-peak time stable within ")
                   + f"{clock.get('sun_peak_max_shift_min')} " + ("мин между годами; " if ru else "min across years; ")
                   + ("на 01.03.2024 00:00 нет дублей/пропусков." if ru else "no duplicates/gaps at 2024-03-01 00:00."))
    out.append("")
    out.append(("_Блок сгенерирован командой `python -m windagent report` из `outputs/" if ru else "_Generated by `python -m windagent report` from `outputs/")
               + f"{site}/…`" + ("; не редактировать вручную._" if ru else "; do not edit by hand._"))
    return "\n".join(out)


def update_readme(path: Path, block: str) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), START + "\n" + block.replace("\\", "\\\\") + "\n" + END, text, flags=re.S)
    path.write_text(new, encoding="utf-8")
    return True


def compare(a: str, b: str, atol: float = 2e-4) -> tuple[bool, str]:
    from .errors import InputError
    for f in (a, b):
        if not Path(f).exists():
            raise InputError(f"File not found: {f}", f"Файл не найден: {f}")
    x, y = pd.read_csv(a, encoding="utf-8"), pd.read_csv(b, encoding="utf-8")
    if len(x) != len(y):
        return False, f"row count differs: {len(x)} vs {len(y)}"
    key = "target_time_utc"
    x, y = x.sort_values(key).reset_index(drop=True), y.sort_values(key).reset_index(drop=True)
    if not (x[key] == y[key]).all():
        return False, "target times differ"
    num = [c for c in x.columns if c in y.columns and pd.api.types.is_numeric_dtype(x[c]) and c != "lead_h"]
    worst = {c: float(np.nanmax(np.abs(x[c].to_numpy(float) - y[c].to_numpy(float)))) for c in num}
    bad = {c: d for c, d in worst.items() if d > atol}
    if bad:
        return False, f"numeric columns differ beyond {atol}: {bad}"
    return True, f"identical within {atol}: {len(x)} rows, columns {num} (max abs diff {max(worst.values()):.2e})"
