"""Reproduce the SCADA data report and figures without modifying inputs or core outputs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
STEP = pd.Timedelta(minutes=10)
MIN_RECORDS = 4
COLORS = ["#126A91", "#C65A29"]
MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def fmt(value: float | int, digits: int = 3) -> str:
    if pd.isna(value):
        return "н/д"
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def link(path: Path, output: Path) -> str:
    return os.path.relpath(path, output).replace(os.sep, "/")


def source_constants() -> dict:
    """Read documented split boundaries as literals; do not execute the forecasting module."""
    names = {"TRAIN_START_DATA_CLOCK", "PRODUCTION_CUTOFF_DATA_CLOCK"}
    tree = ast.parse((ROOT / "src/windagent/forecast.py").read_text(encoding="utf-8"))
    return {node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names}


def load_turbine(path: Path, turbine: str, offset: int) -> dict:
    raw = pd.read_csv(path, encoding="utf-8")
    if len(raw.columns) < 5:
        raise ValueError(f"{path}: expected at least five positional columns")
    wall = pd.to_datetime(raw.iloc[:, 1], errors="coerce", format="mixed")
    if wall.dt.tz is not None:
        raise ValueError(f"{path}: expected naive timestamps on the configured data clock")
    numeric = raw.iloc[:, 2:5].apply(pd.to_numeric, errors="coerce")
    numeric.columns = ["ws", "p", "temp"]
    missing = numeric.isna().sum().to_dict()
    nonfinite = (numeric.notna() & ~np.isfinite(numeric)).sum().to_dict()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    frame = numeric.copy()
    frame["time"] = wall
    valid = frame.dropna(subset=["time"])
    duplicates = int(valid["time"].duplicated().sum())
    backward = int((valid["time"].diff() < pd.Timedelta(0)).sum())
    clean = valid.drop_duplicates("time", keep="first").sort_values("time")
    if clean.empty:
        raise ValueError(f"{path}: no parseable observation timestamps")
    local = pd.DatetimeIndex(clean["time"])
    clean.index = (local - pd.Timedelta(hours=offset)).tz_localize("UTC")
    clean = clean[["ws", "p", "temp"]]
    grid = pd.date_range(local.min().floor("10min"), local.max().ceil("10min"), freq="10min")
    delta = local.to_series().diff().dropna()
    gaps = delta[delta > STEP]
    gap_end = gaps.idxmax() if not gaps.empty else None
    largest = gaps.max() if not gaps.empty else pd.Timedelta(0)
    resampled = clean.resample("1h")
    hourly = resampled.mean()
    hourly["n_records"] = resampled["p"].count()
    hourly = hourly.loc[hourly["n_records"] >= MIN_RECORDS].copy()
    hourly["flag"] = (hourly["p"] < 0.05) & (hourly["ws"] > 6)
    daily = pd.Series(1, index=local).resample("1D").sum()
    daily = daily.reindex(pd.date_range(local.min().normalize(), local.max().normalize(), freq="1D"), fill_value=0)
    hourly_local = hourly.index.tz_localize(None) + pd.Timedelta(hours=offset)
    hour_days = pd.Series(1, index=hourly_local).resample("1D").sum().reindex(daily.index, fill_value=0)
    stats = {
        "rows": len(raw), "start": str(local.min()), "end": str(local.max()),
        "invalid_time": int(wall.isna().sum()), "duplicates": duplicates, "backward": backward,
        "missing_numeric": missing, "nonfinite_numeric": nonfinite,
        "expected_grid": len(grid), "missing_grid": len(grid.difference(local)),
        "off_grid": int(((local.minute % 10 != 0) | (local.second != 0) | (local.microsecond != 0)).sum()),
        "modal_step_minutes": delta.mode().iloc[0].total_seconds() / 60 if len(delta) else np.nan,
        "gap_count": len(gaps), "largest_gap_hours": largest.total_seconds() / 3600,
        "gap_before": str(gap_end - largest) if gap_end is not None else "нет",
        "gap_after": str(gap_end) if gap_end is not None else "нет",
        "hourly_count": len(hourly), "expected_hours": len(resampled.size()),
        "out_of_range_power": int(((numeric["p"] < 0) | (numeric["p"] > 1)).sum()),
        "negative_wind": int((numeric["ws"] < 0).sum()),
        "extreme_temperature": int(((numeric["temp"] < -60) | (numeric["temp"] > 60)).sum()),
        "flagged_hours": int(hourly["flag"].sum()),
        "power_mean": float(hourly["p"].mean()), "power_median": float(hourly["p"].median()),
        "power_q10": float(hourly["p"].quantile(0.1)), "power_q90": float(hourly["p"].quantile(0.9)),
        "low_power_share": float((hourly["p"] < 0.05).mean()),
        "high_power_share": float((hourly["p"] >= 0.9).mean()),
        "wind_mean": float(hourly["ws"].mean()),
        "ranges": {name: [float(numeric[name].min()), float(numeric[name].max())] for name in numeric},
    }
    return {"id": turbine, "path": path, "headers": list(raw.columns), "raw": clean, "hourly": hourly,
            "daily": daily, "hour_days": hour_days, "stats": stats,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def pair_data(data: list[dict], resolution: str, variable: str) -> pd.DataFrame:
    # Inner timestamp join, then pairwise complete values: never interpolate across gaps.
    return pd.concat([d[resolution][variable].rename(d["id"]) for d in data], axis=1, join="inner").dropna()


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=170, facecolor="white", metadata={"Software": "WindAgent reproducible EDA"})
    plt.close(fig)


def make_figures(data: list[dict], output: Path, offset: int) -> list[str]:
    folder = output / "figures/eda"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
                         "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "grid.alpha": 0.2, "axes.axisbelow": True, "savefig.bbox": "tight"})
    names = []

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    for d, color in zip(data, COLORS):
        axes[0].plot(d["daily"].index, d["daily"] / 144 * 100, label=d["id"], color=color, lw=1)
        axes[1].plot(d["hour_days"].index, d["hour_days"] / 24 * 100, label=d["id"], color=color, lw=1)
    axes[0].set(title="Полнота SCADA по дням: пропуски наблюдений", ylabel="10-минутные записи, %")
    axes[1].set(title="Доля часов с ≥ 4 значениями мощности", ylabel="Принятые часы, %",
                xlabel=f"Дата по часам данных (UTC+{offset})")
    for ax in axes:
        ax.set_ylim(-3, 105)
        ax.grid()
        ax.legend(loc="lower left", ncol=2)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    names.append("01_data_coverage.png")
    save(fig, folder / names[-1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    all_power = pd.concat([d["hourly"]["p"] for d in data])
    bins = np.linspace(min(0, all_power.min()), max(1, all_power.max()), 31)
    for d, color in zip(data, COLORS):
        power = d["hourly"]["p"].dropna()
        axes[0].hist(power, bins=bins, weights=np.full(len(power), 100 / len(power)), histtype="step",
                     color=color, lw=2, label=f"{d['id']}: n={len(power):,}".replace(",", " "))
        ordered = power.sort_values().to_numpy()
        axes[1].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered) * 100, color=color, label=d["id"])
    axes[0].set(title="Распределение почасовой мощности", ylabel="Доля принятых часов, %")
    axes[1].set(title="Накопленное распределение", ylabel="Доля часов с мощностью ≤ x, %")
    for ax in axes:
        ax.set_xlabel("Нормированная мощность, отн. ед.")
        ax.grid()
        ax.legend()
    names.append("02_power_distribution.png")
    save(fig, folder / names[-1])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    for d, color in zip(data, COLORS):
        h = d["hourly"]
        months = (h.index.tz_localize(None) + pd.Timedelta(hours=offset)).month
        grouped = h.groupby(months)[["p", "ws"]].mean().reindex(range(1, 13))
        for ax, variable in zip(axes, ["p", "ws"]):
            ax.plot(grouped.index, grouped[variable], marker="o", color=color, label=d["id"])
    axes[0].set(title="Средние по календарным месяцам за доступные годы", ylabel="Мощность, отн. ед.")
    axes[1].set(ylabel="Скорость ветра, м/с", xlabel=f"Месяц по часам данных (UTC+{offset}); разное число наблюдений")
    axes[1].set_xticks(range(1, 13), MONTHS)
    for ax in axes:
        ax.grid()
        ax.legend(ncol=2)
    names.append("03_monthly_seasonality.png")
    save(fig, folder / names[-1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, layout="constrained")
    for d, ax in zip(data, axes):
        h = d["hourly"].dropna(subset=["ws", "p"])
        density = ax.hexbin(h["ws"], h["p"], gridsize=45, mincnt=1, bins="log", cmap="viridis")
        wind_bin = np.floor(h["ws"]).astype(int)
        med = h.groupby(wind_bin)["p"].agg(["median", "count"])
        med = med[med["count"] >= 30]
        ax.plot(med.index + 0.5, med["median"], color="#EF8B2C", lw=2, label="Медиана в бине 1 м/с (n ≥ 30)")
        flagged = h[h["flag"]]
        ax.scatter(flagged["ws"], flagged["p"], facecolors="none", edgecolors="#D52A39", s=23, lw=0.7,
                   label="Низкая мощность при ветре > 6 м/с")
        ax.set(title=f"{d['id']}: ветер и мощность, почасовые пары", xlabel="Наблюдаемый ветер SCADA, м/с")
        ax.legend(loc="upper left", fontsize=9)
        fig.colorbar(density, ax=ax, label="Часы в ячейке (лог. шкала)", shrink=0.77)
    axes[0].set_ylabel("Нормированная мощность, отн. ед.")
    names.append("04_wind_power.png")
    save(fig, folder / names[-1])

    paired = pair_data(data, "hourly", "p")
    a, b = data[0]["id"], data[1]["id"]
    corr = paired[a].corr(paired[b])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    density = axes[0].hexbin(paired[a], paired[b], gridsize=45, mincnt=1, bins="log", cmap="viridis")
    axes[0].plot([0, 1], [0, 1], "--", color="#C65A29", lw=1)
    axes[0].set(title=f"Совпадающие часы: n={len(paired):,}, r={corr:.3f}".replace(",", " "),
                xlabel=f"{a}: мощность, отн. ед.", ylabel=f"{b}: мощность, отн. ед.")
    axes[0].set_aspect("equal", adjustable="box")
    fig.colorbar(density, ax=axes[0], label="Часы в ячейке (лог. шкала)", shrink=0.77)
    difference = paired[a] - paired[b]
    axes[1].hist(difference, bins=50, color=COLORS[0], weights=np.full(len(difference), 100 / len(difference)), alpha=0.85)
    axes[1].axvline(0, color="#C65A29", ls="--", lw=1)
    axes[1].set(title="Разница мощности в совпадающие часы", xlabel=f"{a} − {b}, отн. ед.", ylabel="Доля пар часов, %")
    axes[1].grid()
    names.append("05_cross_turbine.png")
    save(fig, folder / names[-1])
    return names


def report(data: list[dict], site: dict, site_name: str, config_path: Path, output: Path, names: list[str]) -> str:
    offset = int(site["data_clock_utc_offset_h"])
    ids = [d["id"] for d in data]
    sources = "; ".join(f"[{d['id']}: {d['path'].name}]({link(d['path'], output)})" for d in data)
    constants = source_constants()
    cutoff = constants["PRODUCTION_CUTOFF_DATA_CLOCK"]
    validation_path = ROOT / "outputs" / site_name / "validation/metrics.json"
    folds = json.loads(validation_path.read_text(encoding="utf-8"))["validation"]["folds"]
    period = site["test_period"]
    start_test = (pd.Timestamp(period["start"]) - pd.Timedelta(hours=offset)).tz_localize("UTC")
    end_test = (pd.Timestamp(period["end"]) + pd.Timedelta(days=1) - pd.Timedelta(hours=offset)).tz_localize("UTC")
    paired_hourly = pair_data(data, "hourly", "p")
    lines = ["# Анализ исходных данных SCADA", "",
             f"Площадка: **{site_name}**. Источники: {sources}. Параметры площадки и часы взяты из "
             f"[конфигурации]({link(config_path, output)}). Отчёт описывает наблюдения; точность прогноза здесь не оценивается.", "",
             "## Основные результаты", "",
             f"- История охватывает **{data[0]['stats']['start']} — {data[0]['stats']['end']}** по часам данных "
             f"(фиксированный UTC+{offset}). Границы каждой турбины приведены ниже.",
             "- Внутри записанных строк значения могут быть заполнены полностью, но это не означает непрерывную временную историю. "
             + "; ".join(f"{d['id']}: отсутствует {fmt(d['stats']['missing_grid'], 0)} из {fmt(d['stats']['expected_grid'], 0)} "
                          f"ожидаемых 10-минутных меток ({fmt(d['stats']['missing_grid'] / d['stats']['expected_grid'] * 100, 2)}%)" for d in data) + ".",
             f"- На {fmt(len(paired_hourly), 0)} совпадающих принятых часах корреляция Пирсона мощности равна "
             f"**{fmt(paired_hourly.iloc[:, 0].corr(paired_hourly.iloc[:, 1]), 4)}**. Это связь наблюдений соседних турбин, "
             "а не проверка переноса модели на новую площадку.",
             "- Число 10-минутных наблюдений в заданном тестовом периоде: "
             + "; ".join(f"{d['id']} — {fmt(((d['raw'].index >= start_test) & (d['raw'].index < end_test)).sum(), 0)}" for d in data)
             + f" ({period['start']} — {period['end']}).", "",
             "## Источники и обработка", "",
             "Столбцы читаются **по позиции**, чтобы не зависеть от языка заголовков:", "",
             "| Позиция | Значение | Использование |", "|---|---|---|",
             "| 1 | ID строки | Не является временной меткой; не используется как признак |",
             "| 2 | Статистическое время | Наивная метка часов данных |",
             "| 3 | Средняя скорость ветра, м/с | `ws` |",
             "| 4 | Нормированная активная мощность, 0–1 | `p` |",
             "| 5 | Средняя температура окружающей среды, °C | `temp` |", "",
             f"Для вычислений время переводится в timezone-aware UTC вычитанием {offset} часов. На графиках даты и месяцы "
             "показаны по часам данных. Фиксированный сдвиг взят из конфигурации, а не заново установлен этим EDA. "
             "Изменения официального часового пояса не применяются автоматически.", "",
             "Не распознанные времена исключаются; при повторе времени сохраняется первая строка. Отсутствующие, нечисловые "
             "и бесконечные измерения становятся пропусками, **не нулями**. Конечные значения вне проверяемых диапазонов "
             "отмечаются, но не обрезаются и не удаляются. Исходные CSV не изменяются.", "",
             "Часовая метка — начало интервала. Принимается час, в котором есть **не менее 4 конечных значений мощности "
             "из ожидаемых 6**; мощность, ветер и температура усредняются по своим доступным значениям в этом часе. "
             "Это воспроизводит правило `src/windagent/scada.py` для данных без бесконечностей. Порог считается по мощности, "
             "он не гарантирует 4 значений ветра или температуры. Пропуски не интерполируются, в том числе перед корреляциями.", "",
             "## Полнота и качество", "",
             "| Показатель | " + " | ".join(ids) + " |", "|---|" + "---:|" * len(data)]
    metrics = [("Строк в CSV", "rows"), ("Начало (часы данных)", "start"), ("Конец (часы данных)", "end"),
               ("Нераспознанных времён", "invalid_time"), ("Повторных временных меток", "duplicates"),
               ("Шагов назад в исходном порядке строк", "backward"), ("Модальный шаг, мин", "modal_step_minutes"),
               ("Меток вне 10-минутной сетки", "off_grid"), ("Ожидаемых 10-минутных меток", "expected_grid"),
               ("Отсутствующих меток сетки", "missing_grid"), ("Разрывов между соседними метками > 10 мин", "gap_count"),
               ("Наибольший интервал между наблюдениями, ч", "largest_gap_hours"),
               ("Принятых часовых средних", "hourly_count"), ("Часов в полном диапазоне", "expected_hours"),
               ("Мощность вне [0, 1], строк", "out_of_range_power"), ("Отрицательный ветер, строк", "negative_wind"),
               ("Температура вне [−60, +60] °C, строк", "extreme_temperature")]
    for label, key in metrics:
        values = [d["stats"][key] for d in data]
        lines.append("| " + label + " | " + " | ".join(v if isinstance(v, str) else fmt(v, 2 if key == "largest_gap_hours" else 0) for v in values) + " |")
    lines += ["", "Ожидаемая сетка строится между крайними метками каждой турбины. «Разрыв» — интервал между соседними "
              "уникальными метками, превышающий номинальный шаг; это не число подряд отсутствующих часов. "
              "Диапазон температуры — только широкий порог контроля качества, а не технический предел турбины.", "",
              "| Турбина | Последняя метка перед самым большим разрывом | Первая после разрыва |", "|---|---|---|"]
    for d in data:
        s = d["stats"]
        lines.append(f"| {d['id']} | {s['gap_before']} | {s['gap_after']} |")
    lines += ["", "| Турбина / столбец | Пропуск или нечисловое значение | ±∞ | Минимум | Максимум |", "|---|---:|---:|---:|---:|"]
    for d in data:
        for variable, label in [("ws", "ветер, м/с"), ("p", "мощность, отн. ед."), ("temp", "температура, °C")]:
            s = d["stats"]
            lines.append(f"| {d['id']} / {label} | {fmt(s['missing_numeric'][variable], 0)} | {fmt(s['nonfinite_numeric'][variable], 0)} "
                         f"| {fmt(s['ranges'][variable][0], 2)} | {fmt(s['ranges'][variable][1], 2)} |")
    lines += ["", f"![Полнота данных](figures/eda/{names[0]})", "",
              "На графике знаменатели — 144 ожидаемые записи и 24 часа за календарные сутки. "
              "Нулевая полнота означает отсутствие наблюдений, а не нулевую выработку. Причина разрывов из этих столбцов неизвестна.", "",
              "## Распределение и сезонность", "",
              "Статистики ниже рассчитаны по принятым часам каждой турбины, включая часы с флагом низкой выработки; "
              "пропуски за пределами этих часов не восполняются.", "",
              "| Турбина | Часов | Средняя мощность | Медиана | Q10 | Q90 | Доля p < 0,05 | Доля p ≥ 0,90 | Средний ветер, м/с |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in data:
        s = d["stats"]
        lines.append(f"| {d['id']} | {fmt(s['hourly_count'], 0)} | " + " | ".join(fmt(s[key]) for key in ["power_mean", "power_median", "power_q10", "power_q90"])
                     + f" | {fmt(s['low_power_share'] * 100, 2)}% | {fmt(s['high_power_share'] * 100, 2)}% | {fmt(s['wind_mean'], 2)} |")
    lines += ["", f"![Распределение мощности](figures/eda/{names[1]})", "",
              f"![Сезонность](figures/eda/{names[2]})", "",
              "Средние по календарным месяцам объединяют доступные годы. Это описательная сезонность: разная полнота, "
              "состав лет и доступность оборудования могут влиять на сравнение. Это не климатическая норма и не прогноз.", "",
              "| Месяц | " + " | ".join(f"{d['id']}: часов / средняя мощность" for d in data) + " |", "|---|---:|---:|"]
    for month, label in enumerate(MONTHS, 1):
        values = []
        for d in data:
            h = d["hourly"]
            group = h[(h.index.tz_localize(None) + pd.Timedelta(hours=offset)).month == month]
            values.append(f"{fmt(len(group), 0)} / {fmt(group['p'].mean())}")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines += ["", "## Ветер, мощность и флаг доступности", "",
              f"![Связь ветра и мощности](figures/eda/{names[3]})", "",
              "График показывает одновременные наблюдения, а не качество метеопрогноза. Цвет — число часовых пар в ячейке; "
              "медиана рассчитана в бинах ветра шириной 1 м/с с минимум 30 парами. Наблюдаемая зависимость нелинейна; "
              "линейная корреляция не заменяет кривую мощности.", "",
              "Эвристика проекта: час помечается при **p < 0,05 и ветре > 6 м/с**. "
              + "; ".join(f"{d['id']}: {fmt(d['stats']['flagged_hours'], 0)} из {fmt(d['stats']['hourly_count'], 0)} принятых часов "
                           f"({fmt(d['stats']['flagged_hours'] / d['stats']['hourly_count'] * 100, 2)}%)" for d in data) + ". "
              "Это кандидаты для проверки; по мощности, ветру и температуре нельзя установить ремонт, ограничение, "
              "обледенение или ошибку датчика. Флаг не является подтверждённым простоем и не прогнозирует будущий отказ.", "",
              "## Связь между турбинами", "",
              "| Разрешение / величина | Совпадающих полных пар | Корреляция Пирсона |", "|---|---:|---:|"]
    for resolution, label in [("raw", "10 минут"), ("hourly", "1 час")]:
        for variable, title in [("p", "мощность"), ("ws", "ветер"), ("temp", "температура")]:
            paired = pair_data(data, resolution, variable)
            lines.append(f"| {label} / {title} | {fmt(len(paired), 0)} | {fmt(paired.iloc[:, 0].corr(paired.iloc[:, 1]), 4)} |")
    lines += ["", f"![Сопоставление турбин](figures/eda/{names[4]})", "",
              "Пары соединены по одной и той же UTC-метке и отброшены, если хотя бы одно значение отсутствует. "
              "Для почасового сравнения каждая турбина отдельно должна пройти порог полноты. Сдвиги и интерполяция не применялись. "
              "Внутри одного совпадающего часа средние турбин могут опираться на разные наборы 10-минутных наблюдений. "
              "Высокая корреляция может отражать общую погоду и соседство; она не означает независимость турбин и не доказывает причинность.", "",
              "## Границы обучения и честная оценка", "",
              f"- Начало обучающих выпусков в текущем ядре: **{constants['TRAIN_START_DATA_CLOCK']}**; "
              f"cutoff рабочей модели: **{cutoff}** по часам данных. Источник: "
              f"[forecast.py]({link(ROOT / 'src/windagent/forecast.py', output)}), "
              "`TRAIN_START_DATA_CLOCK`, `PRODUCTION_CUTOFF_DATA_CLOCK`. Это границы протокола, а не полный диапазон исходного SCADA.",
              "- При подготовке обучающей строки и свежего наблюдения используется только информация, доступная к выпуску. "
              "Само наличие строки в полном CSV не делает её допустимым признаком для более раннего прогноза.",
              f"- Месяцы текущей walk-forward валидации: **{', '.join(folds)}**; источник: "
              f"[metrics.json]({link(validation_path, output)}), `validation.folds`. Каждый месяц проверяется после обучения "
              "на прошлом. Статистики и рисунки этого EDA рассчитаны по всей истории и не должны использоваться как "
              "готовые параметры преобразований для ранних validation-folds.",
              f"- Тестовый период из конфигурации — **{period['start']} — {period['end']}**. Число фактических строк в нём "
              "посчитано выше. Отсутствие факта исключает измерение февральской точности по этим файлам; опубликованный "
              "прогноз нельзя считать подтверждённым результатом оценки.",
              "- Нормированная мощность не задаёт номинальную мощность в МВт. Средние, разности и доли в отчёте "
              "не являются энергией в МВт·ч, денежным эффектом или взвешенным КИУМ всей станции.", "",
              "## Воспроизведение", "",
              "Из корня репозитория (установленные зависимости из `requirements.txt`):", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe scripts/eda.py", "```", "",
              "Скрипт также работает из другого текущего каталога, если указать его полный путь. "
              "`--data-dir` задаёт корень данных с подкаталогом `raw`; `--output-dir` — каталог отчёта "
              "(в нём создаются `DATA_ANALYSIS.md` и `figures/eda/*.png`); доступны `--site` и `--config`. "
              "По умолчанию используются `data/`, `docs/` и `config/sites.yaml` данного репозитория. "
              "Выполнение не обращается к сети, не использует LLM и не изменяет сырьё, модели или `outputs/`. "
              "В отчёт и PNG не записывается текущее время.", "",
              "SHA-256 входных CSV фиксирует, по каким данным рассчитан отчёт:", "",
              "| Файл | SHA-256 |", "|---|---|"]
    for d in data:
        lines.append(f"| {d['path'].name} | `{d['sha256']}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="shelek")
    parser.add_argument("--config", type=Path, default=ROOT / "config/sites.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    config_path, output = args.config.resolve(), args.output_dir.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    site = config["sites"][args.site]
    if len(site["turbines"]) != 2:
        parser.error("this paired-turbine report expects exactly two configured turbines")
    data = [load_turbine((args.data_dir / t["scada_csv"]).resolve(), t["id"], int(site["data_clock_utc_offset_h"]))
            for t in site["turbines"]]
    output.mkdir(parents=True, exist_ok=True)
    names = make_figures(data, output, int(site["data_clock_utc_offset_h"]))
    (output / "DATA_ANALYSIS.md").write_text(report(data, site, args.site, config_path, output, names), encoding="utf-8")
    pairs = pair_data(data, "hourly", "p")
    print(json.dumps({"report": str(output / "DATA_ANALYSIS.md"), "figures": names,
                      "turbines": {d["id"]: d["stats"] for d in data},
                      "paired_hourly_n": len(pairs), "paired_hourly_power_r": pairs.iloc[:, 0].corr(pairs.iloc[:, 1])},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
