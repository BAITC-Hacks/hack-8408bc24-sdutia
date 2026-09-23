"""Human labels for API facts. No domain calculations or file access."""
from __future__ import annotations

from datetime import timedelta, timezone
import re
import pandas as pd

MONTHS_RU = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря")
MONTHS_EN = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
MONTHS_SHORT_RU = ("янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")
MODELS = {
    "ecmwf_aifs025_single": ("ECMWF AIFS (AI-модель)", "ECMWF AIFS (AI model)"),
    "ecmwf_ifs025": ("ECMWF IFS", "ECMWF IFS"),
    "icon_seamless": ("ICON (DWD)", "ICON (DWD)"),
    "gfs_seamless": ("GFS (NOAA)", "GFS (NOAA)"),
}


def model_name(model, lang="RU") -> str:
    return MODELS.get(str(model), (str(model), str(model)))[lang == "EN"]


def local_stamp(value, offset=6):
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    return stamp.tz_convert(timezone(timedelta(hours=int(offset)))) if pd.notna(stamp) else None


def date_label(value, lang="RU", *, short=False, year=True) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    months = MONTHS_EN if lang == "EN" else (MONTHS_SHORT_RU if short else MONTHS_RU)
    return f"{value.day} {months[value.month - 1]}" + (f" {value.year}" if year else "")


def issue_label(issue_time_utc, lang="RU", data_offset=6, horizon_h=48) -> str:
    stamp = local_stamp(issue_time_utc, data_offset)
    if stamp is None:
        return "n/a"
    last = stamp + pd.Timedelta(hours=max(int(horizon_h) - 1, 0))
    dates = date_label(stamp, lang, short=True, year=False)
    if last.date() != stamp.date():
        dates += " – " + date_label(last, lang, short=True, year=False)
    phrase = "forecast for" if lang == "EN" else "прогноз на"
    return f"{date_label(stamp, lang)}, {stamp:%H:%M} → {phrase} {dates}"


def default_issue(ordered: pd.DataFrame, info: dict):
    if ordered.empty:
        return None
    first = pd.to_datetime((info.get("test_period") or {}).get("start"), errors="coerce")
    dates = ordered["issue_time_utc"].map(lambda value: local_stamp(value, info.get("data_clock_utc_offset_h", 6)))
    if pd.notna(first):
        exact = ordered.loc[dates.map(lambda d: d is not None and d.date() == first.date())]
        if not exact.empty:
            return exact.iloc[-1]["issue_id"]
        eligible = ordered.loc[dates.map(lambda d: d is not None and d.date() >= first.date())]
        if not eligible.empty:
            return eligible.sort_values("issue_time_utc").iloc[0]["issue_id"]
    return ordered.sort_values("issue_time_utc").iloc[0]["issue_id"]


def version_record(meta: dict, version) -> dict:
    return next((item for item in meta.get("versions", []) or []
                 if str(item.get("version")) == str(version)), {})


def version_kpis(meta: dict, version=None) -> dict:
    # A legacy issue summary may describe v2; never present it as v1's KPIs.
    record = meta if version is None else version_record(meta, version)
    return record.get("kpis") or {}


def version_label(meta: dict, version, lang="RU", data_offset=6) -> str:
    record = version_record(meta, version)
    stamp = local_stamp(record.get("as_of_utc") or (meta.get("issue_time_utc") if str(version) == "1" else None), data_offset)
    time = stamp.strftime("%H:%M") if stamp is not None else None
    if str(version) == "1":
        return (f"{time} forecast (initial)" if time else "Initial forecast") if lang == "EN" else (f"Прогноз в {time} (первичный)" if time else "Первичный прогноз")
    return (f"{time} update (newer weather runs)" if time else "Update (newer weather runs)") if lang == "EN" else (f"Уточнение в {time} (вышли новые прогнозы погоды)" if time else "Уточнение (вышли новые прогнозы погоды)")


def human_reason(reason, lang="RU") -> str:
    text = str(reason or "")
    if text.strip().lower() == "initial":
        return "initial forecast" if lang == "EN" else "первичный прогноз"
    for model in MODELS:
        text = text.replace(model, model_name(model, lang))
    if lang != "EN":
        text = text.replace("newer runs for ", "новые прогоны ").replace("new runs: ", "новые прогоны: ")
        text = re.sub(r"(\d+(?:\.\d+)?)% of hours, run [\d\- :]+", r"\1% оставшихся часов", text)
        text = text.replace("new weather runs", "новые прогнозы погоды")
    return text or "n/a"


def source_label(source, lang="RU") -> str:
    names = {"live": ("загружено из API Open-Meteo", "downloaded from the Open-Meteo API"),
             "cache": ("из кэша", "from cache"),
             "mixed": ("API Open-Meteo и кэш", "Open-Meteo API and cache")}
    return names.get(str(source), ("n/a", "n/a"))[lang == "EN"]
