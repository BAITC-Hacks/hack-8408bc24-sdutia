"""Clock handling. Internally every timestamp is tz-aware UTC; clocks are fixed offsets only.

The SCADA data clock is a fixed UTC+6 (it did not follow Kazakhstan's 2024 switch to UTC+5),
so named time zones such as Asia/Almaty must never be used here.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd

from .errors import InputError

UTC = timezone.utc
CLOCK_FMT = "%Y-%m-%d %H:%M"
_ISSUE_ID_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})$")


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(UTC))


def to_utc(ts) -> pd.Timestamp:
    """Timestamp → tz-aware UTC. Naive input is rejected to avoid silent clock bugs."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise ValueError(f"naive timestamp {t!r}: specify the clock explicitly")
    return t.tz_convert(UTC)


def clock_to_utc(local, offset_h: int):
    """Naive wall-clock time(s) on a fixed-offset clock → tz-aware UTC."""
    shift = pd.Timedelta(hours=offset_h)
    if isinstance(local, pd.Series):
        return (pd.to_datetime(local) - shift).dt.tz_localize(UTC)
    if isinstance(local, pd.DatetimeIndex):
        return (local - shift).tz_localize(UTC)
    return (pd.Timestamp(local) - shift).tz_localize(UTC)


def utc_to_clock(ts, offset_h: int):
    """tz-aware UTC → naive wall-clock time on a fixed-offset clock."""
    if isinstance(ts, pd.Series):
        return ts.dt.tz_convert(UTC).dt.tz_localize(None) + pd.Timedelta(hours=offset_h)
    if isinstance(ts, pd.DatetimeIndex):
        return ts.tz_convert(UTC).tz_localize(None) + pd.Timedelta(hours=offset_h)
    return to_utc(ts).tz_localize(None) + pd.Timedelta(hours=offset_h)


def clock_str(ts, offset_h: int):
    """tz-aware UTC → 'YYYY-MM-DD HH:MM' on the given fixed-offset clock."""
    local = utc_to_clock(ts, offset_h)
    if isinstance(local, pd.Series):
        return local.dt.strftime(CLOCK_FMT)
    if isinstance(local, pd.DatetimeIndex):
        return local.strftime(CLOCK_FMT)
    return local.strftime(CLOCK_FMT)


def iso_z(ts):
    """tz-aware UTC → 'YYYY-MM-DDTHH:MM:SSZ'."""
    if isinstance(ts, pd.Series):
        return ts.dt.tz_convert(UTC).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(ts, pd.DatetimeIndex):
        return ts.tz_convert(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return to_utc(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_clock_time(text: str, offset_h: int) -> pd.Timestamp:
    """Parse user input 'YYYY-MM-DD HH:MM' (or 'YYYY-MM-DD') given on a fixed-offset clock → UTC."""
    s = str(text).strip().replace("T", " ")
    for fmt in (CLOCK_FMT, "%Y-%m-%d"):
        try:
            local = datetime.strptime(s, fmt)
            break
        except ValueError:
            continue
    else:
        raise InputError(
            f"Invalid time '{text}'. Expected format YYYY-MM-DD HH:MM (data clock, UTC+{offset_h}).",
            f"Неверное время '{text}'. Ожидается формат ГГГГ-ММ-ДД ЧЧ:ММ (время данных, UTC+{offset_h}).",
        )
    return clock_to_utc(local, offset_h)


def issue_id(issue_time_utc, offset_h: int) -> str:
    return utc_to_clock(issue_time_utc, offset_h).strftime("%Y-%m-%d_%H%M")


def issue_time_from_id(issue_id_str: str, offset_h: int) -> pd.Timestamp:
    m = _ISSUE_ID_RE.match(issue_id_str)
    if not m:
        raise InputError(f"Invalid issue id '{issue_id_str}' (expected YYYY-MM-DD_HHMM)",
                         f"Неверный идентификатор прогноза '{issue_id_str}' (ожидается ГГГГ-ММ-ДД_ЧЧММ)")
    return parse_clock_time(f"{m.group(1)} {m.group(2)}:{m.group(3)}", offset_h)
