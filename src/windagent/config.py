"""Paths, environment and site configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .errors import DataUnavailableError, InputError

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env", override=False)

__version__ = "0.1.0"

# Issue protocol defaults (see README, section 4).
ISSUE_HOUR_DATA_CLOCK = 0
HORIZON_H = 48

# Weather models used as ensemble members. latency_h = conservative publication delay
# (measured delay from Open-Meteo meta.json + safety margin). A run initialised at t0 is
# treated as available only from t0 + latency_h.
WEATHER_MODELS: list[dict] = [
    {"id": "ecmwf_aifs025_single", "label": "ECMWF AIFS 0.25° (AI weather model)", "latency_h": 7,
     "archive_from": "2025-02-19"},
    {"id": "ecmwf_ifs025", "label": "ECMWF IFS 0.25°", "latency_h": 9, "archive_from": "2024-03-07"},
    {"id": "icon_seamless", "label": "DWD ICON", "latency_h": 5, "archive_from": "2024-02-17"},
    {"id": "gfs_seamless", "label": "NOAA GFS", "latency_h": 8, "archive_from": "2024-02-17"},
]


def data_dir() -> Path:
    return Path(os.environ.get("WINDAGENT_DATA_DIR", REPO_ROOT / "data"))


def outputs_dir() -> Path:
    return Path(os.environ.get("WINDAGENT_OUTPUTS_DIR", REPO_ROOT / "outputs"))


def models_dir() -> Path:
    return Path(os.environ.get("WINDAGENT_MODELS_DIR", REPO_ROOT / "models"))


def config_path() -> Path:
    return Path(os.environ.get("WINDAGENT_CONFIG", REPO_ROOT / "config" / "sites.yaml"))


def offline() -> bool:
    return os.environ.get("WINDAGENT_OFFLINE", "0").strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Turbine:
    id: str
    lat: float
    lon: float
    scada_csv: str | None = None

    def scada_path(self) -> Path | None:
        return data_dir() / self.scada_csv if self.scada_csv else None


@dataclass(frozen=True)
class Site:
    site: str
    name: str
    data_clock_utc_offset_h: int
    official_utc_offset_h: int
    turbines: tuple[Turbine, ...]
    rated_mw: float | None = None
    test_period: dict = field(default_factory=dict)

    @property
    def lat(self) -> float:
        return round(sum(t.lat for t in self.turbines) / len(self.turbines), 4)

    @property
    def lon(self) -> float:
        return round(sum(t.lon for t in self.turbines) / len(self.turbines), 4)

    @property
    def turbine_ids(self) -> list[str]:
        return [t.id for t in self.turbines]


@lru_cache(maxsize=4)
def _load_sites(path: str) -> dict[str, Site]:
    p = Path(path)
    if not p.exists():
        raise DataUnavailableError(f"Site config not found: {p}", f"Не найден файл конфигурации площадок: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    sites: dict[str, Site] = {}
    for key, s in (raw.get("sites") or {}).items():
        turbines = tuple(
            Turbine(id=str(t["id"]), lat=float(t["lat"]), lon=float(t["lon"]), scada_csv=t.get("scada_csv"))
            for t in s.get("turbines", [])
        )
        if not turbines:
            raise InputError(f"Site '{key}' has no turbines in {p}", f"У площадки '{key}' нет турбин в {p}")
        sites[key] = Site(
            site=key,
            name=s.get("name", key),
            data_clock_utc_offset_h=int(s.get("data_clock_utc_offset_h", 0)),
            official_utc_offset_h=int(s.get("official_utc_offset_h", s.get("data_clock_utc_offset_h", 0))),
            turbines=turbines,
            rated_mw=s.get("rated_mw"),
            test_period=s.get("test_period") or {},
        )
    return sites


def load_sites() -> dict[str, Site]:
    return _load_sites(str(config_path()))


def get_site(site: str) -> Site:
    sites = load_sites()
    if site not in sites:
        known = ", ".join(sorted(sites)) or "none"
        raise InputError(f"Unknown site '{site}'. Known sites: {known}",
                         f"Неизвестная площадка '{site}'. Доступные: {known}")
    return sites[site]
