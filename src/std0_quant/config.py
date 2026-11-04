"""Configuration loading for std0-quant.

Loads ``config/settings.yaml`` and applies optional ``.env`` overrides.
Research definitions that the spec marks as FROZEN (episode rule, Y30 horizon)
are validated against the constants in :mod:`std0_quant` and loading fails if
they are modified. This is a deliberate guard: nobody should be able to change
the 3-second episode rule or the 30-second Y30 horizon via configuration.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr

from std0_quant import (
    EPISODE_RULE_VERSION,
    EPISODE_WINDOW_SECONDS,
    Y30_HORIZON_SECONDS,
)

logger = logging.getLogger(__name__)


class FrozenDefinitionError(RuntimeError):
    """Raised when a frozen research definition is modified in configuration."""


class TraderConfig(BaseModel):
    name: str = "std0"
    wallet: str


class PathsConfig(BaseModel):
    root: str = "data"
    raw_std0_trades: str = "data/raw/std0_trades"
    raw_api_pages: str = "data/raw/api_pages"
    raw_polymarket_book: str = "data/raw/polymarket_book"
    raw_btc_ticks: str = "data/raw/btc_ticks"
    normalized: str = "data/normalized"
    derived: str = "data/derived"
    reports: str = "data/reports"
    state: str = "data/state"
    logs: str = "data/logs"
    sessions: str = "data/sessions"


class SyncConfig(BaseModel):
    taker_only: bool = False
    page_limit: int = Field(default=500, ge=1, le=1000)
    # Live API hard cap (HTTP 400 beyond offset 10000); keep in sync with reality.
    max_offset: int = Field(default=10000, ge=0)
    time_slice_days: int = Field(default=7, ge=1)
    sleep_between_pages_seconds: float = Field(default=0.35, ge=0.0)


class BookConfig(BaseModel):
    market_slug_prefix: str = "btc-updown-5m-"
    market_window_seconds: int = Field(default=300, gt=0)
    stale_after_seconds: float = 20.0
    reconnect_backoff_base_seconds: float = 1.0
    reconnect_backoff_max_seconds: float = 60.0
    unsubscribe_grace_seconds: float = 120.0


class PolymarketConfig(BaseModel):
    data_api_base: str = "https://data-api.polymarket.com"
    gamma_api_base: str = "https://gamma-api.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    request_timeout_seconds: float = 30.0
    request_max_retries: int = 5
    request_backoff_base_seconds: float = 1.5
    sync: SyncConfig = SyncConfig()
    book: BookConfig = BookConfig()


class BtcConfig(BaseModel):
    source: str = "binance"
    ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    symbol: str = "BTCUSDT"
    stale_after_seconds: float = 30.0
    reconnect_backoff_base_seconds: float = 1.0
    reconnect_backoff_max_seconds: float = 60.0


class EpisodeConfig(BaseModel):
    rule: str = EPISODE_RULE_VERSION
    window_seconds: int = EPISODE_WINDOW_SECONDS


class Y30Config(BaseModel):
    horizon_seconds: int = Y30_HORIZON_SECONDS


class CoverageConfig(BaseModel):
    bucket_seconds: float = 1.0
    gap_threshold_seconds: float = 5.0

class LiveConfig(BaseModel):
    btc_stale_seconds: float = 5.0
    book_stale_seconds: float = 5.0
    market_prediscovery_seconds: float = 60.0
    market_overlap_seconds: float = 15.0
    market_grace_seconds: float = 5.0
    sync_interval_seconds: float = 60.0
    ledger_rebuild_interval_seconds: float = 1200.0
    health_interval_seconds: float = 5.0
    fsync_every_records: int = Field(default=100, ge=1)
    rotation_seconds: int = Field(default=3600, ge=60)
    rotation_max_bytes: int = Field(default=268435456, ge=1024)
    disk_warn_gb: float = 10.0
    disk_critical_gb: float = 2.0
    restart_max_backoff_seconds: float = 30.0
    writer_queue_batches: int = Field(default=1000, ge=10)
    restart_storm_threshold: int = Field(default=5, ge=2)
    restart_storm_window_seconds: float = Field(default=300.0, gt=0)
    proxy_probe_interval_seconds: float = Field(default=30.0, gt=0)


class Settings(BaseModel):
    trader: TraderConfig
    paths: PathsConfig = PathsConfig()
    polymarket: PolymarketConfig = PolymarketConfig()
    btc: BtcConfig = BtcConfig()
    episode: EpisodeConfig = EpisodeConfig()
    y30: Y30Config = Y30Config()
    coverage: CoverageConfig = CoverageConfig()
    live: LiveConfig = LiveConfig()

    _project_root: Path | None = PrivateAttr(default=None)


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root (directory containing ``config/settings.yaml``).

    Walks up from *start* (defaults to cwd), then falls back to the directory
    containing this package's parent (``std0-quant/``).
    """
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start.resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve().parents[2])

    for base in candidates:
        for parent in [base, *base.parents]:
            if (parent / "config" / "settings.yaml").is_file():
                return parent
    return Path(__file__).resolve().parents[2]


def _apply_env_overrides(raw: dict[str, object], *, load_env_file: bool = True) -> dict[str, object]:
    """Apply supported ``.env`` / environment overrides onto the raw YAML dict."""
    if load_env_file:
        load_dotenv()
    env_map = {
        "STD0_NAME": ("trader", "name"),
        "STD0_WALLET": ("trader", "wallet"),
        "POLYMARKET_DATA_API": ("polymarket", "data_api_base"),
        "POLYMARKET_GAMMA_API": ("polymarket", "gamma_api_base"),
        "POLYMARKET_WS_URL": ("polymarket", "ws_url"),
        "BTC_WS_URL": ("btc", "ws_url"),
    }
    for env_key, (section, key) in env_map.items():
        value = os.getenv(env_key)
        if value:
            raw.setdefault(section, {})
            if isinstance(raw[section], dict):
                raw[section][key] = value  # type: ignore[index]
                logger.debug("config override from env: %s -> %s.%s", env_key, section, key)
    return raw


def _validate_frozen(settings: Settings) -> None:
    if settings.episode.rule != EPISODE_RULE_VERSION:
        raise FrozenDefinitionError(
            f"episode.rule is '{settings.episode.rule}' but the frozen research "
            f"definition is '{EPISODE_RULE_VERSION}'. Changing the episode rule "
            "requires explicit approval; refusing to load."
        )
    if settings.episode.window_seconds != EPISODE_WINDOW_SECONDS:
        raise FrozenDefinitionError(
            f"episode.window_seconds is {settings.episode.window_seconds} but the "
            f"frozen research definition is {EPISODE_WINDOW_SECONDS} seconds. "
            "Changing the episode window requires explicit approval; refusing to load."
        )
    if settings.y30.horizon_seconds != Y30_HORIZON_SECONDS:
        raise FrozenDefinitionError(
            f"y30.horizon_seconds is {settings.y30.horizon_seconds} but the frozen "
            f"research definition is {Y30_HORIZON_SECONDS} seconds. Changing the Y30 "
            "horizon requires explicit approval; refusing to load."
        )


def load_settings(
    config_path: Path | str | None = None,
    project_root: Path | None = None,
    *,
    load_env_file: bool = True,
) -> Settings:
    """Load YAML and supported environment overrides; validate frozen definitions.

    Read-only callers can skip dotenv files with load_env_file=False. Existing
    process overrides still apply, and the default dotenv behavior is unchanged.
    """
    if type(load_env_file) is not bool:
        raise TypeError("load_env_file must be a boolean")
    root = find_project_root(Path(project_root) if project_root else None)
    path = Path(config_path) if config_path else root / "config" / "settings.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, object] = yaml.safe_load(fh) or {}
    raw = _apply_env_overrides(raw, load_env_file=load_env_file)
    settings = Settings.model_validate(raw)
    _validate_frozen(settings)
    settings._project_root = root
    return settings


@lru_cache(maxsize=8)
def _cached_settings(config_path_str: str | None, root_str: str | None) -> Settings:
    return load_settings(
        Path(config_path_str) if config_path_str else None,
        Path(root_str) if root_str else None,
    )


def get_settings() -> Settings:
    """Convenience cached accessor (tests should call ``load_settings`` directly)."""
    root = find_project_root()
    return _cached_settings(None, str(root))


def resolve_path(settings: Settings, key: str) -> Path:
    """Resolve a configured path relative to the project root."""
    root = getattr(settings, "_project_root", None)
    if root is None:
        root = find_project_root()
    value = getattr(settings.paths, key)
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path
