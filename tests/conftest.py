"""Shared pytest fixtures for std0-quant tests."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from std0_quant.config import load_settings, resolve_path  # noqa: E402
from std0_quant.storage import (  # noqa: E402
    AppendOnlyNDJSON,
    RawPageStore,
    SqliteState,
)

STD0_WALLET = "0xdf7930e89a2c47560165331863c31deca0733dcd"


@pytest.fixture(autouse=True)
def _agent_shadow_offline(request, monkeypatch):
    """Agent-shadow tests must never reach a network, loopback proxies included."""
    if not request.module.__name__.startswith("test_agent_shadow"):
        yield
        return
    import socket

    # Last-step backstop. Each replacement first raises the audit event the real function would,
    # so the transports' audit-hook guard still sees (and records) the attempt before this refuses.
    def refuse(event):
        def blocked(*args, **_kwargs):
            sys.audit(event, *args[:2])
            raise OSError("network access is blocked in agent-shadow tests")
        return blocked

    monkeypatch.setattr(socket.socket, "connect", refuse("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", refuse("socket.connect"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("socket.getaddrinfo"))
    yield


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A throwaway project directory with the real frozen settings file."""
    (tmp_path / "config").mkdir()
    shutil.copy(
        PROJECT_ROOT / "config" / "settings.yaml",
        tmp_path / "config" / "settings.yaml",
    )
    return tmp_path


@pytest.fixture()
def settings(project: Path):
    return load_settings(project_root=project)


class SyncHarness:
    """Wires a Std0TradesSyncer against a FakeTradesAPI and tmp storage."""

    def __init__(self, settings, records: list[dict], **api_kwargs) -> None:
        self.settings = settings
        self.state = SqliteState(
            resolve_path(settings, "state") / "sync_state.db"
        )
        self.writer = AppendOnlyNDJSON(
            resolve_path(settings, "raw_std0_trades") / "trades.ndjson"
        )
        self.page_store = RawPageStore(resolve_path(settings, "raw_api_pages"))
        self.api = FakeTradesAPI(records, **api_kwargs)
        from std0_quant.collectors.std0_trades import RetryingClient, Std0TradesSyncer

        self.client = RetryingClient(
            base_url="http://fake-api.local",
            fetch_fn=self.api,
            max_retries=0,
            backoff_base_seconds=0.0,
            sleeper=lambda _s: None,
        )
        self.syncer = Std0TradesSyncer(
            settings, self.state, self.writer, self.page_store, client=self.client
        )

    def raw_lines(self) -> list[dict]:
        from std0_quant.storage import read_ndjson

        return list(read_ndjson(self.writer.path))

    def close(self) -> None:
        self.writer.close()
        self.state.close()


@pytest.fixture()
def make_harness(settings):
    """Factory fixture: harness(records, **api_kwargs) -> SyncHarness (auto-closed)."""
    harnesses: list[SyncHarness] = []

    def _make(records: list[dict], **api_kwargs) -> SyncHarness:
        harness = SyncHarness(settings, records, **api_kwargs)
        harnesses.append(harness)
        return harness

    yield _make
    for harness in harnesses:
        harness.close()


class FakeTradesAPI:
    """Emulates ``GET /trades`` with offset/limit pagination (newest first).

    ``enforce_offset_cap`` mimics the live API's hard cap: any offset beyond
    the cap returns HTTP 400 ``{"error":"max historical trades offset of N
    exceeded"}``. ``honor_time_params`` applies the ``start``/``end`` (epoch
    seconds) filters like the live API does.
    """

    def __init__(
        self,
        records: list[dict],
        max_limit: int = 500,
        enforce_offset_cap: int | None = None,
        honor_time_params: bool = False,
    ) -> None:
        self.records = records
        self.max_limit = max_limit
        self.enforce_offset_cap = enforce_offset_cap
        self.honor_time_params = honor_time_params
        self.calls: list[tuple[str, dict]] = []
        self.scripted_failures: list[tuple[int, str]] = []

    def __call__(self, url: str, params: dict) -> tuple[int, str]:
        self.calls.append((url, dict(params)))
        if self.scripted_failures:
            status, body = self.scripted_failures.pop(0)
            return status, body
        offset = int(params.get("offset", 0))
        if (
            self.enforce_offset_cap is not None
            and offset > self.enforce_offset_cap
        ):
            return 400, json.dumps({
                "error": f"max historical trades offset of "
                         f"{self.enforce_offset_cap} exceeded",
            })
        limit = min(int(params.get("limit", 100)), self.max_limit)
        records = self.records
        if self.honor_time_params:
            start = int(params["start"]) if "start" in params else None
            end = int(params["end"]) if "end" in params else None
            records = [
                r for r in records
                if (start is None or int(r["timestamp"]) >= start)
                and (end is None or int(r["timestamp"]) <= end)
            ]
        page = records[offset : offset + limit]
        return 200, json.dumps(page)

    def last_params(self) -> dict:
        return self.calls[-1][1]


def make_trade(
    tx: str,
    ts: int | str,
    *,
    side: str = "BUY",
    outcome: str = "Up",
    outcome_index: int = 0,
    size: str | float = 100,
    price: str | float = 0.5,
    condition_id: str = "0xconditionA",
    asset: str | None = None,
    wallet: str = STD0_WALLET,
    slug: str = "bitcoin-up-or-down-jan-1-1200pm-et",
    title: str = "Bitcoin Up or Down - Jan 1, 12:00 PM ET",
    extra: dict | None = None,
) -> dict:
    """Build one raw data-api trade record (schema documented in README)."""
    record = {
        "proxyWallet": wallet,
        "side": side,
        "asset": asset or f"token-{outcome.lower()}-{condition_id[-6:]}",
        "conditionId": condition_id,
        "size": str(size),
        "price": str(price),
        "timestamp": str(ts),
        "title": title,
        "slug": slug,
        "outcome": outcome,
        "outcomeIndex": outcome_index,
        "transactionHash": tx,
        "name": "std0",
        "pseudonym": "std0",
    }
    if extra:
        record.update(extra)
    return record
