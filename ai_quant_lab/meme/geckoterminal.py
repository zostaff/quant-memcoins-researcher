"""Point-in-time Solana memecoin data from the free GeckoTerminal API.

Survivorship is decided at collection time, not at analysis time. A pool is
registered the moment it shows up in `new_pools`; from then on it stays in
the database whether it moons, bleeds out or gets rugged. Run the collector
continuously (cron / launchd, every minute or two) and the universe you
research on is the universe you could actually have traded.

Three tables:
    pools      one row per pool, first time we saw it
    snapshots  price, pool reserve, FDV and 5-minute flow at observation time
    ohlcv      5-minute candles backfilled per pool

The API has no reserve history, only the current value, so `load_panel`
rebuilds liquidity point-in-time: the last snapshot at or before each bar,
moved by the constant-product rule (TVL ∝ sqrt(price)) until the next one.

The public API is rate limited (about 30 calls per minute); `GeckoClient`
spaces calls accordingly.
"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ai_quant_lab.meme.panel import TokenPanel

API = "https://api.geckoterminal.com/api/v2"
NETWORK = "solana"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pools (
    address TEXT PRIMARY KEY,
    name TEXT,
    base_token TEXT,
    quote_token TEXT,
    dex TEXT,
    created_at TEXT,
    first_seen_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    address TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    price_usd REAL,
    reserve_usd REAL,
    fdv_usd REAL,
    volume_m5 REAL,
    buys_m5 INTEGER,
    sells_m5 INTEGER,
    PRIMARY KEY (address, observed_at)
);
CREATE TABLE IF NOT EXISTS ohlcv (
    address TEXT NOT NULL,
    ts INTEGER NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (address, ts)
);
"""


class GeckoClient:
    """Minimal rate-limited JSON client."""

    def __init__(self, *, min_interval_s: float = 2.2, timeout_s: float = 20.0) -> None:
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._last_call = 0.0

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{API}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        for attempt in range(4):
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return {"data": []}
                if exc.code == 429 or exc.code >= 500:
                    time.sleep(15 * (attempt + 1))
                    continue
                raise
            except urllib.error.URLError:
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"GeckoTerminal request kept failing: {url}")


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    return conn


@dataclass(frozen=True)
class CollectStats:
    new_pools: int
    snapshots: int


def collect_once(
    conn: sqlite3.Connection,
    client: GeckoClient | None = None,
    *,
    pages: int = 3,
    track_hours: float = 72.0,
) -> CollectStats:
    """One collection pass: register new pools, snapshot every tracked pool.

    Pools older than `track_hours` stop being snapshotted (their candles can
    still be backfilled).
    """
    client = client or GeckoClient()
    now = _utcnow()
    before = conn.execute("SELECT COUNT(*) FROM pools").fetchone()[0]
    seen: set[str] = set()
    snapshots = 0

    for page in range(1, pages + 1):
        payload = client.get(f"/networks/{NETWORK}/new_pools", {"page": page})
        for pool in payload.get("data", []):
            _upsert_pool(conn, pool, now)
            snapshots += _insert_snapshot(conn, pool, now)
            seen.add(pool["attributes"]["address"])

    cutoff = (pd.Timestamp(now) - pd.Timedelta(hours=track_hours)).isoformat()
    tracked = [
        row[0]
        for row in conn.execute(
            "SELECT address FROM pools WHERE COALESCE(created_at, first_seen_at) >= ?", (cutoff,)
        )
        if row[0] not in seen
    ]
    for batch in _chunks(tracked, 30):
        payload = client.get(f"/networks/{NETWORK}/pools/multi/{','.join(batch)}")
        returned = set()
        for pool in payload.get("data", []):
            snapshots += _insert_snapshot(conn, pool, now)
            returned.add(pool["attributes"]["address"])
        # A tracked pool the API no longer returns is recorded as empty, so the
        # panel sees it die instead of silently freezing at its last price.
        for address in set(batch) - returned:
            conn.execute(
                "INSERT OR IGNORE INTO snapshots (address, observed_at, price_usd, reserve_usd) "
                "VALUES (?, ?, NULL, 0.0)",
                (address, now),
            )
            snapshots += 1

    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM pools").fetchone()[0]
    return CollectStats(new_pools=after - before, snapshots=snapshots)


def backfill_ohlcv(
    conn: sqlite3.Connection,
    client: GeckoClient | None = None,
    *,
    addresses: Iterable[str] | None = None,
    aggregate_minutes: int = 5,
) -> int:
    """Fetch up to 1000 recent candles per pool. Returns rows inserted."""
    client = client or GeckoClient()
    if addresses is None:
        addresses = [row[0] for row in conn.execute("SELECT address FROM pools")]
    inserted = 0
    for address in addresses:
        payload = client.get(
            f"/networks/{NETWORK}/pools/{address}/ohlcv/minute",
            {"aggregate": aggregate_minutes, "limit": 1000, "currency": "usd"},
        )
        data = payload.get("data") or {}
        candles = (data.get("attributes") or {}).get("ohlcv_list") or []
        rows = [(address, int(c[0]), *map(float, c[1:6])) for c in candles]
        conn.executemany("INSERT OR IGNORE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        inserted += len(rows)
    conn.commit()
    return inserted


def load_panel(
    conn: sqlite3.Connection,
    *,
    bar: str = "5min",
    dead_after_bars: int = 12,
    rug_reserve_usd: float = 100.0,
) -> TokenPanel:
    """Build a point-in-time TokenPanel from the database.

    Price: candle closes; bars with no trades carry the last close forward with
    zero volume. A pool with no trade for `dead_after_bars` bars, or whose
    observed reserve fell below `rug_reserve_usd`, is dead from then on, even
    if it trades again later: a backtest cannot know a corpse will twitch.
    Liquidity: last snapshot at or before the bar, scaled by sqrt(price move).
    Before the first snapshot liquidity is unknown and the token untradeable.
    """
    candles = pd.read_sql("SELECT address, ts, close, volume FROM ohlcv", conn)
    snaps = pd.read_sql(
        "SELECT address, observed_at, price_usd, reserve_usd FROM snapshots ORDER BY observed_at",
        conn,
    )
    pools = pd.read_sql("SELECT address, created_at, first_seen_at FROM pools", conn)
    if candles.empty:
        raise ValueError("No candles in the database. Run backfill_ohlcv first.")

    candles["time"] = pd.to_datetime(candles["ts"], unit="s").dt.floor(bar)
    close = candles.pivot_table(index="time", columns="address", values="close", aggfunc="last")
    volume = candles.pivot_table(index="time", columns="address", values="volume", aggfunc="sum")
    grid = pd.date_range(close.index.min(), close.index.max(), freq=bar)
    close = close.reindex(grid)
    volume = volume.reindex(grid)

    traded = close.notna()
    last_trade_pos = pd.DataFrame(
        np.where(traded, np.arange(len(grid))[:, None], np.nan), index=grid, columns=close.columns
    ).ffill()
    quiet_for = np.arange(len(grid))[:, None] - last_trade_pos
    went_quiet = (quiet_for >= dead_after_bars).cummax()
    close = close.ffill().mask(went_quiet)
    volume = volume.fillna(0.0).where(close.notna())

    liquidity = _point_in_time_liquidity(snaps, close, grid)
    rugged = (liquidity < rug_reserve_usd) & liquidity.notna()
    dead = rugged.cummax()
    close = close.mask(dead)
    volume = volume.mask(dead)
    liquidity = liquidity.mask(dead, 0.0).where(close.notna(), 0.0)

    close.index.name = "time"
    close.columns.name = "token"
    launch = pools.set_index("address")
    launch_time = pd.to_datetime(
        launch["created_at"].fillna(launch["first_seen_at"]), utc=True
    ).dt.tz_localize(None).reindex(close.columns)
    return TokenPanel(
        close=close,
        liquidity=liquidity.reindex_like(close).fillna(0.0),
        volume=volume.reindex_like(close),
        holders=None,
        launch_time=launch_time,
    )


def _point_in_time_liquidity(snaps: pd.DataFrame, close: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=grid, columns=close.columns)
    if snaps.empty:
        return out
    snaps = snaps.copy()
    snaps["time"] = (
        pd.to_datetime(snaps["observed_at"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    )
    bars = pd.DataFrame({"time": grid.astype("datetime64[ns]")})
    for address, group in snaps.groupby("address"):
        if address not in out.columns:
            continue
        group = group.sort_values("time")[["time", "price_usd", "reserve_usd"]]
        merged = pd.merge_asof(bars, group, on="time", direction="backward")
        price_now = close[address].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.sqrt(price_now / merged["price_usd"].to_numpy())
        scaled = merged["reserve_usd"].to_numpy() * np.where(np.isfinite(scale), scale, 1.0)
        out[address] = scaled
    return out


def _upsert_pool(conn: sqlite3.Connection, pool: dict[str, Any], now: str) -> None:
    attrs = pool["attributes"]
    rel = pool.get("relationships", {})
    conn.execute(
        "INSERT OR IGNORE INTO pools VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            attrs["address"],
            attrs.get("name"),
            _rel_id(rel, "base_token"),
            _rel_id(rel, "quote_token"),
            _rel_id(rel, "dex"),
            attrs.get("pool_created_at"),
            now,
        ),
    )


def _insert_snapshot(conn: sqlite3.Connection, pool: dict[str, Any], now: str) -> int:
    attrs = pool["attributes"]
    tx = (attrs.get("transactions") or {}).get("m5") or {}
    cursor = conn.execute(
        "INSERT OR IGNORE INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            attrs["address"],
            now,
            _num(attrs.get("base_token_price_usd")),
            _num(attrs.get("reserve_in_usd")),
            _num(attrs.get("fdv_usd")),
            _num((attrs.get("volume_usd") or {}).get("m5")),
            tx.get("buys"),
            tx.get("sells"),
        ),
    )
    return cursor.rowcount


def _rel_id(rel: dict[str, Any], key: str) -> str | None:
    data = (rel.get(key) or {}).get("data") or {}
    raw = data.get("id")
    return raw.split("_", 1)[1] if raw and "_" in raw else raw


def _num(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _utcnow() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
