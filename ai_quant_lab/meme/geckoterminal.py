"""Point-in-time memecoin data from the free GeckoTerminal API.

Networks: Solana (pump.fun, PumpSwap, Meteora) and Robinhood Chain
(GeckoTerminal id `robinhood`: Uniswap v3/v4 pools and the Pons launchpad).
One database can hold both; every pool row carries its network.

Survivorship is decided at collection time, not at analysis time. A pool is
registered the moment it shows up in `new_pools`; from then on it stays in
the database whether it moons, bleeds out or gets rugged. Run the collector
continuously (cron / launchd, every minute or two) and the universe you
research on is the universe you could actually have traded.

Four tables:
    pools            one row per pool, first time we saw it
    snapshots        price, pool reserve, FDV and 5-minute flow at observation time
    token_snapshots  holders, top-10 share, mint/freeze authority, bonding-curve
                     progress of the base token at observation time
    ohlcv            5-minute candles backfilled per pool

Holders come from the per-token `info` endpoint (one call per token), so each
pass refreshes only a budget of the stalest tokens. Bonding-curve progress
comes from `tokens/multi`, 30 tokens per call, every pass.

The API has no reserve history, only the current value, so `load_panel`
rebuilds liquidity point-in-time: the last snapshot at or before each bar,
moved by the constant-product rule (TVL ∝ sqrt(price)) until the next one.

The public API is rate limited (about 30 calls per minute); `GeckoClient`
spaces calls accordingly.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_quant_lab.meme.panel import TokenPanel

API = "https://api.geckoterminal.com/api/v2"
NETWORK = "solana"
NETWORKS = ("solana", "robinhood")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pools (
    address TEXT PRIMARY KEY,
    name TEXT,
    base_token TEXT,
    quote_token TEXT,
    dex TEXT,
    created_at TEXT,
    first_seen_at TEXT NOT NULL,
    network TEXT NOT NULL DEFAULT 'solana',
    fee_pct REAL
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
CREATE TABLE IF NOT EXISTS token_snapshots (
    token TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    holders INTEGER,
    top10_pct REAL,
    mint_authority INTEGER,
    freeze_authority INTEGER,
    gt_score REAL,
    graduation_pct REAL,
    graduated INTEGER,
    PRIMARY KEY (token, observed_at)
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
    # Databases collected before multi-network support: every pool was Solana.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(pools)")}
    if "network" not in columns:
        conn.execute("ALTER TABLE pools ADD COLUMN network TEXT NOT NULL DEFAULT 'solana'")
    if "fee_pct" not in columns:
        conn.execute("ALTER TABLE pools ADD COLUMN fee_pct REAL")
    conn.commit()
    return conn


@dataclass(frozen=True)
class CollectStats:
    new_pools: int
    snapshots: int
    token_snapshots: int = 0


def collect_once(
    conn: sqlite3.Connection,
    client: GeckoClient | None = None,
    *,
    network: str = NETWORK,
    pages: int = 3,
    track_hours: float = 72.0,
    holder_calls: int = 10,
) -> CollectStats:
    """One collection pass: register new pools, snapshot every tracked pool,
    then snapshot their base tokens (`holder_calls` info calls per pass).

    Pools older than `track_hours` stop being snapshotted (their candles can
    still be backfilled).
    """
    client = client or GeckoClient()
    now = _utcnow()
    before = conn.execute("SELECT COUNT(*) FROM pools WHERE network = ?", (network,)).fetchone()[0]
    seen: set[str] = set()
    snapshots = 0

    for page in range(1, pages + 1):
        payload = client.get(f"/networks/{network}/new_pools", {"page": page})
        for pool in payload.get("data", []):
            _upsert_pool(conn, pool, now, network)
            snapshots += _insert_snapshot(conn, pool, now)
            seen.add(pool["attributes"]["address"])

    cutoff = (pd.Timestamp(now) - pd.Timedelta(hours=track_hours)).isoformat()
    tracked = [
        row[0]
        for row in conn.execute(
            "SELECT address FROM pools WHERE network = ? AND COALESCE(created_at, first_seen_at) >= ?",
            (network, cutoff),
        )
        if row[0] not in seen
    ]
    for batch in _chunks(tracked, 30):
        payload = client.get(f"/networks/{network}/pools/multi/{','.join(batch)}")
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

    token_rows = _snapshot_tokens(conn, client, network, cutoff, now, holder_calls)
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM pools WHERE network = ?", (network,)).fetchone()[0]
    return CollectStats(new_pools=after - before, snapshots=snapshots, token_snapshots=token_rows)


def _snapshot_tokens(
    conn: sqlite3.Connection, client: GeckoClient, network: str, cutoff: str, now: str, holder_calls: int
) -> int:
    tokens = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT base_token FROM pools "
            "WHERE network = ? AND base_token IS NOT NULL AND COALESCE(created_at, first_seen_at) >= ?",
            (network, cutoff),
        )
    ]
    rows: dict[str, dict[str, Any]] = {t: {} for t in tokens}

    for batch in _chunks(tokens, 30):
        payload = client.get(f"/networks/{network}/tokens/multi/{','.join(batch)}")
        for token in payload.get("data", []):
            attrs = token.get("attributes") or {}
            pad = attrs.get("launchpad_details") or {}
            if attrs.get("address") in rows and pad:
                rows[attrs["address"]].update(
                    graduation_pct=_num(pad.get("graduation_percentage")),
                    graduated=int(bool(pad.get("completed"))),
                )

    # Stalest first: never-observed tokens, then the oldest holder reading.
    last_seen = dict(
        conn.execute(
            "SELECT token, MAX(observed_at) FROM token_snapshots "
            "WHERE holders IS NOT NULL OR mint_authority IS NOT NULL GROUP BY token"
        ).fetchall()
    )
    for token in sorted(tokens, key=lambda t: last_seen.get(t) or "")[: max(holder_calls, 0)]:
        data = client.get(f"/networks/{network}/tokens/{token}/info").get("data")
        attrs = (data.get("attributes") or {}) if isinstance(data, dict) else {}
        holders = attrs.get("holders") or {}
        top10 = (holders.get("distribution_percentage") or {}).get("top_10")
        rows[token].update(
            holders=holders.get("count"),
            top10_pct=_num(top10),
            mint_authority=_flag(attrs.get("mint_authority")),
            freeze_authority=_flag(attrs.get("freeze_authority")),
            gt_score=_num(attrs.get("gt_score")),
        )

    columns = ("holders", "top10_pct", "mint_authority", "freeze_authority",
               "gt_score", "graduation_pct", "graduated")
    written = 0
    for token, values in rows.items():
        if not values:
            continue
        cursor = conn.execute(
            f"INSERT OR IGNORE INTO token_snapshots (token, observed_at, {', '.join(columns)}) "
            f"VALUES (?, ?, {', '.join('?' * len(columns))})",
            (token, now, *(values.get(c) for c in columns)),
        )
        written += cursor.rowcount
    return written


def backfill_ohlcv(
    conn: sqlite3.Connection,
    client: GeckoClient | None = None,
    *,
    addresses: Iterable[str] | None = None,
    aggregate_minutes: int = 5,
) -> int:
    """Fetch up to 1000 recent candles per pool. Returns rows inserted."""
    client = client or GeckoClient()
    network_of = dict(conn.execute("SELECT address, network FROM pools"))
    if addresses is None:
        addresses = list(network_of)
    inserted = 0
    for address in addresses:
        payload = client.get(
            f"/networks/{network_of.get(address, NETWORK)}/pools/{address}/ohlcv/minute",
            {"aggregate": aggregate_minutes, "limit": 1000, "currency": "usd"},
        )
        data = payload.get("data") or {}
        candles = (data.get("attributes") or {}).get("ohlcv_list") or []
        rows = [(address, int(c[0]), *map(float, c[1:6])) for c in candles]
        conn.executemany("INSERT OR IGNORE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        inserted += len(rows)
    conn.commit()
    return inserted


def stale_pools(
    conn: sqlite3.Connection,
    *,
    limit: int,
    track_hours: float = 84.0,
    stale_hours: float = 6.0,
    network: str | None = None,
) -> list[str]:
    """Pools worth a candle refresh: launched within `track_hours`, newest stored
    candle older than `stale_hours` (or none). Never-backfilled first, then stalest.

    One call returns the last 1000 five-minute candles (~3.5 days), so a pool
    refreshed every few hours inside that window keeps its full history.
    """
    now = pd.Timestamp.now(tz="UTC")
    cutoff = (now - pd.Timedelta(hours=track_hours)).isoformat()
    fresh_after = int((now - pd.Timedelta(hours=stale_hours)).timestamp())
    rows = conn.execute(
        "SELECT p.address, MAX(o.ts) AS last_ts FROM pools p LEFT JOIN ohlcv o ON o.address = p.address "
        "WHERE COALESCE(p.created_at, p.first_seen_at) >= ? AND (? IS NULL OR p.network = ?) "
        "GROUP BY p.address "
        "HAVING last_ts IS NULL OR last_ts < ? ORDER BY last_ts IS NOT NULL, last_ts LIMIT ?",
        (cutoff, network, network, fresh_after, limit),
    ).fetchall()
    return [row[0] for row in rows]


def load_panel(
    conn: sqlite3.Connection,
    *,
    bar: str = "5min",
    dead_after_bars: int = 12,
    rug_reserve_usd: float = 100.0,
    network: str | None = None,
) -> TokenPanel:
    """Build a point-in-time TokenPanel from the database.

    Price: candle closes; bars with no trades carry the last close forward with
    zero volume. A pool with no trade for `dead_after_bars` bars, or whose
    observed reserve fell below `rug_reserve_usd`, is dead from then on, even
    if it trades again later: a backtest cannot know a corpse will twitch.
    Liquidity: last snapshot at or before the bar, scaled by sqrt(price move).
    Before the first snapshot liquidity is unknown and the token untradeable.
    Token fields (holders, top-10 share, bonding-curve progress, mint risk)
    hold their last observed value from the observation time on; before the
    first observation they are NaN.
    `network` keeps one chain's pools; None loads every network in the file.
    Per-pool fee tiers parsed from pool names land in `panel.pool_fee`.
    """
    candles = pd.read_sql("SELECT address, ts, close, volume FROM ohlcv", conn)
    snaps = pd.read_sql(
        "SELECT address, observed_at, price_usd, reserve_usd FROM snapshots ORDER BY observed_at",
        conn,
    )
    pools = pd.read_sql(
        "SELECT address, base_token, created_at, first_seen_at, network, fee_pct FROM pools", conn
    )
    tokens = pd.read_sql("SELECT * FROM token_snapshots", conn)
    if network is not None:
        pools = pools[pools["network"] == network]
        candles = candles[candles["address"].isin(pools["address"])]
    if candles.empty:
        where = f" for network {network!r}" if network else ""
        raise ValueError(f"No candles in the database{where}. Run backfill_ohlcv first.")

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
    fields = _point_in_time_token_fields(tokens, launch["base_token"], close)
    holders = fields.pop("holders", None)
    fees = (launch["fee_pct"] / 100.0).reindex(close.columns)
    return TokenPanel(
        close=close,
        liquidity=liquidity.reindex_like(close).fillna(0.0),
        volume=volume.reindex_like(close),
        holders=holders,
        launch_time=launch_time,
        extra=fields,
        pool_fee=fees if fees.notna().any() else None,
    )


def _point_in_time_token_fields(
    tokens: pd.DataFrame, base_token: pd.Series, close: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Pool-column frames of token snapshots, as known at each bar."""
    if tokens.empty:
        return {}
    tokens = tokens.copy()
    tokens["time"] = (
        pd.to_datetime(tokens["observed_at"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    )
    tokens["top10_share"] = tokens["top10_pct"] / 100.0
    # The vendor reports negative progress for some fresh curves; clip to [0, 1].
    tokens["graduation"] = (tokens["graduation_pct"] / 100.0).clip(0.0, 1.0)
    tokens["mint_risk"] = tokens[["mint_authority", "freeze_authority"]].max(axis=1, skipna=True)
    names = {"holders": "holders", "top10_share": "top10_share",
             "graduation": "graduation", "mint_risk": "mint_risk"}
    bars = pd.DataFrame({"time": close.index.astype("datetime64[ns]")})
    out = {name: pd.DataFrame(np.nan, index=close.index, columns=close.columns) for name in names}
    by_token = {token: group.sort_values("time") for token, group in tokens.groupby("token")}
    for pool in close.columns:
        group = by_token.get(base_token.get(pool))
        if group is None:
            continue
        for column in names:
            known = group[["time", column]].dropna()
            if known.empty:
                continue
            merged = pd.merge_asof(bars, known, on="time", direction="backward")
            out[column][pool] = merged[column].to_numpy()
    return {name: frame for name, frame in out.items() if frame.notna().any().any()}


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


def _upsert_pool(conn: sqlite3.Connection, pool: dict[str, Any], now: str, network: str = NETWORK) -> None:
    attrs = pool["attributes"]
    rel = pool.get("relationships", {})
    conn.execute(
        "INSERT OR IGNORE INTO pools (address, name, base_token, quote_token, dex, created_at, "
        "first_seen_at, network, fee_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            attrs["address"],
            attrs.get("name"),
            _rel_id(rel, "base_token"),
            _rel_id(rel, "quote_token"),
            _rel_id(rel, "dex"),
            attrs.get("pool_created_at"),
            now,
            network,
            pool_fee_pct(attrs.get("name")),
        ),
    )


_FEE_IN_NAME = re.compile(r"\s(\d+(?:\.\d+)?)%\s*$")


def pool_fee_pct(name: str | None) -> float | None:
    """Fee tier from a pool name such as 'ai / WETH 1%' (Uniswap v3/v4). None if absent."""
    match = _FEE_IN_NAME.search(name or "")
    return float(match.group(1)) if match else None


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


def _flag(value: Any) -> int | None:
    """'yes' / 'no' authority flags as 1 / 0; anything else unknown."""
    if value in ("yes", "no"):
        return int(value == "yes")
    return None


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _utcnow() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
