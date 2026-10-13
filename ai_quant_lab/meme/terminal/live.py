"""Live market data for the terminal: two free APIs behind a TTL cache.

GeckoTerminal has candles, trending lists, holders and contract flags, but its
free tier answers only about ten calls a minute in practice. DexScreener has
no candles, but serves price, liquidity, volume and buy/sell counts for up to
30 pools per call at 300 calls a minute. So the fast-moving numbers come from
DexScreener and GeckoTerminal is spent only on what nobody else has:

    pools in windows       15 s   DexScreener, one call for every watched pool
    trending / new pools  120 s   GeckoTerminal, one call each per network
    candles               120 s   GeckoTerminal; the last candle follows the live price
    token info             30 min GeckoTerminal: holders, top-10 share, authorities, dev share
    curve progress          5 min GeckoTerminal, launchpad pools only

Calls are serialised through one lock. A stale entry is served at once and
refreshed by a background worker, so a slow or rate-limited API never freezes
the page. On HTTP 429 the client stops calling for a cool-down (20 s, growing
to at most a minute) instead of sleeping inside a request.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from ai_quant_lab.meme.geckoterminal import API, NETWORKS, _flag, _num
from ai_quant_lab.meme.narratives import classify

TIMEFRAMES: dict[str, tuple[str, int]] = {
    "1m": ("minute", 1),
    "5m": ("minute", 5),
    "15m": ("minute", 15),
    "1h": ("hour", 1),
    "4h": ("hour", 4),
}
WINDOWS = ("m5", "h1", "h6", "h24")
# Dex ids whose pools are bonding curves. "pumpswap" is the post-migration AMM, not a curve.
CURVE_DEXES = ("pump-fun", "pons", "meteora-dbc", "stonkfun", "bankr", "moonshot", "letsbonk", "launchlab")


class RateLimited(RuntimeError):
    """The API asked us to slow down; try again after the cool-down."""


class TerminalClient:
    """GeckoTerminal client that fails fast instead of sleeping through 429s."""

    def __init__(self, *, min_interval_s: float = 3.0, timeout_s: float = 15.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.clock = clock
        self._last_call = -1e9
        self._cooldown_until = 0.0
        self._backoff = 20.0
        self._lock = threading.Lock()

    @property
    def cooling_down(self) -> float:
        return max(self._cooldown_until - self.clock(), 0.0)

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{API}{path}" + ("?" + urllib.parse.urlencode(params) if params else "")
        with self._lock:
            if self.cooling_down:
                raise RateLimited(f"GeckoTerminal rate limit, retry in {self.cooling_down:.0f}s")
            wait = self.min_interval_s - (self.clock() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = self.clock()
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.loads(response.read())
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return {"data": []}
                if exc.code == 429:
                    self._cooldown_until = self.clock() + self._backoff
                    self._backoff = min(self._backoff * 1.5, 60.0)
                    raise RateLimited("GeckoTerminal rate limit") from None
                raise
            self._backoff = 20.0
            return payload


DEXSCREENER = "https://api.dexscreener.com/latest/dex"


class DexScreenerClient:
    """Batch pair lookups from DexScreener (chain ids match GeckoTerminal's here)."""

    def __init__(self, *, min_interval_s: float = 0.3, timeout_s: float = 10.0) -> None:
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._last_call = -1e9
        self._lock = threading.Lock()

    def pairs(self, network: str, addresses: list[str]) -> list[dict[str, Any]]:
        url = f"{DEXSCREENER}/pairs/{network}/{','.join(addresses)}"
        with self._lock:
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            request = urllib.request.Request(url, headers={"Accept": "application/json",
                                                           "User-Agent": "meme-quant-terminal"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.loads(response.read())
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    raise RateLimited("DexScreener rate limit") from None
                raise
        return payload.get("pairs") or []


@dataclass
class PoolCard:
    """One pool as the terminal shows it."""

    network: str
    address: str
    token: str | None
    name: str
    symbol: str
    dex: str | None
    image: str | None
    narrative: str
    price_usd: float | None
    liquidity_usd: float | None
    fdv_usd: float | None
    created_at: str | None
    age_hours: float | None
    change: dict[str, float | None] = field(default_factory=dict)
    volume: dict[str, float | None] = field(default_factory=dict)
    buys: dict[str, int] = field(default_factory=dict)
    sells: dict[str, int] = field(default_factory=dict)

    def buy_share(self, window: str = "h1") -> float | None:
        b, s = self.buys.get(window, 0), self.sells.get(window, 0)
        return b / (b + s) if b + s else None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["buy_share"] = {w: self.buy_share(w) for w in WINDOWS}
        return out


def parse_pool(pool: dict[str, Any], tokens: dict[str, dict[str, Any]], network: str) -> PoolCard:
    """A GeckoTerminal pool object (plus `included` tokens) as a `PoolCard`."""
    a = pool.get("attributes") or {}
    rel = pool.get("relationships") or {}
    token_id = ((rel.get("base_token") or {}).get("data") or {}).get("id")
    dex = ((rel.get("dex") or {}).get("data") or {}).get("id")
    t = (tokens.get(token_id) or {}).get("attributes") or {}
    pool_name = a.get("name") or ""
    symbol = t.get("symbol") or pool_name.split(" / ")[0] or "?"
    name = t.get("name") or symbol
    created = a.get("pool_created_at")
    age = None
    if created:
        started = datetime.fromisoformat(created.replace("Z", "+00:00"))
        age = (datetime.now(UTC) - started).total_seconds() / 3600
    tx = a.get("transactions") or {}
    return PoolCard(
        network=network,
        address=a.get("address") or "",
        token=t.get("address") or (token_id.split("_", 1)[1] if token_id else None),
        name=name,
        symbol=symbol,
        dex=dex,
        image=t.get("image_url") if (t.get("image_url") or "").startswith("https://") else None,
        narrative=classify(name, symbol),
        price_usd=_num(a.get("base_token_price_usd")),
        liquidity_usd=_num(a.get("reserve_in_usd")),
        fdv_usd=_num(a.get("fdv_usd")),
        created_at=created,
        age_hours=age,
        change={w: _num((a.get("price_change_percentage") or {}).get(w)) for w in WINDOWS},
        volume={w: _num((a.get("volume_usd") or {}).get(w)) for w in WINDOWS},
        buys={w: int((tx.get(w) or {}).get("buys") or 0) for w in WINDOWS},
        sells={w: int((tx.get(w) or {}).get("sells") or 0) for w in WINDOWS},
    )


def parse_dex_pair(pair: dict[str, Any], network: str) -> PoolCard:
    """A DexScreener pair as a `PoolCard`."""
    base = pair.get("baseToken") or {}
    symbol = base.get("symbol") or "?"
    name = base.get("name") or symbol
    created_ms = pair.get("pairCreatedAt")
    created = age = None
    if created_ms:
        started = datetime.fromtimestamp(created_ms / 1000, UTC)
        created = started.isoformat()
        age = (datetime.now(UTC) - started).total_seconds() / 3600
    tx = pair.get("txns") or {}
    image = (pair.get("info") or {}).get("imageUrl") or ""
    labels = pair.get("labels") or []
    dex = pair.get("dexId")
    return PoolCard(
        network=network,
        address=pair.get("pairAddress") or "",
        token=base.get("address"),
        name=name,
        symbol=symbol,
        dex=f"{dex}-{labels[0]}" if dex and labels else dex,
        image=image if image.startswith("https://") else None,
        narrative=classify(name, symbol),
        price_usd=_num(pair.get("priceUsd")),
        liquidity_usd=_num((pair.get("liquidity") or {}).get("usd")),
        fdv_usd=_num(pair.get("fdv")),
        created_at=created,
        age_hours=age,
        change={w: _num((pair.get("priceChange") or {}).get(w)) for w in WINDOWS},
        volume={w: _num((pair.get("volume") or {}).get(w)) for w in WINDOWS},
        buys={w: int((tx.get(w) or {}).get("buys") or 0) for w in WINDOWS},
        sells={w: int((tx.get(w) or {}).get("sells") or 0) for w in WINDOWS},
    )


def parse_token_info(payload: dict[str, Any]) -> dict[str, Any]:
    """Holders, concentration and contract flags from `tokens/{address}/info`."""
    data = payload.get("data")
    a = (data.get("attributes") or {}) if isinstance(data, dict) else {}
    holders = a.get("holders") or {}
    return {
        "holders": holders.get("count"),
        "top10_pct": _num((holders.get("distribution_percentage") or {}).get("top_10")),
        "mint_authority": _flag(a.get("mint_authority")),
        "freeze_authority": _flag(a.get("freeze_authority")),
        "dev_pct": _num(a.get("developer_holding_percentage")),
        "honeypot": _flag(a.get("is_honeypot")),
        "gt_score": _num(a.get("gt_score")),
        "twitter": a.get("twitter_handle"),
        "websites": a.get("websites") or [],
        "description": (a.get("description") or "")[:400],
    }


def _included_tokens(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in payload.get("included") or [] if item.get("type") == "token"}


class LiveData:
    """Cached access to the endpoints the terminal needs."""

    def __init__(self, client: Any | None = None, dex: Any | None = None, *,
                 clock: Callable[[], float] = time.monotonic, background: bool = True) -> None:
        """`dex=None` with a custom `client` keeps everything on that client (tests);
        the default pairs GeckoTerminal with DexScreener."""
        self.dex = dex if dex is not None or client is not None else DexScreenerClient()
        self.client = client or TerminalClient()
        self.clock = clock
        self._cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._queued: set[tuple[Any, ...]] = set()
        self._watched: dict[str, list[str]] = {}
        self._missing: dict[tuple[str, str], set[str]] = {}
        self._jobs: queue.Queue[tuple[tuple[Any, ...], Callable[[], Any]]] = queue.Queue()
        self.background = background
        if background:
            threading.Thread(target=self._worker, daemon=True, name="live-refresh").start()

    def _store(self, key: tuple[Any, ...], fetch: Callable[[], Any]) -> Any:
        # Only GeckoTerminal calls queue behind each other; DexScreener has its own lock.
        if key[0] == "pools":
            value = fetch()
        else:
            with self._lock:
                value = fetch()
        self._cache[key] = (self.clock(), value)
        return value

    def _worker(self) -> None:
        while True:
            key, fetch = self._jobs.get()
            try:
                self._store(key, fetch)
            except Exception:  # noqa: BLE001 - keep serving the stale value
                pass
            finally:
                self._queued.discard(key)

    def _cached(self, key: tuple[Any, ...], ttl: float, fetch: Callable[[], Any]) -> Any:
        """Fresh value, or the stale one while a refresh is queued; fetch inline only when empty."""
        hit = self._cache.get(key)
        if hit and self.clock() - hit[0] < ttl:
            return hit[1]
        if hit and not self.background:
            return self._store(key, fetch)
        if hit:
            if key not in self._queued:
                self._queued.add(key)
                self._jobs.put((key, fetch))
            return hit[1]
        return self._store(key, fetch)

    def cached(self, key: tuple[Any, ...]) -> Any | None:
        hit = self._cache.get(key)
        return hit[1] if hit else None

    @staticmethod
    def _check(network: str) -> None:
        if network not in NETWORKS:
            raise ValueError(f"Unknown network {network!r}; use one of {', '.join(NETWORKS)}")

    def _pools(self, network: str, path: str, params: dict[str, Any]) -> list[PoolCard]:
        payload = self.client.get(path, {"include": "base_token,dex", **params})
        tokens = _included_tokens(payload)
        return [parse_pool(p, tokens, network) for p in payload.get("data") or []]

    def trending(self, network: str) -> list[PoolCard]:
        self._check(network)
        return self._cached(("trending", network), 120,
                            lambda: self._pools(network, f"/networks/{network}/trending_pools", {}))

    def new_pools(self, network: str) -> list[PoolCard]:
        self._check(network)
        return self._cached(("new", network), 120,
                            lambda: self._pools(network, f"/networks/{network}/new_pools", {}))

    def search(self, network: str, query: str) -> list[PoolCard]:
        self._check(network)
        query = query.strip()[:64]
        if not query:
            return []
        return self._cached(("search", network, query.lower()), 120,
                            lambda: self._pools(network, "/search/pools", {"query": query, "network": network}))

    def listed(self, network: str, address: str, max_age: float = 60) -> PoolCard | None:
        """The pool from a fresh trending or new-pools list, without a request."""
        for kind in ("trending", "new"):
            hit = self._cache.get((kind, network))
            if hit and self.clock() - hit[0] < max_age:
                card = next((c for c in hit[1] if c.address == address), None)
                if card is not None:
                    return card
        return None

    def pool(self, network: str, address: str) -> PoolCard | None:
        """One pool. Every pool asked for is watched, and all watched pools of a
        network are refreshed together in one call (DexScreener, or GeckoTerminal
        `pools/multi` without it). A pool missing from the batch falls back to
        the last trending or new-pools list it appeared in."""
        self._check(network)
        watched = self._watched.setdefault(network, [])
        if address not in watched:
            watched.append(address)
            del watched[:-30]                       # both batch endpoints take up to 30 addresses
        key = ("pools", network) if self.dex is not None else ("multi", network)
        ttl = 15 if self.dex is not None else 30
        fetch = (lambda: self._dex_batch(network, list(watched))) if self.dex is not None \
            else (lambda: self._multi(network, list(watched)))
        try:
            cards = self._cached(key, ttl, fetch)
            if address not in cards and address not in self._missing.get(key, set()):
                cards = self._store(key, fetch)     # newly watched: fetch now, not on the next refresh
        except RateLimited:
            cards = self.cached(key) or {}
        card = cards.get(address) or self.listed(network, address, max_age=float("inf"))
        if card is None and not cards and self.cached(key) is None:
            raise RateLimited("Pool data not available yet")
        return card

    def _dex_batch(self, network: str, addresses: list[str]) -> dict[str, PoolCard]:
        cards = {}
        for pair in self.dex.pairs(network, addresses):
            card = parse_dex_pair(pair, network)
            cards[card.address] = card
            cards.setdefault(card.address.lower(), card)
        # Remember who DexScreener does not know, so they do not trigger a refetch every call.
        self._missing[("pools", network)] = {a for a in addresses if a not in cards and a.lower() not in cards}
        return {a: cards.get(a) or cards.get(a.lower()) for a in addresses if a in cards or a.lower() in cards}

    def _multi(self, network: str, addresses: list[str]) -> dict[str, PoolCard]:
        payload = self.client.get(f"/networks/{network}/pools/multi/{','.join(addresses)}",
                                  {"include": "base_token,dex"})
        tokens = _included_tokens(payload)
        cards = {c.address: c for c in (parse_pool(p, tokens, network) for p in payload.get("data") or [])}
        self._missing[("multi", network)] = set(addresses) - set(cards)
        return cards

    def candles(self, network: str, address: str, timeframe: str = "5m", limit: int = 288) -> list[list[float]]:
        """Oldest-first `[ts, open, high, low, close, volume]` rows."""
        self._check(network)
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"Unknown timeframe {timeframe!r}")
        unit, aggregate = TIMEFRAMES[timeframe]

        def fetch() -> list[list[float]]:
            payload = self.client.get(f"/networks/{network}/pools/{address}/ohlcv/{unit}",
                                      {"aggregate": aggregate, "limit": limit, "currency": "usd"})
            data = payload.get("data")
            rows = ((data.get("attributes") or {}).get("ohlcv_list") or []) if isinstance(data, dict) else []
            return sorted([float(v) for v in row[:6]] for row in rows if len(row) >= 6)

        return self._cached(("candles", network, address, timeframe), 120, fetch)

    def graduation(self, network: str, token: str | None, dex: str | None = None) -> float | None:
        """Bonding-curve progress in percent; None without a curve or once it has migrated.

        Only launchpad pools (pump.fun, Pons and the like) have a curve, so other
        dexes skip the request.
        """
        self._check(network)
        if not token or (dex is not None and not any(k in dex for k in CURVE_DEXES)):
            return None

        def fetch() -> float | None:
            data = self.client.get(f"/networks/{network}/tokens/{token}").get("data")
            pad = ((data.get("attributes") or {}).get("launchpad_details") or {}) if isinstance(data, dict) else {}
            if not pad or pad.get("completed"):
                return None
            return _num(pad.get("graduation_percentage"))

        return self._cached(("curve", network, token), 300, fetch)

    def token_info(self, network: str, token: str | None) -> dict[str, Any]:
        self._check(network)
        if not token:
            return {}
        return self._cached(("info", network, token), 1800,
                            lambda: parse_token_info(self.client.get(f"/networks/{network}/tokens/{token}/info")))
