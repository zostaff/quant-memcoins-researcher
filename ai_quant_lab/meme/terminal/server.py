"""HTTP server for the terminal: a JSON API and the static page, standard library only.

    GET  /api/market?network=solana          pulse, trending and new pools
    GET  /api/pool?network=&address=&tf=5m&usd=500
                                             one window: pool, candles, token info,
                                             screen, signals, cost of a ticket
    GET  /api/search?network=&q=             pools by name, symbol or address
    GET  /api/strategies                     the library with loaded verdicts
    GET  /api/paper                          paper account marked to live prices
    POST /api/paper/buy   {network, address, usd, strategy?}
    POST /api/paper/sell  {network, address, fraction}
    POST /api/paper/reset {starting_cash?}

Binds to 127.0.0.1 by default: the paper book has no authentication.
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ai_quant_lab.meme.amm import costs_for_network
from ai_quant_lab.meme.geckoterminal import NETWORKS
from ai_quant_lab.meme.strategies import STRATEGIES
from ai_quant_lab.meme.terminal.live import TIMEFRAMES, LiveData, RateLimited
from ai_quant_lab.meme.terminal.paper import PaperBook, quote
from ai_quant_lab.meme.terminal.signals import Pulse, screen, token_signals

_ADDRESS = re.compile(r"^[A-Za-z0-9]{20,90}$")
_STATIC = resources.files("ai_quant_lab.meme.terminal") / "static"


def load_verdicts(path: str | Path | None) -> dict[str, dict[str, str]]:
    """Verdicts from `zoo --json`: {"verdicts": {name: {"verdict", "reason"}}, "source": ...}."""
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    raw = data.get("verdicts", data)
    out = {}
    for name, value in raw.items():
        if isinstance(value, str):
            value = {"verdict": value, "reason": ""}
        out[name] = {"verdict": value.get("verdict", "untested"), "reason": value.get("reason", ""),
                     "source": data.get("source", "")}
    return out


class Terminal:
    """Everything the handlers need, independent of HTTP."""

    def __init__(self, live: LiveData | None = None, book: PaperBook | None = None,
                 verdicts: dict[str, dict[str, str]] | None = None) -> None:
        self.live = live or LiveData()
        self.book = book or PaperBook()
        self.verdicts = verdicts or {}
        self.pulse = Pulse()
        self._pulse_cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def market(self, network: str) -> dict[str, Any]:
        trending, new = self.live.trending(network), self.live.new_pools(network)
        with self._lock:
            pulse = self.pulse.update(network, trending, new)
            self._pulse_cache[network] = pulse
        return {
            "network": network,
            "pulse": pulse,
            "trending": [{**c.to_dict(), "rank": i} for i, c in enumerate(trending, start=1)],
            "new": [c.to_dict() for c in new],
        }

    def _pulse(self, network: str) -> dict[str, Any]:
        return self._pulse_cache.get(network) or self.market(network)["pulse"]

    def pool(self, network: str, address: str, timeframe: str = "5m", usd: float = 500.0) -> dict[str, Any]:
        card = self.live.pool(network, address)
        if card is None:
            raise LookupError("Pool not found")
        missing: list[str] = []

        def optional(name: str, fetch: Any, empty: Any) -> Any:
            # Candles, holders and the curve can wait for the next refresh; the window still renders.
            try:
                return fetch()
            except RateLimited:
                missing.append(name)
                return empty

        candles_5m = live_tail(optional("candles", lambda: self.live.candles(network, address, "5m"), []),
                               card.price_usd, "5m")
        chart = candles_5m if timeframe == "5m" else live_tail(optional(
            "candles", lambda: self.live.candles(network, address, timeframe), []), card.price_usd, timeframe)
        info = optional("token info", lambda: self.live.token_info(network, card.token), {})
        graduation = optional("curve", lambda: self.live.graduation(network, card.token, card.dex), None)
        rank = next((i for i, c in enumerate(self.live.trending(network), start=1) if c.address == address), None)
        signals = token_signals(card, candles_5m, info, self._pulse(network), rank, graduation,
                                {k: v["verdict"] for k, v in self.verdicts.items()})
        costs = costs_for_network(network)
        return {
            "pool": card.to_dict(),
            "candles": chart,
            "timeframe": timeframe,
            "info": info,
            "graduation_pct": graduation,
            "trending_rank": rank,
            "screen": screen(card, info),
            "signals": [s.to_dict() for s in signals],
            "quote": quote(usd, card.liquidity_usd or 0.0, costs).to_dict(),
            "position": self._position(network, address, card.price_usd, card.liquidity_usd),
            "missing": sorted(set(missing)),
        }

    def _position(self, network: str, address: str, price: float | None, liquidity: float | None) -> dict | None:
        rows = self.book.summary({(network, address): (price, liquidity)})["positions"]
        return next((p for p in rows if p["pool"] == address and p["network"] == network), None)

    def paper(self) -> dict[str, Any]:
        marks = {}
        for (network, address), p in self.book.positions().items():
            if p["tokens"] > 0:
                try:
                    card = self.live.pool(network, address)
                except RateLimited:
                    card = self.live.listed(network, address, max_age=float("inf")) or (self.live.cached(
                        ("multi", network)) or {}).get(address)
                marks[(network, address)] = (card.price_usd, card.liquidity_usd) if card else (None, 0.0)
        return self.book.summary(marks)

    def buy(self, network: str, address: str, usd: float, strategy: str | None = None) -> dict[str, Any]:
        card = self.live.pool(network, address)
        if card is None:
            raise LookupError("Pool not found")
        return self.book.buy(network, address, card.symbol, usd, card.price_usd or 0.0, card.liquidity_usd or 0.0,
                             strategy if strategy in STRATEGIES else None)

    def sell(self, network: str, address: str, fraction: float) -> dict[str, Any]:
        card = self.live.pool(network, address)
        if card is None:
            raise LookupError("Pool not found")
        return self.book.sell(network, address, fraction, card.price_usd or 0.0, card.liquidity_usd or 0.0)

    def strategies(self) -> list[dict[str, Any]]:
        return [{"name": s.name, "family": s.family, "thesis": s.thesis,
                 "variants": len(s.variants()), "requires": list(s.requires),
                 **self.verdicts.get(s.name, {"verdict": "untested", "reason": "", "source": ""})}
                for s in STRATEGIES.values()]


_TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}


def live_tail(candles: list[list[float]], price: float | None, timeframe: str,
              now: float | None = None) -> list[list[float]]:
    """Candles refresh every two minutes; the price every fifteen seconds.
    Move the last candle (or open a new one) to the live price so the chart is current."""
    if not candles or not price:
        return candles
    step = _TF_SECONDS[timeframe]
    now = time.time() if now is None else now
    out = [list(c) for c in candles]
    last = out[-1]
    if now < last[0] + step:
        last[2], last[3], last[4] = max(last[2], price), min(last[3], price), price
    else:
        start = now - now % step
        out.append([start, last[4], max(last[4], price), min(last[4], price), price, 0.0])
    return out


def _network(q: dict[str, list[str]]) -> str:
    network = (q.get("network") or ["solana"])[0]
    if network not in NETWORKS:
        raise ValueError(f"Unknown network {network!r}")
    return network


def _address(value: Any) -> str:
    if not isinstance(value, str) or not _ADDRESS.match(value):
        raise ValueError("Bad pool address")
    return value


def make_handler(terminal: Terminal) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "MemeTerminal/0.4"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            pass

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload, allow_nan=False, default=_clean).encode(), "application/json")

        def _guard(self, fn: Any) -> None:
            try:
                self._json(_sanitize(fn()))
            except RateLimited as exc:
                self._json({"error": str(exc), "rate_limited": True}, HTTPStatus.SERVICE_UNAVAILABLE)
            except (ValueError, KeyError) as exc:
                self._json({"error": str(exc).strip("'\"")}, HTTPStatus.BAD_REQUEST)
            except LookupError as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except Exception as exc:  # upstream API trouble should not kill the page
                self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_GATEWAY)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path == "/api/market":
                return self._guard(lambda: terminal.market(_network(q)))
            if url.path == "/api/pool":
                tf = (q.get("tf") or ["5m"])[0]
                if tf not in TIMEFRAMES:
                    return self._json({"error": "Bad timeframe"}, HTTPStatus.BAD_REQUEST)
                usd = float((q.get("usd") or ["500"])[0])
                return self._guard(lambda: terminal.pool(_network(q), _address((q.get("address") or [""])[0]),
                                                         tf, min(max(usd, 1.0), 1e7)))
            if url.path == "/api/search":
                return self._guard(lambda: [c.to_dict() for c in
                                            terminal.live.search(_network(q), (q.get("q") or [""])[0])])
            if url.path == "/api/strategies":
                return self._guard(terminal.strategies)
            if url.path == "/api/paper":
                return self._guard(terminal.paper)
            return self._static(url.path)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            length = min(int(self.headers.get("Content-Length") or 0), 10_000)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "Bad JSON"}, HTTPStatus.BAD_REQUEST)
            if not isinstance(body, dict):
                return self._json({"error": "Bad JSON"}, HTTPStatus.BAD_REQUEST)
            net = {"network": [str(body.get("network", "solana"))]}
            if url.path == "/api/paper/buy":
                return self._guard(lambda: terminal.buy(_network(net), _address(body.get("address")),
                                                        float(body.get("usd", 0)), body.get("strategy")))
            if url.path == "/api/paper/sell":
                return self._guard(lambda: terminal.sell(_network(net), _address(body.get("address")),
                                                         float(body.get("fraction", 1.0))))
            if url.path == "/api/paper/reset":
                return self._guard(lambda: terminal.book.reset(float(body.get("starting_cash", 10_000)))
                                   or terminal.paper())
            return self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def _static(self, path: str) -> None:
            name = "index.html" if path in ("/", "/index.html") else path.removeprefix("/static/")
            if "/" in name or name.startswith(".") or not name:
                return self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            target = _STATIC / name
            if not target.is_file():
                return self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
            self._send(200, target.read_bytes(), kind + ("; charset=utf-8" if kind.startswith("text") else ""))

    return Handler


def _sanitize(value: Any) -> Any:
    """Replace NaN and infinities with None so the JSON is valid."""
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize(v) for v in value]
    return value


def _clean(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def serve(terminal: Terminal, host: str = "127.0.0.1", port: int = 8787) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(terminal))
    server.daemon_threads = True
    return server
