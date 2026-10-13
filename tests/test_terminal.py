"""Terminal: live data parsing and caching, per-token signals, paper book, HTTP API."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from ai_quant_lab.meme.amm import costs_for_network, trade_cost_usd
from ai_quant_lab.meme.terminal import (
    LiveData,
    PaperBook,
    Pulse,
    Terminal,
    screen,
    serve,
    token_signals,
)
from ai_quant_lab.meme.terminal.live import RateLimited, parse_pool

POOL = "P1" + "a" * 30
OTHER_POOL = "P2" + "b" * 30


def _pool(address, symbol="WIF2", name="dog wif hat 2", price="0.001", tvl="50000", h1=12.0):
    return {
        "attributes": {"address": address, "name": f"{symbol} / SOL", "pool_created_at": "2026-01-01T00:00:00Z",
                       "base_token_price_usd": price, "reserve_in_usd": tvl, "fdv_usd": "900000",
                       "price_change_percentage": {"m5": "1", "h1": str(h1), "h6": "5", "h24": "40"},
                       "volume_usd": {"m5": "100", "h1": "5000", "h6": "20000", "h24": "60000"},
                       "transactions": {"h1": {"buys": 70, "sells": 30}}},
        "relationships": {"base_token": {"data": {"id": f"solana_T{address}"}},
                          "dex": {"data": {"id": "pumpswap"}}},
    }


def _token(address, symbol="WIF2", name="dog wif hat 2"):
    return {"id": f"solana_T{address}", "type": "token",
            "attributes": {"address": f"T{address}", "name": name, "symbol": symbol,
                           "image_url": "https://example.com/i.png"}}


def _candles(n=288, start=1_700_000_000):
    rng = np.random.default_rng(0)
    close = 0.001 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return [[start + 300 * i, c, c * 1.01, c * 0.99, c, 100.0] for i, c in enumerate(close)]


class FakeGecko:
    def __init__(self):
        self.calls = []
        self.limited = False

    def get(self, path, params=None):
        self.calls.append(path)
        if self.limited:
            raise RateLimited("slow down")
        if path.endswith("/trending_pools") or path.endswith("/new_pools"):
            return {"data": [_pool(POOL)], "included": [_token(POOL)]}
        if "/pools/multi/" in path:
            addresses = path.rsplit("/", 1)[1].split(",")
            return {"data": [_pool(a, "CAT", "cat coin") for a in addresses],
                    "included": [_token(a, "CAT", "cat coin") for a in addresses]}
        if "/ohlcv/" in path:
            rows = [[r[0], *r[1:]] for r in reversed(_candles())]          # API is newest first
            return {"data": {"attributes": {"ohlcv_list": rows}}}
        if path.endswith("/info"):
            return {"data": {"attributes": {"holders": {"count": 1234, "distribution_percentage": {"top_10": "22.5"}},
                                            "mint_authority": "no", "freeze_authority": "no",
                                            "developer_holding_percentage": "3", "is_honeypot": "no"}}}
        return {"data": []}


# ---------------------------------------------------------------- live data


def test_parse_pool_reads_flow_and_narrative():
    card = parse_pool(_pool(POOL), {f"solana_T{POOL}": _token(POOL)}, "solana")
    assert (card.symbol, card.narrative, card.dex) == ("WIF2", "dog", "pumpswap")
    assert card.price_usd == pytest.approx(0.001) and card.liquidity_usd == 50_000
    assert card.buy_share("h1") == pytest.approx(0.7)
    assert card.image == "https://example.com/i.png"


def test_cache_serves_fresh_values_without_calls():
    now = [0.0]
    live = LiveData(FakeGecko(), clock=lambda: now[0], background=False)
    live.trending("solana")
    live.trending("solana")
    assert len(live.client.calls) == 1
    now[0] = 121.0
    live.trending("solana")
    assert len(live.client.calls) == 2


def test_watched_pools_are_batched_and_fall_back_to_lists():
    live = LiveData(FakeGecko(), background=False)
    third = "P3" + "c" * 30
    live.pool("solana", OTHER_POOL)
    live.pool("solana", third)
    assert live.pool("solana", OTHER_POOL).narrative == "cat"
    multi = [c for c in live.client.calls if "/pools/multi/" in c]
    assert multi[-1].endswith(f"{OTHER_POOL},{third}")                # one call carries every watched pool
    live.client.limited = True
    assert live.pool("solana", third).symbol == "CAT"                 # stale batch served under a rate limit


class FakeDex:
    def __init__(self):
        self.calls = []

    def pairs(self, network, addresses):
        self.calls.append(addresses)
        return [{"pairAddress": a, "dexId": "uniswap", "labels": ["v4"], "priceUsd": "0.5",
                 "baseToken": {"address": "0xT", "name": "Doggo Coin", "symbol": "DOGGO"},
                 "liquidity": {"usd": 40000}, "txns": {"h1": {"buys": 9, "sells": 1}},
                 "volume": {"h1": 1000}, "priceChange": {"h1": 4.2}, "pairCreatedAt": 1_790_000_000_000}
                for a in addresses if a != OTHER_POOL]


def test_dexscreener_serves_cards_and_misses_fall_back_to_lists():
    gecko, dex = FakeGecko(), FakeDex()
    live = LiveData(gecko, dex, background=False)
    card = live.pool("robinhood", "0x" + "a" * 64)
    assert (card.symbol, card.dex, card.narrative) == ("DOGGO", "uniswap-v4", "dog")
    assert card.buy_share("h1") == pytest.approx(0.9) and gecko.calls == []
    live.trending("solana")
    assert live.pool("solana", OTHER_POOL) is None or live.pool("solana", OTHER_POOL).address == OTHER_POOL
    calls = len(dex.calls)
    live.pool("solana", OTHER_POOL)
    assert len(dex.calls) == calls                                    # an unknown pool does not refetch every call


def test_candles_come_back_oldest_first():
    rows = LiveData(FakeGecko(), background=False).candles("solana", POOL)
    assert len(rows) == 288 and rows[0][0] < rows[-1][0]


def test_unknown_network_is_refused():
    with pytest.raises(ValueError):
        LiveData(FakeGecko(), background=False).trending("ethereum")


# ---------------------------------------------------------------- signals


def _card(**kw):
    card = parse_pool(_pool(POOL, **kw), {f"solana_T{POOL}": _token(POOL)}, "solana")
    card.age_hours = 48.0
    return card


def test_screen_matches_the_library_universe():
    assert screen(_card(), {"mint_authority": 0, "top10_pct": 20}) == []
    fails = screen(_card(tvl="4000"), {"mint_authority": 1, "top10_pct": 80, "honeypot": 1})
    assert len(fails) == 4


def test_signals_fire_and_carry_verdicts():
    candles = _candles()
    candles[-1][5] = 50_000.0                                      # volume spike on the last bar
    candles[-1][4] = candles[-7][4] * 1.2                          # with the price up
    pulse = {"regime": "RISK-ON", "narratives": {"dog": {"heat": 0.4}, "cat": {"heat": -0.2}}}
    sigs = {s.strategy: s for s in token_signals(_card(), candles, {}, pulse, trending_rank=3,
                                                  verdicts={"buy_pressure": "LUCK"})}
    assert sigs["volume_breakout"].state == "on"
    assert sigs["buy_pressure"].state == "on" and sigs["buy_pressure"].verdict == "LUCK"
    assert sigs["trending_entry"].state == "on"
    assert sigs["narrative_rotation"].state == "on"
    assert sigs["curve_graduation"].state == "na"
    assert sigs["volume_breakout"].verdict == "untested"


def test_failed_screen_blocks_signals():
    sigs = token_signals(_card(tvl="3000"), _candles(), {}, {"narratives": {}}, trending_rank=1)
    trending = next(s for s in sigs if s.strategy == "trending_entry")
    assert trending.state == "blocked" and "TVL" in trending.reason


def test_pulse_switches_to_risk_appetite_with_history():
    import pandas as pd

    pulse = Pulse(min_history=5)
    cards = [_card(), parse_pool(_pool(OTHER_POOL, "CAT", h1=-5), {}, "solana")]
    for i in range(6):
        out = pulse.update("solana", cards, [], now=pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i))
    assert out["readings"] == 6 and out["regime"] in {"RISK-ON", "RISK-OFF", "NEUTRAL"}
    assert set(out["narratives"]) == {"dog", "cat"}


# ---------------------------------------------------------------- paper book


def test_round_trip_pays_the_backtest_costs():
    book = PaperBook()
    costs = costs_for_network("solana")
    book.buy("solana", POOL, "WIF2", 1_000, price=0.001, liquidity_usd=50_000)
    position = book.positions()[("solana", POOL)]
    assert position["tokens"] == pytest.approx((1_000 - float(trade_cost_usd(1_000, 50_000, costs))) / 0.001)
    book.sell("solana", POOL, 1.0, price=0.001, liquidity_usd=50_000)
    lost = 10_000 - book.cash()
    assert 0.08 < lost / 1_000 < 0.11      # two legs: 1% fee + ~3.8% curve on a $25k reserve + MEV, each
    summary = book.summary({})
    assert summary["positions"] == [] and summary["realized"] == pytest.approx(-lost)


def test_liquidation_value_is_below_mark():
    book = PaperBook()
    book.buy("solana", POOL, "WIF2", 2_000, price=0.001, liquidity_usd=20_000)
    s = book.summary({("solana", POOL): (0.002, 20_000)})
    assert s["liquidation_value"] < s["mark_value"]
    assert s["positions"][0]["unrealized"] > 0


def test_paper_book_refuses_bad_orders():
    book = PaperBook()
    with pytest.raises(ValueError, match="cash"):
        book.buy("solana", POOL, "X", 20_000, 0.001, 1e6)
    with pytest.raises(ValueError, match="TVL"):
        book.buy("solana", POOL, "X", 100, 0.001, 1_000)
    with pytest.raises(ValueError, match="position"):
        book.sell("solana", POOL, 1.0, 0.001, 1e6)


# ---------------------------------------------------------------- http


@pytest.fixture()
def server():
    gecko = FakeGecko()
    terminal = Terminal(LiveData(gecko, background=False), PaperBook())
    srv = serve(terminal, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", gecko
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_api_end_to_end(server):
    base, _ = server
    market = json.loads(_get(f"{base}/api/market?network=solana")[1])
    assert market["trending"][0]["rank"] == 1 and market["pulse"]["regime"]
    window = json.loads(_get(f"{base}/api/pool?network=solana&address={POOL}&tf=5m&usd=500")[1])
    assert len(window["candles"]) >= 288 and window["info"]["holders"] == 1234
    assert window["quote"]["cost_fraction"] > 0.01 and len(window["signals"]) == 7
    fill = _post(f"{base}/api/paper/buy", {"network": "solana", "address": POOL, "usd": 500})
    assert fill["side"] == "buy" and fill["cost_usd"] > 0
    account = json.loads(_get(f"{base}/api/paper")[1])
    assert account["cash"] == pytest.approx(9_500) and len(account["positions"]) == 1
    assert b"Meme Quant Terminal" in _get(f"{base}/")[1]
    assert _get(f"{base}/static/app.js")[0] == 200


def test_api_rejects_bad_input(server):
    base, _ = server
    for path in ("/api/pool?network=solana&address=../../etc", "/api/market?network=eth",
                 f"/api/pool?network=solana&address={POOL}&tf=7m", "/static/../server.py"):
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base + path)
        assert err.value.code in (400, 404)


def test_live_tail_moves_the_last_candle():
    from ai_quant_lab.meme.terminal.server import live_tail

    rows = [[0, 1.0, 1.0, 1.0, 1.0, 5.0], [300, 1.0, 1.1, 0.9, 1.0, 5.0]]
    same = live_tail(rows, 1.5, "5m", now=400)
    assert len(same) == 2 and same[-1][2:5] == [1.5, 0.9, 1.5] and rows[-1][4] == 1.0
    new = live_tail(rows, 0.8, "5m", now=650)
    assert new[-1] == [600, 1.0, 1.0, 0.8, 0.8, 0.0]


def test_window_renders_partially_when_rate_limited(server):
    base, gecko = server
    _get(f"{base}/api/market?network=solana")                       # the card is in the trending list
    gecko.limited = True
    window = json.loads(_get(f"{base}/api/pool?network=solana&address={POOL}")[1])
    assert window["candles"] == [] and set(window["missing"]) >= {"candles", "token info"}
    assert window["pool"]["symbol"] == "WIF2"                        # from the list, despite the limit
