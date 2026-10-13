# The terminal

```bash
python -m ai_quant_lab.meme terminal --open
```

A local web terminal at `http://127.0.0.1:8787`: four memecoins side by side,
live from GeckoTerminal and DexScreener, with the strategy library read on each of them and a
paper book that pays real execution costs. Standard library only: no extra
dependencies, no build step.

| Flag | Default | |
|---|---|---|
| `--port` | 8787 | |
| `--host` | 127.0.0.1 | The paper book has no authentication; keep it local |
| `--book` | `paper.db` | SQLite file for paper fills |
| `--account` | `default` | Several books can share one file |
| `--verdicts` | none | JSON from `zoo --json`; without it every signal shows `untested` |
| `--open` | off | Open the browser |

## The screen

**Top bar.** Network (Solana or Robinhood Chain), search by name, symbol or
pool address, and the market pulse.

**Watchlist.** GeckoTerminal trending pools and the newest pools. A click loads
the pool into the focused window (keys `1`-`4` change focus), then focus moves
to the next window.

**Windows.** For one pool each:

| Part | What it shows |
|---|---|
| Header | Symbol, chain, dex, age, narrative, price, change over 5m / 1h / 6h / 24h, trending rank |
| Chart | Candles 1m, 5m, 15m, 1h or 4h on a log scale, volume, last price, your entry price |
| Stats | TVL, FDV, 24h volume, holders, top-10 share, buy share and count of the last hour's trades |
| Screen | The library's universe filter, plus mint / freeze authority, honeypot, developer share, curve progress |
| Signals | Seven library strategies asked about this pool, each with the zoo's verdict |
| Trade | Ticket size, one-way and round-trip cost, buy, sell half, sell all, the open position |

**Right column.** The paper account, open positions, recent fills and the
strategy library with its verdicts.

## Signals

The library ranks a whole panel; a window shows one pool, so each signal asks
the strategy's entry question about that pool alone, with the same features
(`features.py`) and thresholds.

| Signal | Lit when | Data |
|---|---|---|
| `volume_breakout` | Volume z-score over 48 five-minute bars above 2 with the price up over 6 bars, within the 3-hour hold | 5m candles |
| `dip_reversal` | 30-60% below the 24h high while buyers outnumber sellers | candles, 1h flow |
| `buy_pressure` | Buys above 55% of the last hour's trades (at least 20 trades) | 1h flow |
| `trending_entry` | Top 10 of the GeckoTerminal trending list | trending list |
| `curve_graduation` | Bonding curve 70-100% complete | launchpad pools only |
| `narrative_rotation` | The token's narrative is one of the two gaining the most share of volume | names, live lists |
| `regime_momentum` | Up over 3 hours while the market pulse is risk-on | candles, pulse |

Two of them differ from the library, and say so:

- `dip_reversal` in the library asks for growing liquidity. A single pool
  request has no liquidity history, so the share of buyers stands in.
- `holder_momentum`, `flow_momentum` and `quality_composite` need history the
  live API does not give per request; they are absent from the windows and
  stay in the zoo.

A token that fails the universe screen has its lit signals shown as
**blocked** with the reason.

### Verdicts

```bash
python -m ai_quant_lab.meme zoo --db data/meme.db --network solana --json verdicts.json
python -m ai_quant_lab.meme terminal --verdicts verdicts.json
```

The verdict beside each signal is what the zoo concluded about the strategy
on the data you gave it: `PASS`, `DUPLICATE`, `THIN`, `LUCK`, `BETA`,
`UNSTABLE`, `LOSS` or `IDLE` (see [STRATEGIES.md](STRATEGIES.md#the-zoo)).
Without a file, every signal is `untested`. A lit signal next to `LUCK` or
`untested` is a reading, not a reason to trade.

## Market pulse

From the trending and new-pool lists, refreshed every minute:

| Component | Meaning |
|---|---|
| breadth | share of listed pools (TVL ≥ $5k) up over the last hour |
| buys | buys as a share of all last-hour trades |
| vol impulse | last-hour volume against the hourly average of the last 24h |
| launches | new pools per hour, from the creation times of the newest pools |

For the first 30 readings the regime comes from fixed thresholds on the raw
components ("calibrating" in the bar). After that it is the library's
`risk_appetite` on the readings collected so far: the trailing z-score of each
component, sign-adjusted and averaged.

**Narrative heat** is the log of each narrative's share of last-hour volume
over its share of 24-hour volume across the listed pools; above zero the
narrative is taking a bigger slice of attention than its daily average.

## Paper trading

Each fill pays what the backtest charges (`amm.py`, chain presets in
`NETWORK_COSTS`):

```
cost = trade × (1 − (1 − fee) · Q / (Q + trade))   curve and pool fee, Q = TVL / 2
     + trade × MEV haircut
     + priority fee
```

A $1,000 round trip in a $50k pool loses about 9.6% before the price moves.
The window shows the cost for the amount typed in before you click.

The account has two values:

- **Mark value:** tokens × last price, what a screener shows.
- **Equity (liquidation value):** what selling every position now would
  return after exit costs. Profit and loss are measured on this.

Positions and P&L are rebuilt from the fills on every request, so the book
cannot drift. `reset` clears the fills and restores the starting cash.

## Data and rate limits

Two free sources, each used for what it does best:

| Data | Source | Refreshed | Calls |
|---|---|---|---|
| Price, TVL, FDV, volume, buys and sells of window pools | DexScreener | 15 s | one call for every watched pool (up to 30) |
| Trending and new pools | GeckoTerminal | 2 min | 1 each per network |
| Candles | GeckoTerminal | 2 min | 1 per window and timeframe; the last candle follows the live price in between |
| Holders, top-10, contract flags | GeckoTerminal | 30 min | 1 per token |
| Curve progress | GeckoTerminal | 5 min | launchpad pools only |

The free GeckoTerminal tier answered about six calls per half minute when the
terminal was built, then returned HTTP 429. DexScreener allows 300 pair calls a
minute, so the numbers that move every second come from there and GeckoTerminal
is spent only on what DexScreener does not have.

A stale value is served at once and refreshed in the background. On HTTP 429
the GeckoTerminal client pauses for 20 seconds (growing to a minute) instead of
waiting inside a request; windows render without the missing part and say
which part is late. A pool DexScreener does not know falls back to the last
GeckoTerminal list it appeared in.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/market?network=solana` | pulse, trending and new pools |
| GET | `/api/pool?network=&address=&tf=5m&usd=500` | one window |
| GET | `/api/search?network=&q=` | pools by name, symbol or address |
| GET | `/api/strategies` | library with loaded verdicts |
| GET | `/api/paper` | account marked to live prices |
| POST | `/api/paper/buy` | `{network, address, usd, strategy?}` |
| POST | `/api/paper/sell` | `{network, address, fraction}` |
| POST | `/api/paper/reset` | `{starting_cash?}` |

## Limits

- **Paper only.** No wallet, no keys, no execution.
- **One user.** The book is local and unauthenticated; do not expose the port.
- **Per-token signals are approximations** of strategies that were designed
  and tested cross-sectionally. The zoo verdict belongs to the strategy, not
  to this token at this minute.
- **Free data sources.** A pool neither GeckoTerminal nor DexScreener lists does
  not exist for the terminal, and candles arrive up to two minutes late.
