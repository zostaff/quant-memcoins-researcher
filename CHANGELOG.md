# Changelog

## 0.4.0

- **Live terminal** (`python -m ai_quant_lab.meme terminal`): four windows on
  one screen, each with candles (1m to 4h), flow, holders, contract flags, the
  library's universe screen, per-token readings of seven library strategies
  with the zoo's verdict beside each, and the real cost of a ticket.
- **Market pulse**: breadth, buy share, volume impulse and launch pace from
  the live lists; switches to the library's `risk_appetite` once enough
  readings exist. Narrative heat from 1h versus 24h volume share.
- **Paper trading**: SQLite book, every fill charged the backtest's fee,
  curve slippage, MEV and priority fee; account valued at mark and at
  liquidation.
- **Rate-limit friendly data layer**: prices and flow from DexScreener (one
  batch call for every watched pool), candles, lists and holders from
  GeckoTerminal; stale-while-revalidate cache, fail-fast cool-down on HTTP 429,
  the last candle follows the live price.
- `zoo --json` writes verdicts for the terminal.
- **Tests**: 193.

## 0.3.0

- **Strategy library**: 12 strategies in momentum, event, reversal, flow,
  narrative and sentiment families (`ai_quant_lab/meme/strategies.py`).
- **Strategy zoo**: every variant counted as a trial, best of each strategy
  judged with a full tear sheet and a verdict (IDLE, LOSS, UNSTABLE, BETA,
  LUCK, THIN, DUPLICATE, PASS); markdown report; `python -m ai_quant_lab.meme zoo`.
- **Narratives**: token themes from names and symbols, narrative heat, return,
  launch saturation (`narratives.py`).
- **Market mood**: breadth, launch rate, death rate, volume impulse, buy
  pressure, curve progress and a composite risk appetite (`sentiment.py`);
  external sentiment feeds with a publication lag.
- **Collector**: token names and symbols, GeckoTerminal categories and socials
  with first-seen time, developer share, honeypot flag, buy/sell ratio,
  trending ranks for tracked pools. Old databases migrate on open.
- **Synthetic tapes**: narratives and names; `narrative_attention` and
  `narrative_strength` switches that never change base-tape prices.
- **Research loop**: Claude receives narrative and mood fields.
- **Tests**: 176, including a look-ahead test for every library strategy.

## 0.2.0

- Robinhood Chain: multi-network collector, per-pool Uniswap fee tiers,
  chain cost presets.
- Token fields: holders, top-10 share, mint and freeze authority, bonding-curve
  progress; rug screen.
- Project website and README.

## 0.1.0

- Memecoin layer: point-in-time universe, AMM execution costs, meme factors,
  capacity, tear sheet, GeckoTerminal collector, synthetic tapes.
