# Memecoin market

`ai_quant_lab.meme` brings the hedge-fund research stack to memecoins. The
validation core (deflated Sharpe, trial counting, correlation and PCA gates,
purged CV, kill switch) is unchanged. What changes is the model of the market
underneath it.

## What carries over, and what has to change

| Fund concept | Equities version | Memecoin version (`ai_quant_lab/meme/`) |
|---|---|---|
| Point-in-time universe | CRSP with delisted names | `TokenPanel`: tokens never leave the panel; dead ones keep a column with no price. Collector registers pools the moment they appear (`geckoterminal.py`) |
| Delisting returns | -30% to -100% on delist | Rug or death while held is written off to zero (`meme_backtest`); factors see death as -100% (`returns_with_death`) |
| Transaction costs | bps + sqrt-impact | Exact constant-product curve loss `1 - (1-f)·Q/(Q+s)` + priority fee + MEV haircut (`amm.py`) |
| Participation limit | % of ADV | % of pool TVL (`max_pool_participation`) |
| No-trade region | rebalance bands | `rebalance_band`: without it fixed per-tx costs eat the book |
| Shorting | borrow | Long only. There is no borrow |
| Risk model | Barra / Fama-French | `meme_factor_returns`: MKT, SIZE, MOM, AGE, FLOW, rugs included |
| Alpha vs beta | factor regression | `meme_attribution`: OLS of strategy on meme factors |
| Capacity | impact-adjusted AUM | `capacity_curve` / `estimate_capacity`: same weights at $1k..$1M |
| Multiple testing | deflated Sharpe | Same `deflated_sharpe`, fed by `ResearchMemory` trial count |
| Tear sheet | | `tear_sheet(...)`: all of the above in one call |

## Networks

| | Solana | Robinhood Chain |
|---|---|---|
| GeckoTerminal id | `solana` | `robinhood` |
| Venues seen | pump.fun, PumpSwap, Meteora | Uniswap v4 / v3, Pons launchpad, Bankr |
| Pool fee | 1% default | fee tier from the pool name (1%, 0.3%, 0.01%) into `panel.pool_fee` |
| Per-transaction cost | priority fee + tip, $0.10 default | L2 gas in ETH, $0.02 default |
| MEV | sandwiches common, 0.2% haircut | sequencer, no public mempool, 0.1% haircut |
| Token fields | holders, top-10, mint / freeze, curve progress | holders, top-10 when GeckoTerminal has them, Pons curve progress |

`costs_for_network(name)` returns the defaults above. They are placeholders.
The constant-product cost model is exact for bonding curves and full-range
pools and conservative for concentrated liquidity (Uniswap v3/v4), where
depth near the price is higher.

One database can hold both chains. `run` collects every network in
`--networks` (default `solana,robinhood`) under one rate limit;
`load_panel(conn, network="robinhood")` and `research --network robinhood`
work on one chain at a time.

## Quick start

```bash
python -m ai_quant_lab.meme demo                     # synthetic, no key, ~1 min
python examples/10_memecoin_multiple_testing.py      # same thing as a script
```

Real data from Solana and Robinhood Chain, collected point-in-time:

```bash
python -m ai_quant_lab.meme run --db data/meme.db              # keep running: both chains, snapshots + token fields + candles
python -m ai_quant_lab.meme collect --db meme.db --every 120   # snapshots only
python -m ai_quant_lab.meme backfill --db meme.db              # 5-minute candles for every pool
python -m ai_quant_lab.meme info --db meme.db
python -m ai_quant_lab.meme research --db meme.db --network robinhood --iterations 20   # needs ANTHROPIC_API_KEY
```

`run` does everything in one process, because all calls share one rate limit
(about 30 per minute): each pass registers new pools, snapshots the tracked
ones, snapshots their tokens, then refreshes candles of the 15 stalest pools.
`deploy/com.quantmeme.collector.plist` is a launchd template. On macOS a
launchd job cannot read an external disk until the launched binary has Full
Disk Access, so either keep the project on the internal disk or grant access.

The research loop is the original Claude loop with a memecoin hypothesis
prompt, a memecoin critic template and a code prompt that writes
`strategy(panel) -> weights`.

## Reading a tear sheet

```python
from ai_quant_lab.meme import UniverseSpec, simulate_universe, tear_sheet
from ai_quant_lab.meme import features as F

panel = simulate_universe(UniverseSpec(n_days=21, edge=1.0))
mask = F.tradeable(panel.close, panel.liquidity, 10_000) & (F.age_bars(panel.close) >= 72)
w = F.top_n_equal_weight(F.holder_growth(panel.holders, 288), 24, mask)
w = w.iloc[::72].reindex(w.index).ffill()          # rebalance every 6 hours
print(tear_sheet(w, panel, n_trials=20).summary())
```

What to look at, in order:

1. **Execution cost per dollar traded.** 2-5% is normal in small pools. If the
   gross edge per trade is below that, nothing else matters.
2. **Halves Sharpe.** One good half and one bad half is a regime, not a strategy.
3. **MKT beta and alpha t-stat.** A high Sharpe with alpha t-stat near zero is
   meme-market beta: long the mania.
4. **Capacity.** The book size where return on capital falls below half of
   the best one. Past the pool-participation cap the extra money sits in cash.
5. **Deflated Sharpe p-value** with the honest number of variants tried.

## Synthetic tapes

`simulate_universe` reproduces what breaks naive backtests: continuous
launches, rugs concentrated in the first hour, liquidity bleeding out,
a shared attention cycle, ~5% volatility per 5-minute bar, TVL ∝ sqrt(price).
`edge=0` is the null tape: holder growth is visible but worthless, and
expected return per bar is zero minus the bleed. `edge=1` makes token quality
pay and regimes persistent. Use the null tape to check that your process
rejects what it should.

Every synthetic token also gets a narrative and a name to match ("Grok Mo",
"Shiba Vu"), and each narrative has its own slow heat cycle.
`narrative_attention` makes that heat visible in launches and volume;
`narrative_strength` makes it pay. Attention without strength is the null
tape for narrative strategies. These switches use their own random stream,
so they never change the prices of the base tape.

## Strategies, narratives, mood

See [STRATEGIES.md](STRATEGIES.md): the strategy library, narrative and
market-mood features, external sentiment, and the zoo that judges them.

## Data caveats

- GeckoTerminal has no reserve history. Liquidity between snapshots is moved
  by the constant-product rule from the last snapshot; before the first one
  the token is untradeable. Collect often.
- Candles exist only for bars with trades. A pool silent for
  `dead_after_bars` bars, or whose reserve fell below `rug_reserve_usd`, is
  treated as dead for good.
- Token fields come from `tokens/{address}/info` (holders, top-10 share,
  mint and freeze authority) and `tokens/multi` (pump.fun bonding-curve
  progress). Each holds its last observed value from observation time on and
  is NaN before. GeckoTerminal fills holder counts lazily: fresh tokens often
  show none for a while. Info is one call per token, so each pass refreshes
  only the `--holder-calls` stalest tokens.
- `tokens/multi` and `info` also give each token's name and symbol, stored in
  `token_meta` and used for narrative labels (they are fixed at launch).
  GeckoTerminal categories and socials are stored with the time they were
  first seen and are not used for labels: they are curated after the fact.
- `info` adds the developer's share of supply (`dev_share`) and a honeypot
  flag; the strategy universe drops flagged honeypots.
- `buy_ratio` is the share of buys in the 5-minute transaction count of the
  last snapshot, held for at most 30 minutes.
- Each pass records the chain's trending list (one call). `trending_rank` is
  the rank in the last list observed at each bar, only for pools already in
  the universe: adding pools because they trend would select on outcome.
- `features.rug_screen(mint_risk, top10_share)` fails tokens whose creator can
  still mint or freeze, or whose top-10 wallets hold more than half of supply;
  unobserved counts as a fail unless `unknown_ok=True`.
- The fee defaults in `ExecutionCosts` are placeholders. Set them from the
  venue you trade on.
