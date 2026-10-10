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

## Quick start

```bash
python -m ai_quant_lab.meme demo                     # synthetic, no key, ~1 min
python examples/10_memecoin_multiple_testing.py      # same thing as a script
```

Real Solana data, collected point-in-time:

```bash
python -m ai_quant_lab.meme collect --db meme.db --every 120   # keep running
python -m ai_quant_lab.meme backfill --db meme.db              # 5-minute candles
python -m ai_quant_lab.meme info --db meme.db
python -m ai_quant_lab.meme research --db meme.db --iterations 20   # needs ANTHROPIC_API_KEY
```

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

## Data caveats

- GeckoTerminal has no reserve history. Liquidity between snapshots is moved
  by the constant-product rule from the last snapshot; before the first one
  the token is untradeable. Collect often.
- Candles exist only for bars with trades. A pool silent for
  `dead_after_bars` bars, or whose reserve fell below `rug_reserve_usd`, is
  treated as dead for good.
- No holder counts from this source: `holders` is `None` on collected data.
- The fee defaults in `ExecutionCosts` are placeholders. Set them from the
  venue you trade on.
