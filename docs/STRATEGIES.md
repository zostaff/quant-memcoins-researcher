# Strategy library, narratives and market mood

The library in [`ai_quant_lab/meme/strategies.py`](../ai_quant_lab/meme/strategies.py)
is a set of ideas a memecoin desk would try first. They are written to be
**killed**: the [zoo](#the-zoo) runs every variant of every strategy, counts
each one as a trial, and judges the best of each the way an allocator would.

```bash
python -m ai_quant_lab.meme zoo --list                  # the library
python -m ai_quant_lab.meme zoo                         # synthetic tape with a planted narrative edge
python -m ai_quant_lab.meme zoo --tape null             # narratives visible, worth nothing
python -m ai_quant_lab.meme zoo --db data/meme.db --network robinhood --out report.md
```

## Ground rules every strategy follows

- **Long only, cash otherwise.** Weights are non-negative and sum to at most 1.
- **Point-in-time.** The weight at bar *t* uses data up to bar *t*; the backtest
  executes it at *t + 1*. This is tested, not promised: for every strategy,
  `tests/test_strategies.py` recomputes the weights on the panel cut at two
  points in time and checks they match the full-sample weights bar for bar.
- **Universe filter.** Pool TVL of at least $10k, at least 3 hours since
  launch, the rug screen (mint/freeze authority, top-10 concentration) when
  contract data was collected, and no tokens flagged as honeypots.
- **Scheduled rebalancing.** Books rotate every 1 to 6 hours, never every bar:
  at memecoin costs (about 2.5% per dollar traded in small pools) a strategy
  that turns over 15% of the book per 5-minute bar is gone within days, and
  the zoo shows exactly that.

## The strategies

| Strategy | Family | Thesis | Needs | Variants |
|---|---|---|---|---:|
| `holder_momentum` | momentum | Tokens adding holders fastest keep attracting buyers | holders | 9 |
| `flow_momentum` | momentum | Liquidity flowing into a pool leads price | — | 6 |
| `quality_composite` | momentum | Blend of holder growth, TVL flow, buy pressure and low concentration | — | 3 |
| `volume_breakout` | event | An abnormal volume spike with a rising price marks new attention | — | 4 |
| `curve_graduation` | event | A bonding curve near completion is about to migrate to a real pool | graduation | 4 |
| `trending_entry` | event | Entering the GeckoTerminal trending list puts a token in front of every screener | trending_rank | 4 |
| `dip_reversal` | reversal | A sharp drawdown while liquidity still grows is a shakeout, not a rug | — | 4 |
| `buy_pressure` | flow | A persistently higher share of buys than sells is one-sided order flow | buy_ratio | 4 |
| `narrative_rotation` | narrative | Own the leaders of the narratives gaining share of volume | names | 8 |
| `narrative_momentum` | narrative | Narratives that have performed keep performing for a while | names | 4 |
| `narrative_early` | narrative | Hot narratives where copycat launches have not exploded yet | names | 4 |
| `regime_momentum` | sentiment | Momentum only when the crowd is risk-on | — | 8 |

`curve_graduation` and `trending_entry` need fields only the live collector
records, so they are skipped on synthetic tapes and listed as such in the
report.

## Narratives

Memecoins trade in themes. A token's narrative is read from its **name and
symbol** with a keyword taxonomy (AI, politics, stocks, dogs, cats, frogs,
celebrities, other animals, anime, food, gaming, space, internet culture,
other) in [`narratives.py`](../ai_quant_lab/meme/narratives.py).

Why names and not GeckoTerminal's categories: the name and symbol are fixed on
chain at launch, so the label is known from the first bar. Categories are
curated later, often after a token has already run; labelling history with
them is look-ahead bias. The collector stores categories with the time they
were first seen, but the library does not use them for labels.

| Feature | Meaning |
|---|---|
| `narrative_volume_share` | narrative's share of all traded volume over a window |
| `narrative_heat` | log of short-window share over long-window share; > 0 means gaining attention |
| `narrative_return` | trailing liquidity-weighted return of the narrative, deaths at −100% |
| `launch_saturation` | copycat launches now versus the narrative's usual pace; > 2 means crowded |

`python -m ai_quant_lab.meme info --db data/meme.db` prints the narrative mix
of the collected universe.

## Market mood

[`sentiment.py`](../ai_quant_lab/meme/sentiment.py) measures the crowd's risk
appetite from the market itself:

| Component | Risk-on when |
|---|---|
| `breadth` | more tradeable tokens are up over the last 3 hours |
| `launch_rate` | launches run above their usual pace |
| `death_rate` | fewer live tokens die (enters with a minus sign) |
| `volume_impulse` | total volume runs above its slower average |
| `buy_pressure` | buys outnumber sells (live data) |
| `curve_progress` | bonding curves are filling (live data) |

`risk_appetite` z-scores each component against its own trailing history and
averages them. `regime(score, threshold)` turns it into a risk-on flag; with
too little history it is off.

### External sentiment

Social and news sentiment (X mentions, LunarCrush, Kaito, an in-house model)
plugs in through `load_external_sentiment`:

```python
from ai_quant_lab.meme import load_external_sentiment

# CSV columns: time, narrative, score   (or token, or market-wide: time, score)
scores = load_external_sentiment("x_mentions.csv", panel.close.index,
                                 key="narrative", lag="15min", max_age="6h")
```

A score stamped at τ is used only from τ + `lag`. The lag stands for the
provider's processing and delivery delay; a feed stamped with the window it
describes rather than with its publication time needs a lag of at least one
window. Scores older than `max_age` count as missing.

## The zoo

[`zoo.py`](../ai_quant_lab/meme/zoo.py) runs the library and returns a
leaderboard and a markdown report.

1. Every variant of every available strategy is backtested net of fees, curve
   slippage, MEV, priority fees and rug write-offs. Each one is a trial.
2. The best variant of each strategy gets the full tear sheet. Its deflated
   Sharpe uses the **total** trial count of the zoo, because choosing which
   family to back is a choice too.
3. A verdict, first match wins:

| Verdict | Rule |
|---|---|
| `IDLE` | the entry signal never fired |
| `LOSS` | net return or Sharpe ≤ 0 |
| `UNSTABLE` | either half of the sample has a negative Sharpe |
| `BETA` | alpha t-stat < 2 while the meme factors explain R² ≥ 0.15 |
| `LUCK` | deflated Sharpe p ≥ 0.05 over all trials |
| `THIN` | capacity below the book size tested |
| `DUPLICATE` | correlation > 0.6 with a strategy that already passed |
| `PASS` | survived all of the above |

`PASS` means the idea deserves fresh data it has never seen. It is not a
signal to trade.

### What the synthetic tapes show

Reports: [edge tape](zoo/edge-tape.md), [null tape](zoo/null-tape.md).

- **Null tape** (narratives visible in volume and launches, worth nothing):
  nothing passes.
- **Edge tape** (narrative heat drives returns, 42 days): the narrative
  rotation strategy passes; its close relative is flagged as a duplicate;
  strategies that caught part of the move are rejected as luck or as a regime.
- **Same edge, 21 days**: nothing passes. Half the history is not enough to
  separate the same edge from the best of 54 tries. The amount of data is a
  research decision, not an afterthought.

## Adding a strategy

```python
from ai_quant_lab.meme.strategies import STRATEGIES, Strategy, _every, _top
from ai_quant_lab.meme import features as F

def _my_idea(ctx, *, lookback, top_n):
    signal = F.volume_to_liquidity(ctx.panel.volume, ctx.panel.liquidity, lookback)
    return _every(_top(signal, top_n, ctx.universe), 36)

STRATEGIES["churn"] = Strategy("churn", "flow", "Pools being churned hard attract more traders.",
                               _my_idea, {"lookback": (12, 36), "top_n": (8, 16)})
```

The look-ahead test in `tests/test_strategies.py` picks it up automatically; then run the zoo. Every
variant you add raises the bar for every other strategy: that is the point.
