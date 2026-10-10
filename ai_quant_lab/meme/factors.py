"""Memecoin risk factors and attribution.

A fund does not pay for beta. On memecoins the "beta" is the meme market
itself: when attention floods in, everything with a pool goes up. A strategy
that is long a basket of fresh tokens will look brilliant in a mania and is
still just market exposure. The factors here make that measurable:

    MKT   liquidity-weighted return of every tradeable token
    SIZE  small pools minus big pools (by TVL)
    MOM   recent winners minus recent losers
    AGE   young tokens minus old tokens
    FLOW  growing liquidity minus shrinking liquidity

Each long/short leg is an equal-weight tercile, formed on bar t-1 data and
earning bar t returns, rugs included at -100%. Factors are frictionless: they
are risk measures, not tradeable strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_quant_lab.meme.panel import TokenPanel, bars_per_year
from ai_quant_lab.validation.factor_attribution import AttributionResult, fama_french_attribution


def meme_factor_returns(
    panel: TokenPanel,
    *,
    min_liquidity_usd: float = 5_000.0,
    momentum_bars: int = 12,
    flow_bars: int = 12,
    min_universe: int = 6,
) -> pd.DataFrame:
    """Per-bar factor returns (time × factor)."""
    returns = panel.returns_with_death()
    universe = panel.alive(min_liquidity_usd).shift(1, fill_value=False)
    liquidity_prev = panel.liquidity.shift(1)

    market_weight = liquidity_prev.where(universe)
    market = (returns * market_weight).sum(axis=1) / market_weight.sum(axis=1).replace(0, np.nan)

    age = _age_bars(panel.close)
    trailing = panel.close / panel.close.shift(momentum_bars) - 1.0
    flow = np.log(panel.liquidity.where(panel.liquidity > 0)).diff(flow_bars)

    factors = pd.DataFrame(
        {
            "MKT": market,
            "SIZE": -_tercile_spread(liquidity_prev, returns, universe, min_universe),
            "MOM": _tercile_spread(trailing.shift(1), returns, universe, min_universe),
            "AGE": -_tercile_spread(age.shift(1), returns, universe, min_universe),
            "FLOW": _tercile_spread(flow.shift(1), returns, universe, min_universe),
        }
    )
    return factors.fillna(0.0)


def meme_attribution(
    strategy_returns: pd.Series,
    factors: pd.DataFrame,
) -> AttributionResult:
    """OLS of strategy returns on meme factors. Alpha is what is left over."""
    return fama_french_attribution(
        strategy_returns,
        factors,
        annualization=bars_per_year(pd.DatetimeIndex(strategy_returns.index)),
    )


def _tercile_spread(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    min_universe: int,
) -> pd.Series:
    """Top-third minus bottom-third equal-weight return of `signal`."""
    masked = signal.where(universe & signal.notna() & returns.notna())
    ranks = masked.rank(axis=1, pct=True)
    count = masked.notna().sum(axis=1)
    top = returns.where(ranks > 2 / 3).mean(axis=1)
    bottom = returns.where(ranks <= 1 / 3).mean(axis=1)
    return (top - bottom).where(count >= min_universe)


def _age_bars(close: pd.DataFrame) -> pd.DataFrame:
    """Bars since each token's first price. NaN before launch."""
    listed = close.notna().cumsum() > 0
    return listed.cumsum().where(listed).astype(float)
