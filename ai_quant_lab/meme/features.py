"""Panel features for memecoin strategies.

Every function takes time × token frames and returns a frame of the same
shape whose value at bar t uses only data up to and including bar t. The
backtest adds the execution lag, so strategies must not shift again to "be
safe" (it only throws away the freshest bar) and must never shift by a
negative number.

These are the cross-sectional building blocks a strategy combines:
attention (volume), flow (liquidity), crowd (holders), and price action.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tradeable(close: pd.DataFrame, liquidity: pd.DataFrame, min_liquidity_usd: float = 5_000.0) -> pd.DataFrame:
    """Token has a price and enough pool TVL to enter at bar t."""
    return close.notna() & (liquidity.fillna(0.0) >= min_liquidity_usd)


def age_bars(close: pd.DataFrame) -> pd.DataFrame:
    """Bars since first price. NaN before launch."""
    listed = close.notna().cumsum() > 0
    return listed.cumsum().where(listed).astype(float)


def trailing_return(close: pd.DataFrame, bars: int = 12) -> pd.DataFrame:
    return close / close.shift(bars) - 1.0


def drawdown_from_high(close: pd.DataFrame) -> pd.DataFrame:
    """Current price over the running max since launch, minus one (≤ 0)."""
    return close / close.cummax() - 1.0


def liquidity_growth(liquidity: pd.DataFrame, bars: int = 12) -> pd.DataFrame:
    """Log change in pool TVL. Positive = money flowing into the pool."""
    positive = liquidity.where(liquidity > 0)
    return np.log(positive).diff(bars)


def volume_to_liquidity(volume: pd.DataFrame, liquidity: pd.DataFrame, bars: int = 12) -> pd.DataFrame:
    """Rolling volume over TVL: how hard the pool is being churned."""
    rolled = volume.rolling(bars, min_periods=1).sum()
    return rolled / liquidity.where(liquidity > 0)


def volume_zscore(volume: pd.DataFrame, bars: int = 48) -> pd.DataFrame:
    """Volume versus the token's own recent history."""
    mean = volume.rolling(bars, min_periods=bars // 2).mean()
    std = volume.rolling(bars, min_periods=bars // 2).std()
    return (volume - mean) / std.replace(0.0, np.nan)


def holder_growth(holders: pd.DataFrame, bars: int = 12) -> pd.DataFrame:
    """Fractional change in holder count."""
    return holders / holders.shift(bars) - 1.0


def cross_rank(signal: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Percentile rank across tokens at each bar, within an optional mask."""
    if mask is not None:
        signal = signal.where(mask)
    return signal.rank(axis=1, pct=True)


def top_n_equal_weight(
    signal: pd.DataFrame,
    n: int,
    mask: pd.DataFrame | None = None,
    gross: float = 1.0,
) -> pd.DataFrame:
    """Long the `n` highest-signal tokens at each bar, equal weight, `gross` total."""
    if mask is not None:
        signal = signal.where(mask)
    ranks = signal.rank(axis=1, ascending=False, method="first")
    chosen = ranks <= n
    count = chosen.sum(axis=1).replace(0, np.nan)
    return chosen.astype(float).div(count, axis=0).fillna(0.0) * gross


def hold_for(entries: pd.DataFrame, bars: int) -> pd.DataFrame:
    """Turn one-bar entry signals (bool) into positions held for `bars` bars."""
    held = entries.astype(float).rolling(bars, min_periods=1).max()
    return held.fillna(0.0) > 0
