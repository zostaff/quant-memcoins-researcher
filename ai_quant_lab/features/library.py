"""Pre-built features. Every function obeys the contract:
    f(price_data, window) -> Series of the same index, where value at time t
    is computed using ONLY data at times <= t.

This is enforced by always using `.rolling()` (which respects the left edge)
or by `.shift(1)` (which uses yesterday's value). No `.center=True`. No raw
forward differences. The leakage detector cross-checks this at runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum(price_data: pd.Series, window: int = 21) -> pd.Series:
    """Past-return momentum over `window` bars, lagged by one bar to be tradeable.

    Returned at time t is `price[t-1] / price[t-1-window] - 1`. The extra lag
    is what most introductory examples get wrong — they use `price[t]` and
    pretend that today's close is available before today's open.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    lagged_price = price_data.shift(1)
    return (lagged_price / lagged_price.shift(window) - 1.0).rename(f"mom_{window}")


def rolling_zscore(price_data: pd.Series, window: int = 21) -> pd.Series:
    """z-score of (price relative to its trailing mean) over `window`.

    Useful for mean-reversion signals. The trailing mean and std are computed
    over `[t-window, t-1]`, never including time t itself.
    """
    if window < 2:
        raise ValueError("window must be >= 2 for std to be defined")
    lagged = price_data.shift(1)
    rolling = lagged.rolling(window=window, min_periods=window)
    mean = rolling.mean()
    std = rolling.std(ddof=1)
    return ((lagged - mean) / std.replace(0.0, np.nan)).rename(f"zscore_{window}")


def realized_volatility(price_data: pd.Series, window: int = 21) -> pd.Series:
    """Trailing realized volatility of log returns over `window`.

    Annualization is intentionally NOT applied — the user should annualize once
    at the metric layer, not multiple times in feature space.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    log_returns = np.log(price_data / price_data.shift(1))
    return (
        log_returns.shift(1)
        .rolling(window=window, min_periods=window)
        .std(ddof=1)
        .rename(f"rv_{window}")
    )


def range_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Mean (high-low)/close over the trailing window.

    A regime feature: high values flag volatile/wide-bar regimes; low values
    flag compression. Each input is shifted by one bar before use.
    """
    high_lag = high.shift(1)
    low_lag = low.shift(1)
    close_lag = close.shift(1)
    bar_range = (high_lag - low_lag) / close_lag.replace(0.0, np.nan)
    return bar_range.rolling(window=window, min_periods=window).mean().rename(f"range_{window}")


def ewma(price_data: pd.Series, halflife: float) -> pd.Series:
    """Exponentially weighted moving average, lagged by one bar.

    `halflife` is in bars. Provided as a convenience for ema-crossover style examples.
    """
    if halflife <= 0:
        raise ValueError("halflife must be > 0")
    return price_data.shift(1).ewm(halflife=halflife, adjust=False).mean().rename(f"ewma_{halflife}")
