"""Label construction.

Triple-barrier labels (López de Prado, 2018): for each event timestamp, walk
forward until one of three things happens — a profit barrier is hit, a loss
barrier is hit, or a time barrier is hit. The label is +1, -1, or 0.

Useful for converting noisy raw returns into a clean classification target
without arbitrary fixed-horizon assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    price_data: pd.Series,
    *,
    upper_pct: float,
    lower_pct: float,
    max_holding: int,
) -> pd.Series:
    """Label each bar by which of three barriers hits first.

    Args:
        price_data: Series of prices, sorted by time, with a sortable index.
        upper_pct: Profit barrier as a fraction (e.g. 0.02 = +2%).
        lower_pct: Loss barrier as a positive fraction (e.g. 0.02 = -2%).
        max_holding: Time barrier in bars. Events that don't hit a price
            barrier within this horizon get label 0.

    Returns:
        Series aligned to `price_data` with values in {-1, 0, +1}. The last
        `max_holding` entries are NaN because their forward window is incomplete.

    The labels are forward-looking BY CONSTRUCTION — that's their purpose.
    They must therefore be used as a *target*, never as a feature.
    """
    if upper_pct <= 0 or lower_pct <= 0:
        raise ValueError("upper_pct and lower_pct must be positive fractions.")
    if max_holding < 1:
        raise ValueError("max_holding must be >= 1")

    prices = price_data.to_numpy(dtype=float)
    n = len(prices)
    labels = np.full(n, np.nan, dtype=float)

    for i in range(n - max_holding):
        entry = prices[i]
        upper = entry * (1.0 + upper_pct)
        lower = entry * (1.0 - lower_pct)
        label = 0
        for step in range(1, max_holding + 1):
            future = prices[i + step]
            if future >= upper:
                label = 1
                break
            if future <= lower:
                label = -1
                break
        labels[i] = label

    return pd.Series(labels, index=price_data.index, name="triple_barrier")


def forward_returns(price_data: pd.Series, horizon: int = 1) -> pd.Series:
    """Plain forward returns over `horizon` bars. The simplest possible target.

    Useful when you want a regression target rather than a classification one.
    Like triple-barrier labels, this is forward-looking by design — use only as a target.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return (price_data.shift(-horizon) / price_data - 1.0).rename(f"fwd_ret_{horizon}")
