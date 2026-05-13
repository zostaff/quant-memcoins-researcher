"""Intraday-aware bar metadata.

The single-asset backtest engine assumes evenly-spaced bars. For intraday
work that's a lie: bars are unevenly spaced across overnight, weekend, and
holiday gaps, and a 1-minute bar carries 1/390th the variance of a daily
bar — but the same "execution lag = 1" rule is wrong by an order of magnitude.

`BarSchedule` carries the metadata needed to do this correctly:

    - bar_interval: "1d" | "1h" | "5m" | "1m" (or pandas Timedelta)
    - market_open / market_close: trading session boundaries
    - annualization: derived from bar_interval and session length

`infer_bar_interval(index)` autodetects the interval from a DatetimeIndex
by taking the median spacing — handles missing bars correctly.

Use case: an intraday strategy reports an annualized Sharpe of 4.0. Is that
because it's amazing, or because the engine annualized assuming 252 bars/year
when there are actually 252 × 390 = 98,280 of them? The BarSchedule prevents
this class of mistake.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

import numpy as np
import pandas as pd


BarInterval = Literal["1d", "4h", "1h", "30m", "15m", "5m", "1m"]


_BARS_PER_TRADING_DAY: dict[str, int] = {
    "1d": 1,
    "4h": 2,
    "1h": 7,
    "30m": 13,
    "15m": 26,
    "5m": 78,
    "1m": 390,  # NYSE: 9:30-16:00
}


@dataclass(frozen=True)
class BarSchedule:
    """Bar interval + session metadata.

    Attributes:
        interval: bar size identifier. Use one of the standard strings or
            pass a Timedelta to `from_timedelta`.
        market_open: session start time (used for intraday overnight gaps).
        market_close: session end time.
        trading_days_per_year: defaults to 252 (US equities). Crypto uses 365.
        bars_per_trading_day: derived; can be overridden for non-standard hours.
    """

    interval: str = "1d"
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)
    trading_days_per_year: int = 252
    bars_per_trading_day: int | None = None

    @property
    def annualization(self) -> int:
        """Periods per year for Sharpe annualization."""
        per_day = self.bars_per_trading_day or _BARS_PER_TRADING_DAY.get(self.interval, 1)
        return self.trading_days_per_year * per_day

    @classmethod
    def daily(cls) -> "BarSchedule":
        return cls(interval="1d", trading_days_per_year=252, bars_per_trading_day=1)

    @classmethod
    def hourly(cls) -> "BarSchedule":
        return cls(interval="1h")

    @classmethod
    def crypto_daily(cls) -> "BarSchedule":
        return cls(
            interval="1d",
            market_open=time(0, 0),
            market_close=time(23, 59),
            trading_days_per_year=365,
            bars_per_trading_day=1,
        )

    @classmethod
    def crypto_hourly(cls) -> "BarSchedule":
        return cls(
            interval="1h",
            market_open=time(0, 0),
            market_close=time(23, 59),
            trading_days_per_year=365,
            bars_per_trading_day=24,
        )


def infer_bar_interval(index: pd.DatetimeIndex) -> str:
    """Heuristic: median bar spacing → standard interval label.

    Args:
        index: DatetimeIndex with at least 10 entries.

    Returns:
        Interval string ("1d", "1h", "5m", ...). Defaults to "1d" if the
        spacing doesn't match a known interval.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if len(index) < 10:
        return "1d"

    deltas = pd.Series(index[1:] - index[:-1])
    median_seconds = float(deltas.dt.total_seconds().median())

    table = [
        ("1m", 60),
        ("5m", 300),
        ("15m", 900),
        ("30m", 1800),
        ("1h", 3600),
        ("4h", 14400),
        ("1d", 86400),
    ]
    # match to within 20% of canonical
    for label, seconds in table:
        if 0.8 * seconds <= median_seconds <= 1.2 * seconds:
            return label
    return "1d"


def annualization_from_index(index: pd.DatetimeIndex) -> int:
    """Convenience: just give me the annualization factor for this series.

    Equivalent to `BarSchedule(interval=infer_bar_interval(index)).annualization`
    but skips constructing the dataclass.
    """
    interval = infer_bar_interval(index)
    per_day = _BARS_PER_TRADING_DAY.get(interval, 1)
    return 252 * per_day


def session_mask(index: pd.DatetimeIndex, schedule: BarSchedule) -> np.ndarray:
    """Boolean mask of bars that fall inside the trading session.

    Useful for intraday strategies: drop overnight bars before computing
    indicators, or zero out positions during after-hours.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    times_of_day = index.time
    return np.array(
        [schedule.market_open <= t <= schedule.market_close for t in times_of_day]
    )
