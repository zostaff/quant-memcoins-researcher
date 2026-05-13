"""Transaction Cost Analysis: calibrate fill quality from real execution logs.

Three standard cost metrics (Perold, 1988; Almgren et al., 2005):

    arrival_shortfall = (avg_fill - arrival_price) * side
        — slippage vs the price at order arrival
    implementation_shortfall = (avg_fill - decision_price) * side
        — slippage vs the price at the trading-decision time
    market_impact_bps = arrival_shortfall / arrival_price * 1e4
        — same thing, in bps

The output feeds `calibrated_cost_bps`: given a history of fills, return
a cost-bps estimate suitable for plugging into `BacktestConfig.cost_bps`.
This closes the loop between live execution and backtest assumptions —
the model gets less wrong over time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fill:
    """One realized execution.

    Attributes:
        timestamp: when the fill happened.
        symbol: instrument id.
        side: +1 for buy, -1 for sell.
        quantity: number of shares/units filled. Always positive.
        fill_price: realized average price for this fill.
        arrival_price: mid/last price at order arrival.
        decision_price: mid/last price at the moment the strategy decided.
    """

    timestamp: pd.Timestamp
    symbol: str
    side: int
    quantity: float
    fill_price: float
    arrival_price: float
    decision_price: float

    def __post_init__(self) -> None:
        if self.side not in (-1, 1):
            raise ValueError("side must be +1 or -1")
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")
        if min(self.fill_price, self.arrival_price, self.decision_price) <= 0:
            raise ValueError("prices must be positive")


@dataclass(frozen=True)
class TCAReport:
    """Aggregate stats over a fill history."""

    n_fills: int
    mean_arrival_shortfall_bps: float
    median_arrival_shortfall_bps: float
    mean_implementation_shortfall_bps: float
    p95_arrival_shortfall_bps: float
    by_symbol: pd.DataFrame                # rows: symbol; cols: same stats

    def calibrated_cost_bps(self, percentile: float = 0.5) -> float:
        """Cost estimate for BacktestConfig.

        Default returns the median realized arrival shortfall — robust to
        outliers, neither pessimistic nor optimistic. Pass `percentile=0.95`
        for a conservative estimate for stress-testing.
        """
        if percentile == 0.5:
            return self.median_arrival_shortfall_bps
        if percentile == 0.95:
            return self.p95_arrival_shortfall_bps
        # Anything else: interpolate from the per-symbol distribution.
        if self.by_symbol.empty:
            return self.median_arrival_shortfall_bps
        return float(self.by_symbol["arrival_shortfall_bps"].quantile(percentile))


def compute_tca(fills: list[Fill]) -> TCAReport:
    """Build a TCA report from a list of fills.

    Args:
        fills: list of `Fill` objects. Can be drawn from `DecisionLog.read()`
            after constructing the appropriate adapter (broker-specific).

    Returns:
        TCAReport with aggregate and per-symbol statistics.

    Sign convention: positive shortfall = paid more than arrival price (or
    sold for less). Negative shortfall = price improvement.
    """
    if not fills:
        raise ValueError("Need at least one fill.")

    rows = []
    for fill in fills:
        arrival_signed = (fill.fill_price - fill.arrival_price) * fill.side
        decision_signed = (fill.fill_price - fill.decision_price) * fill.side
        rows.append(
            {
                "timestamp": fill.timestamp,
                "symbol": fill.symbol,
                "side": fill.side,
                "quantity": fill.quantity,
                "arrival_shortfall_bps": (arrival_signed / fill.arrival_price) * 1e4,
                "implementation_shortfall_bps": (decision_signed / fill.decision_price) * 1e4,
            }
        )
    frame = pd.DataFrame(rows)

    by_symbol = (
        frame.groupby("symbol")
        .agg(
            n_fills=("symbol", "size"),
            arrival_shortfall_bps=("arrival_shortfall_bps", "mean"),
            median_arrival_bps=("arrival_shortfall_bps", "median"),
            implementation_shortfall_bps=("implementation_shortfall_bps", "mean"),
        )
        .reset_index()
        .set_index("symbol")
    )

    return TCAReport(
        n_fills=len(frame),
        mean_arrival_shortfall_bps=float(frame["arrival_shortfall_bps"].mean()),
        median_arrival_shortfall_bps=float(frame["arrival_shortfall_bps"].median()),
        mean_implementation_shortfall_bps=float(frame["implementation_shortfall_bps"].mean()),
        p95_arrival_shortfall_bps=float(frame["arrival_shortfall_bps"].quantile(0.95)),
        by_symbol=by_symbol,
    )


def calibrate_slippage_coefficient(
    fills: list[Fill],
    average_daily_volume: dict[str, float] | None = None,
) -> float:
    """Fit a sqrt-impact slippage coefficient (bps per √participation).

    Args:
        fills: realized fill history.
        average_daily_volume: per-symbol ADV in shares. If None, the
            calibration falls back to a participation-free constant fit and
            the returned number is just the mean shortfall in bps.

    Returns:
        Coefficient k such that `slippage_bps ≈ k * sqrt(participation)`.
        Suitable for `RealisticEventDriven(slippage_coefficient=k)`.
    """
    if not fills:
        raise ValueError("Need at least one fill.")
    if average_daily_volume is None:
        # Constant fit: just the mean of |arrival shortfall|
        shortfalls = []
        for fill in fills:
            shortfall = abs(fill.fill_price - fill.arrival_price) / fill.arrival_price * 1e4
            shortfalls.append(shortfall)
        return float(np.mean(shortfalls))

    shortfalls_bps: list[float] = []
    participations: list[float] = []
    for fill in fills:
        adv = average_daily_volume.get(fill.symbol)
        if not adv or adv <= 0 or fill.quantity <= 0:
            continue
        participation = fill.quantity / adv
        shortfall_bps = abs(fill.fill_price - fill.arrival_price) / fill.arrival_price * 1e4
        shortfalls_bps.append(shortfall_bps)
        participations.append(participation)

    if len(shortfalls_bps) < 5:
        return float(np.mean(shortfalls_bps)) if shortfalls_bps else 0.0

    # OLS through the origin: y = k * sqrt(x). One-parameter.
    x = np.sqrt(np.asarray(participations))
    y = np.asarray(shortfalls_bps)
    denominator = float(np.sum(x * x))
    if denominator == 0:
        return 0.0
    coefficient = float(np.sum(x * y) / denominator)
    return coefficient
