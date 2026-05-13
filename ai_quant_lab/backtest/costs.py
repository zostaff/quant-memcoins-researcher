"""Transaction cost models.

Costs are the difference between a paper edge and a real edge. The default
`realistic_cost_bps` charges a fixed bps per round-turn — enough to kill
most low-edge strategies that survive a frictionless backtest.

Market impact follows the standard sqrt(participation) shape used by Almgren-Chriss
and most practitioner desks; it's a default, not a calibration.
"""

from __future__ import annotations

import numpy as np


def realistic_cost_bps(
    spread_bps: float = 2.0,
    commission_bps: float = 1.0,
    slippage_bps: float = 5.0,
) -> float:
    """Total round-turn cost in basis points.

    Defaults sum to 8 bps, a sober assumption for liquid US equities.
    Crypto and small-cap should override upward; futures may go lower.
    """
    if min(spread_bps, commission_bps, slippage_bps) < 0:
        raise ValueError("Cost components must be non-negative.")
    return spread_bps + commission_bps + slippage_bps


def market_impact_bps(
    trade_size_usd: float,
    average_daily_volume_usd: float,
    coefficient: float = 10.0,
) -> float:
    """Square-root impact: bps = coefficient * sqrt(participation).

    `coefficient` is the impact in bps at 100% participation. 10 bps is a
    reasonable default for liquid equities; illiquid names need much higher.
    """
    if average_daily_volume_usd <= 0:
        raise ValueError("ADV must be positive.")
    participation = max(trade_size_usd, 0.0) / average_daily_volume_usd
    return float(coefficient * np.sqrt(participation))
