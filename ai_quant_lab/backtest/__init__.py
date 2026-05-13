"""Backtesting primitives.

Two engines, intentionally:
    - vectorized_backtest: fast, used inside the research loop where speed dominates
    - EventDrivenBacktest: slow but realistic, used to verify that vectorized results
      survive when costs, slippage, and fill ordering are modeled properly

A strategy that disagrees between the two engines is almost always wrong.
"""

from ai_quant_lab.backtest.costs import market_impact_bps, realistic_cost_bps
from ai_quant_lab.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    performance_metrics,
    vectorized_backtest,
)
from ai_quant_lab.backtest.event_driven import EventDrivenBacktest, RealisticEventDriven

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "EventDrivenBacktest",
    "RealisticEventDriven",
    "market_impact_bps",
    "performance_metrics",
    "realistic_cost_bps",
    "vectorized_backtest",
]
