"""Capacity: how much money a memecoin strategy can actually run.

Every allocator asks this before the Sharpe question. On memecoins it decides
everything: a strategy that makes 300% on $1k can lose money at $100k because
it is buying 10% of a $40k pool on every entry. Run the same weight schedule
at increasing capital and watch the net Sharpe decay.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pandas as pd

from ai_quant_lab.meme.backtest import MemeBacktestConfig, meme_backtest
from ai_quant_lab.meme.panel import TokenPanel

DEFAULT_CAPITALS: tuple[float, ...] = (1_000, 10_000, 100_000, 1_000_000)


def capacity_curve(
    weights: pd.DataFrame,
    panel: TokenPanel,
    *,
    capitals: Sequence[float] = DEFAULT_CAPITALS,
    config: MemeBacktestConfig | None = None,
) -> pd.DataFrame:
    """Net performance of the same weights at each capital level."""
    base = config or MemeBacktestConfig()
    rows = []
    for capital in capitals:
        result = meme_backtest(weights, panel, replace(base, capital_usd=float(capital)))
        m = result.metrics
        rows.append(
            {
                "capital_usd": float(capital),
                "sharpe": m["sharpe_ratio"],
                "total_return": m["total_return"],
                "pnl_usd": m["final_equity_usd"] - capital,
                "cost_per_dollar_traded": m["cost_per_dollar_traded"],
                "capped_fraction": m["capped_fraction"],
                "rug_loss_usd": m["rug_loss_usd"],
            }
        )
    return pd.DataFrame(rows).set_index("capital_usd")


def estimate_capacity(curve: pd.DataFrame, *, keep_fraction: float = 0.5) -> float:
    """Largest tested capital before return on capital first drops below
    `keep_fraction` of the best one.

    Return on capital, not Sharpe: past the pool-participation cap the extra
    money sits in cash, which keeps Sharpe flat while the return per dollar
    collapses. Returns 0.0 when no tested size makes money.
    """
    curve = curve.sort_index()
    best = curve["total_return"].max()
    if best <= 0:
        return 0.0
    capacity = 0.0
    for capital, total_return in curve["total_return"].items():
        if total_return < keep_fraction * best:
            break
        capacity = float(capital)
    return capacity
