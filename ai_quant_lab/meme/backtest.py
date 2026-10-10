"""Long-only memecoin portfolio backtest in dollars, not weights.

Why a separate engine: the vectorized engines charge a fixed bps on turnover,
which is independent of trade size. On a memecoin pool cost is a steep
function of size versus liquidity, so the same strategy at $1k and $1M are
two different strategies. This engine simulates dollar holdings bar by bar.

Rules the simulation enforces:
    - Long only, no leverage. Weight rows are clipped to [0, max_weight] and
      scaled down if they sum above 1; the rest sits in cash.
    - Weights decided at bar t are executed at bar t + execution_lag, at that
      bar's price and liquidity.
    - Entries need liquidity >= costs.min_liquidity_usd. Exits are allowed as
      long as the pool has any liquidity left.
    - Position size is capped at `max_pool_participation` × pool TVL.
    - A token whose price disappears while held is written off to zero. That
      is what a rug pull is; dropping the token instead would be survivorship.
    - No-trade band: a position is only resized when it is off target by more
      than `rebalance_band` of its size. Without it, every price move triggers
      a small trade and the fixed per-transaction cost eats the book.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_quant_lab.backtest.engine import performance_metrics
from ai_quant_lab.meme.amm import ExecutionCosts, swap_loss_fraction
from ai_quant_lab.meme.panel import TokenPanel, bars_per_year


@dataclass(frozen=True)
class MemeBacktestConfig:
    capital_usd: float = 10_000.0
    costs: ExecutionCosts = ExecutionCosts()
    execution_lag: int = 1
    max_weight: float = 0.10
    max_pool_participation: float = 0.02
    rebalance_band: float = 0.25

    def __post_init__(self) -> None:
        if self.capital_usd <= 0:
            raise ValueError("capital_usd must be positive")
        if self.execution_lag < 1:
            raise ValueError("execution_lag must be >= 1: same-bar fills are lookahead")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if not 0 < self.max_pool_participation <= 1:
            raise ValueError("max_pool_participation must be in (0, 1]")
        if self.rebalance_band < 0:
            raise ValueError("rebalance_band must be >= 0")


@dataclass(frozen=True)
class MemeBacktestResult:
    returns: pd.Series          # net return per bar on total equity
    equity: pd.Series           # USD
    holdings: pd.DataFrame      # USD value per token at bar end
    costs: pd.DataFrame         # per-bar USD: fee, impact, mev, priority
    rug_losses: pd.Series       # per-bar USD written off on dead tokens
    metrics: dict[str, float]


def meme_backtest(
    weights: pd.DataFrame,
    panel: TokenPanel,
    config: MemeBacktestConfig | None = None,
) -> MemeBacktestResult:
    """Simulate a long-only weight schedule on a TokenPanel."""
    config = config or MemeBacktestConfig()
    costs = config.costs

    weights = weights.reindex(index=panel.close.index, columns=panel.tokens)
    close = panel.close.to_numpy(dtype=float)
    liquidity = panel.liquidity.fillna(0.0).to_numpy(dtype=float)
    w = np.nan_to_num(weights.to_numpy(dtype=float), nan=0.0)
    w = np.clip(w, 0.0, config.max_weight)
    row_sum = w.sum(axis=1, keepdims=True)
    w = np.where(row_sum > 1.0, w / np.where(row_sum > 0, row_sum, 1.0), w)

    n_bars, n_tokens = close.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_ret = close[1:] / close[:-1] - 1.0

    holdings = np.zeros(n_tokens)
    cash = config.capital_usd
    lag = config.execution_lag

    equity_out = np.empty(n_bars)
    holdings_out = np.zeros((n_bars, n_tokens))
    cost_out = np.zeros((n_bars, 4))
    rug_out = np.zeros(n_bars)
    turnover_out = np.zeros(n_bars)
    capped_out = np.zeros(n_bars)
    wanted_out = np.zeros(n_bars)

    for t in range(n_bars):
        price_ok = np.isfinite(close[t])
        if t > 0:
            r = bar_ret[t - 1]
            moved = np.isfinite(r)
            holdings[moved] *= 1.0 + r[moved]
            dead = (holdings > 0) & ~price_ok
            rug_out[t] = holdings[dead].sum()
            holdings[dead] = 0.0

        equity = cash + holdings.sum()
        if t < lag or equity <= 0:
            equity_out[t] = equity
            holdings_out[t] = holdings
            continue

        target = w[t - lag] * equity
        wanted_out[t] = target.sum()
        cap = config.max_pool_participation * liquidity[t]
        capped = np.minimum(target, cap)
        capped_out[t] = (target - capped).sum()
        target = capped

        delta = target - holdings
        can_enter = price_ok & (liquidity[t] >= costs.min_liquidity_usd)
        can_exit = price_ok & (liquidity[t] > 0)
        delta = np.where((delta > 0) & ~can_enter, 0.0, delta)
        delta = np.where((delta < 0) & ~can_exit, 0.0, delta)
        full_exit = (target == 0) & (delta < 0)
        entry = (holdings == 0) & (delta > 0)
        inside_band = np.abs(delta) <= config.rebalance_band * np.maximum(target, holdings)
        delta = np.where(inside_band & ~full_exit & ~entry, 0.0, delta)
        delta = np.where((np.abs(delta) < costs.min_trade_usd) & ~full_exit, 0.0, delta)

        sells = np.minimum(delta, 0.0)
        buys = np.maximum(delta, 0.0)
        sell_cost = _cost_parts(sells, liquidity[t], costs)
        cash += -sells.sum() - sell_cost.sum()
        holdings += sells

        buy_cost = _cost_parts(buys, liquidity[t], costs)
        budget = max(cash, 0.0)
        spend = buys.sum() + buy_cost.sum()
        if spend > budget and spend > 0:
            buys = buys * (budget / spend)
            buy_cost = _cost_parts(buys, liquidity[t], costs)
        cash -= buys.sum() + buy_cost.sum()
        holdings += buys

        cost_out[t] = sell_cost.sum(axis=1) + buy_cost.sum(axis=1)
        turnover_out[t] = (np.abs(sells).sum() + buys.sum()) / equity
        equity_out[t] = cash + holdings.sum()
        holdings_out[t] = holdings

    index = panel.close.index
    equity_series = pd.Series(equity_out, index=index, name="equity")
    prev = equity_series.shift(1).fillna(config.capital_usd)
    returns = (equity_series / prev - 1.0).rename("returns")

    cost_frame = pd.DataFrame(cost_out, index=index, columns=["fee", "impact", "mev", "priority"])
    rug_series = pd.Series(rug_out, index=index, name="rug_loss")
    holdings_frame = pd.DataFrame(holdings_out, index=index, columns=panel.tokens)

    annualization = bars_per_year(index)
    metrics = performance_metrics(
        returns,
        turnover=pd.Series(turnover_out, index=index),
        annualization=annualization,
    )
    total_cost = float(cost_frame.to_numpy().sum())
    traded = float(turnover_out @ np.maximum(equity_out, 0.0))
    metrics.update(
        {
            "capital_usd": config.capital_usd,
            "final_equity_usd": float(equity_out[-1]),
            "total_return": float(equity_out[-1] / config.capital_usd - 1.0),
            "total_cost_usd": total_cost,
            "cost_per_dollar_traded": total_cost / traded if traded > 0 else 0.0,
            "rug_loss_usd": float(rug_out.sum()),
            "capped_fraction": float(capped_out.sum() / wanted_out.sum()) if wanted_out.sum() > 0 else 0.0,
            "mean_positions": float((holdings_out > 0).sum(axis=1).mean()),
            "mean_invested_fraction": float(
                np.mean(holdings_out.sum(axis=1) / np.where(equity_out > 0, equity_out, np.nan))
            ),
            "annualization": float(annualization),
        }
    )
    return MemeBacktestResult(
        returns=returns,
        equity=equity_series,
        holdings=holdings_frame,
        costs=cost_frame,
        rug_losses=rug_series,
        metrics=metrics,
    )


def _cost_parts(trades: np.ndarray, liquidity: np.ndarray, costs: ExecutionCosts) -> np.ndarray:
    """4 × N array of USD costs: fee, curve impact, MEV, priority fee."""
    size = np.abs(trades)
    curve = size * swap_loss_fraction(size, liquidity, costs.swap_fee)
    fee = size * costs.swap_fee
    impact = np.maximum(curve - fee, 0.0)
    mev = size * costs.mev_haircut
    priority = np.where(size > 0, costs.priority_fee_usd, 0.0)
    return np.vstack([fee, impact, mev, priority])


def meme_backtest_fn(panel: TokenPanel, config: MemeBacktestConfig | None = None):
    """Adapter for `LoopConfig.backtest_fn`: positions in, (returns, metrics) out."""

    def run(positions: pd.DataFrame, _price_data: object) -> tuple[pd.Series, dict[str, float]]:
        result = meme_backtest(positions, panel, config)
        return result.returns, result.metrics

    return run
