"""Cross-sectional portfolio backtest.

The single-instrument `vectorized_backtest` is fine for index futures and
indices. Anything interesting — pairs, factor-neutral baskets, cross-sectional
momentum — needs a multi-asset engine.

Conventions:
    - positions: DataFrame indexed by time, columns by asset id. Each cell is
      the target weight at time t for that asset. Sign = direction.
    - returns: DataFrame with the same shape; each cell is the single-period
      return of the asset.
    - portfolio return at t = sum_i lagged_position[t, i] * return[t, i],
      normalized by gross exposure so the result is a return on capital.

Costs are charged on per-asset turnover (absolute weight change), then summed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_quant_lab.backtest.engine import BacktestConfig, performance_metrics


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Result of a portfolio backtest."""

    returns: pd.Series                # net portfolio returns
    positions: pd.DataFrame           # realized lagged positions per asset
    turnover: pd.DataFrame            # absolute weight change per asset
    gross_exposure: pd.Series         # sum |weight| at each bar
    net_exposure: pd.Series           # sum weight at each bar (long minus short)
    equity_curve: pd.Series
    metrics: dict[str, float]


def vectorized_portfolio_backtest(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    config: BacktestConfig | None = None,
    *,
    normalize: bool = True,
) -> PortfolioBacktestResult:
    """Backtest a portfolio of N assets.

    Args:
        positions: DataFrame (time × asset). Target weights, sign for direction.
            NaN treated as 0. Clipped per-cell to `config.position_bounds`.
        returns: DataFrame (time × asset), single-period fractional returns.
        config: BacktestConfig — costs, execution lag, annualization.
        normalize: if True, divide gross PnL by gross exposure so the result
            is a return on capital deployed (typical for long-short books).
            If False, returns are notional — fine for fully invested portfolios.

    Returns:
        PortfolioBacktestResult with per-bar exposures, metrics, equity curve.

    The lookahead invariant is the same as the single-asset engine: positions
    at time t are shifted by `config.execution_lag` before multiplying by
    returns at time t.
    """
    config = config or BacktestConfig()

    positions, returns = positions.align(returns, axis=None, join="inner")
    if positions.empty or returns.empty or len(positions) < 2:
        raise ValueError("Need at least 2 aligned bars and a non-empty universe.")

    low, high = config.position_bounds
    positions = positions.clip(lower=low, upper=high).fillna(0.0)
    returns = returns.fillna(0.0)

    lagged_positions = positions.shift(config.execution_lag).fillna(0.0)
    turnover = lagged_positions.diff().abs().fillna(lagged_positions.abs())

    gross_pnl = (lagged_positions * returns).sum(axis=1)
    gross_exposure = lagged_positions.abs().sum(axis=1)
    net_exposure = lagged_positions.sum(axis=1)

    if normalize:
        denominator = gross_exposure.replace(0.0, np.nan)
        portfolio_return = (gross_pnl / denominator).fillna(0.0)
    else:
        portfolio_return = gross_pnl

    cost_fraction = config.cost_bps / 1e4
    cost_per_bar = turnover.sum(axis=1) * cost_fraction
    if normalize:
        cost_per_bar = (cost_per_bar / gross_exposure.replace(0.0, np.nan)).fillna(0.0)
    net_returns = portfolio_return - cost_per_bar

    equity = (1.0 + net_returns).cumprod()
    metrics = performance_metrics(
        net_returns,
        turnover=turnover.sum(axis=1),
        annualization=config.annualization,
    )
    metrics["n_assets"] = float(positions.shape[1])
    metrics["mean_gross_exposure"] = float(gross_exposure.mean())
    metrics["mean_net_exposure"] = float(net_exposure.mean())

    return PortfolioBacktestResult(
        returns=net_returns,
        positions=lagged_positions,
        turnover=turnover,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        equity_curve=equity,
        metrics=metrics,
    )


def long_short_quantile_portfolio(
    signal: pd.DataFrame,
    *,
    long_quantile: float = 0.8,
    short_quantile: float = 0.2,
    weight: str = "equal",
) -> pd.DataFrame:
    """Build a dollar-neutral long-short portfolio from a cross-sectional signal.

    Args:
        signal: DataFrame (time × asset). Higher = more bullish.
        long_quantile: per-bar rank threshold to enter long (e.g. 0.8 = top 20%).
        short_quantile: per-bar rank threshold to enter short.
        weight: 'equal' (default) sizes positions equally; 'signal' sizes by
            distance from the median signal.

    Returns:
        DataFrame of weights summing to ~0 per bar (dollar-neutral), gross 1.
    """
    if not 0.0 < short_quantile < long_quantile < 1.0:
        raise ValueError("Need 0 < short_quantile < long_quantile < 1.")

    ranks = signal.rank(axis=1, pct=True)
    positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    long_mask = ranks >= long_quantile
    short_mask = ranks <= short_quantile

    if weight == "equal":
        long_size = long_mask.sum(axis=1).replace(0, np.nan)
        short_size = short_mask.sum(axis=1).replace(0, np.nan)
        positions = positions.add(
            long_mask.div(long_size, axis=0).fillna(0.0), fill_value=0.0
        )
        positions = positions.add(
            -short_mask.div(short_size, axis=0).fillna(0.0), fill_value=0.0
        )
    elif weight == "signal":
        # Distance from cross-sectional median, kept only in extremes
        median = signal.median(axis=1)
        centered = signal.sub(median, axis=0)
        long_weights = centered.where(long_mask, 0.0)
        short_weights = centered.where(short_mask, 0.0)
        # Normalize each side to gross 0.5
        long_gross = long_weights.abs().sum(axis=1).replace(0, np.nan)
        short_gross = short_weights.abs().sum(axis=1).replace(0, np.nan)
        positions = (
            long_weights.div(long_gross, axis=0).fillna(0.0) * 0.5
            + short_weights.div(short_gross, axis=0).fillna(0.0) * 0.5
        )
    else:
        raise ValueError(f"Unknown weight scheme: {weight!r}")

    return positions
