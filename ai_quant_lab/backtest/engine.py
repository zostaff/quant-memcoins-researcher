"""Vectorized backtest engine.

Trades a single position series against a single returns series, with
transaction costs charged on absolute position change. All returns are
in fractional terms (0.01 = 1%), not bps or percent.

The crucial detail is the shift: `position.shift(1) * returns` enforces that
today's position decision is acted on tomorrow's return. Without the shift,
the backtest peeks at the future and inflates Sharpe by 2-5x.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ai_quant_lab.backtest.costs import realistic_cost_bps


@dataclass(frozen=True)
class BacktestConfig:
    """All knobs for a backtest run.

    Attributes:
        cost_bps: Per-trade transaction cost in basis points, charged on
            absolute position change. Default mirrors `realistic_cost_bps()`.
        annualization: Periods per year. 252 for daily equities.
        execution_lag: Bars between decision and fill. 1 = decide on close,
            fill on next bar. Set to 0 only for synthetic tests.
        position_bounds: Hard cap on position size, applied after the strategy.
            Prevents `nan` or runaway signals from blowing up the equity curve.
    """

    cost_bps: float = field(default_factory=realistic_cost_bps)
    annualization: int = 252
    execution_lag: int = 1
    position_bounds: tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self) -> None:
        if self.execution_lag < 0:
            raise ValueError("execution_lag must be >= 0")
        low, high = self.position_bounds
        if low >= high:
            raise ValueError("position_bounds: low must be < high")

    @classmethod
    def from_schedule(
        cls,
        schedule: "BarSchedule",
        *,
        cost_bps: float | None = None,
        execution_lag: int = 1,
    ) -> "BacktestConfig":
        """Build a config whose annualization matches a BarSchedule.

        Lets you do `BacktestConfig.from_schedule(BarSchedule.hourly())` and
        get the right Sharpe annualization for 1-hour bars (≈ 1764 vs the
        usual 252).
        """
        from ai_quant_lab.backtest.bar_engine import BarSchedule  # local to break cycle
        if not isinstance(schedule, BarSchedule):
            raise TypeError("schedule must be a BarSchedule")
        kwargs: dict = {
            "annualization": schedule.annualization,
            "execution_lag": execution_lag,
        }
        if cost_bps is not None:
            kwargs["cost_bps"] = cost_bps
        return cls(**kwargs)


@dataclass(frozen=True)
class BacktestResult:
    """Result of a vectorized backtest. Returns are net of costs."""

    returns: pd.Series
    positions: pd.Series
    turnover: pd.Series
    equity_curve: pd.Series
    metrics: dict[str, float]


def vectorized_backtest(
    positions: pd.Series,
    returns: pd.Series,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Backtest a position series against a return series.

    Args:
        positions: Target positions in units of [-1, 1] (or wider, see config).
            Index must be sortable and align with `returns`.
        returns: Single-period fractional returns of the underlying.
        config: Backtest configuration. Defaults to realistic costs and 1-bar lag.

    Returns:
        BacktestResult with net returns, equity curve, and headline metrics.

    The strategy returns are computed as `lagged_position * returns - costs`.
    `lagged_position` uses `config.execution_lag` to prevent lookahead.
    """
    config = config or BacktestConfig()

    positions, returns = positions.align(returns, join="inner")
    if len(positions) < 2:
        raise ValueError("Need at least 2 aligned observations to backtest.")

    low, high = config.position_bounds
    positions = positions.clip(lower=low, upper=high).fillna(0.0)

    lagged = positions.shift(config.execution_lag).fillna(0.0)
    gross_returns = lagged * returns

    turnover = lagged.diff().abs().fillna(lagged.abs())
    cost_fraction = config.cost_bps / 1e4
    net_returns = gross_returns - turnover * cost_fraction

    equity = (1.0 + net_returns).cumprod()
    metrics = performance_metrics(net_returns, turnover=turnover, annualization=config.annualization)

    return BacktestResult(
        returns=net_returns,
        positions=positions,
        turnover=turnover,
        equity_curve=equity,
        metrics=metrics,
    )


def performance_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    annualization: int = 252,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """Headline performance metrics.

    All metrics are computed in a numerically robust way — zero-volatility
    series return Sharpe=0 rather than NaN/inf, since that's the honest answer.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return {k: 0.0 for k in _METRIC_KEYS}

    mean_return = float(returns.mean())
    std_return = float(returns.std(ddof=1))
    annualized_return = float(mean_return * annualization)
    annualized_volatility = float(std_return * np.sqrt(annualization))

    sharpe = _safe_sharpe(returns, risk_free, annualization)
    sortino = _safe_sortino(returns, risk_free, annualization)
    max_drawdown = _max_drawdown(returns)
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    hit_rate = float((returns > 0).mean())

    avg_turnover = float(turnover.mean()) if turnover is not None else 0.0
    annual_turnover = avg_turnover * annualization

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "hit_rate": hit_rate,
        "n_observations": float(len(returns)),
        "annual_turnover": annual_turnover,
    }


_METRIC_KEYS: tuple[str, ...] = (
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "hit_rate",
    "n_observations",
    "annual_turnover",
)


def _safe_sharpe(returns: pd.Series, risk_free: float, annualization: int) -> float:
    excess = returns - (risk_free / annualization)
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(annualization))


def _safe_sortino(returns: pd.Series, risk_free: float, annualization: int) -> float:
    excess = returns - (risk_free / annualization)
    downside = excess.clip(upper=0.0)
    downside_std = np.sqrt((downside**2).mean())
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(annualization))


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min()) if len(drawdown) else 0.0


Direction = Literal["long", "short", "both"]


def signal_to_position(signal: pd.Series, direction: Direction = "both") -> pd.Series:
    """Convert a continuous signal into a bounded position.

    Helpful for examples where a strategy outputs a z-score or rank and we want
    to feed it directly into the backtest engine.
    """
    if direction == "long":
        return signal.clip(lower=0.0, upper=1.0)
    if direction == "short":
        return signal.clip(lower=-1.0, upper=0.0)
    return signal.clip(lower=-1.0, upper=1.0)
