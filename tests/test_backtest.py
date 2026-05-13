"""Tests for backtest engines.

The key invariant: vectorized and event-driven engines must agree to within
a few bps of total return on the same inputs. Anything else is a bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_quant_lab.backtest import BacktestConfig, vectorized_backtest
from ai_quant_lab.backtest.engine import performance_metrics, signal_to_position
from ai_quant_lab.backtest.event_driven import EventDrivenBacktest, reconcile


def test_vectorized_basic(random_returns):
    positions = pd.Series(np.where(random_returns.shift(1).rolling(10).mean() > 0, 1.0, -1.0),
                          index=random_returns.index)
    result = vectorized_backtest(positions, random_returns)
    assert "sharpe_ratio" in result.metrics
    assert len(result.returns) == len(random_returns)


def test_lookahead_lag_makes_perfect_signal_useless():
    """The execution lag must turn the perfect signal into a non-perfect one.

    With lag=0 (cheating) the perfect signal earns a huge Sharpe. With lag=1
    (honest) it shouldn't — what matters is the *gap* between the two, not the
    absolute value with drift.
    """
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0, 0.01, 1000))  # zero mean to avoid drift bias
    perfect = pd.Series(np.sign(returns), index=returns.index, dtype=float)
    cheat = vectorized_backtest(
        perfect, returns, config=BacktestConfig(cost_bps=0.0, execution_lag=0)
    )
    honest = vectorized_backtest(
        perfect, returns, config=BacktestConfig(cost_bps=0.0, execution_lag=1)
    )
    assert cheat.metrics["sharpe_ratio"] > 5.0  # cheating earns absurd SR
    assert abs(honest.metrics["sharpe_ratio"]) < 1.0  # honest lag breaks the magic


def test_costs_reduce_return(random_returns):
    positions = pd.Series(np.where(random_returns.shift(1) > 0, 1.0, -1.0), index=random_returns.index)
    zero_cost = vectorized_backtest(positions, random_returns, config=BacktestConfig(cost_bps=0.0))
    high_cost = vectorized_backtest(positions, random_returns, config=BacktestConfig(cost_bps=100.0))
    assert high_cost.returns.sum() < zero_cost.returns.sum()


def test_vectorized_event_driven_agreement(random_returns):
    positions = pd.Series(np.where(random_returns.shift(1) > 0, 1.0, -1.0),
                          index=random_returns.index, dtype=float)
    config = BacktestConfig(cost_bps=5.0)
    v = vectorized_backtest(positions, random_returns, config=config)
    e = EventDrivenBacktest(config=config).run(positions, random_returns)
    rec = reconcile(v.returns, e["returns"])
    # Both engines should agree to within 5 bps cumulative.
    assert abs(rec["total_return_gap"] * 1e4) < 5.0, rec


def test_zero_volatility_returns_zero_sharpe():
    returns = pd.Series(np.zeros(100))
    m = performance_metrics(returns)
    assert m["sharpe_ratio"] == 0.0
    assert m["sortino_ratio"] == 0.0


def test_position_bounds_clip():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 200))
    positions = pd.Series(10.0, index=returns.index)
    result = vectorized_backtest(positions, returns)
    assert result.positions.max() <= 1.0


def test_signal_to_position_long_only():
    signal = pd.Series([-0.5, 0.0, 0.5, 2.0])
    long_only = signal_to_position(signal, direction="long")
    assert long_only.min() >= 0.0
    assert long_only.max() <= 1.0


def test_min_observations_raises():
    with pytest.raises(ValueError):
        vectorized_backtest(pd.Series([1.0]), pd.Series([0.01]))
