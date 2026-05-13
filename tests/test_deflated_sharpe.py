"""Tests for the deflated Sharpe gate.

The gate must:
    - reject the best of many random trials (no edge)
    - accept a strategy with a genuine edge after only a few trials
    - be sensitive to n_trials (more trials → harder to pass)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_quant_lab.validation.deflated_sharpe import (
    deflated_sharpe,
    deflated_sharpe_pvalue,
    probabilistic_sharpe,
)


def test_best_of_many_random_trials_does_not_pass():
    """Pick the best of 500 random strategies → should NOT pass at 5%."""
    rng = np.random.default_rng(0)
    n_bars = 1000
    best_sharpe = -np.inf
    best_returns = None
    for _ in range(500):
        returns = pd.Series(rng.normal(0, 0.01, n_bars))
        positions = pd.Series(rng.choice([-1.0, 1.0], size=n_bars))
        strat_returns = positions.shift(1).fillna(0) * returns
        if strat_returns.std() == 0:
            continue
        sharpe = strat_returns.mean() / strat_returns.std()
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_returns = strat_returns

    dsr = deflated_sharpe(best_returns, n_trials=500)
    # The "edge" is noise; honest DSR must not pass at 5%.
    assert not dsr.passes(0.05), dsr


def test_n_trials_monotonically_tightens():
    """Increasing n_trials must monotonically increase the DSR p-value."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 1000))
    pvalues = [deflated_sharpe_pvalue(returns, n_trials=k) for k in (1, 10, 100, 1000)]
    assert pvalues == sorted(pvalues), pvalues


def test_genuine_edge_passes_low_trials():
    """A return series with strong synthetic edge should pass DSR at n_trials=1."""
    rng = np.random.default_rng(0)
    # Annualized SR ~ 3 (very strong)
    returns = pd.Series(rng.normal(3 * 0.01 / np.sqrt(252), 0.01, 1000))
    dsr = deflated_sharpe(returns, n_trials=1)
    assert dsr.passes(0.05), dsr


def test_pvalue_is_in_unit_interval():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 1000))
    p = deflated_sharpe_pvalue(returns, n_trials=100)
    assert 0.0 <= p <= 1.0


def test_psr_is_in_unit_interval():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 1000))
    p = probabilistic_sharpe(returns)
    assert 0.0 <= p <= 1.0


def test_short_series_raises():
    with pytest.raises(ValueError):
        deflated_sharpe(pd.Series([0.01, 0.02, 0.0]), n_trials=1)
