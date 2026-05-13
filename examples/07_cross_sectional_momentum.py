"""Example 07 — cross-sectional momentum, three iterations.

Mirrors the "3-iteration example" from the article: start with naive cross-
sectional momentum, observe what fails on walk-forward, refine. Each iteration
is more constrained than the last. The point is NOT to find a profitable
strategy; the point is to show how validation drives revision.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd

from ai_quant_lab.backtest import BacktestConfig
from ai_quant_lab.features.library import momentum, realized_volatility
from ai_quant_lab.validation import walk_forward_evaluate
from ai_quant_lab.validation.deflated_sharpe import deflated_sharpe


def synthetic_universe(n_bars: int = 1260, n_assets: int = 20, seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic universe with mild cross-sectional momentum."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0003, 0.008, n_bars)
    idiosyncratic = rng.normal(0.0, 0.012, (n_bars, n_assets))
    shocks = common[:, None] + idiosyncratic
    # Inject mild momentum: asset i carries a fraction of its prior shock.
    for t in range(1, n_bars):
        shocks[t] = shocks[t] + 0.04 * shocks[t - 1]
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    index = pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n_bars)
    return pd.DataFrame(prices, index=index, columns=[f"A{i:02d}" for i in range(n_assets)])


def _basket_returns(positions: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    lagged = positions.shift(1).fillna(0.0)
    return (lagged * returns).sum(axis=1) / lagged.abs().sum(axis=1).replace(0, np.nan)


def iteration_1(prices: pd.DataFrame) -> pd.Series:
    """Naïve: rank by 21-day momentum, long top quintile, short bottom."""
    mom = pd.DataFrame({c: momentum(prices[c], 21) for c in prices.columns})
    ranks = mom.rank(axis=1, pct=True)
    positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    positions[ranks >= 0.8] = 1.0
    positions[ranks <= 0.2] = -1.0
    return _basket_returns(positions.fillna(0.0), prices.pct_change()).fillna(0.0)


def iteration_2(prices: pd.DataFrame) -> pd.Series:
    """Vol-scale each position. Reduces drag from high-vol names dominating PnL."""
    mom = pd.DataFrame({c: momentum(prices[c], 21) for c in prices.columns})
    rv = pd.DataFrame({c: realized_volatility(prices[c], 21) for c in prices.columns})
    ranks = mom.rank(axis=1, pct=True)
    positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    positions[ranks >= 0.8] = 1.0
    positions[ranks <= 0.2] = -1.0
    inverse_vol_weight = (1.0 / rv).clip(upper=20.0)
    positions = positions * inverse_vol_weight
    return _basket_returns(positions.fillna(0.0), prices.pct_change()).fillna(0.0)


def iteration_3(prices: pd.DataFrame) -> pd.Series:
    """Iteration 2 + skip-one-month: use t-21 to t-2 momentum, drop the most recent week."""
    skip = 5
    lookback = 21
    skipped_mom = pd.DataFrame(
        {c: prices[c].shift(skip).pct_change(lookback) for c in prices.columns}
    )
    rv = pd.DataFrame({c: realized_volatility(prices[c], 21) for c in prices.columns})
    ranks = skipped_mom.rank(axis=1, pct=True)
    positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    positions[ranks >= 0.8] = 1.0
    positions[ranks <= 0.2] = -1.0
    inverse_vol_weight = (1.0 / rv).clip(upper=20.0)
    positions = positions * inverse_vol_weight
    return _basket_returns(positions.fillna(0.0), prices.pct_change()).fillna(0.0)


def main() -> None:
    prices = synthetic_universe()
    config = BacktestConfig(cost_bps=10.0)

    for i, fn in enumerate([iteration_1, iteration_2, iteration_3], start=1):
        rets = fn(prices)
        sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(252)
        dsr = deflated_sharpe(rets, n_trials=i)
        print(f"Iteration {i}: Sharpe={sharpe:+.2f}, deflated={dsr.deflated_sharpe_ratio:+.2f}, p={dsr.pvalue:.3f}")


if __name__ == "__main__":
    main()
