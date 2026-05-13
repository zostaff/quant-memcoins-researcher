"""Example 09 — full cross-sectional pipeline with PCA gate.

Builds a synthetic universe with mild cross-sectional momentum, runs three
strategies, then shows what the PCA gate catches that pairwise correlation
misses.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ai_quant_lab.backtest import (
    BacktestConfig,
    long_short_quantile_portfolio,
    vectorized_portfolio_backtest,
)
from ai_quant_lab.features.cross_sectional import (
    cross_sectional_momentum,
    rank_within_universe,
    zscore_cross_section,
)
from ai_quant_lab.features.library import realized_volatility
from ai_quant_lab.validation import (
    factor_concentration_score,
    fama_french_attribution,
    pca_decompose,
)


def synthetic_universe(seed: int = 9, n_bars: int = 1500, n_assets: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0003, 0.008, n_bars)[:, None]
    idio = rng.normal(0, 0.012, (n_bars, n_assets))
    shocks = common + idio
    # Mild cross-sectional momentum: yesterday's leaders keep leading
    for t in range(1, n_bars):
        shocks[t] += 0.04 * shocks[t - 1]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(shocks, axis=0)),
        index=pd.bdate_range(end="2026-01-01", periods=n_bars),
        columns=[f"A{i:02d}" for i in range(n_assets)],
    )
    return prices


def strategy_momentum(prices: pd.DataFrame) -> pd.Series:
    signal = cross_sectional_momentum(prices, lookback=21, skip=1)
    positions = long_short_quantile_portfolio(signal, long_quantile=0.8, short_quantile=0.2)
    return vectorized_portfolio_backtest(
        positions, prices.pct_change(), config=BacktestConfig(cost_bps=10.0)
    ).returns


def strategy_momentum_skip(prices: pd.DataFrame) -> pd.Series:
    signal = cross_sectional_momentum(prices, lookback=21, skip=5)
    positions = long_short_quantile_portfolio(signal, long_quantile=0.8, short_quantile=0.2)
    return vectorized_portfolio_backtest(
        positions, prices.pct_change(), config=BacktestConfig(cost_bps=10.0)
    ).returns


def strategy_vol_regime(prices: pd.DataFrame) -> pd.Series:
    """Long-short by inverse vol — a 'low-vol anomaly' style basket."""
    rv = pd.DataFrame({c: realized_volatility(prices[c], 21) for c in prices.columns})
    signal = -rank_within_universe(rv)  # low vol = high signal
    positions = long_short_quantile_portfolio(signal, long_quantile=0.8, short_quantile=0.2)
    return vectorized_portfolio_backtest(
        positions, prices.pct_change(), config=BacktestConfig(cost_bps=10.0)
    ).returns


def main() -> None:
    prices = synthetic_universe()

    strategies = {
        "momentum_21_1": strategy_momentum(prices),
        "momentum_21_5": strategy_momentum_skip(prices),
        "low_vol": strategy_vol_regime(prices),
    }
    matrix = pd.DataFrame(strategies).dropna()
    print("Pairwise correlations:")
    print(matrix.corr().round(2))
    print()

    pca = pca_decompose(matrix)
    print("PCA decomposition:")
    print(f"  PC1 explains {pca.top_concentration():.1%} of variance")
    print(f"  Loadings on PC1:\n{pca.top_loadings.round(2).to_string()}")
    print()

    # The two momentum variants will load similarly on PC1. The low-vol
    # strategy lives on a different component.
    print("Factor concentration of each strategy against the other two:")
    for name in matrix.columns:
        others = [matrix[c] for c in matrix.columns if c != name]
        score = factor_concentration_score(matrix[name], others)
        print(f"  {name}: PC1-overlap score = {score:.2f}")
    print()

    # Optional: regress one strategy on the others to show factor structure.
    print("Regressing momentum_21_1 on the other two (Fama-French style):")
    others_frame = matrix.drop(columns=["momentum_21_1"])
    attribution = fama_french_attribution(matrix["momentum_21_1"], others_frame)
    print(f"  alpha (annualized): {attribution.annualized_alpha:+.2%}")
    print(f"  betas: {attribution.betas.round(2).to_dict()}")
    print(f"  R²: {attribution.r_squared:.2f}")


if __name__ == "__main__":
    main()
