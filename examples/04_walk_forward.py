"""Example 04 — walk-forward evaluation of a momentum strategy.

Demonstrates that out-of-sample Sharpe is what matters. The in-sample
backtest is just a curiosity; the concatenated out-of-sample curve is what
you'd actually have traded.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd

from ai_quant_lab.backtest import BacktestConfig, vectorized_backtest
from ai_quant_lab.features.library import momentum
from ai_quant_lab.validation import walk_forward_evaluate
from ai_quant_lab.validation.diagnostics import degradation_ratio, fold_stability


def strategy(price_data: pd.Series) -> pd.Series:
    """21-day momentum, sign-only. No parameters to fit."""
    signal = momentum(price_data, window=21)
    return np.sign(signal).clip(-1, 1).fillna(0.0)


def main() -> None:
    rng = np.random.default_rng(seed=4)
    n = 2520
    # Add a faint momentum drift so the strategy has SOMETHING to find.
    autocorrelated_shocks = np.zeros(n)
    last = 0.0
    for i in range(n):
        shock = rng.normal(0.0003, 0.011)
        autocorrelated_shocks[i] = 0.05 * last + shock
        last = autocorrelated_shocks[i]
    price_data = pd.Series(
        100.0 * np.exp(np.cumsum(autocorrelated_shocks)),
        index=pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n),
        name="close",
    )
    config = BacktestConfig(cost_bps=8.0)

    in_sample = vectorized_backtest(strategy(price_data), price_data.pct_change(), config=config)
    print(f"In-sample Sharpe (whole series): {in_sample.metrics['sharpe_ratio']:+.2f}")

    wf = walk_forward_evaluate(
        price_data, strategy,
        train_size=504, test_size=126, purge=5, mode="rolling",
        config=config,
    )
    print(f"OOS concatenated Sharpe:        {wf['metrics']['sharpe_ratio']:+.2f}")
    print(f"OOS fold Sharpes:               {[f'{x:+.2f}' for x in wf['fold_sharpes']]}")
    print(f"Degradation ratio (OOS/IS):     {degradation_ratio(in_sample.metrics['sharpe_ratio'], wf['metrics']['sharpe_ratio']):.2f}")
    print(f"Fold stability:                 {fold_stability(wf['fold_sharpes'])}")


if __name__ == "__main__":
    main()
