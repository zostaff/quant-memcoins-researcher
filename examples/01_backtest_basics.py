"""Example 01 — vectorized backtest on synthetic GBM data.

Goal: show the engine end to end, with no Claude calls. Generates a fake price
series, runs a trivial momentum strategy, prints headline metrics.

Expected output: Sharpe is small and noisy (it's GBM with no edge), turnover
reflects the rebalance frequency, and the engine doesn't blow up.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd

from ai_quant_lab.backtest import BacktestConfig, vectorized_backtest
from ai_quant_lab.backtest.utils import text_report
from ai_quant_lab.features.library import momentum


def main() -> None:
    rng = np.random.default_rng(seed=1)
    n = 2520  # ~10 years of daily bars
    daily_returns = rng.normal(loc=0.05 / 252, scale=0.16 / np.sqrt(252), size=n)
    price_data = pd.Series(100.0 * np.exp(np.cumsum(daily_returns)), name="close",
                           index=pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n))

    signal = momentum(price_data, window=21)
    positions = np.sign(signal).fillna(0.0).clip(-1.0, 1.0).rename("position")

    result = vectorized_backtest(positions, price_data.pct_change(),
                                  config=BacktestConfig(cost_bps=8.0))
    print(text_report(result, label="Momentum (21d) on synthetic GBM"))


if __name__ == "__main__":
    main()
