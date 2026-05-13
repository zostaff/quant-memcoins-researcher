"""Example 03 — side-by-side leaky vs clean feature.

Two classic leakage shapes:
    1. Direct forward reference: feature uses `price.shift(-k)`. The detector
       flags this via the absolute-future-correlation rule.
    2. Centered rolling: feature uses `rolling(..., center=True)`. Both past
       and future correlations are inflated, so the ratio test alone misses
       it — but the absolute-future-correlation rule still flags it.

Lesson: an automated detector is necessary but not sufficient. Anything that
survives the detector still has to pass the deflated-Sharpe gate downstream.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd

from ai_quant_lab.backtest import BacktestConfig, vectorized_backtest
from ai_quant_lab.features.leakage_detector import detect_leakage
from ai_quant_lab.features.library import momentum


def forward_reference_leak(price_data: pd.Series, peek: int = 3) -> pd.Series:
    """Deliberately bad: explicitly reads `peek` bars into the future."""
    future_price = price_data.shift(-peek)
    return (future_price / price_data - 1.0).rename(f"peek_{peek}")


def centered_window_leak(price_data: pd.Series, window: int = 21) -> pd.Series:
    """Sneakier: a centered rolling mean. Past and future contributions are
    symmetric, so a future/past correlation ratio cannot detect it."""
    centered_mean = price_data.rolling(window, center=True, min_periods=1).mean()
    return (price_data / centered_mean - 1.0).rename(f"centered_{window}")


def main() -> None:
    rng = np.random.default_rng(seed=3)
    n = 2000
    returns = rng.normal(0.0002, 0.012, n)
    price_data = pd.Series(
        100.0 * np.exp(np.cumsum(returns)),
        index=pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n),
        name="close",
    )

    clean = momentum(price_data, 21)
    peek = forward_reference_leak(price_data, 3)
    centered = centered_window_leak(price_data, 21)

    fwd = price_data.pct_change().shift(-1)
    features = pd.DataFrame({"clean": clean, "peek_3": peek, "centered_21": centered})
    report = detect_leakage(features, fwd)
    print("== Leakage report ==")
    print(report.format_problems())
    print("\nColumn scores (future/past correlation ratio):")
    for col, score in report.column_scores.items():
        print(f"  {col}: {score:.2f}")
    print()

    rets = price_data.pct_change()
    config = BacktestConfig(cost_bps=8.0)
    clean_result = vectorized_backtest(np.sign(clean).clip(-1, 1).fillna(0.0), rets, config=config)
    peek_result = vectorized_backtest(np.sign(peek).clip(-1, 1).fillna(0.0), rets, config=config)
    centered_result = vectorized_backtest(np.sign(centered).clip(-1, 1).fillna(0.0), rets, config=config)
    print(f"Clean Sharpe:        {clean_result.metrics['sharpe_ratio']:+.2f}  (honest)")
    print(f"Peek-3 Sharpe:       {peek_result.metrics['sharpe_ratio']:+.2f}  (detector caught it)")
    print(f"Centered-21 Sharpe:  {centered_result.metrics['sharpe_ratio']:+.2f}  (detector caught it)")


if __name__ == "__main__":
    main()
