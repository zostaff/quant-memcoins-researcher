"""Example 05 — DSR kills 1000 random strategies.

Generate 1000 random buy/sell signals on a no-edge GBM tape. Pick the best by
in-sample Sharpe. Show that the deflated Sharpe p-value is high — the gate
kills it. Then run the same test with N=1 (we only tried one strategy) and
show that the p-value drops dramatically — DSR is multiplicity-aware.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd

from ai_quant_lab.backtest import BacktestConfig, vectorized_backtest
from ai_quant_lab.validation.deflated_sharpe import deflated_sharpe


def main() -> None:
    rng = np.random.default_rng(seed=5)
    n_bars = 1500
    returns = pd.Series(rng.normal(0.0001, 0.012, n_bars))
    config = BacktestConfig(cost_bps=0.0)  # no costs — pure noise test

    n_trials = 1000
    best_sharpe = -np.inf
    best_returns = None
    for _ in range(n_trials):
        random_positions = pd.Series(rng.choice([-1.0, 1.0], size=n_bars))
        result = vectorized_backtest(random_positions, returns, config=config)
        if result.metrics["sharpe_ratio"] > best_sharpe:
            best_sharpe = result.metrics["sharpe_ratio"]
            best_returns = result.returns

    assert best_returns is not None
    dsr_naive = deflated_sharpe(best_returns, n_trials=1)
    dsr_honest = deflated_sharpe(best_returns, n_trials=n_trials)
    print(f"Best of {n_trials} random strategies:")
    print(f"  Naïve report   — SR={dsr_naive.sharpe_ratio:+.2f}, deflated={dsr_naive.deflated_sharpe_ratio:+.2f}, p={dsr_naive.pvalue:.3f}  (n_trials=1, dishonest)")
    print(f"  Honest report  — SR={dsr_honest.sharpe_ratio:+.2f}, deflated={dsr_honest.deflated_sharpe_ratio:+.2f}, p={dsr_honest.pvalue:.3f}  (n_trials={n_trials})")
    print()
    print(f"  Honest gate passes 5%? {dsr_honest.passes(0.05)}")
    print("  Expected: NO. Picking the best of 1000 noise strategies is noise, not edge.")


if __name__ == "__main__":
    main()
