"""Example 08 — simulated paper trading with live diagnostics.

Trains a strategy on the first 80% of a synthetic series, "deploys" it on the
last 20%, runs daily diagnostics and a kill switch. Shows the production
machinery without needing a broker connection.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd

from ai_quant_lab.backtest import BacktestConfig, vectorized_backtest
from ai_quant_lab.features.library import momentum
from ai_quant_lab.production import (
    KillSwitch,
    LiveDiagnostic,
    MetricsCollector,
)
from ai_quant_lab.production.kill_switch import drawdown_trigger, sharpe_collapse_trigger


def main() -> None:
    rng = np.random.default_rng(seed=8)
    n = 2520
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))),
        index=pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n),
        name="close",
    )
    cutoff = int(0.8 * n)

    def strategy(p: pd.Series) -> pd.Series:
        return np.sign(momentum(p, 21)).clip(-1, 1).fillna(0.0)

    config = BacktestConfig(cost_bps=8.0)
    backtest_returns = vectorized_backtest(
        strategy(prices.iloc[:cutoff]), prices.iloc[:cutoff].pct_change(), config=config
    ).returns

    live_returns = vectorized_backtest(
        strategy(prices.iloc[cutoff - 252 :]),  # warm-up window
        prices.iloc[cutoff - 252 :].pct_change(),
        config=config,
    ).returns.iloc[252:]

    diag = LiveDiagnostic(backtest_returns=backtest_returns, window_days=60)
    kill = KillSwitch(
        triggers=[
            drawdown_trigger(0.10),
            sharpe_collapse_trigger(threshold=-0.5, window=60),
        ]
    )
    metrics = MetricsCollector()

    daily_returns_seen: list[float] = []
    for i, (timestamp, ret) in enumerate(live_returns.items()):
        daily_returns_seen.append(ret)
        seen = pd.Series(daily_returns_seen, index=live_returns.index[: i + 1])
        metrics.observe("daily_return", ret)
        if i % 20 == 0 and i >= 60:
            report = diag.diagnose(seen)
            print(f"day {i:3d} ({timestamp.date()}): {report['status']}  flags={report.get('flags', [])}")
        if kill.check(seen):
            print(f"day {i:3d}: KILL SWITCH TRIPPED — {kill.trip_reason}")
            break
    print()
    print(f"Live observations: {len(daily_returns_seen)}")
    print(f"Kill switch tripped: {kill.tripped} (reason: {kill.trip_reason})")
    print("Metrics snapshot:")
    print(metrics.snapshot())


if __name__ == "__main__":
    main()
