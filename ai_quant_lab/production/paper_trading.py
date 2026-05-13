"""Live diagnostic: did the strategy survive its first contact with reality?

Compare a rolling window of live (or paper) returns against the backtest's
distribution. A strategy that's broken usually shows one of:
    - Sharpe collapse (live SR < 0.5 × backtest SR over a 60-day window)
    - Hit rate drift (live hit rate outside the ±5pp band)
    - Drawdown beyond the worst backtest drawdown × 1.5

We emit a diagnosis, not a verdict. The kill_switch decides what to do with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai_quant_lab.backtest.engine import performance_metrics


@dataclass
class LiveDiagnostic:
    """Holds a strategy's backtest baseline so live data can be compared to it.

    Attributes:
        backtest_returns: full backtest return series the strategy was built on.
        window_days: rolling window used to evaluate live performance.
        annualization: periods per year.
    """

    backtest_returns: pd.Series
    window_days: int = 60
    annualization: int = 252
    _baseline: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._baseline = performance_metrics(
            self.backtest_returns, annualization=self.annualization
        )

    @property
    def baseline(self) -> dict[str, float]:
        return dict(self._baseline)

    def diagnose(self, live_returns: pd.Series) -> dict[str, object]:
        return diagnose(
            backtest_returns=self.backtest_returns,
            live_returns=live_returns,
            window_days=self.window_days,
            annualization=self.annualization,
            baseline=self._baseline,
        )


def diagnose(
    backtest_returns: pd.Series,
    live_returns: pd.Series,
    *,
    window_days: int = 60,
    annualization: int = 252,
    baseline: dict[str, float] | None = None,
) -> dict[str, object]:
    """Compare live to backtest. Returns a flat dict of diagnostics + flags."""
    baseline = baseline or performance_metrics(backtest_returns, annualization=annualization)
    recent = live_returns.dropna().tail(window_days)
    if len(recent) < max(20, window_days // 3):
        return {
            "status": "insufficient_data",
            "n_live_observations": int(len(recent)),
            "required_minimum": max(20, window_days // 3),
        }

    live_metrics = performance_metrics(recent, annualization=annualization)
    sharpe_ratio_drop = _ratio(live_metrics["sharpe_ratio"], baseline["sharpe_ratio"])
    hit_rate_drift = live_metrics["hit_rate"] - baseline["hit_rate"]
    drawdown_breach = live_metrics["max_drawdown"] < baseline["max_drawdown"] * 1.5

    flags: list[str] = []
    if sharpe_ratio_drop < 0.5:
        flags.append("sharpe_collapse")
    if abs(hit_rate_drift) > 0.05:
        flags.append("hit_rate_drift")
    if drawdown_breach:
        flags.append("drawdown_breach")

    return {
        "status": "ok" if not flags else "degraded",
        "flags": flags,
        "live_sharpe": live_metrics["sharpe_ratio"],
        "baseline_sharpe": baseline["sharpe_ratio"],
        "sharpe_ratio_drop": sharpe_ratio_drop,
        "live_hit_rate": live_metrics["hit_rate"],
        "baseline_hit_rate": baseline["hit_rate"],
        "live_max_drawdown": live_metrics["max_drawdown"],
        "baseline_max_drawdown": baseline["max_drawdown"],
        "n_live_observations": int(len(recent)),
    }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or np.isnan(denominator):
        return 0.0 if numerator <= 0 else 1.0
    return float(numerator / denominator)
