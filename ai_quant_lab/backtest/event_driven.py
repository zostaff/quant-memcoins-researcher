"""Event-driven backtest used to verify and stress-test vectorized results.

Two engines, both bar-by-bar:

    EventDrivenBacktest      — basic: spread cost, fixed lag.
    RealisticEventDriven     — adds slippage, partial fills via participation
                                cap, and bid-ask side modeling.

Use `EventDrivenBacktest` to verify the vectorized engine agrees on a clean
problem. Use `RealisticEventDriven` to find out how brittle the strategy is
to realistic frictions. A strategy whose vectorized Sharpe is 1.5 and whose
realistic Sharpe is 0.2 is not a strategy that will work in production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_quant_lab.backtest.engine import BacktestConfig, performance_metrics


@dataclass
class EventDrivenBacktest:
    """Bar-by-bar simulator: fixed cost, fixed lag, no fill limits."""

    config: BacktestConfig

    def run(self, positions: pd.Series, returns: pd.Series) -> dict[str, object]:
        positions, returns = positions.align(returns, join="inner")
        if len(positions) < 2:
            raise ValueError("Need at least 2 aligned observations.")

        low, high = self.config.position_bounds
        targets = positions.clip(lower=low, upper=high).fillna(0.0).to_numpy()
        rets = returns.fillna(0.0).to_numpy()

        n = len(targets)
        held = np.zeros(n)
        net = np.zeros(n)
        turnover = np.zeros(n)

        lag = self.config.execution_lag
        cost = self.config.cost_bps / 1e4
        current = 0.0

        for i in range(n):
            new_target = targets[i - lag] if i >= lag else 0.0
            traded = abs(new_target - current)
            turnover[i] = traded
            current = new_target
            held[i] = current
            net[i] = current * rets[i] - traded * cost

        idx = returns.index
        net_returns = pd.Series(net, index=idx, name="returns")
        return {
            "returns": net_returns,
            "positions": pd.Series(held, index=idx, name="positions"),
            "turnover": pd.Series(turnover, index=idx, name="turnover"),
            "equity_curve": (1.0 + net_returns).cumprod(),
            "metrics": performance_metrics(
                net_returns,
                turnover=pd.Series(turnover, index=idx),
                annualization=self.config.annualization,
            ),
        }


@dataclass
class RealisticEventDriven:
    """Bar-by-bar simulator with slippage, participation cap, and side-aware spread.

    Attributes:
        config: standard BacktestConfig (cost_bps treated as spread).
        slippage_coefficient: bps of impact per unit of participation. Default
            10 bps at 100% participation, in line with practitioner desks.
        max_participation_per_bar: maximum fraction of a bar's "capacity" the
            strategy can consume in a single trade. Trades larger than this
            are partially filled, with the residual carried forward.
        side_aware_spread: if True, charge the full spread on every trade
            (paying the bid-ask). If False, average it.

    Notes:
        Participation is measured against `position_capacity` — a synthetic
        proxy for ADV. In a real system you'd pass actual ADV; here we use
        a flat constant scaled by max_participation_per_bar so the engine
        works on plain return series.
    """

    config: BacktestConfig
    slippage_coefficient: float = 10.0
    max_participation_per_bar: float = 0.25
    side_aware_spread: bool = True

    def run(
        self,
        positions: pd.Series,
        returns: pd.Series,
        *,
        participation_capacity: pd.Series | float = 1.0,
    ) -> dict[str, object]:
        """Run the realistic simulator.

        Args:
            positions: target positions.
            returns: single-period returns of the underlying.
            participation_capacity: scalar or per-bar series of "how much can
                we trade this bar at normal cost." Smaller values force the
                strategy to break trades up over multiple bars.
        """
        positions, returns = positions.align(returns, join="inner")
        if len(positions) < 2:
            raise ValueError("Need at least 2 aligned observations.")

        low, high = self.config.position_bounds
        targets = positions.clip(lower=low, upper=high).fillna(0.0).to_numpy()
        rets = returns.fillna(0.0).to_numpy()

        if isinstance(participation_capacity, pd.Series):
            capacity = participation_capacity.reindex(returns.index).fillna(method="ffill").to_numpy()
        else:
            capacity = np.full(len(targets), float(participation_capacity))

        spread = self.config.cost_bps / 1e4
        slip_per_unit = self.slippage_coefficient / 1e4
        n = len(targets)

        held = np.zeros(n)
        net = np.zeros(n)
        turnover = np.zeros(n)
        unfilled = np.zeros(n)

        lag = self.config.execution_lag
        current = 0.0
        pending_target = 0.0  # carried forward across bars when partial-filled

        for i in range(n):
            decision_index = i - lag
            if decision_index >= 0:
                # Newest order arrives; we re-target to it
                pending_target = targets[decision_index]

            desired_trade = pending_target - current
            if desired_trade == 0.0:
                held[i] = current
                continue

            # Cap by participation; the residual stays as pending_target.
            cap = max(capacity[i] * self.max_participation_per_bar, 1e-9)
            executed_size = float(np.sign(desired_trade)) * min(abs(desired_trade), cap)
            unfilled[i] = abs(desired_trade) - abs(executed_size)

            spread_cost = abs(executed_size) * (spread if self.side_aware_spread else spread * 0.5)
            participation = abs(executed_size) / max(capacity[i], 1e-9)
            slippage_cost = abs(executed_size) * slip_per_unit * np.sqrt(participation)

            current = current + executed_size
            turnover[i] = abs(executed_size)
            held[i] = current
            net[i] = current * rets[i] - spread_cost - slippage_cost

        idx = returns.index
        net_returns = pd.Series(net, index=idx, name="returns")
        return {
            "returns": net_returns,
            "positions": pd.Series(held, index=idx, name="positions"),
            "turnover": pd.Series(turnover, index=idx, name="turnover"),
            "unfilled": pd.Series(unfilled, index=idx, name="unfilled"),
            "equity_curve": (1.0 + net_returns).cumprod(),
            "metrics": performance_metrics(
                net_returns,
                turnover=pd.Series(turnover, index=idx),
                annualization=self.config.annualization,
            ),
        }


def reconcile(
    vectorized_returns: pd.Series,
    event_driven_returns: pd.Series,
    tolerance_bps: float = 5.0,
) -> dict[str, float]:
    """Verify two return series agree to within `tolerance_bps` cumulative."""
    aligned_v, aligned_e = vectorized_returns.align(event_driven_returns, join="inner")
    gap = (1.0 + aligned_v).prod() - (1.0 + aligned_e).prod()
    max_bar_gap = float((aligned_v - aligned_e).abs().max())
    return {
        "total_return_gap": float(gap),
        "max_bar_gap_bps": max_bar_gap * 1e4,
        "agrees": float(abs(gap) * 1e4 <= tolerance_bps),
    }
