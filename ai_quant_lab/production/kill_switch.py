"""KillSwitch: trip on hard rules, halt the strategy.

Rules are configured, not negotiated. Examples:
    - Drawdown exceeds X.
    - Daily loss exceeds Y.
    - Live Sharpe over the last N days falls below Z.

Once tripped, the switch stays tripped until manually reset. There is no
"the strategy was about to recover" override — that's how blow-ups happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class KillSwitchTrigger:
    """One rule. `name` is human-readable; `predicate` returns True to trip."""

    name: str
    predicate: Callable[[pd.Series], bool]
    description: str = ""


@dataclass
class KillSwitch:
    """Composite kill switch evaluating a list of triggers each call.

    Once `tripped` is True, subsequent `check()` calls keep it tripped and
    record the original reason. `reset()` is manual and explicit.
    """

    triggers: list[KillSwitchTrigger] = field(default_factory=list)
    tripped: bool = False
    trip_reason: str | None = None
    trip_time: str | None = None
    history: list[dict] = field(default_factory=list)

    def check(self, live_returns: pd.Series) -> bool:
        """Evaluate all triggers. Returns True if tripped (now or previously)."""
        if self.tripped:
            return True
        for trigger in self.triggers:
            try:
                fired = bool(trigger.predicate(live_returns))
            except Exception as exc:  # noqa: BLE001 — predicate errors should be loud
                self.history.append(
                    {"event": "predicate_error", "trigger": trigger.name, "error": str(exc)}
                )
                continue
            if fired:
                self.tripped = True
                self.trip_reason = trigger.name
                self.trip_time = datetime.now(timezone.utc).isoformat()
                self.history.append(
                    {
                        "event": "tripped",
                        "trigger": trigger.name,
                        "description": trigger.description,
                        "time": self.trip_time,
                    }
                )
                return True
        return False

    def reset(self, note: str = "") -> None:
        """Manual reset. Records who and why for the audit trail."""
        self.history.append(
            {
                "event": "reset",
                "previous_reason": self.trip_reason,
                "note": note,
                "time": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.tripped = False
        self.trip_reason = None
        self.trip_time = None


def drawdown_trigger(max_drawdown_pct: float) -> KillSwitchTrigger:
    """Trip when the live equity curve drawdown exceeds `max_drawdown_pct` (e.g. 0.10)."""
    if max_drawdown_pct <= 0:
        raise ValueError("max_drawdown_pct must be positive.")

    def predicate(returns: pd.Series) -> bool:
        if returns.empty:
            return False
        equity = (1.0 + returns).cumprod()
        peak = equity.cummax()
        drawdown = (equity / peak - 1.0).min()
        return bool(drawdown <= -abs(max_drawdown_pct))

    return KillSwitchTrigger(
        name=f"drawdown>{max_drawdown_pct:.2%}",
        predicate=predicate,
        description=f"Halt when drawdown exceeds {max_drawdown_pct:.2%}.",
    )


def daily_loss_trigger(max_daily_loss_pct: float) -> KillSwitchTrigger:
    """Trip on any single-day loss worse than `max_daily_loss_pct` (e.g. 0.03)."""
    if max_daily_loss_pct <= 0:
        raise ValueError("max_daily_loss_pct must be positive.")

    def predicate(returns: pd.Series) -> bool:
        return bool(returns.min() <= -abs(max_daily_loss_pct)) if not returns.empty else False

    return KillSwitchTrigger(
        name=f"daily_loss>{max_daily_loss_pct:.2%}",
        predicate=predicate,
        description=f"Halt on any single-day loss past {max_daily_loss_pct:.2%}.",
    )


def sharpe_collapse_trigger(threshold: float, window: int = 60) -> KillSwitchTrigger:
    """Trip when rolling Sharpe over `window` falls below `threshold`."""

    def predicate(returns: pd.Series) -> bool:
        recent = returns.tail(window).dropna()
        if len(recent) < window:
            return False
        std = recent.std(ddof=1)
        if std == 0:
            return False
        sharpe = (recent.mean() / std) * (252**0.5)
        return bool(sharpe < threshold)

    return KillSwitchTrigger(
        name=f"sharpe<{threshold}",
        predicate=predicate,
        description=f"Halt when {window}-day rolling Sharpe falls below {threshold}.",
    )
