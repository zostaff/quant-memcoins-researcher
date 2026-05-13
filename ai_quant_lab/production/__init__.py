"""Production: the layer between accepted strategies and real money.

The work here is paranoia, not cleverness:
    - LiveDiagnostic checks every day whether the strategy looks like its
      backtest (Sharpe, hit rate, drawdown). If not, flag it.
    - KillSwitch shuts everything down when a hard rule trips.
    - DecisionLog records every order with enough context to audit later.

This module is intentionally lightweight. Production trading needs a broker
integration, position reconciliation, and operational discipline we don't
provide here. What we DO provide is the diagnostic skeleton you can plug
those things into.
"""

from ai_quant_lab.production.audit_log import DecisionLog, DecisionRecord
from ai_quant_lab.production.kill_switch import KillSwitch, KillSwitchTrigger
from ai_quant_lab.production.monitoring import MetricsCollector
from ai_quant_lab.production.paper_trading import LiveDiagnostic, diagnose
from ai_quant_lab.production.tca import (
    Fill,
    TCAReport,
    calibrate_slippage_coefficient,
    compute_tca,
)

__all__ = [
    "DecisionLog",
    "DecisionRecord",
    "Fill",
    "KillSwitch",
    "KillSwitchTrigger",
    "LiveDiagnostic",
    "MetricsCollector",
    "TCAReport",
    "calibrate_slippage_coefficient",
    "compute_tca",
    "diagnose",
]
