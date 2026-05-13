"""Reporting helpers: plot equity curves and emit one-page text reports.

matplotlib is an optional dependency. Functions that need it raise a clean
ImportError if it isn't installed, so the core engine stays light.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ai_quant_lab.backtest.engine import BacktestResult


def equity_curve_plot(result: BacktestResult, title: str = "Equity Curve", path: str | None = None) -> Any:
    """Render an equity curve. Saves to `path` if given, returns the Figure either way.

    Uses a dark style consistent with the article figures.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("Install with: pip install matplotlib") from exc

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 4), dpi=120)
    result.equity_curve.plot(ax=ax, color="#7AD6F8", linewidth=1.5)
    ax.set_title(title, color="#E0E0E0")
    ax.set_ylabel("Equity (start = 1.0)", color="#B0B0B0")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    return fig


def text_report(result: BacktestResult, label: str = "Strategy") -> str:
    """One-screen text report of a backtest. Useful in CI and CLI runs."""
    m = result.metrics
    lines = [
        f"=== {label} ===",
        f"Period:         {_period_str(result.returns)}",
        f"Observations:   {int(m['n_observations'])}",
        f"Ann. return:    {m['annualized_return']:+.2%}",
        f"Ann. vol:       {m['annualized_volatility']:.2%}",
        f"Sharpe ratio:   {m['sharpe_ratio']:.2f}",
        f"Sortino ratio:  {m['sortino_ratio']:.2f}",
        f"Max drawdown:   {m['max_drawdown']:.2%}",
        f"Calmar ratio:   {m['calmar_ratio']:.2f}",
        f"Hit rate:       {m['hit_rate']:.2%}",
        f"Ann. turnover:  {m['annual_turnover']:.2f}",
    ]
    return "\n".join(lines)


def _period_str(series: pd.Series) -> str:
    if len(series) == 0:
        return "(empty)"
    return f"{series.index[0]} → {series.index[-1]}"
