"""Tear sheet: everything an allocator asks before money moves.

One call answers, for a memecoin weight schedule:
    - what it makes net of fees, curve slippage, priority fees and MEV;
    - how much of the PnL went to execution and how much to rugs;
    - how much capital it can take before the edge is gone;
    - whether the return is alpha or just exposure to the meme market factors;
    - whether the Sharpe survives the number of things you tried (deflated);
    - whether the second half of the sample looks like the first.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai_quant_lab.meme.backtest import MemeBacktestConfig, MemeBacktestResult, meme_backtest
from ai_quant_lab.meme.capacity import DEFAULT_CAPITALS, capacity_curve, estimate_capacity
from ai_quant_lab.meme.factors import meme_attribution, meme_factor_returns
from ai_quant_lab.meme.panel import TokenPanel
from ai_quant_lab.validation.deflated_sharpe import DeflatedSharpeResult, deflated_sharpe
from ai_quant_lab.validation.factor_attribution import AttributionResult


@dataclass(frozen=True)
class TearSheet:
    backtest: MemeBacktestResult
    capacity: pd.DataFrame
    capacity_usd: float
    attribution: AttributionResult | None
    dsr: DeflatedSharpeResult | None
    first_half_sharpe: float
    second_half_sharpe: float

    def summary(self) -> str:
        m = self.backtest.metrics
        costs = self.backtest.costs.sum()
        lines = [
            f"capital            ${m['capital_usd']:,.0f} -> ${m['final_equity_usd']:,.0f} "
            f"({m['total_return']:+.1%})",
            f"sharpe (ann.)      {m['sharpe_ratio']:+.2f}   max DD {m['max_drawdown']:.1%}",
            f"execution cost     ${costs.sum():,.0f}  (fee ${costs['fee']:,.0f}, "
            f"impact ${costs['impact']:,.0f}, mev ${costs['mev']:,.0f}, "
            f"priority ${costs['priority']:,.0f}) = {m['cost_per_dollar_traded']:.2%} per $ traded",
            f"rug write-offs     ${m['rug_loss_usd']:,.0f}",
            f"avg positions      {m['mean_positions']:.1f}   invested {m['mean_invested_fraction']:.0%}"
            f"   capped by pool size {m['capped_fraction']:.0%}",
            f"capacity           ${self.capacity_usd:,.0f}",
            f"halves sharpe      {self.first_half_sharpe:+.2f} / {self.second_half_sharpe:+.2f}",
        ]
        if self.attribution is not None:
            a = self.attribution
            betas = ", ".join(f"{k} {v:+.2f}" for k, v in a.betas.items())
            lines.append(
                f"factor betas       {betas}   R² {a.r_squared:.2f}"
            )
            lines.append(f"alpha t-stat       {a.t_stats['alpha']:+.2f}")
        if self.dsr is not None:
            lines.append(
                f"deflated sharpe    p={self.dsr.pvalue:.3f} over {self.dsr.n_trials} trials "
                f"(noise ceiling SR {self.dsr.expected_max_sharpe:+.2f})"
            )
        return "\n".join(lines)


def tear_sheet(
    weights: pd.DataFrame,
    panel: TokenPanel,
    *,
    n_trials: int,
    config: MemeBacktestConfig | None = None,
    capitals: tuple[float, ...] = DEFAULT_CAPITALS,
    factors: pd.DataFrame | None = None,
) -> TearSheet:
    """Full evaluation. `n_trials` is every variant you tried, honestly counted."""
    config = config or MemeBacktestConfig()
    result = meme_backtest(weights, panel, config)
    curve = capacity_curve(weights, panel, capitals=capitals, config=config)

    if factors is None:
        factors = meme_factor_returns(panel, min_liquidity_usd=config.costs.min_liquidity_usd)
    try:
        attribution = meme_attribution(result.returns, factors)
    except ValueError:
        attribution = None

    annualization = int(result.metrics["annualization"])
    try:
        dsr = deflated_sharpe(result.returns, n_trials=n_trials, annualization=annualization)
    except ValueError:
        dsr = None

    half = len(result.returns) // 2
    first = _sharpe(result.returns.iloc[:half], annualization)
    second = _sharpe(result.returns.iloc[half:], annualization)

    return TearSheet(
        backtest=result,
        capacity=curve,
        capacity_usd=estimate_capacity(curve),
        attribution=attribution,
        dsr=dsr,
        first_half_sharpe=first,
        second_half_sharpe=second,
    )


def _sharpe(returns: pd.Series, annualization: int) -> float:
    std = returns.std(ddof=1)
    if not std or pd.isna(std):
        return 0.0
    return float(returns.mean() / std * annualization**0.5)
