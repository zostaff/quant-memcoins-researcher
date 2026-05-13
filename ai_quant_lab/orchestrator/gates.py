"""Three gates a strategy must pass before it earns a paper-trade slot.

    1. Critic gate:        adversarial review must not produce 'kill'.
    2. Deflated Sharpe:    p-value below `dsr_pvalue_max`. Hard gate, no override.
    3. Correlation gate:   max |corr| with any accepted strategy below `max_correlation`.

The gates are an AND. One failure kills. The order matters only for efficiency:
critic is cheap (1 LLM call), DSR is cheap (one closed-form formula), correlation
is the most expensive when there are many survivors. So we check in that order.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai_quant_lab.agents.critic import CriticVerdict
from ai_quant_lab.agents.memory import ResearchMemory
from ai_quant_lab.config import settings
from ai_quant_lab.validation.deflated_sharpe import DeflatedSharpeResult, deflated_sharpe
from ai_quant_lab.validation.factor_attribution import factor_concentration_score


@dataclass(frozen=True)
class GateOutcome:
    """Aggregate gate result.

    `passes` is True only if all sub-gates pass. `rejection_reason` is the
    name of the first gate to fail (None if all pass).

    Gates in evaluation order:
        critic → deflated_sharpe → pairwise_correlation → pca_concentration
    """

    passes: bool
    rejection_reason: str | None
    critic_verdict: CriticVerdict | None
    dsr_result: DeflatedSharpeResult | None
    max_correlation: float | None
    pca_concentration: float | None = None


def evaluate_gates(
    critic_verdict: CriticVerdict,
    strategy_returns: pd.Series,
    *,
    memory: ResearchMemory,
    accepted_returns: list[pd.Series] | None = None,
    annualization: int | None = None,
    dsr_pvalue_max: float | None = None,
    max_correlation: float | None = None,
    max_pca_concentration: float = 0.5,
) -> GateOutcome:
    """Run all three gates in order.

    Args:
        critic_verdict: Output of CriticAgent.
        strategy_returns: Net returns of the candidate.
        memory: Used to fetch n_trials for the deflated_sharpe gate.
        accepted_returns: Returns series of strategies already accepted.
        annualization: Override; falls back to settings.
        dsr_pvalue_max: Override; falls back to settings.
        max_correlation: Override; falls back to settings.
    """
    dsr_pvalue_max = dsr_pvalue_max if dsr_pvalue_max is not None else settings.dsr_pvalue_max
    max_correlation = max_correlation if max_correlation is not None else settings.max_correlation
    annualization = annualization if annualization is not None else settings.annualization

    if not critic_verdict.passes:
        return GateOutcome(
            passes=False,
            rejection_reason="critic",
            critic_verdict=critic_verdict,
            dsr_result=None,
            max_correlation=None,
        )

    n_trials = max(memory.n_trials(), 1)
    try:
        dsr = deflated_sharpe(
            strategy_returns,
            n_trials=n_trials,
            annualization=annualization,
        )
    except ValueError:
        return GateOutcome(
            passes=False,
            rejection_reason="dsr_insufficient_data",
            critic_verdict=critic_verdict,
            dsr_result=None,
            max_correlation=None,
        )

    if dsr.pvalue >= dsr_pvalue_max:
        return GateOutcome(
            passes=False,
            rejection_reason=f"deflated_sharpe_pvalue={dsr.pvalue:.3f}>={dsr_pvalue_max}",
            critic_verdict=critic_verdict,
            dsr_result=dsr,
            max_correlation=None,
        )

    max_corr = _max_abs_correlation(strategy_returns, accepted_returns or [])
    if max_corr >= max_correlation:
        return GateOutcome(
            passes=False,
            rejection_reason=f"correlation={max_corr:.2f}>={max_correlation}",
            critic_verdict=critic_verdict,
            dsr_result=dsr,
            max_correlation=max_corr,
        )

    # PCA concentration: catches "stealthy" redundancy where pairwise corr is
    # OK but the candidate still loads on the survivor set's main driver.
    # Needs >= 2 survivors to even compute; defaults to 0 with fewer.
    pca_score = factor_concentration_score(strategy_returns, accepted_returns or [])
    if pca_score >= max_pca_concentration:
        return GateOutcome(
            passes=False,
            rejection_reason=f"pca_concentration={pca_score:.2f}>={max_pca_concentration}",
            critic_verdict=critic_verdict,
            dsr_result=dsr,
            max_correlation=max_corr,
            pca_concentration=pca_score,
        )

    return GateOutcome(
        passes=True,
        rejection_reason=None,
        critic_verdict=critic_verdict,
        dsr_result=dsr,
        max_correlation=max_corr,
        pca_concentration=pca_score,
    )


def _max_abs_correlation(candidate: pd.Series, accepted: list[pd.Series]) -> float:
    if not accepted:
        return 0.0
    correlations: list[float] = []
    candidate = candidate.dropna()
    for other in accepted:
        joined = pd.concat([candidate, other.dropna()], axis=1, join="inner").dropna()
        if len(joined) < 30:
            continue
        if joined.iloc[:, 0].std(ddof=1) == 0 or joined.iloc[:, 1].std(ddof=1) == 0:
            continue
        c = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        if pd.notna(c):
            correlations.append(abs(float(c)))
    return max(correlations) if correlations else 0.0
