"""Deflated Sharpe Ratio — the unforgiving gate.

When you test N strategies and pick the best, the expected maximum Sharpe rises
with N even when no strategy has any edge. The Deflated Sharpe Ratio (Bailey &
López de Prado, 2014) discounts the observed Sharpe by how much you'd expect
to see by chance given N, the skew, and the kurtosis of returns.

A DSR p-value below the gate (default 5%) is required for a strategy to live.
There is no override. If you find yourself wanting one, you're p-hacking.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Output of the deflated_sharpe gate."""

    sharpe_ratio: float
    expected_max_sharpe: float
    deflated_sharpe_ratio: float
    pvalue: float
    n_trials: int
    n_observations: int

    def passes(self, alpha: float = 0.05) -> bool:
        return self.pvalue < alpha


def deflated_sharpe(
    returns: pd.Series | np.ndarray,
    n_trials: int,
    *,
    annualization: int = 252,
    benchmark_sharpe: float | None = None,
    trial_variance: float | None = None,
) -> DeflatedSharpeResult:
    """Bailey & López de Prado's deflated Sharpe ratio.

    Args:
        returns: Strategy net returns, single period (e.g. daily).
        n_trials: How many strategies were tested to find this one. Must be
            >= 1. Counting honestly matters — every variant, every parameter
            sweep, every "let me just try one more thing".
        annualization: Periods per year. Used to express the result in annual terms.
        benchmark_sharpe: Annualized Sharpe to test against. Defaults to 0 (i.e.
            test that the strategy beats cash).
        trial_variance: Variance of per-period Sharpe ratios across the tested
            universe. If None, defaults to 1/(n-1) — the variance of a Sharpe
            estimator under the null of no edge on a sample of length n. This
            is the López de Prado recommendation when you lack a richer
            characterization of the trial distribution.

    Returns:
        DeflatedSharpeResult including the deflated SR and a p-value. P-values
        below `alpha` indicate the observed SR is unlikely to be noise.
    """
    series = pd.Series(returns).dropna()
    n = len(series)
    if n < 30:
        raise ValueError("Need at least 30 observations to deflate a Sharpe.")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1.")

    sharpe_per_period = _per_period_sharpe(series)
    annualized_sr = sharpe_per_period * math.sqrt(annualization)

    benchmark_per_period = (
        (benchmark_sharpe / math.sqrt(annualization)) if benchmark_sharpe is not None else 0.0
    )

    skew = float(stats.skew(series, bias=False)) if n > 3 else 0.0
    kurt = float(stats.kurtosis(series, fisher=True, bias=False)) if n > 4 else 0.0

    if trial_variance is None:
        trial_variance = 1.0 / max(n - 1, 1)
    expected_max = _expected_max_sharpe(n_trials, trial_variance)

    # PSR conditional on the maximum-Sharpe expectation
    numerator = (sharpe_per_period - benchmark_per_period - expected_max) * math.sqrt(n - 1)
    denominator = math.sqrt(
        max(1.0 - skew * sharpe_per_period + ((kurt) / 4.0) * sharpe_per_period**2, 1e-12)
    )
    z = numerator / denominator
    pvalue = float(1.0 - stats.norm.cdf(z))

    return DeflatedSharpeResult(
        sharpe_ratio=annualized_sr,
        expected_max_sharpe=expected_max * math.sqrt(annualization),
        deflated_sharpe_ratio=(sharpe_per_period - expected_max) * math.sqrt(annualization),
        pvalue=pvalue,
        n_trials=n_trials,
        n_observations=n,
    )


def deflated_sharpe_pvalue(returns: pd.Series, n_trials: int, **kwargs: float) -> float:
    """Convenience accessor: returns just the p-value."""
    return deflated_sharpe(returns, n_trials, **kwargs).pvalue


def probabilistic_sharpe(
    returns: pd.Series,
    *,
    benchmark_sharpe: float = 0.0,
    annualization: int = 252,
) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > benchmark | observed SR).

    The single-trial cousin of deflated SR. Used inside DSR but exposed here
    because it shows up in the literature on its own.
    """
    series = pd.Series(returns).dropna()
    n = len(series)
    if n < 30:
        raise ValueError("Need at least 30 observations.")
    sharpe_per_period = _per_period_sharpe(series)
    benchmark_per_period = benchmark_sharpe / math.sqrt(annualization)
    skew = float(stats.skew(series, bias=False)) if n > 3 else 0.0
    kurt = float(stats.kurtosis(series, fisher=True, bias=False)) if n > 4 else 0.0
    numerator = (sharpe_per_period - benchmark_per_period) * math.sqrt(n - 1)
    denominator = math.sqrt(max(1 - skew * sharpe_per_period + (kurt / 4.0) * sharpe_per_period**2, 1e-12))
    return float(stats.norm.cdf(numerator / denominator))


def _per_period_sharpe(series: pd.Series) -> float:
    std = series.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(series.mean() / std)


def estimate_trial_variance(
    trial_returns: Iterable[pd.Series],
    *,
    annualization: int | None = None,
) -> float:
    """Empirical variance of per-period Sharpe ratios across a trial universe.

    Use this when you have a real sample of strategies tested (e.g. the per-trial
    returns stored in `ResearchMemory.history()`). It produces a tighter, more
    honest variance than the H₀ default `1/(n-1)`, because real trial families
    have correlated returns and the variance of their Sharpes is usually
    smaller than the IID null assumes.

    Args:
        trial_returns: iterable of return Series, one per trial. NaNs dropped.
        annualization: if provided, the returned variance is in annualized
            Sharpe units; otherwise per-period units (the convention DSR uses
            internally).

    Returns:
        Sample variance of the per-period (or annualized) Sharpe ratios.
        Raises ValueError if fewer than 2 trials are supplied.
    """
    sharpes_per_period: list[float] = []
    for returns in trial_returns:
        series = pd.Series(returns).dropna()
        if len(series) < 30:
            continue
        sharpes_per_period.append(_per_period_sharpe(series))
    if len(sharpes_per_period) < 2:
        raise ValueError("Need at least 2 trials with >= 30 observations each.")
    variance = float(np.var(sharpes_per_period, ddof=1))
    if annualization is not None:
        variance *= annualization
    return variance


def _expected_max_sharpe(n_trials: int, trial_variance: float) -> float:
    """Expected maximum of N independent standard normals scaled by sqrt(variance).

    Uses the Bailey & López de Prado closed-form approximation:
        E[max] ≈ sqrt(variance) * ( (1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1 - 1/(N·e)) )
    where γ is the Euler-Mascheroni constant.
    """
    if n_trials == 1:
        return 0.0
    inv_phi_a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    inv_phi_b = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(trial_variance) * (
        (1.0 - EULER_MASCHERONI) * inv_phi_a + EULER_MASCHERONI * inv_phi_b
    )
