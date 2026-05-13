"""Diagnostics that distinguish "robust edge" from "lucky fold".

degradation_ratio: how much performance drops from in-sample to out-of-sample.
fold_stability: how consistent the out-of-sample performance is across folds.
regime_breakdown: per-regime Sharpe, given a regime labeling.

A real strategy degrades by 20-40% IS→OOS and produces fold Sharpes whose
sign is mostly consistent. A fake strategy either matches IS perfectly (signal
of leakage) or has folds that flip sign (signal of overfit to noise).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ai_quant_lab.backtest.engine import performance_metrics


def degradation_ratio(in_sample_sharpe: float, out_of_sample_sharpe: float) -> float:
    """OOS Sharpe / IS Sharpe.

    Interpretation:
        > 1.0  : OOS beat IS. Suspicious unless the sample is short.
        0.5–1  : Healthy. Most edge survives, with some shrinkage.
        0–0.5  : Marginal. Likely overfitting.
        < 0    : Sign flip. The "edge" was noise.
    """
    if in_sample_sharpe == 0:
        return 0.0
    return float(out_of_sample_sharpe / in_sample_sharpe)


def fold_stability(fold_sharpes: np.ndarray | list[float]) -> dict[str, float]:
    """Stability of Sharpe across folds.

    Returns:
        Dict with mean, std, fraction_positive, and a t-statistic of the mean.
        A good strategy has fraction_positive >= 0.7 and a meaningful t-stat.
    """
    arr = np.asarray(fold_sharpes, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "fraction_positive": 0.0, "t_stat": 0.0, "n": 0.0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    fraction_positive = float((arr > 0).mean())
    t_stat = float(mean / (std / math.sqrt(arr.size))) if std > 0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "fraction_positive": fraction_positive,
        "t_stat": t_stat,
        "n": float(arr.size),
    }


def regime_breakdown(
    returns: pd.Series,
    regime: pd.Series,
    *,
    annualization: int = 252,
) -> pd.DataFrame:
    """Performance metrics, grouped by regime.

    `regime` should be a Series with categorical/integer labels aligned to
    `returns`. Common regime sources: realized vol buckets, trend/range
    classifications, or risk-on/risk-off indicators.

    Returns a DataFrame indexed by regime label with Sharpe / hit-rate / count.
    """
    returns, regime = returns.align(regime, join="inner")
    rows: dict[object, dict[str, float]] = {}
    for label, group in returns.groupby(regime):
        metrics = performance_metrics(group, annualization=annualization)
        rows[label] = {
            "n_observations": metrics["n_observations"],
            "ann_return": metrics["annualized_return"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "hit_rate": metrics["hit_rate"],
            "max_drawdown": metrics["max_drawdown"],
        }
    return pd.DataFrame(rows).T
