"""Automated leakage detection.

Two complementary strategies:

1. **Structural (preferred when you have the feature function).** Recompute the
   feature on a truncated tape and compare. If the feature at time t changes
   when bars at times > t are removed, the feature uses future data. This is
   the cleanest possible test and has zero false positives.

2. **Correlation-based (fallback when you only have the values).** Compare the
   feature's correlation with past-shifted vs future-shifted targets. A clean
   feature should not correlate strongly with bars it shouldn't have seen.

The structural test catches centered windows, forward references, and any
exotic peek that hides inside an opaque feature function. The correlation
test is what `FeaturePipeline.fit_transform` runs by default because it
doesn't need the source — only the values and a target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class LeakageReport:
    """Result of a leakage audit."""

    problems: list[str] = field(default_factory=list)
    column_scores: dict[str, float] = field(default_factory=dict)

    @property
    def has_leakage(self) -> bool:
        return bool(self.problems)

    def format_problems(self) -> str:
        if not self.problems:
            return "(no problems detected)"
        return "\n".join(f"  - {p}" for p in self.problems)


def detect_leakage(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    future_horizon: int = 1,
    past_horizon: int = 1,
    threshold: float = 2.0,
    min_correlation: float = 0.05,
    suspicious_future_correlation: float = 0.20,
) -> LeakageReport:
    """Correlation-based leakage audit. Operates on values only.

    Args:
        features: feature matrix to audit.
        target: target series the strategy will eventually predict.
        future_horizon: bars to shift the target into the future when computing
            the "should-not-see" correlation.
        past_horizon: bars to shift the target into the past for the baseline.
        threshold: ratio at which `future_corr / past_corr` triggers a flag.
        min_correlation: features whose past correlation is below this in
            absolute value fall back to the absolute-future check.
        suspicious_future_correlation: features whose future correlation
            exceeds this in absolute value are flagged independently. Catches
            forward-reference leaks where past_corr is near zero.

    Returns:
        LeakageReport listing all suspect columns.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a DataFrame")
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a Series")

    target_past = target.shift(past_horizon)
    target_future = target.shift(-future_horizon)

    report = LeakageReport()
    for column in features.columns:
        series = features[column]
        if series.dropna().empty:
            continue
        past_corr = _safe_abs_corr(series, target_past)
        future_corr = _safe_abs_corr(series, target_future)
        ratio = future_corr / past_corr if past_corr >= min_correlation else float("inf")
        report.column_scores[column] = ratio if past_corr >= min_correlation else future_corr

        if future_corr >= suspicious_future_correlation:
            report.problems.append(
                f"{column}: future_corr={future_corr:.3f} >= {suspicious_future_correlation:.2f} "
                f"(direct forward reference?)"
            )
        elif past_corr >= min_correlation and ratio >= threshold:
            report.problems.append(
                f"{column}: future_corr={future_corr:.3f} > {threshold:.1f}x past_corr={past_corr:.3f}"
            )

    return report


def detect_leakage_structural(
    feature_fn: Callable[[pd.DataFrame], pd.Series],
    price_data: pd.DataFrame,
    *,
    name: str | None = None,
    probe_points: int = 20,
    truncate_horizon: int = 10,
    tolerance: float = 1e-9,
) -> LeakageReport:
    """Structural leakage test: does feature[t] change when bars after t disappear?

    This is the cleanest possible leakage test. We pick `probe_points` times t
    spread across the series. For each one:
        1. Compute the feature on the FULL series. Record value at t.
        2. Compute the feature on the series truncated to [0, t]. Record value at t.
        3. If the two values differ by more than `tolerance`, the feature
           sees data after t — it's leaky.

    There are zero false positives. A feature using only past data must by
    definition produce the same value at t regardless of what's at t+1, t+2, ...

    Args:
        feature_fn: callable `(DataFrame) -> Series` (same as a FeatureStep.func).
        price_data: DataFrame the feature consumes.
        name: feature name for the report. Defaults to feature_fn.__name__.
        probe_points: how many times t to test. More = slower but more thorough.
        truncate_horizon: minimum bars to leave after the probe point in the
            "untruncated" version. Defaults to 10. Set higher if your feature
            uses a longer forward window — though the whole point is that
            cleaner features don't.
        tolerance: numerical tolerance for the equality check.

    Returns:
        LeakageReport. `problems` is empty iff the feature is clean.
    """
    if probe_points < 1:
        raise ValueError("probe_points must be >= 1")
    if truncate_horizon < 1:
        raise ValueError("truncate_horizon must be >= 1")
    if len(price_data) < probe_points + truncate_horizon + 30:
        raise ValueError("price_data is too short for the requested probe configuration.")

    full_values = feature_fn(price_data)
    if not isinstance(full_values, pd.Series):
        raise TypeError("feature_fn must return a Series")

    label = name or getattr(feature_fn, "__name__", "feature")
    report = LeakageReport()

    valid_index = full_values.dropna().index
    if len(valid_index) < probe_points * 2:
        return report

    # Spread probe points across the second half of the valid range, leaving
    # `truncate_horizon` buffer so truncation is meaningful.
    lower = max(30, len(valid_index) // 3)
    upper = len(valid_index) - truncate_horizon
    probe_positions = np.linspace(lower, upper - 1, probe_points, dtype=int)

    max_diff = 0.0
    n_disagreements = 0
    for pos in probe_positions:
        timestamp = valid_index[pos]
        truncated_data = price_data.loc[:timestamp]
        truncated_values = feature_fn(truncated_data)
        full_at_t = full_values.loc[timestamp]
        trunc_at_t = truncated_values.iloc[-1] if len(truncated_values) else np.nan
        if pd.isna(full_at_t) and pd.isna(trunc_at_t):
            continue
        if pd.isna(full_at_t) or pd.isna(trunc_at_t):
            n_disagreements += 1
            continue
        diff = abs(float(full_at_t) - float(trunc_at_t))
        if diff > tolerance:
            n_disagreements += 1
            max_diff = max(max_diff, diff)

    report.column_scores[label] = float(n_disagreements) / probe_points
    if n_disagreements > 0:
        report.problems.append(
            f"{label}: {n_disagreements}/{probe_points} probe points disagree "
            f"between full and truncated computation (max diff {max_diff:.4g}). "
            f"The feature uses future bars."
        )
    return report


def _safe_abs_corr(left: pd.Series, right: pd.Series) -> float:
    aligned_left, aligned_right = left.align(right, join="inner")
    aligned_left = aligned_left.replace([np.inf, -np.inf], np.nan).dropna()
    aligned_right = aligned_right.reindex(aligned_left.index).replace([np.inf, -np.inf], np.nan).dropna()
    aligned_left = aligned_left.reindex(aligned_right.index)
    if len(aligned_left) < 30:
        return 0.0
    if aligned_left.std(ddof=1) == 0 or aligned_right.std(ddof=1) == 0:
        return 0.0
    corr = float(aligned_left.corr(aligned_right))
    return abs(corr) if not np.isnan(corr) else 0.0


def quick_audit(features: pd.DataFrame, target: pd.Series) -> str:
    """Convenience one-liner used in examples and CI."""
    report = detect_leakage(features, target)
    if not report.has_leakage:
        return f"OK — {len(features.columns)} features, no leakage detected."
    return f"LEAKAGE — {len(report.problems)} of {len(features.columns)} features suspect."
