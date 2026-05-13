"""Cross-sectional feature primitives.

A cross-sectional feature looks across the universe at a single time:
who's cheapest, who has the most momentum, who's the most volatile.

Every function obeys the convention: at time t, only data at times <= t is
used. The cross-section over the universe at time t is computed from
already-lagged inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_within_universe(frame: pd.DataFrame, *, pct: bool = True) -> pd.DataFrame:
    """Per-bar cross-sectional rank.

    Args:
        frame: DataFrame indexed by time, columns by asset id.
        pct: if True, returns percentile ranks in [0, 1]; if False, integer ranks.

    Returns:
        DataFrame with the same shape, NaN preserved where the input is NaN.
    """
    return frame.rank(axis=1, pct=pct)


def zscore_cross_section(frame: pd.DataFrame, *, robust: bool = False) -> pd.DataFrame:
    """Per-bar z-score across the universe.

    Args:
        frame: DataFrame indexed by time, columns by asset id.
        robust: if True, use median + MAD instead of mean + std. Less sensitive
            to outliers — useful when one asset blows up.

    Returns:
        DataFrame of same shape with each row standardized.
    """
    if robust:
        center = frame.median(axis=1)
        mad = (frame.sub(center, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        # 1.4826 makes MAD a consistent estimator of sigma under normality.
        scale = 1.4826 * mad
    else:
        center = frame.mean(axis=1)
        scale = frame.std(axis=1, ddof=1).replace(0.0, np.nan)
    return frame.sub(center, axis=0).div(scale, axis=0)


def neutralize_by_factor(
    frame: pd.DataFrame,
    factor: pd.DataFrame,
    *,
    fit_window: int = 252,
) -> pd.DataFrame:
    """Cross-sectionally orthogonalize `frame` against `factor`.

    For each bar t, regresses the cross-section of `frame[t, :]` on
    `factor[t, :]` and returns the residuals. Used to strip a known exposure
    (e.g. beta, size, sector mean) out of a raw signal.

    Args:
        frame: signal to neutralize (time × asset).
        factor: factor exposure to project out (time × asset). Same shape.
        fit_window: ignored at the moment; the regression is per-bar cross-sectional.
            The argument exists so callers can later swap in a rolling-window
            implementation without breaking the API.

    Returns:
        DataFrame of residuals, same shape as `frame`.
    """
    _ = fit_window  # currently per-bar; placeholder for future rolling fit
    frame, factor = frame.align(factor, join="inner")
    residuals = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    for timestamp, row in frame.iterrows():
        y = row.dropna()
        x = factor.loc[timestamp].reindex(y.index).dropna()
        common = y.index.intersection(x.index)
        if len(common) < 3:
            continue
        y_aligned = y.loc[common].to_numpy()
        x_aligned = x.loc[common].to_numpy()
        # Add intercept
        design = np.column_stack([np.ones_like(x_aligned), x_aligned])
        try:
            coeffs, *_ = np.linalg.lstsq(design, y_aligned, rcond=None)
        except np.linalg.LinAlgError:
            continue
        fitted = design @ coeffs
        residuals.loc[timestamp, common] = y_aligned - fitted
    return residuals


def industry_neutralize(frame: pd.DataFrame, industry: pd.Series | dict[str, str]) -> pd.DataFrame:
    """Demean a cross-sectional signal within each industry/group.

    Args:
        frame: signal to neutralize (time × asset).
        industry: mapping asset_id -> industry label. Either a Series indexed
            by asset id or a plain dict.

    Returns:
        DataFrame where each row sums to zero within each industry — the
        common industry tilt is removed.
    """
    if isinstance(industry, dict):
        industry = pd.Series(industry)
    industry = industry.reindex(frame.columns)
    if industry.isna().any():
        missing = industry[industry.isna()].index.tolist()
        raise ValueError(f"Missing industry labels for: {missing[:5]}")

    out = frame.copy()
    for label in industry.dropna().unique():
        cols = industry[industry == label].index
        group_mean = frame[cols].mean(axis=1)
        out[cols] = frame[cols].sub(group_mean, axis=0)
    return out


def cross_sectional_momentum(
    price_data: pd.DataFrame,
    lookback: int = 21,
    skip: int = 1,
) -> pd.DataFrame:
    """Per-asset momentum (% return over lookback bars), lagged by `skip`.

    Skip-month momentum (the Jegadeesh-Titman variant) avoids the short-term
    reversal that dominates the most-recent week.

    Args:
        price_data: DataFrame of prices (time × asset).
        lookback: number of bars in the momentum window.
        skip: bars to skip immediately before t. 1 = simple t-lookback to t-1.
            5 is the classic "skip-one-week" form.
    """
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if skip < 1:
        raise ValueError("skip must be >= 1")
    skipped = price_data.shift(skip)
    return skipped / skipped.shift(lookback) - 1.0
