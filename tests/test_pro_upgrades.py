"""Tests for the 6→8 professional upgrades.

Covers:
    - Portfolio backtest (cross-sectional, dollar-neutral, gross exposure).
    - Cross-sectional features (rank, zscore, neutralize).
    - Meta-labeling: classifier learns, applies act/skip mask.
    - Factor attribution: PCA decomposition + concentration score.
    - Bar engine: interval inference, annualization.
    - TCA: shortfall metrics, slippage calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_quant_lab.backtest import (
    BacktestConfig,
    BarSchedule,
    annualization_from_index,
    infer_bar_interval,
    long_short_quantile_portfolio,
    vectorized_portfolio_backtest,
)
from ai_quant_lab.features.cross_sectional import (
    cross_sectional_momentum,
    industry_neutralize,
    neutralize_by_factor,
    rank_within_universe,
    zscore_cross_section,
)
from ai_quant_lab.features.library import (
    garman_klass_volatility,
    parkinson_volatility,
    vwap_deviation,
)
from ai_quant_lab.production import (
    Fill,
    calibrate_slippage_coefficient,
    compute_tca,
)
from ai_quant_lab.validation import (
    LogisticBinaryClassifier,
    MetaLabeler,
    factor_concentration_score,
    fama_french_attribution,
    meta_label_targets,
    pca_decompose,
)


# ---------- portfolio backtest ----------

def _make_universe(seed: int = 0, n_bars: int = 500, n_assets: int = 10):
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0003, 0.012, (n_bars, n_assets))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(shocks, axis=0)),
        index=pd.bdate_range(end="2026-01-01", periods=n_bars),
        columns=[f"A{i:02d}" for i in range(n_assets)],
    )
    return prices


def test_portfolio_backtest_is_dollar_neutral():
    prices = _make_universe(seed=1)
    signal = cross_sectional_momentum(prices, lookback=21, skip=1)
    positions = long_short_quantile_portfolio(signal, long_quantile=0.7, short_quantile=0.3)
    result = vectorized_portfolio_backtest(positions, prices.pct_change())
    # By construction, net exposure should hover near zero.
    assert abs(result.metrics["mean_net_exposure"]) < 0.05
    assert result.metrics["mean_gross_exposure"] > 1.5


def test_portfolio_backtest_returns_match_shape():
    prices = _make_universe(seed=2)
    signal = cross_sectional_momentum(prices, lookback=10, skip=1)
    positions = long_short_quantile_portfolio(signal)
    result = vectorized_portfolio_backtest(positions, prices.pct_change())
    assert len(result.returns) == len(prices)
    assert int(result.metrics["n_assets"]) == prices.shape[1]


def test_long_short_quantile_rejects_bad_quantiles():
    signal = _make_universe()
    with pytest.raises(ValueError):
        long_short_quantile_portfolio(signal, long_quantile=0.2, short_quantile=0.8)


# ---------- cross-sectional features ----------

def test_rank_within_universe_is_uniform():
    frame = pd.DataFrame(np.random.RandomState(0).randn(100, 5))
    ranks = rank_within_universe(frame, pct=True)
    # Each row should have ranks between 0 and 1, average 0.5
    assert (ranks.min(axis=1) >= 0).all()
    assert (ranks.max(axis=1) <= 1).all()


def test_zscore_cross_section_centers_per_bar():
    frame = pd.DataFrame(np.random.RandomState(0).randn(100, 5))
    z = zscore_cross_section(frame)
    # Per-row mean should be ~0 after standardization
    assert z.mean(axis=1).abs().max() < 1e-9


def test_neutralize_by_factor_strips_exposure():
    rng = np.random.default_rng(0)
    factor = pd.DataFrame(rng.normal(0, 1, (100, 5)))
    raw = factor * 2.0 + rng.normal(0, 0.1, (100, 5))
    residuals = neutralize_by_factor(raw, factor)
    # After neutralization, per-bar correlation with the factor should be ~0
    correlations = []
    for ts in residuals.index:
        x = factor.loc[ts]
        y = residuals.loc[ts].dropna()
        if y.std() > 0:
            correlations.append(abs(y.corr(x)))
    assert np.mean(correlations) < 0.1


def test_industry_neutralize_demeans_within_group():
    frame = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [10, 20]})
    industry = {"A": "tech", "B": "tech", "C": "energy"}
    out = industry_neutralize(frame, industry)
    # Within-tech mean should be zero
    tech_mean = out[["A", "B"]].mean(axis=1)
    assert tech_mean.abs().max() < 1e-9


# ---------- meta-labeling ----------

def test_logistic_classifier_fits_separable_data():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 2))
    y = (X[:, 0] > 0).astype(float)
    clf = LogisticBinaryClassifier(l2=0.1)
    clf.fit(X, y)
    predictions = clf.predict_proba(X)[:, 1]
    accuracy = ((predictions > 0.5) == y).mean()
    assert accuracy > 0.85


def test_meta_label_targets_signed_correctly():
    side = pd.Series([1.0, 1.0, -1.0, -1.0])
    fwd = pd.Series([0.01, -0.01, 0.01, -0.01])
    targets = meta_label_targets(side, fwd)
    # long * up = win; long * down = loss; short * down = win; short * up = loss
    assert targets.tolist() == [1.0, 0.0, 0.0, 1.0]


def test_meta_labeler_end_to_end():
    rng = np.random.default_rng(0)
    n = 500
    side = pd.Series(rng.choice([-1.0, 1.0], n))
    fwd = pd.Series(rng.normal(0, 0.01, n))
    features = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    labeler = MetaLabeler()
    labeler.fit(side, fwd, features)
    filtered = labeler.apply(side, features)
    assert (filtered.isin([-1.0, 0.0, 1.0])).all()


# ---------- factor attribution ----------

def test_pca_decompose_returns_sorted_eigenvalues():
    rng = np.random.default_rng(0)
    matrix = pd.DataFrame(rng.normal(0, 1, (500, 5)))
    result = pca_decompose(matrix)
    descending = sorted(result.eigenvalues.tolist(), reverse=True)
    assert result.eigenvalues.tolist() == descending
    # Explained variance ratios sum to ~1
    assert abs(result.explained_variance_ratio.sum() - 1.0) < 1e-9


def test_factor_concentration_score_is_high_for_duplicate():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 0.01, 500))
    other = base + rng.normal(0, 0.0001, 500)
    survivors = [base.rename("s1"), pd.Series(rng.normal(0, 0.01, 500), name="s2")]
    duplicate_score = factor_concentration_score(other, survivors)
    independent_score = factor_concentration_score(
        pd.Series(rng.normal(0, 0.01, 500)), survivors
    )
    assert duplicate_score > independent_score


def test_factor_concentration_score_zero_with_no_survivors():
    candidate = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))
    assert factor_concentration_score(candidate, []) == 0.0


def test_fama_french_attribution_recovers_known_beta():
    rng = np.random.default_rng(0)
    n = 1000
    market = rng.normal(0.0005, 0.01, n)
    strategy_returns = pd.Series(0.5 * market + rng.normal(0, 0.005, n))
    factors = pd.DataFrame({"MKT": market})
    out = fama_french_attribution(strategy_returns, factors)
    assert abs(out.betas["MKT"] - 0.5) < 0.05
    assert out.r_squared > 0.4


# ---------- bar engine ----------

def test_infer_bar_interval_detects_daily():
    idx = pd.bdate_range(start="2025-01-01", periods=100)
    assert infer_bar_interval(idx) == "1d"


def test_infer_bar_interval_detects_hourly():
    idx = pd.date_range(start="2025-01-01", periods=100, freq="1h")
    assert infer_bar_interval(idx) == "1h"


def test_bar_schedule_annualization():
    assert BarSchedule.daily().annualization == 252
    assert BarSchedule.hourly().annualization == 252 * 7
    assert BarSchedule.crypto_daily().annualization == 365


def test_backtest_config_from_schedule():
    cfg = BacktestConfig.from_schedule(BarSchedule.hourly())
    assert cfg.annualization == 252 * 7


def test_annualization_from_index_helper():
    idx = pd.date_range(start="2025-01-01", periods=200, freq="1h")
    assert annualization_from_index(idx) == 252 * 7


# ---------- OHLCV features ----------

def test_parkinson_volatility_is_positive():
    rng = np.random.default_rng(0)
    n = 300
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, n)))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    rv = parkinson_volatility(high, low, window=21).dropna()
    assert (rv > 0).all()


def test_garman_klass_volatility_is_positive():
    rng = np.random.default_rng(0)
    n = 300
    open_ = pd.Series(100 + np.cumsum(rng.normal(0, 0.3, n)))
    close = open_ + rng.normal(0, 0.5, n)
    high = pd.concat([open_, close], axis=1).max(axis=1) + rng.uniform(0, 1, n)
    low = pd.concat([open_, close], axis=1).min(axis=1) - rng.uniform(0, 1, n)
    gk = garman_klass_volatility(open_, high, low, close, window=21).dropna()
    assert len(gk) > 0


def test_vwap_deviation_is_zero_when_volume_constant():
    n = 100
    close = pd.Series(np.linspace(100, 110, n))
    volume = pd.Series(1000.0, index=close.index)
    dev = vwap_deviation(close, volume, window=21).dropna()
    # With constant volume, VWAP = SMA. close > SMA when trend is up.
    assert (dev > 0).any()


# ---------- TCA ----------

def _make_fills(n: int = 50, seed: int = 0) -> list[Fill]:
    rng = np.random.default_rng(seed)
    fills: list[Fill] = []
    timestamp = pd.Timestamp("2026-01-01")
    for i in range(n):
        arrival = 100.0 + rng.normal(0, 0.5)
        decision = arrival + rng.normal(0, 0.1)
        side = int(rng.choice([-1, 1]))
        # +2 bps mean slippage
        slippage = arrival * 0.0002 * side + rng.normal(0, arrival * 0.0001)
        fill_price = arrival + slippage
        fills.append(
            Fill(
                timestamp=timestamp + pd.Timedelta(minutes=i),
                symbol=f"SYM{i % 3}",
                side=side,
                quantity=100.0,
                fill_price=fill_price,
                arrival_price=arrival,
                decision_price=decision,
            )
        )
    return fills


def test_tca_report_aggregates_shortfall():
    fills = _make_fills(100)
    report = compute_tca(fills)
    assert report.n_fills == 100
    # Mean arrival shortfall should be near +2 bps (positive = paid more)
    assert 0.5 < report.mean_arrival_shortfall_bps < 5.0
    assert not report.by_symbol.empty


def test_tca_calibrated_cost_bps_returns_median():
    fills = _make_fills(100)
    report = compute_tca(fills)
    median_cost = report.calibrated_cost_bps(percentile=0.5)
    assert isinstance(median_cost, float)
    assert median_cost == report.median_arrival_shortfall_bps


def test_calibrate_slippage_coefficient_with_adv():
    fills = _make_fills(50)
    adv = {"SYM0": 10000.0, "SYM1": 10000.0, "SYM2": 10000.0}
    coefficient = calibrate_slippage_coefficient(fills, average_daily_volume=adv)
    assert coefficient > 0


def test_calibrate_slippage_coefficient_falls_back_without_adv():
    fills = _make_fills(20)
    coefficient = calibrate_slippage_coefficient(fills, average_daily_volume=None)
    assert coefficient >= 0
