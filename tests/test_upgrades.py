"""Tests for the post-skeleton upgrades.

Covers:
    - Structural leakage detector (catches centered/forward; clears clean).
    - DSR trial-variance estimator.
    - RealisticEventDriven engine (slippage, partial fills).
    - Per-market critic prompts.
    - Memory's accepted_returns reconciliation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_quant_lab.agents.critic import MarketType, build_system_prompt
from ai_quant_lab.agents.memory import ResearchMemory, TrialRecord
from ai_quant_lab.backtest import BacktestConfig, RealisticEventDriven
from ai_quant_lab.features import detect_leakage_structural, momentum
from ai_quant_lab.validation import deflated_sharpe, estimate_trial_variance


def test_structural_detector_passes_clean_feature(gbm_price_series):
    def clean(df):
        return momentum(df["close"], 21)

    report = detect_leakage_structural(clean, gbm_price_series.to_frame(), name="clean")
    assert not report.has_leakage, report.format_problems()


def test_structural_detector_catches_centered_window(gbm_price_series):
    def centered(df):
        return df["close"].rolling(21, center=True, min_periods=1).mean().rename("centered")

    report = detect_leakage_structural(centered, gbm_price_series.to_frame(), name="centered")
    assert report.has_leakage
    assert "centered" in report.problems[0]


def test_structural_detector_catches_forward_shift(gbm_price_series):
    def peek(df):
        return df["close"].shift(-3).rename("peek")

    report = detect_leakage_structural(peek, gbm_price_series.to_frame(), name="peek")
    assert report.has_leakage


def test_estimate_trial_variance_matches_naive_var():
    rng = np.random.default_rng(0)
    series_list = [pd.Series(rng.normal(0, 0.01, 500)) for _ in range(20)]
    estimated = estimate_trial_variance(series_list)
    expected = np.var(
        [s.mean() / s.std(ddof=1) for s in series_list],
        ddof=1,
    )
    assert estimated == pytest.approx(expected, rel=1e-6)


def test_estimate_trial_variance_used_by_dsr_changes_pvalue():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.001, 0.01, 1000))
    cheap = deflated_sharpe(returns, n_trials=100, trial_variance=0.0001)
    expensive = deflated_sharpe(returns, n_trials=100, trial_variance=0.01)
    # Higher trial variance → more deflation → higher p-value.
    assert expensive.pvalue > cheap.pvalue


def test_realistic_event_driven_charges_more_than_basic():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0, 0.01, 500))
    positions = pd.Series(np.where(returns.shift(1) > 0, 1.0, -1.0), index=returns.index)
    cfg = BacktestConfig(cost_bps=2.0)
    realistic = RealisticEventDriven(config=cfg, slippage_coefficient=20.0).run(positions, returns)
    # Realistic must show some non-zero costs; sum of net returns < gross.
    assert realistic["returns"].sum() < (positions.shift(1).fillna(0) * returns).sum()


def test_realistic_event_driven_partial_fill():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0, 0.01, 100))
    # Hold target=1.0 from bar 10 onward; engine accumulates over many bars.
    positions = pd.Series(0.0, index=returns.index)
    positions.iloc[10:] = 1.0
    cfg = BacktestConfig(cost_bps=1.0, execution_lag=1)
    engine = RealisticEventDriven(config=cfg, max_participation_per_bar=0.1)
    out = engine.run(positions, returns, participation_capacity=1.0)
    # First bar after the request can only fill 10% of capacity.
    assert out["unfilled"].iloc[11] > 0
    # By bar 25, the strategy has had ~14 bars to accumulate at 10% each,
    # so the position is at the target ceiling.
    assert out["positions"].iloc[25] >= 0.9


@pytest.mark.parametrize("market_type", ["equities", "crypto", "futures", "options", "fx"])
def test_per_market_critic_prompts_differ(market_type):
    generic = build_system_prompt("generic")
    specific = build_system_prompt(market_type)
    assert generic != specific
    # Each market template references something the generic doesn't.
    markers = {
        "equities": "Tesla",
        "crypto": "Funding",
        "futures": "Roll-yield",
        "options": "Pin risk",
        "fx": "Carry",
    }
    assert markers[market_type] in specific


def test_memory_accepted_returns_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(4)
    series = pd.Series(
        rng.normal(0, 0.01, 100),
        index=pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=100),
    )
    import json as json_mod
    payload = json_mod.dumps(
        {"index": [str(t) for t in series.index], "values": series.tolist()}
    )

    db = tmp_path / "m.db"
    with ResearchMemory(db) as memory:
        memory.record(
            TrialRecord(
                hypothesis_id="r1", hypothesis_text="t", rationale="",
                code="", metrics={"sharpe_ratio": 1.0}, accepted=True,
                n_trials_at_time=0, iteration=0, returns_json=payload,
            )
        )
        recovered = memory.accepted_returns()
    assert len(recovered) == 1
    assert recovered[0].name == "r1"
    assert len(recovered[0]) == 100


def test_memory_migration_adds_returns_column(tmp_path: Path):
    """Open an existing DB twice: second open must not blow up on the migration."""
    db = tmp_path / "old.db"
    with ResearchMemory(db) as memory:
        memory.record(
            TrialRecord(
                hypothesis_id="x", hypothesis_text="t", rationale="",
                code="", metrics={}, accepted=False,
                n_trials_at_time=0, iteration=0,
            )
        )
    with ResearchMemory(db) as memory:
        assert memory.n_trials() == 1
