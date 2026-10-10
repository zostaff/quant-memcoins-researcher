"""Memecoin module: AMM costs, survivorship, execution rules, factors, data, loop."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest

from ai_quant_lab.agents.code import CodeArtifact
from ai_quant_lab.agents.critic import CriticVerdict, build_system_prompt
from ai_quant_lab.agents.hypothesis import StrategyHypothesis
from ai_quant_lab.agents.memory import ResearchMemory
from ai_quant_lab.meme import (
    ExecutionCosts,
    MemeBacktestConfig,
    TokenPanel,
    UniverseSpec,
    capacity_curve,
    estimate_capacity,
    max_size_for_cost,
    meme_backtest,
    meme_backtest_fn,
    meme_factor_returns,
    round_trip_cost_fraction,
    simulate_universe,
    tear_sheet,
)
from ai_quant_lab.meme import features as F
from ai_quant_lab.meme.amm import swap_loss_fraction
from ai_quant_lab.meme.geckoterminal import _SCHEMA, load_panel
from ai_quant_lab.orchestrator.loop import LoopConfig, run_research_loop
from ai_quant_lab.orchestrator.sandbox import SandboxError, run_strategy

FREE = ExecutionCosts(swap_fee=0.0, priority_fee_usd=0.0, mev_haircut=0.0, min_trade_usd=0.0,
                      min_liquidity_usd=0.0)


def _panel(close: list[list[float]], liquidity: float | list[list[float]] = 1e9) -> TokenPanel:
    index = pd.date_range("2026-01-01", periods=len(close), freq="5min")
    cols = pd.Index([f"T{i}" for i in range(len(close[0]))])
    close_df = pd.DataFrame(close, index=index, columns=cols, dtype=float)
    if np.isscalar(liquidity):
        liq_df = pd.DataFrame(liquidity, index=index, columns=cols).where(close_df.notna(), 0.0)
    else:
        liq_df = pd.DataFrame(liquidity, index=index, columns=cols, dtype=float)
    return TokenPanel(close=close_df, liquidity=liq_df, volume=liq_df * 0.0)


@pytest.fixture(scope="module")
def small_universe() -> TokenPanel:
    return simulate_universe(UniverseSpec(n_days=3, seed=11, edge=1.0))


# ---------------------------------------------------------------- AMM


def test_swap_loss_is_fee_for_tiny_trades_and_grows_with_size():
    tiny = float(swap_loss_fraction(1e-6, 100_000, 0.01))
    assert tiny == pytest.approx(0.01, abs=1e-6)
    # 10% of the quote reserve (TVL/2) at 1% fee: 1 - 0.99 / 1.1 = 10%
    assert float(swap_loss_fraction(5_000, 100_000, 0.01)) == pytest.approx(0.1, abs=1e-9)
    assert float(swap_loss_fraction(50_000, 100_000, 0.01)) > 0.5


def test_no_pool_means_total_loss():
    assert float(swap_loss_fraction(100, 0.0, 0.01)) == 1.0


def test_max_size_for_cost_respects_the_budget():
    costs = ExecutionCosts()
    size = max_size_for_cost(50_000, 0.05, costs)
    assert 0 < size < 50_000
    assert round_trip_cost_fraction(size, 50_000, costs) <= 0.05 + 1e-9
    assert round_trip_cost_fraction(size * 1.1, 50_000, costs) > 0.05


# ---------------------------------------------------------------- backtest


def test_rug_is_written_off_not_dropped():
    panel = _panel([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [np.nan, 1.0], [np.nan, 1.0]])
    weights = pd.DataFrame(0.0, index=panel.close.index, columns=panel.tokens)
    weights["T0"] = 0.5
    config = MemeBacktestConfig(capital_usd=1_000, costs=FREE, max_weight=1.0,
                                max_pool_participation=1.0)
    result = meme_backtest(weights, panel, config)
    assert result.metrics["rug_loss_usd"] == pytest.approx(500.0)
    assert result.equity.iloc[-1] == pytest.approx(500.0)


def test_execution_lag_prevents_trading_on_the_next_bar():
    # T0 alternates up/down. An "oracle" that buys before each up bar must not
    # profit: with a 1-bar lag it is filled one bar late, right before the drop.
    prices = [1.0]
    for i in range(40):
        prices.append(prices[-1] * (1.5 if i % 2 == 0 else 1 / 1.5))
    panel = _panel([[p] for p in prices])
    up_next = panel.close["T0"].shift(-1) > panel.close["T0"]
    weights = up_next.astype(float).to_frame("T0")
    config = MemeBacktestConfig(capital_usd=1_000, costs=FREE, max_weight=1.0,
                                max_pool_participation=1.0, rebalance_band=0.0)
    result = meme_backtest(weights, panel, config)
    assert result.equity.iloc[-1] < 1_000


def test_position_capped_by_pool_participation():
    panel = _panel([[1.0]] * 10, liquidity=10_000)
    weights = pd.DataFrame(1.0, index=panel.close.index, columns=panel.tokens)
    config = MemeBacktestConfig(capital_usd=100_000, costs=FREE, max_weight=1.0,
                                max_pool_participation=0.02)
    result = meme_backtest(weights, panel, config)
    assert result.holdings["T0"].max() <= 0.02 * 10_000 + 1e-6
    assert result.metrics["capped_fraction"] > 0.9


def test_no_entry_below_liquidity_floor():
    panel = _panel([[1.0]] * 10, liquidity=1_000)
    weights = pd.DataFrame(0.5, index=panel.close.index, columns=panel.tokens)
    costs = replace(FREE, min_liquidity_usd=5_000)
    result = meme_backtest(weights, panel, MemeBacktestConfig(capital_usd=100, costs=costs))
    assert result.holdings.to_numpy().max() == 0.0


def test_no_trade_band_avoids_churn():
    rng = np.random.default_rng(0)
    prices = np.exp(np.cumsum(rng.normal(0, 0.02, 200)))
    panel = _panel([[p] for p in prices])
    weights = pd.DataFrame(0.1, index=panel.close.index, columns=panel.tokens)
    banded = meme_backtest(weights, panel, MemeBacktestConfig(rebalance_band=0.25, max_weight=1.0))
    churny = meme_backtest(weights, panel, MemeBacktestConfig(rebalance_band=0.0, max_weight=1.0))
    assert banded.costs.to_numpy().sum() < churny.costs.to_numpy().sum() / 5


def test_long_only_and_cash_never_negative(small_universe):
    weights = pd.DataFrame(
        np.random.default_rng(1).normal(0, 1, small_universe.close.shape),
        index=small_universe.close.index,
        columns=small_universe.tokens,
    )
    result = meme_backtest(weights, small_universe)
    invested = result.holdings.sum(axis=1)
    assert (result.holdings.to_numpy() >= -1e-9).all()
    assert (invested <= result.equity + 1e-6).all()


def test_capacity_return_decays_with_size(small_universe):
    mask = F.tradeable(small_universe.close, small_universe.liquidity)
    weights = F.top_n_equal_weight(F.trailing_return(small_universe.close, 12), 5, mask)
    weights = weights.iloc[::36].reindex(weights.index).ffill()
    curve = capacity_curve(weights, small_universe, capitals=(1_000, 1_000_000))
    assert curve.loc[1_000_000, "capped_fraction"] > curve.loc[1_000, "capped_fraction"]
    assert estimate_capacity(curve) in (0.0, 1_000.0, 1_000_000.0)


# ---------------------------------------------------------------- universe & factors


def test_simulation_is_deterministic_and_has_deaths():
    a = simulate_universe(UniverseSpec(n_days=1, seed=5))
    b = simulate_universe(UniverseSpec(n_days=1, seed=5))
    pd.testing.assert_frame_equal(a.close, b.close)
    assert a.survivorship_report()["n_dead"] > 0


def test_returns_with_death_marks_rugs():
    panel = _panel([[1.0], [1.0], [np.nan]])
    assert panel.returns_with_death()["T0"].iloc[-1] == -1.0


def test_factor_returns_are_finite(small_universe):
    factors = meme_factor_returns(small_universe)
    assert list(factors.columns) == ["MKT", "SIZE", "MOM", "AGE", "FLOW"]
    assert np.isfinite(factors.to_numpy()).all()
    assert factors["MKT"].min() > -1.0 - 1e-9


def test_tear_sheet_runs(small_universe):
    mask = F.tradeable(small_universe.close, small_universe.liquidity)
    weights = F.top_n_equal_weight(F.holder_growth(small_universe.holders, 36), 5, mask)
    sheet = tear_sheet(weights, small_universe, n_trials=5, capitals=(1_000, 10_000))
    text = sheet.summary()
    assert "deflated sharpe" in text and "capacity" in text


# ---------------------------------------------------------------- data loader


def test_load_panel_point_in_time_and_death():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    conn.execute("INSERT INTO pools VALUES ('A', 'A/SOL', 'a', 'sol', 'pump-fun', "
                 "'2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')")
    t0 = int(pd.Timestamp("2026-01-01").timestamp())
    for i in range(30):
        if i < 10:  # trades for 10 bars then goes silent
            conn.execute("INSERT INTO ohlcv VALUES ('A', ?, 1, 1, 1, ?, 100)", (t0 + 300 * i, 1.0 + i))
    conn.execute("INSERT INTO ohlcv VALUES ('A', ?, 1, 1, 1, 1, 100)", (t0 + 300 * 29,))
    # first reserve observation at bar 4, price 5
    conn.execute("INSERT INTO snapshots VALUES ('A', '2026-01-01T00:20:00+00:00', 5.0, 20000, "
                 "NULL, NULL, NULL, NULL)")
    panel = load_panel(conn, dead_after_bars=6)
    liq = panel.liquidity["A"]
    assert liq.iloc[:4].eq(0).all()                      # unknown before first snapshot
    assert liq.iloc[4] == pytest.approx(20_000)
    assert liq.iloc[8] == pytest.approx(20_000 * np.sqrt(9 / 5))  # sqrt(price) rule
    assert panel.close["A"].iloc[16:].isna().all()       # dead for good, despite late trade


# ---------------------------------------------------------------- sandbox & loop


_GOOD = """
from ai_quant_lab.meme import features
def strategy(panel):
    mask = features.tradeable(panel['close'], panel['liquidity'])
    w = features.top_n_equal_weight(features.trailing_return(panel['close'], 12), 5, mask)
    return w.iloc[::36].reindex(w.index).ffill()
"""


def test_sandbox_accepts_panel_strategy(small_universe):
    result = run_strategy(_GOOD, small_universe.as_strategy_input())
    assert result.positions.shape == small_universe.close.shape


def test_sandbox_rejects_wrong_shape(small_universe):
    bad = "def strategy(panel):\n    return panel['close'].iloc[:, :3] * 0"
    with pytest.raises(SandboxError):
        run_strategy(bad, small_universe.as_strategy_input())


def test_memecoin_critic_template_exists():
    assert "Survivorship" in build_system_prompt("memecoin")


@dataclass
class _Hypo:
    def propose(self, *args, **kwargs):
        return StrategyHypothesis("mom12", "12-bar momentum, top 5", "attention", {}, (0, 1), "mania", "bleed")


@dataclass
class _Critic:
    def review(self, hypothesis):
        return CriticVerdict(passes=True, reasoning="ok", kill_reasons=[])


@dataclass
class _Code:
    def render(self, hypothesis):
        return CodeArtifact(source=_GOOD)


def test_research_loop_runs_on_panel(tmp_path, small_universe):
    config = LoopConfig(
        market_description="test",
        market_type="memecoin",
        iterations=2,
        target_survivors=5,
        annualization=105_120,
        backtest_fn=meme_backtest_fn(small_universe),
    )
    with ResearchMemory(tmp_path / "m.db") as memory:
        artifacts, _ = run_research_loop(
            small_universe.as_strategy_input(), config, memory=memory,
            hypothesis_agent=_Hypo(), code_agent=_Code(), critic_agent=_Critic(), log=lambda _: None,
        )
        assert len(artifacts) == 2
        assert memory.n_trials() == 2
        assert all(a.rejection_reason != "sandbox_error" for a in artifacts)


def test_sandbox_import_whitelist_is_exact(small_universe):
    data = small_universe.as_strategy_input()
    ok = "from ai_quant_lab.features.library import momentum\n" + _GOOD
    assert run_strategy(ok, data).positions.shape == small_universe.close.shape
    for bad in ("import ai_quant_lab.config", "from ai_quant_lab import config", "import os"):
        with pytest.raises(SandboxError):
            run_strategy(bad + "\n" + _GOOD, data)
