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
from ai_quant_lab.meme.geckoterminal import _SCHEMA, collect_once, load_panel
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
    conn.execute("INSERT INTO pools (address, name, base_token, quote_token, dex, created_at, first_seen_at) VALUES ('A', 'A/SOL', 'a', 'sol', 'pump-fun', "
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
    assert panel.holders is None and not panel.extra     # no token snapshots collected


def test_token_fields_are_point_in_time():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    conn.execute("INSERT INTO pools (address, name, base_token, quote_token, dex, created_at, first_seen_at) VALUES ('A', 'A/SOL', 'tokA', 'sol', 'pump-fun', "
                 "'2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')")
    t0 = int(pd.Timestamp("2026-01-01").timestamp())
    for i in range(12):
        conn.execute("INSERT INTO ohlcv VALUES ('A', ?, 1, 1, 1, 1, 100)", (t0 + 300 * i,))
    conn.execute("INSERT INTO token_snapshots (token, observed_at, holders, top10_pct, mint_authority, freeze_authority, gt_score, graduation_pct, graduated) VALUES ('tokA', '2026-01-01T00:15:00+00:00', "
                 "120, 45.0, 0, 0, 30.0, 82.0, 0)")
    conn.execute("INSERT INTO token_snapshots (token, observed_at, holders, top10_pct, mint_authority, freeze_authority, gt_score, graduation_pct, graduated) VALUES ('tokA', '2026-01-01T00:40:00+00:00', "
                 "300, NULL, NULL, NULL, NULL, 100.0, 1)")
    panel = load_panel(conn)
    holders = panel.holders["A"]
    assert holders.iloc[:3].isna().all()                 # unknown before first reading
    assert holders.iloc[3] == 120 and holders.iloc[7] == 120
    assert holders.iloc[8] == 300
    assert panel.extra["top10_share"]["A"].iloc[10] == pytest.approx(0.45)  # carried, not erased
    assert panel.extra["graduation"]["A"].iloc[5] == pytest.approx(0.82)
    assert panel.extra["mint_risk"]["A"].iloc[5] == 0
    assert set(panel.as_strategy_input()) >= {"holders", "top10_share", "graduation", "mint_risk"}


def test_rug_screen_fails_unknown_and_risky_tokens():
    idx = pd.RangeIndex(1)
    mint = pd.DataFrame({"a": [0.0], "b": [1.0], "c": [0.0], "d": [np.nan]}, index=idx)
    top10 = pd.DataFrame({"a": [0.3], "b": [0.3], "c": [0.8], "d": [0.3]}, index=idx)
    ok = F.rug_screen(mint, top10).iloc[0]
    assert ok.tolist() == [True, False, False, False]
    assert F.rug_screen(mint, top10, unknown_ok=True).iloc[0]["d"]
    assert F.rug_screen() is None
    grad = pd.DataFrame({"a": [0.5], "b": [0.75], "c": [1.0]})
    assert F.graduation_window(grad).iloc[0].tolist() == [False, True, False]


class _FakeGecko:
    """Offline stand-in for GeckoClient: one new pool, token info and launchpad data."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, path, params=None):
        self.calls.append(path)
        if path.endswith("/new_pools"):
            if params["page"] > 1:
                return {"data": []}
            return {"data": [{
                "attributes": {"address": "P1", "name": "X/SOL", "pool_created_at": "2099-01-01T00:00:00Z",
                               "base_token_price_usd": "0.001", "reserve_in_usd": "9000"},
                "relationships": {"base_token": {"data": {"id": "solana_T1"}},
                                  "dex": {"data": {"id": "pump-fun"}}},
            }]}
        if "/tokens/multi/" in path:
            return {"data": [{"attributes": {"address": "T1", "launchpad_details": {
                "graduation_percentage": 64.0, "completed": False}}}]}
        if path.endswith("/T1/info"):
            return {"data": {"attributes": {"holders": {"count": 77, "distribution_percentage": {"top_10": "51.5"}},
                                            "mint_authority": "no", "freeze_authority": "yes"}}}
        return {"data": []}


def test_collect_once_snapshots_tokens_within_budget():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    client = _FakeGecko()
    stats = collect_once(conn, client, pages=2, holder_calls=1)
    assert stats.new_pools == 1 and stats.token_snapshots == 1
    row = conn.execute("SELECT holders, top10_pct, mint_authority, freeze_authority, graduation_pct, "
                       "graduated FROM token_snapshots").fetchone()
    assert row == (77, 51.5, 0, 1, 64.0, 0)
    assert sum(c.endswith("/info") for c in client.calls) == 1
    collect_once(conn, client, pages=1, holder_calls=0)
    assert sum(c.endswith("/info") for c in client.calls) == 1   # budget 0: no info calls


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


def test_new_models_get_no_temperature():
    from ai_quant_lab.agents.base import _sampling_params

    assert _sampling_params("claude-sonnet-4-6", 2048, 0.4) == {"max_tokens": 2048, "temperature": 0.4}
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-4-8"):
        params = _sampling_params(model, 2048, 0.4)
        assert "temperature" not in params and params["max_tokens"] >= 16_000


def test_stale_pools_picks_unbackfilled_then_stalest():
    from ai_quant_lab.meme.geckoterminal import stale_pools

    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    now = pd.Timestamp.now(tz="UTC")
    for name, age_h in (("new", 1), ("old", 200), ("fresh", 2), ("stale", 30)):
        launched = (now - pd.Timedelta(hours=age_h)).isoformat()
        conn.execute("INSERT INTO pools (address, name, base_token, quote_token, dex, created_at, first_seen_at) VALUES (?, ?, 't', 'sol', 'x', ?, ?)", (name, name, launched, launched))
    conn.execute("INSERT INTO ohlcv VALUES ('fresh', ?, 1, 1, 1, 1, 1)", (int(now.timestamp()) - 600,))
    conn.execute("INSERT INTO ohlcv VALUES ('stale', ?, 1, 1, 1, 1, 1)", (int(now.timestamp()) - 20 * 3600,))
    assert stale_pools(conn, limit=10) == ["new", "stale"]   # 'old' left the window, 'fresh' is fresh
    assert stale_pools(conn, limit=1) == ["new"]


def test_networks_are_kept_apart_and_fee_tiers_reach_the_backtest():
    from ai_quant_lab.meme import costs_for_network
    from ai_quant_lab.meme.geckoterminal import pool_fee_pct

    assert pool_fee_pct("ai / WETH 1%") == 1.0
    assert pool_fee_pct("Glorp / WETH 0.01%") == 0.01
    assert pool_fee_pct("PEPE / SOL") is None
    assert costs_for_network("robinhood").priority_fee_usd < costs_for_network("solana").priority_fee_usd

    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    for addr, net, fee in (("SolPool", "solana", None), ("0xabc", "robinhood", 0.3)):
        conn.execute(
            "INSERT INTO pools (address, name, base_token, quote_token, dex, created_at, first_seen_at, "
            "network, fee_pct) VALUES (?, ?, 't', 'q', 'd', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', ?, ?)",
            (addr, addr, net, fee),
        )
    t0 = int(pd.Timestamp("2026-01-01").timestamp())
    for addr in ("SolPool", "0xabc"):
        for i in range(6):
            conn.execute("INSERT INTO ohlcv VALUES (?, ?, 1, 1, 1, 1, 100)", (addr, t0 + 300 * i))
    rh = load_panel(conn, network="robinhood")
    assert list(rh.tokens) == ["0xabc"]
    assert rh.pool_fee["0xabc"] == pytest.approx(0.003)
    assert list(load_panel(conn, network="solana").tokens) == ["SolPool"]
    assert len(load_panel(conn).tokens) == 2
    with pytest.raises(ValueError, match="robinhood"):
        load_panel(_empty(), network="robinhood")


def _empty():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    return conn


def test_lower_pool_fee_means_lower_cost():
    panel = simulate_universe(UniverseSpec(n_days=2, seed=3))
    w = F.top_n_equal_weight(F.trailing_return(panel.close, 12), 5, F.tradeable(panel.close, panel.liquidity))
    w = w.iloc[::36].reindex(w.index).ffill()
    base = meme_backtest(w, panel).metrics["total_cost_usd"]
    cheap = replace(panel, pool_fee=pd.Series(0.0001, index=panel.tokens))
    assert meme_backtest(w, cheap).metrics["total_cost_usd"] < base


def test_old_database_gets_network_columns(tmp_path):
    from ai_quant_lab.meme.geckoterminal import connect

    db = tmp_path / "old.db"
    old = sqlite3.connect(db)
    old.execute("CREATE TABLE pools (address TEXT PRIMARY KEY, name TEXT, base_token TEXT, quote_token TEXT, "
                "dex TEXT, created_at TEXT, first_seen_at TEXT NOT NULL)")
    old.execute("INSERT INTO pools VALUES ('A', 'A/SOL', 'a', 'sol', 'pump-fun', NULL, '2026-01-01')")
    old.commit()
    old.close()
    conn = connect(db)
    assert conn.execute("SELECT network, fee_pct FROM pools").fetchone() == ("solana", None)
