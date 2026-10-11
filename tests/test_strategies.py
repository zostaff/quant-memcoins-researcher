"""Strategy library, narratives, market mood and the zoo."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from ai_quant_lab.meme import narratives as N
from ai_quant_lab.meme.backtest import MemeBacktestConfig
from ai_quant_lab.meme.enrich import research_panel
from ai_quant_lab.meme.geckoterminal import _SCHEMA, collect_once, connect, load_panel
from ai_quant_lab.meme.sentiment import load_external_sentiment, market_mood, regime, risk_appetite
from ai_quant_lab.meme.simulate import UniverseSpec, simulate_universe
from ai_quant_lab.meme.strategies import STRATEGIES, StrategyContext, catalog, check_weights
from ai_quant_lab.meme.zoo import VERDICTS, run_zoo


@pytest.fixture(scope="module")
def tape():
    return simulate_universe(UniverseSpec(n_days=4, seed=5, edge=0.5, narrative_attention=1.0,
                                          narrative_strength=1.0))


@pytest.fixture(scope="module")
def ctx(tape):
    return StrategyContext(tape)


# ---------------------------------------------------------------- narratives


@pytest.mark.parametrize(("name", "symbol", "expected"), [
    ("dogwifhat", "WIF", "dog"),
    (None, "AIDOGE", "dog"),            # the longer keyword wins
    ("Agent Smith", "SMITH", "ai"),
    ("Trump Coin", "MAGA", "politics"),
    ("PEPE ESCOBAR KILOS", "KILOS", "frog"),
    ("Tesla Moon", "TSLA", "stocks"),
    ("Brain", "BRAIN", "other"),        # 'ai' inside a word is not a match
    (None, None, "other"),
])
def test_classify(name, symbol, expected):
    assert N.classify(name, symbol) == expected


def test_labels_recover_synthetic_themes(tape):
    labels = N.narrative_labels(tape)
    assert (labels == tape.meta["theme"]).mean() > 0.95


def test_labels_without_meta_are_other(tape):
    bare = simulate_universe(UniverseSpec(n_days=1, seed=1))
    object.__setattr__(bare, "meta", None)
    assert set(N.narrative_labels(bare)) == {N.OTHER}


def test_narrative_shares_sum_to_one(tape):
    share = N.narrative_volume_share(tape, N.narrative_labels(tape), 36)
    total = share.sum(axis=1, min_count=1).dropna()
    assert np.allclose(total, 1.0)


# ---------------------------------------------------------------- point-in-time


def _truncation_points(panel):
    n = len(panel.close)
    return [panel.close.index[i] for i in (n // 3, 2 * n // 3)]


@pytest.mark.parametrize("make", [
    lambda p: N.narrative_heat(p, N.narrative_labels(p), fast=12, slow=144),
    lambda p: N.narrative_return(p, N.narrative_labels(p), 36),
    lambda p: N.launch_saturation(p, N.narrative_labels(p), 12, 144),
    lambda p: market_mood(p),
    lambda p: risk_appetite(p).to_frame(),
])
def test_features_do_not_look_ahead(tape, make):
    full = make(tape)
    for end in _truncation_points(tape):
        cut = make(tape.slice(end=end))
        pd.testing.assert_frame_equal(full.loc[:end, cut.columns], cut, check_freq=False)


@pytest.mark.parametrize("name", [n for n, s in STRATEGIES.items()
                                  if not set(s.requires) & {"graduation", "trending_rank"}])
def test_strategy_weights_do_not_look_ahead(tape, ctx, name):
    strategy = STRATEGIES[name]
    params = strategy.variants()[0]
    full = strategy.weights(tape, ctx, **params)
    check_weights(full, tape)
    for end in _truncation_points(tape):
        short = tape.slice(end=end)
        cut = strategy.weights(short, StrategyContext(short), **params)
        np.testing.assert_allclose(full.loc[:end].to_numpy(), cut.to_numpy(), atol=1e-12)


def test_every_library_strategy_produces_valid_weights(tape, ctx):
    found = catalog(ctx)
    assert {"curve_graduation", "trending_entry"} <= set(found.unavailable)
    for strategy in found.available:
        for params in strategy.variants()[:2]:
            check_weights(strategy.weights(tape, ctx, **params), tape)


def test_missing_data_is_refused(tape, ctx):
    with pytest.raises(ValueError, match="graduation"):
        STRATEGIES["curve_graduation"].weights(tape, ctx, low=0.7, top_n=8)


def test_context_must_match_panel(tape, ctx):
    other = simulate_universe(UniverseSpec(n_days=1, seed=2))
    with pytest.raises(ValueError, match="different panel"):
        STRATEGIES["flow_momentum"].weights(other, ctx, lookback=36, top_n=8)


# ---------------------------------------------------------------- simulator


def test_narrative_switches_never_move_prices():
    base = simulate_universe(UniverseSpec(n_days=2, seed=3))
    visible = simulate_universe(UniverseSpec(n_days=2, seed=3, narrative_attention=1.0))
    pd.testing.assert_frame_equal(base.close, visible.close)
    assert not base.volume.equals(visible.volume)       # attention shows up in volume only


# ---------------------------------------------------------------- mood


def test_regime_is_off_without_history(tape):
    on = regime(risk_appetite(tape))
    assert not on.iloc[:50].any()
    assert on.dtype == bool


def test_external_sentiment_respects_publication_lag():
    index = pd.date_range("2026-01-01 10:00", periods=6, freq="5min")
    feed = pd.DataFrame({"time": ["2026-01-01T10:00:00Z", "2026-01-01T10:10:00Z"],
                         "narrative": ["ai", "ai"], "score": [0.3, 0.9]})
    out = load_external_sentiment(feed, index, key="narrative", lag="5min", max_age="10min")
    assert np.isnan(out["ai"].iloc[0])                       # 10:00 score not known at 10:00
    assert out["ai"].iloc[1] == 0.3                          # usable from 10:05
    assert out["ai"].iloc[3] == 0.9                          # 10:10 score usable from 10:15
    assert out["ai"].iloc[5] == 0.9                          # still within max_age at 10:25
    market = load_external_sentiment(feed[["time", "score"]], index)
    assert list(market.columns) == ["market"]


def test_research_panel_adds_point_in_time_fields(tape):
    enriched = research_panel(tape)
    for name in ("narrative_heat", "narrative_return", "launch_saturation", "risk_appetite", "breadth"):
        assert name in enriched.extra
        assert enriched.extra[name].where(tape.close.isna()).isna().all().all()


# ---------------------------------------------------------------- collector


class _FakeGecko:
    def get(self, path, params=None):
        if path.endswith("/new_pools"):
            if params["page"] > 1:
                return {"data": []}
            return {"data": [{
                "attributes": {"address": "P1", "name": "WIF2 / SOL", "pool_created_at": "2099-01-01T00:00:00Z",
                               "base_token_price_usd": "0.001", "reserve_in_usd": "9000",
                               "transactions": {"m5": {"buys": 30, "sells": 10}}},
                "relationships": {"base_token": {"data": {"id": "solana_T1"}}, "dex": {"data": {"id": "pump-fun"}}},
            }]}
        if "/tokens/multi/" in path:
            return {"data": [{"attributes": {"address": "T1", "name": "dog wif hat 2", "symbol": "WIF2"}}]}
        if path.endswith("/T1/info"):
            return {"data": {"attributes": {"name": "dog wif hat 2", "symbol": "WIF2", "gt_category_ids": ["meme"],
                                            "developer_holding_percentage": "12.5", "is_honeypot": "no",
                                            "twitter_handle": "wif2"}}}
        if path.endswith("/trending_pools"):
            return {"data": [{"attributes": {"address": "ELSEWHERE"}}, {"attributes": {"address": "P1"}}]}
        return {"data": []}


def test_collector_records_meta_trending_and_flow():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    stats = collect_once(conn, _FakeGecko(), pages=1, holder_calls=1)
    assert stats.trending == 2
    assert conn.execute("SELECT name, symbol, categories, twitter FROM token_meta").fetchone() == (
        "dog wif hat 2", "WIF2", '["meme"]', "wif2")
    assert conn.execute("SELECT dev_pct, honeypot FROM token_snapshots").fetchone() == (12.5, 0)
    assert conn.execute("SELECT COUNT(*) FROM pools").fetchone()[0] == 1     # trending never adds pools


def test_old_database_gets_new_token_columns(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE token_snapshots (token TEXT, observed_at TEXT, holders INTEGER, top10_pct REAL, "
                "mint_authority INTEGER, freeze_authority INTEGER, gt_score REAL, graduation_pct REAL, "
                "graduated INTEGER, PRIMARY KEY (token, observed_at))")
    old.commit()
    old.close()
    columns = {row[1] for row in connect(path).execute("PRAGMA table_info(token_snapshots)")}
    assert {"dev_pct", "honeypot"} <= columns


def test_panel_gets_names_flow_and_trending():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    conn.execute("INSERT INTO pools (address, name, base_token, quote_token, dex, created_at, first_seen_at, "
                 "network) VALUES ('A', 'DOGE9 / SOL', 'tA', 'sol', 'pump-fun', '2026-01-01T00:00:00Z', "
                 "'2026-01-01T00:00:00Z', 'solana')")
    t0 = int(pd.Timestamp("2026-01-01").timestamp())
    for i in range(12):
        conn.execute("INSERT INTO ohlcv VALUES ('A', ?, 1, 1, 1, 1, 100)", (t0 + 300 * i,))
    conn.execute("INSERT INTO snapshots VALUES ('A', '2026-01-01T00:10:00+00:00', 1, 20000, NULL, NULL, 9, 1)")
    conn.execute("INSERT INTO trending VALUES ('solana', '2026-01-01T00:20:00+00:00', 'A', 3)")
    conn.execute("INSERT INTO trending VALUES ('solana', '2026-01-01T00:30:00+00:00', 'B', 1)")
    panel = load_panel(conn)
    assert panel.meta.loc["A", "symbol"] == "DOGE9"
    assert N.narrative_labels(panel)["A"] == "dog"
    ratio = panel.extra["buy_ratio"]["A"]
    assert ratio.iloc[:2].isna().all() and ratio.iloc[2] == pytest.approx(0.9)
    rank = panel.extra["trending_rank"]["A"]
    assert rank.iloc[:4].isna().all()          # no list observed yet
    assert rank.iloc[4] == 3 and rank.iloc[5] == 3
    assert rank.iloc[6:].isna().all()          # dropped off the 00:30 list


# ---------------------------------------------------------------- zoo


def test_zoo_counts_every_variant_and_kills_the_null_tape():
    panel = simulate_universe(UniverseSpec(n_days=4, seed=6, narrative_attention=1.0))
    names = ["narrative_rotation", "flow_momentum"]
    result = run_zoo(panel, strategies=names, config=MemeBacktestConfig(capital_usd=2_000))
    assert result.n_trials == sum(len(STRATEGIES[n].variants()) for n in names)
    assert all(e.sheet.dsr is None or e.sheet.dsr.n_trials == result.n_trials for e in result.entries)
    assert all(e.verdict in VERDICTS and e.verdict != "PASS" for e in result.entries)
    report = result.to_markdown()
    assert "| `narrative_rotation` |" in report and "Variants tried" in report
