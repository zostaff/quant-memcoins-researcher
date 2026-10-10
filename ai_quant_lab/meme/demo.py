"""Memecoin research done like a fund: sweep, pick the best, then judge it.

We sweep 20 variants of a holder-growth strategy twice: on a synthetic tape
with NO edge, and on one with a planted edge. On the null tape fees, curve
slippage and the post-launch bleed sink every variant. On the edge tape we
take the best variant and judge it the way an allocator would: net of
execution, against the meme factors, at several book sizes, and with a
Sharpe deflated by all 20 tries.
"""

from __future__ import annotations

import itertools

from ai_quant_lab.meme import features as F
from ai_quant_lab.meme.backtest import MemeBacktestConfig, meme_backtest
from ai_quant_lab.meme.simulate import UniverseSpec, simulate_universe
from ai_quant_lab.meme.tearsheet import tear_sheet

LOOKBACKS = (36, 72, 144, 288)
TOP_N = (4, 8, 12, 16, 24)


def weights_for(panel, lookback: int, top_n: int):
    mask = F.tradeable(panel.close, panel.liquidity, 10_000) & (F.age_bars(panel.close) >= 72)
    weights = F.top_n_equal_weight(F.holder_growth(panel.holders, lookback), top_n, mask)
    return weights.iloc[::72].reindex(weights.index).ffill()


def sweep(panel):
    config = MemeBacktestConfig(capital_usd=2_000)
    results = []
    for lookback, top_n in itertools.product(LOOKBACKS, TOP_N):
        metrics = meme_backtest(weights_for(panel, lookback, top_n), panel, config).metrics
        results.append((metrics["sharpe_ratio"], metrics["total_return"], lookback, top_n))
    return sorted(results, reverse=True)


def run_demo(n_days: int = 21, seed: int = 4) -> None:
    tapes = {
        "NULL tape (no edge)": UniverseSpec(n_days=n_days, seed=seed, edge=0.0),
        "tape with a planted edge": UniverseSpec(n_days=n_days, seed=seed, edge=1.0),
    }
    for name, spec in tapes.items():
        panel = simulate_universe(spec)
        ranked = sweep(panel)
        n_trials = len(ranked)
        best_sharpe, best_return, lookback, top_n = ranked[0]
        positive = sum(1 for r in ranked if r[0] > 0)

        print(f"\n=== {name} ===")
        print(f"{n_trials} variants tried, {positive} with positive Sharpe")
        print(f"best: holder growth over {lookback} bars, top {top_n} -> "
              f"Sharpe {best_sharpe:+.2f}, return {best_return:+.1%}")

        sheet = tear_sheet(
            weights_for(panel, lookback, top_n),
            panel,
            n_trials=n_trials,
            config=MemeBacktestConfig(capital_usd=2_000),
            capitals=(1_000, 2_000, 10_000, 50_000),
        )
        print(sheet.summary())
        accepted = sheet.dsr is not None and sheet.dsr.pvalue < 0.05
        print("verdict            " + ("ACCEPT" if accepted else "REJECT: the best of many tries"))
