"""CLI: `python -m ai_quant_lab.meme <command>`.

    collect   register new Solana pools and snapshot tracked ones (loop with --every)
    backfill  pull 5-minute candles for collected pools
    info      what is in the database
    demo      tear sheets on synthetic tapes, no API key or data needed
    research  Claude research loop on a panel (database or synthetic)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ai_quant_lab.meme.amm import ExecutionCosts
from ai_quant_lab.meme.backtest import MemeBacktestConfig, meme_backtest_fn
from ai_quant_lab.meme.panel import bars_per_year
from ai_quant_lab.meme.simulate import UniverseSpec, simulate_universe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ai_quant_lab.meme", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect")
    p.add_argument("--db", type=Path, default=Path("meme.db"))
    p.add_argument("--pages", type=int, default=3)
    p.add_argument("--every", type=float, default=0.0, help="Seconds between passes; 0 = once.")

    p = sub.add_parser("backfill")
    p.add_argument("--db", type=Path, default=Path("meme.db"))

    p = sub.add_parser("info")
    p.add_argument("--db", type=Path, default=Path("meme.db"))

    p = sub.add_parser("demo")
    p.add_argument("--days", type=int, default=21)
    p.add_argument("--seed", type=int, default=4)

    p = sub.add_parser("research")
    p.add_argument("--db", type=Path, default=None, help="Collected database; synthetic if absent.")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--target", type=int, default=2)
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--memory-db", type=Path, default=Path("meme_memory.db"))

    args = parser.parse_args(argv)
    return {
        "collect": _collect,
        "backfill": _backfill,
        "info": _info,
        "demo": _demo,
        "research": _research,
    }[args.command](args)


def _collect(args: argparse.Namespace) -> int:
    from ai_quant_lab.meme.geckoterminal import GeckoClient, collect_once, connect

    conn, client = connect(args.db), GeckoClient()
    while True:
        stats = collect_once(conn, client, pages=args.pages)
        print(f"[collect] +{stats.new_pools} pools, {stats.snapshots} snapshots", flush=True)
        if args.every <= 0:
            return 0
        time.sleep(args.every)


def _backfill(args: argparse.Namespace) -> int:
    from ai_quant_lab.meme.geckoterminal import GeckoClient, backfill_ohlcv, connect

    rows = backfill_ohlcv(connect(args.db), GeckoClient())
    print(f"[backfill] {rows} candles")
    return 0


def _info(args: argparse.Namespace) -> int:
    from ai_quant_lab.meme.geckoterminal import connect, load_panel

    conn = connect(args.db)
    for table in ("pools", "snapshots", "ohlcv"):
        print(f"{table:10s} {conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]}")
    for dex, n in conn.execute("SELECT dex, COUNT(*) FROM pools GROUP BY dex ORDER BY 2 DESC"):
        print(f"  {dex}: {n}")
    try:
        panel = load_panel(conn)
    except ValueError as exc:
        print(exc)
        return 0
    print(f"panel {panel.close.shape[0]} bars × {panel.close.shape[1]} tokens, "
          f"{panel.close.index[0]} → {panel.close.index[-1]}")
    print(panel.survivorship_report())
    return 0


def _demo(args: argparse.Namespace) -> int:
    from ai_quant_lab.meme.demo import run_demo

    run_demo(n_days=args.days, seed=args.seed)
    return 0


def _research(args: argparse.Namespace) -> int:
    from ai_quant_lab.agents.memory import ResearchMemory
    from ai_quant_lab.orchestrator.loop import LoopConfig, run_research_loop

    if args.db is not None:
        from ai_quant_lab.meme.geckoterminal import connect, load_panel

        panel = load_panel(connect(args.db))
        source = f"Solana pools collected point-in-time from GeckoTerminal ({args.db})"
    else:
        panel = simulate_universe(UniverseSpec(n_days=30, edge=1.0))
        source = "synthetic 30-day memecoin universe"
    fields = ", ".join(panel.as_strategy_input())
    config = LoopConfig(
        market_description=(
            f"{source}: {panel.close.shape[1]} tokens, 5-minute bars, fields: {fields}. "
            f"Long-only, ${args.capital:,.0f} book, AMM execution costs, rugs written off."
        ),
        market_type="memecoin",
        iterations=args.iterations,
        target_survivors=args.target,
        annualization=bars_per_year(panel.close.index),
        backtest_fn=meme_backtest_fn(
            panel, MemeBacktestConfig(capital_usd=args.capital, costs=ExecutionCosts())
        ),
    )
    with ResearchMemory(args.memory_db) as memory:
        artifacts, survivors = run_research_loop(panel.as_strategy_input(), config, memory=memory)
    print(f"\n[research] {len(artifacts)} iterations, {len(survivors)} survivors")
    for trial in survivors:
        print(f"   {trial.hypothesis_id}: SR={trial.metrics.get('sharpe_ratio', 0):+.2f}  "
              f"{trial.hypothesis_text[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
