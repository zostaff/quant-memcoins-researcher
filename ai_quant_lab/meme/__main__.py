"""CLI: `python -m ai_quant_lab.meme <command>`.

Networks (GeckoTerminal ids): solana, robinhood (Robinhood Chain).

    collect   register new pools and snapshot tracked ones (loop with --every)
    backfill  pull 5-minute candles for collected pools
    run       collector for launchd: collect + refresh the stalest candles, forever
    info      what is in the database
    demo      tear sheets on synthetic tapes, no API key or data needed
    zoo       run the strategy library through every gate (database or synthetic)
    research  Claude research loop on a panel (database or synthetic)
    terminal  four-window live terminal with library signals and paper trading
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ai_quant_lab.meme.amm import costs_for_network
from ai_quant_lab.meme.backtest import MemeBacktestConfig, meme_backtest_fn
from ai_quant_lab.meme.panel import bars_per_year
from ai_quant_lab.meme.simulate import UniverseSpec, simulate_universe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ai_quant_lab.meme", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect")
    p.add_argument("--db", type=Path, default=Path("meme.db"))
    p.add_argument("--network", default="solana", help="solana or robinhood")
    p.add_argument("--pages", type=int, default=3)
    p.add_argument("--every", type=float, default=0.0, help="Seconds between passes; 0 = once.")
    p.add_argument("--holder-calls", type=int, default=10,
                   help="Token info calls per pass (holders, top-10, mint/freeze).")

    p = sub.add_parser("backfill")
    p.add_argument("--db", type=Path, default=Path("meme.db"))

    p = sub.add_parser("run")
    p.add_argument("--db", type=Path, default=Path("meme.db"))
    p.add_argument("--networks", default="solana,robinhood", help="Comma-separated GeckoTerminal ids.")
    p.add_argument("--pages", type=int, default=3)
    p.add_argument("--every", type=float, default=120.0, help="Target seconds per pass.")
    p.add_argument("--holder-calls", type=int, default=10)
    p.add_argument("--backfill-per-pass", type=int, default=15)
    p.add_argument("--stale-hours", type=float, default=6.0)

    p = sub.add_parser("info")
    p.add_argument("--db", type=Path, default=Path("meme.db"))
    p.add_argument("--network", default=None, help="Panel for one network; default all.")

    p = sub.add_parser("demo")
    p.add_argument("--days", type=int, default=21)
    p.add_argument("--seed", type=int, default=4)

    p = sub.add_parser("zoo")
    p.add_argument("--db", type=Path, default=None, help="Collected database; synthetic if absent.")
    p.add_argument("--network", default="solana", help="Which network's pools (with --db).")
    p.add_argument("--tape", choices=("edge", "null"), default="edge",
                   help="Synthetic tape: narratives that pay (edge) or that are only visible (null).")
    p.add_argument("--days", type=int, default=42, help="Synthetic tape length.")
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--capital", type=float, default=2_000.0)
    p.add_argument("--strategies", default=None, help="Comma-separated subset; default all.")
    p.add_argument("--out", type=Path, default=None, help="Write a markdown report here.")
    p.add_argument("--json", type=Path, default=None, help="Write verdicts as JSON (for the terminal).")
    p.add_argument("--list", action="store_true", help="List the library and exit.")

    p = sub.add_parser("terminal")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--book", type=Path, default=Path("paper.db"), help="Paper trading database.")
    p.add_argument("--account", default="default")
    p.add_argument("--verdicts", type=Path, default=None, help="Zoo verdicts from `zoo --json`.")
    p.add_argument("--open", action="store_true", help="Open the browser.")

    p = sub.add_parser("research")
    p.add_argument("--db", type=Path, default=None, help="Collected database; synthetic if absent.")
    p.add_argument("--network", default="solana", help="Which network's pools to research.")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--target", type=int, default=2)
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--memory-db", type=Path, default=Path("meme_memory.db"))

    args = parser.parse_args(argv)
    return {
        "collect": _collect,
        "backfill": _backfill,
        "run": _run,
        "info": _info,
        "demo": _demo,
        "zoo": _zoo,
        "research": _research,
        "terminal": _terminal,
    }[args.command](args)


def _collect(args: argparse.Namespace) -> int:
    from ai_quant_lab.meme.geckoterminal import GeckoClient, collect_once, connect

    conn, client = connect(args.db), GeckoClient()
    while True:
        stats = collect_once(conn, client, network=args.network, pages=args.pages,
                             holder_calls=args.holder_calls)
        print(f"[collect] {time.strftime('%H:%M:%S')} {args.network}: +{stats.new_pools} pools, "
              f"{stats.snapshots} snapshots, {stats.token_snapshots} token snapshots", flush=True)
        if args.every <= 0:
            return 0
        time.sleep(args.every)


def _backfill(args: argparse.Namespace) -> int:
    from ai_quant_lab.meme.geckoterminal import GeckoClient, backfill_ohlcv, connect

    rows = backfill_ohlcv(connect(args.db), GeckoClient())
    print(f"[backfill] {rows} candles")
    return 0


def _run(args: argparse.Namespace) -> int:
    from ai_quant_lab.meme.geckoterminal import (
        GeckoClient,
        backfill_ohlcv,
        collect_once,
        connect,
        stale_pools,
    )

    conn, client = connect(args.db), GeckoClient()
    networks = [n.strip() for n in args.networks.split(",") if n.strip()]
    while True:
        started = time.monotonic()
        for network in networks:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                stats = collect_once(conn, client, network=network, pages=args.pages,
                                     holder_calls=args.holder_calls)
                pools = stale_pools(conn, limit=args.backfill_per_pass, stale_hours=args.stale_hours,
                                    network=network)
                candles = backfill_ohlcv(conn, client, addresses=pools)
                print(f"[run] {stamp} {network}: +{stats.new_pools} pools, "
                      f"{stats.snapshots} snapshots, {stats.token_snapshots} token snapshots, "
                      f"{candles} candles from {len(pools)} pools", flush=True)
            except Exception as exc:  # noqa: BLE001 — a collector must outlive API hiccups
                print(f"[run] {stamp} {network}: pass failed: {exc!r}", flush=True)
        time.sleep(max(args.every - (time.monotonic() - started), 5.0))


def _info(args: argparse.Namespace) -> int:
    from ai_quant_lab.meme.geckoterminal import connect, load_panel

    conn = connect(args.db)
    for table in ("pools", "snapshots", "token_snapshots", "token_meta", "trending", "ohlcv"):
        print(f"{table:10s} {conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]}")
    for network, dex, n in conn.execute(
        "SELECT network, dex, COUNT(*) FROM pools GROUP BY network, dex ORDER BY 1, 3 DESC"
    ):
        print(f"  {network:10s} {dex}: {n}")
    try:
        panel = load_panel(conn, network=args.network)
    except ValueError as exc:
        print(exc)
        return 0
    print(f"panel {panel.close.shape[0]} bars × {panel.close.shape[1]} tokens, "
          f"{panel.close.index[0]} → {panel.close.index[-1]}")
    print(panel.survivorship_report())
    fields = ["holders"] if panel.holders is not None else []
    print("token fields:", ", ".join(fields + list(panel.extra)) or "none yet")
    from ai_quant_lab.meme.narratives import narrative_summary

    print("narratives (from names and symbols):")
    print(narrative_summary(panel).to_string(formatters={"volume_share": "{:.1%}".format}))
    return 0


def _zoo(args: argparse.Namespace) -> int:
    from ai_quant_lab.meme.strategies import STRATEGIES
    from ai_quant_lab.meme.zoo import run_zoo

    if args.list:
        for s in STRATEGIES.values():
            needs = f"  needs: {', '.join(s.requires)}" if s.requires else ""
            print(f"{s.name:20} {s.family:10} {len(s.variants()):2} variants  {s.thesis}{needs}")
        return 0
    if args.db is not None:
        from ai_quant_lab.meme.geckoterminal import connect, load_panel

        panel = load_panel(connect(args.db), network=args.network)
        costs = costs_for_network(args.network)
        source = f"{args.network} pools from {args.db}"
    else:
        strength = 1.0 if args.tape == "edge" else 0.0
        panel = simulate_universe(UniverseSpec(n_days=args.days, seed=args.seed, edge=0.5 * strength,
                                               narrative_attention=1.0, narrative_strength=strength))
        costs = costs_for_network("solana")
        source = f"synthetic {args.tape} tape, {args.days} days, seed {args.seed}"
    names = [n.strip() for n in args.strategies.split(",")] if args.strategies else None
    print(f"[zoo] {source}: {panel.close.shape[1]} tokens, {panel.close.shape[0]} bars")
    result = run_zoo(panel, strategies=names, config=MemeBacktestConfig(capital_usd=args.capital, costs=costs),
                     source=source, progress=True)
    print()
    print(result.summary())
    if args.out is not None:
        args.out.write_text(result.to_markdown(f"Strategy zoo: {source}"))
        print(f"\nreport written to {args.out}")
    if args.json is not None:
        import json

        args.json.write_text(json.dumps({
            "source": source,
            "trials": result.n_trials,
            "verdicts": {e.strategy: {"verdict": e.verdict, "reason": e.reason} for e in result.entries},
        }, indent=2) + "\n")
        print(f"verdicts written to {args.json}")
    return 0


def _terminal(args: argparse.Namespace) -> int:
    from ai_quant_lab.meme.terminal import PaperBook, Terminal, load_verdicts, serve

    terminal = Terminal(book=PaperBook(args.book, account=args.account), verdicts=load_verdicts(args.verdicts))
    server = serve(terminal, args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"[terminal] {url}  paper book {args.book} ({args.account}), "
          f"verdicts: {args.verdicts or 'none loaded (signals show untested)'}", flush=True)
    if args.open:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
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

        panel = load_panel(connect(args.db), network=args.network)
        chain = {"solana": "Solana", "robinhood": "Robinhood Chain"}.get(args.network, args.network)
        source = f"{chain} pools collected point-in-time from GeckoTerminal ({args.db})"
    else:
        panel = simulate_universe(UniverseSpec(n_days=30, edge=1.0))
        source = "synthetic 30-day memecoin universe"
    from ai_quant_lab.meme.enrich import research_panel

    panel = research_panel(panel)
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
            panel, MemeBacktestConfig(capital_usd=args.capital, costs=costs_for_network(args.network))
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
