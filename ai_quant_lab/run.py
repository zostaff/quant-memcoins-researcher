"""CLI entry point: `python -m ai_quant_lab.run`.

Wires the research loop to a price series and writes results to memory.db.
Defaults to a synthetic GBM tape so the CLI works without you needing data;
pass --csv to use your own.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_lab.agents.memory import ResearchMemory
from ai_quant_lab.backtest.engine import BacktestConfig
from ai_quant_lab.config import settings
from ai_quant_lab.orchestrator.loop import LoopConfig, run_research_loop


def main(argv: list[str] | None = None) -> int:
    parser = _argparser()
    args = parser.parse_args(argv)

    price_data = _load_prices(args)
    backtest_config = BacktestConfig(
        cost_bps=args.cost_bps,
        annualization=args.annualization,
    )
    loop_config = LoopConfig(
        market_description=args.market_description,
        market_type=args.market_type,
        iterations=args.iterations,
        target_survivors=args.target,
        max_llm_calls=args.max_llm_calls,
        backtest_config=backtest_config,
        annualization=args.annualization,
    )

    db_path = Path(args.memory_db)
    print(f"[run] memory db: {db_path.resolve()}")
    print(f"[run] series: {len(price_data)} bars from {price_data.index[0]} to {price_data.index[-1]}")
    print(f"[run] target survivors: {args.target}  max iterations: {args.iterations}")

    with ResearchMemory(db_path) as memory:
        artifacts, survivors = run_research_loop(price_data, loop_config, memory=memory)

    print()
    print(f"[run] artifacts: {len(artifacts)} iterations, {sum(a.accepted for a in artifacts)} accepted")
    print(f"[run] survivors after loop: {len(survivors)}")
    for trial in survivors:
        print(
            f"   {trial.hypothesis_id}: SR={trial.metrics.get('sharpe_ratio', 0):+.2f}  "
            f"{trial.hypothesis_text[:80]}"
        )
    return 0


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-quant-lab", description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV with at least a 'close' column and a sortable date index. "
        "If absent, synthetic GBM data is used.",
    )
    parser.add_argument(
        "--date-column",
        default="date",
        help="Date column name when reading --csv. Default: 'date'.",
    )
    parser.add_argument(
        "--close-column",
        default="close",
        help="Close-price column name when reading --csv. Default: 'close'.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Maximum loop iterations.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=settings.target_survivors,
        help="Stop early when this many accepted strategies exist.",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=settings.max_llm_calls,
        help="Hard cap on Claude API calls.",
    )
    parser.add_argument(
        "--market-description",
        default="Daily bars on a single liquid US equity index futures contract.",
        help="Plain-English description fed to the hypothesis agent.",
    )
    parser.add_argument(
        "--market-type",
        default="generic",
        choices=["generic", "equities", "crypto", "futures", "options", "fx"],
        help="Selects the critic agent's failure-mode template.",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=settings.cost_bps,
        help="Round-turn transaction cost.",
    )
    parser.add_argument(
        "--annualization",
        type=int,
        default=settings.annualization,
        help="Periods per year (252 daily equities, 365 crypto).",
    )
    parser.add_argument(
        "--memory-db",
        type=Path,
        default=settings.memory_db,
        help="SQLite path. Append-only; existing trials are kept.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for synthetic data when --csv is absent.",
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=2520,
        help="Length of synthetic series (default ~10 years of daily bars).",
    )
    return parser


def _load_prices(args: argparse.Namespace) -> pd.Series:
    if args.csv is not None:
        if not args.csv.exists():
            raise SystemExit(f"--csv not found: {args.csv}")
        frame = pd.read_csv(args.csv)
        if args.date_column not in frame.columns:
            raise SystemExit(f"missing date column '{args.date_column}' in {args.csv}")
        if args.close_column not in frame.columns:
            raise SystemExit(f"missing close column '{args.close_column}' in {args.csv}")
        frame[args.date_column] = pd.to_datetime(frame[args.date_column])
        frame = frame.sort_values(args.date_column).set_index(args.date_column)
        return frame[args.close_column].rename("close")

    rng = np.random.default_rng(args.seed)
    daily_drift = 0.05 / args.annualization
    daily_vol = 0.16 / np.sqrt(args.annualization)
    shocks = rng.normal(loc=daily_drift, scale=daily_vol, size=args.n_bars)
    prices = 100.0 * np.exp(np.cumsum(shocks))
    index = pd.bdate_range(end=pd.Timestamp.utcnow().normalize(), periods=args.n_bars)
    return pd.Series(prices, index=index, name="close")


if __name__ == "__main__":
    sys.exit(main())
