"""Example 06 — full research loop with Claude (requires ANTHROPIC_API_KEY).

End-to-end demo: Claude proposes a hypothesis, critic reviews it, code agent
renders it, the gates evaluate it, memory stores the trial. Set
AI_QUANT_LAB_MAX_LLM_CALLS small (e.g. 6) for a quick feel.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import os
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_lab.agents.memory import ResearchMemory
from ai_quant_lab.backtest import BacktestConfig
from ai_quant_lab.orchestrator.loop import LoopConfig, run_research_loop


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY first. See .env.example.")

    rng = np.random.default_rng(seed=6)
    n = 2520
    price_data = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))),
        index=pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n),
        name="close",
    )

    loop_config = LoopConfig(
        market_description="Daily bars on a single liquid US large-cap equity, 10 years.",
        iterations=10,
        target_survivors=2,
        max_llm_calls=30,
        backtest_config=BacktestConfig(cost_bps=8.0),
    )

    db_path = Path("./memory_example06.db")
    with ResearchMemory(db_path) as memory:
        artifacts, survivors = run_research_loop(price_data, loop_config, memory=memory)

    print()
    print(f"Iterations run: {len(artifacts)}")
    print(f"Survivors:      {len(survivors)}")
    for trial in survivors:
        print(f"  {trial.hypothesis_id}: SR={trial.metrics.get('sharpe_ratio', 0):+.2f}  {trial.hypothesis_text}")


if __name__ == "__main__":
    main()
