"""Orchestrator: the research loop and its gates.

The loop:
    1. HypothesisAgent proposes a strategy.
    2. CriticAgent attacks the idea. Kill on fail.
    3. CodeAgent renders code. Sandbox runs it.
    4. Vectorized backtest measures Sharpe.
    5. Three gates: critic, deflated_sharpe, correlation. ALL must pass.
    6. Survivors written to ResearchMemory; n_trials counter increments.
    7. Repeat until target_survivors or max_llm_calls.

The deflated_sharpe gate is the only place an LLM cannot talk you out of.
"""

from ai_quant_lab.orchestrator.gates import GateOutcome, evaluate_gates
from ai_quant_lab.orchestrator.loop import LoopConfig, run_research_loop
from ai_quant_lab.orchestrator.sandbox import SandboxError, SandboxResult, run_strategy

__all__ = [
    "GateOutcome",
    "LoopConfig",
    "SandboxError",
    "SandboxResult",
    "evaluate_gates",
    "run_research_loop",
    "run_strategy",
]
