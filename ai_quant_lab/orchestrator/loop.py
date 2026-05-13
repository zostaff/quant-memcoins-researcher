"""run_research_loop: the orchestrator's entry point.

The signature is small on purpose. Everything tunable lives in `LoopConfig`,
which is itself derived from `settings`. The function is callable from a
notebook, from `run.py`, or from a test (with mocked agents).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Sequence

import pandas as pd

from ai_quant_lab.agents import (
    CodeAgent,
    CriticAgent,
    HypothesisAgent,
    ResearchMemory,
    TrialRecord,
)
from ai_quant_lab.agents.critic import MarketType
from ai_quant_lab.backtest import BacktestConfig, vectorized_backtest
from ai_quant_lab.config import settings
from ai_quant_lab.orchestrator.gates import GateOutcome, evaluate_gates
from ai_quant_lab.orchestrator.sandbox import SandboxError, run_strategy


@dataclass
class LoopConfig:
    """All tunable knobs for the research loop."""

    market_description: str
    market_type: MarketType = "generic"
    iterations: int = 50
    target_survivors: int = field(default_factory=lambda: settings.target_survivors)
    max_llm_calls: int = field(default_factory=lambda: settings.max_llm_calls)
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)
    annualization: int = field(default_factory=lambda: settings.annualization)


@dataclass
class LoopArtifact:
    """One iteration's full record: useful for debugging and writing reports."""

    iteration: int
    hypothesis_id: str
    title: str
    accepted: bool
    rejection_reason: str | None
    sharpe: float
    gate_outcome: GateOutcome | None


def run_research_loop(
    price_data: pd.Series,
    config: LoopConfig,
    *,
    memory: ResearchMemory,
    hypothesis_agent: HypothesisAgent | None = None,
    code_agent: CodeAgent | None = None,
    critic_agent: CriticAgent | None = None,
    log: Callable[[str], None] = print,
) -> tuple[list[LoopArtifact], list[TrialRecord]]:
    """Drive the loop until `target_survivors` or `iterations` is hit.

    Returns:
        (artifacts, survivors) — full per-iteration log and the accepted set.
    """
    hypothesis_agent = hypothesis_agent or HypothesisAgent()
    code_agent = code_agent or CodeAgent()
    critic_agent = critic_agent or CriticAgent(market_type=config.market_type)

    artifacts: list[LoopArtifact] = []
    survivors: list[TrialRecord] = list(memory.survivors())
    # Reload accepted returns from memory so we survive process restarts and
    # honor every prior survivor when checking the correlation gate.
    accepted_returns: list[pd.Series] = list(memory.accepted_returns())
    llm_calls = 0

    for iteration in range(config.iterations):
        if len(survivors) >= config.target_survivors:
            log(f"[done] reached target_survivors={config.target_survivors}")
            break
        if llm_calls >= config.max_llm_calls:
            log(f"[done] max_llm_calls={config.max_llm_calls} reached")
            break

        artifact = _run_iteration(
            iteration=iteration,
            price_data=price_data,
            config=config,
            memory=memory,
            survivors=survivors,
            accepted_returns=accepted_returns,
            hypothesis_agent=hypothesis_agent,
            code_agent=code_agent,
            critic_agent=critic_agent,
            log=log,
        )
        artifacts.append(artifact)
        llm_calls += _llm_calls_per_iteration(artifact)

        if artifact.accepted:
            # Re-source from memory so every consumer sees the same truth.
            survivors = list(memory.survivors())
            accepted_returns = list(memory.accepted_returns())

    return artifacts, survivors


def _run_iteration(
    *,
    iteration: int,
    price_data: pd.Series,
    config: LoopConfig,
    memory: ResearchMemory,
    survivors: Sequence[TrialRecord],
    accepted_returns: list[pd.Series],
    hypothesis_agent: HypothesisAgent,
    code_agent: CodeAgent,
    critic_agent: CriticAgent,
    log: Callable[[str], None],
) -> LoopArtifact:
    summary = memory.summarize_for_prompt(limit=10)
    survivor_titles = [t.hypothesis_text for t in survivors]
    hypothesis = hypothesis_agent.propose(
        market_description=config.market_description,
        prior_trials_summary=summary,
        avoid_correlation_with=survivor_titles,
    )
    log(f"[{iteration:03d}] propose: {hypothesis.hypothesis_id} — {hypothesis.title}")

    verdict = critic_agent.review(hypothesis)
    if not verdict.passes:
        log(f"[{iteration:03d}] critic killed: {verdict.reasoning[:90]}")
        memory.record(
            TrialRecord(
                hypothesis_id=hypothesis.hypothesis_id,
                hypothesis_text=hypothesis.title,
                rationale=hypothesis.rationale,
                code="",
                metrics={},
                accepted=False,
                rejection_reason="critic",
                n_trials_at_time=memory.n_trials(),
                iteration=iteration,
            )
        )
        return LoopArtifact(
            iteration=iteration,
            hypothesis_id=hypothesis.hypothesis_id,
            title=hypothesis.title,
            accepted=False,
            rejection_reason="critic",
            sharpe=0.0,
            gate_outcome=None,
        )

    code = code_agent.render(hypothesis)
    try:
        sandbox_result = run_strategy(code.source, price_data)
    except SandboxError as exc:
        log(f"[{iteration:03d}] sandbox error: {exc}")
        memory.record(
            TrialRecord(
                hypothesis_id=hypothesis.hypothesis_id,
                hypothesis_text=hypothesis.title,
                rationale=hypothesis.rationale,
                code=code.source,
                metrics={},
                accepted=False,
                rejection_reason="sandbox_error",
                n_trials_at_time=memory.n_trials(),
                iteration=iteration,
            )
        )
        return LoopArtifact(
            iteration=iteration,
            hypothesis_id=hypothesis.hypothesis_id,
            title=hypothesis.title,
            accepted=False,
            rejection_reason="sandbox_error",
            sharpe=0.0,
            gate_outcome=None,
        )

    returns = price_data.pct_change().fillna(0.0)
    result = vectorized_backtest(sandbox_result.positions, returns, config=config.backtest_config)
    gate_outcome = evaluate_gates(
        critic_verdict=verdict,
        strategy_returns=result.returns,
        memory=memory,
        accepted_returns=accepted_returns,
        annualization=config.annualization,
    )

    returns_payload = ""
    if gate_outcome.passes:
        returns_payload = json.dumps(
            {
                "index": [str(t) for t in result.returns.index],
                "values": [float(v) for v in result.returns.tolist()],
            }
        )
    memory.record(
        TrialRecord(
            hypothesis_id=hypothesis.hypothesis_id,
            hypothesis_text=hypothesis.title,
            rationale=hypothesis.rationale,
            code=code.source,
            metrics=result.metrics,
            accepted=gate_outcome.passes,
            rejection_reason=gate_outcome.rejection_reason,
            n_trials_at_time=memory.n_trials(),
            iteration=iteration,
            returns_json=returns_payload,
        )
    )
    log(
        f"[{iteration:03d}] SR={result.metrics['sharpe_ratio']:+.2f} "
        f"→ {'ACCEPT' if gate_outcome.passes else f'REJECT ({gate_outcome.rejection_reason})'}"
    )

    if gate_outcome.passes:
        accepted_returns.append(result.returns)

    return LoopArtifact(
        iteration=iteration,
        hypothesis_id=hypothesis.hypothesis_id,
        title=hypothesis.title,
        accepted=gate_outcome.passes,
        rejection_reason=gate_outcome.rejection_reason,
        sharpe=result.metrics["sharpe_ratio"],
        gate_outcome=gate_outcome,
    )


def _llm_calls_per_iteration(artifact: LoopArtifact) -> int:
    # Hypothesis + critic always. Code only if critic passes.
    if artifact.rejection_reason == "critic":
        return 2
    return 3
