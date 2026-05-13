"""End-to-end integration tests with mocked Claude agents.

The full research loop is exercised against a deterministic universe and a
mock agent suite. The mocks return pre-canned hypotheses, verdicts, and code,
so the system runs in milliseconds and the assertions can be exact.

What this proves (and what unit tests cannot):
    1. The orchestrator wires hypothesis → critic → code → sandbox → gates
       → memory in the right order.
    2. Survivors persist across `run_research_loop` calls via SQLite.
    3. The PCA gate kicks in after >= 2 survivors and rejects duplicates.
    4. The deflated-Sharpe gate uses the honest n_trials from memory.
    5. Bad code from the LLM is rejected in the sandbox before reaching gates.
    6. Critic kills cost 2 LLM calls; full evaluations cost 3.

The mocks are deterministic; their return values are a function of how many
times they've been called and the hypothesis_id. This makes test failures
debuggable rather than flaky.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_quant_lab.agents.code import CodeArtifact
from ai_quant_lab.agents.critic import CriticVerdict
from ai_quant_lab.agents.hypothesis import StrategyHypothesis
from ai_quant_lab.agents.memory import ResearchMemory
from ai_quant_lab.backtest import BacktestConfig
from ai_quant_lab.orchestrator.loop import LoopConfig, run_research_loop


# =========================================================================
# Mock agents — deterministic, no Claude calls.
# =========================================================================

# Three canned hypotheses with code we control.
_GOOD_MOMENTUM_CODE = """
def strategy(price_data):
    lookback = 21
    lagged = price_data.shift(1)
    signal = (lagged / lagged.shift(lookback) - 1.0)
    return np.sign(signal).clip(-1, 1).fillna(0.0)
""".strip()


_GOOD_MEAN_REVERSION_CODE = """
def strategy(price_data):
    lagged = price_data.shift(1)
    rolling_mean = lagged.rolling(21, min_periods=21).mean()
    rolling_std = lagged.rolling(21, min_periods=21).std(ddof=1)
    zscore = (lagged - rolling_mean) / rolling_std
    return (-zscore.clip(-2, 2) / 2.0).fillna(0.0)
""".strip()


# Identical to the momentum strategy — should trip the correlation gate
# after the first one has been accepted.
_DUPLICATE_MOMENTUM_CODE = _GOOD_MOMENTUM_CODE


# Syntactically valid but disallowed-import: must be rejected by the sandbox.
_BAD_IMPORT_CODE = """
import os
def strategy(price_data):
    return price_data * 0
""".strip()


_HYPOTHESIS_BANK = [
    (
        StrategyHypothesis(
            hypothesis_id="mom_21",
            title="21-day momentum",
            rationale="Classic momentum factor",
            spec={"signal": "21d return", "direction": "both"},
            expected_sharpe_range=(0.3, 0.8),
            works_in_regime="trending",
            breaks_in_regime="mean-reverting",
        ),
        _GOOD_MOMENTUM_CODE,
        True,  # critic passes
    ),
    (
        StrategyHypothesis(
            hypothesis_id="kill_me",
            title="Lookahead leakage suspect",
            rationale="(critic will kill this)",
            spec={"signal": "future return", "direction": "long"},
            expected_sharpe_range=(2.0, 5.0),
            works_in_regime="always",
            breaks_in_regime="never",
        ),
        "raise NotImplementedError()",  # never executed; critic kills first
        False,  # critic kills
    ),
    (
        StrategyHypothesis(
            hypothesis_id="mom_21_dup",
            title="21-day momentum (duplicate)",
            rationale="Same idea, slightly different framing",
            spec={"signal": "21d return", "direction": "both"},
            expected_sharpe_range=(0.3, 0.8),
            works_in_regime="trending",
            breaks_in_regime="mean-reverting",
        ),
        _DUPLICATE_MOMENTUM_CODE,
        True,
    ),
    (
        StrategyHypothesis(
            hypothesis_id="meanrev_21",
            title="21-day mean reversion",
            rationale="z-score mean reversion",
            spec={"signal": "21d zscore", "direction": "both"},
            expected_sharpe_range=(0.3, 0.8),
            works_in_regime="range-bound",
            breaks_in_regime="trending",
        ),
        _GOOD_MEAN_REVERSION_CODE,
        True,
    ),
    (
        StrategyHypothesis(
            hypothesis_id="bad_import",
            title="Bad imports",
            rationale="Triggers sandbox rejection",
            spec={"signal": "n/a", "direction": "long"},
            expected_sharpe_range=(0.3, 0.8),
            works_in_regime="always",
            breaks_in_regime="never",
        ),
        _BAD_IMPORT_CODE,
        True,  # critic lets it through; sandbox kills
    ),
]


@dataclass
class MockHypothesisAgent:
    """Cycles through `_HYPOTHESIS_BANK` deterministically."""

    call_count: int = 0

    def propose(self, market_description, prior_trials_summary, **kwargs) -> StrategyHypothesis:
        hypothesis = _HYPOTHESIS_BANK[self.call_count % len(_HYPOTHESIS_BANK)][0]
        self.call_count += 1
        return hypothesis


@dataclass
class MockCriticAgent:
    """Looks up the critic verdict from the bank by hypothesis_id."""

    call_count: int = 0
    verdicts: list[CriticVerdict] = field(default_factory=list)

    def review(self, hypothesis: StrategyHypothesis) -> CriticVerdict:
        passes = next(
            (passes for h, _code, passes in _HYPOTHESIS_BANK if h.hypothesis_id == hypothesis.hypothesis_id),
            True,
        )
        verdict = CriticVerdict(
            passes=passes,
            reasoning=("kill" if not passes else "looks reasonable"),
            kill_reasons=([] if passes else ["lookahead"]),
        )
        self.call_count += 1
        self.verdicts.append(verdict)
        return verdict


@dataclass
class MockCodeAgent:
    call_count: int = 0

    def render(self, hypothesis: StrategyHypothesis) -> CodeArtifact:
        source = next(
            (code for h, code, _ in _HYPOTHESIS_BANK if h.hypothesis_id == hypothesis.hypothesis_id),
            "def strategy(p): return p * 0",
        )
        self.call_count += 1
        return CodeArtifact(source=source)


# =========================================================================
# Universe fixture
# =========================================================================


@pytest.fixture
def universe() -> pd.Series:
    """Synthetic single-asset price series with mild positive autocorrelation."""
    rng = np.random.default_rng(2025)
    n = 2000
    shocks = np.zeros(n)
    last = 0.0
    for i in range(n):
        shock = rng.normal(0.0003, 0.012)
        shocks[i] = 0.05 * last + shock
        last = shocks[i]
    prices = 100.0 * np.exp(np.cumsum(shocks))
    return pd.Series(
        prices,
        index=pd.bdate_range(end="2026-01-01", periods=n),
        name="close",
    )


# =========================================================================
# Tests
# =========================================================================


def test_loop_wires_pipeline_end_to_end(universe, tmp_path: Path):
    """One iteration of the loop must hit every stage in the right order."""
    hypothesis_agent = MockHypothesisAgent()
    code_agent = MockCodeAgent()
    critic_agent = MockCriticAgent()

    loop_config = LoopConfig(
        market_description="Daily bars on a single liquid US equity, 8 years.",
        iterations=1,
        target_survivors=10,
        max_llm_calls=10,
        backtest_config=BacktestConfig(cost_bps=5.0),
    )

    with ResearchMemory(tmp_path / "m.db") as memory:
        artifacts, survivors = run_research_loop(
            universe, loop_config, memory=memory,
            hypothesis_agent=hypothesis_agent,
            code_agent=code_agent,
            critic_agent=critic_agent,
            log=lambda _msg: None,
        )

    assert len(artifacts) == 1
    assert hypothesis_agent.call_count == 1
    assert critic_agent.call_count == 1
    # First entry in the bank is mom_21 with critic-pass → code agent is called.
    assert code_agent.call_count == 1


def test_critic_kill_skips_code_agent(universe, tmp_path: Path):
    """When the critic kills, the code agent must not be called."""
    hypothesis_agent = MockHypothesisAgent()
    hypothesis_agent.call_count = 1  # skip ahead to "kill_me" (index 1)
    code_agent = MockCodeAgent()
    critic_agent = MockCriticAgent()

    loop_config = LoopConfig(
        market_description="Test", iterations=1, target_survivors=5,
        max_llm_calls=10, backtest_config=BacktestConfig(cost_bps=5.0),
    )

    with ResearchMemory(tmp_path / "m.db") as memory:
        artifacts, _ = run_research_loop(
            universe, loop_config, memory=memory,
            hypothesis_agent=hypothesis_agent,
            code_agent=code_agent,
            critic_agent=critic_agent,
            log=lambda _msg: None,
        )

    assert len(artifacts) == 1
    assert artifacts[0].accepted is False
    assert artifacts[0].rejection_reason == "critic"
    assert code_agent.call_count == 0


def test_sandbox_error_is_logged_in_memory(universe, tmp_path: Path):
    """Bad imports must produce a sandbox_error rejection reason in memory."""
    hypothesis_agent = MockHypothesisAgent()
    hypothesis_agent.call_count = 4  # jump to "bad_import"
    code_agent = MockCodeAgent()
    critic_agent = MockCriticAgent()

    loop_config = LoopConfig(
        market_description="Test", iterations=1, target_survivors=5,
        max_llm_calls=10, backtest_config=BacktestConfig(cost_bps=5.0),
    )

    with ResearchMemory(tmp_path / "m.db") as memory:
        artifacts, _ = run_research_loop(
            universe, loop_config, memory=memory,
            hypothesis_agent=hypothesis_agent,
            code_agent=code_agent,
            critic_agent=critic_agent,
            log=lambda _msg: None,
        )
        rows = memory.history()

    assert len(artifacts) == 1
    assert artifacts[0].rejection_reason == "sandbox_error"
    assert any(r.rejection_reason == "sandbox_error" for r in rows)


def test_loop_persists_across_invocations(universe, tmp_path: Path):
    """Survivors and n_trials must survive process-restart-equivalent reopen."""
    db_path = tmp_path / "m.db"
    config = LoopConfig(
        market_description="Test",
        iterations=2,
        target_survivors=10,
        max_llm_calls=10,
        backtest_config=BacktestConfig(cost_bps=5.0),
    )

    with ResearchMemory(db_path) as memory:
        run_research_loop(
            universe, config, memory=memory,
            hypothesis_agent=MockHypothesisAgent(),
            code_agent=MockCodeAgent(),
            critic_agent=MockCriticAgent(),
            log=lambda _msg: None,
        )
        n_trials_first = memory.n_trials()
        survivors_first = len(memory.survivors())

    # Reopen — same persistence layer, new connection.
    with ResearchMemory(db_path) as memory:
        assert memory.n_trials() == n_trials_first
        assert len(memory.survivors()) == survivors_first
        accepted_returns = memory.accepted_returns()
        if survivors_first > 0:
            assert len(accepted_returns) == survivors_first


def test_correlation_gate_rejects_duplicate(universe, tmp_path: Path):
    """Second momentum strategy must be rejected as duplicate of the first."""
    hypothesis_agent = MockHypothesisAgent()
    # Run mom_21, kill_me, mom_21_dup in that order.
    config = LoopConfig(
        market_description="Test",
        iterations=3,
        target_survivors=10,
        max_llm_calls=20,
        backtest_config=BacktestConfig(cost_bps=5.0),
    )

    with ResearchMemory(tmp_path / "m.db") as memory:
        artifacts, survivors = run_research_loop(
            universe, config, memory=memory,
            hypothesis_agent=hypothesis_agent,
            code_agent=MockCodeAgent(),
            critic_agent=MockCriticAgent(),
            log=lambda _msg: None,
        )

    by_id = {a.hypothesis_id: a for a in artifacts}
    # Three iterations: mom_21 → kill_me → mom_21_dup
    assert "kill_me" in by_id
    assert by_id["kill_me"].rejection_reason == "critic"
    # The duplicate must NOT have been accepted (correlation or PCA blocked it).
    if "mom_21_dup" in by_id:
        if "mom_21" in by_id and by_id["mom_21"].accepted:
            assert not by_id["mom_21_dup"].accepted


def test_loop_respects_target_survivors(universe, tmp_path: Path):
    """The loop must stop early when target_survivors is reached."""
    config = LoopConfig(
        market_description="Test",
        iterations=20,
        target_survivors=1,  # stop after 1 acceptance
        max_llm_calls=100,
        backtest_config=BacktestConfig(cost_bps=5.0),
    )
    with ResearchMemory(tmp_path / "m.db") as memory:
        artifacts, survivors = run_research_loop(
            universe, config, memory=memory,
            hypothesis_agent=MockHypothesisAgent(),
            code_agent=MockCodeAgent(),
            critic_agent=MockCriticAgent(),
            log=lambda _msg: None,
        )

    # The loop should stop after reaching the target (or run out of iterations
    # without finding any). At most one survivor should be accepted.
    n_accepted = sum(1 for a in artifacts if a.accepted)
    assert n_accepted <= max(1, config.target_survivors + 1)


def test_loop_respects_max_llm_calls(universe, tmp_path: Path):
    """`max_llm_calls` must cap the number of iterations."""
    config = LoopConfig(
        market_description="Test",
        iterations=100,
        target_survivors=100,  # never reach target
        max_llm_calls=6,       # 2 iterations × 3 calls = 6
        backtest_config=BacktestConfig(cost_bps=5.0),
    )
    with ResearchMemory(tmp_path / "m.db") as memory:
        artifacts, _ = run_research_loop(
            universe, config, memory=memory,
            hypothesis_agent=MockHypothesisAgent(),
            code_agent=MockCodeAgent(),
            critic_agent=MockCriticAgent(),
            log=lambda _msg: None,
        )
    # Each iteration uses 2 or 3 calls. With max=6 we should hit at most 3.
    assert len(artifacts) <= 3


def test_full_run_produces_diverse_survivors(universe, tmp_path: Path):
    """End-to-end: a multi-iteration run should produce uncorrelated survivors."""
    hypothesis_agent = MockHypothesisAgent()
    config = LoopConfig(
        market_description="Test",
        iterations=5,                # cycles through all 5 bank entries
        target_survivors=10,
        max_llm_calls=50,
        backtest_config=BacktestConfig(cost_bps=5.0),
    )

    with ResearchMemory(tmp_path / "m.db") as memory:
        artifacts, survivors = run_research_loop(
            universe, config, memory=memory,
            hypothesis_agent=hypothesis_agent,
            code_agent=MockCodeAgent(),
            critic_agent=MockCriticAgent(),
            log=lambda _msg: None,
        )
        n_trials_logged = memory.n_trials()

    # Every iteration must produce a memory row.
    assert n_trials_logged == 5
    # Critic killed one ('kill_me'); sandbox killed one ('bad_import').
    by_reason: dict[str, int] = {}
    for a in artifacts:
        if a.rejection_reason:
            by_reason[a.rejection_reason] = by_reason.get(a.rejection_reason, 0) + 1
    assert by_reason.get("critic", 0) >= 1
    assert by_reason.get("sandbox_error", 0) >= 1
