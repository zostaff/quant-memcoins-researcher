"""Tests for orchestrator gates.

Failing strategies must fail the appropriate gate. Each test confirms that
exactly one expected failure mode trips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_lab.agents.critic import CriticVerdict
from ai_quant_lab.agents.memory import ResearchMemory, TrialRecord
from ai_quant_lab.orchestrator.gates import evaluate_gates
from ai_quant_lab.orchestrator.sandbox import SandboxError, run_strategy


def test_critic_kill_blocks_pipeline(tmp_path: Path):
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    verdict = CriticVerdict(passes=False, reasoning="bad idea", kill_reasons=["lookahead"])
    with ResearchMemory(tmp_path / "m.db") as memory:
        outcome = evaluate_gates(verdict, returns, memory=memory)
    assert not outcome.passes
    assert outcome.rejection_reason == "critic"


def test_dsr_fails_lucky_strategy(tmp_path: Path):
    rng = np.random.default_rng(0)
    # Pick the best of 200 random series; DSR with n_trials=200 should reject.
    best = None
    for _ in range(200):
        cand = pd.Series(rng.normal(0, 0.01, 800))
        if best is None or cand.mean() / cand.std() > best.mean() / best.std():
            best = cand
    verdict = CriticVerdict(passes=True, reasoning="ok", kill_reasons=[])

    # Record 200 trials before evaluating
    with ResearchMemory(tmp_path / "m.db") as memory:
        for i in range(200):
            memory.record(
                TrialRecord(
                    hypothesis_id=f"h{i}", hypothesis_text="t", rationale="",
                    code="", metrics={}, accepted=False, n_trials_at_time=i, iteration=i,
                )
            )
        outcome = evaluate_gates(verdict, best, memory=memory, accepted_returns=[])
    assert not outcome.passes
    assert outcome.rejection_reason.startswith("deflated_sharpe")


def test_correlation_gate_blocks_redundant(tmp_path: Path):
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0.001, 0.01, 500))
    # nearly identical series — high correlation
    duplicate = base + rng.normal(0, 0.0001, 500)
    verdict = CriticVerdict(passes=True, reasoning="ok", kill_reasons=[])
    with ResearchMemory(tmp_path / "m.db") as memory:
        memory.record(TrialRecord(
            hypothesis_id="h0", hypothesis_text="t", rationale="",
            code="", metrics={}, accepted=True, n_trials_at_time=0, iteration=0,
        ))
        outcome = evaluate_gates(
            verdict, duplicate, memory=memory, accepted_returns=[base],
            dsr_pvalue_max=0.99,  # let DSR pass to isolate the correlation gate
            max_correlation=0.5,
        )
    assert not outcome.passes
    assert outcome.rejection_reason.startswith("correlation")


def test_sandbox_rejects_disallowed_imports(gbm_price_series):
    bad = "import os\ndef strategy(price_data):\n    return price_data * 0"
    try:
        run_strategy(bad, gbm_price_series)
    except SandboxError as exc:
        assert "Disallowed import" in str(exc)
    else:
        raise AssertionError("expected SandboxError")


def test_sandbox_rejects_wrong_signature(gbm_price_series):
    src = "def not_strategy(x): return x"
    try:
        run_strategy(src, gbm_price_series)
    except SandboxError as exc:
        assert "strategy" in str(exc)
    else:
        raise AssertionError("expected SandboxError")


def test_sandbox_accepts_good_strategy(gbm_price_series):
    # `pd` and `np` are pre-injected into the sandbox namespace; no imports needed.
    src = "def strategy(price_data):\n    return pd.Series(0.0, index=price_data.index)\n"
    result = run_strategy(src, gbm_price_series)
    assert (result.positions == 0.0).all()
