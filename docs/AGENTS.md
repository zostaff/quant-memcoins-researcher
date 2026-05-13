# Agents

Five Claude-powered agents power the research loop. Each lives in
`ai_quant_lab/agents/`. They share a thin SDK wrapper (`base.py`) that handles
retries, prompt caching, and JSON extraction.

## HypothesisAgent (`agents/hypothesis.py`)

**Role**: propose a new strategy given a market description and a memory of
prior trials.

**System prompt**: instructs Claude that creativity is _not_ wanted; only
strategies grounded in known microstructure (momentum, mean reversion, vol
carry, term structure, dispersion). Every proposal must include the regime
where the edge exists AND the regime where it dies. Output is a strict JSON
schema (`hypothesis_id`, `title`, `rationale`, `spec`, `expected_sharpe_range`,
`works_in_regime`, `breaks_in_regime`).

**Defaults**: `temperature=0.6`. Slightly higher than the other agents because
we want diversity in the proposal space, not diversity in the syntax.

## CriticAgent (`agents/critic.py`)

**Role**: argue against the hypothesis BEFORE we spend a backtest budget on it.

**System prompt**: explicitly adversarial. Looks for six failure modes:
implicit lookahead, survivorship/selection bias, already-arbitraged factors,
single-parameter overfit, cost-fragility, IS/OOS conflation. Bias: when in
doubt, kill.

**Output**: `{"verdict": "pass"|"kill", "reasoning": ..., "kill_reasons": [...]}`.

**Defaults**: `temperature=0.3`. Skepticism doesn't need warmth.

## CodeAgent (`agents/code.py`)

**Role**: translate a hypothesis into runnable Python.

**System prompt**: hard constraints — single function named `strategy(price_data)`,
imports limited to `numpy`, `pandas`, `ai_quant_lab.features.library`, no
forward references, clipped to `[-1, 1]`. Returns code in a ` ```python ` block.

**Defaults**: `temperature=0.2`. Code generation rewards precision.

## RiskAgent (`agents/risk.py`)

**Role**: post-gate sanity check on the realized positions. NOT a substitute
for a real risk model — it's a checklist runner.

**Output**: `{"risk_score": int 0-10, "concerns": [...], "size_recommendation": float ∈ (0, 1]}`.

**Defaults**: `temperature=0.2`. Conservative.

## ResearchMemory (`agents/memory.py`)

Not an LLM — the SQLite store that backs the deflated Sharpe gate. Every
hypothesis ever proposed (accepted, rejected, killed mid-loop) gets a row.
`n_trials()` is the honest count fed to the gate.

Schema is intentionally narrow: ten columns, two indexes. The point is
auditability, not analytics.

## Prompt caching

`base.call_claude` marks the system prompt with `cache_control: ephemeral` by
default. Since the system prompt is invariant across every iteration, all
calls after the first hit the 5-minute cache. Cost reduction is roughly 10×
on a long loop.

## Why these five

The split is functional, not bureaucratic. Each agent has a single job that
either ends the iteration (rejection) or hands off to the next stage. There
is no `OrchestrationAgent` or `ConsensusAgent` — orchestration is a Python
function (`orchestrator/loop.py`), because LLMs are bad at deterministic
control flow.

## What an iteration looks like

```
1. HypothesisAgent.propose(market, memory_summary)            ──► proposal
2. CriticAgent.review(proposal)                                ──► verdict
3. if verdict.passes:
4.     CodeAgent.render(proposal)                              ──► strategy.py
5.     run_strategy(source, prices)                             ──► positions
6.     vectorized_backtest(positions, returns)                  ──► metrics
7.     evaluate_gates(verdict, returns, memory, accepted)       ──► outcome
8. memory.record(TrialRecord(...))
```

Steps 1-4 cost three Claude calls per iteration when the critic passes, two
when it kills. With prompt caching, a typical run of 50 iterations is under
$2 of API usage.
