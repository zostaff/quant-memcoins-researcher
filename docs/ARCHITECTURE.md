# Architecture

`ai-quant-lab` is intentionally small. Every module fits in a single sitting and
does one thing.

## Module map

```
ai_quant_lab/
├── config.py            Pydantic settings sourced from env vars.
├── backtest/            Vectorized + event-driven engines and cost models.
├── features/            Leakage-proof feature pipeline + automated detector.
├── validation/          Walk-forward, purged CV, deflated Sharpe, diagnostics.
├── agents/              Claude-powered hypothesis / code / critic / risk / memory.
├── orchestrator/        Research loop, gates, sandbox.
├── production/          Paper-trading diagnostics, kill switch, audit log.
└── run.py               CLI entry point.
```

## Data flow

```
        ┌──────────────────────────────────────────────────────────────┐
        │                       Research Loop                          │
        │                                                              │
        │   memory ──► HypothesisAgent                                 │
        │                  │                                           │
        │                  ▼                                           │
        │              CriticAgent ── reject ──► memory (rejected)     │
        │                  │  pass                                     │
        │                  ▼                                           │
        │              CodeAgent ──► Sandbox ──► vectorized_backtest   │
        │                                  │                           │
        │                                  ▼                           │
        │                            evaluate_gates                    │
        │                            (critic · DSR · correlation)      │
        │                                  │                           │
        │                                  ▼                           │
        │                              memory ──► next iteration       │
        └──────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                            Survivors (accepted strategies)
                                       │
                                       ▼
                            Paper trading + LiveDiagnostic + KillSwitch
```

## Three gates

Every candidate strategy must pass all three of:

1. **Critic gate** — an adversarial LLM review of the hypothesis BEFORE the
   backtest runs. Cheap, catches the obvious nonsense.
2. **Deflated Sharpe gate** — penalty for multiple-testing, parameterized by
   `n_trials` (which `ResearchMemory` tracks honestly). No override.
3. **Correlation gate** — the new candidate's returns must be uncorrelated
   enough with already-accepted strategies. Keeps the survivor set diverse.

## Sandbox

The strategy code coming out of `CodeAgent` runs through `orchestrator/sandbox.py`,
which:

- AST-walks the source to reject any import outside `numpy`, `pandas`, `math`,
  and `ai_quant_lab.features.library`.
- Runs the strategy in an `exec` namespace with a whitelist of builtins
  (`abs`, `len`, `range`, etc. — but no `open`, no `__import__`, no `eval`).
- Enforces a wall-clock timeout via `signal.SIGALRM` on POSIX.

This is _not_ a security boundary. Anyone with shell access can do anything.
The sandbox catches accidents, not adversaries.

## Where to extend

- **More features**: drop a function in `features/library.py`. The pipeline
  picks them up via `FeatureStep`.
- **More gates**: add to `orchestrator/gates.py` and update `evaluate_gates`.
  Keep them deterministic and cheap.
- **More agents**: any module that talks to Claude through `agents/base.py`
  benefits from retries and prompt caching for free.

## What's deliberately out of scope

- Broker integrations. Production trading needs reconciliation, idempotency,
  and operational discipline that don't belong in a research tool.
- Real-time feeds. The whole system assumes a pandas Series of historical
  prices. Add your own ingestion layer.
- Multi-asset portfolio construction. Each strategy is single-instrument by
  design; combine them downstream with whatever sizing scheme fits.
