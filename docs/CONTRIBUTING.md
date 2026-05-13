# Contributing

ai-quant-lab is small and tries to stay that way. PRs that add bells and
whistles will be politely declined; PRs that improve the validation gates,
catch new leakage shapes, or sharpen the docs are welcome.

## Setup

```bash
git clone https://github.com/yourname/ai-quant-lab
cd ai-quant-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## What kind of contributions

### Welcome

- New features in `features/library.py`. Each must be lagged by construction
  and accompanied by a test that proves the leakage detector lets it through.
- New diagnostics in `validation/diagnostics.py`. Cheap, deterministic checks
  the orchestrator can call after the gates.
- New kill-switch triggers in `production/kill_switch.py`. Add a function,
  add a test.
- Better leakage detection. The current detector catches the obvious cases;
  it doesn't catch survivorship bias or universe selection. Improvements
  welcome.
- Docs corrections. Especially in `VALIDATION.md`.

### Not welcome

- New ML/DL models. ai-quant-lab is not a modeling framework; it's a
  validation framework. Use the gates to test whatever model you want from
  the outside.
- New brokers, data feeds, or infrastructure. Out of scope. Use a separate
  layer.
- Renaming the deflated_sharpe gate or making its threshold easier to
  override. The whole point is that the gate has no override.

## Style

- Python 3.11+, type hints everywhere.
- Each module < 300 lines. If you need more, split.
- Docstrings in Google style.
- Variable names are full words: `price_data`, not `df`. Exception: standard
  quant abbreviations (`OOS`, `IS`, `PnL`, `AUM`, `ADV`).
- Comments explain WHY, not WHAT. Prefer no comments to obvious ones.
- Run `ruff check ai_quant_lab` and `pytest` before opening a PR.

## Tests

Every PR should keep `pytest` green. New code needs new tests.

Tests do NOT hit the Claude API. If you're adding agent behavior, mock the
`call_claude` call or test the parsing logic separately (see
`tests/test_agents.py`).
