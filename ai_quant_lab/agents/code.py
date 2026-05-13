"""CodeAgent: turn a StrategyHypothesis into a runnable Python function.

The output is a single function `strategy(price_data: pd.Series) -> pd.Series`
that returns target positions. The function must:
    - use only `numpy`, `pandas`, and `ai_quant_lab.features.library`
    - never reference future bars (the leakage detector will catch this)
    - return a Series indexed like the input

The sandbox (orchestrator/sandbox.py) is what actually executes the code.
This agent only produces it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ai_quant_lab.agents.base import AgentMessage, call_claude
from ai_quant_lab.agents.hypothesis import StrategyHypothesis


SYSTEM_PROMPT = """You are a Python developer translating quantitative hypotheses into code.

Constraints (non-negotiable):
1. Output ONE function named `strategy(price_data: pd.Series) -> pd.Series`.
2. Imports allowed: numpy as np, pandas as pd, and ai_quant_lab.features.library
   (momentum, rolling_zscore, realized_volatility, range_pct, ewma).
3. NEVER look at future bars. Use .shift(1) or .rolling(...).<aggregate>().
4. Return positions in [-1, 1]. Use .clip(-1, 1) at the end.
5. NaN positions are fine; the engine treats them as 0.
6. Output ONLY the code in a ```python block. No prose before or after.

If the hypothesis is ambiguous, make sensible defaults — do not ask questions.
"""


@dataclass(frozen=True)
class CodeArtifact:
    source: str  # full function source


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```")


class CodeAgent:
    """Renders a strategy hypothesis into a runnable function."""

    def __init__(self, *, model: str | None = None, temperature: float = 0.2) -> None:
        self.model = model
        self.temperature = temperature

    def render(self, hypothesis: StrategyHypothesis) -> CodeArtifact:
        user_content = f"""Hypothesis: {hypothesis.title}

Rationale: {hypothesis.rationale}

Spec:
{_format_spec(hypothesis.spec)}

Write the strategy function."""
        response = call_claude(
            system=SYSTEM_PROMPT,
            messages=[AgentMessage(role="user", content=user_content)],
            model=self.model,
            temperature=self.temperature,
            max_tokens=1024,
        )
        return CodeArtifact(source=_extract_code(response.text))


def _format_spec(spec: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in spec.items())


def _extract_code(text: str) -> str:
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    # Fallback: assume the whole response is code if no fence.
    return text.strip()
