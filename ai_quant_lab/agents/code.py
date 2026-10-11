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

SYSTEM_PROMPT_SINGLE = """You are a Python developer translating quantitative hypotheses into code.

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


SYSTEM_PROMPT_CROSS_SECTIONAL = """You are a Python developer translating quantitative hypotheses into code.

The strategy is CROSS-SECTIONAL: it ranks/scores assets at each bar and goes
long the best, short the worst (or whatever the hypothesis specifies).

Constraints (non-negotiable):
1. Output ONE function: `strategy(price_data: pd.DataFrame) -> pd.DataFrame`.
   `price_data` has time on the index, asset id on columns. Return a DataFrame
   of the SAME shape, where each cell is the target weight for that asset at
   that time.
2. Imports allowed: numpy as np, pandas as pd,
   ai_quant_lab.features.library, ai_quant_lab.features.cross_sectional
   (rank_within_universe, zscore_cross_section, neutralize_by_factor,
    industry_neutralize, cross_sectional_momentum).
3. NEVER look at future bars. Always .shift(1) before computing signals.
4. The portfolio should be roughly dollar-neutral: sum(weights per row) ≈ 0.
   Use long_short_quantile_portfolio shape: long top quantile, short bottom.
5. Weights are unbounded per asset but the engine clips to [-1, 1] per cell.
   Typical magnitudes are 1/N where N is universe size.
6. Output ONLY the code in a ```python block.
"""


SYSTEM_PROMPT_MEMECOIN = """You are a Python developer translating quantitative hypotheses into code.

The strategy trades a MEMECOIN universe: tokens launch and die all the time.

Constraints (non-negotiable):
1. Output ONE function: `strategy(panel: dict) -> pd.DataFrame`.
   `panel` holds time × token DataFrames: 'close' (USD price, NaN before launch
   and after death), 'liquidity' (pool TVL in USD, 0 when dead), 'volume'
   (USD per bar) and, when available: 'holders' (holder count), 'top10_share'
   (supply share of the 10 largest wallets, 0..1), 'graduation' (pump.fun
   bonding-curve progress, 0..1), 'mint_risk' (1 if the creator can still
   mint or freeze), 'dev_share' (developer's share of supply), 'buy_ratio'
   (share of buys in recent transactions) and 'trending_rank' (1 = top of the
   GeckoTerminal trending list, NaN when not listed). Narrative and mood
   fields, broadcast per token: 'narrative_heat' (the token's narrative, short
   vs long share of volume; > 0 = gaining attention), 'narrative_return',
   'launch_saturation' (copycat launches vs usual pace; > 2 = crowded),
   'risk_appetite' (market-wide z-score; > 0 = risk-on) and 'breadth'.
   Token fields are NaN until first observed; use `panel.get(name)` and
   handle None.
2. Return LONG-ONLY target weights shaped exactly like panel['close'].
   Weights >= 0, each row sums to <= 1 (the rest is cash). The engine caps
   each weight and each position's share of the pool.
3. A weight at bar t is executed at bar t+1 by the engine. Use only data up to
   bar t. Never shift by a negative number, never use future rows.
4. Only hold tokens that are tradeable at bar t (price present, liquidity
   above a floor): use meme.features.tradeable.
5. Keep turnover low. Every trade pays swap fee + curve slippage + priority
   fee. Rebalance on a schedule (e.g. `w.iloc[::k].reindex(w.index).ffill()`)
   or hold positions for many bars; never rotate the book every bar.
6. Imports allowed: numpy as np, pandas as pd, and
   `from ai_quant_lab.meme import features` (tradeable, age_bars,
   trailing_return, drawdown_from_high, liquidity_growth, volume_to_liquidity,
   volume_zscore, holder_growth, rug_screen, graduation_window, cross_rank,
   top_n_equal_weight, hold_for).
7. Output ONLY the code in a ```python block.
"""


# Default to single-asset for backward compatibility.
SYSTEM_PROMPT = SYSTEM_PROMPT_SINGLE


@dataclass(frozen=True)
class CodeArtifact:
    source: str  # full function source


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```")


class CodeAgent:
    """Renders a strategy hypothesis into a runnable function.

    Switches between single-asset and cross-sectional system prompts based on
    the `mode` argument. Cross-sectional mode is used when the universe is
    a basket (e.g. equities) and dollar-neutral long-short is the goal.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        mode: str = "single",  # 'single' | 'cross_sectional' | 'memecoin'
    ) -> None:
        prompts = {
            "single": SYSTEM_PROMPT_SINGLE,
            "cross_sectional": SYSTEM_PROMPT_CROSS_SECTIONAL,
            "memecoin": SYSTEM_PROMPT_MEMECOIN,
        }
        if mode not in prompts:
            raise ValueError(f"mode must be one of {sorted(prompts)}")
        self.model = model
        self.temperature = temperature
        self.mode = mode
        self._system_prompt = prompts[mode]

    def render(self, hypothesis: StrategyHypothesis) -> CodeArtifact:
        user_content = f"""Hypothesis: {hypothesis.title}

Rationale: {hypothesis.rationale}

Spec:
{_format_spec(hypothesis.spec)}

Write the strategy function."""
        response = call_claude(
            system=self._system_prompt,
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
