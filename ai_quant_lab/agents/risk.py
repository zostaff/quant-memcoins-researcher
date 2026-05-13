"""RiskAgent: sanity-check the position sizing of an accepted strategy.

Triggered AFTER the validation gates pass, BEFORE the strategy is added to a
paper-trade allocation. Looks at the realized positions and asks:
    - Does it concentrate beyond what was approved?
    - Does it leverage up during high-volatility regimes (the bad time)?
    - Does it have implicit short-vol or carry exposure?

The agent doesn't have a sophisticated risk model; it has a checklist and an
ability to read recent positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_quant_lab.agents.base import AgentMessage, call_claude, extract_first_json


SYSTEM_PROMPT = """You are a risk officer reviewing a quantitative strategy before it goes live.

You will be shown:
- The strategy hypothesis (one sentence).
- Summary statistics of the realized positions over the backtest period.
- Realized returns aggregated by volatility regime.

Identify risks. Output JSON ONLY:
{
  "risk_score": <integer 0-10, higher = more risky>,
  "concerns": [<list of specific concerns>],
  "size_recommendation": <float in (0, 1] - fraction of approved size to actually deploy>
}

Be conservative. Default size_recommendation is 0.5. Only go above 0.8 if the
strategy has no concentration concerns AND no short-volatility exposure.
"""


@dataclass(frozen=True)
class RiskAssessment:
    risk_score: int
    concerns: list[str]
    size_recommendation: float


class RiskAgent:
    """LLM-driven risk review. Stateless."""

    def __init__(self, *, model: str | None = None, temperature: float = 0.2) -> None:
        self.model = model
        self.temperature = temperature

    def assess(
        self,
        hypothesis_title: str,
        positions: pd.Series,
        returns: pd.Series,
        volatility: pd.Series | None = None,
    ) -> RiskAssessment:
        positions = positions.dropna()
        returns = returns.dropna()
        position_summary = {
            "mean_abs_position": float(positions.abs().mean()),
            "max_abs_position": float(positions.abs().max()),
            "fraction_at_max": float((positions.abs() >= positions.abs().quantile(0.99)).mean()),
            "concentration_share": float(positions.abs().max() / max(positions.abs().sum(), 1e-9)),
        }
        regime_summary = _regime_breakdown(returns, volatility)

        user_content = f"""Hypothesis: {hypothesis_title}

Position statistics:
{_pretty(position_summary)}

Returns by volatility regime:
{regime_summary}

Output JSON only."""

        response = call_claude(
            system=SYSTEM_PROMPT,
            messages=[AgentMessage(role="user", content=user_content)],
            model=self.model,
            temperature=self.temperature,
            max_tokens=512,
        )
        payload = extract_first_json(response.text)
        return RiskAssessment(
            risk_score=int(payload.get("risk_score", 5)),
            concerns=[str(c) for c in payload.get("concerns", [])],
            size_recommendation=float(payload.get("size_recommendation", 0.5)),
        )


def _pretty(d: dict[str, float]) -> str:
    return "\n".join(f"  {k}: {v:.4f}" for k, v in d.items())


def _regime_breakdown(returns: pd.Series, volatility: pd.Series | None) -> str:
    if volatility is None:
        volatility = returns.rolling(21, min_periods=21).std()
    aligned_returns, vol = returns.align(volatility, join="inner")
    valid = vol.dropna()
    if valid.empty:
        return "  (insufficient data)"
    quantiles = valid.quantile([1 / 3, 2 / 3])
    low, high = quantiles.iloc[0], quantiles.iloc[1]
    buckets = pd.cut(vol, bins=[-np.inf, low, high, np.inf], labels=["low_vol", "mid_vol", "high_vol"])
    grouped = aligned_returns.groupby(buckets, observed=True)
    rows = []
    for label, slice_ in grouped:
        rows.append(f"  {label}: mean={slice_.mean():+.5f}, sharpe={_safe_sharpe(slice_):+.2f}, n={len(slice_)}")
    return "\n".join(rows)


def _safe_sharpe(series: pd.Series) -> float:
    std = series.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(series.mean() / std * np.sqrt(252))
