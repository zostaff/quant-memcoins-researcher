"""HypothesisAgent: propose a new strategy.

Given a market description and a summary of prior trials, returns a structured
hypothesis (rationale + concise spec). The spec is small on purpose — Code-
Agent fills in the details. This separation lets us critique the *idea* before
spending tokens turning it into code.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_quant_lab.agents.base import AgentMessage, call_claude, extract_first_json


SYSTEM_PROMPT = """You are a quantitative researcher proposing trading strategies.

Your job is NOT to be creative. Your job is to propose strategies grounded in
known market microstructure: momentum, mean reversion, volatility carry, term
structure, cross-sectional dispersion. You may combine these.

Rules:
1. Never propose a strategy that requires data you don't have (no fundamentals,
   no alt data, no news). Only OHLCV.
2. Be specific. "Buy when momentum is strong" is rejected. "Long when 21-day
   return is in the top decile of trailing 252-day distribution" is acceptable.
3. State the expected Sharpe range based on academic literature, honestly. If
   you don't know, say 0.3-0.8.
4. Cite the regime where you expect this to work and the regime where it will
   fail. Both are required.
5. Output VALID JSON only, with no commentary, in this schema:

{
  "hypothesis_id": "<short snake_case id>",
  "title": "<one-line description>",
  "rationale": "<2-4 sentences citing the microstructure reason>",
  "spec": {
    "signal": "<precise signal description with parameters>",
    "direction": "long" | "short" | "both",
    "holding_period_bars": <integer>,
    "rebalance_bars": <integer>,
    "position_bounds": [<low>, <high>]
  },
  "expected_sharpe_range": [<low>, <high>],
  "works_in_regime": "<regime where edge exists>",
  "breaks_in_regime": "<regime where edge dies>"
}
"""


SYSTEM_PROMPT_MEMECOIN = """You are a quantitative researcher at a fund that trades memecoins.

Treat the memecoin market the way a fund treats any market: as a cross-section
of assets with risk factors, execution costs and a capacity limit. Ground ideas
in mechanisms: attention cycles, liquidity flowing into or out of pools, holder
base growth, post-launch decay, the survivor premium of tokens that outlive
the rug window, overreaction and exhaustion after vertical moves.

Rules:
1. Data available: per-token close, pool liquidity (TVL), volume, holder
   count, top-10 wallet share of supply, pump.fun bonding-curve progress and
   whether the creator can still mint or freeze. Nothing else: no social
   data, no wallet labels, no news.
2. Long only. No shorting, no leverage.
3. Costs are brutal (several % per round trip in small pools), so prefer
   holding periods of hours to days and liquidity filters over fast rotation.
4. Be specific: windows in bars, thresholds, liquidity floor, max positions.
5. State the regime where it works and where it dies. Expected Sharpe: be honest.
6. Output VALID JSON only, same schema as below.

{
  "hypothesis_id": "<short snake_case id>",
  "title": "<one-line description>",
  "rationale": "<2-4 sentences citing the mechanism>",
  "spec": {
    "signal": "<precise signal with parameters>",
    "direction": "long",
    "holding_period_bars": <integer>,
    "rebalance_bars": <integer>,
    "position_bounds": [0, <max weight per token>]
  },
  "expected_sharpe_range": [<low>, <high>],
  "works_in_regime": "<regime where edge exists>",
  "breaks_in_regime": "<regime where edge dies>"
}
"""


@dataclass(frozen=True)
class StrategyHypothesis:
    hypothesis_id: str
    title: str
    rationale: str
    spec: dict
    expected_sharpe_range: tuple[float, float]
    works_in_regime: str
    breaks_in_regime: str


class HypothesisAgent:
    """Claude-powered hypothesis proposer."""

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float = 0.6,
        market_type: str = "generic",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._system_prompt = SYSTEM_PROMPT_MEMECOIN if market_type == "memecoin" else SYSTEM_PROMPT

    def propose(
        self,
        market_description: str,
        prior_trials_summary: str,
        *,
        avoid_correlation_with: list[str] | None = None,
    ) -> StrategyHypothesis:
        """Return a new hypothesis informed by what's already been tried.

        Args:
            market_description: e.g. "Daily bars on a basket of 50 large-cap US equities."
            prior_trials_summary: Output of `ResearchMemory.summarize_for_prompt()`.
            avoid_correlation_with: human-readable list of accepted strategies to
                steer away from. The agent will try to propose something with
                a different return profile.
        """
        avoid = ""
        if avoid_correlation_with:
            avoid = (
                "\n\nAlready accepted, look for something uncorrelated with these:\n"
                + "\n".join(f"  - {s}" for s in avoid_correlation_with)
            )
        user_content = f"""Market: {market_description}

Prior trials (most recent last):
{prior_trials_summary}
{avoid}

Propose ONE new strategy. Output JSON only, matching the schema."""
        response = call_claude(
            system=self._system_prompt,
            messages=[AgentMessage(role="user", content=user_content)],
            model=self.model,
            temperature=self.temperature,
            max_tokens=1024,
        )
        payload = extract_first_json(response.text)
        return _payload_to_hypothesis(payload)


def _payload_to_hypothesis(payload: dict) -> StrategyHypothesis:
    sharpe_low, sharpe_high = payload["expected_sharpe_range"]
    return StrategyHypothesis(
        hypothesis_id=str(payload["hypothesis_id"]),
        title=str(payload["title"]),
        rationale=str(payload["rationale"]),
        spec=dict(payload["spec"]),
        expected_sharpe_range=(float(sharpe_low), float(sharpe_high)),
        works_in_regime=str(payload["works_in_regime"]),
        breaks_in_regime=str(payload["breaks_in_regime"]),
    )
