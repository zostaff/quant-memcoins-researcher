"""Claude-powered agents.

Five roles:
    HypothesisAgent — proposes new strategies given a context of past results
    CodeAgent       — turns a hypothesis into runnable strategy code
    CriticAgent     — argues against a hypothesis, adversarially
    RiskAgent       — sanity-checks position sizing and exposure
    ResearchMemory  — SQLite store of every trial; the source of truth for n_trials

All agents share the `call_claude` wrapper for retries, JSON parsing, and
prompt caching where it makes sense.
"""

from ai_quant_lab.agents.base import (
    AgentMessage,
    AgentResponse,
    call_claude,
    extract_first_json,
)
from ai_quant_lab.agents.code import CodeAgent
from ai_quant_lab.agents.critic import CriticAgent, CriticVerdict
from ai_quant_lab.agents.hypothesis import HypothesisAgent, StrategyHypothesis
from ai_quant_lab.agents.memory import ResearchMemory, TrialRecord
from ai_quant_lab.agents.risk import RiskAgent, RiskAssessment

__all__ = [
    "AgentMessage",
    "AgentResponse",
    "CodeAgent",
    "CriticAgent",
    "CriticVerdict",
    "HypothesisAgent",
    "ResearchMemory",
    "RiskAgent",
    "RiskAssessment",
    "StrategyHypothesis",
    "TrialRecord",
    "call_claude",
    "extract_first_json",
]
