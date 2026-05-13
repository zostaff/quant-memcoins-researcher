"""ai-quant-lab: AI-powered quant research engine.

Public API surface is intentionally small. Most users interact through:
    - `ai_quant_lab.run` — CLI entry point for the research loop
    - `ai_quant_lab.backtest` — vectorized and event-driven backtesting
    - `ai_quant_lab.validation` — deflated Sharpe, walk-forward, purged CV
    - `ai_quant_lab.agents` — Claude-powered hypothesis / code / critic agents

Design principle: generation is cheap, validation is expensive.
The whole architecture is built around making bad strategies easy to kill.
"""

from ai_quant_lab.config import Settings, settings

__version__ = "0.1.0"
__all__ = ["Settings", "settings", "__version__"]
