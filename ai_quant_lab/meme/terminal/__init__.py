"""Memecoin quant terminal: four live windows, library signals and paper trading.

    python -m ai_quant_lab.meme terminal            # http://127.0.0.1:8787

Each window shows one pool: candles, flow, holders and contract flags, the
library's universe screen, what each strategy reads on it (with the zoo's
verdict for that strategy) and what a ticket would really cost to trade.
Trades go to a paper book that charges the backtest's execution costs.
"""

from ai_quant_lab.meme.terminal.live import LiveData, PoolCard
from ai_quant_lab.meme.terminal.paper import PaperBook, quote
from ai_quant_lab.meme.terminal.server import Terminal, load_verdicts, serve
from ai_quant_lab.meme.terminal.signals import Pulse, Signal, screen, token_signals

__all__ = [
    "LiveData",
    "PaperBook",
    "PoolCard",
    "Pulse",
    "Signal",
    "Terminal",
    "load_verdicts",
    "quote",
    "screen",
    "serve",
    "token_signals",
]
