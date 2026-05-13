"""Decision log: every order with enough context to audit later.

Each entry is a frozen dataclass with the strategy id, intended position,
realized fill, prices, and a free-form reason field for the LLM-side
rationale. Persisted as JSONL.

You can't debug a blow-up without this. Print is not a logging strategy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DecisionRecord:
    """One trading decision and its outcome."""

    timestamp: str
    strategy_id: str
    symbol: str
    intent_position: float
    realized_position: float
    intent_price: float
    realized_price: float
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionLog:
    """Append-only JSONL log of decisions.

    Designed for single-process use. For multi-process or distributed setups,
    wrap with a queue or use a real broker for persistence.
    """

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, record: DecisionRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), default=str) + "\n")

    def record(
        self,
        *,
        strategy_id: str,
        symbol: str,
        intent_position: float,
        realized_position: float,
        intent_price: float,
        realized_price: float,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DecisionRecord:
        """Build and append in one call."""
        record = DecisionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy_id=strategy_id,
            symbol=symbol,
            intent_position=intent_position,
            realized_position=realized_position,
            intent_price=intent_price,
            realized_price=realized_price,
            reason=reason,
            metadata=dict(metadata or {}),
        )
        self.append(record)
        return record

    def read(self) -> list[DecisionRecord]:
        """Read all records back. Useful for tests and offline analysis."""
        records: list[DecisionRecord] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                records.append(DecisionRecord(**data))
        return records
