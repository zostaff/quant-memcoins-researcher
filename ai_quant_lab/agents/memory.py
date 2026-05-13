"""Research memory: SQLite-backed log of every trial.

The deflated Sharpe gate needs an honest count of how many strategies have
been tested. This module is that source of truth. Every hypothesis ever
proposed — accepted, rejected, killed mid-loop — gets a row.

The schema is intentionally narrow. The point is auditability, not data lake.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class TrialRecord:
    """One trial: a single hypothesis × backtest evaluation.

    `returns_json` carries the realized return series for accepted trials.
    Stored as compact JSON {"index": [...], "values": [...]}. Empty for
    rejected trials — we don't need their curves and they take space.
    """

    hypothesis_id: str
    hypothesis_text: str
    rationale: str
    code: str
    metrics: dict[str, float]
    accepted: bool
    rejection_reason: str | None = None
    n_trials_at_time: int = 0
    iteration: int = 0
    returns_json: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT NOT NULL,
    hypothesis_text TEXT NOT NULL,
    rationale TEXT,
    code TEXT,
    metrics_json TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    rejection_reason TEXT,
    n_trials_at_time INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    returns_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trials_accepted ON trials(accepted);
CREATE INDEX IF NOT EXISTS idx_trials_iteration ON trials(iteration);
"""


class ResearchMemory:
    """Persistent store of every research trial.

    Use `record()` after each hypothesis is evaluated; use `n_trials()` to
    feed the deflated-Sharpe gate; use `survivors()` to inspect the accepted set.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        for statement in _SCHEMA.strip().split(";"):
            if statement.strip():
                self._conn.execute(statement)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that were introduced after the initial schema.

        SQLite has no IF NOT EXISTS for ALTER TABLE; we check pragma_table_info.
        """
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(trials)")}
        if "returns_json" not in existing:
            self._conn.execute("ALTER TABLE trials ADD COLUMN returns_json TEXT")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ResearchMemory:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def record(self, trial: TrialRecord) -> int:
        """Insert a trial. Returns the new row id."""
        cursor = self._conn.execute(
            """
            INSERT INTO trials (
                hypothesis_id, hypothesis_text, rationale, code,
                metrics_json, accepted, rejection_reason,
                n_trials_at_time, iteration, returns_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trial.hypothesis_id,
                trial.hypothesis_text,
                trial.rationale,
                trial.code,
                json.dumps(trial.metrics),
                int(trial.accepted),
                trial.rejection_reason,
                trial.n_trials_at_time,
                trial.iteration,
                trial.returns_json,
                trial.created_at,
            ),
        )
        row_id = cursor.lastrowid
        return int(row_id) if row_id is not None else -1

    def accepted_returns(self) -> list[pd.Series]:
        """Reconstruct the return series of every accepted trial.

        Used by `evaluate_gates` for the correlation gate. Survives process
        restarts and shared-memory scenarios since everything is persisted.
        """
        rows = self._conn.execute(
            "SELECT hypothesis_id, returns_json FROM trials WHERE accepted=1 ORDER BY id"
        ).fetchall()
        series_list: list[pd.Series] = []
        for row in rows:
            payload = row["returns_json"]
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict) or "index" not in data or "values" not in data:
                continue
            try:
                idx = pd.to_datetime(data["index"])
            except (TypeError, ValueError):
                idx = pd.Index(data["index"])
            series_list.append(pd.Series(data["values"], index=idx, name=row["hypothesis_id"]))
        return series_list

    def n_trials(self) -> int:
        """How many trials we've recorded. Feeds deflated_sharpe(n_trials=...)."""
        row = self._conn.execute("SELECT COUNT(*) AS c FROM trials").fetchone()
        return int(row["c"]) if row else 0

    def survivors(self) -> list[TrialRecord]:
        """All accepted trials. Used to check correlation against new candidates."""
        rows = self._conn.execute("SELECT * FROM trials WHERE accepted=1 ORDER BY id").fetchall()
        return [self._row_to_trial(row) for row in rows]

    def history(self, limit: int | None = None) -> list[TrialRecord]:
        """Recent trials in insertion order; limit applies from the end."""
        if limit is not None:
            rows = self._conn.execute(
                "SELECT * FROM (SELECT * FROM trials ORDER BY id DESC LIMIT ?) ORDER BY id",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM trials ORDER BY id").fetchall()
        return [self._row_to_trial(row) for row in rows]

    def summarize_for_prompt(self, limit: int = 10) -> str:
        """Compact text summary of recent trials, suitable for an LLM prompt.

        Keeps each line under ~150 chars so a window of 10 fits cheaply.
        """
        recent = self.history(limit=limit)
        if not recent:
            return "(no prior trials)"
        lines = []
        for trial in recent:
            sharpe = trial.metrics.get("sharpe_ratio", 0.0)
            verdict = "ACCEPT" if trial.accepted else f"REJECT ({trial.rejection_reason or 'unspecified'})"
            lines.append(
                f"- {trial.hypothesis_id}: SR={sharpe:+.2f} → {verdict} | {trial.hypothesis_text[:80]}"
            )
        return "\n".join(lines)

    @staticmethod
    def _row_to_trial(row: sqlite3.Row) -> TrialRecord:
        keys = row.keys() if hasattr(row, "keys") else []
        return TrialRecord(
            hypothesis_id=row["hypothesis_id"],
            hypothesis_text=row["hypothesis_text"],
            rationale=row["rationale"] or "",
            code=row["code"] or "",
            metrics=json.loads(row["metrics_json"]),
            accepted=bool(row["accepted"]),
            rejection_reason=row["rejection_reason"],
            n_trials_at_time=int(row["n_trials_at_time"]),
            iteration=int(row["iteration"]),
            returns_json=(row["returns_json"] if "returns_json" in keys else "") or "",
            created_at=row["created_at"],
        )

    def to_dicts(self) -> list[dict[str, Any]]:
        """Dump everything as dicts for inspection / export."""
        return [asdict(t) for t in self.history()]
