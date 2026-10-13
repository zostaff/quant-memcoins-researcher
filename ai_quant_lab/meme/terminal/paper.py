"""Paper trading with the backtest's execution costs.

Every fill pays what a real swap would: pool fee plus constant-product
slippage against the pool's current TVL, the MEV haircut and the priority
fee, from the chain's `ExecutionCosts` preset. Positions are valued twice:

    mark         tokens × last price, what a screener shows
    liquidation  what selling everything now would actually return

The gap between the two is the cost of getting out, and on a $20k pool it is
not small. The book lives in SQLite, one row per fill; positions and P&L are
rebuilt from the fills, so nothing can drift.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_quant_lab.meme.amm import (
    ExecutionCosts,
    costs_for_network,
    swap_loss_fraction,
    trade_cost_usd,
)

STARTING_CASH = 10_000.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    name TEXT PRIMARY KEY,
    starting_cash REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT NOT NULL,
    at TEXT NOT NULL,
    network TEXT NOT NULL,
    pool TEXT NOT NULL,
    symbol TEXT,
    side TEXT NOT NULL,
    usd REAL NOT NULL,
    price REAL NOT NULL,
    tokens REAL NOT NULL,
    cost_usd REAL NOT NULL,
    liquidity_usd REAL,
    strategy TEXT
);
"""


@dataclass(frozen=True)
class Quote:
    """What a ticket of `usd` costs on one leg in this pool right now."""

    usd: float
    liquidity_usd: float
    cost_usd: float
    slippage: float
    fee: float
    fixed_usd: float

    @property
    def cost_fraction(self) -> float:
        return self.cost_usd / self.usd if self.usd else 0.0

    def to_dict(self) -> dict[str, float]:
        return {"usd": self.usd, "liquidity_usd": self.liquidity_usd, "cost_usd": self.cost_usd,
                "cost_fraction": self.cost_fraction, "round_trip": min(2 * self.cost_fraction, 1.0),
                "slippage": self.slippage, "fee": self.fee, "fixed_usd": self.fixed_usd}


def quote(usd: float, liquidity_usd: float, costs: ExecutionCosts) -> Quote:
    usd = abs(float(usd))
    curve = float(swap_loss_fraction(usd, liquidity_usd, costs.swap_fee))
    total = float(trade_cost_usd(usd, liquidity_usd, costs))
    return Quote(usd=usd, liquidity_usd=liquidity_usd, cost_usd=min(total, usd),
                 slippage=max(curve - costs.swap_fee, 0.0), fee=costs.swap_fee + costs.mev_haircut,
                 fixed_usd=costs.priority_fee_usd if usd > 0 else 0.0)


class PaperBook:
    def __init__(self, path: str | Path = ":memory:", *, account: str = "default") -> None:
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(_SCHEMA)
        self.account = account
        self._lock = threading.Lock()
        self._ensure_account()

    def _ensure_account(self) -> None:
        self.conn.execute("INSERT OR IGNORE INTO accounts VALUES (?, ?, ?)",
                          (self.account, STARTING_CASH, _now()))
        self.conn.commit()

    @property
    def starting_cash(self) -> float:
        return self.conn.execute("SELECT starting_cash FROM accounts WHERE name = ?",
                                 (self.account,)).fetchone()[0]

    def reset(self, starting_cash: float = STARTING_CASH) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM fills WHERE account = ?", (self.account,))
            self.conn.execute("UPDATE accounts SET starting_cash = ?, created_at = ? WHERE name = ?",
                              (float(starting_cash), _now(), self.account))
            self.conn.commit()

    def fills(self, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM fills WHERE account = ? ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
        cur = self.conn.execute(sql, (self.account,))
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]

    def cash(self) -> float:
        flows = self.conn.execute(
            "SELECT COALESCE(SUM(CASE WHEN side = 'buy' THEN -usd ELSE usd - cost_usd END), 0) "
            "FROM fills WHERE account = ?", (self.account,)).fetchone()[0]
        return self.starting_cash + flows

    def positions(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Open positions keyed by (network, pool), average cost basis in USD."""
        book: dict[tuple[str, str], dict[str, Any]] = {}
        for f in reversed(self.fills()):
            key = (f["network"], f["pool"])
            p = book.setdefault(key, {"network": f["network"], "pool": f["pool"], "symbol": f["symbol"],
                                      "tokens": 0.0, "cost_basis": 0.0, "realized": 0.0, "costs_paid": 0.0})
            p["costs_paid"] += f["cost_usd"]
            if f["side"] == "buy":
                p["tokens"] += f["tokens"]
                p["cost_basis"] += f["usd"]
            else:
                share = f["tokens"] / p["tokens"] if p["tokens"] else 0.0
                released = p["cost_basis"] * share
                p["realized"] += f["usd"] - f["cost_usd"] - released
                p["tokens"] -= f["tokens"]
                p["cost_basis"] -= released
                if p["tokens"] <= 1e-12:
                    p["tokens"], p["cost_basis"] = 0.0, 0.0
        return book

    def buy(self, network: str, pool: str, symbol: str, usd: float, price: float, liquidity_usd: float,
            strategy: str | None = None) -> dict[str, Any]:
        usd = float(usd)
        costs = costs_for_network(network)
        if usd < costs.min_trade_usd:
            raise ValueError(f"Minimum ticket is ${costs.min_trade_usd:.0f}")
        if not price or price <= 0:
            raise ValueError("No live price for this pool")
        if (liquidity_usd or 0) < costs.min_liquidity_usd:
            raise ValueError(f"Pool TVL below ${costs.min_liquidity_usd:,.0f}: untradeable")
        with self._lock:
            if usd > self.cash() + 1e-9:
                raise ValueError(f"Not enough cash: ${self.cash():,.2f} available")
            q = quote(usd, liquidity_usd, costs)
            tokens = (usd - q.cost_usd) / price
            return self._record(network, pool, symbol, "buy", usd, price, tokens, q.cost_usd, liquidity_usd, strategy, q)

    def sell(self, network: str, pool: str, fraction: float, price: float, liquidity_usd: float) -> dict[str, Any]:
        fraction = min(max(float(fraction), 0.0), 1.0)
        costs = costs_for_network(network)
        with self._lock:
            p = self.positions().get((network, pool))
            if not p or p["tokens"] <= 0:
                raise ValueError("No open position in this pool")
            if not price or price <= 0:
                raise ValueError("No live price for this pool")
            tokens = p["tokens"] * fraction
            gross = tokens * price
            q = quote(gross, liquidity_usd or 0.0, costs)
            return self._record(network, pool, p["symbol"], "sell", gross, price, tokens, q.cost_usd,
                                liquidity_usd, None, q)

    def _record(self, network: str, pool: str, symbol: str, side: str, usd: float, price: float, tokens: float,
                cost: float, liquidity: float, strategy: str | None, q: Quote) -> dict[str, Any]:
        self.conn.execute(
            "INSERT INTO fills (account, at, network, pool, symbol, side, usd, price, tokens, cost_usd, "
            "liquidity_usd, strategy) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self.account, _now(), network, pool, symbol, side, usd, price, tokens, cost, liquidity, strategy))
        self.conn.commit()
        return {"side": side, "symbol": symbol, "usd": usd, "tokens": tokens, "price": price,
                "cost_usd": cost, "quote": q.to_dict()}

    def summary(self, marks: dict[tuple[str, str], tuple[float | None, float | None]]) -> dict[str, Any]:
        """Account value given `marks[(network, pool)] = (price, liquidity_usd)`."""
        cash = self.cash()
        rows, mark_total, liq_total = [], 0.0, 0.0
        realized = costs_paid = 0.0
        for key, p in self.positions().items():
            realized += p["realized"]
            costs_paid += p["costs_paid"]
            if p["tokens"] <= 0:
                continue
            price, liquidity = marks.get(key, (None, None))
            mark = p["tokens"] * price if price else 0.0
            exit_cost = float(trade_cost_usd(mark, liquidity or 0.0, costs_for_network(p["network"]))) if mark else 0.0
            liquidation = max(mark - exit_cost, 0.0)
            mark_total += mark
            liq_total += liquidation
            rows.append({**p, "price": price, "mark": mark, "liquidation": liquidation,
                         "unrealized": liquidation - p["cost_basis"],
                         "return": liquidation / p["cost_basis"] - 1 if p["cost_basis"] else None})
        start = self.starting_cash
        return {
            "account": self.account, "starting_cash": start, "cash": cash,
            "mark_value": cash + mark_total, "liquidation_value": cash + liq_total,
            "pnl": cash + liq_total - start, "return": (cash + liq_total) / start - 1,
            "realized": realized, "costs_paid": costs_paid, "positions": rows, "fills": self.fills(25),
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
