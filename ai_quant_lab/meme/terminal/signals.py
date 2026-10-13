"""Library strategies read for one token, plus the market pulse.

The strategies in `ai_quant_lab.meme.strategies` rank a whole panel. A
terminal window shows one pool, so each signal here asks the strategy's entry
question about that pool alone, with the same features and thresholds:

    volume_breakout     volume z-score over 48 bars > 2 with the price up over 6 bars;
                        active for the 36-bar hold after an entry
    dip_reversal        30-60% below the 24h high while buyers outnumber sellers
                        (the library asks for growing liquidity, which a single
                        pool request cannot show; buy share stands in)
    buy_pressure        buys > 55% of the last hour's transactions
    trending_entry      in the top 10 of the GeckoTerminal trending list
    curve_graduation    bonding curve 70-100% complete
    narrative_rotation  the token's narrative is one of the two gaining most share of volume
    regime_momentum     up over 36 bars while the market pulse is risk-on

Each signal carries the verdict the zoo gave its strategy (from a JSON file
written by `zoo --json`), or "untested" when no verdict is loaded. A signal is
a reading, not advice: most of these strategies have not passed the zoo.

The universe screen is the library's: TVL of at least $10k, 3 hours since
launch, no mint or freeze authority, top-10 holders at most 50%, not a
honeypot. A token that fails it gets its signals shown as blocked.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ai_quant_lab.meme import features as F
from ai_quant_lab.meme.narratives import OTHER
from ai_quant_lab.meme.sentiment import risk_appetite
from ai_quant_lab.meme.terminal.live import PoolCard

MIN_LIQUIDITY_USD = 10_000.0
MIN_AGE_HOURS = 3.0
MAX_TOP10_PCT = 50.0
HEAT_FLOOR = 0.005


@dataclass
class Signal:
    strategy: str
    family: str
    state: str            # "on", "off", "na"
    value: str
    reason: str
    verdict: str = "untested"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def screen(card: PoolCard, info: dict[str, Any]) -> list[str]:
    """Reasons the token fails the library's universe filter; empty if it passes."""
    fails = []
    if (card.liquidity_usd or 0.0) < MIN_LIQUIDITY_USD:
        fails.append(f"TVL below ${MIN_LIQUIDITY_USD:,.0f}")
    if card.age_hours is not None and card.age_hours < MIN_AGE_HOURS:
        fails.append(f"younger than {MIN_AGE_HOURS:.0f}h")
    if info.get("mint_authority") == 1:
        fails.append("mint authority on")
    if info.get("freeze_authority") == 1:
        fails.append("freeze authority on")
    if (info.get("top10_pct") or 0.0) > MAX_TOP10_PCT:
        fails.append(f"top-10 hold {info['top10_pct']:.0f}%")
    if info.get("honeypot") == 1:
        fails.append("honeypot")
    return fails


def _frame(candles: list[list[float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = np.asarray(candles, dtype=float).reshape(-1, 6)
    index = pd.to_datetime(rows[:, 0], unit="s")
    close = pd.DataFrame({"t": rows[:, 4]}, index=index)
    volume = pd.DataFrame({"t": rows[:, 5]}, index=index)
    return close, volume


def _last(frame: pd.DataFrame) -> float:
    value = frame["t"].iloc[-1] if len(frame) else float("nan")
    return float(value)


def token_signals(
    card: PoolCard,
    candles_5m: list[list[float]],
    info: dict[str, Any],
    pulse: dict[str, Any],
    trending_rank: int | None,
    graduation_pct: float | None = None,
    verdicts: dict[str, str] | None = None,
) -> list[Signal]:
    verdicts = verdicts or {}
    out: list[Signal] = []
    close, volume = _frame(candles_5m)
    enough = len(close) >= 48

    # volume_breakout
    if enough:
        z = F.volume_zscore(volume, 48)
        entries = (z > 2.0) & (F.trailing_return(close, 6) > 0)
        recent = entries["t"].iloc[-36:]
        ago = int(len(recent) - 1 - np.flatnonzero(recent.to_numpy())[-1]) if recent.any() else None
        out.append(Signal(
            "volume_breakout", "event", "on" if ago is not None else "off", f"z {_last(z):+.1f}",
            f"entry {ago * 5} min ago, held 3h" if ago is not None else "no volume spike with rising price in 3h",
        ))
    else:
        out.append(Signal("volume_breakout", "event", "na", "—", "needs 4h of 5-minute candles"))

    # dip_reversal
    share = card.buy_share("h1")
    if len(close):
        dd = _last(F.drawdown_from_high(close))
        fired = -0.6 < dd <= -0.3 and share is not None and share > 0.5
        out.append(Signal(
            "dip_reversal", "reversal", "on" if fired else "off", f"{dd:+.0%} from high",
            "deep dip and buyers outnumber sellers" if fired else "needs a 30-60% dip with buyers in control",
        ))
    else:
        out.append(Signal("dip_reversal", "reversal", "na", "—", "no candles"))

    # buy_pressure
    txs = card.buys.get("h1", 0) + card.sells.get("h1", 0)
    if share is None or txs < 20:
        out.append(Signal("buy_pressure", "flow", "na", "—", "fewer than 20 trades in the last hour"))
    else:
        out.append(Signal("buy_pressure", "flow", "on" if share > 0.55 else "off", f"{share:.0%} buys",
                          f"{card.buys['h1']} buys / {card.sells['h1']} sells in 1h"))

    # trending_entry
    on = trending_rank is not None and trending_rank <= 10
    out.append(Signal("trending_entry", "event", "on" if on else "off",
                      f"#{trending_rank}" if trending_rank else "not listed",
                      "top 10 on GeckoTerminal trending" if on else "outside the trending top 10"))

    # curve_graduation
    if graduation_pct is None:
        out.append(Signal("curve_graduation", "event", "na", "—", "no bonding curve (or already migrated)"))
    else:
        g = graduation_pct / 100.0
        out.append(Signal("curve_graduation", "event", "on" if 0.7 <= g < 1.0 else "off", f"{g:.0%} of curve",
                          "close to migrating" if 0.7 <= g < 1.0 else "curve not in the 70-100% window"))

    # narrative_rotation
    heat = pulse.get("narratives") or {}
    hot = [n for n, row in sorted(heat.items(), key=lambda kv: -kv[1]["heat"])
           if n != OTHER and row["heat"] > 0][:2]
    mine = heat.get(card.narrative, {}).get("heat")
    if card.narrative == OTHER:
        out.append(Signal("narrative_rotation", "narrative", "off", "other", "no narrative in name or symbol"))
    else:
        on = card.narrative in hot
        out.append(Signal("narrative_rotation", "narrative", "on" if on else "off",
                          f"{card.narrative} {mine:+.2f}" if mine is not None else card.narrative,
                          f"{card.narrative} is gaining share of volume" if on
                          else "narrative not among the two heating fastest"))

    # regime_momentum
    if len(close) > 36:
        ret = _last(F.trailing_return(close, 36))
        risk_on = pulse.get("regime") == "RISK-ON"
        on = risk_on and ret > 0
        out.append(Signal("regime_momentum", "sentiment", "on" if on else "off", f"{ret:+.0%} 3h",
                          "trend up and market risk-on" if on
                          else ("market not risk-on" if not risk_on else "3h trend not up")))
    else:
        out.append(Signal("regime_momentum", "sentiment", "na", "—", "needs 3h of 5-minute candles"))

    fails = screen(card, info)
    for s in out:
        s.verdict = verdicts.get(s.strategy, "untested")
        if fails and s.state == "on":
            s.state, s.reason = "blocked", "screen: " + "; ".join(fails)
    return out


# ---------------------------------------------------------------- market pulse


def _heat(cards: list[PoolCard]) -> dict[str, dict[str, float]]:
    """Each narrative's share of 1h volume against its share of 24h volume."""
    h1: dict[str, float] = {}
    h24: dict[str, float] = {}
    count: dict[str, int] = {}
    for c in cards:
        h1[c.narrative] = h1.get(c.narrative, 0.0) + (c.volume.get("h1") or 0.0)
        h24[c.narrative] = h24.get(c.narrative, 0.0) + (c.volume.get("h24") or 0.0)
        count[c.narrative] = count.get(c.narrative, 0) + 1
    t1, t24 = sum(h1.values()) or 1.0, sum(h24.values()) or 1.0
    return {
        n: {"heat": math.log((h1[n] / t1 + HEAT_FLOOR) / (h24[n] / t24 + HEAT_FLOOR)),
            "share_1h": h1[n] / t1, "tokens": count[n]}
        for n in h1
    }


def snapshot_mood(trending: list[PoolCard], new: list[PoolCard]) -> dict[str, float]:
    """Raw mood components from the live lists, same meaning as `market_mood`."""
    cards = {c.address: c for c in trending + new}.values()
    live = [c for c in cards if (c.liquidity_usd or 0) >= 5_000 and c.change.get("h1") is not None]
    breadth = sum(c.change["h1"] > 0 for c in live) / len(live) if live else float("nan")
    buys = sum(c.buys.get("h1", 0) for c in cards)
    sells = sum(c.sells.get("h1", 0) for c in cards)
    v1 = sum(c.volume.get("h1") or 0 for c in cards)
    v24 = sum(c.volume.get("h24") or 0 for c in cards)
    stamps = sorted(datetime.fromisoformat(c.created_at.replace("Z", "+00:00")) for c in new if c.created_at)
    span_h = (stamps[-1] - stamps[0]).total_seconds() / 3600 if len(stamps) > 1 else 0.0
    return {
        "breadth": breadth,
        "buy_pressure": buys / (buys + sells) if buys + sells else float("nan"),
        "volume_impulse": math.log((24 * v1 + 1) / (v24 + 1)),
        "launch_rate": (len(stamps) - 1) / span_h if span_h > 0 else float("nan"),
    }


class Pulse:
    """Market pulse per network, with a trailing history for the z-scored score.

    The first readings use fixed thresholds on the raw components; once
    `min_history` readings exist the score is `risk_appetite` on the history,
    the same composite the strategies use.
    """

    def __init__(self, *, history: int = 1_440, min_history: int = 30) -> None:
        self.min_history = min_history
        self._history: dict[str, deque[tuple[pd.Timestamp, dict[str, float]]]] = {}
        self._maxlen = history

    def update(self, network: str, trending: list[PoolCard], new: list[PoolCard],
               now: pd.Timestamp | None = None) -> dict[str, Any]:
        mood = snapshot_mood(trending, new)
        log = self._history.setdefault(network, deque(maxlen=self._maxlen))
        now = now or pd.Timestamp.now("UTC").floor("min")
        if not log or log[-1][0] < now:
            log.append((now, mood))

        score = None
        if len(log) >= self.min_history:
            frame = pd.DataFrame([m for _, m in log], index=[t for t, _ in log])
            value = risk_appetite(mood=frame, window=len(log), min_periods=self.min_history).iloc[-1]
            score = None if np.isnan(value) else float(value)
        if score is not None:
            regime = "RISK-ON" if score > 0.25 else "RISK-OFF" if score < -0.25 else "NEUTRAL"
        else:
            b, p, v = mood["breadth"], mood["buy_pressure"], mood["volume_impulse"]
            if b > 0.55 and p > 0.52 and v > 0:
                regime = "RISK-ON"
            elif b < 0.4 and p < 0.5:
                regime = "RISK-OFF"
            else:
                regime = "NEUTRAL"

        heat = _heat(list({c.address: c for c in trending + new}.values()))
        return {
            "network": network,
            "regime": regime,
            "score": score,
            "readings": len(log),
            "min_history": self.min_history,
            "components": {k: (None if np.isnan(v) else v) for k, v in mood.items()},
            "narratives": heat,
        }
