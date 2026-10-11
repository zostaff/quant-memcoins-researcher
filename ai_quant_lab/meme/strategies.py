"""Strategy library: the ideas a memecoin desk actually tries, as code.

Each entry is a family with a thesis, a parameter grid and the data it needs.
They are starting points to be killed, not recommendations: run them through
`ai_quant_lab.meme.zoo`, which counts every variant as a trial and judges the
best of each family net of costs, against the meme factors, and deflated by
the total number of things tried.

Families
    momentum   holder_momentum, flow_momentum, quality_composite
    event      volume_breakout, curve_graduation, trending_entry
    reversal   dip_reversal
    flow       buy_pressure
    narrative  narrative_rotation, narrative_momentum, narrative_early
    sentiment  regime_momentum

Every weight schedule is long-only, uses only data up to bar t at bar t (the
backtest adds the execution lag), and trades only tokens that pass the
universe filter: enough liquidity, old enough to have a history, and the rug
screen when contract data was collected.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from ai_quant_lab.meme import features as F
from ai_quant_lab.meme import narratives as N
from ai_quant_lab.meme.panel import TokenPanel
from ai_quant_lab.meme.sentiment import market_mood, regime, risk_appetite


class StrategyContext:
    """Per-panel cache of the expensive shared inputs (labels, heat, mood)."""

    def __init__(self, panel: TokenPanel, *, min_liquidity_usd: float = 10_000.0, min_age_bars: int = 36):
        self.panel = panel
        self.min_liquidity_usd = min_liquidity_usd
        self.min_age_bars = min_age_bars
        self._cache: dict[tuple[Any, ...], pd.DataFrame | pd.Series] = {}

    def has(self, field_name: str) -> bool:
        if field_name == "holders":
            return self.panel.holders is not None
        if field_name == "meta":
            return self.panel.meta is not None
        return field_name in self.panel.extra

    @cached_property
    def universe(self) -> pd.DataFrame:
        p = self.panel
        mask = F.tradeable(p.close, p.liquidity, self.min_liquidity_usd)
        mask &= F.age_bars(p.close) >= self.min_age_bars
        screen = F.rug_screen(p.extra.get("mint_risk"), p.extra.get("top10_share"), unknown_ok=True)
        if screen is not None:
            mask &= screen
        if "honeypot" in p.extra:
            mask &= p.extra["honeypot"].fillna(0) != 1
        return mask

    @cached_property
    def labels(self) -> pd.Series:
        return N.narrative_labels(self.panel)

    def memo(self, key: tuple[Any, ...], make: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def heat(self, fast: int) -> pd.DataFrame:
        return self.memo(("heat", fast), lambda: N.narrative_heat(self.panel, self.labels, fast=fast))

    def narrative_return(self, bars: int) -> pd.DataFrame:
        return self.memo(("nret", bars), lambda: N.narrative_return(self.panel, self.labels, bars))

    def saturation(self) -> pd.DataFrame:
        return self.memo(("sat",), lambda: N.launch_saturation(self.panel, self.labels))

    def risk(self) -> pd.Series:
        return self.memo(("risk",), lambda: risk_appetite(mood=market_mood(self.panel)))


@dataclass(frozen=True)
class Strategy:
    name: str
    family: str
    thesis: str
    build: Callable[..., pd.DataFrame]
    grid: Mapping[str, Sequence[Any]]
    requires: tuple[str, ...] = ()
    notes: str = ""

    def missing(self, ctx: StrategyContext) -> list[str]:
        return [r for r in self.requires if not ctx.has(r)]

    def variants(self) -> list[dict[str, Any]]:
        keys = list(self.grid)
        return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*self.grid.values())]

    def weights(self, panel: TokenPanel, ctx: StrategyContext | None = None, **params: Any) -> pd.DataFrame:
        ctx = ctx or StrategyContext(panel)
        if ctx.panel is not panel:
            raise ValueError("Context was built for a different panel")
        missing = self.missing(ctx)
        if missing:
            raise ValueError(f"{self.name} needs {', '.join(missing)}, which this panel does not have")
        return self.build(ctx, **params)


# ---------------------------------------------------------------- helpers


def _every(weights: pd.DataFrame, bars: int) -> pd.DataFrame:
    """Rebalance only every `bars` bars; hold the schedule in between."""
    return weights.iloc[::bars].reindex(weights.index).ffill().fillna(0.0)


def _top(signal: pd.DataFrame, n: int, mask: pd.DataFrame) -> pd.DataFrame:
    return F.top_n_equal_weight(signal, n, mask)


def _held(entries: pd.DataFrame, hold: int, score: pd.DataFrame, n: int, mask: pd.DataFrame) -> pd.DataFrame:
    """Hold each entry for `hold` bars; if more than `n` are open, keep the best `score`."""
    open_ = F.hold_for(entries.fillna(False), hold) & mask
    best = score.where(entries).rolling(hold, min_periods=1).max()
    return _top(best, n, open_)


def _in_top_narratives(ctx: StrategyContext, narrative_score: pd.DataFrame, k: int) -> pd.DataFrame:
    """Token mask: belongs to one of the `k` best-scoring narratives (score > 0) at bar t."""
    score = narrative_score.drop(columns=[N.OTHER], errors="ignore")
    ranks = score.where(score > 0).rank(axis=1, ascending=False, method="first")
    chosen = (ranks <= k).astype(float)
    return N.to_tokens(chosen, ctx.labels, ctx.panel.tokens).fillna(0.0) > 0


# ---------------------------------------------------------------- families


def _holder_momentum(ctx: StrategyContext, *, lookback: int, top_n: int, every: int = 72) -> pd.DataFrame:
    signal = F.holder_growth(ctx.panel.holders, lookback)
    return _every(_top(signal, top_n, ctx.universe), every)


def _flow_momentum(ctx: StrategyContext, *, lookback: int, top_n: int, every: int = 36) -> pd.DataFrame:
    signal = F.liquidity_growth(ctx.panel.liquidity, lookback)
    return _every(_top(signal, top_n, ctx.universe), every)


def _quality_composite(ctx: StrategyContext, *, top_n: int, every: int = 72) -> pd.DataFrame:
    p, mask = ctx.panel, ctx.universe
    parts = [F.cross_rank(F.liquidity_growth(p.liquidity, 72), mask)]
    if p.holders is not None:
        parts.append(F.cross_rank(F.holder_growth(p.holders, 144), mask))
    if "buy_ratio" in p.extra:
        parts.append(F.cross_rank(p.extra["buy_ratio"].rolling(12, min_periods=3).mean(), mask))
    if "top10_share" in p.extra:
        parts.append(F.cross_rank(-p.extra["top10_share"], mask))
    score = sum(parts) / len(parts)
    return _every(_top(score, top_n, mask), every)


def _volume_breakout(ctx: StrategyContext, *, z: float, hold: int, top_n: int = 12) -> pd.DataFrame:
    p, mask = ctx.panel, ctx.universe
    zscore = F.volume_zscore(p.volume, 48)
    entries = (zscore > z) & (F.trailing_return(p.close, 6) > 0) & mask
    return _every(_held(entries, hold, zscore, top_n, mask), 12)


def _curve_graduation(ctx: StrategyContext, *, low: float, top_n: int, every: int = 12) -> pd.DataFrame:
    grad = ctx.panel.extra["graduation"]
    mask = ctx.universe & F.graduation_window(grad, low, 1.0)
    return _every(_top(grad, top_n, mask), every)


def _trending_entry(ctx: StrategyContext, *, max_rank: int, hold: int, top_n: int = 10) -> pd.DataFrame:
    rank = ctx.panel.extra["trending_rank"]
    listed = rank <= max_rank
    entries = listed & ~listed.shift(1, fill_value=False) & ctx.universe
    return _every(_held(entries, hold, -rank, top_n, ctx.universe), 6)


def _dip_reversal(ctx: StrategyContext, *, depth: float, hold: int, top_n: int = 12) -> pd.DataFrame:
    p, mask = ctx.panel, ctx.universe
    dd = F.drawdown_from_high(p.close)
    entries = (dd <= -depth) & (dd > -depth - 0.3) & (F.liquidity_growth(p.liquidity, 12) > 0) & mask
    return _every(_held(entries, hold, -dd, top_n, mask), 12)


def _buy_pressure(ctx: StrategyContext, *, threshold: float, top_n: int, every: int = 36) -> pd.DataFrame:
    ratio = ctx.panel.extra["buy_ratio"].rolling(12, min_periods=3).mean()
    mask = ctx.universe & (ratio > threshold)
    return _every(_top(ratio, top_n, mask), every)


def _narrative_rotation(ctx: StrategyContext, *, k: int, fast: int, top_n: int, every: int = 36) -> pd.DataFrame:
    p = ctx.panel
    mask = ctx.universe & _in_top_narratives(ctx, ctx.heat(fast), k)
    leaders = p.volume.fillna(0.0).rolling(fast, min_periods=1).sum()
    return _every(_top(leaders, top_n, mask), every)


def _narrative_momentum(ctx: StrategyContext, *, lookback: int, k: int, top_n: int = 8, every: int = 36) -> pd.DataFrame:
    p = ctx.panel
    mask = ctx.universe & _in_top_narratives(ctx, ctx.narrative_return(lookback), k)
    signal = F.trailing_return(p.close, min(lookback, 36))
    return _every(_top(signal, top_n, mask), every)


def _narrative_early(ctx: StrategyContext, *, max_saturation: float, top_n: int, every: int = 36) -> pd.DataFrame:
    p = ctx.panel
    heat, sat = ctx.heat(36), ctx.saturation()
    early = heat.where((heat > 0) & (sat.reindex_like(heat) < max_saturation), 0.0)
    mask = ctx.universe & _in_top_narratives(ctx, early, 2)
    signal = F.liquidity_growth(p.liquidity, 36)
    return _every(_top(signal, top_n, mask), every)


def _regime_momentum(ctx: StrategyContext, *, lookback: int, threshold: float, top_n: int, every: int = 36) -> pd.DataFrame:
    p = ctx.panel
    on = regime(ctx.risk(), threshold).reindex(p.close.index).fillna(False)
    weights = _top(F.trailing_return(p.close, lookback), top_n, ctx.universe)
    return _every(weights.mul(on.astype(float), axis=0), every)


STRATEGIES: dict[str, Strategy] = {s.name: s for s in (
    Strategy(
        "holder_momentum", "momentum",
        "Tokens adding holders fastest keep attracting buyers; the crowd is the product.",
        _holder_momentum, {"lookback": (72, 144, 288), "top_n": (8, 16, 24)}, ("holders",),
    ),
    Strategy(
        "flow_momentum", "momentum",
        "Money flowing into a pool (TVL growth) leads price: liquidity providers and buyers arrive first.",
        _flow_momentum, {"lookback": (36, 72, 144), "top_n": (8, 16)},
    ),
    Strategy(
        "quality_composite", "momentum",
        "Blend of holder growth, liquidity flow, buy pressure and low concentration: no single signal, a robust rank.",
        _quality_composite, {"top_n": (8, 16, 24)},
    ),
    Strategy(
        "volume_breakout", "event",
        "An abnormal volume spike with a rising price marks new attention; ride the first hours of it.",
        _volume_breakout, {"z": (2.0, 3.0), "hold": (36, 144)},
    ),
    Strategy(
        "curve_graduation", "event",
        "A bonding curve close to completion is about to migrate to a real pool and a wider audience.",
        _curve_graduation, {"low": (0.7, 0.85), "top_n": (8, 16)}, ("graduation",),
        notes="pump.fun and Pons only; needs collected token snapshots.",
    ),
    Strategy(
        "trending_entry", "event",
        "Entering the GeckoTerminal trending list puts a token in front of every screener user.",
        _trending_entry, {"max_rank": (5, 10), "hold": (36, 144)}, ("trending_rank",),
        notes="Real data only; ranks are used only for pools already in the universe.",
    ),
    Strategy(
        "dip_reversal", "reversal",
        "A sharp drawdown while liquidity still grows is a shakeout, not a rug; buy the panic.",
        _dip_reversal, {"depth": (0.3, 0.55), "hold": (36, 144)},
    ),
    Strategy(
        "buy_pressure", "flow",
        "A persistently higher share of buys than sells means the order flow is one-sided.",
        _buy_pressure, {"threshold": (0.55, 0.6), "top_n": (8, 16)}, ("buy_ratio",),
    ),
    Strategy(
        "narrative_rotation", "narrative",
        "Attention rotates between stories; own the leaders of the narratives gaining share of volume.",
        _narrative_rotation, {"k": (1, 2), "fast": (12, 36), "top_n": (6, 12)}, ("meta",),
    ),
    Strategy(
        "narrative_momentum", "narrative",
        "Narratives that have performed keep performing for a while (sector momentum).",
        _narrative_momentum, {"lookback": (36, 144), "k": (1, 2)}, ("meta",),
    ),
    Strategy(
        "narrative_early", "narrative",
        "Be early, not crowded: hot narratives where copycat launches have not exploded yet.",
        _narrative_early, {"max_saturation": (1.5, 3.0), "top_n": (8, 16)}, ("meta",),
    ),
    Strategy(
        "regime_momentum", "sentiment",
        "Momentum only pays when the crowd is risk-on; step aside when breadth, launches and flow turn.",
        _regime_momentum, {"lookback": (36, 144), "threshold": (0.0, 0.5), "top_n": (8, 16)},
    ),
)}

FAMILIES: tuple[str, ...] = tuple(dict.fromkeys(s.family for s in STRATEGIES.values()))


@dataclass(frozen=True)
class Catalog:
    """What a panel can run: available strategies and why the others cannot."""

    available: list[Strategy] = field(default_factory=list)
    unavailable: dict[str, list[str]] = field(default_factory=dict)


def catalog(ctx: StrategyContext, names: Sequence[str] | None = None) -> Catalog:
    chosen = [STRATEGIES[n] for n in names] if names else list(STRATEGIES.values())
    out = Catalog()
    for strategy in chosen:
        missing = strategy.missing(ctx)
        if missing:
            out.unavailable[strategy.name] = missing
        else:
            out.available.append(strategy)
    return out


def check_weights(weights: pd.DataFrame, panel: TokenPanel) -> None:
    """Shape, sign and gross checks every library strategy must pass."""
    if weights.shape != panel.close.shape:
        raise ValueError("weights shape does not match the panel")
    values = weights.to_numpy(dtype=float)
    if np.isnan(values).any() or (values < -1e-12).any() or (values.sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("weights must be finite, non-negative and sum to at most 1")
