"""Synthetic memecoin universe, so everything runs without a data vendor.

The generator is not a forecast of the real market. It reproduces the
features that break naive backtests:

    - tokens launch continuously and most die: by rug (price ~ -98% in a bar,
      then gone) or by liquidity bleeding out below the delisting floor;
    - a shared attention cycle moves every token at once (the MKT factor);
    - volatility of several percent per 5-minute bar;
    - pool TVL follows the constant-product rule (∝ sqrt(price)).

Each token has a hidden quality that always drives how fast it gains
holders. `edge` decides whether that quality is also worth money: it adds a
quality-proportional drift and makes hot/cold regimes persistent (momentum).
At edge=0 the holder signal is still visible but carries no return, which is
the null tape for checking that the validation gates kill what they should.

Narratives work the same way. Every token gets a theme and a name to match
(an AI ticker, a dog coin, ...), and each narrative has its own slow heat
cycle. `narrative_attention` makes the heat visible: hot narratives get more
launches and more volume. `narrative_strength` makes it worth money: hot
narratives drift up. Attention without strength is the null tape for
narrative rotation: the story is visible and still pays nothing.

Names, narratives and the buy/sell ratio draw from their own random stream,
so turning them on or off never changes the prices of the base tape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_quant_lab.meme.narratives import NARRATIVES, OTHER
from ai_quant_lab.meme.panel import TokenPanel

_THEMES = (*NARRATIVES, OTHER)
_SYLLABLES = ("ba", "ko", "ri", "zu", "mo", "ta", "ne", "pi", "lo", "xa", "vu", "do", "ki", "sha")


@dataclass(frozen=True)
class UniverseSpec:
    n_days: int = 7
    bar_minutes: int = 5
    launches_per_hour: float = 2.0
    rug_probability: float = 0.30
    mean_bars_to_rug: float = 8.0
    median_launch_liquidity_usd: float = 20_000.0
    delist_liquidity_usd: float = 500.0
    bar_volatility: float = 0.05
    regime_drift: float = 0.006
    bleed_per_bar: float = 0.0005
    market_beta: float = 1.0
    quality_drift: float = 0.0015
    edge: float = 0.0
    narrative_attention: float = 0.0
    narrative_strength: float = 0.0
    seed: int = 7


def simulate_universe(spec: UniverseSpec | None = None) -> TokenPanel:
    spec = spec or UniverseSpec()
    if not 0.0 <= spec.edge <= 1.0:
        raise ValueError("edge must be in [0, 1]")
    rng = np.random.default_rng(spec.seed)
    side = np.random.default_rng([spec.seed, 7919])  # narratives, names, flow

    bars_per_hour = 60 / spec.bar_minutes
    n_bars = int(spec.n_days * 24 * bars_per_hour)
    index = pd.date_range("2026-01-01", periods=n_bars, freq=f"{spec.bar_minutes}min")

    # Shared attention: slow AR(1), scaled to a per-bar drift.
    attention = np.zeros(n_bars)
    for t in range(1, n_bars):
        attention[t] = 0.995 * attention[t - 1] + rng.normal(0, 0.1)
    market_drift = spec.market_beta * 0.0008 * attention / (attention.std() or 1.0)

    n_tokens = rng.poisson(spec.launches_per_hour * spec.n_days * 24)
    launch_bars = np.sort(rng.integers(0, n_bars - 12, size=n_tokens))
    p_stay = 0.5 + 0.45 * spec.edge

    # Narrative heat: one slow AR(1) per theme ("other" stays cold), unit variance.
    n_themes = len(_THEMES)
    heat = np.zeros((n_themes, n_bars))
    shocks = side.normal(0, 1, size=(n_themes, n_bars))
    for t in range(1, n_bars):
        heat[:, t] = 0.997 * heat[:, t - 1] + 0.077 * shocks[:, t]
    heat[-1] = 0.0
    base_weight = np.full(n_themes, 0.65 / (n_themes - 1))
    base_weight[-1] = 0.35
    theme = np.empty(n_tokens, dtype=int)
    for j, start in enumerate(launch_bars):
        w = base_weight * np.exp(spec.narrative_attention * 1.2 * heat[:, start])
        theme[j] = side.choice(n_themes, p=w / w.sum())
    hot_state = np.zeros((n_bars, n_tokens), dtype=bool)

    close = np.full((n_bars, n_tokens), np.nan)
    liquidity = np.zeros((n_bars, n_tokens))
    volume = np.zeros((n_bars, n_tokens))
    holders = np.full((n_bars, n_tokens), np.nan)

    for j, start in enumerate(launch_bars):
        rug_at = None
        if rng.random() < spec.rug_probability:
            rug_at = start + 1 + int(rng.exponential(spec.mean_bars_to_rug))
        price = 1e-5 * float(np.exp(rng.normal(0, 0.5)))
        liq = spec.median_launch_liquidity_usd * float(np.exp(rng.normal(0, 0.8)))
        hot = rng.random() < 0.5
        n_holders = float(rng.integers(20, 80))
        quality = rng.normal()
        token_drift = spec.edge * spec.quality_drift * quality
        holder_rate = float(np.exp(0.7 * quality))

        for t in range(start, n_bars):
            if rug_at is not None and t == rug_at:
                close[t, j] = price * 0.02
                liquidity[t, j] = 50.0
                volume[t, j] = liq * 0.5
                holders[t, j] = n_holders
                break
            if t > start:
                hot = hot if rng.random() < p_stay else not hot
                drift = (spec.regime_drift if hot else -spec.regime_drift) - spec.bleed_per_bar + token_drift
                if spec.narrative_strength:
                    drift += spec.narrative_strength * 0.002 * heat[theme[j], t]
                # -σ²/2 makes the arithmetic mean return equal `drift`. Without it a
                # rebalanced basket harvests +σ²/2 per bar out of thin air.
                log_ret = (
                    drift + market_drift[t] - 0.5 * spec.bar_volatility**2
                    + rng.normal(0, spec.bar_volatility)
                )
                price *= float(np.exp(log_ret))
                liq *= float(np.exp(0.5 * log_ret + rng.normal(-0.002, 0.01)))
            if liq < spec.delist_liquidity_usd:
                break
            heat_mult = 3.0 if hot else 1.0
            close[t, j] = price
            liquidity[t, j] = liq
            volume[t, j] = liq * 0.05 * heat_mult * float(np.exp(rng.normal(0, 0.5)))
            hot_state[t, j] = hot
            n_holders += holder_rate * volume[t, j] / 500.0 * float(rng.uniform(0.5, 1.5))
            holders[t, j] = n_holders

    if spec.narrative_attention:
        volume *= np.exp(spec.narrative_attention * 0.8 * heat[theme].T)

    # Share of buys in recent transactions: tilted by the token's hot/cold state.
    buy_ratio = np.clip(0.5 + 0.06 * (2 * hot_state - 1) + side.normal(0, 0.08, size=close.shape), 0.0, 1.0)
    buy_ratio[np.isnan(close)] = np.nan

    columns = pd.Index([f"TOKEN{j:04d}" for j in range(n_tokens)], name="token")
    names, symbols = zip(*(_token_name(_THEMES[k], side) for k in theme), strict=True) if n_tokens else ((), ())
    meta = pd.DataFrame(
        {"name": names, "symbol": symbols, "network": "synthetic", "dex": "sim", "theme": [_THEMES[k] for k in theme]},
        index=columns,
    )
    return TokenPanel(
        close=pd.DataFrame(close, index=index, columns=columns),
        liquidity=pd.DataFrame(liquidity, index=index, columns=columns),
        volume=pd.DataFrame(volume, index=index, columns=columns),
        holders=pd.DataFrame(holders, index=index, columns=columns),
        launch_time=pd.Series(index[launch_bars], index=columns, name="launch_time"),
        extra={"buy_ratio": pd.DataFrame(buy_ratio, index=index, columns=columns)},
        meta=meta,
    )


def _token_name(theme: str, rng: np.random.Generator) -> tuple[str, str]:
    """A ticker that reads like its narrative: 'Grok Mo' / 'GROKMO' for an AI coin."""
    tail = "".join(rng.choice(_SYLLABLES, size=int(rng.integers(1, 3))))
    if theme == OTHER:
        word = "".join(rng.choice(_SYLLABLES, size=int(rng.integers(2, 4))))
    else:
        word = str(rng.choice([k for k in NARRATIVES[theme] if len(k) >= 3]))
    return f"{word.title()} {tail.title()}", (word + tail).upper()[:10]
