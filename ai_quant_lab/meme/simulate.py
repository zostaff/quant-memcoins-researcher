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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_quant_lab.meme.panel import TokenPanel


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
    seed: int = 7


def simulate_universe(spec: UniverseSpec | None = None) -> TokenPanel:
    spec = spec or UniverseSpec()
    if not 0.0 <= spec.edge <= 1.0:
        raise ValueError("edge must be in [0, 1]")
    rng = np.random.default_rng(spec.seed)

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
            heat = 3.0 if hot else 1.0
            close[t, j] = price
            liquidity[t, j] = liq
            volume[t, j] = liq * 0.05 * heat * float(np.exp(rng.normal(0, 0.5)))
            n_holders += holder_rate * volume[t, j] / 500.0 * float(rng.uniform(0.5, 1.5))
            holders[t, j] = n_holders

    columns = pd.Index([f"TOKEN{j:04d}" for j in range(n_tokens)], name="token")
    return TokenPanel(
        close=pd.DataFrame(close, index=index, columns=columns),
        liquidity=pd.DataFrame(liquidity, index=index, columns=columns),
        volume=pd.DataFrame(volume, index=index, columns=columns),
        holders=pd.DataFrame(holders, index=index, columns=columns),
        launch_time=pd.Series(index[launch_bars], index=columns, name="launch_time"),
    )
