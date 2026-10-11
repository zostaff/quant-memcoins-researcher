"""Market mood: is the memecoin crowd risk-on or risk-off right now?

Funds gate strategies by regime: momentum works in trending markets and
bleeds in choppy ones. On memecoins the regime is the crowd's appetite for
risk, and it leaves fingerprints in the data before it shows up in prices:

    breadth        share of tradeable tokens up over the window
    launch_rate    new launches versus the usual pace (froth)
    death_rate     share of live tokens that died in the window (fear)
    volume_impulse total volume versus its slower average
    buy_pressure   average share of buys in recent transactions (when collected)
    curve_progress average bonding-curve progress of live tokens (when collected)

`risk_appetite` z-scores each component against its own trailing history and
averages them, sign-adjusted so that higher always means more risk-on. Every
value at bar t uses only bars up to t; the z-score uses a trailing window,
never the full sample.

External sentiment (X mentions, LunarCrush, Kaito, an in-house NLP model)
plugs in through `load_external_sentiment`, which applies a publication lag
so a score is only used once it could actually have been known.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_lab.meme.features import tradeable, trailing_return
from ai_quant_lab.meme.panel import TokenPanel

# Higher = more risk-on. Death rate is fear, so it enters with a minus sign.
_SIGN = {
    "breadth": 1.0,
    "launch_rate": 1.0,
    "death_rate": -1.0,
    "volume_impulse": 1.0,
    "buy_pressure": 1.0,
    "curve_progress": 1.0,
}


def market_mood(
    panel: TokenPanel,
    *,
    bars: int = 36,
    slow: int = 864,
    min_liquidity_usd: float = 5_000.0,
) -> pd.DataFrame:
    """Raw mood components, time × component."""
    live = tradeable(panel.close, panel.liquidity, min_liquidity_usd)
    up = (trailing_return(panel.close, bars) > 0) & live
    counted = live & trailing_return(panel.close, bars).notna()
    breadth = up.sum(axis=1) / counted.sum(axis=1).replace(0, np.nan)

    first = panel.close.notna() & (panel.close.notna().cumsum() == 1)
    launches = first.sum(axis=1).astype(float)
    launch_rate = (launches.rolling(bars, min_periods=1).sum() + 1.0) / (
        launches.rolling(slow, min_periods=1).mean() * bars + 1.0
    )

    died = panel.close.shift(1).notna() & panel.close.isna()
    alive = panel.close.notna().sum(axis=1).replace(0, np.nan)
    death_rate = died.sum(axis=1).rolling(bars, min_periods=1).sum() / alive

    volume = panel.volume.fillna(0.0).sum(axis=1)
    volume_impulse = np.log(
        (volume.rolling(bars, min_periods=1).mean() + 1.0) / (volume.rolling(slow, min_periods=1).mean() + 1.0)
    )

    mood = pd.DataFrame({
        "breadth": breadth,
        "launch_rate": np.log(launch_rate),
        "death_rate": death_rate,
        "volume_impulse": volume_impulse,
    })
    if "buy_ratio" in panel.extra:
        mood["buy_pressure"] = panel.extra["buy_ratio"].where(live).mean(axis=1)
    if "graduation" in panel.extra:
        mood["curve_progress"] = panel.extra["graduation"].where(panel.close.notna()).mean(axis=1)
    return mood


def risk_appetite(
    panel: TokenPanel | None = None,
    *,
    mood: pd.DataFrame | None = None,
    window: int = 864,
    min_periods: int = 72,
    **mood_kwargs: object,
) -> pd.Series:
    """Composite risk-on score: mean of trailing z-scores of the mood components."""
    if mood is None:
        if panel is None:
            raise ValueError("Pass a panel or a precomputed mood frame")
        mood = market_mood(panel, **mood_kwargs)  # type: ignore[arg-type]
    mean = mood.rolling(window, min_periods=min_periods).mean()
    std = mood.rolling(window, min_periods=min_periods).std().replace(0.0, np.nan)
    z = (mood - mean) / std
    signs = pd.Series({c: _SIGN.get(c, 1.0) for c in mood.columns})
    return (z * signs).clip(-4, 4).mean(axis=1, skipna=True).rename("risk_appetite")


def regime(score: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Risk-on flag. NaN score (not enough history yet) counts as risk-off."""
    return (score > threshold).fillna(False).rename("risk_on")


def load_external_sentiment(
    source: str | Path | pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    key: str = "market",
    lag: str | pd.Timedelta = "5min",
    max_age: str | pd.Timedelta | None = "6h",
) -> pd.DataFrame:
    """Align an external sentiment feed to the panel's bars, point-in-time.

    `source` is a CSV path or a DataFrame with columns `time`, `score` and,
    unless `key == "market"`, a column named `key` (`token` or `narrative`)
    saying what each score is about. A score stamped at time τ is usable from
    bar τ + `lag` on: the lag covers the provider's processing and delivery
    delay, and it must not be zero for a feed that is timestamped by the
    window it describes rather than by when it was published. Scores older
    than `max_age` at a bar are treated as missing.

    Returns time × key (one column "market" for a market-wide feed).
    """
    frame = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source.copy()
    if "time" not in frame or "score" not in frame:
        raise ValueError("External sentiment needs 'time' and 'score' columns")
    if key != "market" and key not in frame:
        raise ValueError(f"External sentiment has no {key!r} column")
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None) + pd.Timedelta(lag)
    if key == "market":
        frame[key] = "market"
    bars = pd.DataFrame({"time": pd.DatetimeIndex(index).astype("datetime64[ns]")})
    tolerance = pd.Timedelta(max_age) if max_age is not None else None
    out = {}
    for name, group in frame.groupby(key):
        group = group.sort_values("time")[["time", "score"]].astype({"time": "datetime64[ns]"})
        merged = pd.merge_asof(bars, group, on="time", direction="backward", tolerance=tolerance)
        out[name] = merged["score"].to_numpy()
    return pd.DataFrame(out, index=index)
