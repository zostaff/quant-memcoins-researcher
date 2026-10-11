"""Narratives: which story the money is chasing, measured point-in-time.

Memecoins trade in themes. An AI-agent token, a dog coin and a political coin
launched in the same hour live in different markets, and attention rotates
between them: one week every new AI ticker runs, the next week the dogs do.
A fund calls this sector rotation. Here the sectors are narratives.

Labels come from the token's name and symbol only. Both are fixed on chain at
launch, so a label is known from the first bar a token exists. GeckoTerminal
categories are deliberately not used for labels: they are curated after the
fact, often after a token has already run, and labelling the past with them
leaks the future.

Every feature below is a time × narrative frame (or broadcast to time ×
token) whose value at bar t uses only bars up to t.

    narrative_volume_share  share of all traded volume over a window
    narrative_heat          short-window share over long-window share (rotation)
    narrative_return        liquidity-weighted return of the narrative, rugs at -100%
    narrative_launches      new tokens launched in the narrative over a window
    launch_saturation       launches now versus the narrative's usual pace (crowding)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ai_quant_lab.meme.panel import TokenPanel

OTHER = "other"

# Ordered: on a tie the earlier narrative wins. Short keywords (< 4 letters) only
# match a whole word or the start/end of a glued symbol ("AIDOGE", "CASHCAT").
NARRATIVES: dict[str, tuple[str, ...]] = {
    "ai": ("ai", "agent", "agents", "gpt", "llm", "neural", "bot", "robot", "cyber", "sentient",
           "agi", "claude", "grok", "openai", "deepseek", "model"),
    "politics": ("trump", "maga", "biden", "kamala", "election", "president", "melania", "vance",
                 "usa", "america", "senate"),
    "stocks": ("tsla", "nvda", "aapl", "hood", "stock", "stocks", "nasdaq", "spy", "gme", "amc",
               "robinhood", "tesla", "nvidia", "apple"),
    "dog": ("dog", "doge", "inu", "shib", "shiba", "wif", "pup", "puppy", "bonk", "woof", "corgi",
            "floki", "doggo"),
    "cat": ("cat", "cats", "kitty", "kitten", "meow", "popcat", "mew", "nyan", "purr", "neko"),
    "frog": ("pepe", "frog", "kek", "toad", "ribbit"),
    "celebrity": ("elon", "musk", "kanye", "drake", "taylor", "mrbeast", "celeb", "tate"),
    "animals": ("monkey", "ape", "penguin", "pengu", "bear", "bull", "hippo", "goat", "fish", "duck",
                "hamster", "moodeng", "panda", "rabbit", "bunny", "fox", "wolf", "horse"),
    "anime": ("anime", "waifu", "chan", "kun", "senpai", "sama", "otaku", "manga"),
    "food": ("pizza", "burger", "taco", "sushi", "banana", "coffee", "cheese", "bread", "milk",
             "donut", "cookie"),
    "gaming": ("game", "gaming", "play", "quest", "pixel", "arcade", "nft", "minecraft"),
    "space": ("moon", "rocket", "mars", "space", "nasa", "galaxy", "star", "alien", "ufo"),
    "culture": ("wojak", "chad", "based", "giga", "sigma", "npc", "meme", "doomer", "degen", "chill",
                "brainrot", "skibidi"),
}

_WORD = re.compile(r"[a-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")


def classify(name: str | None, symbol: str | None = None) -> str:
    """Narrative of a token from its name and symbol; `OTHER` when nothing matches."""
    text = " ".join(part for part in (name, symbol) if isinstance(part, str))
    if not text.strip():
        return OTHER
    words = set(_WORD.findall(_CAMEL.sub(" ", text).lower()))
    glued = [w for w in _WORD.findall(text.lower()) if len(w) > 3]
    best, best_hits = OTHER, 0
    for narrative, keywords in NARRATIVES.items():
        # Longer keywords are more specific: "AIDOGE" is a dog coin, not an AI one.
        hits = 0
        for kw in keywords:
            if kw in words:
                hits += 2 * len(kw)
            elif len(kw) >= 4 and any(kw in g for g in glued):
                hits += len(kw)
            elif any(g.startswith(kw) or g.endswith(kw) for g in glued):
                hits += len(kw)
        if hits > best_hits:
            best, best_hits = narrative, hits
    return best


def narrative_labels(panel: TokenPanel) -> pd.Series:
    """Narrative per token column. Tokens without metadata are `OTHER`."""
    if panel.meta is None:
        return pd.Series(OTHER, index=panel.tokens, name="narrative")
    meta = panel.meta.reindex(panel.tokens)
    labels = [classify(n, s) for n, s in zip(meta.get("name", pd.Series(index=meta.index)),
                                             meta.get("symbol", pd.Series(index=meta.index)), strict=True)]
    return pd.Series(labels, index=panel.tokens, name="narrative")


def _by_narrative(frame: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Sum a time × token frame into time × narrative."""
    return frame.T.groupby(labels.reindex(frame.columns).fillna(OTHER)).sum(min_count=1).T


def narrative_volume_share(panel: TokenPanel, labels: pd.Series, bars: int = 36) -> pd.DataFrame:
    """Each narrative's share of all traded volume over the trailing `bars`."""
    volume = _by_narrative(panel.volume.fillna(0.0), labels).rolling(bars, min_periods=1).sum()
    total = volume.sum(axis=1).replace(0.0, np.nan)
    return volume.div(total, axis=0)


def narrative_heat(
    panel: TokenPanel, labels: pd.Series, fast: int = 12, slow: int = 288, floor: float = 0.005
) -> pd.DataFrame:
    """Log of short-window volume share over long-window share.

    Positive: the narrative is taking a bigger slice of attention than usual,
    the rotation signal. `floor` keeps tiny narratives from producing huge
    ratios out of a few trades.
    """
    short = narrative_volume_share(panel, labels, fast)
    long = narrative_volume_share(panel, labels, slow)
    return np.log((short + floor) / (long + floor))


def narrative_return(panel: TokenPanel, labels: pd.Series, bars: int = 36) -> pd.DataFrame:
    """Trailing liquidity-weighted return of each narrative, deaths at -100%."""
    returns = panel.returns_with_death()
    weight = panel.liquidity.shift(1).where(returns.notna())
    num = _by_narrative((returns * weight).fillna(0.0), labels)
    den = _by_narrative(weight.fillna(0.0), labels).replace(0.0, np.nan)
    per_bar = (num / den).fillna(0.0)
    return np.expm1(np.log1p(per_bar.clip(lower=-0.99)).rolling(bars, min_periods=1).sum())


def narrative_launches(panel: TokenPanel, labels: pd.Series, bars: int = 36) -> pd.DataFrame:
    """Tokens that got their first price within the trailing `bars`, per narrative."""
    first = panel.close.notna() & panel.close.shift(1).isna() & (panel.close.notna().cumsum() == 1)
    return _by_narrative(first.astype(float), labels).rolling(bars, min_periods=1).sum()


def launch_saturation(panel: TokenPanel, labels: pd.Series, fast: int = 36, slow: int = 864) -> pd.DataFrame:
    """Launches in the fast window over the narrative's average pace (1 = normal).

    A narrative where everyone is now launching copycats is crowded: the
    attention is split across more tickers and the late entrants are exit
    liquidity.
    """
    now = narrative_launches(panel, labels, fast)
    usual = narrative_launches(panel, labels, slow) * (fast / slow)
    return (now + 1.0) / (usual + 1.0)


def to_tokens(frame: pd.DataFrame, labels: pd.Series, columns: pd.Index) -> pd.DataFrame:
    """Broadcast a time × narrative frame to time × token by each token's label."""
    labels = labels.reindex(columns).fillna(OTHER)
    out = pd.DataFrame(np.nan, index=frame.index, columns=columns)
    for narrative, tokens in labels.groupby(labels).groups.items():
        if narrative in frame.columns:
            out[list(tokens)] = np.repeat(frame[narrative].to_numpy()[:, None], len(tokens), axis=1)
    return out


def narrative_summary(panel: TokenPanel, labels: pd.Series | None = None) -> pd.DataFrame:
    """One row per narrative: tokens, still alive, died, share of total volume."""
    labels = narrative_labels(panel) if labels is None else labels
    alive_end = panel.close.iloc[-1].notna()
    listed = panel.close.notna().any()
    volume = panel.volume.fillna(0.0).sum()
    table = pd.DataFrame({
        "tokens": listed.groupby(labels).sum(),
        "alive": (alive_end & listed).groupby(labels).sum(),
        "volume_share": volume.groupby(labels).sum() / max(volume.sum(), 1e-12),
    })
    table["died"] = table["tokens"] - table["alive"]
    return table.sort_values("volume_share", ascending=False)[["tokens", "alive", "died", "volume_share"]]
