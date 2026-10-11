"""TokenPanel: a point-in-time memecoin universe.

The equity world learned the hard way that a universe defined by "what trades
today" is survivorship-biased. Memecoins are the extreme case: most tokens die
within hours, many by a rug pull. A backtest that only sees tokens that still
have a chart is a backtest of the winners.

Conventions:
    - Every frame is time × token. Columns never disappear: a dead token keeps
      its column, with NaN close and zero liquidity after death.
    - `liquidity` is total pool TVL in USD (both sides), the number most data
      vendors report. The quote-side reserve an AMM trade hits is half of it.
    - A token is tradeable at bar t only if it has a price and liquidity above
      the caller's floor at bar t. Nothing about bar t+1 is used.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class TokenPanel:
    """Aligned time × token frames for one research universe."""

    close: pd.DataFrame
    liquidity: pd.DataFrame
    volume: pd.DataFrame
    holders: pd.DataFrame | None = None
    launch_time: pd.Series = field(default_factory=lambda: pd.Series(dtype="datetime64[ns]"))
    # Further time × token fields by name (top10_share, graduation, mint_risk, ...).
    extra: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    # Per-token pool fee as a fraction (Uniswap fee tiers differ by pool); None = config default.
    pool_fee: pd.Series | None = None

    def __post_init__(self) -> None:
        frames = {"liquidity": self.liquidity, "volume": self.volume, **self.extra}
        if self.holders is not None:
            frames["holders"] = self.holders
        for name, frame in frames.items():
            if not frame.index.equals(self.close.index):
                raise ValueError(f"{name} index does not match close index")
            if not frame.columns.equals(self.close.columns):
                raise ValueError(f"{name} columns do not match close columns")
        if not self.close.index.is_monotonic_increasing:
            raise ValueError("Panel index must be sorted ascending")

    @property
    def tokens(self) -> pd.Index:
        return self.close.columns

    def returns(self) -> pd.DataFrame:
        """Single-bar returns. NaN before launch and after death."""
        return self.close.pct_change(fill_method=None)

    def returns_with_death(self) -> pd.DataFrame:
        """Like `returns`, but the bar where a token's price disappears is -100%.

        Factor returns and attribution must see rugs, or the "market" factor
        is a market of survivors.
        """
        returns = self.returns()
        died = self.close.shift(1).notna() & self.close.isna()
        return returns.mask(died, -1.0)

    def alive(self, min_liquidity_usd: float = 0.0) -> pd.DataFrame:
        """Boolean mask: token has a price and enough liquidity at bar t."""
        return self.close.notna() & (self.liquidity.fillna(0.0) > min_liquidity_usd)

    def as_strategy_input(self) -> dict[str, pd.DataFrame]:
        """What a sandboxed strategy receives. Plain dict: no methods to misuse."""
        data = {
            "close": self.close,
            "liquidity": self.liquidity,
            "volume": self.volume,
        }
        if self.holders is not None:
            data["holders"] = self.holders
        data.update(self.extra)
        return data

    def slice(self, start: object = None, end: object = None) -> TokenPanel:
        """Time slice. Tokens are kept even if they are dead in the window."""
        sl = slice(start, end)
        return TokenPanel(
            close=self.close.loc[sl],
            liquidity=self.liquidity.loc[sl],
            volume=self.volume.loc[sl],
            holders=None if self.holders is None else self.holders.loc[sl],
            launch_time=self.launch_time,
            extra={name: frame.loc[sl] for name, frame in self.extra.items()},
            pool_fee=self.pool_fee,
        )

    def death_bars(self) -> pd.Series:
        """Last bar with a price for each token, or NaT if still alive at the end."""
        last_valid = self.close.apply(lambda col: col.last_valid_index())
        still_alive = self.close.iloc[-1].notna()
        return last_valid.where(~still_alive)

    def survivorship_report(self) -> dict[str, float]:
        """How much of the universe a 'tokens that exist today' sample would drop."""
        n = len(self.tokens)
        dead = int(self.close.iloc[-1].isna().sum())
        return {
            "n_tokens": float(n),
            "n_dead": float(dead),
            "dead_fraction": dead / n if n else 0.0,
        }


def bars_per_year(index: pd.DatetimeIndex) -> int:
    """Annualization for a 24/7 market from the median bar spacing."""
    if len(index) < 2:
        raise ValueError("Need at least 2 bars to infer spacing")
    step = pd.Series(index).diff().median()
    seconds = step.total_seconds()
    if seconds <= 0:
        raise ValueError("Non-increasing index")
    return int(round(365 * 24 * 3600 / seconds))


def bars_per_day(index: pd.DatetimeIndex) -> int:
    return max(int(round(bars_per_year(index) / 365)), 1)


def empty_like(panel: TokenPanel, value: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(value, index=panel.close.index, columns=panel.close.columns, dtype=float)

