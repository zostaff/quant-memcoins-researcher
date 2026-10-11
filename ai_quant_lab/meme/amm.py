"""Execution cost model for constant-product pools and bonding curves.

On a listed equity, trading costs are a few bps and impact follows the
square-root law. On a memecoin pool the "order book" is an x·y=k curve with a
few thousand dollars on each side, so the trade size relative to the pool is
the whole story. pump.fun-style bonding curves are constant-product curves on
virtual reserves, so the same formula applies with the virtual quote reserve.

For a trade of `s` USD against a quote reserve `Q` with fee `f`, the fraction
of value lost versus the pre-trade spot price is

    loss(s) = 1 - (1 - f) * Q / (Q + s)

Buy and sell are symmetric to first order. A round trip of 10% of the reserve
at a 1% fee loses about 20% before priority fees and MEV.

Defaults below are placeholders, not venue quotes: fees change. Set them from
the venue you trade on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutionCosts:
    """Everything that sits between a paper fill and a real one.

    Attributes:
        swap_fee: Pool/curve fee as a fraction of the trade (0.01 = 1%).
        priority_fee_usd: Fixed cost per transaction: priority fee + bundle tip.
            Matters a lot for small tickets.
        mev_haircut: Extra fraction lost to sandwiches and adverse ordering.
        min_trade_usd: Trades smaller than this are skipped (not worth the tx).
        min_liquidity_usd: Pools below this TVL are untradeable: no entries,
            and exits are only possible while liquidity is still above zero.
    """

    swap_fee: float = 0.01
    priority_fee_usd: float = 0.10
    mev_haircut: float = 0.002
    min_trade_usd: float = 5.0
    min_liquidity_usd: float = 5_000.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.swap_fee < 1.0:
            raise ValueError("swap_fee must be in [0, 1)")
        if min(self.priority_fee_usd, self.mev_haircut, self.min_trade_usd) < 0:
            raise ValueError("cost components must be non-negative")


# Starting points per chain. Placeholders: set them from the venue you trade on.
# Robinhood Chain is an Arbitrum L2 with ETH gas and a sequencer instead of a
# public mempool, so a transaction costs cents and sandwiching is harder than
# on Solana. Uniswap pools there carry their own fee tier, which `load_panel`
# reads from the pool name into `panel.pool_fee`.
NETWORK_COSTS: dict[str, ExecutionCosts] = {
    "solana": ExecutionCosts(),
    "robinhood": ExecutionCosts(swap_fee=0.01, priority_fee_usd=0.02, mev_haircut=0.001),
}


def costs_for_network(network: str) -> ExecutionCosts:
    """Default `ExecutionCosts` for a GeckoTerminal network id."""
    try:
        return NETWORK_COSTS[network]
    except KeyError:
        raise ValueError(f"No cost preset for network {network!r}; pass ExecutionCosts yourself") from None


def quote_reserve(liquidity_usd: np.ndarray | float) -> np.ndarray | float:
    """Quote-side reserve from total pool TVL (both sides at equal value)."""
    return np.asarray(liquidity_usd, dtype=float) / 2.0


def swap_loss_fraction(
    trade_usd: np.ndarray | float,
    liquidity_usd: np.ndarray | float,
    swap_fee: np.ndarray | float,
) -> np.ndarray:
    """Fraction of `trade_usd` lost to fee + curve slippage on one leg.

    Exact for constant-product pools and bonding curves. For concentrated
    liquidity (Uniswap v3/v4) it is conservative: depth near the price is
    higher than a full-range pool of the same TVL.
    """
    trade = np.abs(np.asarray(trade_usd, dtype=float))
    reserve = quote_reserve(liquidity_usd)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = 1.0 - (1.0 - swap_fee) * reserve / (reserve + trade)
    # No pool: a trade cannot happen at any price.
    return np.where(reserve > 0, loss, 1.0)


def trade_cost_usd(
    trade_usd: np.ndarray | float,
    liquidity_usd: np.ndarray | float,
    costs: ExecutionCosts,
) -> np.ndarray:
    """Total USD cost of executing each trade: curve + fee + MEV + priority fee."""
    trade = np.abs(np.asarray(trade_usd, dtype=float))
    curve = trade * swap_loss_fraction(trade, liquidity_usd, costs.swap_fee)
    mev = trade * costs.mev_haircut
    fixed = np.where(trade > 0, costs.priority_fee_usd, 0.0)
    return curve + mev + fixed


def round_trip_cost_fraction(
    trade_usd: float,
    liquidity_usd: float,
    costs: ExecutionCosts | None = None,
) -> float:
    """Cost of buying and later selling `trade_usd` at unchanged liquidity."""
    costs = costs or ExecutionCosts()
    one_leg = float(trade_cost_usd(trade_usd, liquidity_usd, costs))
    return 2.0 * one_leg / trade_usd if trade_usd > 0 else 0.0


def max_size_for_cost(
    liquidity_usd: float,
    max_round_trip_cost: float,
    costs: ExecutionCosts | None = None,
) -> float:
    """Largest ticket whose round trip costs at most `max_round_trip_cost`.

    The capacity question from the other direction: given a pool, how big can
    a position be before execution eats the edge you expect.
    """
    costs = costs or ExecutionCosts()
    lo, hi = 0.0, float(liquidity_usd)
    if round_trip_cost_fraction(max(hi, 1e-9), liquidity_usd, costs) <= max_round_trip_cost:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid > 0 and round_trip_cost_fraction(mid, liquidity_usd, costs) <= max_round_trip_cost:
            lo = mid
        else:
            hi = mid
    return lo
