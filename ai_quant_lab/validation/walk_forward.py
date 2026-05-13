"""Walk-forward validation with optional purge.

Walks the timeline left to right, training on `train_size` then testing on
`test_size`, sliding by `test_size`. Optionally inserts a purge gap between
train and test to prevent label-overlap leakage (relevant when labels look
forward more than one bar).

Two modes:
    - 'rolling' (default): training window slides
    - 'expanding': training window grows
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Literal

import numpy as np
import pandas as pd

from ai_quant_lab.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    vectorized_backtest,
)


@dataclass(frozen=True)
class WalkForwardFold:
    """One fold: train slice, test slice, and the indices that produced them."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def train_slice(self, frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        return frame.iloc[self.train_start : self.train_end]

    def test_slice(self, frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        return frame.iloc[self.test_start : self.test_end]


def walk_forward_splits(
    n_observations: int,
    train_size: int,
    test_size: int,
    *,
    purge: int = 0,
    mode: Literal["rolling", "expanding"] = "rolling",
) -> Iterator[WalkForwardFold]:
    """Yield walk-forward folds.

    Args:
        n_observations: Length of the time series.
        train_size: Bars in each training window.
        test_size: Bars in each test window.
        purge: Bars to skip between train end and test start. Set to the maximum
            label-overlap horizon for triple-barrier labels.
        mode: 'rolling' keeps the training window fixed-size; 'expanding' grows it.

    Raises:
        ValueError: if the arguments don't fit the series.
    """
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must be >= 1.")
    if purge < 0:
        raise ValueError("purge must be >= 0.")
    if train_size + purge + test_size > n_observations:
        raise ValueError("train_size + purge + test_size exceeds the series length.")

    fold_index = 0
    train_start = 0
    train_end = train_size
    while train_end + purge + test_size <= n_observations:
        test_start = train_end + purge
        test_end = test_start + test_size
        yield WalkForwardFold(
            fold=fold_index,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        fold_index += 1
        train_end += test_size
        if mode == "rolling":
            train_start += test_size


def walk_forward_evaluate(
    price_data: pd.Series,
    strategy: Callable[[pd.Series], pd.Series],
    *,
    train_size: int,
    test_size: int,
    purge: int = 0,
    mode: Literal["rolling", "expanding"] = "rolling",
    config: BacktestConfig | None = None,
) -> dict[str, object]:
    """Run a strategy across walk-forward folds.

    `strategy` is a callable `(price_data) -> positions`. It receives the
    training slice for fitting (if it needs to fit) AND THEN is called on the
    full window through the end of the test slice — but only the test slice
    of its output is evaluated. This mirrors the way you'd deploy live: the
    feature pipeline sees the whole tape, but trades only happen out-of-sample.

    Returns:
        Dict with:
            - 'folds': list of per-fold backtest results
            - 'concatenated_returns': all test-slice net returns end-to-end
            - 'fold_sharpes': per-fold Sharpe ratios
            - 'metrics': metrics of the concatenated curve
    """
    config = config or BacktestConfig()
    returns = price_data.pct_change()
    folds_iter = walk_forward_splits(
        n_observations=len(price_data),
        train_size=train_size,
        test_size=test_size,
        purge=purge,
        mode=mode,
    )

    fold_results: list[BacktestResult] = []
    fold_sharpes: list[float] = []
    all_test_returns: list[pd.Series] = []

    for fold in folds_iter:
        full_window = price_data.iloc[fold.train_start : fold.test_end]
        positions = strategy(full_window)
        if not isinstance(positions, pd.Series):
            raise TypeError("strategy must return a pandas Series of positions.")
        positions = positions.reindex(full_window.index).fillna(0.0)

        # Slice to test window only.
        test_index = price_data.index[fold.test_start : fold.test_end]
        test_returns = returns.reindex(test_index).fillna(0.0)
        test_positions = positions.reindex(test_index).fillna(0.0)

        result = vectorized_backtest(test_positions, test_returns, config=config)
        fold_results.append(result)
        fold_sharpes.append(result.metrics["sharpe_ratio"])
        all_test_returns.append(result.returns)

    if not fold_results:
        raise ValueError("No folds produced. Check train_size / test_size / purge.")

    concatenated = pd.concat(all_test_returns).sort_index()
    from ai_quant_lab.backtest.engine import performance_metrics  # local import to avoid cycle

    metrics = performance_metrics(concatenated, annualization=config.annualization)

    return {
        "folds": fold_results,
        "concatenated_returns": concatenated,
        "fold_sharpes": np.array(fold_sharpes),
        "metrics": metrics,
    }
