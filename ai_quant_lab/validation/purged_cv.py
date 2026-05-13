"""Combinatorial purged cross-validation (López de Prado, 2018, ch. 12).

Standard k-fold CV fails on time series because:
    1. Information at the boundary of consecutive folds leaks across the split.
    2. Random shuffling destroys temporal structure.

Purged CV addresses both by:
    1. Splitting the data into N contiguous groups.
    2. Choosing K groups as the test set, the rest as train.
    3. PURGING bars from the train set whose labels overlap the test set.
    4. EMBARGOING the bars immediately AFTER each test block from the train set
       to prevent leakage in the other direction.

`purged_kfold_splits` is the simpler 1-out-of-N case; `combinatorial_purged_cv`
is the K-out-of-N case used to estimate the variance of out-of-sample Sharpe.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class PurgedSplit:
    """A train/test index pair from a purged splitter."""

    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def purged_kfold_splits(
    n_observations: int,
    *,
    n_splits: int = 5,
    purge: int = 0,
    embargo: int = 0,
) -> Iterator[PurgedSplit]:
    """Standard purged k-fold. K=1 test group per split.

    Args:
        n_observations: Length of the series.
        n_splits: Number of folds.
        purge: Bars of label-overlap to drop adjacent to the test window.
        embargo: Additional bars to drop AFTER each test window to prevent
            information from the test block influencing future training data.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be >= 0.")

    indices = np.arange(n_observations)
    fold_boundaries = np.linspace(0, n_observations, n_splits + 1, dtype=int)

    for fold in range(n_splits):
        test_start, test_end = fold_boundaries[fold], fold_boundaries[fold + 1]
        test_indices = indices[test_start:test_end]
        mask = np.ones(n_observations, dtype=bool)
        # drop the test block plus purge before it and embargo after it
        mask[max(0, test_start - purge) : min(n_observations, test_end + embargo)] = False
        train_indices = indices[mask]
        yield PurgedSplit(fold=fold, train_indices=train_indices, test_indices=test_indices)


def combinatorial_purged_cv(
    n_observations: int,
    *,
    n_groups: int = 10,
    n_test_groups: int = 2,
    purge: int = 0,
    embargo: int = 0,
) -> Iterator[PurgedSplit]:
    """Combinatorial purged CV: choose `n_test_groups` out of `n_groups`.

    Produces `C(n_groups, n_test_groups)` splits. Useful for estimating the
    distribution of out-of-sample performance — every observation appears in
    multiple test sets, giving a robustness check that single-shot walk-forward
    can't provide.

    Args:
        n_observations: Length of the series.
        n_groups: Number of contiguous groups to partition into.
        n_test_groups: How many of those groups to combine as the test set.
        purge: Bars to purge adjacent to each test block.
        embargo: Bars to embargo after each test block.

    Yields:
        PurgedSplit objects in deterministic combination order.
    """
    if n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be < n_groups.")
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2.")

    indices = np.arange(n_observations)
    boundaries = np.linspace(0, n_observations, n_groups + 1, dtype=int)
    fold_id = 0

    for combo in combinations(range(n_groups), n_test_groups):
        test_mask = np.zeros(n_observations, dtype=bool)
        drop_mask = np.zeros(n_observations, dtype=bool)
        for group in combo:
            start, end = boundaries[group], boundaries[group + 1]
            test_mask[start:end] = True
            drop_mask[max(0, start - purge) : min(n_observations, end + embargo)] = True

        train_indices = indices[~drop_mask]
        test_indices = indices[test_mask]
        yield PurgedSplit(fold=fold_id, train_indices=train_indices, test_indices=test_indices)
        fold_id += 1
