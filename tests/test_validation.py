"""Tests for walk-forward and purged CV splitters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_quant_lab.backtest import BacktestConfig
from ai_quant_lab.features.library import momentum
from ai_quant_lab.validation import (
    combinatorial_purged_cv,
    purged_kfold_splits,
    walk_forward_evaluate,
    walk_forward_splits,
)


def test_walk_forward_split_count():
    folds = list(walk_forward_splits(n_observations=1000, train_size=500, test_size=100))
    # (1000-500)/100 = 5 folds
    assert len(folds) == 5
    # No overlap between train and test of the same fold
    for f in folds:
        assert f.train_end <= f.test_start


def test_walk_forward_purge_separates_train_test():
    folds = list(
        walk_forward_splits(n_observations=1000, train_size=500, test_size=100, purge=20)
    )
    for f in folds:
        assert f.test_start - f.train_end == 20


def test_walk_forward_expanding_grows_train():
    folds = list(
        walk_forward_splits(
            n_observations=1000, train_size=500, test_size=100, mode="expanding"
        )
    )
    train_sizes = [f.train_end - f.train_start for f in folds]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] == 500
    assert train_sizes[-1] > 500


def test_purged_kfold_excludes_test_block():
    splits = list(purged_kfold_splits(100, n_splits=5, purge=2, embargo=2))
    assert len(splits) == 5
    for split in splits:
        assert not set(split.train_indices) & set(split.test_indices)


def test_combinatorial_purged_cv_combo_count():
    splits = list(combinatorial_purged_cv(n_observations=100, n_groups=5, n_test_groups=2))
    # C(5, 2) = 10 splits
    assert len(splits) == 10
    # Test indices must all be valid
    for s in splits:
        assert s.test_indices.min() >= 0 and s.test_indices.max() < 100


def test_walk_forward_evaluate_end_to_end(gbm_price_series):
    def strategy(price):
        return np.sign(momentum(price, 21)).clip(-1, 1).fillna(0.0)

    out = walk_forward_evaluate(
        gbm_price_series,
        strategy,
        train_size=300,
        test_size=100,
        purge=5,
        config=BacktestConfig(cost_bps=5.0),
    )
    assert "concatenated_returns" in out
    assert len(out["folds"]) > 0


def test_walk_forward_rejects_oversized_window():
    with pytest.raises(ValueError):
        list(walk_forward_splits(n_observations=100, train_size=80, test_size=50))
