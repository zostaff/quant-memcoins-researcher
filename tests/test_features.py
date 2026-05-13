"""Tests for the feature pipeline and leakage detector.

The detector must:
    - flag a feature that uses `price.shift(-k)` directly
    - flag a feature whose future correlation is suspiciously high
    - NOT flag a properly shifted clean feature
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_quant_lab.features import (
    FeaturePipeline,
    FeatureStep,
    detect_leakage,
    momentum,
    realized_volatility,
    rolling_zscore,
)
from ai_quant_lab.features.labels import forward_returns, triple_barrier_labels


def test_clean_features_pass(gbm_price_series):
    pipeline = FeaturePipeline(
        [
            FeatureStep("mom_21", lambda df: momentum(df["close"], 21), ("close",)),
            FeatureStep("zscore_21", lambda df: rolling_zscore(df["close"], 21), ("close",)),
            FeatureStep("rv_21", lambda df: realized_volatility(df["close"], 21), ("close",)),
        ]
    )
    target = forward_returns(gbm_price_series, 1)
    features = pipeline.fit_transform(gbm_price_series.to_frame(), target=target)
    assert features.shape[1] == 3
    report = pipeline.last_report
    assert not report.has_leakage, report.format_problems()


def test_forward_reference_is_caught(gbm_price_series):
    """A feature that uses price.shift(-3) must trip the detector."""
    leaky = (gbm_price_series.shift(-3) / gbm_price_series - 1.0).rename("peek")
    target = forward_returns(gbm_price_series, 1)
    report = detect_leakage(pd.DataFrame({"peek": leaky}), target)
    assert report.has_leakage
    assert "peek" in report.problems[0]


def test_pipeline_raises_on_leakage(gbm_price_series):
    def peek(df):
        return (df["close"].shift(-3) / df["close"] - 1.0).rename("peek")

    pipeline = FeaturePipeline([FeatureStep("peek", peek, ("close",))])
    target = forward_returns(gbm_price_series, 1)
    with pytest.raises(ValueError, match="Leakage detected"):
        pipeline.fit_transform(gbm_price_series.to_frame(), target=target)


def test_triple_barrier_labels_shape(gbm_price_series):
    labels = triple_barrier_labels(
        gbm_price_series, upper_pct=0.02, lower_pct=0.02, max_holding=20
    )
    valid = labels.dropna()
    assert set(valid.unique()) <= {-1.0, 0.0, 1.0}
    assert valid.index.equals(gbm_price_series.index[: -20])


def test_pipeline_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate"):
        FeaturePipeline(
            [
                FeatureStep("x", lambda df: df.iloc[:, 0]),
                FeatureStep("x", lambda df: df.iloc[:, 0]),
            ]
        )


def test_momentum_is_lagged(gbm_price_series):
    """momentum(price, 21) at time t must use only prices up to t-1."""
    mom = momentum(gbm_price_series, 21)
    # First 22 bars should be NaN: needs 21-lookback PLUS 1-bar shift.
    assert mom.iloc[:22].isna().all()
    assert not mom.iloc[22:].isna().any()
