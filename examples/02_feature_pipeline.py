"""Example 02 — building leakage-proof features via FeaturePipeline.

The pipeline composes three features. The leakage detector audits them against
the forward-return target. With clean features, the audit passes.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd

from ai_quant_lab.features import (
    FeaturePipeline,
    FeatureStep,
    momentum,
    realized_volatility,
    rolling_zscore,
)
from ai_quant_lab.features.labels import forward_returns


def main() -> None:
    rng = np.random.default_rng(seed=2)
    n = 1500
    returns = rng.normal(0.0003, 0.012, n)
    price_data = pd.Series(
        100.0 * np.exp(np.cumsum(returns)),
        index=pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n),
        name="close",
    )
    frame = price_data.to_frame()

    pipeline = FeaturePipeline(
        [
            FeatureStep("mom_21", lambda df: momentum(df["close"], 21), ("close",)),
            FeatureStep("zscore_21", lambda df: rolling_zscore(df["close"], 21), ("close",)),
            FeatureStep("rv_21", lambda df: realized_volatility(df["close"], 21), ("close",)),
        ]
    )
    target = forward_returns(price_data, horizon=1)
    features = pipeline.fit_transform(frame, target=target)

    print("Feature head:")
    print(features.tail(5))
    print()
    report = pipeline.last_report
    assert report is not None
    print(f"Leakage problems: {len(report.problems)}")
    for col, score in report.column_scores.items():
        print(f"  {col}: future/past corr ratio = {score:.2f}")


if __name__ == "__main__":
    main()
