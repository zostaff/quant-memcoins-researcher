"""Feature engineering: leakage-proof by construction.

The pipeline forces every transformation to register its "look-back window."
Any operation that would read the future raises immediately, so leakage is
caught at construction time rather than dressed up as 90% out-of-sample accuracy.
"""

from ai_quant_lab.features.labels import triple_barrier_labels
from ai_quant_lab.features.leakage_detector import (
    LeakageReport,
    detect_leakage,
    detect_leakage_structural,
)
from ai_quant_lab.features.library import (
    momentum,
    range_pct,
    realized_volatility,
    rolling_zscore,
)
from ai_quant_lab.features.pipeline import FeaturePipeline, FeatureStep

__all__ = [
    "FeaturePipeline",
    "FeatureStep",
    "LeakageReport",
    "detect_leakage",
    "detect_leakage_structural",
    "momentum",
    "range_pct",
    "realized_volatility",
    "rolling_zscore",
    "triple_barrier_labels",
]
