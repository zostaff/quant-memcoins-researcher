"""Validation: the hard gates that separate edge from noise.

The full pipeline is:
    1. walk_forward_evaluate(...)         — honest out-of-sample performance
    2. combinatorial_purged_cv(...)       — robustness across overlapping splits
    3. deflated_sharpe(...)               — penalize the Sharpe for multiple-testing

If a strategy passes all three, it's worth a paper-trade slot. If it fails any,
it's killed before it costs real money. The deflated_sharpe gate has no
override — that's the whole point.
"""

from ai_quant_lab.validation.deflated_sharpe import (
    deflated_sharpe,
    deflated_sharpe_pvalue,
    estimate_trial_variance,
    probabilistic_sharpe,
)
from ai_quant_lab.validation.factor_attribution import (
    AttributionResult,
    PCAResult,
    factor_concentration_score,
    fama_french_attribution,
    pca_decompose,
)
from ai_quant_lab.validation.meta_labeling import (
    LogisticBinaryClassifier,
    MetaLabeler,
    meta_label_targets,
)
from ai_quant_lab.validation.diagnostics import (
    degradation_ratio,
    fold_stability,
    regime_breakdown,
)
from ai_quant_lab.validation.purged_cv import combinatorial_purged_cv, purged_kfold_splits
from ai_quant_lab.validation.walk_forward import (
    WalkForwardFold,
    walk_forward_evaluate,
    walk_forward_splits,
)

__all__ = [
    "AttributionResult",
    "LogisticBinaryClassifier",
    "MetaLabeler",
    "PCAResult",
    "WalkForwardFold",
    "combinatorial_purged_cv",
    "deflated_sharpe",
    "deflated_sharpe_pvalue",
    "degradation_ratio",
    "estimate_trial_variance",
    "factor_concentration_score",
    "fama_french_attribution",
    "fold_stability",
    "meta_label_targets",
    "pca_decompose",
    "probabilistic_sharpe",
    "purged_kfold_splits",
    "regime_breakdown",
    "walk_forward_evaluate",
    "walk_forward_splits",
]
