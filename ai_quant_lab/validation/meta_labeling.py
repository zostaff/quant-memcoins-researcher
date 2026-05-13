"""Meta-labeling (López de Prado, Advances in Financial Machine Learning, ch.3).

Idea:
    1. Primary model produces a side (long/short) for each bar — e.g. our
       cross-sectional momentum signal.
    2. Secondary "meta" model takes the primary side AND a feature matrix
       and predicts whether to ACT (1) or SKIP (0).

The secondary model is a binary classifier whose target is "did the primary
side make money?" It learns to refuse trades when the primary signal is
unreliable — typically lifting Sharpe from 0.6 to 1.2+ on the same primary
without finding new edge.

We deliberately use logistic regression as the default classifier:
    - Cheap to fit, deterministic, no PyTorch dependency.
    - Coefficients are interpretable — you can see WHY it skips.
    - Tree models would over-fit on the small samples typical in quant.

If you want a fancier model later, the `MetaLabeler` interface accepts any
sklearn-style estimator with .fit / .predict_proba.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


class BinaryClassifier(Protocol):
    """Minimal interface for a meta-classifier."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryClassifier": ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class LogisticBinaryClassifier:
    """Stand-alone logistic regression with L2 regularization.

    We don't depend on sklearn. The implementation uses scipy's optimizer.
    Coefficients are stored on `self.coef_` (no intercept-handling magic;
    we prepend a 1-column to X internally).
    """

    l2: float = 1.0
    max_iter: int = 200
    coef_: np.ndarray | None = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBinaryClassifier":
        from scipy.optimize import minimize

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y row counts must match")

        n, d = X.shape
        design = np.column_stack([np.ones(n), X])

        def loss(w: np.ndarray) -> float:
            logits = design @ w
            log_likelihood = np.sum(y * logits - np.log1p(np.exp(logits)))
            reg = 0.5 * self.l2 * np.sum(w[1:] ** 2)
            return -log_likelihood / n + reg / n

        def grad(w: np.ndarray) -> np.ndarray:
            logits = design @ w
            p = 1.0 / (1.0 + np.exp(-logits))
            base = design.T @ (p - y) / n
            base[1:] += self.l2 * w[1:] / n
            return base

        result = minimize(
            loss, x0=np.zeros(d + 1), jac=grad,
            method="L-BFGS-B", options={"maxiter": self.max_iter},
        )
        self.coef_ = result.x
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict_proba()")
        X = np.asarray(X, dtype=float)
        design = np.column_stack([np.ones(X.shape[0]), X])
        p = 1.0 / (1.0 + np.exp(-(design @ self.coef_)))
        return np.column_stack([1.0 - p, p])


@dataclass
class MetaLabeler:
    """Trains a secondary classifier on top of a primary signal.

    Args:
        classifier: any object exposing .fit and .predict_proba. Defaults to
            LogisticBinaryClassifier(l2=1.0).
        threshold: probability above which to ACT. Lower → trade more often
            but with lower precision; higher → skip more aggressively.
    """

    classifier: BinaryClassifier = field(default_factory=LogisticBinaryClassifier)
    threshold: float = 0.5

    def fit(
        self,
        primary_side: pd.Series,
        forward_returns: pd.Series,
        features: pd.DataFrame,
    ) -> "MetaLabeler":
        """Train the meta-classifier.

        Args:
            primary_side: in {-1, +1}, the direction the primary signal would
                take at each bar. NaN means primary is silent — skipped.
            forward_returns: returns over the holding period the primary
                strategy uses.
            features: per-bar features used by the meta-classifier.

        Notes:
            Target y = 1 if primary_side * forward_return > 0, else 0.
            i.e. "did following the primary make money on this bar?"
        """
        df = pd.concat(
            [primary_side.rename("side"), forward_returns.rename("fwd"), features],
            axis=1,
        ).dropna()
        if len(df) < 50:
            raise ValueError("Need at least 50 aligned observations to fit a meta-labeler.")
        y = ((df["side"] * df["fwd"]) > 0).astype(float).to_numpy()
        feature_cols = [c for c in df.columns if c not in ("side", "fwd")]
        X = df[feature_cols].to_numpy(dtype=float)
        self.classifier.fit(X, y)
        self._feature_names: list[str] = feature_cols  # type: ignore[attr-defined]
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Probability of "primary side will be right" at each bar."""
        if not hasattr(self, "_feature_names"):
            raise RuntimeError("Call fit() before predict()")
        X = features[self._feature_names].to_numpy(dtype=float)
        mask = ~np.isnan(X).any(axis=1)
        probabilities = np.full(len(features), np.nan)
        if mask.any():
            probabilities[mask] = self.classifier.predict_proba(X[mask])[:, 1]
        return pd.Series(probabilities, index=features.index, name="meta_probability")

    def apply(
        self,
        primary_side: pd.Series,
        features: pd.DataFrame,
    ) -> pd.Series:
        """Multiply the primary side by the act/skip mask.

        Returned series has the same shape as `primary_side`. Bars where the
        meta-model predicts < threshold are set to 0 (skip).
        """
        probabilities = self.predict(features)
        act_mask = (probabilities >= self.threshold).astype(float)
        return primary_side.fillna(0.0) * act_mask.fillna(0.0)


def meta_label_targets(primary_side: pd.Series, forward_returns: pd.Series) -> pd.Series:
    """Build the binary target series used by `MetaLabeler.fit`.

    Returns a Series of {0, 1}: 1 = primary made money on that bar, 0 = didn't.
    Useful for diagnostics outside the labeler (e.g. computing base rate).
    """
    return ((primary_side * forward_returns) > 0).astype(float).where(~primary_side.isna())
