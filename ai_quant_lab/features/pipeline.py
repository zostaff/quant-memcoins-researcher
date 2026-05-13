"""FeaturePipeline: composable, leakage-proof feature construction.

Each `FeatureStep` declares its name and a callable. The pipeline runs them
in order on a DataFrame and runs the leakage detector on every output before
returning. If any step's output correlates with the *future* target, the
pipeline raises rather than silently inflating the backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from ai_quant_lab.features.leakage_detector import (
    LeakageReport,
    detect_leakage,
    detect_leakage_structural,
)


@dataclass(frozen=True)
class FeatureStep:
    """One step in a feature pipeline.

    Attributes:
        name: Output column name.
        func: Callable `(DataFrame) -> Series`. Must return a Series indexed like the input.
        required_columns: Input columns the step depends on. Checked before running.
    """

    name: str
    func: Callable[[pd.DataFrame], pd.Series]
    required_columns: tuple[str, ...] = ()


class FeaturePipeline:
    """Pipeline that runs feature steps and audits for leakage.

    Usage:
        pipeline = FeaturePipeline([
            FeatureStep("mom_21", lambda df: momentum(df["close"], 21), ("close",)),
            FeatureStep("rv_21", lambda df: realized_volatility(df["close"], 21), ("close",)),
        ])
        features = pipeline.fit_transform(price_data, target=price_data["close"].pct_change())
    """

    def __init__(
        self,
        steps: list[FeatureStep],
        *,
        audit: bool = True,
        structural_audit: bool = False,
    ) -> None:
        if not steps:
            raise ValueError("Pipeline must have at least one step.")
        names = [step.name for step in steps]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate step names: {names}")
        self.steps = steps
        self.audit = audit
        self.structural_audit = structural_audit
        self._last_report: LeakageReport | None = None

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Run every step and return a DataFrame of features. No audit."""
        outputs: dict[str, pd.Series] = {}
        for step in self.steps:
            missing = [col for col in step.required_columns if col not in frame.columns]
            if missing:
                raise KeyError(f"Step '{step.name}' missing columns: {missing}")
            result = step.func(frame)
            if not isinstance(result, pd.Series):
                raise TypeError(f"Step '{step.name}' must return a Series, got {type(result).__name__}.")
            if not result.index.equals(frame.index):
                raise ValueError(f"Step '{step.name}' returned a Series with a different index.")
            outputs[step.name] = result.rename(step.name)
        return pd.DataFrame(outputs, index=frame.index)

    def fit_transform(
        self,
        frame: pd.DataFrame,
        target: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Transform and (if `target` is provided and `audit` is True) audit for leakage.

        Raises:
            ValueError: if leakage is detected and `audit` is True.
        """
        features = self.transform(frame)
        report = LeakageReport()

        if self.audit and target is not None:
            corr_report = detect_leakage(features, target)
            report.problems.extend(corr_report.problems)
            report.column_scores.update(corr_report.column_scores)

        if self.structural_audit:
            for step in self.steps:
                try:
                    structural = detect_leakage_structural(step.func, frame, name=step.name)
                except (ValueError, TypeError):
                    continue
                report.problems.extend(structural.problems)
                report.column_scores.update(structural.column_scores)

        self._last_report = report
        if report.has_leakage:
            raise ValueError("Leakage detected:\n" + report.format_problems())
        return features

    @property
    def last_report(self) -> LeakageReport | None:
        """The audit report from the last `fit_transform` call, if any."""
        return self._last_report
