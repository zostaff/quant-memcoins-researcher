"""Shared test fixtures and path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make ai_quant_lab importable when running `pytest` from repo root or tests dir.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def gbm_price_series() -> pd.Series:
    """Deterministic GBM price series of length 1000."""
    rng = np.random.default_rng(seed=42)
    daily_returns = rng.normal(0.05 / 252, 0.16 / np.sqrt(252), 1000)
    prices = 100.0 * np.exp(np.cumsum(daily_returns))
    index = pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=1000)
    return pd.Series(prices, index=index, name="close")


@pytest.fixture
def random_returns() -> pd.Series:
    rng = np.random.default_rng(seed=42)
    return pd.Series(rng.normal(0.0005, 0.01, 1000))
